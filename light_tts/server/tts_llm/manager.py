import inspect
import pickle
import torch
import time
import multiprocessing
import uvloop
import asyncio

from light_tts.server.core.objs.req import ReqRunStatus
from light_tts.utils.process_check import start_parent_check_thread

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import concurrent.futures
import zmq
import zmq.asyncio
from typing import Dict, List, Optional
from .model_infer.model_rpc import start_model_process, ModelRpcClient
from .req_queue import build_req_queue
from light_tts.utils.infer_utils import calculate_time
from .batch import Batch, Req
from .stats import Stats
from .pause_strategy import Fcfs, select_paused_reqs
from light_tts.server.core.objs import ShmReqManager
from light_tts.utils.log_utils import init_logger
from light_tts.server.core.objs.shm_speech_manager import SharedSpeechManager
from light_tts.common.mem_manager import ReadOnlyStaticsMemoryManager
import multiprocessing as mp
from light_tts.utils.load_utils import CosyVoiceVersion, load_yaml_lite
from itertools import chain
import threading
from light_tts.server.tts_llm.token_load import TokenLoad
from light_tts.utils.envs_utils import get_unique_server_name
from light_tts.utils.graceful_utils import graceful_registry

logger = init_logger(__name__)


class RouterManager:
    def __init__(self, args, tts_llm_port, tts_decode_port, style_name, gpt_parall_lock):
        self.args = args
        self.world_size = 1
        self.node_world_size = 1
        self.nnodes = 1
        self.node_rank = 1
        self.dp_size = 1
        self.dp_size_in_node = 1
        # 判断是否是保守调度，保守调度不会发生暂停 req 的情况，但是有些场景可能影响吞吐
        self.is_safe_schedule = args.router_token_ratio == 0.0
        self.load_way = args.load_way
        self.mode = args.mode

        # tts
        self.style_name = style_name
        self.gpt_parall_lock = gpt_parall_lock
        self.has_lock = False
        self.parall_step_counter = 0
        self.parall_step_max_num = args.gpt_paral_step_num

        self.shm_req_manager = ShmReqManager()
        # 用共享内存进行共享，router 模块读取进行精确的调度估计
        self.read_only_statics_mem_manager = ReadOnlyStaticsMemoryManager()

        # 共享变量，用于存储router端调度分析得到的机器负载信息
        self.shared_token_load = TokenLoad(f"{get_unique_server_name()}_shared_token_load", self.dp_size_in_node)
        for dp_index in range(self.dp_size_in_node):
            self.shared_token_load.set_estimated_peak_token_count(0, dp_index)
            self.shared_token_load.set_frozened_token_count(0, dp_index)
            self.shared_token_load.set_current_load(0.0, dp_index)
            self.shared_token_load.set_logical_max_load(0.0, dp_index)
            self.shared_token_load.set_dynamic_max_load(0.0, dp_index)

        self.pause_strategy = Fcfs()
        self.running_batch: Batch = None
        self.has_wait_tokens = 0
        self.max_wait_tokens = args.router_max_wait_tokens

        context = zmq.asyncio.Context(2)
        self.recv_from_tts1_encode = context.socket(zmq.PULL)
        self.recv_from_tts1_encode.bind(f"{args.zmq_mode}127.0.0.1:{tts_llm_port}")

        self.send_to_tts_decode = context.socket(zmq.PUSH)
        self.send_to_tts_decode.connect(f"{args.zmq_mode}127.0.0.1:{tts_decode_port}")

        self.stats_tool = Stats(not args.disable_log_stats, args.log_stats_interval)

        self.shared_speech_manager = SharedSpeechManager(f"{args.port}_cosyvoice", args.cache_capacity)
        self.max_req_total_len = self.args.max_req_total_len
        self.max_total_token_num = args.max_total_token_num
        assert self.max_req_total_len <= self.max_total_token_num

        configs = load_yaml_lite(args.model_dir)
        self.model_config = configs["llm"].llm.model.model.config
        self.speech_token_size = configs["llm"].speech_token_size
        self.mix_ratio = configs["llm"].mix_ratio
        self.decode_token_hop_len = 25
        self.flow_pre_lookahead_len = configs["flow"].pre_lookahead_len
        self.vocab_size = self.model_config.vocab_size
        if configs["cosyvoice_version"] == CosyVoiceVersion.VERSION_3:
            self.embed_offset = self.vocab_size
        else:
            self.embed_offset = self.vocab_size + 2
        self.eos_id = configs["eos_token"]
        self.fill_token_id = configs["fill_token"]
        self.sos = configs["sos"]
        self.task_id = configs["task_id"]
        self.max_semantic_position = self.model_config.max_position_embeddings
        self.version = configs["cosyvoice_version"]
        del configs

        # 调度和推理进行折叠使用的线程池
        self.overlap_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.schedule_task = None
        # cpu和gpu overlap
        self.overlap_event = threading.Event()
        return

    async def wait_to_model_ready(self):
        # 初始化模型
        self.model_rpc_servers = []

        self.rpc_event = multiprocessing.Event()
        self.rpc_finished_event = multiprocessing.Event()

        for rank_id in range(self.world_size):
            rpc_model = await start_model_process(
                args=self.args,
                rank=rank_id,
                rank_in_node=0,
                node_world_size=1,
                rpc_event=self.rpc_event,
                rpc_finished_event=self.rpc_finished_event,
            )
            self.model_rpc_servers.append(rpc_model)

        # 相比于之前，现在只用单个model_rpc_client，通过shm和event号令所有server
        self.model_rpc_client = ModelRpcClient(
            model_infer_servers=self.model_rpc_servers,
            world_size=self.world_size,
            rpc_event=self.rpc_event,
            rpc_finished_event=self.rpc_finished_event,
        )

        kvargs = {
            "args": self.args,
            "rank_id": None,
            "world_size": self.world_size,
            "weight_dir": self.args.model_dir,
            "load_way": self.load_way,
            "max_total_token_num": self.max_total_token_num,
            "mode": self.mode,
            "max_req_num": self.args.running_max_req_size + 8,
            "max_seq_length": self.args.max_req_total_len + 8,  # 留一点余量
            "style_name": self.style_name,
            "cache_capacity": self.args.cache_capacity,
            "port": self.args.port,
            "speech_token_size": self.speech_token_size,
            "mix_ratio": self.mix_ratio,
            "batch_max_tokens": self.args.batch_max_tokens,
            "disable_cudagraph": self.args.disable_cudagraph,
            "graph_max_batch_size": self.args.graph_max_batch_size,
            "graph_max_len_in_batch": self.args.graph_max_len_in_batch,
            "data_type": getattr(self.args, "data_type", "float16"),
            "version": self.version,
        }
        await self.model_rpc_client.init_model(kvargs=kvargs)
        if self.max_total_token_num is None:
            self.max_total_token_num = await self.model_rpc_client.get_max_total_token_num()
            self.args.max_total_token_num = self.max_total_token_num

        self.req_queue = build_req_queue(self.args, self, self.dp_size_in_node)
        logger.info(f"use req queue {self.req_queue.__class__.__name__}")
        return

    def add_req(self, req: Req):
        req.start_time = time.time()
        if req.bistream:
            req.audio_ids = (
                (self.shared_speech_manager.get_index_speech_token(req.speech_index).arr[0] + self.embed_offset)
                .flatten()
                .tolist()
            )
            req.mix_ratio = self.mix_ratio
            req.next_fill_index = (int(len(req.audio_ids) / self.mix_ratio[1]) + 1) * self.mix_ratio[1] - len(
                req.audio_ids
            )
            self.req_queue.append_bistream(req)
        else:
            self.req_queue.append(req)
        self.send_to_tts_decode.send_pyobj(req.index_in_shm_mem, protocol=pickle.HIGHEST_PROTOCOL)
        return

    def check_and_wait_to_has_lock(self):
        if not self.has_lock:
            ans = self.gpt_parall_lock.acquire(block=True)
            assert ans is True
            self.has_lock = True
        self.parall_step_counter += 1
        return

    def check_and_release_lock(self):
        assert self.has_lock is True
        if self.has_lock and self.parall_step_counter >= self.parall_step_max_num:
            self.parall_step_counter = 0
            self.gpt_parall_lock.release()
            self.has_lock = False
        return

    def release_lock_when_all_finish(self):
        if self.has_lock:
            self.parall_step_counter = 0
            self.gpt_parall_lock.release()
            self.has_lock = False
        return

    async def loop_for_fwd(
        self,
    ):
        counter_count = 0
        while True:
            await self._step()
            counter_count += 1
            if self.running_batch is not None:
                if counter_count % 50 == 0:
                    token_ratio = self.get_used_tokens(0) / self.max_total_token_num
                    logger.debug(
                        f"current batch size: {len(self.running_batch.reqs)} \n"
                        f"paused req num: {self.req_queue.get_paused_req_num()} \n"
                        f"token used ratio: {token_ratio} \n"
                    )
                self.req_queue.update_token_load(self.running_batch, force_update=False)
                self.stats_tool.print_stats()
            else:
                self.req_queue.update_token_load(self.running_batch, force_update=True)

            if self.running_batch is None:
                self.release_lock_when_all_finish()
                await asyncio.sleep(0.01)  # 10ms

    async def get_schedule_result(self, running_batch: Batch):
        if self.schedule_task is None:

            def get_new_batch():
                limit_router_queue_length = None

                self.overlap_event.wait(timeout=0.020)
                self.overlap_event.clear()
                time.sleep(0.003)
                new_batch = self.req_queue.generate_new_batch(running_batch, limit_router_queue_length)
                return new_batch

            self.schedule_task = asyncio.get_running_loop().run_in_executor(self.overlap_thread_pool, get_new_batch)
        else:
            result = await self.schedule_task
            self.schedule_task = None
            return result

    async def _step(self):
        """
        事件处理循环
        """
        # 删除所有已经 finished 的 req
        # 当前无运行请求时
        if self.running_batch is None:
            new_batch = await self.get_schedule_result(self.running_batch)
            if new_batch is not None:
                self.stats_tool.count_prompt_tokens(new_batch)
                self.running_batch = new_batch
                await self._prefill_batch(self.running_batch)
                self._filter_runing_batch()
                self.has_wait_tokens = self.max_wait_tokens
            return

        # 有运行请求，当持续decode的次数到达一个阈值，或者有上次预调度的结果存在的时。
        if self.has_wait_tokens >= self.max_wait_tokens or self.schedule_task is not None:
            new_mini_batch = await self.get_schedule_result(self.running_batch)
            self.has_wait_tokens = 0
            if new_mini_batch is not None:
                self.has_wait_tokens = self.max_wait_tokens
                self.stats_tool.count_prompt_tokens(new_mini_batch)
                await self._prefill_batch(new_mini_batch)
                if not new_mini_batch.is_clear():
                    self.running_batch.merge(new_mini_batch)
                return

        # 正常 decode 阶段， 如果可以直接decode就直接decode，否则通过暂停策略暂停一些请求
        # 释放一些管理的 token
        if self._can_decode(self.running_batch):
            self.stats_tool.count_output_tokens(self.running_batch)
            await self._decode_batch(self.running_batch)
            self._filter_runing_batch()
            self.has_wait_tokens += 1
            return
        else:
            # pause strategy
            paused_reqs = select_paused_reqs(
                self.running_batch, self.pause_strategy, self.req_queue, self.max_total_token_num
            )
            await self._pause_reqs(paused_reqs)
            logger.debug(f"pasued req num: {self.req_queue.get_paused_req_num()}")
            self.has_wait_tokens = 0
            return
        return

    async def _prefill_batch(self, batch: Batch):
        self.check_and_wait_to_has_lock()
        reqs = [r.to_router_rpc_obj() for r in batch.reqs]
        self.overlap_event.set()
        await self.model_rpc_client.prefill(reqs)
        # 在 非 splitfsyncio.gather(*rets)
        self.check_and_release_lock()

        self._send_to_tts2_decodec_proc(batch)
        append_prefill_req_ids = [
            req.request_id for req in batch.reqs if req.req_status.get_status() == ReqRunStatus.WAIT_FOR_TEXT
        ]
        append_prefill_reqs = [batch.id_to_reqs[req_id] for req_id in append_prefill_req_ids]
        self.req_queue.waiting_req_bistream_list.extend(append_prefill_reqs)
        batch.filter_out_finished_req(self.shm_req_manager, append_prefill_req_ids)
        logger.debug(f"Prefill Batch: {batch.simple_log()} \n")
        return

    async def _decode_batch(self, batch: Batch):
        self.check_and_wait_to_has_lock()
        self.overlap_event.set()
        await self.model_rpc_client.decode()

        self.check_and_release_lock()
        self._send_to_tts2_decodec_proc(batch)
        append_prefill_req_ids = [
            req.request_id for req in batch.reqs if req.req_status.get_status() == ReqRunStatus.WAIT_FOR_TEXT
        ]
        append_prefill_reqs = [batch.id_to_reqs[req_id] for req_id in append_prefill_req_ids]
        self.req_queue.waiting_req_bistream_list.extend(append_prefill_reqs)
        batch.filter_out_finished_req(self.shm_req_manager, append_prefill_req_ids)
        return

    async def _filter_batch(self, batch: Batch, unfinished_req_ids, finished_req_ids: List, append_prefill_req_ids):
        rets = [
            self.model_rpcs[tp_rank].filter_batch(
                batch.batch_id, unfinished_req_ids, finished_req_ids, append_prefill_req_ids
            )
            for tp_rank in range(self.world_size)
        ]
        await asyncio.gather(*rets)
        return

    async def _merge_batch(self, batch1, batch2):
        rets = [
            self.model_rpcs[tp_rank].merge_batch(batch1.batch_id, batch2.batch_id) for tp_rank in range(self.world_size)
        ]
        await asyncio.gather(*rets)
        return

    async def _remove_batch(self, batch, append_prefill_req_ids):
        rets = [
            self.model_rpcs[tp_rank].remove_batch(batch.batch_id, append_prefill_req_ids)
            for tp_rank in range(self.world_size)
        ]
        await asyncio.gather(*rets)
        return

    async def _pause_reqs(self, pasue_reqs):
        pasue_req_ids = [r.request_id for r in pasue_reqs]
        await self.model_rpc_client.pause_reqs(pasue_req_ids)
        return

    def _filter_runing_batch(self):
        if self.running_batch is not None and self.running_batch.is_clear():
            self.running_batch = None
            return

    def _can_decode(self, batch: Batch):
        # return batch.get_batch_decode_need_tokens()[0] + self.get_used_tokens(0) <= self.max_total_token_num
        return True

    def get_used_tokens(self, dp_index):
        return self.max_total_token_num - self.read_only_statics_mem_manager.get_unrefed_token_num(dp_index)

    def _send_to_tts2_decodec_proc(self, batch: Batch):
        def process_send(req: Req):
            output_len = req.get_output_len()
            is_finished = req.finish_status.is_finished()
            self.send_to_tts_decode.send_pyobj((req.request_id, output_len), protocol=pickle.HIGHEST_PROTOCOL)
            if is_finished:
                logger.info(
                    f"Send:    tts_llm_{self.style_name} | req_id {req.request_id} | "
                    f"{output_len} token_ids bilv {output_len / max(1, req.input_len)}"
                )
                cost_time = (time.time() - req.start_time) * 1000
                logger.info(f"module tts_llm req_id {req.request_id} cost_time {cost_time} ms")
            else:
                logger.info(
                    f"Send:    tts_llm_{self.style_name} | req_id {req.request_id} | "
                    f"{output_len} token_ids | offset {offset}"
                )

        for req in batch.reqs:
            if req.router_aborted:
                continue
            if not req.stream:
                if req.finish_status.is_finished():
                    process_send(req)
            else:
                output_len = req.get_output_len()
                this_token_hop_len = (
                    self.decode_token_hop_len + req.prompt_token_pad
                    if req.token_offset == 0
                    else self.decode_token_hop_len
                )
                offset = this_token_hop_len + self.flow_pre_lookahead_len + req.token_offset
                if output_len == offset:
                    process_send(req)
                    req.token_offset += this_token_hop_len

                if req.finish_status.is_finished():
                    process_send(req)

    async def loop_for_netio_req(self):
        while True:
            recv_req = await self.recv_from_tts1_encode.recv_pyobj()
            if isinstance(recv_req, int):
                shm_req_index = recv_req
                req = self.shm_req_manager.get_req_obj_by_index(shm_req_index)

                logger.info(
                    f"Receive: tts_llm | req_id {req.request_id} | {req.input_len} {req.semantic_len} "
                    f"token_ids | speech_index: {req.speech_index}"
                )
                self.add_req(req)
            else:
                assert False, f"Error Req Inf {recv_req}"

    def clean_up(self):
        self.model_rpcs = None
        return


def start_tts_llm_process(args, tts_llm_port, tts_decode_port, style_name, gpt_parall_lock, pipe_writer):
    graceful_registry(inspect.currentframe().f_code.co_name)
    start_parent_check_thread()

    try:
        router = RouterManager(
            args,
            tts_llm_port=tts_llm_port,
            tts_decode_port=tts_decode_port,
            style_name=style_name,
            gpt_parall_lock=gpt_parall_lock,
        )

        asyncio.run(router.wait_to_model_ready())
    except Exception:
        import traceback
        import sys

        etype, evalue, tb = sys.exc_info()
        err_str = "\n".join(traceback.format_exception(etype, evalue, tb))
        logger.error(err_str)
        pipe_writer.send(err_str)
        router.clean_up()
        raise

    pipe_writer.send("init ok")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(router.loop_for_fwd())
    loop.run_until_complete(router.loop_for_netio_req())
    return
