import inspect
import pickle
from light_tts.server.core.objs.shm_req_manager import ShmReqManager
from light_tts.utils.graceful_utils import graceful_registry
import traceback
import torch
import torch.nn.functional as F
import time
import uvloop
import asyncio

from light_tts.utils.load_utils import load_yaml_lite
from light_tts.utils.process_check import start_parent_check_thread

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import zmq
import zmq.asyncio
from typing import Dict, Union
from .model_infer.model_rpc import start_model_process, TTS2DecodeModelRpcClient

from light_tts.utils.infer_utils import calculate_time, mark_start, mark_end
from light_tts.utils.log_utils import init_logger
from .decode_req import DecodeReq
from typing import List

logger = init_logger(__name__)


class TTSDecodeManager:
    def __init__(self, args, tts_decode_port, httpserver_port, style_name, decode_parall_lock, decode_proc_index):
        self.args = args
        context = zmq.asyncio.Context(2)
        self.recv_from_tts2_gpt = context.socket(zmq.PULL)
        self.recv_from_tts2_gpt.bind(f"{args.zmq_mode}127.0.0.1:{tts_decode_port}")

        self.send_to_httpserver = context.socket(zmq.PUSH)
        self.send_to_httpserver.connect(f"{args.zmq_mode}127.0.0.1:{httpserver_port}")
        self.style_name = style_name
        self.waiting_reqs = []
        self.decode_parall_lock = decode_parall_lock
        self.decode_proc_index = decode_proc_index

        self.req_id_to_out: Dict[int, DecodeReq] = {}
        self.new_req_id_to_req: Dict[int, DecodeReq] = {}
        self.shm_req_manager = ShmReqManager()

        configs = load_yaml_lite(args.model_dir)
        self.decode_token_hop_len = 25
        self.flow_pre_lookahead_len = configs["flow"].pre_lookahead_len
        self.waiting_reqs = []
        self.speech_token_size = configs["llm"].speech_token_size
        self.eos_id = configs["eos_token"]
        self.decode_max_batch_size = 1

    async def wait_to_model_ready(self):
        gpu_num = torch.cuda.device_count()
        self.gpu_id = self.decode_proc_index % gpu_num

        self.rpc_model = await start_model_process()

        kvargs = {
            "gpu_id": self.gpu_id,
            "model_dir": self.args.model_dir,
            "port": self.args.port,
            "shared_cache_capacity": self.args.cache_capacity,
            "load_jit": self.args.load_jit,
            "load_trt": self.args.load_trt,
        }
        await self.rpc_model.init_model(kvargs)
        return

    async def infer_decodec_batch(self, batch):
        await self.rpc_model.decode(batch)
        return

    def get_batch(self):
        if len(self.waiting_reqs) == 0:
            return []
        batch = []
        appended_reqs = []
        while len(self.waiting_reqs) != 0:
            request_id = self.waiting_reqs.pop(0)
            decode_req = self.req_id_to_out[request_id]
            if decode_req.out_queue_is_full():
                appended_reqs.append(request_id)
                continue
            batch.append(decode_req)
            # 同时进行推理的请求数量限制
            if len(batch) >= self.decode_max_batch_size:
                break
        self.waiting_reqs += appended_reqs
        return batch

    def remove_finished_reqs(self, batch: List[DecodeReq]):
        finished_reqs: List[DecodeReq] = []
        for decode_req in batch:
            if decode_req.can_set_release_mark():
                finished_reqs.append(decode_req)

        for decode_req in finished_reqs:
            decode_req.req.can_released_mark = True
            logger.info(f"detoken release req_id {decode_req.req.request_id}")
            logger.info(
                f"req_id {decode_req.req.request_id}, stream {decode_req.req.stream}, "
                f"finished {decode_req.req.finish_status.is_finished()}"
            )
            self.shm_req_manager.put_back_req_obj(decode_req.req)
            self.req_id_to_out.pop(decode_req.request_id, None)
        return

    async def loop_for_fwd(self):
        idle_count = 1000
        while True:
            if len(self.waiting_reqs) == 0:
                await asyncio.sleep(0.01)  # 10ms
            else:
                while len(self.waiting_reqs) > 0:
                    batch: List[DecodeReq] = self.get_batch()
                    try:
                        await self.infer_decodec_batch(batch)
                        self.send_to_httpserver.send_pyobj(None, protocol=pickle.HIGHEST_PROTOCOL)
                        logger.debug(f"decode send to httpserver req_id {batch[0].req.request_id}")
                    except Exception as e:
                        logger.exception(str(e))
                    idle_count -= 1
                    if idle_count <= 0:
                        torch.cuda.empty_cache()
                        torch.cuda.current_stream().synchronize()
                        idle_count = 1000
                    logger.debug(f"{self.style_name} current waiting queue in tts_decode: {len(self.waiting_reqs)}")
                    self.remove_finished_reqs(batch)

    async def handle_loop(self):
        # asyncio.create_task(self.timer_to_detoken())
        while True:
            try:
                recv_obj = await self.recv_from_tts2_gpt.recv_pyobj()

                if isinstance(recv_obj, int):
                    shm_req_index = recv_obj
                    req = self.shm_req_manager.get_req_obj_by_index(shm_req_index)
                    logger.info(f"decode_manager recv req_id {req.request_id}")

                    decode_req = DecodeReq(req, self.decode_token_hop_len, self.flow_pre_lookahead_len, self.eos_id)
                    self.req_id_to_out[req.request_id] = decode_req

                elif isinstance(recv_obj, tuple):
                    logger.info(f"Receive: | req_id: {recv_obj[0]} | {recv_obj[1]} token_ids")
                    self.waiting_reqs.append(recv_obj[0])
                else:
                    assert False, f"Error Req Inf {recv_obj}"

            except Exception as e:
                logger.error(f"decode process has exception {str(e)}")
                traceback.print_exc()
                pass


def start_tts_decode_process(params_list, pipe_writer):
    graceful_registry(inspect.currentframe().f_code.co_name)

    torch.backends.cudnn.enabled = True
    managers = []
    try:
        for params in params_list:
            args, tts_decode_port, httpserver_port, style_name, decode_parall_lock, decode_proc_index = params

            tts_decodec = TTSDecodeManager(
                args,
                tts_decode_port=tts_decode_port,
                httpserver_port=httpserver_port,
                style_name=style_name,
                decode_parall_lock=decode_parall_lock,
                decode_proc_index=decode_proc_index,
            )
            asyncio.run(tts_decodec.wait_to_model_ready())
            managers.append(tts_decodec)
    except Exception:
        import traceback
        import sys

        etype, evalue, tb = sys.exc_info()
        err_str = "\n".join(traceback.format_exception(etype, evalue, tb))
        logger.error(err_str)
        pipe_writer.send(err_str)
        raise

    pipe_writer.send("init ok")
    loop = asyncio.new_event_loop()
    for manager in managers:
        loop.create_task(manager.loop_for_fwd())
        loop.create_task(manager.handle_loop())

    loop.run_forever()
    return
