import datetime
from itertools import chain
import zmq
import zmq.asyncio
import asyncio
import uvloop
import time
import sys

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from light_tts.utils.config_utils import get_config_json
from light_tts.utils.log_utils import init_logger
from light_tts.server.core.objs.shm_speech_manager import SharedSpeechManager
from light_tts.server.core.objs.shm_req_manager import ShmReqManager
from light_tts.server.core.objs import Req, FinishStatus
from typing import Union, List, Tuple, Dict, Optional
from light_tts.server.core.objs.io_objs import GroupReqObjs
import pickle
from light_tts.server.core.objs import SamplingParams
from fastapi import Request
from light_tts.utils.load_utils import load_yaml

logger = init_logger(__name__)


class HttpServerManager:
    def __init__(self, args, httpserver_port, tts1_encode_ports):
        self.args = args
        self.total_config = get_config_json(args.model_dir)

        self.send_to_tts1_encode_dict = {}

        context = zmq.asyncio.Context(2)

        for index, lora_item in enumerate(self.total_config["lora_info"]):
            tts_encode_port = tts1_encode_ports[index % args.encode_process_num]
            style_name = lora_item["style_name"]
            self.send_to_tts1_encode_dict[style_name] = context.socket(zmq.PUSH)
            self.send_to_tts1_encode_dict[style_name].connect(f"{args.zmq_mode}127.0.0.1:{tts_encode_port}")

        self.recv_from_tts_decode = context.socket(zmq.PULL)
        self.recv_from_tts_decode.bind(f"{args.zmq_mode}127.0.0.1:{httpserver_port}")

        self.req_id_to_out_inf: Dict[int, ReqStatus] = {}

        self.shared_speech_manager = SharedSpeechManager(f"{args.port}_cosyvoice", args.cache_capacity)
        self.shm_req_manager = ShmReqManager()

        configs = load_yaml(args.model_dir)
        self.model_config = configs["llm"].llm.model.model.config
        self.vocab_size = self.model_config.vocab_size
        self.sos = self.vocab_size + configs["sos"]
        self.task_id = self.vocab_size + configs["task_id"]

        self.max_req_total_len = args.max_req_total_len
        get_tokenizer = configs["get_tokenizer"]
        self.tokenizer = get_tokenizer()
        self.allowed_special = configs["allowed_special"]

        # from github source code
        self.min_token_text_ratio = 2
        self.max_token_text_ratio = 20
        return

    def alloc_speech_mem(self, speech_md5, prompt_wav):
        index, have_alloc = self.shared_speech_manager.alloc(speech_md5)
        if not have_alloc:
            self.shared_speech_manager.set_index_data(index, prompt_wav.shape, prompt_wav)
        return index, have_alloc

    async def append_bistream(self, request_dict, request_id):
        # 等待 request_id 出现在 dict 中（generate 可能还没执行完初始化）
        while request_id not in self.req_id_to_out_inf:
            await asyncio.sleep(0.01)

        req_status = self.req_id_to_out_inf[request_id]
        req = req_status.group_req_objs.shm_req_objs[0]
        finish = request_dict.get("finish", False)
        if finish:
            req.bistream_input_finished = True
        else:
            text = request_dict.get("text", "")
            text_ids = self._encode(text)
            with self.shm_req_manager.get_req_lock_by_index(req.index_in_shm_mem):
                req.append_bistream(text_ids, self.min_token_text_ratio, self.max_token_text_ratio)

    async def transfer_to_next_module(
        self,
        style,
        group_req_objs: Optional[GroupReqObjs] = None,
    ):
        self.send_to_tts1_encode_dict[style].send_pyobj(
            group_req_objs.to_group_req_index(),
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    async def _wait_to_token_package(
        self,
        request_id: int,
        req_status: "ReqStatus",
        request: Request = None,
    ):
        event = req_status.event

        while True:
            try:
                await asyncio.wait_for(event.wait(), timeout=1)
            except asyncio.TimeoutError:
                pass

            if request is not None and await request.is_disconnected():
                # Abort the request if the client disconnects.
                await self.abort(request_id)
                raise Exception(f"req_id {request_id} disconnected")

            async with req_status.lock:
                event.clear()
                logger.info(
                    f"req_id {request_id} get out data, len(out_data_info_list) {len(req_status.out_data_info_list)}"
                )
                if len(req_status.out_data_info_list) == 0:
                    continue

                for tts_speech, finish_status, finialize in req_status.out_data_info_list:
                    logger.debug(f"req_id {request_id} yield data {tts_speech.shape}")
                    yield tts_speech, finish_status, finialize
                    if finialize:
                        return
                req_status.out_data_info_list.clear()

    async def _log_req_header(self, request_headers, group_request_id: int):

        x_request_id = request_headers.get("X-Request-Id", "")
        x_session_id = request_headers.get("X-Session-Id", "")

        format_in_time = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")
        logger.info(
            f"recieved req X-Request-Id:{x_request_id} "
            f"X-Session-Id:{x_session_id} start_time:{format_in_time} "
            f"lightllm_req_id:{group_request_id} "
        )
        return

    def _encode(self, text: str):
        return self.tokenizer.encode(
            text,
            allowed_special=self.allowed_special,
        )

    async def _async_encode(self, text: str):
        return self._encode(text)

    async def _check_and_repair_length(self, prompt_ids: List[int], semantic_len: int, sampling_params: SamplingParams):
        prompt_tokens = len(prompt_ids) + semantic_len
        if prompt_tokens + sampling_params.max_new_tokens > self.max_req_total_len:
            sampling_params.max_new_tokens = self.max_req_total_len - prompt_tokens
        if sampling_params.max_new_tokens < sampling_params.min_new_tokens:
            raise ValueError("The input is too long and the resulting audio will be incomplete")

    async def generate(self, request_dict, request_id, sampling_params, request=None):
        # 记录请求到达的相关信息
        start_time = time.time()
        try:
            semantic_len = request_dict["semantic_len"]
            bistream = request_dict.get("bistream", False)
            speed = request_dict.get("speed", 1.0)
            request_headers = request.headers if request is not None else {}
            await self._log_req_header(request_headers, request_id)

            prompt_text_ids = await self._async_encode(request_dict["prompt_text"])
            text_ids = await self._async_encode(request_dict["text"])
            if not bistream:
                prompt_ids = list(chain([self.sos], prompt_text_ids, text_ids, [self.task_id]))
                sampling_params.min_new_tokens = len(text_ids) * self.min_token_text_ratio
                sampling_params.max_new_tokens = len(text_ids) * self.max_token_text_ratio
                await self._check_and_repair_length(prompt_ids, semantic_len, sampling_params)
            else:
                prompt_ids = prompt_text_ids + text_ids  # text_cache
                sampling_params.min_new_tokens = len(text_ids) * self.min_token_text_ratio
                sampling_params.max_new_tokens = len(text_ids) * self.max_token_text_ratio

            req_index = await self.shm_req_manager.async_alloc_req_index()
            while req_index is None:
                await asyncio.sleep(0.1)
                req_index = await self.shm_req_manager.async_alloc_req_index()

            style = request_dict["tts_model_name"]
            req_obj = await self.shm_req_manager.async_get_req_obj_by_index(req_index)
            req_objs = []
            req_obj.init(request_id, prompt_ids, request_dict, sampling_params, self.sos, self.task_id, speed)
            req_objs.append(req_obj)
            req_status = ReqStatus(request_id, req_objs, start_time)
            self.req_id_to_out_inf[request_id] = req_status

            await self.transfer_to_next_module(style, req_status.group_req_objs)

            results_generator = self._wait_to_token_package(
                request_id,
                req_status,
                request,
            )

            async for tts_speech, finish_status, finialize in results_generator:
                yield {"tts_speech": tts_speech}

        except Exception as e:
            logger.error(f"request_id: {request_id} has exception {str(e)}")
            await self.abort(request_id)
            raise e
        return

    async def abort(self, request_id):
        if request_id in self.req_id_to_out_inf:
            req_status = self.req_id_to_out_inf[request_id]
            group_req_objs: GroupReqObjs = req_status.group_req_objs
            for req in group_req_objs.shm_req_objs:
                req.is_aborted = True
            logger.warning(f"aborted request_id {group_req_objs.group_req_id}")
        else:
            logger.warning("aborted request_id not exist")
        return

    async def recycle_resource_loop(self):
        pre_time_mark = time.time()

        while True:
            try:
                await asyncio.wait_for(self.recycle_event.wait(), timeout=0.02)
            except asyncio.TimeoutError:
                pass
            self.recycle_event.clear()

            # 清理已经处理完的可以删除的请求
            release_req_status: List[ReqStatus] = []
            for group_req_id_ in list(self.req_id_to_out_inf.keys()):
                req_status: ReqStatus = self.req_id_to_out_inf.get(group_req_id_, None)
                if req_status is not None and req_status.can_release():
                    release_req_status.append(req_status)

            for req_status in release_req_status:
                logger.info(f"release req_id {req_status.group_req_objs.shm_req_objs[0].request_id}")
                self.req_id_to_out_inf.pop(req_status.group_req_objs.group_req_id, None)
                for req in req_status.group_req_objs.shm_req_objs:
                    await self.shm_req_manager.async_put_back_req_obj(req)
                    await self.shm_req_manager.async_release_req_index(req.index_in_shm_mem)

            if time.time() - pre_time_mark > 20:
                pre_time_mark = time.time()
                for req_status in self.req_id_to_out_inf.values():
                    logger.info(
                        f"left req_id {req_status.group_req_objs.group_req_id}"
                        f"can release {req_status.group_req_objs.shm_req_objs[0].can_released_mark} "
                        f"refcount {req_status.group_req_objs.shm_req_objs[0].ref_count}"
                    )

    async def handle_loop(self):
        self.recycle_event = asyncio.Event()
        asyncio.create_task(self.recycle_resource_loop())

        while True:
            try:
                await asyncio.wait_for(self.recv_from_tts_decode.recv_pyobj(), timeout=0.05)
            except asyncio.TimeoutError:
                pass

            for group_req_id_ in list(self.req_id_to_out_inf.keys()):
                req_status = self.req_id_to_out_inf.get(group_req_id_, None)
                if req_status is None:
                    continue

                for req in req_status.group_req_objs.shm_req_objs:
                    if req.stream:
                        if not req.out_tokens_queue.is_empty():
                            tts_speech, token_offset, finalize = req.out_tokens_queue.peek()
                            tts_speech = tts_speech.copy()

                            if finalize:
                                finish_status = FinishStatus(req.finish_status.status)
                            else:
                                finish_status = FinishStatus()

                            req.out_tokens_queue.pop_no_ret()
                            async with req_status.lock:
                                logger.debug(f"req_id {req.request_id} shm_index {req.index_in_shm_mem} get chunk")
                                req_status.out_data_info_list.append((tts_speech, finish_status, finalize))
                                req_status.event.set()
                    elif req.gen_finished:
                        tts_speech = req.get_gen_audios()
                        finalize = True
                        finish_status = FinishStatus(req.finish_status.status)
                        async with req_status.lock:
                            req_status.out_data_info_list.append((tts_speech.copy(), finish_status, finalize))
                            logger.info(f"req_id {req.request_id} shm_index {req.index_in_shm_mem} gen finished")
                            req_status.event.set()

            self.recycle_event.set()
        return


class ReqStatus:
    def __init__(self, req_id, req_objs: List[Req], start_time) -> None:
        self.lock = asyncio.Lock()
        self.event = asyncio.Event()
        self.group_req_objs = GroupReqObjs(
            group_req_id=req_id,
            shm_req_objs=req_objs,
            time_mark=start_time,
        )
        self.out_data_info_list = []

    def can_release(self):
        for req in self.group_req_objs.shm_req_objs:
            if not req.can_release():
                return False
        return True
