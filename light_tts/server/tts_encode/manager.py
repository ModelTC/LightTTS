import pickle
import numpy as np
from light_tts.server.core.objs.io_objs.group_req import GroupReqIndexes
from light_tts.server.core.objs.shm_req_manager import ShmReqManager, Req
import torch
import zmq
import zmq.asyncio
import asyncio
import time
import uvloop

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from light_tts.utils.config_utils import get_config_json
from light_tts.utils.log_utils import init_logger
from light_tts.server.core.objs.shm_speech_manager import SharedSpeechManager
from light_tts.utils.load_utils import CosyVoiceVersion, load_yaml_frontend
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from light_tts.utils.graceful_utils import graceful_registry
from light_tts.utils.process_check import start_parent_check_thread
import inspect
from typing import List

logger = init_logger(__name__)


class TTS1EncodeManager:
    def __init__(
        self,
        args,
        tts_llm_ports,
        tts1_encode_port,
        index_id,
        encode_parall_lock,
    ):
        self.index_id = index_id
        self.encode_parall_lock = encode_parall_lock

        context = zmq.asyncio.Context(2)

        self.recv_from_httpserver = context.socket(zmq.PULL)
        self.recv_from_httpserver.bind(f"{args.zmq_mode}127.0.0.1:{tts1_encode_port}")
        self.waiting_reqs: List[Req] = []
        self.model_cfg = get_config_json(args.model_dir)
        self.send_to_tts_llms = {}
        for i, lora_w in enumerate(self.model_cfg["lora_info"]):
            # context = zmq.asyncio.Context()
            if i % args.encode_process_num == self.index_id:  # 通过id分配他需要处理的lora
                send_to_tts_llm_i = context.socket(zmq.PUSH)
                send_to_tts_llm_i.connect(f"{args.zmq_mode}127.0.0.1:{tts_llm_ports[i]}")
                self.send_to_tts_llms[lora_w["style_name"]] = send_to_tts_llm_i
        # 只能有一个进程刷新内部的标记值
        self.shared_speech_manager = SharedSpeechManager(f"{args.port}_cosyvoice", args.cache_capacity)
        self.shm_req_manager = ShmReqManager()

        self.world_size = 1
        self.trust_remote_code = args.trust_remote_code
        self.args = args

        configs = load_yaml_frontend(args.model_dir)
        self.configs = configs
        self.resample_rate = configs["sample_rate"]
        self.token_hop_len = 25

        if configs["cosyvoice_version"] == CosyVoiceVersion.VERSION_2:
            speech_tokenizer_model = "{}/speech_tokenizer_v2.onnx".format(args.model_dir)
        else:
            speech_tokenizer_model = "{}/speech_tokenizer_v3.onnx".format(args.model_dir)

        self.frontend = CosyVoiceFrontEnd(
            configs["get_tokenizer"],
            configs["feat_extractor"],
            "{}/campplus.onnx".format(args.model_dir),
            speech_tokenizer_model,
            "{}/spk2info.pt".format(args.model_dir),
            configs["allowed_special"],
        )
        self.model_config = configs["llm"].llm.model.model.config
        vocab_size = self.model_config.vocab_size
        if configs["cosyvoice_version"] == CosyVoiceVersion.VERSION_3:
            self.embed_offset = vocab_size
        else:
            self.embed_offset = vocab_size + 2
        del configs

    def add_req(self, group_req_indexes: GroupReqIndexes):
        req_group = []
        for req_index in group_req_indexes.shm_req_indexes:
            req = self.shm_req_manager.get_req_obj_by_index(req_index)
            req.start_time = group_req_indexes.time_mark
            req_group.append(req)

            logger.info(f"tts_encode recive req_id {req.request_id} cost time {time.time() - req.start_time} s")
        self.waiting_reqs.extend(req_group)

    async def loop_for_fwd(self):
        module_name = "tts1_encoder"
        idle_count = 0
        while True:
            if len(self.waiting_reqs) == 0:
                await asyncio.sleep(0.01)  # 10ms
                idle_count -= 1
                if idle_count == 0:
                    torch.cuda.empty_cache()
            else:
                idle_count = 1000
                n = len(self.waiting_reqs)
                while n > 0:
                    req = self.waiting_reqs.pop(0)
                    if req.is_aborted:
                        n -= 1
                        req.router_aborted = True
                        self.shm_req_manager.put_back_req_obj(req)
                        req.can_released_mark = True
                        continue
                    tts_model_name = "CosyVoice2"
                    speech_index = req.speech_index
                    need_extract_speech = req.need_extract_speech

                    n -= 1

                    if need_extract_speech:
                        logger.debug(f"tts_encode req_id {req.request_id} generate speech index {speech_index} cache")
                        prompt_speech_16k = self.shared_speech_manager.get_index_data(speech_index)
                        if prompt_speech_16k is None:
                            raise RuntimeError(f"In encode, get_index_data {speech_index} not found")
                        prompt_speech_16k = torch.from_numpy(prompt_speech_16k.arr)
                        model_input = self.frontend.frontend_zero_shot(
                            "", "", prompt_speech_16k, self.resample_rate, ""
                        )
                        speech_token = model_input["llm_prompt_speech_token"].cpu().numpy()
                        speech_feat = model_input["prompt_speech_feat"].squeeze(0).cpu().numpy()
                        embedding = model_input["llm_embedding"].cpu().numpy()
                        self.shared_speech_manager.set_index_speech(speech_index, speech_token, speech_feat, embedding)
                    else:
                        if not self.shared_speech_manager.speech_data_ready(speech_index):
                            self.waiting_reqs.append(req)
                            continue
                        else:
                            logger.debug(f"tts_encode req_id {req.request_id} use speech index {speech_index} cache")
                            speech_token = self.shared_speech_manager.get_index_speech_token(speech_index).arr[0]

                    if not req.bistream:
                        speech_token = speech_token + self.embed_offset
                        audio_ids = speech_token.flatten().tolist()
                        with self.shm_req_manager.get_req_lock_by_index(req.index_in_shm_mem):
                            req.set_speech_token(audio_ids)

                    req.prompt_token_pad = int(
                        np.ceil(speech_token.size / self.token_hop_len) * self.token_hop_len - speech_token.size
                    )
                    logger.info(
                        f"Send:    {module_name:<14} | req_id {req.request_id} | "
                        f"semantic length {req.semantic_len} | text length {req.text_len} to tts_llm"
                    )
                    self.shm_req_manager.put_back_req_obj(req)
                    self.send_to_tts_llms[tts_model_name].send_pyobj(req.index_in_shm_mem)
                    cost_time = (time.time() - req.start_time) * 1000
                    logger.info(f"module {module_name} req_id {req.request_id} cost_time {cost_time} ms")

    async def loop_for_netio_req(self):
        while True:
            recv_req = await self.recv_from_httpserver.recv_pyobj()

            if isinstance(recv_req, GroupReqIndexes):
                self.add_req(recv_req)
            else:
                assert False, f"Error Req Inf {recv_req}"

    def clean_up(self):
        self.model_rpcs = None
        return


def start_tts1_encode_process(args, tts_llm_ports, tts1_encode_port, index_id, encode_parall_lock, pipe_writer):
    # 注册 graceful 退出的处理
    # 返回当前这行代码的位置的代码对象的当前函数名字
    graceful_registry(inspect.currentframe().f_code.co_name)
    start_parent_check_thread()

    try:
        encodeserver = TTS1EncodeManager(args, tts_llm_ports, tts1_encode_port, index_id, encode_parall_lock)
    except Exception:
        import traceback
        import sys

        etype, evalue, tb = sys.exc_info()
        err_str = "\n".join(traceback.format_exception(etype, evalue, tb))
        logger.error(err_str)
        pipe_writer.send(err_str)
        encodeserver.clean_up()
        raise

    pipe_writer.send("init ok")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(encodeserver.loop_for_fwd())
    loop.run_until_complete(encodeserver.loop_for_netio_req())
    return
