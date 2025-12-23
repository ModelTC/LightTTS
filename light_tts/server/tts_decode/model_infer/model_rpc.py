from collections import defaultdict
import rpyc
import torch
import json
import os
import numpy as np
from light_tts.utils.infer_utils import set_random_seed
from light_tts.utils.log_utils import init_logger
from typing import List
from ..decode_req import DecodeReq
from light_tts.utils.load_utils import CosyVoiceVersion, load_yaml
from light_tts.server.tts_decode.model_infer import patch_conditional_cfm
from cosyvoice.cli.model import CosyVoice2Model, CosyVoice3Model
from light_tts.server.core.objs.shm_speech_manager import SharedSpeechManager

logger = init_logger(__name__)


class TTS2DecodeModelRpcServer:
    def init_model(self, kvargs):
        gpu_id = kvargs["gpu_id"]
        model_dir = kvargs["model_dir"]
        torch.cuda.set_device(gpu_id)

        configs = load_yaml(model_dir)
        version = configs["cosyvoice_version"]

        if version == CosyVoiceVersion.VERSION_2:
            self.fp16 = True
            self.model = CosyVoice2Model(configs["llm"], configs["flow"], configs["hift"], fp16=self.fp16)
        elif version == CosyVoiceVersion.VERSION_3:
            self.fp16 = False
            self.model = CosyVoice3Model(configs["llm"], configs["flow"], configs["hift"], fp16=self.fp16)
        self.model.load("{}/llm.pt".format(model_dir), "{}/flow.pt".format(model_dir), "{}/hift.pt".format(model_dir))

        load_jit = kvargs.get("load_jit", False)
        load_trt = kvargs.get("load_trt", False)
        trt_concurrent = 1

        if version == CosyVoiceVersion.VERSION_3:
            load_jit = False

        if load_jit:
            self.model.load_jit("{}/flow.encoder.{}.zip".format(model_dir, "fp16" if self.fp16 is True else "fp32"))
        if load_trt:
            capability = torch.cuda.get_device_capability(0)
            self.model.load_trt(
                "{}/flow.decoder.estimator.{}.sm{}{}.plan".format(
                    model_dir, "fp16" if self.fp16 is True else "fp32", capability[0], capability[1]
                ),
                "{}/flow.decoder.estimator.fp32.onnx".format(model_dir),
                trt_concurrent,
                self.fp16,
            )
        self.model.hift_cache_dict = defaultdict(lambda: None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        set_random_seed(2147483647)
        self.shared_speech_manager = SharedSpeechManager(f"{kvargs['port']}_cosyvoice", kvargs["shared_cache_capacity"])
        del configs

    # @calculate_time(show=True, min_cost_ms=150)
    @torch.no_grad()
    def forward(self, batch: List[DecodeReq]):
        for decode_req in batch:
            output_ids, speech_index, request_id, token_offset, finalize = decode_req.get_infer_data()
            speech_token, speech_feat, spk_embedding = self.shared_speech_manager.get_index_speech(speech_index)
            speech_token, speech_feat, spk_embedding = speech_token.arr, speech_feat.arr, spk_embedding.arr
            logger.info(f"req_id {request_id} start decode")
            tts_speech = self.model.token2wav(
                torch.tensor(output_ids, device="cuda").unsqueeze(0),
                torch.as_tensor(speech_token, device="cuda"),
                torch.as_tensor(speech_feat, device="cuda").unsqueeze(0),
                torch.as_tensor(spk_embedding, device="cuda"),
                token_offset,
                request_id,
                stream=decode_req.req.stream,
                finalize=finalize,
                speed=decode_req.req.speed,
            )
            decode_req.update_one_decode(finalize)
            tts_speech = tts_speech.view(-1).cpu().numpy()
            if decode_req.req.stream:
                decode_req.req.out_tokens_queue.push(tts_speech, token_offset, finalize)
                logger.info(f"req_id {request_id} decode stream and push")
            else:
                logger.info(f"req_id {request_id} decode set_gen_audios")
                decode_req.req.set_gen_audios(tts_speech)
            if finalize:
                self.model.hift_cache_dict.pop(request_id)
        return

    def decode(self, batch: List[DecodeReq]):
        return self.forward(batch)


class TTS2DecodeModelRpcClient:
    def __init__(self, model_rpc):
        self.model_infer_server: TTS2DecodeModelRpcServer = model_rpc
        return

    async def init_model(self, kvargs):
        self.model_infer_server.init_model(kvargs)
        return

    async def decode(self, batch: List[DecodeReq]):
        return self.model_infer_server.decode(batch)


async def start_model_process():
    return TTS2DecodeModelRpcClient(TTS2DecodeModelRpcServer())
