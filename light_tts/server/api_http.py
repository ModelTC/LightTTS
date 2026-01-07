# Adapted from vllm/entrypoints/api_server.py
# of the vllm-project/vllm GitHub repository.
#
# Copyright 2023 ModelTC Team
# Copyright 2023 vLLM Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import io
import asyncio
import traceback
from typing import List
import torch
import uvloop
import hashlib
import base64
import subprocess
import time
import copy
import os
from pathlib import Path
import sys

# 抑制常见的库警告
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")
warnings.filterwarnings("ignore", category=UserWarning, message=".*weight_norm.*")
warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "cosyvoice"))
from light_tts.utils.config_utils import get_config_json
from light_tts.utils.load_utils import CosyVoiceVersion, load_yaml_frontend

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import argparse
import json
from http import HTTPStatus
import numpy as np
from fastapi import FastAPI, UploadFile, Form, File, BackgroundTasks, Request, WebSocketDisconnect, WebSocket
from fastapi.responses import Response, StreamingResponse, JSONResponse
import uvicorn
import soundfile as sf
from io import BytesIO
import multiprocessing as mp

from .httpserver.manager import HttpServerManager
from .tts_encode.manager import start_tts1_encode_process
from .tts_llm.manager import start_tts_llm_process
from .tts_decode.manager import start_tts_decode_process
from .req_id_generator import ReqIDGenerator

from light_tts.utils.net_utils import alloc_can_use_network_port
from light_tts.utils.param_utils import check_request
from light_tts.utils.health_utils import health_check
from light_tts.static_config import dict_language
from cosyvoice.utils.file_utils import load_wav
from light_tts.utils.envs_utils import get_env_start_args, get_unique_server_name
from light_tts.server.core.objs.sampling_params import SamplingParams
from light_tts.utils.start_utils import process_manager
from .metrics import histogram_timer
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from light_tts.utils.process_check import is_process_active
from prometheus_client import Counter, Histogram, generate_latest

all_request_counter = Counter("lightllm_request_count", "The total number of requests")
failure_request_counter = Counter("lightllm_request_failure", "The total number of requests")
sucess_request_counter = Counter("lightllm_request_success", "The total number of requests")
request_latency_histogram = Histogram("lightllm_request_latency", "Request latency", ["route"])
from dataclasses import dataclass
from .health_monitor import start_health_check_process
from light_tts.utils.log_utils import init_logger
from light_tts.server import TokenLoad


logger = init_logger(__name__)
g_id_gen = ReqIDGenerator()


@dataclass
class G_Objs:
    app: FastAPI = None
    args: object = None
    httpserver_manager: HttpServerManager = None
    shared_token_load: TokenLoad = None
    lora_styles: List[str] = None
    version: CosyVoiceVersion = None

    def set_args(self, args):
        self.args = args
        self.httpserver_manager = HttpServerManager(
            args, httpserver_port=args.httpserver_port, tts1_encode_ports=args.tts1_encode_ports
        )
        self.shared_token_load = TokenLoad(f"{get_unique_server_name()}_shared_token_load", 1)
        all_config = get_config_json(args.model_dir)
        self.lora_styles = [item["style_name"] for item in all_config["lora_info"]]
        configs = load_yaml_frontend(args.model_dir)
        self.version = configs["cosyvoice_version"]
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
        del self.frontend.feat_extractor
        del self.frontend.campplus_session
        del self.frontend.speech_tokenizer_session
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


g_objs = G_Objs()
app = FastAPI()
g_objs.app = app


def create_error_response(status_code: HTTPStatus, message: str) -> JSONResponse:
    return JSONResponse({"message": message}, status_code=status_code.value)


# 建议探测频率为60s的间隔，同时认为探测失败的超时时间为60s.连续3次探测失败则重启容器。
@app.get("/healthz")
@app.get("/health")
async def healthcheck():
    if os.environ.get("DEBUG_HEALTHCHECK_RETURN_FAIL") == "true":
        return JSONResponse({"message": "Error"}, status_code=404)

    if await health_check(g_objs.httpserver_manager, g_id_gen, g_objs.lora_styles, g_objs.version):
        return JSONResponse({"message": "Ok"}, status_code=200)
    else:
        return JSONResponse({"message": "Error"}, status_code=404)


@app.get("/liveness")
def liveness():
    return {"status": "ok"}


@app.get("/readiness")
def readiness():
    return {"status": "ok"}


def generate_data(model_output):
    for i in model_output:
        tts_audio = (i["tts_speech"] * (2 ** 15)).astype(np.int16).tobytes()
        yield tts_audio


async def generate_data_stream(generate_objs):
    for generator in generate_objs:
        async for i in generator:
            tts_audio = (i["tts_speech"] * (2 ** 15)).astype(np.int16).tobytes()
            yield tts_audio


def calculate_md5(file: UploadFile) -> str:
    hash_md5 = hashlib.md5()
    while chunk := file.read(8192):  # 分块读取以支持大文件
        hash_md5.update(chunk)
    return hash_md5.hexdigest()


async def send_wav(websocket: WebSocket, generator):
    async for result in generator:
        if isinstance(result, dict):
            await websocket.send_bytes((result["tts_speech"] * (2 ** 15)).astype(np.int16).tobytes())
            continue

        websocket.close()
        return


@app.websocket("/inference_zero_shot_bistream")
async def inference_zero_shot_bistream(websocket: WebSocket):
    # 接受 WebSocket 连接
    await websocket.accept()

    # 接收初始化参数
    init_params = await websocket.receive_json()

    # 解析固定参数
    prompt_text = init_params.get("prompt_text")
    tts_model_name = init_params.get("tts_model_name", "default")

    # 处理音频文件（假设客户端发送 base64 编码的音频）
    prompt_wav_data = await websocket.receive_bytes()
    wav_bytes_io = BytesIO(prompt_wav_data)
    prompt_speech_16k = load_wav(wav_bytes_io, 16000)
    semantic_len = (prompt_speech_16k.shape[1] + 239) // 640 + 10  # + 10 for safe

    wav_bytes_io.seek(0)
    speech_md5 = calculate_md5(wav_bytes_io)

    if tts_model_name == "default":
        tts_model_name = g_objs.lora_styles[0]

    speech_index, have_alloc = g_objs.httpserver_manager.alloc_speech_mem(speech_md5, prompt_speech_16k)
    request_id = g_id_gen.generate_id()
    first_text = True
    prompt_text = g_objs.frontend.text_normalize(prompt_text, split=False)
    process_task = None
    sample_params_dict = {}
    sampling_params = SamplingParams()
    sampling_params.init(**sample_params_dict)
    sampling_params.verify()
    try:
        while True:
            input_data = await websocket.receive_json()
            tts_text = input_data.get("tts_text", "")
            cur_req_dict = {
                "text": tts_text,
                "prompt_text": prompt_text,
                "tts_model_name": tts_model_name,
                "speech_md5": speech_md5,
                "need_extract_speech": first_text and not have_alloc,
                "stream": True,
                "speech_index": speech_index,
                "semantic_len": semantic_len,
                "bistream": True,
                "append": not first_text,
            }
            if input_data.get("finish", False):
                cur_req_dict["finish"] = True
                cur_req_dict["append"] = True
                await g_objs.httpserver_manager.append_bistream(cur_req_dict, request_id)
                break
            elif first_text:
                generator = g_objs.httpserver_manager.generate(cur_req_dict, request_id, sampling_params)
                process_task = asyncio.create_task(send_wav(websocket, generator))
                first_text = False
            else:
                await g_objs.httpserver_manager.append_bistream(cur_req_dict, request_id)
        await process_task
    except WebSocketDisconnect:
        # 处理客户端断开连接
        await g_objs.httpserver_manager.abort(request_id)
    except Exception as e:
        traceback.print_exc()
        await websocket.send_json({"error": f"Server error: {str(e)}"})
    finally:
        await websocket.close()
        if process_task is not None:
            process_task.cancel()


@histogram_timer(request_latency_histogram)
@app.post("/inference_zero_shot")
async def inference_zero_shot(
    request: Request,
    tts_text: str = Form(),
    prompt_text: str = Form(),
    prompt_wav: UploadFile = File(),
    stream: bool = Form(default=False),
    tts_model_name: str = Form(default="default"),
    speed: float = Form(default=1.0),
):
    all_request_counter.inc()

    if stream and speed != 1.0:
        return create_error_response(HTTPStatus.BAD_REQUEST, "speed change only support non-stream inference mode")

    if tts_model_name == "default":
        tts_model_name = g_objs.lora_styles[0]

    # 检查请求参数
    sample_params_dict = {}
    sampling_params = SamplingParams()
    sampling_params.init(**sample_params_dict)
    sampling_params.verify()
    prompt_text = g_objs.frontend.text_normalize(prompt_text, split=False)
    tts_texts = g_objs.frontend.text_normalize(tts_text, split=True)
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    semantic_len = (prompt_speech_16k.shape[1] + 239) // 640 + 10  # + 10 for safe

    prompt_wav.file.seek(0)
    speech_md5 = calculate_md5(prompt_wav.file)

    generate_objs = []
    need_extract_speech = True
    speech_index, have_alloc = g_objs.httpserver_manager.alloc_speech_mem(speech_md5, prompt_speech_16k)

    request_ids = []
    for text in tts_texts:
        cur_req_dict = {
            "text": text,
            "prompt_text": prompt_text,
            "tts_model_name": tts_model_name,
            "speech_md5": speech_md5,
            "need_extract_speech": need_extract_speech and not have_alloc,
            "stream": stream,
            "speech_index": speech_index,
            "semantic_len": semantic_len,
            "speed": speed,
        }
        need_extract_speech = False
        request_id = g_id_gen.generate_id()
        results_generator = g_objs.httpserver_manager.generate(
            cur_req_dict, request_id, sampling_params, request=request
        )
        generate_objs.append(results_generator)
        request_ids.append(request_id)

    print(f"split to request_ids: {request_ids}")
    if stream:
        try:
            return StreamingResponse(generate_data_stream(generate_objs))
        except Exception as e:
            logger.error("An error occurred: %s", str(e), exc_info=True)
            return create_error_response(HTTPStatus.EXPECTATION_FAILED, str(e))
    else:
        try:
            ans_objs = []
            for generator in generate_objs:
                async for result in generator:
                    ans_objs.append(result)
            return StreamingResponse(generate_data(ans_objs))
        except Exception as e:
            logger.error("An error occurred: %s", str(e), exc_info=True)
            return create_error_response(HTTPStatus.EXPECTATION_FAILED, str(e))


@app.post("/query_tts_model")
@app.get("/query_tts_model")
async def show_available_styles(request: Request) -> Response:
    data = {"tts_models": g_objs.lora_styles}
    json_data = json.dumps(data)
    return Response(content=json_data, media_type="application/json")


@app.get("/metrics")
async def metrics() -> Response:
    metrics_data = generate_latest()
    response = Response(metrics_data)
    response.mimetype = "text/plain"
    return response


@app.on_event("shutdown")
async def shutdown():
    logger.info("Received signal to shutdown. Performing graceful shutdown...")
    await asyncio.sleep(3)

    # 杀掉所有子进程
    import psutil
    import signal

    parent = psutil.Process(os.getpid())
    children = parent.children(recursive=True)
    for child in children:
        os.kill(child.pid, signal.SIGKILL)
    logger.info("Graceful shutdown completed.")
    return


@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("🚀 Application startup: Loading models...")
    loop = asyncio.get_event_loop()
    g_objs.set_args(get_env_start_args())
    loop.create_task(g_objs.httpserver_manager.handle_loop())

    # 启动时进行健康检查，确保系统正常工作
    logger.info("🔍 Running startup health check...")
    if await health_check(g_objs.httpserver_manager, g_id_gen, g_objs.lora_styles, g_objs.version):
        logger.info("✅ Startup health check passed!")
    else:
        logger.critical("❌ Startup health check failed! Server may not function correctly.")

    logger.info("✅ Application ready! Server is now accepting requests")
    logger.info(f"🌐 Listening at: http://{g_objs.args.host}:{g_objs.args.port}")
    logger.info("=" * 60)
    return
