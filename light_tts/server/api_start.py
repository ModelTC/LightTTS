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
import asyncio
import uvloop
import subprocess
import time
import os
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import multiprocessing as mp
from .httpserver.manager import HttpServerManager
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'cosyvoice'))
from .tts_encode.manager import start_tts1_encode_process
from .tts_llm.manager import start_tts_llm_process
from .tts_decode.manager import start_tts_decode_process

from light_tts.utils.net_utils import alloc_can_use_network_port, PortLocker
from light_tts.utils.envs_utils import get_unique_server_name, set_env_start_args, set_unique_server_name
from light_tts.utils.start_utils import process_manager
import signal
import sys
import uvicorn
from light_tts.utils.process_check import is_process_active
from .health_monitor import start_health_check_process
from light_tts.utils.log_utils import init_logger

logger = init_logger(__name__)

def setup_signal_handlers(http_server_process, process_manager):
    def signal_handler(sig, frame):
        if sig == signal.SIGINT:
            logger.info("Received SIGINT (Ctrl+C), forcing immediate exit...")
            if http_server_process and http_server_process.poll() is None:
                http_server_process.kill()

            process_manager.terminate_all_processes()
            logger.info("All processes have been forcefully terminated.")
            sys.exit(0)
        elif sig == signal.SIGTERM:
            logger.info("Received SIGTERM, shutting down gracefully...")
            if http_server_process and http_server_process.poll() is None:
                http_server_process.send_signal(signal.SIGTERM)

                start_time = time.time()
                while (time.time() - start_time) < 60:
                    if not is_process_active(http_server_process.pid):
                        logger.info("httpserver exit")
                        break
                    time.sleep(1)

                if time.time() - start_time < 60:
                    logger.info("HTTP server has exited gracefully")
                else:
                    logger.warning("HTTP server did not exit in time, killing it...")
                    http_server_process.kill()

            process_manager.terminate_all_processes()
            logger.info("All processes have been terminated gracefully.")
            sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    logger.info(f"start process pid {os.getpid()}")
    logger.info(f"http server pid {http_server_process.pid}")
    return

def normal_start(args):
    set_unique_server_name(args)
    
    assert args.max_req_total_len <= args.max_total_token_num
    assert args.zmq_mode in ["tcp://", "ipc:///tmp/"]
    # 确保单机上多实列不冲突
    if args.zmq_mode == "ipc:///tmp/":
        zmq_mode = f"{args.zmq_mode}_{get_unique_server_name()}_"
        args.zmq_mode = None  # args 的参数不能直接设置，只能先设置None，再设置才能成功
        args.zmq_mode = zmq_mode
        logger.info(f"zmq mode head: {args.zmq_mode}")

    # 普通模式下
    if args.batch_max_tokens is None:
        batch_max_tokens = int(1 / 6 * args.max_total_token_num)
        batch_max_tokens = max(batch_max_tokens, args.max_req_total_len)
        args.batch_max_tokens = batch_max_tokens
    else:
        assert (
            args.batch_max_tokens >= args.max_req_total_len
        ), "batch_max_tokens must >= max_req_total_len"

    # 提前锁定端口，防止在单个机器上启动多个实列的时候，要到模型启动的时候才能
    # 捕获到端口设置冲突的问题
    ports_locker = PortLocker([args.port])
    ports_locker.lock_port()

    num_loras = 1
    assert args.decode_process_num <= num_loras

    can_use_ports = alloc_can_use_network_port(
        num=num_loras * 2 + args.encode_process_num + 100, used_nccl_port=None
    )

    httpserver_port = can_use_ports[0]
    del can_use_ports[0]
    tts1_encode_ports = can_use_ports[0:args.encode_process_num]
    del can_use_ports[0:args.encode_process_num]
    
    args.httpserver_port = httpserver_port
    args.tts1_encode_ports = tts1_encode_ports

    tts_llm_ports = can_use_ports[0 : num_loras]
    del can_use_ports[0 : num_loras]

    set_env_start_args(args)
    logger.info(f"all start args:{args}")
    ports_locker.release_port()
    
    logger.info("=" * 80)
    logger.info("🚀 Starting LightTTS Server - FULL PARALLEL Mode")
    logger.info("⚡ All 3 components will start simultaneously!")
    logger.info("=" * 80)
    
    tts_decode_ports = can_use_ports[0 : num_loras]
    del can_use_ports[0 : num_loras]
    
    # 准备所有进程参数
    all_funcs = []
    all_start_args = []
    
    # 1️⃣ 准备 Encode 进程
    encode_parall_lock = mp.Semaphore(args.encode_paral_num)
    for index_id in range(args.encode_process_num):
        all_funcs.append(start_tts1_encode_process)
        all_start_args.append((args, tts_llm_ports, tts1_encode_ports[index_id], index_id, encode_parall_lock))
    
    # 2️⃣ 准备 LLM 进程
    gpt_parall_lock = mp.Semaphore(args.gpt_paral_num)
    for style_name, tts_llm_port, tts_decode_port in zip(["CosyVoice2"], tts_llm_ports, tts_decode_ports): 
        all_funcs.append(start_tts_llm_process)
        all_start_args.append((args, tts_llm_port, tts_decode_port, style_name, gpt_parall_lock))
    
    # 3️⃣ 准备 Decode 进程
    decode_parall_lock = mp.Semaphore(args.decode_paral_num)
    for decode_proc_index in range(args.decode_process_num):
        tmp_args = []
        for style_name, tts_decode_port in zip(["CosyVoice2"], tts_decode_ports):
            tmp_args.append((args, tts_decode_port, httpserver_port, style_name, decode_parall_lock, decode_proc_index))
        all_funcs.append(start_tts_decode_process)
        all_start_args.append((tmp_args,))
    
    # 🔥 激进模式：一次性启动 Encode + LLM + Decode，完全并行初始化！
    logger.info(f"🚀 Launching {len(all_funcs)} processes in parallel...")
    process_manager.start_submodule_processes(start_funcs=all_funcs, start_args=all_start_args)
    
    logger.info("✅ All components started successfully!")
    logger.info("=" * 80)

    if os.getenv("LIGHTLLM_DEBUG") == "1":
        from light_tts.server.api_http import app
        server = uvicorn.Server(uvicorn.Config(app))
        server.install_signal_handlers()
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="debug",
            timeout_keep_alive=5,
            loop="uvloop",
        )
    else:
        command = [
            "gunicorn",
            "--workers",
            f"{args.httpserver_workers}",
            "--worker-class",
            "uvicorn.workers.UvicornWorker",
            "--bind",
            f"{args.host}:{args.port}",
            "--log-level",
            "warning",  # 降低gunicorn日志级别，避免误导性的"Listening"消息
            "--access-logfile",
            "-",
            "--error-logfile",
            "-",
            "light_tts.server.api_http:app",
        ]

        # 启动子进程
        http_server_process = subprocess.Popen(command)

        if args.health_monitor:
            process_manager.start_submodule_processes(start_funcs=[start_health_check_process], start_args=[(args,)])

        setup_signal_handlers(http_server_process, process_manager)
        http_server_process.wait()
    return
