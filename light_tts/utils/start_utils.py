import sys
import multiprocessing as mp
import psutil
from light_tts.utils.log_utils import init_logger
import traceback
from pathlib import Path
import sys

logger = init_logger(__name__)


class SubmoduleManager:
    def __init__(self):
        self.processes = []

    def start_submodule_processes(self, start_funcs=[], start_args=[]):
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'cosyvoice'))  # 具体看你的目录结构
        assert len(start_funcs) == len(start_args)
        pipe_readers = []
        processes = []
        
        # 进程名称映射（用于更友好的日志）
        process_names = {
            'start_tts1_encode_process': '📝 Text Encoder',
            'start_tts_llm_process': '🎯 LLM Model',
            'start_tts_decode_process': '🎵 Audio Decoder',
            'start_health_check_process': '❤️ Health Monitor'
        }

        for start_func, start_arg in zip(start_funcs, start_args):
            pipe_reader, pipe_writer = mp.Pipe(duplex=False)
            process = mp.Process(
                target=start_func,
                args=start_arg + (pipe_writer,),
            )
            process.start()
            pipe_readers.append(pipe_reader)
            processes.append(process)
            
            # 打印启动提示
            func_name = start_func.__name__
            display_name = process_names.get(func_name, func_name)
            logger.info(f"⏳ Starting {display_name}...")

        # Wait for all processes to initialize
        for index, pipe_reader in enumerate(pipe_readers):
            init_state = pipe_reader.recv()
            func_name = start_funcs[index].__name__
            display_name = process_names.get(func_name, func_name)
            
            if init_state != "init ok":
                logger.error(f"❌ {display_name} failed: {str(init_state)}")
                for proc in processes:
                    proc.kill()
                sys.exit(1)
            else:
                logger.info(f"✅ {display_name} initialized successfully")

        assert all([proc.is_alive() for proc in processes])
        self.processes.extend(processes)
        return

    def terminate_all_processes(self):
        def kill_recursive(proc):
            try:
                parent = psutil.Process(proc.pid)
                children = parent.children(recursive=True)
                for child in children:
                    logger.info(f"Killing child process {child.pid}")
                    child.kill()
                logger.info(f"Killing parent process {proc.pid}")
                parent.kill()
            except psutil.NoSuchProcess:
                logger.warning(f"Process {proc.pid} does not exist.")

        for proc in self.processes:
            if proc.is_alive():
                kill_recursive(proc)
                proc.join()
        logger.info("All processes terminated gracefully.")


def start_submodule_processes(start_funcs=[], start_args=[]):
    assert len(start_funcs) == len(start_args)
    pipe_readers = []
    processes = []
    for start_func, start_arg in zip(start_funcs, start_args):
        pipe_reader, pipe_writer = mp.Pipe(duplex=False)
        process = mp.Process(
            target=start_func,
            args=start_arg + (pipe_writer,),
        )
        process.start()
        pipe_readers.append(pipe_reader)
        processes.append(process)
    
    # wait to ready
    for index, pipe_reader in enumerate(pipe_readers):
        init_state = pipe_reader.recv()
        if init_state != 'init ok':
            logger.error(f"init func {start_funcs[index].__name__} : {str(init_state)}")
            for proc in processes:
                proc.kill()
            sys.exit(1)
        else:
            logger.info(f"init func {start_funcs[index].__name__} : {str(init_state)}")
    
    assert all([proc.is_alive() for proc in processes])
    return


process_manager = SubmoduleManager()
