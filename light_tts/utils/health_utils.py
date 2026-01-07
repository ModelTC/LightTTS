import hashlib
import asyncio
from functools import lru_cache
from pathlib import Path
from light_tts.utils.load_utils import CosyVoiceVersion
from light_tts.utils.param_utils import check_request
from light_tts.server.httpserver.manager import HttpServerManager
from light_tts.utils.log_utils import init_logger
from light_tts.server.core.objs.sampling_params import SamplingParams
from cosyvoice.utils.file_utils import load_wav

logger = init_logger(__name__)


# 获取项目根目录下的测试音频文件路径
def _get_test_wav_path() -> Path:
    return Path(__file__).parent.parent.parent / "cosyvoice" / "asset" / "zero_shot_prompt.wav"


@lru_cache(maxsize=1)
def _get_cached_wav_info() -> tuple:
    """
    缓存加载的wav文件信息，避免重复加载
    返回: (prompt_speech_16k, speech_md5, semantic_len)
    """
    wav_path = _get_test_wav_path()
    prompt_speech_16k = load_wav(str(wav_path), 16000)
    semantic_len = (prompt_speech_16k.shape[1] + 239) // 640 + 10  # + 10 for safe

    # 计算md5
    with open(wav_path, "rb") as f:
        speech_md5 = hashlib.md5(f.read()).hexdigest()

    return prompt_speech_16k, speech_md5, semantic_len


async def _consume_generator(generator):
    """消费async generator并返回所有结果"""
    results = []
    async for result in generator:
        results.append(result)
    return results


async def health_check(httpserver_manager: HttpServerManager, g_id_gen, lora_styles, version: CosyVoiceVersion):
    try:
        # 获取缓存的wav信息
        prompt_speech_16k, speech_md5, semantic_len = _get_cached_wav_info()

        gen_ans_list = []
        for style in lora_styles:
            if version == CosyVoiceVersion.VERSION_2:
                prompt_text = "希望你以后能够做的比我还好呦。"
            else:
                prompt_text = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"

            # 分配speech内存
            speech_index, have_alloc = httpserver_manager.alloc_speech_mem(speech_md5, prompt_speech_16k)

            # 构建完整的request_dict
            request_dict = {
                "text": "你好",
                "prompt_text": prompt_text,
                "tts_model_name": style,
                "speech_md5": speech_md5,
                "need_extract_speech": not have_alloc,
                "stream": False,
                "speech_index": speech_index,
                "semantic_len": semantic_len,
                "speed": 1.0,
            }
            # 创建sampling_params
            sampling_params = SamplingParams()
            sampling_params.init()
            sampling_params.verify()

            request_id = g_id_gen.generate_id()
            results_generator = httpserver_manager.generate(request_dict, request_id, sampling_params)
            # 包装成coroutine以便asyncio.gather使用
            gen_ans_list.append(_consume_generator(results_generator))

        await asyncio.gather(*gen_ans_list)
        return True
    except Exception as e:
        logger.critical("health_check error:", e)
        return False
