import os
import json
from functools import lru_cache
from schema import Schema, And, Use, Optional, SchemaError
from light_tts.utils.path_utils import trans_relative_to_abs_path
from light_tts.utils.log_utils import init_logger
from .envs_utils import get_env_start_args
import sys

logger = init_logger(__name__)


def get_config_json(model_dir):
    config_path = os.path.join(model_dir, "style_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)

        # 相对路径转绝对路径
        def modify_paths(data):
            if isinstance(data, dict):
                for k in list(data.keys()):  # 转换为列表避免迭代时修改字典大小
                    if k in ["llm_path", "gpt_model_path"]:
                        data["llm_path"] = trans_relative_to_abs_path(model_dir, data[k])
                    else:
                        modify_paths(data[k])
            elif isinstance(data, list):
                for item in data:
                    modify_paths(item)

        modify_paths(config)
        return config
    else:
        # 默认配置
        return {
            "lora_info": [
                {
                    "style_name": "cosyvoice",
                    "llm_path": os.path.join(model_dir, "llm.pt"),
                }
            ],
        }


@lru_cache(maxsize=None)
def get_fixed_kv_len():
    start_args = get_env_start_args()
    model_cfg = get_config_json(start_args.model_dir)
    if "prompt_cache_token_ids" in model_cfg:
        return len(model_cfg["prompt_cache_token_ids"])
    else:
        return 0


def get_style_gpt_path(model_dir, style_name):
    config = get_config_json(model_dir)
    for item in config["lora_info"]:
        if item["style_name"] == style_name:
            return item["llm_path"]
    assert False, f"can not find {style_name} llm path"
