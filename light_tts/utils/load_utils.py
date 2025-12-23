import os
from hyperpyyaml import load_hyperpyyaml
import sys
from enum import Enum

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append("{}/../../third_party/Matcha-TTS".format(ROOT_DIR))


class CosyVoiceVersion(Enum):
    """CosyVoice版本枚举类"""

    VERSION_2 = 2
    VERSION_3 = 3

    @classmethod
    def from_config_file(cls, model_dir):
        """从模型目录中检测配置文件版本"""
        config_v3 = os.path.join(model_dir, "cosyvoice3.yaml")
        config_v2 = os.path.join(model_dir, "cosyvoice2.yaml")

        if os.path.exists(config_v3):
            return cls.VERSION_3
        elif os.path.exists(config_v2):
            return cls.VERSION_2
        else:
            raise FileNotFoundError(f"未找到配置文件：在 {model_dir} 中找不到 cosyvoice2.yaml 或 cosyvoice3.yaml")

    def get_config_path(self, model_dir):
        """获取配置文件名"""
        return os.path.join(model_dir, f"cosyvoice{self.value}.yaml")


# 对于多进程来说不行
def load_yaml(model_dir):
    """加载配置文件，自动检测版本"""
    version = CosyVoiceVersion.from_config_file(model_dir)
    config_path = version.get_config_path(model_dir)

    with open(config_path, "r") as f:
        configs = load_hyperpyyaml(f, overrides={"qwen_pretrain_path": os.path.join(model_dir, "CosyVoice-BlankEN")})

    # 将版本信息添加到配置中
    configs["cosyvoice_version"] = version

    # 根据版本设置特殊 token
    speech_token_size = configs["llm"].speech_token_size

    if version == CosyVoiceVersion.VERSION_3:
        # CosyVoice3 的特殊 token 配置
        configs["sos"] = speech_token_size + 0
        configs["task_id"] = speech_token_size + 2
        configs["eos_token"] = speech_token_size + 1
        configs["fill_token"] = speech_token_size + 3
    else:  # VERSION_2
        # CosyVoice2 的特殊 token 配置
        configs["sos"] = 0
        configs["task_id"] = 1
        configs["eos_token"] = speech_token_size
        configs["fill_token"] = speech_token_size + 2

    return configs
