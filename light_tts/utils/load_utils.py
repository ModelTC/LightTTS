import os
import json
import yaml
from hyperpyyaml import load_hyperpyyaml
import sys
from enum import Enum
from typing import Optional, Set
from functools import lru_cache

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


class _LazyPlaceholder:
    """延迟加载的占位符，用于跳过 !new: 等标签的实例化"""

    def __init__(self, tag, value=None):
        self._tag = tag
        self._value = value

    def __repr__(self):
        return f"<LazyPlaceholder: {self._tag}>"


class LiteYamlLoader(yaml.SafeLoader):
    """
    轻量级 YAML Loader，不实例化 !new:, !apply:, !name: 等标签定义的对象，
    而是返回占位符或原始值。用于只需要读取配置值而不需要实例化模型的场景。
    """

    pass


def _construct_lazy_placeholder(loader, tag_suffix, node):
    """处理 !new:xxx, !apply:xxx, !name:xxx 等标签，返回占位符"""
    return _LazyPlaceholder(tag_suffix)


def _construct_ref(loader, node):
    """处理 !ref 标签，返回引用的键名"""
    return _LazyPlaceholder("ref", node.value)


# 注册自定义标签处理器
for tag in ["new", "apply", "name", "copy", "include"]:
    LiteYamlLoader.add_multi_constructor(f"!{tag}:", _construct_lazy_placeholder)

LiteYamlLoader.add_constructor("!ref", _construct_ref)


def _extract_yaml_value(yaml_content: str, key_path: str):
    """
    从 YAML 内容中提取指定路径的值（支持简单类型）。
    key_path 格式如 "llm.speech_token_size" 或 "flow.pre_lookahead_len"
    """
    data = yaml.load(yaml_content, Loader=LiteYamlLoader)
    keys = key_path.split(".")
    value = data
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None
    return value


@lru_cache(maxsize=8)
def _get_qwen_config(model_dir: str):
    """从 Qwen 模型目录加载 config.json，获取 vocab_size 等信息"""
    qwen_path = os.path.join(model_dir, "CosyVoice-BlankEN")
    config_path = os.path.join(qwen_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return None


class LiteConfig:
    """轻量级配置类，提供与完整配置兼容的接口但不加载模型"""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.version = CosyVoiceVersion.from_config_file(model_dir)
        config_path = self.version.get_config_path(model_dir)

        with open(config_path, "r") as f:
            self._yaml_content = f.read()
            f.seek(0)
            self._raw_data = yaml.load(f, Loader=LiteYamlLoader)

        # 加载 Qwen 配置
        self._qwen_config = _get_qwen_config(model_dir)

        # 缓存常用值
        self._cache = {}

    @property
    def cosyvoice_version(self):
        return self.version

    @property
    def sample_rate(self):
        return self._get_value("sample_rate", 24000)

    @property
    def speech_token_size(self):
        """从 YAML 中提取 llm.speech_token_size"""
        if "speech_token_size" not in self._cache:
            # 从 llm 定义中提取
            llm_def = self._raw_data.get("llm", {})
            if isinstance(llm_def, dict):
                self._cache["speech_token_size"] = llm_def.get("speech_token_size", 6561)
            else:
                self._cache["speech_token_size"] = 6561
        return self._cache["speech_token_size"]

    @property
    def mix_ratio(self):
        """从 YAML 中提取 llm.mix_ratio"""
        llm_def = self._raw_data.get("llm", {})
        if isinstance(llm_def, dict):
            return llm_def.get("mix_ratio", [5, 15])
        return [5, 15]

    @property
    def pre_lookahead_len(self):
        """从 YAML 中提取 flow.pre_lookahead_len"""
        flow_def = self._raw_data.get("flow", {})
        if isinstance(flow_def, dict):
            return flow_def.get("pre_lookahead_len", 3)
        return 3

    @property
    def vocab_size(self):
        """从 Qwen config.json 获取 vocab_size"""
        if self._qwen_config:
            return self._qwen_config.get("vocab_size", 151936)
        return 151936  # Qwen2 默认值

    @property
    def max_position_embeddings(self):
        """从 Qwen config.json 获取 max_position_embeddings"""
        if self._qwen_config:
            return self._qwen_config.get("max_position_embeddings", 32768)
        return 32768

    @property
    def sos(self):
        if self.version == CosyVoiceVersion.VERSION_3:
            return self.speech_token_size + 0
        else:
            return 0

    @property
    def task_id(self):
        if self.version == CosyVoiceVersion.VERSION_3:
            return self.speech_token_size + 2
        else:
            return 1

    @property
    def eos_token(self):
        if self.version == CosyVoiceVersion.VERSION_3:
            return self.speech_token_size + 1
        else:
            return self.speech_token_size

    @property
    def fill_token(self):
        if self.version == CosyVoiceVersion.VERSION_3:
            return self.speech_token_size + 3
        else:
            return self.speech_token_size + 2

    def _get_value(self, key, default=None):
        return self._raw_data.get(key, default)

    def __getitem__(self, key):
        """支持字典式访问"""
        if key == "cosyvoice_version":
            return self.version
        if key == "sample_rate":
            return self.sample_rate
        if key == "sos":
            return self.sos
        if key == "task_id":
            return self.task_id
        if key == "eos_token":
            return self.eos_token
        if key == "fill_token":
            return self.fill_token
        return self._raw_data.get(key)

    def get(self, key, default=None):
        try:
            val = self[key]
            return val if val is not None else default
        except KeyError:
            return default


class _FakeLLMConfig:
    """模拟 llm.llm.model.model.config 的接口"""

    def __init__(self, lite_config: LiteConfig):
        self._lite_config = lite_config

    @property
    def vocab_size(self):
        return self._lite_config.vocab_size

    @property
    def max_position_embeddings(self):
        return self._lite_config.max_position_embeddings


class _FakeLLM:
    """模拟 configs["llm"] 的接口，提供必要属性而不实际加载模型"""

    def __init__(self, lite_config: LiteConfig):
        self._lite_config = lite_config
        self._fake_config = _FakeLLMConfig(lite_config)

    @property
    def speech_token_size(self):
        return self._lite_config.speech_token_size

    @property
    def mix_ratio(self):
        return self._lite_config.mix_ratio

    @property
    def llm(self):
        return self

    @property
    def model(self):
        return self

    @property
    def config(self):
        return self._fake_config


class _FakeFlow:
    """模拟 configs["flow"] 的接口"""

    def __init__(self, lite_config: LiteConfig):
        self._lite_config = lite_config

    @property
    def pre_lookahead_len(self):
        return self._lite_config.pre_lookahead_len


def load_yaml_frontend(model_dir: str) -> dict:
    """
    加载配置文件，只实例化前端需要的组件（get_tokenizer, feat_extractor, allowed_special），
    跳过大型模型（llm, flow, hift）。

    适用于需要 tokenizer 和 frontend 但不需要推理模型的场景。

    Args:
        model_dir: 模型目录路径

    Returns:
        配置字典，包含前端组件和轻量级配置
    """
    version = CosyVoiceVersion.from_config_file(model_dir)
    config_path = version.get_config_path(model_dir)

    # 使用 overrides 将大型模型设为 None，避免实例化
    # 注意：这里我们保留需要的组件定义，只跳过模型实例化
    with open(config_path, "r") as f:
        # 先读取原始配置获取 speech_token_size 等值
        lite_config = LiteConfig(model_dir)

    with open(config_path, "r") as f:
        configs = load_hyperpyyaml(
            f,
            overrides={
                "qwen_pretrain_path": os.path.join(model_dir, "CosyVoice-BlankEN"),
                # 跳过大型模型实例化 - 设为简单占位符
                "llm": None,
                "flow": None,
                "hift": None,
                "hifigan": None,
            },
        )

    # 添加版本信息
    configs["cosyvoice_version"] = version

    # 添加轻量级配置（从 lite_config 获取，因为 llm 和 flow 已被跳过）
    configs["sos"] = lite_config.sos
    configs["task_id"] = lite_config.task_id
    configs["eos_token"] = lite_config.eos_token
    configs["fill_token"] = lite_config.fill_token

    # 添加 fake llm 和 flow 用于访问配置值
    configs["llm"] = _FakeLLM(lite_config)
    configs["flow"] = _FakeFlow(lite_config)

    return configs


def load_yaml_lite(model_dir: str) -> dict:
    """
    轻量级加载配置文件，不实例化模型对象。

    适用于只需要读取配置值（如 speech_token_size, vocab_size, pre_lookahead_len 等）
    而不需要实际模型对象的场景。

    返回的配置字典支持以下访问方式：
    - configs["cosyvoice_version"]
    - configs["sample_rate"]
    - configs["sos"], configs["task_id"], configs["eos_token"], configs["fill_token"]
    - configs["llm"].speech_token_size
    - configs["llm"].mix_ratio
    - configs["llm"].llm.model.model.config.vocab_size
    - configs["llm"].llm.model.model.config.max_position_embeddings
    - configs["flow"].pre_lookahead_len

    Args:
        model_dir: 模型目录路径

    Returns:
        轻量级配置字典
    """
    lite_config = LiteConfig(model_dir)

    # 构建返回的配置字典，模拟完整加载的结构
    configs = {
        "cosyvoice_version": lite_config.version,
        "sample_rate": lite_config.sample_rate,
        "sos": lite_config.sos,
        "task_id": lite_config.task_id,
        "eos_token": lite_config.eos_token,
        "fill_token": lite_config.fill_token,
        "llm": _FakeLLM(lite_config),
        "flow": _FakeFlow(lite_config),
    }

    return configs


# 对于多进程来说不行
def load_yaml(model_dir):
    """
    完整加载配置文件，包括实例化所有模型对象。

    警告：此函数会实例化 llm, flow, hift 等大型模型，
    如果只需要读取配置值，请使用 load_yaml_lite() 代替。

    Args:
        model_dir: 模型目录路径

    Returns:
        完整配置字典，包含实例化的模型对象
    """
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
