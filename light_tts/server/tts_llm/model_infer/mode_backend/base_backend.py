import os
import asyncio
import numpy as np
import rpyc
import torch
import socket
from datetime import timedelta
from typing import Dict, List, Tuple
from light_tts.models.cosyvoice3.model import CosyVoice3TpPartModel
from light_tts.utils.infer_utils import set_random_seed
from light_tts.utils.infer_utils import calculate_time, mark_start, mark_end
from light_tts.utils.load_utils import CosyVoiceVersion
from light_tts.utils.log_utils import init_logger
from light_tts.server.tts_llm.token_load import TokenLoad
from light_tts.utils.dist_utils import init_distributed_env
from light_tts.utils.envs_utils import get_unique_server_name
from light_tts.server.core.objs import ShmReqManager
from light_tts.server.tts_llm.model_infer.infer_batch import g_infer_context
from light_tts.utils.dist_utils import get_global_rank, get_global_world_size, get_dp_size
from light_tts.utils.dist_utils import get_dp_world_size, get_global_dp_rank, get_current_rank_in_dp
from light_tts.utils.dist_utils import get_current_device_id, get_current_rank_in_node, get_node_world_size
from light_tts.utils.dist_utils import get_dp_rank_in_node
from light_tts.models.cosyvoice2.model import CosyVoice2TpPartModel
import torch.distributed as dist

logger = init_logger(__name__)


class ModeBackend:
    def __init__(self) -> None:
        self.shm_req_manager = ShmReqManager()
        pass

    def init_model(self, kvargs):
        self.args = kvargs.get("args", None)
        self.is_multimodal = False
        self.tp_rank = kvargs["rank_id"]
        self.world_size = kvargs["world_size"]
        self.mix_ratio = kvargs.get("mix_ratio", [5, 15])
        # dp_size_in_node 计算兼容多机纯tp的运行模式，这时候 1 // 2 == 0, 需要兼容
        self.dp_size_in_node = 1

        self.cache = {}
        self.logger = init_logger(__name__)

        self.weight_dir = kvargs["weight_dir"]
        version = kvargs["version"]

        torch.cuda.set_device(0)
        init_distributed_env(kvargs)
        self.init_rank_infos()

        self.shared_token_load = TokenLoad(f"{get_unique_server_name()}_shared_token_load", self.dp_size_in_node)

        # 多卡需要
        # from lightllm.distributed import custom_comm_ops

        # custom_comm_ops.set_custom_reduce()
        # custom_comm_ops.set_custom_gather()

        model_kvargs = {
            "weight_dir": os.path.join(self.weight_dir, "CosyVoice-BlankEN"),
            "max_total_token_num": kvargs["max_total_token_num"],
            "load_way": kvargs["load_way"],
            "pt_dir": os.path.join(self.weight_dir, "llm.pt"),
            "mode": kvargs["mode"],
            "max_req_num": kvargs.get("max_req_num", 1000),
            "max_seq_length": kvargs.get("max_seq_length", 1024 * 5),
            "use_dynamic_prompt_cache": True,  # for bistream mode
            "data_type": kvargs.get("data_type", "float16"),
            "style_name": kvargs["style_name"],
            "speech_token_size": kvargs.get("speech_token_size"),
            "graph_max_batch_size": kvargs.get("graph_max_batch_size", 16),
            "graph_max_len_in_batch": kvargs.get("graph_max_len_in_batch", 8196),
            "disable_cudagraph": kvargs.get("disable_cudagraph", False),
            "batch_max_tokens": kvargs.get("batch_max_tokens", None),
            "quant_type": kvargs.get("quant_type", None),
            "quant_cfg": kvargs.get("quant_cfg", None),
        }

        try:
            if version == CosyVoiceVersion.VERSION_2:
                self.model = CosyVoice2TpPartModel(model_kvargs)
            elif version == CosyVoiceVersion.VERSION_3:
                self.model = CosyVoice3TpPartModel(model_kvargs)
        except Exception as e:
            self.logger.exception(str(e))
            raise e

        torch.cuda.empty_cache()
        set_random_seed(2147483647)

        self.init_custom()

        g_infer_context.register(
            req_manager=self.model.req_manager,
            shm_req_manager=self.shm_req_manager,
            vocab_size=self.model.vocab_size,
        )
        return

    def init_custom(self):
        pass

    def get_max_total_token_num(self):
        return self.model.mem_manager.size

    def prefill(self, reqs: List[Tuple]):
        """This method can be overridden in subclasses."""
        raise NotImplementedError()

    # @calculate_time(show=True, min_cost_ms=200)
    def decode(self):
        """This method can be overridden in subclasses."""
        raise NotImplementedError()

    def pause_reqs(self, req_ids):
        if self.dp_size_in_node != 1:
            req_ids = [req_id for req_id in req_ids if req_id in g_infer_context.requests_mapping]

        g_infer_context.pause_reqs(req_ids)
        return

    # 一些可以复用的单元功能函数
    def _init_reqs(self, reqs: List[Tuple], init_req_obj=True):
        g_infer_context.add_reqs(reqs, init_req_obj=init_req_obj)
        req_ids = [e[0] for e in reqs]
        return req_ids

    def init_rank_infos(self):
        self.node_world_size = get_node_world_size()
        self.rank_in_node = get_current_rank_in_node()
        self.current_device_id = get_current_device_id()
        self.rank_in_dp = get_current_rank_in_dp()
        self.global_dp_rank = get_global_dp_rank()
        self.dp_rank_in_node = get_dp_rank_in_node()
        self.dp_world_size = get_dp_world_size()
        self.global_rank = get_global_rank()
        self.global_world_size = get_global_world_size()
        self.dp_size = get_dp_size()
        self.nnodes = 1

        if self.nnodes > 1 and self.dp_size == 1:
            if self.rank_in_node == 0:
                self.is_master_in_dp = True
            else:
                self.is_master_in_dp = False
        else:
            if self.rank_in_dp == 0:
                self.is_master_in_dp = True
            else:
                self.is_master_in_dp = False
        return
