import asyncio
import multiprocessing.synchronize
import os
import numpy as np
import rpyc
import torch
import traceback
from datetime import timedelta
import torch.multiprocessing as mp
import multiprocessing
from typing import Dict, List, Tuple

from light_tts.server.tts_llm.model_infer.mode_backend.continues_batch.impl import ContinuesBatchBackend
from light_tts.utils.log_utils import init_logger
from light_tts.server.core.objs.shm_speech_manager import SharedSpeechManager
from light_tts.server.core.objs import RpcShmParams, RpcShmResults, ShmSyncStatusArray


logger = init_logger(__name__)


class ModelRpcServer:
    def __init__(
        self,
        args,
        rank: int,
        rank_in_node: int,
        node_world_size: int,
        rpc_event: multiprocessing.synchronize.Event,
        rpc_finished_event: multiprocessing.synchronize.Event,
    ):
        super().__init__()
        self.args = args
        self.node_world_size = node_world_size
        self.rpc_event = rpc_event
        self.rpc_finished_event = rpc_finished_event

        self.rpc_shm_params = RpcShmParams()
        self.rpc_shm_params.create_or_link_shm()
        self.rpc_shm_results = RpcShmResults()
        self.rpc_shm_results.create_or_link_shm()
        self.rpc_shm_sync_status = ShmSyncStatusArray(self.node_world_size)
        self.rpc_shm_sync_status.create_or_link_shm()

        self.rank = rank
        self.rank_in_node = rank_in_node
        logger.info(f"Initialized RPC server for rank {self.rank}.")
        return

    def init_model(self, kvargs):
        kvargs["rank_id"] = self.rank
        # 初始化 backend，所有参数直接通过 kvargs 传递
        self.backend = ContinuesBatchBackend()
        logger.info(f"use {self.backend.__class__.__name__}")
        self.backend.init_model(kvargs)

        return

    def prefill(self, reqs):
        try:
            return self.backend.prefill(reqs)
        except Exception as e:
            err_msg = str(e)
            logger.exception(f"Batch prefill encountered an unexpected ERROR: {err_msg}")
            raise e

    def decode(self):
        try:
            return self.backend.decode()
        except Exception as e:
            err_msg = str(e)
            logger.exception(f"Batch decode encountered an unexpected ERROR: {err_msg}")
            raise e

    def pause_reqs(self, req_ids):
        return self.backend.pause_reqs(req_ids)

    def get_max_total_token_num(self):
        return self.backend.get_max_total_token_num()


class ModelRpcClient:
    def __init__(self, model_infer_servers: List[ModelRpcServer], world_size, rpc_event, rpc_finished_event):
        # model_infer_servers 是传入的推理服务对象，但是在重构后，
        # 单卡不使用rpc 通信的时候，里面才有真实对象，当多卡使用rpc
        # 以后，model_infer_servers 传入的是 None 数组
        if world_size == 1:
            self.model_infer_server: ModelRpcServer = model_infer_servers[0]
        else:
            self.model_infer_server: ModelRpcServer = None

        self.world_size = world_size
        self.use_rpc = self.world_size != 1
        self.rpc_shm_params = RpcShmParams()
        self.rpc_shm_params.create_or_link_shm()
        self.rpc_shm_results = RpcShmResults()
        self.rpc_shm_results.create_or_link_shm()

        self.rpc_event = rpc_event
        self.rpc_finished_event = rpc_finished_event
        return

    async def init_model(self, kvargs):
        if self.use_rpc:
            self.rpc_shm_params.write_func_params("init_model", (kvargs,))
            self.rpc_event.set()

            self.rpc_finished_event.wait()
            self.rpc_finished_event.clear()
            return
        else:
            self.model_infer_server.init_model(kvargs)
            return

    async def prefill(self, reqs):
        if self.use_rpc:
            self.rpc_shm_params.write_func_params("prefill", (reqs,))
            self.rpc_event.set()

            await asyncio.to_thread(self.rpc_finished_event.wait)
            self.rpc_finished_event.clear()
            return
        else:
            self.model_infer_server.prefill(reqs)
            return

    async def decode(self):
        if self.use_rpc:
            self.rpc_shm_params.write_func_params("decode", ())
            self.rpc_event.set()

            await asyncio.to_thread(self.rpc_finished_event.wait)
            self.rpc_finished_event.clear()
            return
        else:
            self.model_infer_server.decode()
            return

    async def pause_reqs(self, req_ids):
        if self.use_rpc:
            self.rpc_shm_params.write_func_params("pause_reqs", (req_ids,))
            self.rpc_event.set()

            self.rpc_finished_event.wait()
            self.rpc_finished_event.clear()
            return
        else:
            self.model_infer_server.pause_reqs(req_ids)
            return

    async def get_max_total_token_num(self):
        if self.use_rpc:
            self.rpc_shm_params.write_func_params("get_max_total_token_num", ())
            self.rpc_event.set()

            self.rpc_finished_event.wait()
            self.rpc_finished_event.clear()
            func_name, ret = self.rpc_shm_results.read_func_result()
            assert func_name == "get_max_total_token_num"
            return ret
        else:
            return self.model_infer_server.get_max_total_token_num()


async def start_model_process(
    args,
    rank,
    rank_in_node,
    node_world_size,
    rpc_event,
    rpc_finished_event,
):
    import light_tts.utils.rpyc_fix_utils as _

    return ModelRpcServer(
        args,
        rank,
        rank_in_node,
        node_world_size,
        rpc_event,
        rpc_finished_event,
    )
