import triton
import torch
from light_tts.utils.log_utils import init_logger

logger = init_logger(__name__)

from light_tts.utils.tuning_utils import mp_tuning, set_seed, tuning_configs
import sys
import os
import numpy as np

from light_tts.models.llama.triton_kernel.flash_decoding_stage1 import flash_decode_stage1
from light_tts.models.llama.triton_kernel.flash_decoding_stage2 import flash_decode_stage2
from light_tts.models.llama.triton_kernel.flash_decoding import LlamaFlashDecodingStage1KernelConfig


@torch.no_grad()
def test_func_stage1(
    batch_size,
    seq_len,
    head_dim,
    q_head_num,
    kv_head_num,
    dtype,
    test_count: int = 20,
    **run_config,
):
    """只测试 stage1 的性能"""
    set_seed()
    max_len = 2048
    tmp_class = type("TestObj", (object,), {})
    infer_state = tmp_class()
    infer_state.batch_size = batch_size
    infer_state.max_len_in_batch = 2048

    infer_state.req_manager = tmp_class()
    infer_state.req_manager.req_to_token_indexs = torch.zeros(
        (infer_state.batch_size, seq_len), dtype=torch.int32, device="cuda"
    )
    infer_state.req_manager.req_to_token_indexs.view(-1)[:] = torch.arange(
        0, infer_state.batch_size * seq_len, step=1, dtype=torch.int32
    ).cuda()
    infer_state.b_req_idx = torch.arange(0, infer_state.batch_size, step=1, dtype=torch.int32).cuda()
    infer_state.b_seq_len = torch.full((infer_state.batch_size,), fill_value=seq_len, dtype=torch.int32).cuda()
    infer_state.total_token_num_tensor = torch.sum(infer_state.b_seq_len)

    q = torch.empty((batch_size, q_head_num, head_dim), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    cache_k = torch.empty((batch_size * max_len, kv_head_num, head_dim), dtype=dtype, device="cuda").normal_(
        mean=0.1, std=0.2
    )
    cache_v = torch.empty((batch_size * max_len, kv_head_num, head_dim), dtype=dtype, device="cuda").normal_(
        mean=0.1, std=0.2
    )
    mid_o = torch.empty(
        [batch_size, q_head_num, max_len // run_config["BLOCK_SEQ"] + 1, head_dim], dtype=torch.float32, device="cuda"
    )
    mid_o_logexpsum = torch.empty(
        [batch_size, q_head_num, max_len // run_config["BLOCK_SEQ"] + 1], dtype=torch.float32, device="cuda"
    )

    fn = lambda: flash_decode_stage1(
        q,
        cache_k,
        cache_v,
        infer_state.req_manager.req_to_token_indexs,
        infer_state.b_req_idx,
        infer_state.b_seq_len,
        infer_state.max_len_in_batch,
        mid_o,
        mid_o_logexpsum,
        **run_config,
    )

    cost_time = triton.testing.do_bench_cudagraph(fn, rep=test_count)
    # logger.info(f"stage1 {batch_size, seq_len} cost time: {cost_time} ms, config: {run_config}")
    return cost_time


@torch.no_grad()
def test_func_stage2(
    batch_size,
    seq_len,
    head_dim,
    q_head_num,
    kv_head_num,
    dtype,
    test_count: int = 20,
    **run_config,
):
    """只测试 stage2 的性能"""
    set_seed()
    max_len = 2048

    mid_o = torch.empty(
        [batch_size, q_head_num, max_len // run_config["BLOCK_SEQ"] + 1, head_dim], dtype=torch.float32, device="cuda"
    ).normal_(mean=0.1, std=0.2)
    mid_o_logexpsum = torch.empty(
        [batch_size, q_head_num, max_len // run_config["BLOCK_SEQ"] + 1], dtype=torch.float32, device="cuda"
    ).normal_(mean=0.1, std=0.2)

    b_seq_len = torch.full((batch_size,), fill_value=seq_len, dtype=torch.int32).cuda()
    o_tensor = torch.empty((batch_size, q_head_num, head_dim), dtype=dtype, device="cuda")

    fn = lambda: flash_decode_stage2(mid_o, mid_o_logexpsum, b_seq_len, o_tensor, **run_config)

    cost_time = triton.testing.do_bench_cudagraph(fn, rep=test_count)
    logger.info(f"stage2 {batch_size, max_len} cost time: {cost_time} ms, config: {run_config}")
    return cost_time


def get_test_configs_stage1(split_id, split_count, **kwargs):
    """Stage1 配置生成器"""
    index = 0
    for BLOCK_SEQ in [16, 32, 64, 128, 256]:
        for BLOCK_N in [8, 16, 32, 64]:
            if BLOCK_SEQ % BLOCK_N != 0:
                continue
            for stage1_num_warps in [1, 2, 4, 8]:
                for stage1_num_stages in [1, 2, 3, 4, 5, 6, 7, 8]:
                    t_config = {
                        "BLOCK_SEQ": BLOCK_SEQ,
                        "BLOCK_N": BLOCK_N,
                        "stage1_num_warps": stage1_num_warps,
                        "stage1_num_stages": stage1_num_stages,
                        # stage2 使用默认值，不影响 stage1 测试
                        "stage2_num_warps": 4,
                        "stage2_num_stages": 2,
                    }
                    if index % split_count == split_id:
                        yield t_config
                    index += 1


def get_test_configs_stage2(split_id, split_count, **kwargs):
    """Stage2 配置生成器，基于 stage1 的最优配置（从 kwargs 中获取）"""
    best_stage1_config = kwargs.get("best_stage1_config", {})
    index = 0
    for stage2_num_warps in [1, 2, 4, 8]:
        for stage2_num_stages in [1, 2, 3, 4, 5, 6, 7, 8]:
            t_config = {
                # 使用 stage1 的最优配置
                "BLOCK_SEQ": best_stage1_config["BLOCK_SEQ"],
                "BLOCK_N": best_stage1_config["BLOCK_N"],
                "stage1_num_warps": best_stage1_config["stage1_num_warps"],
                "stage1_num_stages": best_stage1_config["stage1_num_stages"],
                # stage2 参数
                "stage2_num_warps": stage2_num_warps,
                "stage2_num_stages": stage2_num_stages,
            }
            if index % split_count == split_id:
                yield t_config
            index += 1


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    import collections

    store_json_ans = collections.defaultdict(dict)

    head_dim = 64
    q_head_num = 14
    kv_head_num = 2

    for seq_len in [2048]:
        for batch_size in [1, 2, 4, 8]:
            test_func_args = {
                "batch_size": batch_size,
                "seq_len": seq_len,
                "head_dim": head_dim,
                "q_head_num": q_head_num,
                "kv_head_num": kv_head_num,
                "dtype": torch.float16,
                "test_count": 20,
            }

            # ========== Stage 1 调优 ==========
            logger.info(f"===== Tuning Stage1 for seq_len={seq_len}, batch_size={batch_size} =====")
            best_stage1_config = mp_tuning(
                tuning_configs,
                {
                    "test_func": test_func_stage1,
                    "test_func_args": test_func_args,
                    "get_test_configs_func": get_test_configs_stage1,
                },
            )
            # 如果 stage1 调优失败，使用默认配置
            if best_stage1_config is None:
                best_stage1_config = {
                    "BLOCK_SEQ": 64,
                    "BLOCK_N": 16,
                    "stage1_num_warps": 4,
                    "stage1_num_stages": 2,
                }
                logger.warning(f"Stage1 tuning failed, using default config: {best_stage1_config}")
            else:
                logger.info(f"Stage1 best config: {best_stage1_config}")

            # ========== Stage 2 调优 ==========
            logger.info(f"===== Tuning Stage2 for seq_len={seq_len}, batch_size={batch_size} =====")

            # 把 best_stage1_config 放到 test_func_args 里传递给 get_test_configs_stage2
            test_func_args_stage2 = test_func_args.copy()
            test_func_args_stage2["best_stage1_config"] = best_stage1_config

            best_stage2_config = mp_tuning(
                tuning_configs,
                {
                    "test_func": test_func_stage2,
                    "test_func_args": test_func_args_stage2,
                    "get_test_configs_func": get_test_configs_stage2,
                },
            )
            # 如果 stage2 调优失败，使用默认配置
            if best_stage2_config is None:
                best_stage2_config = {
                    "stage2_num_warps": 4,
                    "stage2_num_stages": 2,
                }
                logger.warning(f"Stage2 tuning failed, using default config: {best_stage2_config}")
            else:
                logger.info(f"Stage2 best config: {best_stage2_config}")

            # ========== 合并最终配置 ==========
            final_config = {
                "BLOCK_SEQ": best_stage1_config["BLOCK_SEQ"],
                "BLOCK_N": best_stage1_config["BLOCK_N"],
                "stage1_num_warps": best_stage1_config["stage1_num_warps"],
                "stage1_num_stages": best_stage1_config["stage1_num_stages"],
                "stage2_num_warps": best_stage2_config["stage2_num_warps"],
                "stage2_num_stages": best_stage2_config["stage2_num_stages"],
            }
            logger.info(f"Final combined config: {final_config}")

            store_json_ans[seq_len][batch_size] = final_config
            LlamaFlashDecodingStage1KernelConfig.save_config(
                head_dim=head_dim,
                q_head_num=q_head_num,
                kv_head_num=kv_head_num,
                out_dtype=str(torch.float16),
                store_json_ans=store_json_ans,
            )
