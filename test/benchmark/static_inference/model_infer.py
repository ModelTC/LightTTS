import os
import torch
import numpy as np
from multiprocessing import Queue
import multiprocessing
from transformers import PretrainedConfig
from light_tts.models.cosyvoice3.model import CosyVoice3TpPartModel
from light_tts.utils.dist_utils import init_distributed_env, get_current_rank_in_dp
from light_tts.utils.envs_utils import get_env_start_args
from torch.profiler import profile, record_function, ProfilerActivity
from light_tts.utils.load_utils import load_yaml_lite
from light_tts.utils.log_utils import init_logger
import torch.cuda as cuda

logger = init_logger(__name__)


def test_model_inference(args):
    ans_queue = Queue()
    workers = []
    dp_size = args.get("dp", 1)
    configs = load_yaml_lite(args.model_dir)
    print(configs)

    model_kvargs = {
        "weight_dir": os.path.join(args.model_dir, "CosyVoice-BlankEN"),
        "max_total_token_num": args.max_total_token_num,
        "load_way": args.load_way,
        "pt_dir": "/data/Fun-CosyVoice3-0.5B/llm.pt",
        "mode": args.mode,
        "max_req_num": args.get("max_req_num", 1000),
        "max_seq_length": args.get("max_seq_length", 1024 * 5),
        "use_dynamic_prompt_cache": True,  # for bistream mode
        "data_type": "fp16",
        "style_name": "cosyvoice",
        "speech_token_size": configs["llm"].speech_token_size,
        "graph_max_batch_size": args.get("graph_max_batch_size", 16),
        "graph_max_len_in_batch": args.get("graph_max_len_in_batch", 8196),
        "disable_cudagraph": args.get("disable_cudagraph", False),
        "batch_max_tokens": args.get("batch_max_tokens", None),
        "quant_type": args.get("quant_type", None),
        "quant_cfg": args.get("quant_cfg", None),
    }
    proc = multiprocessing.Process(
        target=tppart_model_infer,
        args=(args, model_kvargs, args.batch_size, args.input_len, args.output_len, ans_queue),
    )
    proc.start()
    workers.append(proc)

    for proc in workers:
        proc.join()

    assert not ans_queue.empty()
    while not ans_queue.empty():
        assert ans_queue.get()
    return

def prefill(
    model_part: CosyVoice3TpPartModel,
    batch_size,
    max_len_in_batch,
    input_ids,
    b_req_idx,
    b_seq_len,
    b_start_loc,
    total_token_num,
    b_ready_cache_len,
):
    model_output = model_part.forward(
        batch_size,
        total_token_num,
        max_len_in_batch,
        input_ids,
        b_req_idx,
        b_start_loc,
        b_seq_len,
        b_ready_cache_len,
        is_prefill=True,
    )
    return model_output


def decode(
    model_part: CosyVoice3TpPartModel, batch_size, max_len_in_batch, input_ids, b_req_idx, b_seq_len, b_start_loc, total_token_num
):
    model_output = model_part.forward(
        batch_size,
        total_token_num,
        max_len_in_batch,
        input_ids,
        b_req_idx,
        b_start_loc,
        b_seq_len,
        is_prefill=False,
    )
    return model_output


def torch_profile(fn, log_dir=None):
    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
    ) as prof:
        fn()
    if get_current_rank_in_dp() == 0:
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


def run_forward_once(
    model_kvargs, input_len, output_len, batch_size, model_part: CosyVoice3TpPartModel, enable_torch_profile=False
):
    test_data = np.vstack([np.random.randint(0, 50256, input_len) for _ in range(batch_size)])
    test_data = test_data.reshape(-1)
    test_data = torch.from_numpy(test_data)
    import time

    torch.cuda.synchronize()
    prefill_start_time = time.time()

    b_req_idx = torch.tensor(
        [model_part.req_manager.alloc() for _ in range(batch_size)], dtype=torch.int32, device="cpu"
    )
    b_seq_len = torch.zeros(batch_size, dtype=torch.int32, device="cpu")
    b_ready_cache_len = torch.zeros(batch_size, dtype=torch.int32, device="cpu")
    b_start_loc = torch.zeros(batch_size, dtype=torch.int32, device="cpu")
    for i in range(batch_size):
        b_seq_len[i] = input_len
        if i > 0:
            b_start_loc[i] = b_start_loc[i - 1] + b_seq_len[i - 1]

    total_token_num = batch_size * input_len

    prefill_fn = prefill
    decode_fn = decode

    test_data = test_data.cuda()
    b_req_idx = b_req_idx.cuda()
    b_seq_len = b_seq_len.cuda()
    b_start_loc = b_start_loc.cuda()
    b_ready_cache_len = b_ready_cache_len.cuda()

    logits = prefill_fn(
        model_part,
        batch_size,
        input_len,
        test_data,
        b_req_idx,
        b_seq_len,
        b_start_loc,
        total_token_num,
        b_ready_cache_len,  # b_ready_cache_len
    )

    prob_out = torch.softmax(logits, dim=-1)
    predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
    _ = predict_ids.detach().cpu().numpy()
    predict_ids = predict_ids.cuda()

    torch.cuda.synchronize()
    rank_id = 0
    if rank_id == 0:
        print(
            f"prefill time cost: {(time.time() - prefill_start_time) * 1000}, "
            f"prefill throughput: {batch_size * input_len / (time.time() - prefill_start_time)} tokens/s"
        )

    if enable_torch_profile:
        print("Profile Prefill")
        try:
            torch_profile(
                lambda: prefill_fn(
                    model_part,
                    batch_size,
                    input_len,
                    test_data,
                    b_req_idx,
                    b_seq_len,
                    b_start_loc,
                    total_token_num,
                    b_ready_cache_len,  # b_ready_cache_len
                ),
                log_dir=f"./logs/forward_prefill_{model_kvargs['rank_id']}/{batch_size}",
            )
        except Exception as e:
            print(str(e))
            raise

    for i in range(output_len):
        torch.cuda.synchronize()
        step_start = time.time()
        total_token_num += batch_size
        b_seq_len += 1
        max_len_in_batch = input_len + i + 1
        logits = decode_fn(
            model_part,
            batch_size,
            max_len_in_batch,
            predict_ids.view(-1),
            b_req_idx,
            b_seq_len,
            b_start_loc,
            total_token_num,
        )
        if enable_torch_profile and (i == output_len // 2 or i == output_len // 2 + 1):
            try:
                torch_profile(
                    lambda: decode_fn(
                        model_part,
                        batch_size,
                        max_len_in_batch,
                        predict_ids.view(-1),
                        b_req_idx,
                        b_start_loc,
                        b_seq_len,
                        total_token_num,
                    ),
                    log_dir=f"./logs/forward_decode_{model_kvargs['rank_id']}/{batch_size}",
                )
            except Exception as e:
                print(str(e))
                raise

        prob_out = torch.softmax(logits, dim=-1)
        predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
        _ = predict_ids.detach().cpu().numpy()
        predict_ids = predict_ids.cuda()
        torch.cuda.synchronize()
        if i % 100 == 0 or i == output_len - 1:
            if rank_id == 0:
                print(
                    f"i: {i}, step cost time: {(time.time() - step_start) * 1000} ms, "
                    f"throughput: {batch_size / (time.time() - step_start)} tokens/s"
                )

    model_part.mem_manager.free_all()
    model_part.req_manager.free_all()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def tppart_model_infer(args, model_kvargs, batch_size, input_len, output_len, ans_queue):
    args = get_env_start_args()
    import triton.profiler as proton
    import torch
    from light_tts.utils.dist_utils import set_current_device_id

    if isinstance(batch_size, int):
        batch_size = [batch_size]
    else:
        batch_size = [2, 8, 16, 32, 64, 128]
    print(batch_size)
    model_kvargs["world_size"] = 1
    model_kvargs["rank_id"] = 0
    init_distributed_env(model_kvargs)

    torch.cuda.empty_cache()

    model_part = CosyVoice3TpPartModel(model_kvargs)

    rank_id = get_current_rank_in_dp()
    for b in batch_size:
        if rank_id == 0:
            print(f"Testing batch size {b}")

        # warm up
        run_forward_once(
            model_kvargs,
            input_len,
            output_len=10,
            batch_size=b,
            model_part=model_part,
            enable_torch_profile=False,
        )
        # test
        run_forward_once(
            model_kvargs,
            input_len,
            output_len,
            batch_size=b,
            model_part=model_part,
            enable_torch_profile=args.torch_profile,
        )
        if rank_id == 0:
            print("=" * 50)

    ans_queue.put(True)

    return
