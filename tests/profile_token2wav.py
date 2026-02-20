#!/usr/bin/env python3
"""
torch.profiler 测试脚本，用于分析 CosyVoice3Model.token2wav 中 flow 和 hift 的性能瓶颈。

使用方式:
    python tests/profile_token2wav.py --model_dir <model_dir> [--token_len 50] [--prompt_token_len 25] [--warmup 3] [--repeat 5]

输出:
    1. 终端打印 key_averages 表格（按 CUDA 耗时排序）
    2. 生成 Chrome trace JSON 文件，可用 chrome://tracing 查看
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import torch
import torch.profiler
import numpy as np

# 确保项目根目录在 sys.path 中
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "cosyvoice"))
sys.path.insert(0, os.path.join(ROOT_DIR, "third_party", "Matcha-TTS"))

from light_tts.utils.load_utils import load_yaml, CosyVoiceVersion
from cosyvoice.cli.model import CosyVoice3Model, CosyVoice2Model


def parse_args():
    parser = argparse.ArgumentParser(description="Profile CosyVoice3Model.token2wav")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="模型目录路径，包含 cosyvoice3.yaml / flow.pt / hift.pt 等")
    parser.add_argument("--token_len", type=int, default=50,
                        help="生成的 speech token 长度 (默认 50)")
    parser.add_argument("--prompt_token_len", type=int, default=25,
                        help="prompt speech token 长度 (默认 25)")
    parser.add_argument("--prompt_feat_len", type=int, default=50,
                        help="prompt mel feature 帧数 (默认 50, 应 = prompt_token_len * token_mel_ratio)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="warmup 次数 (默认 3)")
    parser.add_argument("--repeat", type=int, default=5,
                        help="profiling 重复次数 (默认 5)")
    parser.add_argument("--output_dir", type=str, default="./profiler_output",
                        help="profiler trace 输出目录 (默认 ./profiler_output)")
    parser.add_argument("--finalize", action="store_true", default=True,
                        help="是否 finalize (默认 True)")
    parser.add_argument("--stream", action="store_true", default=False,
                        help="是否 streaming 模式 (默认 False)")
    return parser.parse_args()


def load_model(model_dir: str):
    """使用与 model_rpc.py 相同的方式初始化模型"""
    configs = load_yaml(model_dir)
    version = configs["cosyvoice_version"]

    if version == CosyVoiceVersion.VERSION_3:
        fp16 = True
        model = CosyVoice3Model(configs["llm"], configs["flow"], configs["hift"], fp16=fp16)
    elif version == CosyVoiceVersion.VERSION_2:
        fp16 = True
        model = CosyVoice2Model(configs["llm"], configs["flow"], configs["hift"], fp16=fp16)
    else:
        raise ValueError(f"Unsupported version: {version}")

    model.load(
        f"{model_dir}/llm.pt",
        f"{model_dir}/flow.pt",
        f"{model_dir}/hift.pt",
    )
    model.hift_cache_dict = defaultdict(lambda: None)

    print(f"[INFO] 模型加载完成, version={version}, device={model.device}, fp16={model.fp16}")
    return model


def build_dummy_inputs(model, token_len: int, prompt_token_len: int, prompt_feat_len: int):
    """构造假的输入数据，形状与真实推理一致"""
    device = model.device

    # speech token: [1, token_len], int32, 范围 [0, vocab_size)
    vocab_size = model.flow.vocab_size if hasattr(model.flow, 'vocab_size') else 6561
    token = torch.randint(0, vocab_size, (1, token_len), dtype=torch.int32)

    # prompt speech token: [1, prompt_token_len], int32
    prompt_token = torch.randint(0, vocab_size, (1, prompt_token_len), dtype=torch.int32)

    # prompt mel feature: [1, prompt_feat_len, 80]
    output_size = model.flow.output_size if hasattr(model.flow, 'output_size') else 80
    prompt_feat = torch.randn(1, prompt_feat_len, output_size)

    # speaker embedding: [1, 192] (需要 2D 用于 F.normalize(dim=1))
    spk_embed_dim = 192
    embedding = torch.randn(1, spk_embed_dim)

    return token, prompt_token, prompt_feat, embedding


def run_token2wav(model, token, prompt_token, prompt_feat, embedding,
                  token_offset=0, uuid_str="profile_test",
                  stream=False, finalize=True, speed=1.0):
    """执行一次 token2wav 调用"""
    # 每次调用前重置 hift_cache
    model.hift_cache_dict[uuid_str] = None

    tts_speech = model.token2wav(
        token=token,
        prompt_token=prompt_token,
        prompt_feat=prompt_feat,
        embedding=embedding,
        token_offset=token_offset,
        uuid=uuid_str,
        stream=stream,
        finalize=finalize,
        speed=speed,
    )
    return tts_speech


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ========== 1. 加载模型 ==========
    print("=" * 60)
    print("[STEP 1] 加载模型...")
    model = load_model(args.model_dir)

    # ========== 2. 构造输入 ==========
    print("=" * 60)
    print("[STEP 2] 构造假输入数据...")
    token, prompt_token, prompt_feat, embedding = build_dummy_inputs(
        model, args.token_len, args.prompt_token_len, args.prompt_feat_len
    )
    print(f"  token:        shape={token.shape}, dtype={token.dtype}")
    print(f"  prompt_token: shape={prompt_token.shape}, dtype={prompt_token.dtype}")
    print(f"  prompt_feat:  shape={prompt_feat.shape}, dtype={prompt_feat.dtype}")
    print(f"  embedding:    shape={embedding.shape}, dtype={embedding.dtype}")

    # ========== 3. Warmup ==========
    print("=" * 60)
    print(f"[STEP 3] Warmup ({args.warmup} 次)...")
    for i in range(args.warmup):
        tts_speech = run_token2wav(model, token, prompt_token, prompt_feat, embedding,
                                   stream=args.stream, finalize=args.finalize)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"  warmup {i+1}/{args.warmup} done, output shape={tts_speech.shape}")

    # ========== 4. 手动计时 (作为参考) ==========
    print("=" * 60)
    print(f"[STEP 4] 手动计时 ({args.repeat} 次)...")
    times = []
    for i in range(args.repeat):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        tts_speech = run_token2wav(model, token, prompt_token, prompt_feat, embedding,
                                   stream=args.stream, finalize=args.finalize)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000
        times.append(elapsed_ms)
        print(f"  run {i+1}/{args.repeat}: {elapsed_ms:.2f} ms, output shape={tts_speech.shape}")

    print(f"  平均耗时: {np.mean(times):.2f} ms, 标准差: {np.std(times):.2f} ms")

    # ========== 5. torch.profiler 详细分析 ==========
    print("=" * 60)
    print(f"[STEP 5] torch.profiler 分析 ({args.repeat} 次)...")

    trace_path = os.path.join(args.output_dir, "token2wav_trace")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=0,
            warmup=1,
            active=args.repeat,
            repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_path),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        for step_i in range(1 + args.repeat):  # 1 warmup + repeat active
            run_token2wav(model, token, prompt_token, prompt_feat, embedding,
                          stream=args.stream, finalize=args.finalize)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            prof.step()

    # ========== 6. 打印汇总表 ==========
    print("=" * 60)
    print("[STEP 6] Profiler 结果汇总 (按 CUDA 总耗时排序):")
    print()

    # 按 CUDA 时间排序
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
    print()

    # 按 CPU 时间排序
    print("-" * 60)
    print("按 CPU 总耗时排序:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
    print()

    # 按 self CUDA 时间排序 (找出真正的 kernel 热点)
    print("-" * 60)
    print("按 Self CUDA 耗时排序 (kernel 热点):")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))

    # ========== 7. 分别测量 flow 和 hift ==========
    print("=" * 60)
    print("[STEP 7] 分别测量 flow 和 hift 耗时...")

    # 先用 flow 单独测
    flow_times = []
    hift_times = []

    for i in range(args.repeat):
        model.hift_cache_dict["bench_test"] = None

        with torch.cuda.amp.autocast(model.fp16):
            # ---- flow ----
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            tts_mel, _ = model.flow.inference(
                token=token.to(model.device, dtype=torch.int32),
                token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(model.device),
                prompt_token=prompt_token.to(model.device),
                prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(model.device),
                prompt_feat=prompt_feat.to(model.device),
                prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(model.device),
                embedding=embedding.to(model.device),
                streaming=args.stream,
                finalize=args.finalize,
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            flow_times.append((t1 - t0) * 1000)

            # ---- hift ----
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t2 = time.perf_counter()

            tts_speech, _ = model.hift.inference(speech_feat=tts_mel, finalize=args.finalize)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t3 = time.perf_counter()
            hift_times.append((t3 - t2) * 1000)

        print(f"  run {i+1}/{args.repeat}: flow={flow_times[-1]:.2f} ms, hift={hift_times[-1]:.2f} ms, "
              f"mel shape={tts_mel.shape}, speech shape={tts_speech.shape}")

    print(f"\n  flow 平均: {np.mean(flow_times):.2f} ms (std={np.std(flow_times):.2f})")
    print(f"  hift 平均: {np.mean(hift_times):.2f} ms (std={np.std(hift_times):.2f})")
    print(f"  flow 占比: {np.mean(flow_times) / (np.mean(flow_times) + np.mean(hift_times)) * 100:.1f}%")
    print(f"  hift 占比: {np.mean(hift_times) / (np.mean(flow_times) + np.mean(hift_times)) * 100:.1f}%")

    # ========== 8. 分别 profile flow 和 hift ==========
    print("=" * 60)
    print("[STEP 8] 分别 profile flow 和 hift...")

    # Profile flow
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as flow_prof:
        for _ in range(args.repeat):
            with torch.cuda.amp.autocast(model.fp16):
                model.flow.inference(
                    token=token.to(model.device, dtype=torch.int32),
                    token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(model.device),
                    prompt_token=prompt_token.to(model.device),
                    prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(model.device),
                    prompt_feat=prompt_feat.to(model.device),
                    prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(model.device),
                    embedding=embedding.to(model.device),
                    streaming=args.stream,
                    finalize=args.finalize,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    print("\n--- Flow Profiler (Self CUDA 耗时 Top 20) ---")
    print(flow_prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))

    # Profile hift
    # 先生成 mel 用于 hift 输入
    with torch.no_grad(), torch.cuda.amp.autocast(model.fp16):
        tts_mel, _ = model.flow.inference(
            token=token.to(model.device, dtype=torch.int32),
            token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(model.device),
            prompt_token=prompt_token.to(model.device),
            prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(model.device),
            prompt_feat=prompt_feat.to(model.device),
            prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(model.device),
            embedding=embedding.to(model.device),
            streaming=args.stream,
            finalize=args.finalize,
        )

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as hift_prof:
        for _ in range(args.repeat):
            model.hift.inference(speech_feat=tts_mel, finalize=args.finalize)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    print("\n--- HiFT Profiler (Self CUDA 耗时 Top 20) ---")
    print(hift_prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))

    # ========== Done ==========
    print("=" * 60)
    print(f"[DONE] Trace 文件已保存到: {trace_path}")
    print("  可以用 TensorBoard 查看:")
    print(f"    tensorboard --logdir {trace_path}")
    print("  或者在 chrome://tracing 中打开生成的 .json 文件")


if __name__ == "__main__":
    main()
