import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True
    if v.lower() in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def make_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--httpserver_workers", type=int, default=1)
    parser.add_argument(
        "--zmq_mode",
        type=str,
        default="ipc:///tmp/",
        help="use socket mode or ipc mode, only can be set in ['tcp://', 'ipc:///tmp/']",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="the model weight dir path, the app will load config, weights and tokenizer from this dir",
    )
    parser.add_argument(
        "--tokenizer_mode",
        type=str,
        default="slow",
        help="""tokenizer load mode, can be slow or auto, slow mode load fast but run slow, slow mode is good for debug and test,
                        when you want to get best performance, try auto mode""",
    )
    parser.add_argument(
        "--load_way",
        type=str,
        default="HF",
        help="the way of loading model weights, the default is HF(Huggingface format), "
        "llama also supports DS(Deepspeed)",
    )
    parser.add_argument(
        "--data_type", type=str, default="float16", help="the data type for model inference, default is float16"
    )
    parser.add_argument(
        "--max_total_token_num",
        type=int,
        default=64 * 1024,
        help="the total token nums the gpu and model can support, equals = max_batch * (input_len + output_len)",
    )
    parser.add_argument(
        "--batch_max_tokens",
        type=int,
        default=None,
        help="max tokens num for new cat batch, it control prefill batch size to Preventing OOM",
    )
    parser.add_argument(
        "--running_max_req_size", type=int, default=30, help="the max size for forward requests in the same time"
    )
    parser.add_argument(
        "--max_req_total_len", type=int, default=8192, help="the max value for req_input_len + req_output_len"
    )
    parser.add_argument("--encode_process_num", type=int, default=1)
    parser.add_argument("--decode_process_num", type=int, default=1)
    parser.add_argument("--encode_paral_num", type=int, default=50)
    parser.add_argument("--gpt_paral_num", type=int, default=50)
    parser.add_argument("--gpt_paral_step_num", type=int, default=200)
    parser.add_argument("--decode_paral_num", type=int, default=1)
    parser.add_argument(
        "--decode_max_batch_size",
        type=int,
        default=1,
        help="the max batch size for token2wav, currently only support 1",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=["triton_flashdecoding"],
        nargs="+",
        help="""Model mode: [triton_int8kv | ppl_int8kv | ppl_fp16 | triton_flashdecoding
                        | triton_gqa_attention | triton_gqa_flashdecoding]
                        [triton_int8weight | triton_int4weight | lmdeploy_int4weight | ppl_int4weight],
                        triton_flashdecoding mode is for long context, current support llama llama2 qwen;
                        triton_gqa_attention and triton_gqa_flashdecoding is fast kernel for model which use GQA;
                        triton_int8kv mode use int8 to store kv cache, can increase token capacity, use triton kernel;
                        ppl_int8kv mode use int8 to store kv cache, and use ppl fast kernel;
                        ppl_fp16 mode use ppl fast fp16 decode attention kernel;
                        triton_int8weight and triton_int4weight and lmdeploy_int4weight or ppl_int4weight mode use int8
                        and int4 to store weights, you can use --data_type to specify the data type for model inference;
                        you need to read source code to make sure the supported detail mode for all models""",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether or not to allow for custom models defined on the Hub in their own modeling files.",
    )
    parser.add_argument("--disable_log_stats", action="store_true", help="disable logging throughput stats.")
    parser.add_argument("--log_stats_interval", type=int, default=10, help="log stats interval in second.")
    parser.add_argument("--router_token_ratio", type=float, default=0.0, help="token ratio to control router dispatch")
    parser.add_argument(
        "--router_max_new_token_len", type=int, default=1024, help="the request max new token len for router"
    )
    parser.add_argument(
        "--router_max_wait_tokens",
        type=int,
        default=8,
        help="schedule new requests after every router_max_wait_tokens decode steps.",
    )
    parser.add_argument(
        "--cache_capacity",
        type=int,
        default=200,
        help="prompt wav cache capacity, cache server capacity for multimodal resources",
    )
    parser.add_argument(
        "--cache_reserved_ratio", type=float, default=0.5, help="cache server reserved capacity ratio after clear"
    )
    parser.add_argument("--sample_close", action="store_true", help="close sample function for tts_llm")
    parser.add_argument("--health_monitor", action="store_true", help="health check time interval")
    parser.add_argument("--disable_cudagraph", action="store_true", help="Disable the cudagraph of the decoding stage")
    parser.add_argument(
        "--graph_max_batch_size",
        type=int,
        default=16,
        help="""Maximum batch size that can be captured by the cuda graph for decodign stage.
                The default value is 16. It will turn into eagar mode if encounters a larger value.""",
    )
    parser.add_argument(
        "--graph_max_len_in_batch",
        type=int,
        default=8192,
        help="""Maximum sequence length that can be captured by the cuda graph for decodign stage.
                The default value is 8192. It will turn into eagar mode if encounters a larger value. """,
    )
    parser.add_argument(
        "--load_jit",
        type=str2bool,
        default=False,
        help="Whether to load the flow_encoder in JIT mode.",
    )
    parser.add_argument(
        "--load_trt",
        type=str2bool,
        default=True,
        help="Whether to load the flow_decoder in trt.",
    )
    return parser
