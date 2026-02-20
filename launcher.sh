export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
python -m light_tts.server.api_server --model_dir /data/Fun-CosyVoice3-0.5B --load_jit False --load_trt False
