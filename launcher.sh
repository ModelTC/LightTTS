export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export MIOPEN_FIND_MODE=2
# Encrypted model weights:
#   - Encrypt once:  python scripts/encrypt_model_weights.py --input_dir <plain> --output_dir <enc>
#   - LIGHT_TTS_MODEL_KEY: hex-encoded 32-byte key (must match the key used at encryption time)
#   Weights are decrypted in-memory at load time; nothing is written to disk.
export LIGHT_TTS_MODEL_KEY=$(cat /tmp/light_tts_key.hex)
python -m light_tts.server.api_server --model_dir /data/tts_model/tts_2026030901/tts_model_2026031001_enc/ --load_jit False --load_trt False --port 8101 --host 0.0.0.0
