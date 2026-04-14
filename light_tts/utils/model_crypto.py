import io
import os
import shutil
from typing import Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


MAGIC = b"LTTS"
VERSION = 1
HEADER_LEN = len(MAGIC) + 1 + 12 + 16  # magic + version + nonce + tag
NONCE_LEN = 12
TAG_LEN = 16
KEY_ENV = "LIGHT_TTS_MODEL_KEY"
ENC_SUFFIX = ".enc"
WEIGHT_EXTS = (".safetensors", ".bin", ".pt", ".onnx")


def load_key_from_env() -> bytes:
    raw = os.environ.get(KEY_ENV)
    if not raw:
        raise RuntimeError(
            f"Environment variable {KEY_ENV} is not set. "
            f"Generate one via: python -c 'import secrets; print(secrets.token_hex(32))'"
        )
    try:
        key = bytes.fromhex(raw.strip())
    except ValueError as e:
        raise RuntimeError(f"{KEY_ENV} must be a hex string, got: {e}")
    if len(key) != 32:
        raise RuntimeError(
            f"{KEY_ENV} must decode to 32 bytes (AES-256), got {len(key)} bytes"
        )
    return key


def is_weight_file(filename: str) -> bool:
    lower = filename.lower()
    return any(lower.endswith(ext) for ext in WEIGHT_EXTS)


def encrypt_file(src_path: str, dst_path: str, key: bytes) -> None:
    with open(src_path, "rb") as f:
        plaintext = f.read()
    nonce = os.urandom(NONCE_LEN)
    aesgcm = AESGCM(key)
    ct_with_tag = aesgcm.encrypt(nonce, plaintext, None)
    ciphertext = ct_with_tag[:-TAG_LEN]
    tag = ct_with_tag[-TAG_LEN:]

    tmp_path = dst_path + ".partial"
    with open(tmp_path, "wb") as f:
        f.write(MAGIC)
        f.write(bytes([VERSION]))
        f.write(nonce)
        f.write(tag)
        f.write(ciphertext)
    os.replace(tmp_path, dst_path)


def decrypt_to_bytes(src_path: str, key: bytes) -> bytes:
    with open(src_path, "rb") as f:
        header = f.read(HEADER_LEN)
        if len(header) < HEADER_LEN or header[:4] != MAGIC:
            raise RuntimeError(f"Not a LightTTS encrypted file: {src_path}")
        version = header[4]
        if version != VERSION:
            raise RuntimeError(
                f"Unsupported encryption format version {version} in {src_path}"
            )
        nonce = header[5:5 + NONCE_LEN]
        tag = header[5 + NONCE_LEN:HEADER_LEN]
        ciphertext = f.read()

    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext + tag, None)


def decrypt_file(src_path: str, dst_path: str, key: bytes) -> None:
    plaintext = decrypt_to_bytes(src_path, key)
    os.makedirs(os.path.dirname(dst_path) or ".", mode=0o700, exist_ok=True)
    tmp_path = dst_path + ".partial"
    with open(tmp_path, "wb") as f:
        f.write(plaintext)
    os.replace(tmp_path, dst_path)


_PATCHED_TORCH = False
_PATCHED_ONNX = False
_PATCHED_QWEN2 = False


def apply_decryption_patches(
    key: Optional[bytes] = None,
    patch_onnx: bool = True,
    patch_qwen2: bool = True,
) -> None:
    """Monkey-patch model loaders so loading <path> falls back to in-memory
    decryption of <path>.enc when <path> does not exist on disk.

    torch.load is always patched. onnxruntime.InferenceSession is only
    patched when patch_onnx=True (default); disable in processes that don't
    use onnxruntime to avoid an unnecessary ``import onnxruntime`` and the
    CUDA/ROCm provider init it triggers. Same for patch_qwen2, which pulls
    in ``cosyvoice.llm.llm`` + ``transformers``.

    Idempotent per-patch: each patch installs at most once even across
    repeated calls. Must be invoked in every process that loads weights,
    since mp.Process re-imports modules from scratch.
    """
    global _PATCHED_TORCH, _PATCHED_ONNX, _PATCHED_QWEN2

    if key is None:
        key = load_key_from_env()

    if not _PATCHED_TORCH:
        import torch

        _orig_torch_load = torch.load

        def _patched_torch_load(f, *args, **kwargs):
            if isinstance(f, (str, os.PathLike)):
                path = os.fspath(f)
                if not os.path.exists(path):
                    enc = path + ENC_SUFFIX
                    if os.path.exists(enc):
                        plaintext = decrypt_to_bytes(enc, key)
                        return _orig_torch_load(io.BytesIO(plaintext), *args, **kwargs)
            return _orig_torch_load(f, *args, **kwargs)

        torch.load = _patched_torch_load
        _PATCHED_TORCH = True

    if patch_onnx and not _PATCHED_ONNX:
        try:
            import onnxruntime as ort

            _orig_session_cls = ort.InferenceSession

            def _patched_inference_session(path_or_bytes, *args, **kwargs):
                if isinstance(path_or_bytes, (str, os.PathLike)):
                    p = os.fspath(path_or_bytes)
                    if not os.path.exists(p):
                        enc = p + ENC_SUFFIX
                        if os.path.exists(enc):
                            return _orig_session_cls(decrypt_to_bytes(enc, key), *args, **kwargs)
                return _orig_session_cls(path_or_bytes, *args, **kwargs)

            ort.InferenceSession = _patched_inference_session
            _PATCHED_ONNX = True
        except ImportError:
            pass

    # transformers.from_pretrained walks the directory listing and only
    # recognizes plain "model.safetensors" — it has no idea about our .enc
    # mirror. Replace Qwen2Encoder.__init__ with a variant that builds the
    # model from config and loads weights directly from the decrypted bytes.
    if patch_qwen2 and not _PATCHED_QWEN2:
        try:
            from cosyvoice.llm import llm as _cos_llm_mod

            _orig_qwen2_init = _cos_llm_mod.Qwen2Encoder.__init__

            def _patched_qwen2_init(self, pretrain_path):
                safetensors_path = os.path.join(pretrain_path, "model.safetensors")
                enc_path = safetensors_path + ENC_SUFFIX
                if os.path.exists(enc_path) and not os.path.exists(safetensors_path):
                    import torch as _torch
                    from transformers import Qwen2ForCausalLM, Qwen2Config
                    from safetensors.torch import load as _st_load_bytes

                    _torch.nn.Module.__init__(self)
                    config = Qwen2Config.from_pretrained(pretrain_path)
                    self.model = Qwen2ForCausalLM(config)
                    plaintext = decrypt_to_bytes(enc_path, key)
                    state_dict = _st_load_bytes(plaintext)
                    del plaintext
                    # Qwen2 ties lm_head.weight to embed_tokens.weight when
                    # config.tie_word_embeddings is True; the safetensors file
                    # only stores one copy, so load with strict=False and re-tie.
                    missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
                    del state_dict
                    self.model.tie_weights()
                    tied_ok = bool(getattr(config, "tie_word_embeddings", False))
                    missing = [k for k in missing if not (tied_ok and k == "lm_head.weight")]
                    if missing or unexpected:
                        raise RuntimeError(
                            f"Qwen2 weight load mismatch — missing={missing}, unexpected={unexpected}"
                        )
                else:
                    _orig_qwen2_init(self, pretrain_path)

            _cos_llm_mod.Qwen2Encoder.__init__ = _patched_qwen2_init
            _PATCHED_QWEN2 = True
        except ImportError:
            pass
