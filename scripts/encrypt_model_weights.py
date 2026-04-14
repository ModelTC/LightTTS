"""Offline tool: produce an encrypted mirror of a LightTTS model directory.

Weight files (.safetensors/.bin/.pt/.onnx) are encrypted with AES-256-GCM and
written as <name>.enc in the output directory. All other files (config.json,
*.yaml, tokenizer, ...) are copied verbatim via shutil.copy2. The output
directory is fully self-contained — no symlinks.

Usage:
    python scripts/encrypt_model_weights.py \
        --input_dir  /path/to/plain_model_dir \
        --output_dir /path/to/encrypted_model_dir

Key source (in priority order):
    1) --key-hex <64-char-hex>
    2) --key-env <env_var_name>   (default: LIGHT_TTS_MODEL_KEY)

Generate a key:
    python -c "import secrets; print(secrets.token_hex(32))"
"""
import argparse
import os
import shutil
import sys

# Allow running directly from repo root without installing as package.
_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_THIS)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from light_tts.utils.model_crypto import (  # noqa: E402
    ENC_SUFFIX,
    KEY_ENV,
    WEIGHT_EXTS,
    encrypt_file,
    is_weight_file,
)


def _resolve_key(args) -> bytes:
    if args.key_hex:
        raw = args.key_hex.strip()
    else:
        raw = os.environ.get(args.key_env, "").strip()
        if not raw:
            raise SystemExit(
                f"Key not found: env var {args.key_env} is unset and --key-hex not given.\n"
                f"Generate one via: python -c 'import secrets; print(secrets.token_hex(32))'"
            )
    try:
        key = bytes.fromhex(raw)
    except ValueError as e:
        raise SystemExit(f"Key must be a hex string: {e}")
    if len(key) != 32:
        raise SystemExit(f"Key must decode to 32 bytes (AES-256), got {len(key)}")
    return key


def _human(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    f = float(n)
    for u in units:
        if f < 1024.0:
            return f"{f:.2f} {u}"
        f /= 1024.0
    return f"{f:.2f} PB"


def _check_output_dir(output_dir: str, force: bool) -> None:
    if os.path.exists(output_dir):
        if not os.path.isdir(output_dir):
            raise SystemExit(f"--output_dir exists and is not a directory: {output_dir}")
        if os.listdir(output_dir) and not force:
            raise SystemExit(
                f"--output_dir is not empty: {output_dir}\n"
                f"Refusing to overwrite. Pass --force to remove its contents first."
            )
        if force:
            for name in os.listdir(output_dir):
                p = os.path.join(output_dir, name)
                if os.path.isdir(p) and not os.path.islink(p):
                    shutil.rmtree(p)
                else:
                    os.remove(p)
    else:
        os.makedirs(output_dir, mode=0o755, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_dir", required=True, help="Plain-text model directory")
    parser.add_argument("--output_dir", required=True, help="Destination for encrypted model directory")
    parser.add_argument("--key-env", default=KEY_ENV, help=f"Env var holding hex key (default: {KEY_ENV})")
    parser.add_argument("--key-hex", default=None, help="Hex-encoded key (overrides --key-env; use with care)")
    parser.add_argument("--force", action="store_true", help="Allow overwriting a non-empty --output_dir")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.isdir(input_dir):
        raise SystemExit(f"--input_dir does not exist or is not a directory: {input_dir}")
    if os.path.commonpath([input_dir, output_dir]) == input_dir and input_dir != output_dir:
        raise SystemExit("--output_dir must not be inside --input_dir")
    if input_dir == output_dir:
        raise SystemExit("--input_dir and --output_dir must differ")

    key = _resolve_key(args)
    _check_output_dir(output_dir, args.force)

    n_encrypted = 0
    n_copied = 0
    enc_bytes = 0
    cp_bytes = 0

    for root, _dirs, files in os.walk(input_dir):
        rel = os.path.relpath(root, input_dir)
        dst_subdir = output_dir if rel == "." else os.path.join(output_dir, rel)
        os.makedirs(dst_subdir, mode=0o755, exist_ok=True)

        for name in files:
            src_path = os.path.join(root, name)
            if os.path.islink(src_path):
                # follow & materialize the linked content so output_dir stays symlink-free
                src_path = os.path.realpath(src_path)
                if not os.path.isfile(src_path):
                    print(f"  SKIP broken symlink: {os.path.join(root, name)}", file=sys.stderr)
                    continue

            size = os.path.getsize(src_path)

            if name.endswith(ENC_SUFFIX):
                print(f"  SKIP already-encrypted: {os.path.join(rel, name)}")
                dst_path = os.path.join(dst_subdir, name)
                shutil.copy2(src_path, dst_path)
                n_copied += 1
                cp_bytes += size
                continue

            if is_weight_file(name):
                dst_path = os.path.join(dst_subdir, name + ENC_SUFFIX)
                encrypt_file(src_path, dst_path, key)
                n_encrypted += 1
                enc_bytes += size
                print(f"  ENC  {os.path.join(rel, name)}  ({_human(size)})")
            else:
                dst_path = os.path.join(dst_subdir, name)
                shutil.copy2(src_path, dst_path)
                n_copied += 1
                cp_bytes += size
                print(f"  COPY {os.path.join(rel, name)}  ({_human(size)})")

    print("-" * 60)
    print(f"Encrypted: {n_encrypted} files, total {_human(enc_bytes)}")
    print(f"Copied   : {n_copied} files, total {_human(cp_bytes)}")
    print(f"Output   : {output_dir}")
    print()
    print("Weight extensions encrypted:", ", ".join(WEIGHT_EXTS))
    print("Next steps:")
    print(f"  export {KEY_ENV}=<same hex key you used above>")
    print(f"  # point --model_dir in launcher.sh to {output_dir}")


if __name__ == "__main__":
    main()
