import torch
import os
import gc
from safetensors import safe_open
import light_tts.utils.petrel_helper as utils
from light_tts.utils.dist_utils import get_current_device_id
from light_tts.utils.model_crypto import ENC_SUFFIX, decrypt_to_bytes, load_key_from_env


def load_func(file_, use_safetensors=False, pre_post_layer=None, transformer_layer_list=None, weight_dir=None):
    # fix bug for 多线程加载的时候，每个线程内部的cuda device 会切回 0， 修改后来保证不会出现bug
    import torch.distributed as dist

    torch.cuda.set_device(get_current_device_id())

    full_path = os.path.join(weight_dir, file_)
    enc_path = full_path + ENC_SUFFIX
    use_enc = (not os.path.exists(full_path)) and os.path.exists(enc_path)

    if use_safetensors:
        if use_enc:
            from safetensors.torch import load as st_load_bytes
            plaintext = decrypt_to_bytes(enc_path, load_key_from_env())
            weights = st_load_bytes(plaintext)
            del plaintext
        else:
            weights = safe_open(full_path, "pt", "cpu")
            weights = {k: weights.get_tensor(k) for k in weights.keys()}
    else:
        weights = utils.PetrelHelper.load(full_path, map_location="cpu")

    if pre_post_layer is not None:
        pre_post_layer.load_hf_weights(weights)
    if transformer_layer_list is not None:
        for layer in transformer_layer_list:
            layer.load_hf_weights(weights)
    del weights
    gc.collect()


def load_hf_weights(data_type, weight_dir, pre_post_layer=None, transformer_layer_list=None, weight_dict=None):
    if isinstance(data_type, str):
        data_type = torch.float16 if data_type == "fp16" else torch.float32
    if pre_post_layer is not None:
        assert pre_post_layer.data_type_ == data_type, "type is not right"
    if transformer_layer_list is not None:
        assert transformer_layer_list[0].data_type_ == data_type, "type is not right"
    if weight_dict:
        if pre_post_layer is not None:
            pre_post_layer.load_hf_weights(weight_dict)
        if transformer_layer_list is not None:
            for layer in transformer_layer_list:
                layer.load_hf_weights(weight_dict)
        del weight_dict
        return
    use_safetensors = True
    files = utils.PetrelHelper.list(weight_dir, extension="all")
    # Treat <name>.safetensors.enc as <name>.safetensors so the rest of the
    # pipeline keeps working with the plaintext filename; load_func decrypts
    # at read time when the plaintext file is missing.
    normalized = [f[: -len(ENC_SUFFIX)] if f.endswith(ENC_SUFFIX) else f for f in files]
    candidate_files = list(filter(lambda x: x.endswith(".safetensors"), normalized))
    if len(candidate_files) == 0:
        use_safetensors = False
        candidate_files = list(filter(lambda x: x.endswith(".bin"), normalized))
    # de-dup in case both plain and .enc appear
    candidate_files = sorted(set(candidate_files))
    assert len(candidate_files) != 0, "can only support pytorch tensor and safetensors format for weights."
    from functools import partial
    from multiprocessing.pool import ThreadPool as Pool

    partial_func = partial(
        load_func,
        use_safetensors=use_safetensors,
        pre_post_layer=pre_post_layer,
        transformer_layer_list=transformer_layer_list,
        weight_dir=weight_dir,
    )  # noqa
    worker = int(os.environ.get("LOADWORKER", 1))
    with Pool(worker) as p:
        _ = p.map(partial_func, candidate_files)

    return
