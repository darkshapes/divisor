# SPDX-License-Identifier:Apache-2.0
# original XFlux code from https://github.com/TencentARC/FluxKits

import os

from nnll.console import nfo
from safetensors import safe_open
import torch

from divisor.flux1.loading import retrieve_model


def load_checkpoint(local_path, repo_id, name):
    nfo(f": {os.path.basename(name)}")
    if local_path is not None:
        if ".safetensors" in local_path:
            checkpoint = load_safetensors(local_path)
        else:
            checkpoint = torch.load(local_path, map_location="cpu")
    elif repo_id is not None and name is not None:
        checkpoint = retrieve_model(repo_id, name)
    else:
        raise ValueError("LOADING ERROR: you must specify local_path or repo_id with name in HF to download")
    return checkpoint


def load_safetensors(path):
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def get_lora_rank(checkpoint):
    for k in checkpoint.keys():
        if k.endswith(".down.weight"):
            return checkpoint[k].shape[0]
