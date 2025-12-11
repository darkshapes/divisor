# SPDX-License-Identifier:Apache-2.0
# original BFL Flux code from https://github.com/black-forest-labs/flux2

import os

from nnll.console import nfo
from nnll.init_gpu import device
from safetensors.torch import load_file as load_sft
import torch

from divisor.flux1.loading import retrieve_model
from divisor.flux1.spec import configs as flux_configs
from divisor.flux2 import precision
from divisor.flux2.autoencoder import AutoEncoder, AutoEncoderParams
from divisor.flux2.model import Flux2, Flux2Params
from divisor.flux2.text_encoder import Mistral3SmallEmbedder
from divisor.spec import get_model_spec


FLUX2_MODEL_INFO = {
    "model.dit.flux2-dev": {
        "repo_id": "black-forest-labs/FLUX.2-dev",
        "filename": "flux2-dev.safetensors",
        "params": Flux2Params(),
    }
}
FLUX2_VAE_INFO = {
    "model.vae.flux2-dev": {
        "repo_id": "black-forest-labs/FLUX.2-dev",
        "filename": "ae.safetensors",
        "params": AutoEncoderParams(),
    }
}

FLUX2_FP8_MODEL_INFO = {
    "model.dit.flux2-dev:fp8-sai": {
        "repo_id": "Comfy-Org/flux2-dev",
        "filename": "split_files/diffusion_models/flux2_dev_fp8mixed.safetensors",
        "filename_ae": "split_files/vae/flux2-vae.safetensors",
        "params": Flux2Params(),
    }
}


def load_flow_model(mir_id: str, device: torch.device = device) -> Flux2:
    model_spec = get_model_spec(mir_id, flux_configs)
    weight_path = retrieve_model(
        repo_id=model_spec["repo_id"],
        file_name=model_spec["file_name"],
    )
    with torch.device("meta"):
        model = Flux2(model_spec["params"]).to(precision)
    nfo(f": {os.path.basename(weight_path)}")
    sd = load_sft(weight_path, device=str(device))
    model.load_state_dict(sd, strict=False, assign=True)
    return model.to(device)


def load_mistral_small_embedder(device: str | torch.device = device) -> Mistral3SmallEmbedder:
    return Mistral3SmallEmbedder().to(device)


def load_ae(model_name: str, device: str | torch.device = device) -> AutoEncoder:
    config = FLUX2_VAE_INFO[model_name.lower()]

    weight_path = retrieve_model(repo_id=config["repo_id"], file_name=config["filename"])

    if isinstance(device, str):
        device = torch.device(device)
    with torch.device("meta"):
        ae = AutoEncoder(AutoEncoderParams())

    nfo(f": {os.path.basename(weight_path)}")
    sd = load_sft(weight_path, device=str(device))
    ae.load_state_dict(sd, strict=True, assign=True)
    return ae.to(device)
