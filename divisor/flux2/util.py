# SPDX-License-Identifier:Apache-2.0
# original BFL Flux code from https://github.com/black-forest-labs/flux2

import base64
import io
import os
import torch
from PIL import Image
from safetensors.torch import load_file as load_sft
from nnll.init_gpu import device
from nnll.console import nfo

from divisor.flux1.loading import retrieve_model
from divisor.flux2.autoencoder import AutoEncoder, AutoEncoderParams
from divisor.flux2.model import Flux2, Flux2Params
from divisor.flux2.text_encoder import Mistral3SmallEmbedder
from divisor.flux2 import precision


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


def load_flow_model(model_name: str, device: torch.device = device) -> Flux2:
    config = FLUX2_MODEL_INFO[model_name.lower()]

    weight_path = retrieve_model(
        repo_id=config["repo_id"],
        file_name=config["filename"],
    )

    with torch.device("meta"):
        model = Flux2(FLUX2_MODEL_INFO[model_name.lower()]["params"]).to(precision)
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


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str
