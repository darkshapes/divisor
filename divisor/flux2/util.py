import base64
import io
import os
import sys

import huggingface_hub
import torch
from PIL import Image
from safetensors.torch import load_file as load_sft

from divisor.flux2.autoencoder import AutoEncoder, AutoEncoderParams
from divisor.flux2.model import Flux2, Flux2Params
from divisor.flux2.text_encoder import Mistral3SmallEmbedder
from divisor.flux2 import precision


FLUX2_MODEL_INFO = {
    "flux.2-dev": {
        "repo_id": "black-forest-labs/FLUX.2-dev",
        "filename": "flux2-dev.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Flux2Params(),
    }
}
FLUX2_FP8_MODEL_INFO = {
    "flux.2-dev": {
        "repo_id": "Comfy-Org/flux2-dev",
        "filename": "split_files/diffusion_models/flux2_dev_fp8mixed.safetensors",
        "filename_ae": "split_files/vae/flux2-vae.safetensors",
        "params": Flux2Params(),
    }
}

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
device_name = device.type


def load_flow_model(model_name: str, debug_mode: bool = False, device=device_name) -> Flux2:
    config = FLUX2_MODEL_INFO[model_name.lower()]

    if debug_mode:
        config["params"].depth = 1
        config["params"].depth_single_blocks = 1
    else:
        if "FLUX2_MODEL_PATH" in os.environ:
            weight_path = os.environ["FLUX2_MODEL_PATH"]
            assert os.path.exists(weight_path), f"Provided weight path {weight_path} does not exist"
        else:
            # download from huggingface
            try:
                weight_path = huggingface_hub.hf_hub_download(
                    repo_id=config["repo_id"],
                    filename=config["filename"],
                    repo_type="model",
                )
            except huggingface_hub.errors.RepositoryNotFoundError:
                print(f"Failed to access the model repository. Please check your internet connection and make sure you've access to {config['repo_id']}.Stopping.")
                sys.exit(1)

    if not debug_mode:
        with torch.device("meta"):
            model = Flux2(FLUX2_MODEL_INFO[model_name.lower()]["params"]).to(precision)
        print(f"Loading {weight_path} for the FLUX.2 weights")
        sd = load_sft(weight_path, device=device_name)
        model.load_state_dict(sd, strict=False, assign=True)
        return model.to(device)
    else:
        with torch.device(device_name):
            return Flux2(FLUX2_MODEL_INFO[model_name.lower()]["params"]).to(precision)


def load_mistral_small_embedder(device: str | torch.device = device_name) -> Mistral3SmallEmbedder:
    return Mistral3SmallEmbedder().to(device)


def load_ae(model_name: str, device: str | torch.device = device_name) -> AutoEncoder:
    config = FLUX2_MODEL_INFO[model_name.lower()]

    if "AE_MODEL_PATH" in os.environ:
        weight_path = os.environ["AE_MODEL_PATH"]
        assert os.path.exists(weight_path), f"Provided weight path {weight_path} does not exist"
    else:
        # download from huggingface
        try:
            weight_path = huggingface_hub.hf_hub_download(
                repo_id=config["repo_id"],
                filename=config["filename_ae"],
                repo_type="model",
            )
        except huggingface_hub.errors.RepositoryNotFoundError:
            print(f"Failed to access the model repository. Please check your internet connection and make sure you've access to {config['repo_id']}.Stopping.")
            sys.exit(1)

    if isinstance(device, str):
        device = torch.device(device)
    with torch.device("meta"):
        ae = AutoEncoder(AutoEncoderParams())

    print(f"Loading {weight_path} for the AutoEncoder weights")
    sd = load_sft(weight_path, device=str(device))
    ae.load_state_dict(sd, strict=True, assign=True)
    return ae.to(device)


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str
