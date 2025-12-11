# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->
# adapted BFL Flux code from https://github.com/black-forest-labs/flux

from dataclasses import dataclass

from diffusers.models.autoencoders.autoencoder_tiny import AutoencoderTiny
from nnll.console import nfo
import torch

from divisor.flux1.autoencoder import AutoEncoderParams
from divisor.flux1.model import FluxLoraWrapper, FluxParams
from divisor.flux2.autoencoder import AutoEncoderParams as AutoEncoder2Params
from divisor.flux2.model import Flux2Params
from divisor.xflux1.model import XFluxParams


@dataclass
class CompatibilitySpec:
    repo_id: str
    file_name: str


@dataclass
class InitialParams:
    num_steps: int | None = None
    max_length: int | None = None
    guidance: float | None = None
    shift: bool | None = None
    width: int = 1360
    height: int = 768


@dataclass
class ModelSpec:
    repo_id: str
    file_name: str
    params: FluxParams | AutoEncoderParams | XFluxParams | Flux2Params | AutoEncoder2Params | type[AutoencoderTiny] | type[FluxLoraWrapper]
    init: InitialParams


configs: dict[str, dict[str, ModelSpec | CompatibilitySpec]] = {
    "model.dit.flux1-dev": {
        "*": ModelSpec(
            repo_id="black-forest-labs/FLUX.1-dev",
            file_name="flux1-dev.safetensors",
            init=InitialParams(
                num_steps=28,
                max_length=512,
                guidance=4.0,
                shift=True,
            ),
            params=FluxParams(
                in_channels=64,
                vec_in_dim=768,
                context_in_dim=4096,
                hidden_size=3072,
                mlp_ratio=4.0,
                num_heads=24,
                depth=19,
                depth_single_blocks=38,
                axes_dim=[16, 56, 56],
                theta=10_000,
                qkv_bias=True,
                guidance_embed=True,
            ),
        ),
        "fp8-e5m2-sai": CompatibilitySpec(
            repo_id="Kijai/flux-fp8",
            file_name="flux1-dev-fp8-e5m2.safetensors",
        ),
        "fp8-e4m3fn-sai": CompatibilitySpec(
            repo_id="Kijai/flux-fp8",
            file_name="flux1-dev-fp8-e4m3fn.safetensors",
        ),
        "fp8-sai": CompatibilitySpec(
            repo_id="XLabs-AI/flux-dev-fp8",
            file_name="flux-dev-fp8.safetensors",
        ),
        "mini": ModelSpec(
            repo_id="TencentARC/flux-mini",
            file_name="flux-mini.safetensors",
            init=InitialParams(
                num_steps=25,
                max_length=512,
                guidance=3.5,
                shift=True,
            ),
            params=XFluxParams(
                in_channels=64,
                vec_in_dim=768,
                context_in_dim=4096,
                hidden_size=3072,
                mlp_ratio=4.0,
                num_heads=24,
                depth=5,
                depth_single_blocks=10,
                axes_dim=[16, 56, 56],
                theta=10_000,
                qkv_bias=True,
                guidance_embed=True,
            ),
        ),
    },
    "model.vae.flux1-dev": {
        "*": ModelSpec(
            repo_id="black-forest-labs/FLUX.1-dev",
            file_name="ae.safetensors",
            init=InitialParams(
                num_steps=25,
                max_length=512,
                guidance=3.5,
                shift=True,
            ),
            params=AutoEncoderParams(
                resolution=256,
                in_channels=3,
                ch=128,
                out_ch=3,
                ch_mult=[1, 2, 4, 4],
                num_res_blocks=2,
                z_channels=16,
                scale_factor=0.3611,
                shift_factor=0.1159,
            ),
        ),
    },
    "model.taesd.flux1-dev": {
        "*": ModelSpec(
            repo_id="madebyollin/taef1",
            file_name="diffusion_pytorch_model.safetensors",
            init=InitialParams(),
            params=AutoencoderTiny,
        ),
    },
    "model.dit.flux1-schnell": {
        "*": ModelSpec(
            repo_id="black-forest-labs/FLUX.1-schnell",
            file_name="flux1-schnell.safetensors",
            init=InitialParams(
                num_steps=4,
                max_length=256,
                guidance=2.5,
                shift=False,
            ),
            params=FluxParams(
                in_channels=64,
                vec_in_dim=768,
                context_in_dim=4096,
                hidden_size=3072,
                mlp_ratio=4.0,
                num_heads=24,
                depth=19,
                depth_single_blocks=38,
                axes_dim=[16, 56, 56],
                theta=10_000,
                qkv_bias=True,
                guidance_embed=False,
            ),
        ),
        "fp8-sai": CompatibilitySpec(
            repo_id="Comfy-Org/flux1-schnell",
            file_name="flux1-schnell-fp8.safetensors",
        ),
        "fp8-e4m3fn-sai": CompatibilitySpec(
            repo_id="Kijai/flux-fp8",
            file_name="flux1-schnell-fp8-e4m3fn.safetensors",
        ),
    },
    "model.dit.flux2-dev": {
        "*": ModelSpec(
            repo_id="black-forest-labs/FLUX.2-dev",
            file_name="flux2-dev.safetensors",
            params=Flux2Params(),
            init=InitialParams(),
        )
    },
    "model.vae.flux2-dev": {
        "*": ModelSpec(
            repo_id="black-forest-labs/FLUX.2-dev",
            file_name="ae.safetensors",
            params=AutoEncoder2Params(),
            init=InitialParams(),
        )
    },
    "model.dit.flux2-dev:fp8-sai": {
        "*": ModelSpec(
            repo_id="Comfy-Org/flux2-dev",
            file_name="split_files/diffusion_models/flux2_dev_fp8mixed.safetensors",
            params=Flux2Params(),
            init=InitialParams(),
        )
    },
}


def optionally_expand_state_dict(model: torch.nn.Module, state_dict: dict) -> dict:
    """Optionally expand the state dict to match the model's parameters shapes.\n
    :param model: The model to match parameters against
    :param state_dict: The state dictionary to expand
    :returns: The expanded state dictionary
    """
    for name, param in model.named_parameters():
        if name in state_dict:
            if state_dict[name].shape != param.shape:
                nfo(f"Expanding '{name}' with shape {state_dict[name].shape} to model parameter with shape {param.shape}.")
                # expand with zeros:
                expanded_state_dict_weight = torch.zeros_like(param, device=state_dict[name].device)
                slices = tuple(slice(0, dim) for dim in state_dict[name].shape)
                expanded_state_dict_weight[slices] = state_dict[name]
                state_dict[name] = expanded_state_dict_weight

    return state_dict
