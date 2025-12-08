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
    num_steps: int
    max_length: int
    guidance: float
    shift: bool
    width: int = 1360
    height: int = 768


@dataclass
class ModelSpec:
    repo_id: str
    file_name: str
    params: FluxParams | AutoEncoderParams | XFluxParams | Flux2Params | AutoEncoder2Params | type[AutoencoderTiny] | type[FluxLoraWrapper]
    init: InitialParams | None = None


configs = {
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
        "*": ModelSpec(repo_id="madebyollin/taef1", file_name="diffusion_pytorch_model.safetensors", params=AutoencoderTiny),
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
        )
    },
    "model.vae.flux2-dev": {
        "*": ModelSpec(
            repo_id="black-forest-labs/FLUX.2-dev",
            file_name="ae.safetensors",
            params=AutoEncoder2Params(),
        )
    },
    "model.dit.flux2-dev:fp8-sai": {
        "*": ModelSpec(
            repo_id="Comfy-Org/flux2-dev",
            file_name="split_files/diffusion_models/flux2_dev_fp8mixed.safetensors",
            params=Flux2Params(),
        )
    },
}


def get_model_spec(mir_id: str, compatibility_key: str | None = None) -> ModelSpec | CompatibilitySpec | None:
    """Get a ModelSpec or CompatibilitySpec for a given model ID.\n
    :param mir_id: Model ID (e.g., "model.dit.flux1-dev")
    :param compatibility_key: Optional compatibility key (e.g., "fp8-sai"). If None, returns base ModelSpec.
    :returns: ModelSpec if compatibility_key is None, CompatibilitySpec if provided and available, None if provided but not found
    """
    if mir_id not in configs:
        if compatibility_key is None:
            available = ", ".join(configs.keys())
            raise ValueError(f"Unknown model ID: {mir_id}. Available: {available}")
        return None

    config_dict = configs[mir_id]

    # If compatibility_key is provided, try to get compatibility spec
    if compatibility_key is not None:
        compat_spec = config_dict.get(compatibility_key)
        if compat_spec is None:
            return None
        return compat_spec
    else:
        # Otherwise, return base ModelSpec from "*" key
        if "*" not in config_dict:
            raise ValueError(f"Model {mir_id} does not have a base spec (missing '*' key)")

        base_spec = config_dict["*"]
        if not isinstance(base_spec, ModelSpec):
            raise ValueError(f"Model {mir_id} base spec is not a ModelSpec")

        return base_spec


def get_merged_model_spec(mir_id: str, compatibility_key: str | None = None) -> ModelSpec:
    """Get a ModelSpec with compatibility overrides merged in.\n
    :param mir_id: Model ID (e.g., "model.dit.flux1-dev")
    :param compatibility_key: Optional compatibility key (e.g., "fp8-sai")
    :returns: ModelSpec with compatibility overrides applied
    """
    base_spec = get_model_spec(mir_id)
    if not isinstance(base_spec, ModelSpec):
        raise ValueError(f"Model {mir_id} does not have a base ModelSpec")
    if compatibility_key is None:
        return base_spec

    compat_spec = get_model_spec(mir_id, compatibility_key)
    if compat_spec is None:
        raise ValueError(f"Model {mir_id} does not have compatibility spec '{compatibility_key}'")

    if isinstance(compat_spec, CompatibilitySpec):
        return ModelSpec(
            repo_id=compat_spec.repo_id,
            file_name=compat_spec.file_name,
            params=base_spec.params,
            init=base_spec.init,
        )
    return base_spec


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
