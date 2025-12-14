# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->
# adapted BFL Flux code from https://github.com/black-forest-labs/flux and  https://github.com/Gen-Verse/MMaDA


from dataclasses import replace
from dataclasses import dataclass
from typing import Any

from nnll.console import nfo
import torch

from divisor.contents import build_available_models
from divisor.flux1.autoencoder import AutoEncoderParams as AutoEncoder1Params
from divisor.flux1.model import FluxLoraWrapper, FluxParams
from divisor.flux2.autoencoder import AutoEncoderParams as AutoEncoder2Params
from divisor.flux2.model import Flux2Params
from divisor.flux2.model import Flux2
from divisor.xflux1.model import XFluxParams
from divisor.xflux1.model import XFlux
from divisor.mmada.modeling_mmada import MMadaConfig as MMaDAParams, MMadaModelLM


@dataclass
class CompatibilitySpec:
    repo_id: str
    file_name: str


@dataclass
class InitialParamsFlux:
    num_steps: int
    max_length: int
    guidance: float
    shift: bool
    width: int = 1360
    height: int = 768


@dataclass
class InitialParamsMMaDA:
    """Default initialization parameters for MMaDA models."""

    steps: int
    gen_length: int
    block_length: int
    temperature: float
    cfg_scale: float
    remasking_strategy: str
    mask_id: int
    max_position_embeddings: int
    max_text_len: int


@dataclass
class AutoencoderTinyParams:
    """"""


@dataclass
class ModelSpec:
    repo_id: str
    params: FluxParams | AutoEncoder1Params | XFluxParams | Flux2Params | MMaDAParams | AutoEncoder2Params | AutoencoderTinyParams | FluxLoraWrapper
    file_name: str
    init: InitialParamsFlux | InitialParamsMMaDA | None = None


flux_configs: dict[str, dict[str, ModelSpec | CompatibilitySpec]] = {
    "model.dit.flux1-dev": {
        "*": ModelSpec(
            repo_id="black-forest-labs/FLUX.1-dev",
            file_name="flux1-dev.safetensors",
            init=InitialParamsFlux(
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
        "@@fp8-e5m2-sai": CompatibilitySpec(
            repo_id="Kijai/flux-fp8",
            file_name="flux1-dev-fp8-e5m2.safetensors",
        ),
        "*@fp8-e4m3fn-sai": CompatibilitySpec(
            repo_id="Kijai/flux-fp8",
            file_name="flux1-dev-fp8-e4m3fn.safetensors",
        ),
        "*@fp8-sai": CompatibilitySpec(
            repo_id="XLabs-AI/flux-dev-fp8",
            file_name="flux-dev-fp8.safetensors",
        ),
        "mini": ModelSpec(
            repo_id="TencentARC/flux-mini",
            file_name="flux-mini.safetensors",
            init=InitialParamsFlux(
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
            params=AutoEncoder1Params(
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
        "*": ModelSpec(repo_id="madebyollin/taef1", file_name="diffusion_pytorch_model.safetensors", params=AutoencoderTinyParams()),
    },
    "model.dit.flux1-schnell": {
        "*": ModelSpec(
            repo_id="black-forest-labs/FLUX.1-schnell",
            file_name="flux1-schnell.safetensors",
            init=InitialParamsFlux(
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
        "*@fp8-sai": CompatibilitySpec(
            repo_id="Comfy-Org/flux1-schnell",
            file_name="flux1-schnell-fp8.safetensors",
        ),
        "*@fp8-e4m3fn-sai": CompatibilitySpec(
            repo_id="Kijai/flux-fp8",
            file_name="flux1-schnell-fp8-e4m3fn.safetensors",
        ),
    },
    "model.dit.flux2-dev": {
        "*": ModelSpec(
            repo_id="black-forest-labs/FLUX.2-dev",
            file_name="flux2-dev.safetensors",
            params=Flux2Params(),
        ),
        "*@fp8-sai": CompatibilitySpec(
            repo_id="Comfy-Org/flux2-dev",
            file_name="split_files/diffusion_models/flux2_dev_fp8mixed.safetensors",
        ),
    },
    "model.vae.flux2-dev": {
        "*": ModelSpec(
            repo_id="black-forest-labs/FLUX.2-dev",
            file_name="ae.safetensors",
            params=AutoEncoder2Params(),
        )
    },
}

mmada_configs = {
    "model.mldm.mmada": {
        "*": ModelSpec(
            repo_id="Gen-Verse/MMaDA-8B-Base",
            file_name="model.safetensors",
            init=InitialParamsMMaDA(
                steps=256,
                gen_length=512,
                block_length=128,
                temperature=1.0,
                cfg_scale=0.0,
                remasking_strategy="low_confidence",
                mask_id=126336,
                max_position_embeddings=2048,
                max_text_len=512,
            ),
            params=MMaDAParams(
                vocab_size=50257,
                llm_vocab_size=50257,
                llm_model_path="",
                codebook_size=8192,
                num_vq_tokens=1024,
                num_new_special_tokens=0,
            ),
        ),
        "mixcot": CompatibilitySpec(
            repo_id="Gen-Verse/MMaDA-8B-MixCoT",
            file_name="model.safetensors",
        ),
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


def merge_spec(base_spec: Any, subkey_spec: Any) -> ModelSpec:
    """Merge two dataclass or nested dataclass specs with overlapping subkey values taking precedence over base values.\n
    :param base_spec: Base specification dataclass
    :param subkey_spec: Subkey specification dataclass (values take precedence)
    :returns: Merged specification with subkey values overriding base values
    """
    if not hasattr(subkey_spec, "__dataclass_fields__"):
        return base_spec

    merge_kwargs = {}
    for field_name in subkey_spec.__dataclass_fields__:
        subkey_value = getattr(subkey_spec, field_name, None)
        base_value = getattr(base_spec, field_name, None)

        if subkey_value is not None:
            if hasattr(subkey_value, "__dataclass_fields__") and base_value is not None and hasattr(base_value, "__dataclass_fields__"):
                nested_merge_kwargs = {}
                for nested_field in subkey_value.__dataclass_fields__:
                    nested_subkey_val = getattr(subkey_value, nested_field, None)
                    if nested_subkey_val is not None:
                        nested_merge_kwargs[nested_field] = nested_subkey_val
                if nested_merge_kwargs:
                    merge_kwargs[field_name] = replace(base_value, **nested_merge_kwargs)
                else:
                    merge_kwargs[field_name] = subkey_value
            else:
                merge_kwargs[field_name] = subkey_value

    if merge_kwargs:
        return replace(base_spec, **merge_kwargs)
    return base_spec


def get_model_spec(mir_id: str, configs: dict[str, dict[str, ModelSpec | CompatibilitySpec]]) -> ModelSpec:
    """Get a ModelSpec or CompatibilitySpec for a given model ID. Use to point to a known model spec.\n
    :param mir_id: Model ID (e.g., "model.dit.flux1-dev")
    :param configs: Configuration mapping containing model specs
    :returns: ModelSpec if compatibility_key is None, CompatibilitySpec if provided and available, None if provided but not found
    :raises ValueError: If model ID does not have a base ModelSpec
    """

    if ":" in mir_id:
        series_key, compatibility_key = mir_id.split(":")
        if base_spec := configs.get(series_key, {}).get("*", None):
            if compatibility_spec := configs.get(series_key, {}).get(compatibility_key, None):
                merged_spec = merge_spec(base_spec, compatibility_spec)
                return merged_spec
    else:
        if model_spec := configs.get(mir_id, {}).get("*", None):
            if isinstance(model_spec, ModelSpec):
                return model_spec

    raise ValueError(f"{mir_id} has no defined model spec")


mmada_map = build_available_models(mmada_configs)

flux_map = build_available_models(flux_configs)
