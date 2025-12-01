# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Divisor class definitions and configuration dataclasses."""

from dataclasses import dataclass, replace
from typing import Any, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class DenoisingState:
    """State of the denoising process at a given timestep."""

    current_timestep: float
    previous_timestep: Optional[float]
    current_sample: torch.Tensor
    timestep_index: int
    total_timesteps: int
    guidance: float
    layer_dropout: Optional[list[int]] = None
    width: Optional[int] = None
    height: Optional[int] = None
    seed: Optional[int] = None
    prompt: Optional[str] = None
    num_steps: Optional[int] = None
    vae_shift_offset: float = 0.0
    vae_scale_offset: float = 0.0
    use_previous_as_mask: bool = False
    variation_seed: Optional[int] = None
    variation_strength: float = 0.0
    deterministic: bool = False


@dataclass
class GetPredictionSettings:
    """Base configuration class for get_prediction function creation.

    Consolidates all arguments needed to create a get_prediction function.
    """

    model_ref: list[Any]
    img_ids: Tensor
    img: Tensor
    state: Any
    img_cond: Optional[Tensor]
    img_cond_seq: Optional[Tensor]
    img_cond_seq_ids: Optional[Tensor]
    current_txt: list[Tensor]
    current_txt_ids: list[Tensor]
    current_vec: list[Tensor]
    cached_prediction: list[Optional[Tensor]]
    cached_prediction_state: list[Optional[dict]]
    neg_pred_enabled: bool = False
    current_neg_txt: Optional[Tensor] | str | None = ""
    current_neg_txt_ids: Optional[Tensor] | str | None = ""
    current_neg_vec: Optional[Tensor] | str | None = ""
    true_gs: Optional[int] = None


@dataclass
class AdditionalPredictionSettings:
    """Additional configuration for XFlux1-specific prediction settings.

    Consolidates XFlux1-specific arguments for get_prediction function creation.
    """

    timestep_to_start_cfg: int
    image_proj: Tensor | None
    neg_image_proj: Tensor | None
    ip_scale: Tensor
    neg_ip_scale: Tensor
    current_timestep_index: list[int]


@dataclass
class DenoiseSettings:
    """Base configuration class for denoise function parameters.

    Consolidates common arguments needed for denoise functions across Flux1, Flux2, and XFlux1.
    """

    # Model and core inputs (required)
    img: Tensor
    img_ids: Tensor
    txt: Tensor
    txt_ids: Tensor
    state: Any  # DenoisingState
    ae: Any  # AutoEncoder
    timesteps: list[float]

    # Optional image conditioning
    img_cond: Tensor | None = None  # Channel-wise image conditioning (Flux1 only)
    img_cond_seq: Tensor | None = None  # Sequence-wise image conditioning
    img_cond_seq_ids: Tensor | None = None

    # Optional device and layer dropout
    device: torch.device | None = None
    initial_layer_dropout: Optional[list[int]] = None

    # Flux1/XFlux1 specific - CLIP embeddings
    vec: Tensor | None = None  # CLIP embeddings (Flux1/XFlux1)

    # Flux1/XFlux1 specific - negative prompt support
    neg_pred_enabled: bool = False
    neg_txt: Tensor | None = None
    neg_txt_ids: Tensor | None = None
    neg_vec: Tensor | None = None
    true_gs: float | int = 1

    # XFlux1 specific - IP-Adapter and CFG
    timestep_to_start_cfg: int = 0
    image_proj: Tensor | None = None
    neg_image_proj: Tensor | None = None
    ip_scale: Tensor | None = None
    neg_ip_scale: Tensor | None = None

    # Text embedders for prompt changes
    t5: Any | None = None  # T5 embedder (Flux1/XFlux1)
    clip: Any | None = None  # CLIP embedder (Flux1/XFlux1)
    text_embedder: Any | None = None  # Mistral embedder (Flux2)


@dataclass
class SimpleDenoiseSettingsFlux2:
    """Configuration for simple (non-interactive) Flux2 denoising.

    Consolidates arguments for the simple denoise function in Flux2.
    """

    model: Any  # Flux2
    img: Tensor
    img_ids: Tensor
    txt: Tensor
    txt_ids: Tensor
    timesteps: list[float]
    guidance: float = 4.0
    img_cond_seq: Tensor | None = None
    img_cond_seq_ids: Tensor | None = None


@dataclass
class SimpleDenoiseSettingsXFlux1:
    """Configuration for simple (non-interactive) XFlux1 denoising.

    Consolidates arguments for the denoise_simple function in XFlux1.
    """

    model: Any  # XFlux
    img: Tensor
    img_ids: Tensor
    txt: Tensor
    txt_ids: Tensor
    vec: Tensor  # CLIP embeddings (required for XFlux1)
    timesteps: list[float]
    guidance: float = 4.0


def find_mir_spec(
    model_id: str,
    ae_id: str,
    configs: dict,
    tiny: bool = False,
    prefix: str = "model.dit.",
) -> Tuple[str, Optional[str], str]:
    """Find/validate model specifications by MIR (Machine Intelligence Resource) ID.\n
    :param model_id: Model ID, optionally with subkey (e.g., "flux1-dev" or "flux1-dev:mini")
    :param ae_id: Autoencoder ID
    :param configs: Configuration dictionary containing model specs
    :param tiny: Whether to use tiny autoencoder prefix (model.taesd. instead of model.vae.)
    :param prefix: Prefix to add to model_id (default: "model.dit.")
    :returns: Tuple of (normalized_model_id, subkey, normalized_ae_id)
    :raises ValueError: If model_id, subkey, or ae_id is not found in configs
    """

    def _validate_in_configs(key: str, key_type: str, available: list[str] | None = None) -> None:
        """Helper to validate a key exists in configs."""
        if key not in configs:
            available_keys = available if available is not None else list(configs.keys())
            available_str = ", ".join(available_keys)
            raise ValueError(f"Got unknown {key_type}: {key}, chose from {available_str}")

    # Handle model_id with optional subkey
    subkey = None
    if ":" in model_id:
        base_model_id, subkey = model_id.split(":", 1)
        normalized_model_id = f"{prefix}{base_model_id}".lower()
        subkey = subkey.lower()
        _validate_in_configs(
            normalized_model_id,
            "base model id",
        )
        base_model = configs[normalized_model_id]

        if subkey not in base_model:
            available_subkeys = [k for k in base_model.keys() if k != "*"]
            available_str = ", ".join(available_subkeys)
            raise ValueError(f"Got unknown subkey '{subkey}' for model {normalized_model_id}. Available subkeys: {available_str}")

        base_spec = base_model["*"]
        subkey_spec = base_model[subkey]

        # Merge subkey spec with base spec (subkey values take precedence)
        merged_spec = merge_spec(base_spec, subkey_spec)
        if merged_spec is not base_spec:
            # Update configs with merged spec so subsequent lookups use merged values
            configs[normalized_model_id] = {**base_model, "*": merged_spec}
    else:
        normalized_model_id = f"{prefix}{model_id}".lower()
        _validate_in_configs(normalized_model_id, "model id")

    ae_prefix = "model.taesd." if tiny else "model.vae."
    normalized_ae_id = f"{ae_prefix}{ae_id}".lower()
    _validate_in_configs(normalized_ae_id, "ae id")

    return normalized_model_id, subkey, normalized_ae_id


def merge_spec(base_spec: Any, subkey_spec: Any) -> Any:
    """Merge two dataclass specs with subkey values taking precedence over base values.

    Handles nested dataclasses recursively (e.g., init: InitialParams).
    Only merges if both specs are dataclasses with the same structure.

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
            # Handle nested dataclasses (e.g., init: InitialParams)
            if hasattr(subkey_value, "__dataclass_fields__") and base_value is not None and hasattr(base_value, "__dataclass_fields__"):
                # Merge nested dataclass: subkey fields override base fields
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
