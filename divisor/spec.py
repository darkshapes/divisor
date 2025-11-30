# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Divisor class definitions and configuration dataclasses."""

from dataclasses import dataclass
from typing import Any, Optional

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
    current_neg_txt: Optional[Tensor] | None = None
    current_neg_txt_ids: Optional[Tensor] | None = None
    current_neg_vec: Optional[Tensor] | None = None
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
