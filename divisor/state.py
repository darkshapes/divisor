# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Divisor class definitions and configuration dataclasses."""

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional

from nnll.random import RNGState
import torch
from torch import Tensor
from torch import nn


@dataclass
class TimestepState:
    """Runtime state that changes at each denoising step."""

    current_timestep: float
    current_sample: torch.Tensor
    timestep_index: int
    total_timesteps: int
    previous_timestep: float | None = None

    def with_runtime_state(
        self,
        current_timestep: float | None = None,
        current_sample: torch.Tensor | None = None,
        timestep_index: int | None = None,
        total_timesteps: int | None = None,
        previous_timestep: float | None = None,
    ) -> "TimestepState":
        """Create a new TimestepState with updated runtime fields.

        :param current_timestep: Current timestep value (if None, keeps existing)
        :param current_sample: Current sample tensor (if None, keeps existing)
        :param timestep_index: Current timestep index (if None, keeps existing)
        :param total_timesteps: Total number of timesteps (if None, keeps existing)
        :param previous_timestep: Previous timestep value (if None, keeps existing)
        :returns: New TimestepState with updated fields
        """
        return replace(
            self,
            current_timestep=current_timestep if current_timestep is not None else self.current_timestep,
            current_sample=current_sample if current_sample is not None else self.current_sample,
            timestep_index=timestep_index if timestep_index is not None else self.timestep_index,
            total_timesteps=total_timesteps if total_timesteps is not None else self.total_timesteps,
            previous_timestep=previous_timestep if previous_timestep is not None else self.previous_timestep,
        )


@dataclass
class DenoisingState:
    """State of the denoising process at a given timestep.

    This is the single source of truth for denoising configuration and runtime state.
    Use from_cli_args() to create from command-line arguments, and with_runtime_state()
    to update runtime fields during denoising.
    """

    # Runtime state (changes every step)
    timestep_state: TimestepState

    # Configuration (set at start, may change via controller)
    guidance: float
    layer_dropout: List[int] | None = None
    width: int = 1024
    height: int = 1024
    seed: int = 0
    prompt: str | None = None
    num_steps: int | None = None
    neg_prompt: str | None = None
    vae_shift_offset: float = 0.0
    vae_scale_offset: float = 0.0
    use_previous_as_mask: bool = False
    variation_seed: int = 0
    variation_strength: float = 0.0
    deterministic: bool = False

    # Convenience properties for backward compatibility
    @property
    def current_timestep(self) -> float:
        """Current timestep value."""
        return self.timestep_state.current_timestep

    @property
    def current_sample(self) -> torch.Tensor:
        """Current sample tensor."""
        return self.timestep_state.current_sample

    @property
    def timestep_index(self) -> int:
        """Current timestep index."""
        return self.timestep_state.timestep_index

    @property
    def total_timesteps(self) -> int:
        """Total number of timesteps."""
        return self.timestep_state.total_timesteps

    @property
    def previous_timestep(self) -> float | None:
        """Previous timestep value."""
        return self.timestep_state.previous_timestep

    @classmethod
    def from_cli_args(
        cls,
        prompt: str,
        width: int,
        height: int,
        num_steps: int,
        guidance: float,
        seed: int = 0,
        neg_prompt: str | None = None,
        current_sample: torch.Tensor | None = None,
        timesteps: List[float] | None = None,
        **kwargs,
    ) -> "DenoisingState":
        """Create DenoisingState from CLI arguments.

        :param prompt: Text prompt
        :param width: Image width
        :param height: Image height
        :param num_steps: Number of denoising steps
        :param guidance: Guidance scale
        :param seed: Random seed
        :param neg_prompt: Negative prompt
        :param current_sample: Initial sample tensor (if available, otherwise empty tensor)
        :param timesteps: List of timesteps (if available, used to determine total_timesteps)
        :param kwargs: Additional fields to set (e.g., deterministic, vae_shift_offset, etc.)
        :returns: DenoisingState initialized from CLI args
        """
        total_timesteps = len(timesteps) if timesteps else num_steps

        timestep_state = TimestepState(
            current_timestep=0.0,
            previous_timestep=None,
            current_sample=current_sample if current_sample is not None else torch.empty(0),
            timestep_index=0,
            total_timesteps=total_timesteps,
        )

        return cls(
            timestep_state=timestep_state,
            guidance=guidance,
            width=width,
            height=height,
            seed=seed,
            prompt=prompt,
            num_steps=num_steps,
            neg_prompt=neg_prompt,
            deterministic=bool(torch.get_deterministic_debug_mode()) if "deterministic" not in kwargs else kwargs.pop("deterministic"),
            **kwargs,
        )

    def with_runtime_state(
        self,
        current_timestep: float,
        current_sample: torch.Tensor,
        timestep_index: int,
        total_timesteps: int | None = None,
        previous_timestep: float | None = None,
    ) -> "DenoisingState":
        """Create a new DenoisingState with updated runtime state.

        Delegates to TimestepState.with_runtime_state() and replaces the timestep.

        :param current_timestep: Current timestep value
        :param current_sample: Current sample tensor
        :param timestep_index: Current timestep index
        :param total_timesteps: Total number of timesteps (if changed)
        :param previous_timestep: Previous timestep value
        :returns: New DenoisingState with updated runtime fields
        """
        new_timestep = self.timestep_state.with_runtime_state(
            current_timestep=current_timestep,
            current_sample=current_sample,
            timestep_index=timestep_index,
            total_timesteps=total_timesteps,
            previous_timestep=previous_timestep,
        )
        return replace(self, timestep_state=new_timestep)


@dataclass
class GetImagePredictionSettings:
    """Image-related configuration for get_prediction function creation."""

    img_ids: Tensor
    img: Tensor
    img_cond: Tensor | None = None
    img_cond_seq: Tensor | None = None
    img_cond_seq_ids: Tensor | None = None
    image_proj: Tensor | None = None
    neg_image_proj: Tensor | None = None
    ip_scale: Tensor | None = None
    neg_ip_scale: Tensor | None = None


@dataclass
class GetPredictionSettings:
    """Base configuration class for get_prediction function creation."""

    model_ref: List[Any]
    state: Any
    current_txt: List[Tensor]
    current_txt_ids: List[Tensor]
    current_vec: List[Tensor]
    cached_prediction: List[Optional[Tensor]] = field(default_factory=lambda: [None])
    cached_prediction_state: List[Optional[Dict]] = field(default_factory=lambda: [None])
    neg_pred_enabled: bool = False
    current_neg_txt: Tensor | str | None = ""
    current_neg_txt_ids: Tensor | str | None = ""
    current_neg_vec: Tensor | str | None = ""
    true_gs: float | None = None


@dataclass
class AdditionalPredictionSettings:
    """Additional configuration for XFlux1-specific prediction settings."""

    timestep_to_start_cfg: int
    current_timestep_index: List[int]


@dataclass
class DenoiseSettings:
    """Base configuration class for denoise function parameters."""

    # Model and core inputs (required)
    img: Tensor
    img_ids: Tensor
    txt: Tensor
    txt_ids: Tensor
    state: Any  # DenoisingState
    ae: Any  # AutoEncoder
    timesteps: List[float]
    vec: Tensor | None = None  # CLIP embeddings (Flux1/XFlux1)
    neg_pred_enabled: bool = False
    neg_txt: Tensor | None = None
    neg_txt_ids: Tensor | None = None
    neg_vec: Tensor | None = None
    true_gs: float = 1.0

    # Text embedders for prompt changes
    t5: Any | None = None  # T5 embedder (Flux1/XFlux1)
    clip: Any | None = None  # CLIP embedder (Flux1/XFlux1)
    text_embedder: Any | None = None  # Mistral embedder (Flux2)

    img_cond: Tensor | None = None  # Channel-wise image conditioning (Flux1 only)
    img_cond_seq: Tensor | None = None  # Sequence-wise image conditioning
    img_cond_seq_ids: Tensor | None = None
    device: torch.device | None = None
    initial_layer_dropout: List[int] | None = None
    timestep_to_start_cfg: int = 0
    image_proj: Tensor | None = None
    neg_image_proj: Tensor | None = None
    ip_scale: Tensor | None = None
    neg_ip_scale: Tensor | None = None


@dataclass
class DenoiseSettingsFlux2:
    """Configuration for simple (non-interactive) Flux2 denoising."""

    model: Any  # Flux2
    img: Tensor
    img_ids: Tensor
    txt: Tensor
    txt_ids: Tensor
    timesteps: List[float]
    guidance: float = 4.0
    img_cond_seq: Tensor | None = None
    img_cond_seq_ids: Tensor | None = None


@dataclass
class RouteProcesses:
    """Functions to call for each choice."""

    clear_prediction_cache: Callable[[], None]
    rng: RNGState | None = None
    variation_rng: RNGState | None = None
    ae: Optional[nn.Module] | None = None
    t5: Optional[nn.Module] | None = None
    clip: Optional[nn.Module] | None = None
    recompute_text_embeddings: Optional[Callable[[str], None]] | None = None
