# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""
Interactive denoising with manual timestep control.
Allows users to manually increment through timesteps one at a time.
"""

import json
from dataclasses import asdict
from typing import Any, Callable, Optional

import torch
from nnll.console import nfo
from nnll.hyperchain import HyperChain
from nnll.random import RNGState

from divisor.interaction_context import InteractionContext
from divisor.registry import gfx_device
from divisor.state import MenuState, StepState

rng = RNGState(device=gfx_device.type)
variation_rng = RNGState(device=gfx_device.type)


def time_shift(
    schedule_mu: float,
    schedule_sigma: float,
    original_timestep_tensor: torch.Tensor,
    desired_step_count: int,
    compression_factor: float = 1.0,
) -> torch.Tensor:
    """Adjustable noise schedule. Compress or stretch any schedule to match a dynamic step sequence length.\n
    :param schedule_mu: Original schedule parameter.
    :param schedule_sigma: Original schedule parameter.
    :param original_timestep_tensor: Tensor of original timesteps in [0,1].
    :param desired_step_count: Desired number of timesteps.
    :param compression_factor: >1 compresses (fewer steps), <1 stretches (more steps).
    :returns: Adjusted timestep tensor."""

    if desired_step_count == 1:
        return original_timestep_tensor

    new_index = (original_timestep_tensor * (desired_step_count - 1) * compression_factor).clamp(0, desired_step_count - 1)
    scaled_timestep = new_index / (desired_step_count - 1)
    scaled_timestep = torch.clamp(scaled_timestep, min=1e-8, max=1.0 - 1e-8)
    exponential_mu = torch.exp(torch.tensor(schedule_mu))
    return exponential_mu / (exponential_mu + (1 / scaled_timestep - 1) ** torch.tensor(schedule_sigma))


def serialize_state_for_chain(state: "MenuState", current_seed: int) -> str:
    """Serialize MenuState for HyperChain storage, excluding current_sample and adding current_seed.\n
    :param state: The MenuState to serialize
    :param current_seed: The current seed value to include instead of current_sample
    :returns: JSON string representation of the state"""

    state_dict = asdict(state)
    if "timestep" in state_dict and isinstance(state_dict["timestep"], dict):
        state_dict["timestep"].pop("current_sample", None)
    if "timestep" in state_dict and isinstance(state_dict["timestep"], dict):
        timestep_dict = state_dict.pop("timestep")
        state_dict.update(timestep_dict)
    state_dict["current_seed"] = current_seed
    return json.dumps(state_dict, default=str)


def reconstruct_state_from_dict(state_dict: dict, current_sample: torch.Tensor) -> "MenuState":
    """Reconstruct MenuState from dictionary and current sample tensor.

    :param state_dict: Dictionary containing state fields
    :param current_sample: The current sample tensor to include in the state
    :returns: Reconstructed MenuState object"""
    step_state = StepState(
        current_timestep=state_dict["current_timestep"],
        previous_timestep=state_dict["previous_timestep"],
        current_sample=current_sample,
        timestep_index=state_dict["timestep_index"],
        total_timesteps=state_dict["total_timesteps"],
    )
    return MenuState(
        step_state=step_state,
        guidance=state_dict["guidance"],
        layer_dropout=state_dict["layer_dropout"],
        width=state_dict["width"],
        height=state_dict["height"],
        seed=state_dict["seed"],
        prompt=state_dict["prompt"],
        num_steps=state_dict["num_steps"],
        vae_shift_offset=state_dict["vae_shift_offset"],
        vae_scale_offset=state_dict["vae_scale_offset"],
        use_previous_as_mask=state_dict["use_previous_as_mask"],
        variation_seed=state_dict["variation_seed"],
        variation_strength=state_dict["variation_strength"],
        deterministic=state_dict["deterministic"],
    )


class ManualTimestepController:
    """
    Controller for manually stepping through denoising timesteps.\n
    Instead of automatically processing all timesteps, this allows
    the user to increment timesteps one at a time, with the ability
    to intervene between steps."""

    hyperchain: HyperChain = HyperChain()

    def __init__(
        self,
        timesteps: list[float],
        initial_sample: Any,
        denoise_step_fn: Callable[[Any, float, float, float], Any],
        schedule_mu: float = 0.0,
        schedule_sigma: float = 1.0,
        initial_guidance: float = 7.5,
    ) -> None:
        """Manipulate denoising process.\n
        :param timesteps: List of timestep values to process (typically from 1.0 to 0.0)
        :param initial_sample: The initial noisy sample to start denoising from
        :param denoise_step_fn: Function that performs one denoising step. Signature: (sample, t_curr, t_prev, guidance) -> new_sample
        :param schedule_mu: Schedule parameter for time_shift (default: 0.0)
        :param schedule_sigma: Schedule parameter for time_shift (default: 1.0)
        :param initial_guidance: Initial guidance (CFG) value (default: 7.5)
        :param hyperchain: Optional HyperChain instance for storing state history (default: None)"""
        self.timesteps = timesteps
        self.original_timesteps = timesteps.copy()
        self.denoise_step_fn = denoise_step_fn
        self.current_index = 0
        self.current_sample = initial_sample
        self.state_history: list[MenuState] = []
        self.schedule_mu = schedule_mu
        self.schedule_sigma = schedule_sigma
        self.guidance = initial_guidance
        self.guidance_history: list[float] = [initial_guidance]
        self.layer_dropout: Optional[list[int]] = None
        self.layer_dropout_history: list[Optional[list[int]]] = [None]
        self.width: Optional[int] = None
        self.height: Optional[int] = None
        self.seed: Optional[int] = None
        self.prompt: Optional[str] = None
        self.num_steps: Optional[int] = None
        self.vae_shift_offset: float = 0.0
        self.vae_scale_offset: float = 0.0
        self.use_previous_as_mask: bool = False
        self.variation_seed: Optional[int] = None
        self.variation_strength: float = 0.0
        self.deterministic: bool = False
        self.rewind_steps: int = 0

    @property
    def is_complete(self) -> bool:
        """Check if all timesteps have been processed."""
        return self.current_index >= len(self.timesteps) - 1

    @property
    def current_state(self) -> MenuState:
        """Get the current state of the denoising process."""
        t_curr = self.timesteps[self.current_index]
        t_prev = self.timesteps[self.current_index + 1] if self.current_index + 1 < len(self.timesteps) else None

        step_state = StepState(
            current_timestep=t_curr,
            previous_timestep=t_prev,
            current_sample=self.current_sample,
            timestep_index=self.current_index,
            total_timesteps=len(self.timesteps),
        )

        return MenuState(
            step_state=step_state,
            guidance=self.guidance,
            layer_dropout=self.layer_dropout,
            width=self.width,
            height=self.height,
            seed=self.seed,
            prompt=self.prompt,
            num_steps=self.num_steps,
            vae_shift_offset=self.vae_shift_offset,
            vae_scale_offset=self.vae_scale_offset,
            use_previous_as_mask=self.use_previous_as_mask,
            variation_seed=self.variation_seed,
            variation_strength=self.variation_strength,
            deterministic=self.deterministic,
        )

    def step(self) -> MenuState:
        """Manually increment to the next timestep and perform one denoising step. Uses the current guidance value.\n
        :returns: The new state after the step.
        :raises ValueError: If all timesteps have already been processed."""
        if self.is_complete:
            raise ValueError("All timesteps have been processed. Cannot step further.")

        t_curr = self.timesteps[self.current_index]
        t_prev = self.timesteps[self.current_index + 1]

        self.current_sample = self.denoise_step_fn(self.current_sample, t_curr, t_prev, self.guidance)
        self.current_index += 1

        state = self.current_state
        self.state_history.append(state)
        self.guidance_history.append(self.guidance)
        self.layer_dropout_history.append(self.layer_dropout)

        return state

    def rewind(self, num_steps: int = 1) -> None:
        """Move the controller back `n` timesteps (if possible).\n
        :param n: Number of timesteps to rewind"""

        if num_steps < 0:
            nfo("Rewind count must be non‑negative")
            return

        self.current_index = max(0, self.current_index - num_steps)
        self.rewind_steps += num_steps

    def set_guidance(self, guidance: float) -> None:
        """Set the guidance value for the next denoising step.\n
        :param guidance: New guidance (CFG) value to use. Typically ranges from 1.0 to 20.0. Higher values provide stronger adherence to the conditioning."""
        self.guidance = guidance

    def set_layer_dropout(self, layer_dropout: Optional[list[int]]):
        """Set the layer dropout configuration for the next denoising step.\n
        :param layer_dropout: List of block indices to skip during inference, or None to skip none. Blocks are indexed starting from 0."""
        self.layer_dropout = layer_dropout

    def set_resolution(self, width: int, height: int) -> None:
        """Set the width and height resolution for the denoising process.\n
        :param width: Width in pixels
        :param height: Height in pixels"""
        self.width = width
        self.height = height

    def set_seed(self, seed: int) -> None:
        """Set the seed value for the denoising process.\n
        :param seed: Seed value for random number generation"""
        self.seed = seed

    def set_prompt(self, prompt: str) -> None:
        """Set the prompt text for the denoising process.\n
        :param prompt: Prompt text"""
        self.prompt = prompt

    def set_num_steps(self, num_steps: int) -> None:
        """Set the number of steps for the denoising process.\n
        :param num_steps: Number of denoising steps"""
        self.num_steps = num_steps

    def set_vae_shift_offset(self, offset: float) -> None:
        """Set the VAE shift offset for autoencoder decode.\n
        :param offset: Offset to add to shift_factor in autoencoder decode"""
        self.vae_shift_offset = offset

    def set_vae_scale_offset(self, offset: float) -> None:
        """Set the VAE scale offset for autoencoder decode.\n
        :param offset: Offset to add to scale_factor in autoencoder decode"""
        self.vae_scale_offset = offset

    def set_use_previous_as_mask(self, use_mask: bool) -> None:
        """Set whether to use previous step tensor as mask.\n
        :param use_mask: Whether to use previous step tensor as mask for next step"""
        self.use_previous_as_mask = use_mask

    def set_variation_seed(self, seed: int | None = None) -> None:
        """Set the variation seed for adding variation noise.\n
        :param seed: Variation seed value, or None to disable"""
        self.variation_seed = seed

    def set_variation_strength(self, strength: float) -> None:
        """Set the variation strength (0.0 to 1.0).\n
        :param strength: Variation strength, where 0.0 = no variation, 1.0 = full variation"""
        self.variation_strength = max(0.0, min(1.0, strength))

    def set_deterministic(self, deterministic: bool) -> None:
        """Set deterministic mode for PyTorch operations.\n
        :param deterministic: Whether to use deterministic algorithms (False = non-deterministic, True = deterministic)"""

        self.deterministic = deterministic

    def store_state_in_chain(self, current_seed: int | None = None, serialized_state_int: int | None = None) -> Optional[Any]:
        """Store the current MenuState in HyperChain, excluding current_sample and adding current_seed.

        :param current_seed: The current seed value to include instead of current_sample. Required if serialized_state_int is None.
        :param serialized_state_int: Optional pre-serialized state as integer. If provided, current_seed is ignored.
        :returns: The created Block if hyperchain is configured, None otherwise
        """

        if serialized_state_int is not None:
            json_bytes = serialized_state_int.to_bytes((serialized_state_int.bit_length() + 7) // 8, byteorder="big", signed=False)
            serialized = json_bytes.decode("utf-8")
        else:
            if current_seed is None:
                raise ValueError("Either current_seed or serialized_state_int must be provided")
            state = self.current_state
            serialized = serialize_state_for_chain(state, current_seed)

        return self.hyperchain.add_block(serialized)


def update_state_and_cache(
    controller: ManualTimestepController,
    setter_func: Callable,
    value: Any,
    interaction_context: InteractionContext,
    success_message: str,
) -> MenuState:
    """Generic state update helper that sets value, clears cache, and refreshes state.\n
    :param controller: ManualTimestepController instance
    :param setter_func: Controller setter method to call
    :param value: Value to set
    :param clear_prediction_cache: Function to clear prediction cache
    :param success_message: Message to display on success
    :returns: Updated MenuState"""
    setter_func(value)
    interaction_context.clear_prediction_cache()
    state = controller.current_state
    nfo(success_message)
    return state
