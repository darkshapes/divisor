# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""
Interactive denoising with manual timestep control.
Allows users to manually increment through timesteps one at a time.
"""

from typing import Optional, Callable, Any
from dataclasses import dataclass, asdict
import json
import torch
from nnll.hyperchain import HyperChain
from nnll.random import RNGState

rng = RNGState(device="cpu")


def time_shift(
    mu: float,
    sigma: float,
    tensor_step: torch.Tensor,
    steps: int,
    compress: float = 1.0,
) -> torch.Tensor:
    """Adjustable noise schedule. Compress or stretch any schedule to match a dynamic step sequence length.

    :param mu: Original schedule parameter.
    :param sigma: Original schedule parameter.
    :param tensor_step: Tensor of original timesteps in [0,1].
    :param steps: Desired number of timesteps.
    :param compress: >1 compresses (fewer steps), <1 stretches (more steps).
    :returns: Adjusted timestep tensor.
    """
    # Handle edge case where steps is 1
    if steps == 1:
        return tensor_step

    # map original tensor_step to new index space
    new_idx = (tensor_step * (steps - 1) * compress).clamp(0, steps - 1)
    # rescale back to [0,1]
    t_adj = new_idx / (steps - 1)
    # Avoid division by zero
    t_adj = torch.clamp(t_adj, min=1e-8, max=1.0 - 1e-8)
    return torch.exp(torch.tensor(mu)) / (torch.exp(torch.tensor(mu)) + (1 / t_adj - 1) ** torch.tensor(sigma))


def serialize_state_for_chain(state: "DenoisingState", current_seed: int) -> str:
    """Serialize DenoisingState for HyperChain storage, excluding current_sample and adding current_seed.

    :param state: The DenoisingState to serialize
    :param current_seed: The current seed value to include instead of current_sample
    :returns: JSON string representation of the state
    """
    state_dict = asdict(state)
    # Remove the tensor (current_sample) as it's not serializable
    state_dict.pop("current_sample", None)
    # Add the seed instead
    state_dict["current_seed"] = current_seed
    return json.dumps(state_dict, default=str)


def reconstruct_state_from_dict(state_dict: dict, current_sample: torch.Tensor) -> "DenoisingState":
    """Reconstruct DenoisingState from dictionary and current sample tensor.

    :param state_dict: Dictionary containing state fields
    :param current_sample: The current sample tensor to include in the state
    :returns: Reconstructed DenoisingState object
    """
    return DenoisingState(
        current_timestep=state_dict["current_timestep"],
        previous_timestep=state_dict.get("previous_timestep"),
        current_sample=current_sample,
        timestep_index=state_dict["timestep_index"],
        total_timesteps=state_dict["total_timesteps"],
        guidance=state_dict["guidance"],
        layer_dropout=state_dict.get("layer_dropout"),
        width=state_dict.get("width"),
        height=state_dict.get("height"),
        seed=state_dict.get("seed"),
        prompt=state_dict.get("prompt"),
        num_steps=state_dict.get("num_steps"),
        vae_shift_offset=state_dict.get("vae_shift_offset", 0.0),
        vae_scale_offset=state_dict.get("vae_scale_offset", 0.0),
        use_previous_as_mask=state_dict.get("use_previous_as_mask", False),
    )


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


class ManualTimestepController:
    """
    Controller for manually stepping through denoising timesteps.

    Instead of automatically processing all timesteps, this allows
    the user to increment timesteps one at a time, with the ability
    to intervene between steps.
    """

    hyperchain: HyperChain = HyperChain()

    def __init__(
        self,
        timesteps: list[float],
        initial_sample: Any,
        denoise_step_fn: Callable[[Any, float, float, float], Any],
        mu: float = 0.0,
        sigma: float = 1.0,
        initial_guidance: float = 7.5,
    ):
        """Initialize the controller.

        :param timesteps: List of timestep values to process (typically from 1.0 to 0.0)
        :param initial_sample: The initial noisy sample to start denoising from
        :param denoise_step_fn: Function that performs one denoising step. Signature: (sample, t_curr, t_prev, guidance) -> new_sample
        :param mu: Schedule parameter for time_shift (default: 0.0)
        :param sigma: Schedule parameter for time_shift (default: 1.0)
        :param initial_guidance: Initial guidance (CFG) value (default: 7.5)
        :param hyperchain: Optional HyperChain instance for storing state history (default: None)
        """
        self.timesteps = timesteps
        self.original_timesteps = timesteps.copy()
        self.denoise_step_fn = denoise_step_fn
        self.current_index = 0
        self.current_sample = initial_sample
        self.state_history: list[DenoisingState] = []
        self.mu = mu
        self.sigma = sigma
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
        if self.hyperchain is not None and len(self.hyperchain.chain) == 0:
            self.hyperchain.synthesize_genesis_block()

    @property
    def is_complete(self) -> bool:
        """Check if all timesteps have been processed."""
        return self.current_index >= len(self.timesteps) - 1

    @property
    def current_state(self) -> DenoisingState:
        """Get the current state of the denoising process."""
        t_curr = self.timesteps[self.current_index]
        t_prev = self.timesteps[self.current_index + 1] if self.current_index + 1 < len(self.timesteps) else None

        return DenoisingState(
            current_timestep=t_curr,
            previous_timestep=t_prev,
            current_sample=self.current_sample,
            timestep_index=self.current_index,
            total_timesteps=len(self.timesteps),
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
        )

    def step(self) -> DenoisingState:
        """Manually increment to the next timestep and perform one denoising step. Uses the current guidance value.

        :returns: The new state after the step.
        :raises ValueError: If all timesteps have already been processed.
        """
        if self.is_complete:
            raise ValueError("All timesteps have been processed. Cannot step further.")

        t_curr = self.timesteps[self.current_index]
        t_prev = self.timesteps[self.current_index + 1]

        # Perform the denoising step with current guidance
        self.current_sample = self.denoise_step_fn(self.current_sample, t_curr, t_prev, self.guidance)

        # Move to next timestep
        self.current_index += 1

        # Save state and guidance
        state = self.current_state
        self.state_history.append(state)
        self.guidance_history.append(self.guidance)
        self.layer_dropout_history.append(self.layer_dropout)

        return state

    def reset(
        self,
        new_initial_sample: Optional[Any] = None,
        reset_guidance: Optional[float] = None,
    ):
        """Reset the controller to the beginning.

        :param new_initial_sample: Optional new initial sample to use. If None, keeps the original initial sample.
        :param reset_guidance: Optional new guidance value to use. If None, keeps the current guidance value.
        """
        self.current_index = 0
        if new_initial_sample is not None:
            self.current_sample = new_initial_sample
        if reset_guidance is not None:
            self.guidance = reset_guidance
        self.state_history.clear()
        self.guidance_history = [self.guidance]
        self.layer_dropout_history = [self.layer_dropout]

    def intervene(self, modified_sample: Any):
        """Allow user intervention by modifying the current sample. This replaces the current sample with a user-modified version.

        :param modified_sample: The user-modified sample to use going forward.
        """
        self.current_sample = modified_sample

    def preview_final(self, preview_fn: Callable[[Any], Any]) -> Any:
        """Generate a preview of what the final result would look like from the current state, using a single-step x₀ prediction.

        :param preview_fn: Function that generates a preview from current sample. Signature: (sample) -> preview_sample
        :returns: Preview of the final result.
        """
        return preview_fn(self.current_sample)

    def stretch_compress_schedule(
        self,
        compress: float,
        steps: Optional[int] = None,
    ) -> list[float]:
        """Stretch or compress the remaining timesteps in the schedule using time_shift.

        :param compress: >1 compresses (fewer steps), <1 stretches (more steps)
        :param steps: Desired number of timesteps for the remaining schedule. If None, uses the current number of remaining timesteps.
        :returns: New list of timesteps for the remaining schedule.
        """
        if self.is_complete:
            return []

        # Get remaining timesteps
        remaining_timesteps = self.timesteps[self.current_index :]

        if steps is None:
            steps = len(remaining_timesteps)

        # Convert to tensor and normalize to [0, 1] if needed
        # Assuming timesteps are already in [0, 1] range
        tensor_steps = torch.tensor(remaining_timesteps, dtype=torch.float32)

        # Apply time_shift
        adjusted_timesteps = time_shift(self.mu, self.sigma, tensor_steps, steps, compress)

        # Convert back to list
        new_timesteps = adjusted_timesteps.tolist()

        # Update the schedule (keep processed timesteps, replace remaining)
        self.timesteps = self.timesteps[: self.current_index] + new_timesteps

        return new_timesteps

    def stretch_compress_current_step(
        self,
        sub_steps: int,
        compress: float = 1.0,
    ) -> list[float]:
        """Stretch or compress the current step by subdividing it into multiple steps. This allows finer control over a single denoising step.

        :param sub_steps: Number of sub-steps to divide the current step into
        :param compress: >1 compresses the sub-steps (closer spacing), <1 stretches them (wider spacing). Affects the distribution of sub-steps.
        :returns: List of new timesteps that replace the current step.
        """
        if self.is_complete:
            raise ValueError("No current step to subdivide.")

        t_curr = self.timesteps[self.current_index]
        t_prev = self.timesteps[self.current_index + 1] if self.current_index + 1 < len(self.timesteps) else 0.0

        if t_curr == t_prev:
            # No step to subdivide
            return []

        # Create evenly spaced sub-timesteps between t_curr and t_prev
        sub_timesteps = torch.linspace(t_curr, t_prev, sub_steps + 1)

        # Apply compression/stretching using time_shift if needed
        if compress != 1.0:
            # Normalize to [0, 1] range where 1.0 = t_curr and 0.0 = t_prev
            # This maps the interval [t_prev, t_curr] to [0, 1]
            normalized = (sub_timesteps - t_prev) / (t_curr - t_prev)

            # Apply time_shift to adjust the spacing
            adjusted_normalized = time_shift(self.mu, self.sigma, normalized, sub_steps + 1, compress)

            # Denormalize back to original range [t_prev, t_curr]
            sub_timesteps = adjusted_normalized * (t_curr - t_prev) + t_prev

            # Ensure we maintain the endpoints
            sub_timesteps[0] = t_curr
            sub_timesteps[-1] = t_prev

        # Convert to list (exclude the first one as it's the current timestep)
        new_sub_timesteps = sub_timesteps[1:].tolist()

        # Replace current step with sub-steps
        self.timesteps = self.timesteps[: self.current_index + 1] + new_sub_timesteps + self.timesteps[self.current_index + 1 :]

        return new_sub_timesteps

    def apply_time_shift_to_remaining(
        self,
        compress: float = 1.0,
        steps: Optional[int] = None,
    ):
        """Apply time_shift to the remaining schedule and update it. This is a convenience method that calls stretch_compress_schedule.

        :param compress: >1 compresses (fewer steps), <1 stretches (more steps)
        :param steps: Desired number of timesteps for the remaining schedule. If None, uses the current number of remaining timesteps.
        """
        return self.stretch_compress_schedule(compress, steps)

    def set_guidance(self, guidance: float):
        """Set the guidance value for the next denoising step.

        :param guidance: New guidance (CFG) value to use. Typically ranges from 1.0 to 20.0. Higher values provide stronger adherence to the conditioning.
        """
        self.guidance = guidance

    def adjust_guidance(self, delta: float):
        """Adjust the guidance value by a delta amount.

        :param delta: Amount to add to the current guidance value (can be negative).
        """
        self.guidance = max(0.0, self.guidance + delta)

    def set_layer_dropout(self, layer_dropout: Optional[list[int]]):
        """Set the layer dropout configuration for the next denoising step.

        :param layer_dropout: List of block indices to skip during inference, or None to skip none. Blocks are indexed starting from 0.
        """
        self.layer_dropout = layer_dropout

    def set_resolution(self, width: int, height: int):
        """Set the width and height resolution for the denoising process.

        :param width: Width in pixels
        :param height: Height in pixels
        """
        self.width = width
        self.height = height

    def set_seed(self, seed: int):
        """Set the seed value for the denoising process.

        :param seed: Seed value for random number generation
        """
        self.seed = seed

    def set_prompt(self, prompt: str):
        """Set the prompt text for the denoising process.

        :param prompt: Prompt text
        """
        self.prompt = prompt

    def set_num_steps(self, num_steps: int):
        """Set the number of steps for the denoising process.

        :param num_steps: Number of denoising steps
        """
        self.num_steps = num_steps

    def set_vae_shift_offset(self, offset: float):
        """Set the VAE shift offset for autoencoder decode.

        :param offset: Offset to add to shift_factor in autoencoder decode
        """
        self.vae_shift_offset = offset

    def set_vae_scale_offset(self, offset: float):
        """Set the VAE scale offset for autoencoder decode.

        :param offset: Offset to add to scale_factor in autoencoder decode
        """
        self.vae_scale_offset = offset

    def set_use_previous_as_mask(self, use_mask: bool):
        """Set whether to use previous step tensor as mask.

        :param use_mask: Whether to use previous step tensor as mask for next step
        """
        self.use_previous_as_mask = use_mask

    def store_state_in_chain(self, current_seed: int | None = None, serialized_state_int: int | None = None) -> Optional[Any]:
        """Store the current DenoisingState in HyperChain, excluding current_sample and adding current_seed.

        :param current_seed: The current seed value to include instead of current_sample. Required if serialized_state_int is None.
        :param serialized_state_int: Optional pre-serialized state as integer. If provided, current_seed is ignored.
        :returns: The created Block if hyperchain is configured, None otherwise
        """
        if self.hyperchain is None:
            return None

        if serialized_state_int is not None:
            # Deserialize the int back to JSON string for HyperChain
            json_bytes = serialized_state_int.to_bytes((serialized_state_int.bit_length() + 7) // 8, byteorder="big", signed=False)
            serialized = json_bytes.decode("utf-8")
        else:
            if current_seed is None:
                raise ValueError("Either current_seed or serialized_state_int must be provided")
            state = self.current_state
            serialized = serialize_state_for_chain(state, current_seed)

        return self.hyperchain.add_block(serialized)
