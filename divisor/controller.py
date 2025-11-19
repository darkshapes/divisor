# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""
Interactive denoising with manual timestep control.
Allows users to manually increment through timesteps one at a time.
"""

from typing import Optional, Callable, Any
from dataclasses import dataclass
import torch


def time_shift(
    mu: float,
    sigma: float,
    tensor_step: torch.Tensor,
    steps: int,
    compress: float = 1.0,
) -> torch.Tensor:
    """
    Adjustable noise schedule.
    Compress or stretch any schedule to match a dynamic step sequence length.

    Args:
        mu: Original schedule parameter.
        sigma: Original schedule parameter.
        tensor_step: Tensor of original timesteps in [0,1].
        steps: Desired number of timesteps.
        compress: >1 compresses (fewer steps), <1 stretches (more steps).

    Returns:
        Adjusted timestep tensor.
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
    return torch.exp(mu) / (torch.exp(mu) + (1 / t_adj - 1) ** sigma)


@dataclass
class DenoisingState:
    """State of the denoising process at a given timestep."""

    current_timestep: float
    previous_timestep: Optional[float]
    current_sample: torch.Tensor
    timestep_index: int
    total_timesteps: int
    guidance: float


class ManualTimestepController:
    """
    Controller for manually stepping through denoising timesteps.

    Instead of automatically processing all timesteps, this allows
    the user to increment timesteps one at a time, with the ability
    to intervene between steps.
    """

    def __init__(
        self,
        timesteps: list[float],
        initial_sample: Any,
        denoise_step_fn: Callable[[Any, float, float, float], Any],
        mu: float = 0.0,
        sigma: float = 1.0,
        initial_guidance: float = 7.5,
    ):
        """
        Initialize the controller.

        Args:
            timesteps: List of timestep values to process (typically from 1.0 to 0.0)
            initial_sample: The initial noisy sample to start denoising from
            denoise_step_fn: Function that performs one denoising step.
                            Signature: (sample, t_curr, t_prev, guidance) -> new_sample
            mu: Schedule parameter for time_shift (default: 0.0)
            sigma: Schedule parameter for time_shift (default: 1.0)
            initial_guidance: Initial guidance (CFG) value (default: 7.5)
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

    @property
    def is_complete(self) -> bool:
        """Check if all timesteps have been processed."""
        return self.current_index >= len(self.timesteps) - 1

    @property
    def current_state(self) -> DenoisingState:
        """Get the current state of the denoising process."""
        t_curr = self.timesteps[self.current_index]
        t_prev = (
            self.timesteps[self.current_index + 1]
            if self.current_index + 1 < len(self.timesteps)
            else None
        )

        return DenoisingState(
            current_timestep=t_curr,
            previous_timestep=t_prev,
            current_sample=self.current_sample,
            timestep_index=self.current_index,
            total_timesteps=len(self.timesteps),
            guidance=self.guidance,
        )

    def step(self) -> DenoisingState:
        """
        Manually increment to the next timestep and perform one denoising step.
        Uses the current guidance value.

        Returns:
            The new state after the step.

        Raises:
            ValueError: If all timesteps have already been processed.
        """
        if self.is_complete:
            raise ValueError("All timesteps have been processed. Cannot step further.")

        t_curr = self.timesteps[self.current_index]
        t_prev = self.timesteps[self.current_index + 1]

        # Perform the denoising step with current guidance
        self.current_sample = self.denoise_step_fn(
            self.current_sample, t_curr, t_prev, self.guidance
        )

        # Move to next timestep
        self.current_index += 1

        # Save state and guidance
        state = self.current_state
        self.state_history.append(state)
        self.guidance_history.append(self.guidance)

        return state

    def reset(
        self,
        new_initial_sample: Optional[Any] = None,
        reset_guidance: Optional[float] = None,
    ):
        """
        Reset the controller to the beginning.

        Args:
            new_initial_sample: Optional new initial sample to use.
                               If None, keeps the original initial sample.
            reset_guidance: Optional new guidance value to use.
                          If None, keeps the current guidance value.
        """
        self.current_index = 0
        if new_initial_sample is not None:
            self.current_sample = new_initial_sample
        if reset_guidance is not None:
            self.guidance = reset_guidance
        self.state_history.clear()
        self.guidance_history = [self.guidance]

    def intervene(self, modified_sample: Any):
        """
        Allow user intervention by modifying the current sample.
        This replaces the current sample with a user-modified version.

        Args:
            modified_sample: The user-modified sample to use going forward.
        """
        self.current_sample = modified_sample

    def preview_final(self, preview_fn: Callable[[Any], Any]) -> Any:
        """
        Generate a preview of what the final result would look like
        from the current state, using a single-step x₀ prediction.

        Args:
            preview_fn: Function that generates a preview from current sample.
                       Signature: (sample) -> preview_sample

        Returns:
            Preview of the final result.
        """
        return preview_fn(self.current_sample)

    def stretch_compress_schedule(
        self,
        compress: float,
        steps: Optional[int] = None,
    ) -> list[float]:
        """
        Stretch or compress the remaining timesteps in the schedule using time_shift.

        Args:
            compress: >1 compresses (fewer steps), <1 stretches (more steps)
            steps: Desired number of timesteps for the remaining schedule.
                   If None, uses the current number of remaining timesteps.

        Returns:
            New list of timesteps for the remaining schedule.
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
        adjusted_timesteps = time_shift(
            self.mu, self.sigma, tensor_steps, steps, compress
        )

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
        """
        Stretch or compress the current step by subdividing it into multiple steps.
        This allows finer control over a single denoising step.

        Args:
            sub_steps: Number of sub-steps to divide the current step into
            compress: >1 compresses the sub-steps (closer spacing), <1 stretches them
                     (wider spacing). Affects the distribution of sub-steps.

        Returns:
            List of new timesteps that replace the current step.
        """
        if self.is_complete:
            raise ValueError("No current step to subdivide.")

        t_curr = self.timesteps[self.current_index]
        t_prev = (
            self.timesteps[self.current_index + 1]
            if self.current_index + 1 < len(self.timesteps)
            else 0.0
        )

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
            adjusted_normalized = time_shift(
                self.mu, self.sigma, normalized, sub_steps + 1, compress
            )

            # Denormalize back to original range [t_prev, t_curr]
            sub_timesteps = adjusted_normalized * (t_curr - t_prev) + t_prev

            # Ensure we maintain the endpoints
            sub_timesteps[0] = t_curr
            sub_timesteps[-1] = t_prev

        # Convert to list (exclude the first one as it's the current timestep)
        new_sub_timesteps = sub_timesteps[1:].tolist()

        # Replace current step with sub-steps
        self.timesteps = (
            self.timesteps[: self.current_index + 1]
            + new_sub_timesteps
            + self.timesteps[self.current_index + 1 :]
        )

        return new_sub_timesteps

    def apply_time_shift_to_remaining(
        self,
        compress: float = 1.0,
        steps: Optional[int] = None,
    ):
        """
        Apply time_shift to the remaining schedule and update it.
        This is a convenience method that calls stretch_compress_schedule.

        Args:
            compress: >1 compresses (fewer steps), <1 stretches (more steps)
            steps: Desired number of timesteps for the remaining schedule.
                   If None, uses the current number of remaining timesteps.
        """
        return self.stretch_compress_schedule(compress, steps)

    def set_guidance(self, guidance: float):
        """
        Set the guidance value for the next denoising step.

        Args:
            guidance: New guidance (CFG) value to use. Typically ranges from 1.0 to 20.0.
                     Higher values provide stronger adherence to the conditioning.
        """
        self.guidance = guidance

    def adjust_guidance(self, delta: float):
        """
        Adjust the guidance value by a delta amount.

        Args:
            delta: Amount to add to the current guidance value (can be negative).
        """
        self.guidance = max(0.0, self.guidance + delta)
