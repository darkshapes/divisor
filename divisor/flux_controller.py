# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""
Flux-specific controller for interactive denoising with manual timestep control.
Mirrors the controller.py structure but is specifically designed for Flux models.
"""

from typing import Optional
from dataclasses import dataclass
import torch
from torch import Tensor

from divisor.controller import DenoisingState
from divisor.flux_modules.model import Flux


@dataclass
class FluxDenoisingState(DenoisingState):
    """Flux-specific extended state for denoising process."""


class FluxController:
    """
    Controller for manually stepping through Flux denoising timesteps.

    Mirrors the ManualTimestepController structure but is specifically
    designed to work with Flux models and their denoise function parameters.
    """

    def __init__(
        self,
        model: Flux,
        timesteps: list[float],
        initial_img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        vec: Tensor,
        mu: float = 0.0,
        sigma: float = 1.0,
        initial_guidance: float = 4.0,
        img_cond: Optional[Tensor] = None,
        img_cond_seq: Optional[Tensor] = None,
        img_cond_seq_ids: Optional[Tensor] = None,
    ):
        """Initialize the Flux controller.

        :param model: The Flux model instance
        :param timesteps: List of timestep values to process (typically from 1.0 to 0.0)
        :param initial_img: The initial noisy image tensor
        :param img_ids: Image position IDs tensor
        :param txt: Text embeddings tensor
        :param txt_ids: Text position IDs tensor
        :param vec: CLIP embeddings vector tensor
        :param mu: Schedule parameter for time_shift (default: 0.0)
        :param sigma: Schedule parameter for time_shift (default: 1.0)
        :param initial_guidance: Initial guidance (CFG) value (default: 4.0)
        :param img_cond: Optional channel-wise image conditioning tokens
        :param img_cond_seq: Optional sequence-wise image conditioning tokens
        :param img_cond_seq_ids: Optional sequence-wise image conditioning IDs
        """
        self.model = model
        self.timesteps = timesteps
        self.original_timesteps = timesteps.copy()
        self.current_index = 0
        self.img = initial_img
        self.img_ids = img_ids
        self.txt = txt
        self.txt_ids = txt_ids
        self.vec = vec
        self.img_cond = img_cond
        self.img_cond_seq = img_cond_seq
        self.img_cond_seq_ids = img_cond_seq_ids

        self.state_history: list[FluxDenoisingState] = []
        self.mu = mu
        self.sigma = sigma
        self.guidance = initial_guidance
        self.guidance_history: list[float] = [initial_guidance]

    @property
    def is_complete(self) -> bool:
        """Check if all timesteps have been processed."""
        return self.current_index >= len(self.timesteps) - 1

    @property
    def current_state(self) -> FluxDenoisingState:
        """Get the current state of the denoising process."""
        t_curr = self.timesteps[self.current_index]
        t_prev = (
            self.timesteps[self.current_index + 1]
            if self.current_index + 1 < len(self.timesteps)
            else None
        )

        return FluxDenoisingState(
            current_timestep=t_curr,
            previous_timestep=t_prev,
            current_sample=self.img,
            timestep_index=self.current_index,
            total_timesteps=len(self.timesteps),
            guidance=self.guidance,
        )

    def _denoise_step(
        self,
        img: Tensor,
        t_curr: float,
        t_prev: float,
        guidance: float,
    ) -> Tensor:
        """Perform a single denoising step using the Flux model. This extracts the single-step logic from the denoise function.

        :param img: Current image tensor
        :param t_curr: Current timestep
        :param t_prev: Previous timestep
        :param guidance: Guidance (CFG) value
        :returns: Updated image tensor after one denoising step
        """
        guidance_vec = torch.full(
            (img.shape[0],), guidance, device=img.device, dtype=img.dtype
        )
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)

        # Prepare input with optional conditioning
        img_input = img
        img_input_ids = self.img_ids

        if self.img_cond is not None:
            img_input = torch.cat((img, self.img_cond), dim=-1)

        if self.img_cond_seq is not None:
            assert self.img_cond_seq_ids is not None, (
                "You need to provide either both or neither of the sequence conditioning"
            )
            img_input = torch.cat((img_input, self.img_cond_seq), dim=1)
            img_input_ids = torch.cat((img_input_ids, self.img_cond_seq_ids), dim=1)

        # Run model prediction
        pred = self.model(
            img=img_input,
            img_ids=img_input_ids,
            txt=self.txt,
            txt_ids=self.txt_ids,
            y=self.vec,
            timesteps=t_vec,
            guidance=guidance_vec,
        )

        # Extract prediction for original image (excluding conditioning tokens)
        if img_input_ids is not None:
            pred = pred[:, : img.shape[1]]

        # Update image using rectified flow step
        img = img + (t_prev - t_curr) * pred

        return img

    def step(self) -> FluxDenoisingState:
        """Manually increment to the next timestep and perform one denoising step. Uses the current guidance value.

        :returns: The new state after the step.
        :raises ValueError: If all timesteps have already been processed.
        """
        if self.is_complete:
            raise ValueError("All timesteps have been processed. Cannot step further.")

        t_curr = self.timesteps[self.current_index]
        t_prev = self.timesteps[self.current_index + 1]

        # Perform the denoising step with current guidance
        self.img = self._denoise_step(self.img, t_curr, t_prev, self.guidance)

        # Move to next timestep
        self.current_index += 1

        # Save state and guidance
        state = self.current_state
        self.state_history.append(state)
        self.guidance_history.append(self.guidance)

        return state

    def reset(
        self,
        new_initial_img: Optional[Tensor] = None,
        reset_guidance: Optional[float] = None,
    ):
        """Reset the controller to the beginning.

        :param new_initial_img: Optional new initial image to use. If None, keeps the original initial image.
        :param reset_guidance: Optional new guidance value to use. If None, keeps the current guidance value.
        """
        self.current_index = 0
        if new_initial_img is not None:
            self.img = new_initial_img
        if reset_guidance is not None:
            self.guidance = reset_guidance
        self.state_history.clear()
        self.guidance_history = [self.guidance]

    def intervene(self, modified_img: Tensor):
        """Allow user intervention by modifying the current image. This replaces the current image with a user-modified version.

        :param modified_img: The user-modified image tensor to use next.
        """
        self.img = modified_img

    def preview_final(self, preview_fn) -> Tensor:
        """Generate a preview of what the final result would look like from the current state, using a single-step x₀ prediction.

        :param preview_fn: Function that generates a preview from current image. Signature: (img) -> preview_img
        :returns: Preview of the final result.
        """
        return preview_fn(self.img)

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

    def update_conditioning(
        self,
        txt: Optional[Tensor] = None,
        txt_ids: Optional[Tensor] = None,
        vec: Optional[Tensor] = None,
        img_cond: Optional[Tensor] = None,
        img_cond_seq: Optional[Tensor] = None,
        img_cond_seq_ids: Optional[Tensor] = None,
    ):
        """Update conditioning parameters during denoising. Useful for dynamic prompt changes or conditioning adjustments.

        :param txt: Optional new text embeddings tensor
        :param txt_ids: Optional new text position IDs tensor
        :param vec: Optional new CLIP embeddings vector tensor
        :param img_cond: Optional new channel-wise image conditioning tokens
        :param img_cond_seq: Optional new sequence-wise image conditioning tokens
        :param img_cond_seq_ids: Optional new sequence-wise image conditioning IDs
        """
        if txt is not None:
            self.txt = txt
        if txt_ids is not None:
            self.txt_ids = txt_ids
        if vec is not None:
            self.vec = vec
        if img_cond is not None:
            self.img_cond = img_cond
        if img_cond_seq is not None:
            self.img_cond_seq = img_cond_seq
        if img_cond_seq_ids is not None:
            self.img_cond_seq_ids = img_cond_seq_ids
