# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""
FluxDivide - Advanced intervention and branching capabilities for Flux denoising.

This module provides the "division" functionality - the ability to branch, cache,
and explore different parameter paths during interactive denoising. This is the
core feature that gives Divisor its name.
"""

from typing import Optional
import torch
from torch import Tensor

from .controller import time_shift
from .flux_controller import FluxController


class FluxDivide:
    """
    Advanced intervention and branching wrapper for FluxController.

    Provides branching, caching, and schedule manipulation capabilities
    that allow users to explore different parameter paths during denoising.
    This is the core "division" functionality of Divisor.
    """

    def __init__(self, controller: FluxController):
        """
        Initialize FluxDivide with a FluxController instance.

        Args:
            controller: The FluxController instance to wrap and extend
        """
        self.controller = controller
        self._preview_cache: Optional[dict] = None

    def preview_branches(
        self,
        parameter_variations: list[dict],
        preview_steps: int = 1,
    ) -> list[tuple[dict, Tensor]]:
        """
        Generate multiple preview branches with different parameter variations
        without modifying the main controller state.

        This creates parallel exploration paths from the current state, allowing
        you to compare different parameter settings (e.g., different guidance values)
        before committing to a path.

        The results are cached so they can be restored later.

        Args:
            parameter_variations: List of dictionaries, each containing parameter
                                overrides. Keys can include:
                                - 'guidance': float (guidance value)
                                - 'txt': Tensor (text embeddings)
                                - 'vec': Tensor (CLIP embeddings)
                                - 'img_cond': Tensor (image conditioning)
                                - 'img_cond_seq': Tensor (sequence conditioning)
                                - 'img_cond_seq_ids': Tensor (sequence conditioning IDs)
            preview_steps: Number of denoising steps to take for each branch
                          (default: 1 for single-step preview)

        Returns:
            List of tuples: (parameter_dict, preview_img_tensor) for each branch
        """
        if self.controller.is_complete:
            return []

        # Snapshot current state
        current_img = self.controller.img.clone()
        current_index = self.controller.current_index

        results = []

        for params in parameter_variations:
            # Create a working copy of the image
            branch_img = current_img.clone()

            # Extract parameter overrides (use current values as defaults)
            branch_guidance = params.get("guidance", self.controller.guidance)
            branch_txt = params.get("txt", self.controller.txt)
            branch_vec = params.get("vec", self.controller.vec)
            branch_img_cond = params.get("img_cond", self.controller.img_cond)
            branch_img_cond_seq = params.get(
                "img_cond_seq", self.controller.img_cond_seq
            )
            branch_img_cond_seq_ids = params.get(
                "img_cond_seq_ids", self.controller.img_cond_seq_ids
            )

            # Apply preview steps
            branch_index = current_index
            for step in range(preview_steps):
                if branch_index >= len(self.controller.timesteps) - 1:
                    break

                t_curr = self.controller.timesteps[branch_index]
                t_prev = self.controller.timesteps[branch_index + 1]

                # Use temporary parameter overrides
                guidance_vec = torch.full(
                    (branch_img.shape[0],),
                    branch_guidance,
                    device=branch_img.device,
                    dtype=branch_img.dtype,
                )
                t_vec = torch.full(
                    (branch_img.shape[0],),
                    t_curr,
                    dtype=branch_img.dtype,
                    device=branch_img.device,
                )

                # Prepare input with branch-specific conditioning
                img_input = branch_img
                img_input_ids = self.controller.img_ids

                if branch_img_cond is not None:
                    img_input = torch.cat((branch_img, branch_img_cond), dim=-1)

                if branch_img_cond_seq is not None:
                    assert branch_img_cond_seq_ids is not None, (
                        "You need to provide either both or neither of the sequence conditioning"
                    )
                    img_input = torch.cat((img_input, branch_img_cond_seq), dim=1)
                    img_input_ids = torch.cat(
                        (img_input_ids, branch_img_cond_seq_ids), dim=1
                    )

                # Run model prediction with branch parameters
                pred = self.controller.model(
                    img=img_input,
                    img_ids=img_input_ids,
                    txt=branch_txt,
                    txt_ids=self.controller.txt_ids,
                    y=branch_vec,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                )

                # Extract prediction for original image
                if img_input_ids is not None:
                    pred = pred[:, : branch_img.shape[1]]

                # Update image
                branch_img = branch_img + (t_prev - t_curr) * pred
                branch_index += 1

            results.append((params, branch_img))

        # Cache the results and snapshot state
        self._preview_cache = {
            "results": results,
            "snapshot_img": current_img,
            "snapshot_index": current_index,
            "preview_steps": preview_steps,
            "snapshot_guidance": self.controller.guidance,
            "snapshot_txt": (
                self.controller.txt.clone() if self.controller.txt is not None else None
            ),
            "snapshot_vec": (
                self.controller.vec.clone() if self.controller.vec is not None else None
            ),
            "snapshot_img_cond": (
                self.controller.img_cond.clone()
                if self.controller.img_cond is not None
                else None
            ),
            "snapshot_img_cond_seq": (
                self.controller.img_cond_seq.clone()
                if self.controller.img_cond_seq is not None
                else None
            ),
            "snapshot_img_cond_seq_ids": (
                self.controller.img_cond_seq_ids.clone()
                if self.controller.img_cond_seq_ids is not None
                else None
            ),
        }

        return results

    def preview_guidance_branches(
        self,
        guidance_values: list[float],
        preview_steps: int = 1,
    ) -> list[tuple[float, Tensor]]:
        """
        Generate preview branches with different guidance values.
        Convenience method that wraps preview_branches for guidance-only variations.

        Args:
            guidance_values: List of guidance values to test
            preview_steps: Number of steps to preview (default: 1)

        Returns:
            List of tuples: (guidance_value, preview_img_tensor)
        """
        variations = [{"guidance": g} for g in guidance_values]
        results = self.preview_branches(variations, preview_steps)
        return [(r[0]["guidance"], r[1]) for r in results]

    def restore_from_preview_cache(self, branch_index: int) -> bool:
        """
        Restore the controller state from a cached preview branch.
        This allows you to "commit" to one of the preview branches.

        Args:
            branch_index: Index of the branch to restore (from preview_branches results)

        Returns:
            True if restoration was successful, False if no cache exists or index is invalid

        Raises:
            ValueError: If branch_index is out of range
        """
        if self._preview_cache is None:
            return False

        results = self._preview_cache["results"]
        if branch_index < 0 or branch_index >= len(results):
            raise ValueError(
                f"Branch index {branch_index} out of range [0, {len(results) - 1}]"
            )

        # Get the selected branch
        params, preview_img = results[branch_index]

        # Restore snapshot state
        self.controller.img = self._preview_cache["snapshot_img"].clone()
        self.controller.current_index = self._preview_cache["snapshot_index"]

        # Apply parameter overrides from the selected branch
        if "guidance" in params:
            self.controller.guidance = params["guidance"]
        if "txt" in params:
            self.controller.txt = params["txt"]
        if "vec" in params:
            self.controller.vec = params["vec"]
        if "img_cond" in params:
            self.controller.img_cond = params["img_cond"]
        if "img_cond_seq" in params:
            self.controller.img_cond_seq = params["img_cond_seq"]
        if "img_cond_seq_ids" in params:
            self.controller.img_cond_seq_ids = params["img_cond_seq_ids"]

        # Apply the preview image (the result of the preview steps)
        # This represents the state after preview_steps have been taken
        self.controller.img = preview_img.clone()

        # Update current_index to reflect that preview_steps have been taken
        preview_steps = self._preview_cache["preview_steps"]
        self.controller.current_index = min(
            self.controller.current_index + preview_steps,
            len(self.controller.timesteps) - 1,
        )

        # Clear cache after restoration
        self._preview_cache = None

        return True

    def get_preview_cache(self) -> Optional[dict]:
        """
        Get the current preview cache without modifying it.

        Returns:
            Dictionary containing cached preview results and snapshot state, or None
        """
        return self._preview_cache

    def clear_preview_cache(self):
        """
        Clear the preview cache.
        """
        self._preview_cache = None

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
        if self.controller.is_complete:
            return []

        # Get remaining timesteps
        remaining_timesteps = self.controller.timesteps[self.controller.current_index :]

        if steps is None:
            steps = len(remaining_timesteps)

        # Convert to tensor and normalize to [0, 1] if needed
        # Assuming timesteps are already in [0, 1] range
        tensor_steps = torch.tensor(remaining_timesteps, dtype=torch.float32)

        # Apply time_shift
        adjusted_timesteps = time_shift(
            self.controller.mu, self.controller.sigma, tensor_steps, steps, compress
        )

        # Convert back to list
        new_timesteps = adjusted_timesteps.tolist()

        # Update the schedule (keep processed timesteps, replace remaining)
        self.controller.timesteps = (
            self.controller.timesteps[: self.controller.current_index] + new_timesteps
        )

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
        if self.controller.is_complete:
            raise ValueError("No current step to subdivide.")

        t_curr = self.controller.timesteps[self.controller.current_index]
        t_prev = (
            self.controller.timesteps[self.controller.current_index + 1]
            if self.controller.current_index + 1 < len(self.controller.timesteps)
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
                self.controller.mu,
                self.controller.sigma,
                normalized,
                sub_steps + 1,
                compress,
            )

            # Denormalize back to original range [t_prev, t_curr]
            sub_timesteps = adjusted_normalized * (t_curr - t_prev) + t_prev

            # Ensure we maintain the endpoints
            sub_timesteps[0] = t_curr
            sub_timesteps[-1] = t_prev

        # Convert to list (exclude the first one as it's the current timestep)
        new_sub_timesteps = sub_timesteps[1:].tolist()

        # Replace current step with sub-steps
        self.controller.timesteps = (
            self.controller.timesteps[: self.controller.current_index + 1]
            + new_sub_timesteps
            + self.controller.timesteps[self.controller.current_index + 1 :]
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
