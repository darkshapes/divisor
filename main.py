"""
Interactive denoising with manual timestep control.
Allows users to manually increment through timesteps one at a time.
"""

from typing import Optional, Callable, Any
from dataclasses import dataclass


@dataclass
class DenoisingState:
    """State of the denoising process at a given timestep."""
    current_timestep: float
    previous_timestep: Optional[float]
    current_sample: Any
    timestep_index: int
    total_timesteps: int


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
        denoise_step_fn: Callable[[Any, float, float], Any],
    ):
        """
        Initialize the controller.
        
        Args:
            timesteps: List of timestep values to process (typically from 1.0 to 0.0)
            initial_sample: The initial noisy sample to start denoising from
            denoise_step_fn: Function that performs one denoising step.
                            Signature: (sample, t_curr, t_prev) -> new_sample
        """
        self.timesteps = timesteps
        self.denoise_step_fn = denoise_step_fn
        self.current_index = 0
        self.current_sample = initial_sample
        self.state_history: list[DenoisingState] = []
        
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
        )
    
    def step(self) -> DenoisingState:
        """
        Manually increment to the next timestep and perform one denoising step.
        
        Returns:
            The new state after the step.
            
        Raises:
            ValueError: If all timesteps have already been processed.
        """
        if self.is_complete:
            raise ValueError("All timesteps have been processed. Cannot step further.")
        
        t_curr = self.timesteps[self.current_index]
        t_prev = self.timesteps[self.current_index + 1]
        
        # Perform the denoising step
        self.current_sample = self.denoise_step_fn(
            self.current_sample,
            t_curr,
            t_prev
        )
        
        # Move to next timestep
        self.current_index += 1
        
        # Save state
        state = self.current_state
        self.state_history.append(state)
        
        return state
    
    def reset(self, new_initial_sample: Optional[Any] = None):
        """
        Reset the controller to the beginning.
        
        Args:
            new_initial_sample: Optional new initial sample to use.
                               If None, keeps the original initial sample.
        """
        self.current_index = 0
        if new_initial_sample is not None:
            self.current_sample = new_initial_sample
        self.state_history.clear()
    
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


def example_usage():
    """
    Example of how to use the ManualTimestepController.
    """
    # Example timesteps (typically from 1.0 to 0.0)
    timesteps = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    
    # Example initial sample (would be actual tensor/image in real usage)
    initial_sample = "noisy_sample"
    
    # Example denoising function
    def denoise_step(sample, t_curr, t_prev):
        # In real usage, this would call the diffusion model
        print(f"Denoising from t={t_curr} to t={t_prev}")
        return f"denoised_sample_at_{t_prev}"
    
    # Create controller
    controller = ManualTimestepController(
        timesteps=timesteps,
        initial_sample=initial_sample,
        denoise_step_fn=denoise_step,
    )
    
    # Manual stepping - user controls when to advance
    while not controller.is_complete:
        state = controller.current_state
        print(f"\nCurrent timestep: {state.current_timestep} ({state.timestep_index}/{state.total_timesteps})")
        print(f"Sample: {state.current_sample}")
        
        # User can intervene here if desired
        # controller.intervene(modified_sample)
        
        # User manually triggers next step
        input("Press Enter to step to next timestep...")
        state = controller.step()
        print(f"Stepped to timestep: {state.current_timestep}")
    
    print("\nDenoising complete!")


if __name__ == "__main__":
    example_usage()

