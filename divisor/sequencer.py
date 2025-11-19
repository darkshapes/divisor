# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import torch
import math
from controller import ManualTimestepController
from flux_controller import FluxController
from flux_divide import FluxDivide


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
dtype = torch.float16
seed = torch.random.seed()


# BFL Flux noise generation
def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    return torch.randn(
        num_samples,
        16,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        dtype=dtype,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    ).to(device)


def flux_sequencer(controller: FluxController):
    while not controller.is_complete:
        state = controller.current_state
        print(
            f"\nCurrent timestep: {state.current_timestep} ({state.timestep_index}/{state.total_timesteps})"
        )
        print(f"Current guidance: {state.guidance:.2f}")
        print(f"Sample: {state.current_sample}")
        controller.step()
        controller.set_guidance(7.0)

        # Advanced usage - with branching
        controller = FluxController(...)
        divide = FluxDivide(controller)

        # Generate branches
        branches = divide.preview_guidance_branches([3.5, 7.0, 10.0])

        # Restore a branch
        divide.restore_from_preview_cache(branch_index=1)

        # Continue with controller
        controller.step()


def example_usage():
    """
    Example of how to use the ManualTimestepController.
    """
    # Example timesteps (typically from 1.0 to 0.0)
    timesteps = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

    # Example initial sample (would be actual tensor/image in real usage)
    initial_sample = get_noise(1, 256, 256, device, dtype, seed)

    # Example denoising function (now accepts guidance parameter)
    def denoise_step(sample, t_curr, t_prev, guidance):
        # In real usage, this would call the diffusion model with the guidance value
        print(f"Denoising from t={t_curr} to t={t_prev} with guidance={guidance:.2f}")
        return f"denoised_sample_at_{t_prev}_guidance_{guidance:.2f}"

    # Create controller
    controller = ManualTimestepController(
        timesteps=timesteps,
        initial_sample=initial_sample,
        denoise_step_fn=denoise_step,
    )

    # Manual stepping - user controls when to advance
    while not controller.is_complete:
        state = controller.current_state
        print(
            f"\nCurrent timestep: {state.current_timestep} ({state.timestep_index}/{state.total_timesteps})"
        )
        print(f"Current guidance: {state.guidance:.2f}")
        print(f"Sample: {state.current_sample}")

        # User can intervene here if desired
        # controller.intervene(modified_sample)

        # Example: User can stretch/compress the current step
        # controller.stretch_compress_current_step(sub_steps=3, compress=1.5)

        # Example: User can stretch/compress the remaining schedule
        # controller.stretch_compress_schedule(compress=0.8, steps=20)

        # User manually triggers next step or adjusts settings
        user_input = input(
            "Press Enter to step, 's' to stretch current step, 'c' to compress schedule, "
            "'g' to set guidance, '+/-' to adjust guidance: "
        )

        if user_input.lower() == "s":
            sub_steps = int(input("Enter number of sub-steps (default 3): ") or "3")
            compress = float(input("Enter compress factor (default 1.0): ") or "1.0")
            new_steps = controller.stretch_compress_current_step(sub_steps, compress)
            print(
                f"Subdivided current step into {len(new_steps)} sub-steps: {new_steps}"
            )
            continue
        elif user_input.lower() == "c":
            compress = float(input("Enter compress factor (default 1.0): ") or "1.0")
            steps_input = input(
                "Enter desired number of steps (press Enter to keep current): "
            )
            steps = int(steps_input) if steps_input else None
            new_timesteps = controller.stretch_compress_schedule(compress, steps)
            print(f"Adjusted remaining schedule to {len(new_timesteps)} timesteps")
            continue
        elif user_input.lower() == "g":
            new_guidance = float(
                input(f"Enter new guidance value (current: {state.guidance:.2f}): ")
            )
            controller.set_guidance(new_guidance)
            print(f"Guidance set to {new_guidance:.2f}")
            continue
        elif user_input == "+":
            delta = float(
                input("Enter amount to increase guidance (default 0.5): ") or "0.5"
            )
            controller.adjust_guidance(delta)
            print(f"Guidance adjusted to {controller.guidance:.2f}")
            continue
        elif user_input == "-":
            delta = float(
                input("Enter amount to decrease guidance (default 0.5): ") or "0.5"
            )
            controller.adjust_guidance(-delta)
            print(f"Guidance adjusted to {controller.guidance:.2f}")
            continue

        state = controller.step()
        print(
            f"Stepped to timestep: {state.current_timestep} with guidance: {state.guidance:.2f}"
        )

    print("\nDenoising complete!")


if __name__ == "__main__":
    example_usage()
