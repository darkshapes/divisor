# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""
Choice registry decorator.

Usage:
```
@choice("g", "Guidance")
def change_guidance(controller: ManualTimestepController,
state: MenuState,
clear_prediction_cache: Callable[[], None],) -> MenuState:

    ...

    return state
```

Returns:
```
{"fn": callable, "desc": description}
```

"""

from collections import OrderedDict
from typing import Any, Callable, Dict

from nnll.console import nfo
from nnll.helpers import generate_valid_resolutions

from divisor.cli_input import (
    get_float_input,
    get_int_input,
    handle_float_setting,
    handle_toggle,
)
from divisor.controller import ManualTimestepController, update_state_and_cache
from divisor.interaction_context import InteractionContext
from divisor.noise import prepare_4d_noise_for_3d_model
from divisor.state import MenuState


def choice(key: str, description: str) -> Callable[[Callable], Callable]:
    """Decorator that registers a function as a menu choice.\n
    :param key: Single-character option the user will type (e.g. ``"g"``).Use ``""`` for the *Enter* (advance) option.
    :param description: Short human-readable description that will be shown in the prompt and self-document the registry"""

    def wrapper(fn: Callable) -> Callable:
        normalized_key = key.lower()
        _CHOICE_REGISTRY[normalized_key] = {"fn": fn, "desc": description}  # case‑insensitive
        return fn

    return wrapper


_CHOICE_REGISTRY: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()


@choice("", "Advance (Enter)")
def _advance(
    controller: ManualTimestepController,
    state: MenuState,
    interaction_context: InteractionContext,
) -> MenuState:
    """Advance one step (default action when Enter is pressed)."""
    nfo("Advancing...")
    interaction_context.clear_prediction_cache()
    controller.step()
    return controller.current_state


@choice("g", "Guidance")
def change_guidance(
    controller: ManualTimestepController,
    state: MenuState,
    interaction_context: InteractionContext,
) -> MenuState:
    """Handle guidance value change.\n"""
    new_guidance = get_float_input(
        f"Enter new guidance value (current: {state.guidance:.2f}): ",
        state.guidance,
        allow_empty=False,
    )
    if new_guidance is not None:
        return update_state_and_cache(
            controller,
            controller.set_guidance,
            new_guidance,
            interaction_context,
            f"Guidance set to {new_guidance}",
        )
    nfo("Invalid guidance value, keeping current value")
    return state


@choice("l", "Layer Dropout")
def change_layer_dropout(
    controller: ManualTimestepController,
    state: MenuState,
    interaction_context: InteractionContext,
) -> MenuState:
    """Handle layer dropout change.\n
    :param controller: ManualTimestepController instance
    :param state: Current input state
    :param interaction_context: InteractionContext arguments
    :returns: Updated input state"""
    try:
        dropout_input = input("Enter layer indices to drop (comma-separated, or 'none' to clear): ").strip()
        if dropout_input.lower() == "none" or dropout_input == "":
            layer_indices = None
        else:
            layer_indices = [int(x.strip()) for x in dropout_input.split(",")]

        controller.set_layer_dropout(layer_indices)
        interaction_context.clear_prediction_cache()
        state = controller.current_state
        nfo(f"Layer dropout set to: {layer_indices}")
        return state
    except ValueError:
        nfo("Invalid layer indices, keeping current value")
    return controller.current_state


@choice("r", "Resolution")
def change_resolution(
    controller: ManualTimestepController,
    state: MenuState,
    interaction_context: InteractionContext,
) -> MenuState:
    """Handle resolution change.\n
    :param controller: ManualTimestepController instance
    :param state: Current input state
    :param clear_prediction_cache: Function to clear prediction cache
    :returns: Updated input state"""

    try:
        if state.width is None or state.height is None:
            nfo("Cannot generate resolutions: width or height not set")
        else:
            valid_resolutions = generate_valid_resolutions(state.width, state.height)
            nfo("Valid resolutions (same patch count):")
            for i, (w, h) in enumerate(valid_resolutions):
                current_marker = ""
                if state.width == w and state.height == h:
                    current_marker = " (current)"
                nfo(f"  {i}: {w}x{h}{current_marker}")
            resolution_input = input(f"\nEnter resolution index (0-{len(valid_resolutions) - 1}) or 'custom' for custom: ").strip()
            if resolution_input.lower() == "custom":
                width_input = input("Enter width: ").strip()
                height_input = input("Enter height: ").strip()
                new_width = int(width_input)
                new_height = int(height_input)
            else:
                resolution_idx = int(resolution_input)
                if 0 <= resolution_idx < len(valid_resolutions):
                    new_width, new_height = valid_resolutions[resolution_idx]
                else:
                    nfo("Invalid resolution index, keeping current value")
                    return state
            if new_width is not None and new_height is not None:
                controller.set_resolution(new_width, new_height)
                interaction_context.clear_prediction_cache()
                state = controller.current_state
                nfo(f"Resolution set to: {new_width}x{new_height}")
                return state
    except (ValueError, IndexError):
        nfo("Invalid resolution input, keeping current value")
    return state


@choice("s", "Seed")
def change_seed(
    controller: ManualTimestepController,
    state: MenuState,
    interaction_context: InteractionContext,
) -> MenuState:
    """Handle seed change.\n
    :param controller: ManualTimestepController instance
    :param state: Current menu state
    :param rng: Random number generator instance
    :param clear_prediction_cache: Function to clear prediction cache
    :param t5: Optional T5 embedder instance (required for Flux1/XFlux1 to prepare sample)
    :param clip: Optional CLIP embedder instance (required for Flux1/XFlux1 to prepare sample)
    :returns: Updated menu state"""

    current_seed = state.seed if state.seed is not None else 0
    if new_seed := get_int_input(
        f"Enter new seed number (current: {current_seed}, or press Enter for random): ",
        current_seed,
        generate_random=lambda: interaction_context.rng.next_seed(),
    ):
        controller.set_seed(new_seed)
        new_sample = prepare_4d_noise_for_3d_model(
            height=state.height,  # type: ignore
            width=state.width,  # type: ignore
            seed=new_seed,
            t5=interaction_context.t5,
            clip=interaction_context.clip,
            prompt=state.prompt,
        )

        controller.current_sample = new_sample
        interaction_context.clear_prediction_cache()
        updated_state = controller.current_state
        nfo(f"Seed set to: {new_seed}")
        return updated_state
    nfo("Invalid seed value, keeping current seed")
    return state


@choice("b", "Buffer Mask")
def toggle_buffer_mask(
    controller: ManualTimestepController,
    state: MenuState,
    interaction_context: InteractionContext,
) -> MenuState:
    """Handle buffer mask toggle.\n
    :param controller: ManualTimestepController instance
    :param state: Current input state
    :returns: Updated input state"""

    return handle_toggle(
        controller,
        state,
        state.use_previous_as_mask,
        controller.set_use_previous_as_mask,
        enabled_msg="Previous step tensor mask: ENABLED",
        disabled_msg="Previous step tensor mask: DISABLED",
    )


@choice("a", "Autoencoder Offset")
def change_vae_offset(
    controller: ManualTimestepController,
    state: MenuState,
    interaction_context: InteractionContext,
) -> MenuState:
    """Handle VAE shift/scale offset change.\n
    :param controller: ManualTimestepController instance
    :param state: Current input state
    :param ae: Optional AutoEncoder instance
    :param clear_prediction_cache: Function to clear prediction cache
    :returns: Updated input state"""

    if interaction_context.ae is None:
        nfo("AutoEncoder not available, cannot set VAE offset")
        return state

    vae_input = input("\nChoose [S]hift or [C]scale: ").strip().lower()
    if vae_input == "c":
        return handle_float_setting(
            controller,
            state,
            f"Enter VAE scale offset (current: {state.vae_scale_offset:.4f}, or press Enter to reset to 0.0): ",
            state.vae_scale_offset,
            controller.set_vae_scale_offset,
            interaction_context.clear_prediction_cache,
            default_value=0.0,
            value_name="VAE scale offset",
        )
    elif vae_input == "s":
        return handle_float_setting(
            controller,
            state,
            f"Enter VAE shift offset (current: {state.vae_shift_offset:.4f}, or press Enter to reset to 0.0): ",
            state.vae_shift_offset,
            controller.set_vae_shift_offset,
            interaction_context.clear_prediction_cache,
            default_value=0.0,
            value_name="VAE shift offset",
        )
    return state


@choice("d", "Deterministic")
def toggle_deterministic(
    controller: ManualTimestepController,
    state: MenuState,
    interaction_context: InteractionContext,
) -> MenuState:
    """Handle deterministic mode toggle.\n
    :param controller: ManualTimestepController instance
    :param state: Current input state
    :param clear_prediction_cache: Function to clear prediction cache
    :returns: Updated input state"""

    deterministic = not state.deterministic
    interaction_context.rng.random_mode(reproducible=deterministic)
    interaction_context.variation_rng.random_mode(reproducible=deterministic)

    return handle_toggle(
        controller,
        state,
        state.deterministic,
        controller.set_deterministic,
        interaction_context.clear_prediction_cache,
        enabled_msg="Deterministic mode: ENABLED",
        disabled_msg="Deterministic mode: DISABLED",
    )


@choice("p", "Prompt")
def change_prompt(
    controller: ManualTimestepController,
    state: MenuState,
    interaction_context: InteractionContext,
) -> MenuState:
    """Handle prompt change.\n
    :param controller: ManualTimestepController instance
    :param state: Current input state
    :param clear_prediction_cache: Function to clear prediction cache
    :param recompute_text_embeddings: Optional function to recompute text embeddings
    :returns: Updated input state"""

    current_prompt = state.prompt if state.prompt is not None else ""
    new_prompt = input(f"Enter new prompt (current: {current_prompt}): ").strip()

    if new_prompt:
        controller.set_prompt(new_prompt)
        if interaction_context.recompute_text_embeddings is not None:
            interaction_context.recompute_text_embeddings(new_prompt)
        else:
            interaction_context.clear_prediction_cache()
        state = controller.current_state
        nfo(f"Prompt set to: {new_prompt}")
    else:
        nfo("Prompt unchanged")

    return state


@choice("e", "Edit Mode")
def edit_mode(
    controller: ManualTimestepController,
    state: MenuState,
    interaction_context: InteractionContext,
) -> None:
    """Handle edit mode (debugger breakpoint).\n
    :param clear_prediction_cache: Function to clear prediction cache"""
    nfo("Entering edit mode (use c/cont to exit)...")
    breakpoint()
    interaction_context.clear_prediction_cache()


@choice("j", "Jump to Step")
def jump_to_step(
    controller: ManualTimestepController,
    state: MenuState,
    interaction_context: InteractionContext,
) -> MenuState:
    """Run the controller forward until a user‑specified step index.\n
    :param controller: ManualTimestepController instance
    :param state: Current input state (unused – we return the controller’s final state)
    :param clear_prediction_cache: Routine to nvalidate cache prediction.
    :returns: Input state post jump (or the current state if the target is out of range)."""

    target_str = input(f"Jump to step (0‑{controller.current_index} …{len(controller.timesteps) - 1}): ").strip()
    if not target_str.isdigit():
        nfo("Invalid step number – aborting jump.")
        return controller.current_state

    target = int(target_str)

    if target <= controller.current_index:
        nfo("Target step is before or equal to the current step – nothing to do.")
        return controller.current_state
    if target >= len(controller.timesteps):
        nfo("Target step exceeds the schedule – jumping to the end.")
        target = len(controller.timesteps) - 1

    interaction_context.clear_prediction_cache()
    while controller.current_index < target and not controller.is_complete:
        controller.step()

    nfo(f"Jumped to step {controller.current_index}/{len(controller.timesteps) - 1}")
    return controller.current_state


@choice("v", "Variation")
def change_variation(
    controller: ManualTimestepController,
    state: MenuState,
    interaction_context: InteractionContext,
) -> MenuState:
    """Handle variation seed/strength change.\n
    :param controller: ManualTimestepController instance
    :param state: Current input state
    :param variation_rng: Variation random number generator instance
    :param clear_prediction_cache: Function to clear prediction cache
    :returns: Updated input state"""
    try:
        var_input = input(
            f"Variation (current integer seed: {state.variation_seed}, float strength: {state.variation_strength:.3f}. type a number, leave empty for random seed, or use 0.0 to disable): "
        ).strip()

        if not var_input or "." not in var_input:
            # Try to parse as integer (seed)
            try:
                if var_input != "":
                    variation_seed = interaction_context.variation_rng.next_seed(int(var_input))
                else:
                    variation_seed = interaction_context.variation_rng.next_seed()
                controller.set_variation_seed(variation_seed)
                interaction_context.clear_prediction_cache()
                state = controller.current_state
                nfo(f"Variation seed set to: {state.variation_seed}")
            except ValueError:
                nfo("Invalid integer seed value, keeping current value")
        else:
            # Try to parse as float (strength)
            try:
                strength_value = float(var_input)
                if strength_value < 0.0 or strength_value > 1.0:
                    state = controller.current_state
                    nfo("Variation strength must be between 0.0 and 1.0, keeping current value")
                else:
                    controller.set_variation_strength(strength_value)
                    interaction_context.clear_prediction_cache()
                    state = controller.current_state
                    nfo(f"Variation strength set to: {strength_value:.3f}")
            except ValueError:
                nfo("Invalid float strength value, keeping current value")
    except (ValueError, KeyboardInterrupt):
        nfo("Invalid variation value, keeping current value")
    return state


@choice("w", "Rewind")
def rewind(
    controller: ManualTimestepController,
    state: MenuState,
    interaction_context: InteractionContext,
) -> MenuState:
    """Rewind the controller by a specified number of steps.\n
    :param controller: ManualTimestepController instance
    :param state: Current input state
    :param clear_prediction_cache: Function to clear prediction cache
    :returns: Updated input state"""
    import random as prng

    num_steps = get_int_input(
        f"Enter number of steps to rewind (current: {controller.rewind_steps}): ",
        controller.rewind_steps,
        generate_random=lambda: prng.randint(0, controller.timesteps.index(state.timestep_index)),
    )
    if num_steps is not None:
        controller.rewind(num_steps)
        interaction_context.clear_prediction_cache()
        state = controller.current_state
        nfo(f"Rewind step {num_steps} to {controller.current_index}")
        return state
    nfo("Invalid number of steps, keeping current value")
    return state


# Optionally recompute the *remaining* schedule with a different compression
# remaining = controller.timesteps[controller.current_index + 1 :]
# new_tail = time_shift(
#    schedule_mu=controller.mu,
#    schedule_sigma=controller.sigma,
#    original_timestep_tensor=torch.tensor(remaining),
#    desired_step_count=len(remaining),
#    compression_factor=0.9,               # e.g. stretch the tail a bit
# )
# controller.timesteps[controller.current_index + 1 :] = new_tail.tolist()
