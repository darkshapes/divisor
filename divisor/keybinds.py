# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from collections import OrderedDict
from typing import Any, Callable, Dict, Optional

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
from divisor.noise import prepare_noise_for_model
from divisor.state import DenoisingState


def choice(key: str, description: str) -> Callable[[Callable], Callable]:
    """Decorator that registers a function as a menu choice.\n
    :param key: Single-character option the user will type
    (e.g. ``"g"``).Use ``""`` for the *Enter* (advance) option.
    :param description: Short human-readable description that will be shown in the prompt and self-document the registry

    @# key → {"fn": callable, "desc": description}
    def example_function(controller: ManualTimestepController, state: DenoisingState, clear_prediction_cache: Callable[[], None]) -> DenoisingState:
        `Example function.\n
        :param controller: ManualTimestepController instance
        :param state: Current DenoisingState
        :param clear_prediction_cache: Function to clear prediction cache
        :returns: Updated DenoisingState
        `
        return state
    """

    def wrapper(fn: Callable) -> Callable:
        # Normalise to lower‑case – the UI works case‑insensitively.
        _CHOICE_REGISTRY[key.lower()] = {"fn": fn, "desc": description}
        return fn

    return wrapper


_CHOICE_REGISTRY: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()


@choice("", "Advance (Enter)")
def _advance(
    controller: ManualTimestepController,
    state: DenoisingState,
    interaction_context: InteractionContext,
) -> DenoisingState:
    """Advance one step (default action when the user presses Enter)."""
    nfo("Advancing...")
    interaction_context.clear_prediction_cache()
    controller.step()
    return controller.current_state


@choice("g", "Guidance")
def change_guidance(
    controller: ManualTimestepController,
    state: DenoisingState,
    interaction_context: InteractionContext,
) -> DenoisingState:
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
    state: DenoisingState,
    interaction_context: InteractionContext,
) -> DenoisingState:
    """Handle layer dropout change.\n
    :param controller: ManualTimestepController instance
    :param state: Current DenoisingState
    :param interaction_context: InteractionContext arguments
    :returns: Updated DenoisingState
    """
    try:
        dropout_input = input("Enter layer indices to drop (comma-separated, or 'none' to clear): ").strip()
        if dropout_input.lower() == "none" or dropout_input == "":
            layer_indices = None
        else:
            layer_indices = [int(x.strip()) for x in dropout_input.split(",")]

        # Update controller first
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
    state: DenoisingState,
    interaction_context: InteractionContext,
) -> DenoisingState:
    """Handle resolution change.\n
    :param controller: ManualTimestepController instance
    :param state: Current DenoisingState
    :param clear_prediction_cache: Function to clear prediction cache
    :returns: Updated DenoisingState
    """
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
    state: DenoisingState,
    interaction_context: InteractionContext,
) -> DenoisingState:
    """Handle seed change.\n
    :param controller: ManualTimestepController instance
    :param state: Current DenoisingState
    :param rng: Random number generator instance
    :param clear_prediction_cache: Function to clear prediction cache
    :param t5: Optional T5 embedder instance (required for Flux1/XFlux1 to prepare sample)
    :param clip: Optional CLIP embedder instance (required for Flux1/XFlux1 to prepare sample)
    :returns: Updated DenoisingState
    """
    current_seed = state.seed if state.seed is not None else 0
    if new_seed := get_int_input(
        f"Enter new seed number (current: {current_seed}, or press Enter for random): ",
        current_seed,
        generate_random=lambda: interaction_context.rng.next_seed(),
    ):
        controller.set_seed(new_seed)
        new_sample = prepare_noise_for_model(
            height=state.height,  # type: ignore
            width=state.width,  # type: ignore
            seed=new_seed,
            t5=interaction_context.t5,
            clip=interaction_context.clip,
            prompt=state.prompt,
        )

        # Update controller's current_sample
        controller.current_sample = new_sample
        # Get updated state from controller (it will use the updated current_sample)
        interaction_context.clear_prediction_cache()
        updated_state = controller.current_state
        nfo(f"Seed set to: {new_seed}")
        return updated_state
    nfo("Invalid seed value, keeping current seed")
    return state


@choice("b", "Buffer Mask")
def toggle_buffer_mask(
    controller: ManualTimestepController,
    state: DenoisingState,
) -> DenoisingState:
    """Handle buffer mask toggle.\n
    :param controller: ManualTimestepController instance
    :param state: Current DenoisingState
    :returns: Updated DenoisingState
    """
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
    state: DenoisingState,
    interaction_context: InteractionContext,
) -> DenoisingState:
    """Handle VAE shift/scale offset change.\n
    :param controller: ManualTimestepController instance
    :param state: Current DenoisingState
    :param ae: Optional AutoEncoder instance
    :param clear_prediction_cache: Function to clear prediction cache
    :returns: Updated DenoisingState
    """
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
    state: DenoisingState,
    interaction_context: InteractionContext,
) -> DenoisingState:
    """Handle deterministic mode toggle.\n
    :param controller: ManualTimestepController instance
    :param state: Current DenoisingState
    :param clear_prediction_cache: Function to clear prediction cache
    :returns: Updated DenoisingState
    """

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
    state: DenoisingState,
    interaction_context: InteractionContext,
    recompute_text_embeddings: Optional[Callable[[str], None]],
) -> DenoisingState:
    """Handle prompt change.\n
    :param controller: ManualTimestepController instance
    :param state: Current DenoisingState
    :param clear_prediction_cache: Function to clear prediction cache
    :param recompute_text_embeddings: Optional function to recompute text embeddings
    :returns: Updated DenoisingState
    """
    current_prompt = state.prompt if state.prompt is not None else ""
    new_prompt = input(f"Enter new prompt (current: {current_prompt}): ").strip()

    if new_prompt:
        controller.set_prompt(new_prompt)
        if recompute_text_embeddings is not None:
            recompute_text_embeddings(new_prompt)
        else:
            interaction_context.clear_prediction_cache()
        state = controller.current_state
        nfo(f"Prompt set to: {new_prompt}")
    else:
        nfo("Prompt unchanged")

    return state


@choice("e", "Edit Mode")
def edit_mode(
    interaction_context: InteractionContext,
) -> None:
    """Handle edit mode (debugger breakpoint).\n
    :param clear_prediction_cache: Function to clear prediction cache
    """
    nfo("Entering edit mode (use c/cont to exit)...")
    breakpoint()
    interaction_context.clear_prediction_cache()


@choice("j", "Jump to Step")
def jump_to_step(
    controller: ManualTimestepController,
    state: DenoisingState,
    interaction_context: InteractionContext,
) -> DenoisingState:
    """Run the controller forward until a user‑specified step index.\n
    The user enters the *target* step number (0‑based, inclusive).  The
    function repeatedly calls ``controller.step()`` until the controller
    reaches that index or the process finishes.

    :param controller: ManualTimestepController instance
    :param state: Current DenoisingState (unused – we return the
                  controller’s final state)
    :param clear_prediction_cache: Called once before the jump so any
                                   cached predictions are invalidated.
    :returns: The DenoisingState after the jump (or the current state if
              the target is out of range).
    """
    # Ask the user for the target step
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
    state: DenoisingState,
    interaction_context: InteractionContext,
) -> DenoisingState:
    """Handle variation seed/strength change.\n
    :param controller: ManualTimestepController instance
    :param state: Current DenoisingState
    :param variation_rng: Variation random number generator instance
    :param clear_prediction_cache: Function to clear prediction cache
    :returns: Updated DenoisingState
    """
    try:
        var_input = input(
            f"Variation (current integer seed: {state.variation_seed}, float strength: {state.variation_strength:.3f}. type a number, leave empty for random, or use 0.0 to disable): "
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
