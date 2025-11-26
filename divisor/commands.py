# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Command handlers for interactive denoising state changes."""

from typing import Any, Callable, Optional

from nnll.console import nfo
from nnll.helpers import generate_valid_resolutions

from divisor.cli_helpers import (
    get_float_input,
    get_int_input,
    handle_float_setting,
    handle_toggle,
    update_state_and_cache,
)
from divisor.controller import (
    DenoisingState,
    ManualTimestepController,
    rng,
    variation_rng,
)
from divisor.flux_modules.autoencoder import AutoEncoder
from divisor.variant import change_variation


def process_choice(
    controller: ManualTimestepController,
    state: DenoisingState,
    clear_prediction_cache: Callable[[], None],
    current_layer_dropout: list[Optional[list[int]]],
    rng,
    variation_rng,
    ae: Optional[AutoEncoder] = None,
    t5: Optional[Any] = None,
    clip: Optional[Any] = None,
    recompute_text_embeddings: Optional[Callable[[str], None]] = None,
) -> DenoisingState:
    """Process user choice input and return updated state.\n
    :param controller: ManualTimestepController instance
    :param state: Current DenoisingState
    :param clear_prediction_cache: Function to clear prediction cache
    :param current_layer_dropout: Mutable list containing current layer dropout
    :param rng: Random number generator instance
    :param variation_rng: Variation random number generator instance
    :param ae: Optional AutoEncoder instance
    :param t5: Optional T5 embedder instance
    :param clip: Optional CLIP embedder instance
    :param recompute_text_embeddings: Optional function to recompute text embeddings
    :returns: Updated DenoisingState
    """
    # Display status
    step = state.timestep_index
    nfo(f"Step {step}/{state.total_timesteps} @ noise level {state.current_timestep:.4f}")
    nfo(f"[G]uidance: {state.guidance:.2f}")
    nfo(f"[S]eed: {state.seed}")
    if state.width is not None and state.height is not None:
        nfo(f"[R]esolution: {state.width}x{state.height}")
    if state.layer_dropout:
        nfo(f"[L]ayer dropout: {state.layer_dropout}")
    else:
        nfo("[L]ayer dropout: None")
    nfo(f"[B]uffer mask: {'ON' if state.use_previous_as_mask else 'OFF'}")
    if ae is not None:
        nfo(f"[A]utoencoder shift offset: {state.vae_shift_offset:.4f}")
        nfo(f"[A]utoencoder scale offset: {state.vae_scale_offset:.4f}")
    if state.variation_seed is not None:
        nfo(f"[V]Variation seed: {state.variation_seed}, strength: {state.variation_strength:.3f}")
    else:
        nfo("[V]Variation: OFF")
    nfo(f"[D]eterministic: {'ON' if state.deterministic else 'OFF'}")
    if state.prompt is not None:
        prompt_display = state.prompt[:60] + "..." if len(state.prompt) > 60 else state.prompt
        nfo(f"[P]rompt: {prompt_display}")

    choice = input(": [BDGLSRVXP] advance with Enter: ").lower().strip()

    choice_handlers = {
        "": lambda: (
            nfo("Advancing..."),
            clear_prediction_cache(),
            controller.step(),
            controller.current_state,
        ),
        "g": lambda: change_guidance(controller, state, clear_prediction_cache),
        "l": lambda: change_layer_dropout(controller, state, current_layer_dropout, clear_prediction_cache),
        "r": lambda: change_resolution(controller, state, clear_prediction_cache),
        "s": lambda: change_seed(controller, state, rng, clear_prediction_cache),
        "b": lambda: toggle_buffer_mask(controller, state),
        "a": lambda: change_vae_offset(controller, state, ae, clear_prediction_cache),
        "v": lambda: change_variation(controller, state, variation_rng, clear_prediction_cache),
        "d": lambda: toggle_deterministic(controller, state, clear_prediction_cache),
        "e": lambda: edit_mode(clear_prediction_cache),
        "p": lambda: change_prompt(controller, state, clear_prediction_cache, recompute_text_embeddings),
    }

    if choice in choice_handlers:
        result = choice_handlers[choice]()
        state = result if isinstance(result, type(state)) else state
    else:
        nfo("Invalid choice, please try again")

    return controller.current_state


def change_guidance(
    controller: ManualTimestepController,
    state: DenoisingState,
    clear_prediction_cache: Callable[[], None],
) -> DenoisingState:
    """Handle guidance value change.\n
    :param controller: ManualTimestepController instance
    :param state: Current DenoisingState
    :param clear_prediction_cache: Function to clear prediction cache
    :returns: Updated DenoisingState
    """
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
            clear_prediction_cache,
            f"Guidance set to {new_guidance:.2f}",
        )
    nfo("Invalid guidance value, keeping current value")
    return state


def change_layer_dropout(
    controller: ManualTimestepController,
    state: DenoisingState,
    current_layer_dropout: list[Optional[list[int]]],
    clear_prediction_cache: Callable[[], None],
) -> DenoisingState:
    """Handle layer dropout change.\n
    :param controller: ManualTimestepController instance
    :param state: Current DenoisingState
    :param current_layer_dropout: Mutable list containing current layer dropout
    :param clear_prediction_cache: Function to clear prediction cache
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
        # Keep current_layer_dropout in sync (for backward compatibility)
        current_layer_dropout[0] = layer_indices
        clear_prediction_cache()
        state = controller.current_state
        nfo(f"Layer dropout set to: {layer_indices}")
        return state
    except ValueError:
        nfo("Invalid layer indices, keeping current value")
    return controller.current_state


def change_resolution(
    controller: ManualTimestepController,
    state: DenoisingState,
    clear_prediction_cache: Callable[[], None],
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
            nfo("\nValid resolutions (same patch count):")
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
                return update_state_and_cache(
                    controller,
                    controller.set_resolution,
                    (new_width, new_height),
                    clear_prediction_cache,
                    f"Resolution set to: {new_width}x{new_height}",
                )
    except (ValueError, IndexError):
        nfo("Invalid resolution input, keeping current value")
    return state


def change_seed(
    controller: ManualTimestepController,
    state: DenoisingState,
    rng,
    clear_prediction_cache: Callable[[], None],
) -> DenoisingState:
    """Handle seed change.\n
    :param controller: ManualTimestepController instance
    :param state: Current DenoisingState
    :param rng: Random number generator instance
    :param clear_prediction_cache: Function to clear prediction cache
    :returns: Updated DenoisingState
    """
    current_seed = state.seed if state.seed is not None else 0
    new_seed = get_int_input(
        f"Enter new seed number (current: {current_seed}, or press Enter for random): ",
        current_seed,
        generate_random=lambda: rng.next_seed(),
    )
    if new_seed is not None:
        return update_state_and_cache(
            controller,
            controller.set_seed,
            new_seed,
            clear_prediction_cache,
            f"Seed set to: {new_seed}",
        )
    nfo("Invalid seed value, keeping current seed")
    return state


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


def change_vae_offset(
    controller: ManualTimestepController,
    state: DenoisingState,
    ae: Optional[AutoEncoder],
    clear_prediction_cache: Callable[[], None],
) -> DenoisingState:
    """Handle VAE shift/scale offset change.\n
    :param controller: ManualTimestepController instance
    :param state: Current DenoisingState
    :param ae: Optional AutoEncoder instance
    :param clear_prediction_cache: Function to clear prediction cache
    :returns: Updated DenoisingState
    """
    if ae is None:
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
            clear_prediction_cache,
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
            clear_prediction_cache,
            default_value=0.0,
            value_name="VAE shift offset",
        )
    return state


def toggle_deterministic(
    controller: ManualTimestepController,
    state: DenoisingState,
    clear_prediction_cache: Callable[[], None],
) -> DenoisingState:
    """Handle deterministic mode toggle.\n
    :param controller: ManualTimestepController instance
    :param state: Current DenoisingState
    :param clear_prediction_cache: Function to clear prediction cache
    :returns: Updated DenoisingState
    """

    deterministic = not state.deterministic
    rng.random_mode(reproducible=deterministic)
    variation_rng.random_mode(reproducible=deterministic)

    return handle_toggle(
        controller,
        state,
        state.deterministic,
        controller.set_deterministic,
        clear_prediction_cache,
        enabled_msg="Deterministic mode: ENABLED",
        disabled_msg="Deterministic mode: DISABLED",
    )


def change_prompt(
    controller: ManualTimestepController,
    state: DenoisingState,
    clear_prediction_cache: Callable[[], None],
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
    new_prompt = input(f"Enter new prompt (current: {current_prompt[:60]}...): ").strip()

    if new_prompt:
        controller.set_prompt(new_prompt)
        if recompute_text_embeddings is not None:
            recompute_text_embeddings(new_prompt)
        else:
            clear_prediction_cache()
        state = controller.current_state
        nfo(f"Prompt set to: {new_prompt[:60]}...")
    else:
        nfo("Prompt unchanged")

    return state


def edit_mode(
    clear_prediction_cache: Callable[[], None],
) -> None:
    """Handle edit mode (debugger breakpoint).\n
    :param clear_prediction_cache: Function to clear prediction cache
    """
    nfo("Entering edit mode (use c/cont to exit)...")
    breakpoint()
    clear_prediction_cache()
