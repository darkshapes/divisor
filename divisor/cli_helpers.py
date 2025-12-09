# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""CLI helper functions for interactive input handling and state management."""

from typing import Any, Callable, Optional

from nnll.console import nfo

from divisor.controller import ManualTimestepController
from divisor.state import DenoisingState


def update_state_and_cache(
    controller: ManualTimestepController,
    setter_func: Callable,
    value: Any,
    clear_prediction_cache: Callable[[], None],
    success_message: str,
) -> DenoisingState:
    """Generic state update helper that sets value, clears cache, and refreshes state.\n
    :param controller: ManualTimestepController instance
    :param setter_func: Controller setter method to call
    :param value: Value to set
    :param clear_prediction_cache: Function to clear prediction cache
    :param success_message: Message to display on success
    :returns: Updated DenoisingState
    """
    setter_func(value)
    clear_prediction_cache()
    state = controller.current_state
    nfo(success_message)
    return state


def get_float_input(
    prompt: str,
    current_value: float,
    default_value: float = 0.0,
    allow_empty: bool = True,
) -> Optional[float]:
    """Get float input with optional reset to default.\n
    :param prompt: Input prompt string
    :param current_value: Current value to display
    :param default_value: Value to use if input is empty
    :param allow_empty: Whether empty input is allowed
    :returns: Parsed float value or None if invalid
    """
    try:
        user_input = input(prompt).strip()
        if allow_empty and user_input == "":
            return default_value
        return float(user_input)
    except ValueError:
        return None


def get_int_input(
    prompt: str,
    current_value: int,
    generate_random: Optional[Callable[[], int]] = None,
) -> Optional[int]:
    """Get integer input with optional random generation.\n
    :param prompt: Input prompt string
    :param current_value: Current value to display
    :param generate_random: Optional function to generate random value if input is empty
    :returns: Parsed integer value or None if invalid
    """
    try:
        user_input = input(prompt).strip()
        if user_input == "" and generate_random is not None:
            return generate_random()
        return int(user_input)
    except ValueError:
        return None


def handle_toggle(
    controller: ManualTimestepController,
    state: DenoisingState,
    current_value: bool,
    setter_func: Callable[[bool], None],
    clear_prediction_cache: Optional[Callable[[], None]] = None,
    enabled_msg: str = "ENABLED",
    disabled_msg: str = "DISABLED",
) -> DenoisingState:
    """Generic toggle handler.\n
    :param controller: ManualTimestepController instance
    :param state: Current DenoisingState
    :param current_value: Current boolean value
    :param setter_func: Controller setter method
    :param clear_prediction_cache: Optional function to clear cache
    :param enabled_msg: Message when enabled
    :param disabled_msg: Message when disabled
    :returns: Updated DenoisingState
    """
    new_value = not current_value
    setter_func(new_value)
    if clear_prediction_cache is not None:
        clear_prediction_cache()
    state = controller.current_state
    status = enabled_msg if new_value else disabled_msg
    nfo(f"{status}")
    return state


def handle_float_setting(
    controller: ManualTimestepController,
    state: DenoisingState,
    prompt: str,
    current_value: float,
    setter_func: Callable[[float], None],
    clear_prediction_cache: Callable[[], None],
    default_value: float = 0.0,
    value_name: str = "value",
    format_str: str = ".4f",
) -> DenoisingState:
    """Handle float setting with reset option.\n
    :param controller: ManualTimestepController instance
    :param state: Current DenoisingState
    :param prompt: Input prompt
    :param current_value: Current float value
    :param setter_func: Controller setter method
    :param clear_prediction_cache: Function to clear cache
    :param default_value: Default value for empty input
    :param value_name: Name of the value for messages
    :param format_str: Format string for display
    :returns: Updated DenoisingState
    """
    try:
        user_input = input(prompt).strip()
        if user_input == "":
            new_value = default_value
            reset_msg = f"{value_name} reset to {default_value:{format_str}}"
        else:
            new_value = float(user_input)
            reset_msg = f"{value_name} set to {new_value:{format_str}}"

        setter_func(new_value)
        clear_prediction_cache()
        state = controller.current_state
        nfo(reset_msg)
    except ValueError:
        nfo(f"Invalid {value_name}, keeping current value")
    return state


def build_model_arguments(configs: dict[str, Any]) -> dict[str, str]:
    """Build model arguments from configs.\n
    :param configs: Configuration mapping containing model specs
    :returns: List of model arguments
    """
    from divisor.spec import populate_model_choices

    model_choices = populate_model_choices(configs)
    model_args: dict = {}
    filters = ["fp8-", ".vae."]
    for model in model_choices:
        if not any(filter in model for filter in filters):
            if ":" in model:
                key = model.split(":")[-1]
            else:
                key = model.split(".")[-1]
            if key not in model_args:
                model_args[key] = model

    return model_args
