# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Command handlers for interactive denoising state changes."""

import inspect
from typing import Any, Callable

from nnll.console import nfo

from divisor.controller import ManualTimestepController
from divisor.interaction_context import InteractionContext
from divisor.keybinds import _CHOICE_REGISTRY
from divisor.state import MenuState


def _format_menu_line(
    key: str,
    entry: dict,
    state: MenuState,
) -> str:
    """
    Build a printable menu line for a registry entry.\n
    param key: Shortcut key used in the menu (e.g. ``"g"`` for guidance).
    param entry: Registry entry containing at least a ``desc`` string and optionally a `state_keys`` list that maps to attributes on ``state``.
    param state: The current input state instance.
    returns: Formatted line ready for ``nfo`` output."""
    import torch

    shortcut = f"[{key.upper()}]"  # e.g. "[G]"
    description = entry.get("desc", "")  # human‑readable label

    state_keys = entry.get("state_keys", [])
    if not state_keys:
        return f"{shortcut}{description}"

    state_values = []
    for attr_name in state_keys:
        attr_value = getattr(state, attr_name, "?")
        if isinstance(attr_value, torch.Tensor):
            attr_value = attr_value.item() if attr_value.numel() == 1 else f"tensor{tuple(attr_value.shape)}"
        state_values.append(f"{attr_value}")

    current_state = " / ".join(state_values)  # join multiple values
    return f"{shortcut}{description}: {current_state}"


def route_choices(
    controller: ManualTimestepController,
    state: MenuState,
    interaction_context: InteractionContext,
    **kwargs: Any,
) -> MenuState:
    """Process user choice input and return updated state.\n
    :param controller: ManualTimestepController instance
    :param state: Current input state
    :param interaction_context: InteractionContext instance
    :returns: Updated input state
    """

    step = state.timestep_index
    nfo(f"Step {step}/{state.total_timesteps - 1} @ noise level {state.current_timestep:.4f}")
    for choice_letter in sorted(_CHOICE_REGISTRY):
        choice_description = _CHOICE_REGISTRY[choice_letter]
        if not choice_letter:
            continue
        line = _format_menu_line(choice_letter, choice_description, state)
        nfo(line)

    menu_keybinds: dict[str, Callable[[], Any]] = {}
    for choice_letter, choice_function in _CHOICE_REGISTRY.items():
        fn = choice_function["fn"]
        sig = inspect.signature(fn)
        param_names = list(sig.parameters.keys())
        kwargs = {}
        for name in param_names:
            if name not in kwargs and hasattr(controller, name):
                kwargs[name] = getattr(interaction_context, name, None)
        menu_keybinds[choice_letter] = lambda fn=fn, kwargs=kwargs: fn(controller, state, interaction_context, **kwargs)
    prompt = "".join(k.upper() for k in _CHOICE_REGISTRY if k) + "/q"
    choice = input(f": [{prompt}] or advance with Enter:").lower().strip()

    if choice == "q":
        import sys

        nfo("Quitting...")
        sys.exit(0)

    elif choice == "/":
        return controller.current_state
    elif choice in menu_keybinds:
        result = menu_keybinds[choice]()
        state = result if isinstance(result, MenuState) else state
    else:
        nfo("Invalid choice, please try again")

    return controller.current_state
