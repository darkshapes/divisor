# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""
Main entry point for divisor CLI.
Routes to different inference modes based on flags.
Exclusive import location of submodules to avoid circular imports.
"""

import argparse
import importlib
from pathlib import Path
from typing import Any, Callable, Dict
from inspect import signature, Parameter, Signature

from fire import Fire
from nnll.console import nfo

from divisor.spec import flux_map
from divisor.spec import mmada_map

model_args = flux_map | mmada_map


def _load_main_from_folder(folder: Path) -> Callable | None:
    """Find `main` inside `prompt.py` or `gradio.py` in *subfolders*.
    :param folder: The folder to search for `main`
    :return: The `main` function or None if not found"""
    for candidate in ("prompt.py", "gradio.py"):
        module_path = folder / candidate
        if module_path.is_file():
            module_name = f"{folder.parent.name}.{folder.name}.{candidate[:-3]}"
            try:
                mod = importlib.import_module(module_name)
                return getattr(mod, "main", None)
            except (AttributeError, ModuleNotFoundError, ImportError) as e:
                nfo(f"Could not import {module_name}: {e}")


def build_main_mapping(base_dir: Path = Path(__file__).parent) -> Dict[str, Callable]:
    """Create a mapping of model keys to their main functions.
    :param base_dir: The base directory to search for subfolders
    :return: A dictionary of model keys to their main functions"""
    mapping: Dict[str, Callable] = {}

    for subfolder in base_dir.iterdir():
        if not subfolder.is_dir():
            continue

        key = subfolder.name
        for model_key in model_args:
            if model_key.startswith(key):
                key = model_key
                break
            continue  # skip folders we don’t have a config for

        main_fn = _load_main_from_folder(subfolder)
        if main_fn is None:
            continue

        mapping[key] = main_fn

    return mapping


MAIN_ROUTINES: Dict[str, Callable] = build_main_mapping()


def _patch_run_signature() -> None:
    """Patch the run function to accept the mode and keyword arguments."""
    all_params: dict[str, Parameter] = {}
    for _, module_object in MAIN_ROUTINES.items():
        main_fn = module_object
        for p in signature(main_fn).parameters.values():
            if p.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY):
                all_params.setdefault(p.name, p)

    run_sig = Signature(
        parameters=[
            Parameter("model_id", Parameter.POSITIONAL_OR_KEYWORD, default="flux1-dev"),
            *all_params.values(),
        ]
    )
    run.__signature__ = run_sig  # type: ignore[attr-defined]


def run(model_id="flux1-dev", *args, **kwargs: Any) -> None:
    """Run the main function for the given model ID.
    :param mir_id: MIR series ID (e.g., "flux1-dev")
    :param args: Additional arguments
    :param kwargs: Additional keyword arguments"""
    import sys

    if main_fn := MAIN_ROUTINES.get(model_id):
        target_sig = signature(main_fn)
        allowed = {p.name for p in target_sig.parameters.values() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
        args = {k: v for k, v in kwargs.items() if k in allowed}
        remaining_argv = ["--mir-id", model_args[model_id]] + sys.argv[1:]

        sys.argv = [sys.argv[0]] + remaining_argv
        print(sys.argv)
        main_fn(**args)
    else:
        raise ValueError(f"Model ID {model_id} not found")


def main() -> None:
    _patch_run_signature()
    Fire(run)


if __name__ == "__main__":
    main()
