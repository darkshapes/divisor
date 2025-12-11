# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

# Consolidated entry‑point for all supported inference modes.

from typing import Any, Callable, Dict

from fire import Fire
from inspect import signature, Parameter

all_params: dict[str, Parameter] = {}

_MODE_MODULE: Dict[str, str] = {
    "flux1": "divisor.flux1.prompt",
    "xflux1": "divisor.xflux1.prompt",
    "flux2": "divisor.flux2.prompt",
    "mmada": "divisor.mmada.gradio",  # UI entry‑point
}


def _load_main(module_path: str) -> Callable[..., Any]:
    """
    Import *module_path* and return its ``main`` function.
    """
    module = __import__(module_path, fromlist=["main"])
    return getattr(module, "main")


def _patch_run_signature() -> None:
    from inspect import signature, Parameter, Signature

    all_params: dict[str, Parameter] = {}
    for mod_path in _MODE_MODULE.values():
        mod = __import__(mod_path, fromlist=["main"])
        main_fn = getattr(mod, "main")
        for p in signature(main_fn).parameters.values():
            if p.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY):
                all_params.setdefault(p.name, p)

    run_sig = Signature(
        parameters=[
            Parameter("mode", Parameter.POSITIONAL_OR_KEYWORD, default="flux1"),
            *all_params.values(),
        ]
    )
    run.__signature__ = run_sig  # type: ignore[attr-defined]


def run(mode: str = "flux1", **kwargs: Any) -> None:
    main_fn = _load_main(_MODE_MODULE[mode])

    # Keep only arguments that the concrete main expects.
    target_sig = signature(main_fn)
    allowed = {p.name for p in target_sig.parameters.values() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
    args = {k: v for k, v in kwargs.items() if k in allowed}
    main_fn(**args)


def main() -> None:
    """
    CLI wrapper – ``python -m divisor.cli --mode flux2 ...``.
    """
    _patch_run_signature()
    Fire(run)


if __name__ == "__main__":
    main()
