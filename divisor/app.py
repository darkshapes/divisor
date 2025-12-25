# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""
Main entry point for divisor CLI.
Routes to different inference modes based on flags.
Exclusive import location of submodules to avoid circular imports.
"""

import argparse
import sys

from fire import Fire

from divisor.spec import flux_map, mmada_map, acestep_map

model_args = flux_map | mmada_map | acestep_map


def main():
    """Main entry point that routes to appropriate inference function."""
    parser = argparse.ArgumentParser(description="Divisor Multimodal CLI")
    parser.usage = "divisor --model-type dev --quantization <args>"
    parser.epilog = """Valid arguments :
    --ae_id, --width, --height, --guidance, --seed, --prompt,
    --tiny, --device, --num_steps, --loop,
    --offload, --compile, --verbose
    """
    parser.add_argument(
        "--quantization",
        action="store_true",
        help="Enable quantization (fp8, e5m2, e4m3fn) for the model",
    )

    parser.add_argument(
        "-m",
        "--model-type",
        choices=model_args,
        default=list(model_args)[0],
        help=f"""
        Model type to use: {list(model_args)}, Default: {list(model_args)[0]}
        """,
    )

    args, remaining_argv = parser.parse_known_args()
    model_id = args.model_type
    if args.model_type in mmada_map:
        from divisor.mmada.gradio import main

        remaining_argv = [""]  # Gradio app doesn't need arguments
    elif args.model_type in flux_map:
        model_id = args.model_type
        if args.model_type == "flux2-dev":
            from divisor.flux2.prompt import main
        else:
            if args.model_type == "mini":
                from divisor.xflux1.prompt import main
            else:
                from divisor.flux1.prompt import main
        remaining_argv = ["--mir-id", model_args[model_id]] + remaining_argv  # change to     model_args[model_id]
    else:
        from divisor.acestep.gradio import main

    sys.argv = [sys.argv[0]] + remaining_argv
    Fire(main)


if __name__ == "__main__":
    main()
