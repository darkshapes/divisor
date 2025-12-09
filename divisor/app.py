# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""
Main entry point for divisor CLI.
Routes to different inference modes based on flags.
"""

import argparse
import sys

from fire import Fire

from divisor.cli_helpers import build_model_arguments
from divisor.flux1.spec import configs as flux1_configs
from divisor.mmada.spec import configs as mmada_configs

flux_args = build_model_arguments(flux1_configs)
mmada_args = build_model_arguments(mmada_configs)
model_args = flux_args | mmada_args


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
        Model type to use: {model_args}, Default: {list(model_args)[0]}
        """,
    )

    args, remaining_argv = parser.parse_known_args()
    if args.model_type in mmada_args:
        from divisor.mmada.gradio import main

        model_id = "Gen-Verse/MMaDA-8B-Base"  # Gen-Verse/MMaDA-8B-Base, Gen-Verse/TraDo-4B-Instruct, Gen-Verse/TraDo-8B-Instruct

    else:
        model_id = args.model_type
        if args.model_type == "flux2-dev":
            from divisor.flux2.prompt import main
        else:
            if args.model_type == "mini":
                from divisor.xflux1.prompt import main

                model_id = f"flux1-dev:{args.model_type}"
            else:
                from divisor.flux1.prompt import main

    remaining_argv = ["--model-id", model_id] + remaining_argv
    sys.argv = [sys.argv[0]] + remaining_argv
    Fire(main)


if __name__ == "__main__":
    main()
