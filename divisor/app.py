# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""
Main entry point for divisor CLI.
Routes to different inference modes based on flags.
"""

import argparse
import sys
from fire import Fire

from divisor.flux2.prompt import main as flux2_main
from divisor.xflux1.prompt import main as xflux1_main
from divisor.flux1.prompt import main as flux1_main
from divisor.mmada.gradio import main as mmada_main


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
        choices=["dev", "schnell", "dev2", "mini", "llm"],
        default="dev",
        help="Model type to use: 'dev' (flux1-dev), 'schnell' (flux1-schnell), or 'dev2' (flux2-dev), 'mini' (flux1-mini). Default: dev",
    )

    args, remaining_argv = parser.parse_known_args()
    if args.model_type == "llm":
        main = mmada_main
        model_id = "Gen-Verse/MMaDA-8B-MixCoT"  # Gen-Verse/MMaDA-8B-Base, Gen-Verse/TraDo-4B-Instruct, Gen-Verse/TraDo-8B-Instruct

    elif args.model_type == "dev2":
        main = flux2_main

        model_id = f"flux2-{args.model_type.strip('2')}"

    else:
        if args.model_type == "mini":
            main = xflux1_main
            model_id = f"flux1-dev:{args.model_type}"
        else:
            main = flux1_main
            model_id = f"flux1-{args.model_type}"
    remaining_argv = ["--model-id", model_id] + remaining_argv
    sys.argv = [sys.argv[0]] + remaining_argv
    Fire(main)


if __name__ == "__main__":
    main()
