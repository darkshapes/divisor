# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""
Main entry point for divisor CLI.
Routes to different inference modes based on flags.
"""

import argparse
import sys
from fire import Fire


def main():
    """Main entry point that routes to appropriate inference function."""
    parser = argparse.ArgumentParser(description="Divisor CLI - Flux image generation and multimodal understanding")
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
        choices=["dev", "schnell", "dev2"],
        default="dev",
        help="Model type to use: 'dev' (flux1-dev), 'schnell' (flux1-schnell), or 'dev2' (flux2-dev). Default: dev",
    )

    # Parse known args to separate our args from Fire's args
    args, remaining_argv = parser.parse_known_args()

    # Route to appropriate Flux mode based on model type
    if args.model_type == "dev2":
        # Route to Flux2
        from divisor.flux2.cli import main as flux2_main

        # Flux2 uses model_name parameter, default is "flux.2-dev"
        # Fire will handle the remaining arguments
        sys.argv = [sys.argv[0]] + remaining_argv
        Fire(flux2_main)
    else:
        # Route to Flux1
        from divisor.flux1.prompt import main as flux_main

        # Add model_id argument to remaining argv for Fire to parse
        # Fire converts underscores to hyphens, so model_id becomes --model-id
        model_id = f"flux1-{args.model_type}"
        # Insert model_id argument before other arguments
        remaining_argv = ["--model-id", model_id] + remaining_argv
        sys.argv = [sys.argv[0]] + remaining_argv

        # Flux uses Fire, which automatically handles sys.argv
        Fire(flux_main)


if __name__ == "__main__":
    main()
