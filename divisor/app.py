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
    parser.add_argument(
        "--quantization",
        action="store_true",
        help="Enable quantization (fp8, e5m2, e4m3fn) for the model",
    )

    parser.add_argument(
        "-m",
        "--model-type",
        choices=["dev", "schnell"],
        default="dev",
        help="Model type to use: 'dev' (flux1-dev) or 'schnell' (flux1-schnell). Default: dev",
    )

    # Parse known args to separate our args from Fire's args
    args, remaining_argv = parser.parse_known_args()

    if args.omni:
        # Remove --omni/-o from argv and route to omni mode
        filtered_argv = [arg for arg in sys.argv[1:] if arg not in ["-o", "--omni"]]
        sys.argv = [sys.argv[0]] + filtered_argv
        # TODO: Import and call omni main function when implemented
        # from divisor.omni_modules.prompt import main as omni_main
        # Fire(omni_main)
        raise NotImplementedError("Omni mode not yet implemented")
    else:
        # Route to Flux mode
        from divisor.flux_modules.prompt import main as flux_main

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
