# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""
Main entry point for divisor CLI.
Routes to different inference modes based on flags.
"""

import argparse
import sys

from fire import Fire

# XD: The ONLY place in root folder that should import from submodules is here.
from divisor.flux1.prompt import main as flux1_main
from divisor.flux2.prompt import main as flux2_main
from divisor.mmada.gradio import main as mmada_main
from divisor.xflux1.prompt import main as xflux1_main
from divisor.flux1.spec import configs as flux_configs
from divisor.mmada.spec import configs as mmada_configs
from divisor.spec import populate_model_choices


def main():
    """Main entry point that routes to appropriate inference function."""
    parser = argparse.ArgumentParser(description="Divisor Multimodal CLI")
    parser.usage = "divisor --model-type dev --quantization <args>"
    parser.epilog = """Valid arguments : 
    --ae_id, --width, --height, --guidance, --seed, --prompt,
    --tiny, --device, --num_steps, --loop,
    --offload, --compile, --verbose
    """
    flux_models = populate_model_choices(flux_configs)
    mmada_models = populate_model_choices(mmada_configs)
    all_models = flux_models + mmada_models
    model_args = []
    model_args_map = {}
    for model in all_models:
        if ":" in model:
            key = model.split(":")[-1]
        else:
            key = model.split(".")[-1]

        # Only add to map if key doesn't exist, or if this is a model.dit.* model (prioritize it)
        if key not in model_args_map or "model.dit." in model:
            model_args_map[key] = model
            if key not in model_args:
                model_args.append(key)

    # Fix 2: Set default to first model or "flux1-dev" if it exists
    default_model = "flux1-dev" if "flux1-dev" in model_args else model_args[0] if model_args else "dev"

    parser.add_argument(
        "-m",
        "--model-type",
        choices=model_args,
        default=default_model,  # Use the computed default
        help=f"""
        Model type to use: {model_args}, Default: {default_model}
        """,
    )

    args, remaining_argv = parser.parse_known_args()
    if args.model_type in mmada_models:
        main = mmada_main  # Gen-Verse/MMaDA-8B-Base, Gen-Verse/TraDo-4B-Instruct, Gen-Verse/TraDo-8B-Instruct

    elif "flux2" in args.model_type:
        main = flux2_main
    elif "mini" in args.model_type:
        main = xflux1_main
    else:
        main = flux1_main

    remaining_argv = ["--model-id", model_args_map[args.model_type]] + remaining_argv
    print(model_args_map[args.model_type])
    sys.argv = [sys.argv[0]] + remaining_argv
    Fire(main)


if __name__ == "__main__":
    main()
