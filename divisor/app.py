# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""
Main entry point for divisor CLI.
Routes to different inference modes based on flags.
"""

import sys
from fire import Fire


def main():
    """Main entry point that routes to appropriate inference function.

    Usage:
        dvzr                    # Default: Flux image generation mode
        dvzr -o / --omni        # DiMOO multimodal understanding mode
    """
    # Check for --omni or -o flag (as standalone arguments, not part of other args)
    has_omni_flag = any(arg in ["-o", "--omni"] for arg in sys.argv)

    if has_omni_flag:
        # Route to DiMOO inference_mmu.py main()
        from divisor.dimoo_modules.inference_mmu import main as omni_main

        # Remove the omni flag from argv before passing to omni_main
        # since it uses argparse and doesn't know about this flag
        original_argv = sys.argv.copy()
        try:
            # Filter out only exact matches to avoid removing -o from --output, etc.
            filtered_argv = [arg for arg in original_argv if arg not in ["-o", "--omni"]]
            sys.argv = filtered_argv
            omni_main()
        finally:
            sys.argv = original_argv
    else:
        # Route to Flux prompt.py main() (default behavior)
        from divisor.flux_modules.prompt import main as flux_main

        # Flux uses Fire, which automatically handles sys.argv
        Fire(flux_main)


if __name__ == "__main__":
    main()
