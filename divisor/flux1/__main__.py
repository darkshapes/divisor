# SPDX-License-Identifier:Apache-2.0
# original BFL Flux code from https://github.com/black-forest-labs/flux

from fire import Fire

from divisor.flux1.prompt import main as cli_main

if __name__ == "__main__":
    Fire(
        {
            "t2i": cli_main,
        }
    )
