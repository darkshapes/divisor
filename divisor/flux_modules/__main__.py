from fire import Fire

from divisor.flux_modules.prompt import main as cli_main

if __name__ == "__main__":
    Fire(
        {
            "t2i": cli_main,
        }
    )
