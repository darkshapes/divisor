from fire import Fire

from divisor.flux1.prompt import main as cli_main

if __name__ == "__main__":
    Fire(
        {
            "t2i": cli_main,
        }
    )
