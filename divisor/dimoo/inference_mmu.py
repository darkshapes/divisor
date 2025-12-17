# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/Alpha-VLLM/Lumina-DiMOO

"""
Text understanding inference script
"""

import argparse
import os
import sys
import time

from huggingface_hub import snapshot_download
from nnll.init_gpu import device
import torch
from transformers import AutoTokenizer

from divisor.dimoo.config import SPECIAL_TOKENS
from divisor.dimoo.prompt_utils import generate_text_prompt
from divisor.dimoo.text_understanding_generator import generate_text_understanding
from divisor.mmada.modeling_llada import LLaDAModelLM

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Text understanding inference")
    parser.add_argument("--model_id", type=str, required=False, default=snapshot_download("Alpha-VLLM/Lumina-DiMOO"), help="Model ID")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--steps", type=int, default=128, help="Generation steps")
    parser.add_argument("--gen_length", type=int, default=1024, help="Generation length")
    parser.add_argument("--block_length", type=int, default=256, help="Block length")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    parser.add_argument("--cfg_scale", type=float, default=0.0, help="CFG scale")
    parser.add_argument("--vae_ckpt", type=str, default="./vae_ckpt", help="VAE checkpoint path")
    parser.add_argument("--output_dir", type=str, default="outputs_text_understanding", help="Output directory")

    args = parser.parse_args()

    # Special tokens
    MASK = SPECIAL_TOKENS["mask_token"]
    BOA = SPECIAL_TOKENS["answer_start"]  # Begin of Answer
    EOA = SPECIAL_TOKENS["answer_end"]  # End of Answer

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    input_prompt = generate_text_prompt(args.prompt)
    input_token = tokenizer(input_prompt)["input_ids"]

    # Prediction text token start index
    code_start = len(input_token) + 1

    # Build text mask predition sequence
    input_token = input_token + [BOA] + args.gen_length * [MASK] + [EOA]
    input_ids = torch.tensor(input_token, device=device).unsqueeze(0)

    # Generate text
    start_time = time.time()
    out_new = generate_text_understanding(
        model,
        input_ids,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        remasking="low_confidence",
        code_start=code_start,
    )

    text_new = tokenizer.batch_decode(out_new[:, code_start:-1], skip_special_tokens=True)[0]

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"[✓] (Time {elapsed_time:.2f}s)")

    print(f"Generated text: {text_new}")


if __name__ == "__main__":
    main()
