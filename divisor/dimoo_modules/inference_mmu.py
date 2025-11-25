# -*- coding: utf-8 -*-
"""
Text understanding inference script
"""

import os
from PIL import Image
import torch
import time
from transformers import AutoTokenizer
import sys
from nnll.init_gpu import device
from huggingface_hub import snapshot_download

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from divisor.dimoo_modules.config import SPECIAL_TOKENS
from divisor.dimoo_modules.model import LLaDAForMultiModalGeneration
from divisor.dimoo_modules.utils.image_utils import encode_img_with_breaks, calculate_vq_params, generate_crop_size_list, var_center_crop, add_break_line
from divisor.dimoo_modules.generators.text_understanding_generator import generate_text_understanding
from divisor.dimoo_modules.utils.prompt_utils import generate_multimodal_understanding_prompt

CHECKPOINTS_PATH = snapshot_download(repo_id="Alpha-VLLM/Lumina-DiMOO")


def main(
    checkpoint: str = CHECKPOINTS_PATH,
    prompt: str = "Describe what a cat is.",
    image_path: str | None = None,
    steps: int = 128,
    gen_length: int = 1024,
    block_length: int = 256,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    vae_ckpt: str = "./vae_ckpt",
    output_dir: str = ".output",
):
    """Text understanding inference using DiMOO (multimodal understanding) model.

    :param checkpoint: Fine-tuned checkpoint path. Defaults to Alpha-VLLM/Lumina-DiMOO from HuggingFace.
    :param prompt: Text prompt/question for understanding. Default: "Describe what a cat is."
    :param image_path: Optional path to input image for multimodal understanding.
    :param steps: Number of generation steps. Default: 128.
    :param gen_length: Maximum generation length in tokens. Default: 1024.
    :param block_length: Block length for generation. Default: 256.
    :param temperature: Sampling temperature. Default: 0.0 (deterministic).
    :param cfg_scale: Classifier-free guidance scale. Default: 0.0 (no guidance).
    :param vae_ckpt: VAE checkpoint path for image encoding. Default: "./vae_ckpt".
    :param output_dir: Directory to save outputs. Default: ".output".
    """

    # Special tokens
    MASK = SPECIAL_TOKENS["mask_token"]
    NEW_LINE = SPECIAL_TOKENS["newline_token"]
    BOA = SPECIAL_TOKENS["answer_start"]  # Begin of Answer
    EOA = SPECIAL_TOKENS["answer_end"]  # End of Answer

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = LLaDAForMultiModalGeneration.from_pretrained(
        checkpoint,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    # Get prompt and image
    question = prompt
    # Generate prompt using utility function
    input_prompt = generate_multimodal_understanding_prompt(question)
    input_ids = tokenizer(input_prompt)["input_ids"]
    input_token = input_ids[:-1] + input_ids[-1:]

    if image_path is not None:
        from diffusers.models.autoencoders.vq_model import VQModel

        vqvae = VQModel.from_pretrained(vae_ckpt, subfolder="vqvae").to(device)

        # Calculate VQ parameters
        vae_scale = 2 ** (len(vqvae.config.block_out_channels) - 1)

        print(f"Processing image: {image_path}")
        img = Image.open(image_path)
        crop_size_list = generate_crop_size_list((1024 // 32) ** 2, 32)
        image = var_center_crop(img, crop_size_list=crop_size_list)
        image_width, image_height = image.size
        seq_len, newline_every, token_grid_height, token_grid_width = calculate_vq_params(image_height, image_width, vae_scale)
        input_img_token = encode_img_with_breaks(image, vqvae=vqvae)
        img_token = add_break_line(input_img_token, token_grid_height, token_grid_width, new_number=NEW_LINE)
        input_img_token = img_token
        input_token = input_ids[:-1] + input_img_token + input_ids[-1:]
    print(f"Question: {question}")

    # Prediction text token start index
    code_start = len(input_token) + 1

    # Build text mask prediction sequence
    input_token = input_token + [BOA] + gen_length * [MASK] + [EOA]
    input_ids = torch.tensor(input_token, device=device).unsqueeze(0)

    # Generate text
    start_time = time.time()
    out_new = generate_text_understanding(
        model,
        input_ids,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=temperature,
        cfg_scale=cfg_scale,
        remasking="low_confidence",
        code_start=code_start,
    )

    text_new = tokenizer.batch_decode(out_new[:, code_start:-1], skip_special_tokens=True)[0]

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"[✓] (Time {elapsed_time:.2f}s)")

    print(f"Generated text: {text_new}")


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
