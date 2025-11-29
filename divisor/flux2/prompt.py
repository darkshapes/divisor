import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from einops import rearrange
from PIL import ExifTags, Image
from nnll.console import nfo
from nnll.init_gpu import device, sync_torch, clear_cache

from divisor.controller import rng
from divisor.flux1.sampling import SamplingOptions, DenoisingState
from divisor.noise import get_noise
from divisor.flux2.openrouter_api_client import DEFAULT_SAMPLING_PARAMS, OpenRouterAPIClient
from divisor.flux2.sampling import (
    batched_prc_img,
    batched_prc_txt,
    denoise_interactive,
    encode_image_refs,
    get_schedule,
    scatter_ids,
)
from divisor.flux2.util import FLUX2_MODEL_INFO, load_ae, load_flow_model, load_mistral_small_embedder

from divisor.flux2 import precision


def parse_prompt(options: SamplingOptions) -> SamplingOptions | None:
    user_question = "Next prompt (write /h for help, /q to quit and leave empty to repeat):\n"
    usage = (
        "Usage: Either write your prompt directly, leave this field empty "
        "to repeat the prompt or write a command starting with a slash:\n"
        "- '/w <width>' will set the width of the generated image\n"
        "- '/h <height>' will set the height of the generated image\n"
        "- '/s <seed>' sets the next seed\n"
        "- '/g <guidance>' sets the guidance (flux-dev only)\n"
        "- '/n <steps>' sets the number of steps\n"
        "- '/q' to quit"
    )

    while (prompt := input(user_question)).startswith("/"):
        if prompt.startswith("/w"):
            if prompt.count(" ") != 1:
                nfo(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, width = prompt.split()
            options.width = 16 * (int(width) // 16)
            nfo(f"Setting resolution to {options.width} x {options.height} ({options.height * options.width / 1e6:.2f}MP)")
        elif prompt.startswith("/h"):
            if prompt.count(" ") != 1:
                nfo(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, height = prompt.split()
            options.height = 16 * (int(height) // 16)
            nfo(f"Setting resolution to {options.width} x {options.height} ({options.height * options.width / 1e6:.2f}MP)")
        elif prompt.startswith("/g"):
            if prompt.count(" ") != 1:
                nfo(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, guidance = prompt.split()
            options.guidance = float(guidance)
            nfo(f"Setting guidance to {options.guidance}")
        elif prompt.startswith("/s"):
            if prompt.count(" ") != 1:
                nfo(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, seed = prompt.split()
            options.seed = int(seed)
            nfo(f"Setting seed to {options.seed}")
        elif prompt.startswith("/n"):
            if prompt.count(" ") != 1:
                nfo(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, steps = prompt.split()
            options.num_steps = int(steps)
            nfo(f"Setting number of steps to {options.num_steps}")
        elif prompt.startswith("/q"):
            nfo("Quitting")
            return None
        else:
            if not prompt.startswith("/h"):
                nfo(f"Got invalid command '{prompt}'\n{usage}")
            nfo(usage)
    if prompt != "":
        options.prompt = prompt
    return options


def main(
    model_name: str = "flux.2-dev",
    single_eval: bool = False,
    prompt: str = "",
    width: int = 1360,
    height: int = 768,
    guidance: float = 4,
    seed: int | None = rng.next_seed(),
    num_steps: int = 50,
    device: torch.device = device,
    upsample_prompt: bool = False,
    loop: bool = False,
    offload: bool = False,
    compile: bool = False,
    input_images: list[str] | None = None,
):
    assert model_name.lower() in FLUX2_MODEL_INFO, f"{model_name} is not available, choose from {FLUX2_MODEL_INFO.keys()}"

    prompt_parts = prompt.split("|")
    if len(prompt_parts) == 1:
        prompt = prompt_parts[0]
        additional_prompts = None
    else:
        additional_prompts = prompt_parts[1:]
        prompt = prompt_parts[0]

    assert not ((additional_prompts is not None) and loop), "Do not provide additional prompts and set loop to True"

    mistral = load_mistral_small_embedder()
    model = load_flow_model(model_name, device=torch.device("cpu") if offload else device)

    is_compiled = False
    if compile and not offload:
        # Compile only if not offloading (compiled models can't be easily moved between devices)
        nfo("Compilation enabled.")
        model = torch.compile(model)  # type: ignore[assignment]
        is_compiled = True

    ae = load_ae(model_name)
    ae.eval()
    mistral.eval()

    # Validate user inputs
    opts = SamplingOptions(
        prompt=prompt,
        width=width,
        height=height,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed,
    )

    if loop:
        opts = parse_prompt(opts)

    while opts is not None:
        if opts.seed is None:
            opts.seed = rng.next_seed()
        else:
            rng.next_seed(opts.seed)
        # At this point, opts.seed is guaranteed to be an int
        assert opts.seed is not None, "Seed must be set"
        nfo(f"Generating with seed {rng.seed}: {opts.prompt}")

        x = get_noise(
            1,
            opts.height,
            opts.width,
            dtype=torch.bfloat16,
            seed=rng.seed,  # type: ignore
            device=device,
            version_2=True,
        )

        with torch.no_grad():
            if input_images:
                img_ctx = [Image.open(input_image) for input_image in input_images]
                ref_tokens, ref_ids = encode_image_refs(ae, img_ctx)  # type: ignore
            else:
                ref_tokens = None
                ref_ids = None

            if upsample_prompt:
                # Use local model for upsampling
                upsampled_prompts = mistral.upsample_prompt([prompt], img=[img_ctx] if img_ctx else None)  # type: ignore
                prompt = upsampled_prompts[0] if upsampled_prompts else prompt
            else:
                prompt = prompt

            ctx = mistral([prompt]).to(precision)  # type: ignore
            ctx, ctx_ids = batched_prc_txt(ctx)

            randn = get_noise(
                1,
                opts.height,
                opts.width,
                dtype=torch.bfloat16,
                seed=opts.seed,
                device=device,
                version_2=False,
            )

            x, x_ids = batched_prc_img(randn)

            timesteps = get_schedule(opts.num_steps, x.shape[1])
            state = DenoisingState(
                current_timestep=0,
                previous_timestep=None,
                current_sample=x,
                timestep_index=0,
                total_timesteps=len(timesteps),
                layer_dropout=None,
                guidance=opts.guidance,
                seed=rng.seed,
                width=opts.width,
                height=opts.height,
                prompt=opts.prompt,
                num_steps=opts.num_steps,
                deterministic=bool(torch.get_deterministic_debug_mode()),
            )

            if offload:
                if device.type == "cuda":
                    mistral = mistral.cpu()  # type: ignore
                else:
                    mistral = None
                    del mistral
                clear_cache()
                if is_compiled:
                    raise RuntimeError("Cannot use offload=True with compile=True. Compile after model is on device, or disable compilation when offloading.")
                model = model.to(device)  # type: ignore[attr-defined]
                if compile:
                    nfo("Compiling model on device.")
                    model = torch.compile(model, mode="max-autotune")  # type: ignore[assignment]
                    is_compiled = True

            x = denoise_interactive(
                model,  # type: ignore
                x,
                x_ids,
                ctx,
                ctx_ids,
                timesteps=timesteps,
                img_cond_seq=ref_tokens,
                img_cond_seq_ids=ref_ids,
                state=state,
                ae=ae,
            )
            if loop:
                nfo("-" * 80)
                opts = parse_prompt(opts)
            elif additional_prompts:
                next_prompt = additional_prompts.pop(0)
                opts.prompt = next_prompt
            else:
                opts = None


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
