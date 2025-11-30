# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->
# adapted BFL Flux code from https://github.com/black-forest-labs/flux2

import torch
from PIL import Image
from fire import Fire
from nnll.console import nfo
from nnll.init_gpu import device, clear_cache

from divisor.controller import rng
from divisor.noise import get_noise
from divisor.flux1.sampling import SamplingOptions, DenoisingState
from divisor.flux1.prompt import parse_prompt
from divisor.flux2.sampling import (
    batched_prc_img,
    batched_prc_txt,
    denoise_interactive,
    encode_image_refs,
    get_schedule,
)
from divisor.flux2.util import FLUX2_MODEL_INFO, load_ae, load_flow_model, load_mistral_small_embedder
from divisor.flux2 import precision


def main(
    model_id: str = "flux2-dev",
    ae_id: str = "flux2-dev",
    width: int = 1360,
    height: int = 768,
    guidance: float = 4,
    seed: int | None = rng.next_seed(),
    prompt: str = "",
    device: torch.device = device,
    num_steps: int = 50,
    upsample_prompt: bool = False,
    loop: bool = False,
    offload: bool = False,
    compile: bool = False,
    input_images: list[str] | None = None,
):
    model_id = f"model.dit.{model_id}".lower()
    ae_id = f"model.vae.{ae_id}".lower()
    assert model_id.lower() in FLUX2_MODEL_INFO, f"{model_id} is not available, choose from {FLUX2_MODEL_INFO.keys()}"

    prompt_parts = prompt.split("|")
    if len(prompt_parts) == 1:
        prompt = prompt_parts[0]
        additional_prompts = None
    else:
        additional_prompts = prompt_parts[1:]
        prompt = prompt_parts[0]

    assert not ((additional_prompts is not None) and loop), "Do not provide additional prompts and set loop to True"

    mistral = load_mistral_small_embedder()
    model = load_flow_model(model_id, device=torch.device("cpu") if offload else device)

    is_compiled = False
    if compile and not offload:
        # Compile only if not offloading (compiled models can't be easily moved between devices)
        nfo("Compilation enabled.")
        model = torch.compile(model)  # type: ignore[assignment]
        is_compiled = True

    ae = load_ae(ae_id)
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
                version_2=True,
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
    Fire(main)
