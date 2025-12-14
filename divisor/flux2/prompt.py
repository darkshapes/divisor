# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->
# adapted BFL Flux code from https://github.com/black-forest-labs/flux2

from dataclasses import replace

from PIL import Image
from fire import Fire
from nnll.console import nfo
from nnll.init_gpu import clear_cache, device
import torch

from divisor.contents import get_dtype
from divisor.controller import rng
from divisor.flux1.loading import load_flow_model, load_mistral_small_embedder
from divisor.flux1.prompt import parse_prompt
from divisor.flux2.sampling import (
    batched_prc_img,
    batched_prc_txt,
    denoise_interactive,
    encode_image_refs,
    get_schedule,
)
from divisor.noise import get_noise
from divisor.spec import flux_configs, get_model_spec, ModelSpec
from divisor.state import DenoiseSettings, DenoisingState


def main(
    mir_id: str = "model.dit.flux2-dev",
    ae_id: str = "model.vae.flux2-dev",
    width: int = 1360,
    height: int = 768,
    guidance: float = 4,
    seed: int = rng.next_seed(),
    prompt: str = "",
    quantization: bool = False,
    device: torch.device = device,
    num_steps: int = 50,
    upsample_prompt: bool = False,
    loop: bool = False,
    offload: bool = False,
    compile: bool = False,
    verbose: bool = False,
    input_images: list[str] | None = None,
) -> None:
    """Sample the flux model. Either interactively (set `--loop`) or run for a single image.\n
    :param name: Name of the model to load
    :param height: height of the sample in pixels (should be a multiple of 16)
    :param width: width of the sample in pixels (should be a multiple of 16)
    :param seed: Set a seed for sampling
    :param output_name: where to save the output image, `{idx}` will be replaced by the index of the sample
    :param prompt: Prompt used for sampling
    :param device: Pytorch device
    :param num_steps: number of sampling steps (default 4 for schnell, 28 for guidance distilled)
    :param loop: start an interactive session and sample multiple times
    :param guidance: guidance value used for guidance distillation
    """

    precision = get_dtype(device)
    prompt_parts = prompt.split("|")
    if len(prompt_parts) == 1:
        prompt = prompt_parts[0]
        additional_prompts = None
    else:
        additional_prompts = prompt_parts[1:]
        prompt = prompt_parts[0]

    mistral = load_mistral_small_embedder()
    if quantization:
        mir_id += ":@fp8-sai"
    model_spec: ModelSpec = get_model_spec(mir_id, flux_configs)
    ae_spec = get_model_spec(ae_id, flux_configs)
    model = load_flow_model(
        model_spec,
        device=torch.device("cpu") if offload else device,
        verbose=verbose,
    )

    is_compiled = False
    if compile and not offload:  # Compiled models can't be easily moved between devices
        nfo("Compilation enabled.")
        model = torch.compile(model)  # type: ignore[assignment]
        is_compiled = True

    ae = load_flow_model(ae_spec, device=torch.device("cpu") if offload else device)
    ae.eval()
    mistral.eval()

    state = DenoisingState.from_cli_args(
        prompt=prompt,
        width=width,
        height=height,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed,
    )

    if loop:
        state = parse_prompt(state)

    while state is not None:
        if state.seed is None:
            seed = rng.next_seed()
            state = replace(state, seed=seed)
        else:
            rng.next_seed(state.seed)
        # At this point, state.seed is guaranteed to be an int
        assert state.seed is not None, "Seed must be set"
        assert state.width is not None and state.height is not None, "Width and height must be set"
        assert state.num_steps is not None, "num_steps must be set"
        assert state.prompt is not None, "Prompt must be set"
        nfo(f"Generating with seed {rng.seed}: {state.prompt}")

        x = get_noise(
            1,
            state.height,
            state.width,
            dtype=precision,
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
                upsampled_prompts = mistral.upsample_prompt([state.prompt], img=[img_ctx] if img_ctx else None)  # type: ignore
                prompt = upsampled_prompts[0] if upsampled_prompts else state.prompt
                state = replace(state, prompt=prompt)
            else:
                prompt = state.prompt

            ctx = mistral([prompt]).to(precision)  # type: ignore
            ctx, ctx_ids = batched_prc_txt(ctx)

            randn = get_noise(
                1,
                state.height,  # type: ignore
                state.width,  # type: ignore
                dtype=torch.bfloat16,
                seed=state.seed,  # type: ignore
                device=device,
                version_2=True,
            )

            x, x_ids = batched_prc_img(randn)

            timesteps = get_schedule(state.num_steps, x.shape[1])  # type: ignore
            # Update state with runtime information
            state = state.with_runtime_state(
                current_timestep=0.0,
                current_sample=x,
                timestep_index=0,
                total_timesteps=len(timesteps),
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
                DenoiseSettings(
                    img=x,
                    img_ids=x_ids,
                    txt=ctx,
                    txt_ids=ctx_ids,
                    timesteps=timesteps,
                    img_cond_seq=ref_tokens,
                    img_cond_seq_ids=ref_ids,
                    state=state,
                    ae=ae,
                ),
            )
            if loop:
                nfo("-" * 80)
                state = parse_prompt(state)
            elif additional_prompts:
                next_prompt = additional_prompts.pop(0)
                state = replace(state, prompt=next_prompt)
            else:
                state = None


if __name__ == "__main__":
    Fire(main)
