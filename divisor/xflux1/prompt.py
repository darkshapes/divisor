# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->
# adapted XFlux code from https://github.com/TencentARC/FluxKits

import torch
from fire import Fire
from nnll.init_gpu import device, clear_cache
from nnll.console import nfo

from divisor.controller import DenoisingState, rng
from divisor.flux1.sampling import get_noise, get_schedule, prepare, SamplingOptions
from divisor.flux1.prompt import parse_prompt
from divisor.flux1.spec import configs, get_model_spec, CompatibilitySpec, ModelSpec, InitialParams
from divisor.flux1.loading import load_ae, load_clip, load_flow_model, load_t5
from divisor.xflux1.sampling import denoise


@torch.inference_mode()
def main(
    model_id: str = "flux1-dev",
    ae_id: str = "flux1-dev",
    width: int = 1360,
    height: int = 768,
    guidance: float = 2.5,
    seed: int | None = rng.next_seed(),
    prompt: str = "",
    # ('a photo of a forest with mist swirling around the tree trunks. The word "FLUX" is painted over it in big, red brush strokes with visible texture'),
    quantization: bool = False,
    tiny: bool = False,
    device: torch.device = device,
    num_steps: int | None = None,
    loop: bool = False,
    # 2.5, 4.0
    offload: bool = False,
    compile: bool = False,
    verbose: bool = False,
):
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

    model_id = f"model.dit.{model_id}".lower()
    ae_id = f"model.vae.{ae_id}".lower() if not tiny else f"model.taesd.{ae_id}".lower()
    if model_id not in configs or ae_id not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Got unknown model id: {model_id} or {ae_id}, chose from {available}")

    prompt_parts = prompt.split("|")
    if len(prompt_parts) == 1:
        prompt = prompt_parts[0]
        additional_prompts = None
    else:
        additional_prompts = prompt_parts[1:]
        prompt = prompt_parts[0]

    assert not ((additional_prompts is not None) and loop), "Do not provide additional prompts and set loop to True"

    compat_spec = {}
    if quantization:
        spec = get_model_spec(model_id, "fp8-sai")
        if spec is None:
            raise ValueError(f"Model {model_id} does not have a compatibility spec configured")
        if isinstance(spec, CompatibilitySpec):
            compat_spec = {
                "repo_id": spec.repo_id,
                "file_name": spec.file_name,
                "verbose": verbose,
            }
        elif isinstance(spec, ModelSpec):
            pass
    else:
        spec = get_model_spec(model_id)

    init = getattr(
        spec,
        "init",
        ValueError(f"Model {model_id} does not have initialization parameters (init) configured"),
    )
    assert isinstance(init, InitialParams), "init must be an InitialParams"

    height = 16 * (height // 16)
    width = 16 * (width // 16)

    t5 = load_t5(device, init.max_length or 512)
    clip = load_clip(device)
    # Load model to final device if not offloading (compile requires model to be on target device)
    model = load_flow_model(
        model_id,
        device=torch.device("cpu") if offload else device,
        **compat_spec,
    )

    is_compiled = False
    if compile and not offload:
        # Compile only if not offloading (compiled models can't be easily moved between devices)
        nfo("Compilation enabled.")
        model = torch.compile(model)  # type: ignore[assignment]
        is_compiled = True

    ae = load_ae(ae_id, device="cpu" if offload else device)

    # Validate user inputs
    opts = SamplingOptions(
        prompt=prompt,
        width=width,
        height=height,
        num_steps=num_steps or init.num_steps,
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
            device=device,
            dtype=torch.bfloat16,
            seed=rng.seed,  # type: ignore
        )

        # prepare input
        if offload:
            ae = ae.cpu()
            clear_cache()

            t5, clip = t5.to(device), clip.to(device)
        inp = prepare(t5, clip, x, prompt=opts.prompt)
        timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=init.shift)
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
        # offload TEs to CPU, load model to gpu
        if offload:
            t5, clip = t5.cpu(), clip.cpu()
            clear_cache()
            # Move model to device
            if is_compiled:
                # Can't move compiled models, so recompile after moving
                # This requires getting the underlying model, which is tricky
                # For now, just move and recompile
                # Note: This is a limitation - ideally compile after moving
                raise RuntimeError("Cannot use offload=True with compile=True. Compile after model is on device, or disable compilation when offloading.")
            # At this point, model is not compiled, so .to() is safe
            model = model.to(device)  # type: ignore[attr-defined]
            # Compile after moving to device if requested
            if compile:
                nfo("Compiling model on device.")

                model = torch.compile(model, mode="max-autotune")  # type: ignore[assignment]
                is_compiled = True

        # denoise initial noise
        x = denoise(
            model,  # type: ignore[arg-type]
            **inp,  # type: ignore
            timesteps=timesteps,
            state=state,
            ae=ae,
            device=device,
            t5=t5,
            clip=clip,
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

if __name__ == "__main__":
    Fire(main)
