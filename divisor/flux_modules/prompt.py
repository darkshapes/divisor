# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->
# adapted BFL Flux code from https://github.com/black-forest-labs/flux

import torch
from fire import Fire
from nnll.init_gpu import clear_cache, device
from nnll.console import nfo

from divisor.controller import DenoisingState, rng
from divisor.flux_modules.sampling import (
    SamplingOptions,
    denoise,
    get_noise,
    get_schedule,
    prepare,
)
from divisor.flux_modules.util import (
    configs,
    get_model_spec,
    get_compatibility_spec,
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
)


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


@torch.inference_mode()
def main(
    model_id: str = "model.dit.flux1-dev",
    ae_id: str = "model.vae.flux-dev",
    width: int = 1360,
    height: int = 768,
    guidance: float = 2.5,
    seed: int | None = rng.next_seed(),
    prompt: str = (""),
    quantization: bool = False,
    # ('a photo of a forest with mist swirling around the tree trunks. The word "FLUX" is painted over it in big, red brush strokes with visible texture'),
    device: torch.device = device,
    num_steps: int | None = None,
    loop: bool = False,
    # 2.5, 4.0
    offload: bool = False,
    compile: bool = False,
    verbose: bool = False,
):
    """Sample the flux model. Either interactively (set `--loop`) or run for a single image.

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

    if model_id not in configs or ae_id not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Got unknown model id: {model_id} or {ae_id}, chose from {available}")

    spec = get_model_spec(model_id)
    init = spec.init

    # Ensure init params are available
    if init is None:
        raise ValueError(f"Model {model_id} does not have initialization parameters (init) configured")

    prompt_parts = prompt.split("|")
    if len(prompt_parts) == 1:
        prompt = prompt_parts[0]
        additional_prompts = None
    else:
        additional_prompts = prompt_parts[1:]
        prompt = prompt_parts[0]

    assert not ((additional_prompts is not None) and loop), "Do not provide additional prompts and set loop to True"

    override = {}
    if quantization:
        compat_spec = get_compatibility_spec(model_id, "fp8-sai")
        if compat_spec is None:
            raise ValueError(f"Model {model_id} does not have a compatibility spec configured")
        # Use base ModelSpec but substitute repo_id and file_name from compatibility spec
        override = {
            "repo_id": compat_spec.repo_id,
            "file_name": compat_spec.file_name,
            "verbose": verbose,
        }
    # allow for packing and conversion to latent space
    height = 16 * (height // 16)
    width = 16 * (width // 16)

    t5 = load_t5(device, init.max_length or 512)
    clip = load_clip(device)
    # Load model to final device if not offloading (compile requires model to be on target device)
    model = load_flow_model(
        model_id,
        device="cpu" if offload else device,
        **override,
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
        nfo(f"Generating with seed {rng.seed}:\n{opts.prompt}")

        x = get_noise(
            1,
            opts.height,
            opts.width,
            device=device,
            dtype=torch.bfloat16,
            seed=rng.seed,  # type: ignore
        )

        # prepare input
        opts.seed = None
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
            torch_device=device,
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
