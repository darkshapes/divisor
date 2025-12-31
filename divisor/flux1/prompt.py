# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->
# adapted BFL Flux code from https://github.com/black-forest-labs/flux

from dataclasses import replace

from fire import Fire
from nnll.console import nfo
from divisor.registry import device, gfx
import torch

from divisor.controller import rng
from divisor.flux1.loading import load_ae, load_clip, load_flow_model, load_t5
from divisor.flux1.sampling import denoise, get_schedule, prepare
from divisor.noise import prepare_4d_noise_for_3d_model
from divisor.spec import InitialParamsFlux, ModelSpec, flux_configs, get_model_spec
from divisor.state import MenuState


def parse_prompt(state: MenuState) -> MenuState | None:
    """Parse user input and update input display.

    :param state: Current state to update
    :returns: Updated input display or None to quit"""
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
            width = 16 * (int(width) // 16)
            state = replace(state, width=width)
            nfo(f"Setting resolution to {state.width} x {state.height} ({state.height * state.width / 1e6:.2f}MP)")
        elif prompt.startswith("/h"):
            if prompt.count(" ") != 1:
                nfo(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, height = prompt.split()
            height = 16 * (int(height) // 16)
            state = replace(state, height=height)
            nfo(f"Setting resolution to {state.width} x {state.height} ({state.height * state.width / 1e6:.2f}MP)")
        elif prompt.startswith("/g"):
            if prompt.count(" ") != 1:
                nfo(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, guidance = prompt.split()
            state = replace(state, guidance=float(guidance))
            nfo(f"Setting guidance to {state.guidance}")
        elif prompt.startswith("/s"):
            if prompt.count(" ") != 1:
                nfo(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, seed = prompt.split()
            state = replace(state, seed=int(seed))
            nfo(f"Setting seed to {state.seed}")
        elif prompt.startswith("/n"):
            if prompt.count(" ") != 1:
                nfo(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, steps = prompt.split()
            state = replace(state, num_steps=int(steps))
            nfo(f"Setting number of steps to {state.num_steps}")
        elif prompt.startswith("/q"):
            nfo("Quitting")
            return None
        else:
            if not prompt.startswith("/h"):
                nfo(f"Got invalid command '{prompt}'\n{usage}")
            nfo(usage)
    if prompt != "":
        state = replace(state, prompt=prompt)
    return state


@torch.inference_mode()
def main(
    mir_id: str = "model.dit.flux1-dev",
    ae_id: str = "model.vae.flux1-dev",
    width: int = 1360,
    height: int = 768,
    guidance: float = 4.0,
    seed: int = rng.next_seed(),
    prompt: str = "",
    quantization: bool = False,
    device: torch.device = device,
    num_steps: int = 50,
    loop: bool = False,
    offload: bool = False,
    compile: bool = False,
    verbose: bool = False,
    input_images: list[str] | None = None,
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
    :param guidance: guidance value used for guidance distillation"""

    if quantization:
        mir_id += ":*@fp8-sai"
    model_spec: ModelSpec = get_model_spec(mir_id, flux_configs)
    ae_spec: ModelSpec = get_model_spec(ae_id, flux_configs)
    init: InitialParamsFlux = model_spec.init

    prompt_parts = prompt.split("|")
    if len(prompt_parts) == 1:
        prompt = prompt_parts[0]
        additional_prompts = None
    else:
        additional_prompts = prompt_parts[1:]
        prompt = prompt_parts[0]

    assert not ((additional_prompts is not None) and loop), "Do not provide additional prompts and set loop to True"

    height = 16 * (height // 16)
    width = 16 * (width // 16)

    t5 = load_t5(device, init.max_length or 512)
    clip = load_clip(device)
    model = load_flow_model(
        model_spec,
        device=torch.device("cpu") if offload else device,
        verbose=verbose,
    )

    is_compiled = False
    if compile and not offload:  # (compiled models should be on target device)
        nfo("Compilation enabled.")
        model = torch.compile(model)  # type: ignore[assignment]
        is_compiled = True

    ae = load_ae(ae_spec, device=torch.device("cpu") if offload else device)

    state = MenuState.from_cli_args(
        prompt=prompt,
        width=width,
        height=height,
        num_steps=num_steps or init.num_steps,
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
        assert state.seed is not None, "Seed must be set"
        assert state.width is not None and state.height is not None, "Width and height must be set"
        assert state.num_steps is not None, "num_steps must be set"
        assert state.prompt is not None, "Prompt must be set"
        nfo(f"Generating with seed {rng.seed}: {state.prompt}")

        from divisor.noise import get_noise

        x_4d = get_noise(
            1,
            state.height,
            state.width,
            device=device,
            dtype=torch.bfloat16,
            seed=rng.seed,  # type: ignore
        )

        if offload:
            ae = ae.cpu()
            gfx.empty_cache()

            t5, clip = t5.to(device), clip.to(device)
        x_3d = prepare_4d_noise_for_3d_model(
            height=state.height,  # type: ignore
            width=state.width,  # type: ignore
            seed=rng.seed,  # type: ignore
            t5=t5,
            clip=clip,
            prompt=state.prompt,
            device=device,
            dtype=torch.bfloat16,
        )
        inp = prepare(t5, clip, x_4d, prompt=state.prompt)
        timesteps = get_schedule(state.num_steps, inp["img"].shape[1], shift=init.shift)
        # Update state with runtime information (use 3D format for current_sample)
        state = state.with_runtime_state(
            current_timestep=0.0,
            current_sample=x_3d,
            timestep_index=0,
            total_timesteps=len(timesteps),
        )
        # offload TEs to CPU, load model to gpu
        if offload:
            t5, clip = t5.cpu(), clip.cpu()
            gfx.empty_cache()
            if is_compiled:
                raise RuntimeError("Can't move compiled models. Compile on target device, or disable compilation when offloading.")
            model = model.to(device)  # type: ignore[attr-defined]
            if compile:
                nfo("Compiling model on device.")
                model = torch.compile(model, mode="max-autotune")  # type: ignore[assignment]
                is_compiled = True

        from divisor.state import InferenceState

        settings = InferenceState(
            img=inp["img"],
            img_ids=inp["img_ids"],
            txt=inp["txt"],
            txt_ids=inp["txt_ids"],
            vec=inp["vec"],
            state=state,
            ae=ae,
            timesteps=timesteps,
            device=device,
            t5=t5,
            clip=clip,
        )
        x_4d = denoise(
            model,  # type: ignore[arg-type]
            settings=settings,
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
