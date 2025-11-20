# SPDX-License-Identifier:Apache-2.0
# original BFL Flux code from https://github.com/black-forest-labs/flux
import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from einops import rearrange, repeat
from torch import Tensor

from divisor.controller import ManualTimestepController
from divisor.flux_modules.autoencoder import AutoEncoder
from divisor.flux_modules.model import Flux
from divisor.flux_modules.text_embedder import HFEmbedder
from divisor.flux_modules.util import save_image_simple
from divisor.hardware import seed_planter


@dataclass
class SamplingOptions:
    prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None


def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    return torch.randn(
        num_samples,
        16,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        dtype=dtype,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    ).to(device)


def prepare(
    t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: str | list[str]
) -> dict[str, Tensor]:
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


@torch.inference_mode()
def denoise(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    # extra img tokens (channel-wise)
    img_cond: Tensor | None = None,
    # extra img tokens (sequence-wise)
    img_cond_seq: Tensor | None = None,
    img_cond_seq_ids: Tensor | None = None,
    ae: AutoEncoder | None = None,
    torch_device: torch.device = torch.device("cuda"),
    opts: SamplingOptions | None = None,
    initial_layer_dropout: Optional[list[int]] = None,
):
    """
    Denoise using Flux model with optional ManualTimestepController.

    Args:
        model: Flux model instance
        img: Initial noisy image tensor
        img_ids: Image position IDs tensor
        txt: Text embeddings tensor
        txt_ids: Text position IDs tensor
        vec: CLIP embeddings vector tensor
        timesteps: List of timestep values
        guidance: Guidance (CFG) value
        img_cond: Optional channel-wise image conditioning
        img_cond_seq: Optional sequence-wise image conditioning
        img_cond_seq_ids: Optional sequence-wise image conditioning IDs
        ae: AutoEncoder for decoding previews
        torch_device: PyTorch device
        opts: Sampling options
        initial_layer_dropout: Initial layer dropout configuration
        use_controller: Whether to use ManualTimestepController (default: True)
    """
    # this is ignored for schnell
    guidance_vec = (
        torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype) * 0.0
    ) * 0.0

    # Store layer_dropout in a mutable container for closure
    current_layer_dropout = [initial_layer_dropout]

    def denoise_step_fn(
        sample: Tensor, t_curr: float, t_prev: float, guidance_val: float
    ) -> Tensor:
        """Single denoising step function for the controller."""
        t_vec = torch.full(
            (sample.shape[0],), t_curr, dtype=sample.dtype, device=sample.device
        )
        img_input = sample
        img_input_ids = img_ids

        if img_cond is not None:
            img_input = torch.cat((sample, img_cond), dim=-1)
        if img_cond_seq is not None:
            assert img_cond_seq_ids is not None, (
                "You need to provide either both or neither of the sequence conditioning"
            )
            img_input = torch.cat((img_input, img_cond_seq), dim=1)
            img_input_ids = torch.cat((img_input_ids, img_cond_seq_ids), dim=1)

        # Get current layer_dropout from mutable container
        layer_dropouts = current_layer_dropout[0]

        pred = model(
            img=img_input,
            img_ids=img_input_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            layer_dropouts=layer_dropouts,
        )

        if img_input_ids is not None:
            pred = pred[:, : sample.shape[1]]

        dt = t_prev - t_curr
        x0 = sample - t_curr * pred

        return (1 - (t_curr + dt)) * x0 + (t_curr + dt) * torch.randn_like(x0)

    # if use_controller:
    # Create controller
    controller = ManualTimestepController(
        timesteps=timesteps,
        initial_sample=img,
        denoise_step_fn=denoise_step_fn,
        initial_guidance=guidance,
    )
    controller.set_layer_dropout(initial_layer_dropout)

    # Initialize current seed (use from opts if available, otherwise generate random)
    if opts is not None and opts.seed is not None:
        current_seed = int(opts.seed)
    else:
        current_seed = int(torch.randint(0, 2**31, (1,)).item())
    seed_planter(current_seed)

    # Interactive loop
    while not controller.is_complete:
        state = controller.current_state
        step = state.timestep_index

        print(
            f"\nStep {step}/{state.total_timesteps} @ noise level {state.current_timestep:.4f}"
        )
        print(f"Guidance: {state.guidance:.2f}")
        print(f"Seed: {current_seed}")
        if state.layer_dropout:
            print(f"Layer dropout: {state.layer_dropout}")
        else:
            print("Layer dropout: None")

        # Generate preview
        if ae is not None and opts is not None:
            # Prepare input for preview
            preview_img_input = state.current_sample
            preview_img_input_ids = img_ids
            if img_cond is not None:
                preview_img_input = torch.cat([state.current_sample, img_cond], dim=-1)
            if img_cond_seq is not None and img_cond_seq_ids is not None:
                preview_img_input = torch.cat([preview_img_input, img_cond_seq], dim=1)
                preview_img_input_ids = torch.cat(
                    [preview_img_input_ids, img_cond_seq_ids], dim=1
                )

            t_vec_preview = torch.full(
                (state.current_sample.shape[0],),
                state.current_timestep,
                dtype=state.current_sample.dtype,
                device=state.current_sample.device,
            )

            # TODO: we can reuse the prediction from denoise_step_fn to avoid two model calls per step.

            pred_preview = model(
                img=preview_img_input,
                img_ids=preview_img_input_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec_preview,
                guidance=guidance_vec,
                layer_dropouts=state.layer_dropout,
            )

            if preview_img_input_ids is not None:
                pred_preview = pred_preview[:, : state.current_sample.shape[1]]

            intermediate = state.current_sample - state.current_timestep * pred_preview
            intermediate = unpack(intermediate.float(), opts.height, opts.width)
            with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
                intermediate_image = ae.decode(intermediate)
                save_image_simple("preview.jpg", intermediate_image)

        # User input
        choice = (
            input(
                "Choose an action: (a)dvance, (g)uidance, (l)ayer_dropout, (s)eed, (e)dit: "
            )
            .lower()
            .strip()
        )

        if choice == "a":
            controller.step()
        elif choice == "g":
            try:
                new_guidance = float(
                    input(f"Enter new guidance value (current: {state.guidance:.2f}): ")
                )
                controller.set_guidance(new_guidance)
                print(f"Guidance set to {new_guidance:.2f}")
            except ValueError:
                print("Invalid guidance value, keeping current value")
        elif choice == "l":
            try:
                dropout_input = input(
                    "Enter layer indices to drop (comma-separated, or 'none' to clear): "
                ).strip()
                if dropout_input.lower() == "none" or dropout_input == "":
                    controller.set_layer_dropout(None)
                    current_layer_dropout[0] = None
                    print("Layer dropout cleared")
                else:
                    layer_indices = [int(x.strip()) for x in dropout_input.split(",")]
                    controller.set_layer_dropout(layer_indices)
                    current_layer_dropout[0] = layer_indices
                    print(f"Layer dropout set to: {layer_indices}")
            except ValueError:
                print("Invalid layer indices, keeping current value")
        elif choice == "s":
            try:
                seed_input = input(
                    f"Enter new seed number (current: {current_seed}, or press Enter for random): "
                ).strip()
                if seed_input == "":
                    # Generate random seed
                    current_seed = int(torch.randint(0, 2**31, (1,)).item())
                else:
                    current_seed = int(seed_input)
                seed_planter(current_seed)
                print(f"Seed set to: {current_seed}")
            except ValueError:
                print("Invalid seed value, keeping current seed")
        elif choice == "e":
            print("Entering edit mode (use c/cont to exit)...")
            breakpoint()
        else:
            print("Invalid choice, please try again")

    return controller.current_sample


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )
