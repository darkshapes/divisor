# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->
# adapted BFL Flux code from https://github.com/black-forest-labs/flux

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
from divisor.flux_modules.util import PREFERRED_KONTEXT_RESOLUTIONS, save_image_simple
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


def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: str | list[str]) -> dict[str, Tensor]:
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


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15) -> Callable[[float], float]:
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
    timesteps: list[float],  # sampling parameters
    guidance: float = 4.0,
    img_cond: Tensor | None = None,  # extra img tokens (channel-wise)
    img_cond_seq: Tensor | None = None,  # extra img tokens (sequence-wise)
    img_cond_seq_ids: Tensor | None = None,
    ae: AutoEncoder | None = None,
    torch_device: torch.device = torch.device("cuda"),
    opts: SamplingOptions | None = None,
    initial_layer_dropout: Optional[list[int]] = None,
):
    """Denoise using Flux model with optional ManualTimestepController.

    :param model: Flux model instance
    :param img: Initial noisy image tensor
    :param img_ids: Image position IDs tensor
    :param txt: Text embeddings tensor
    :param txt_ids: Text position IDs tensor
    :param vec: CLIP embeddings vector tensor
    :param timesteps: List of timestep values
    :param guidance: Guidance (CFG) value
    :param img_cond: Optional channel-wise image conditioning
    :param img_cond_seq: Optional sequence-wise image conditioning
    :param img_cond_seq_ids: Optional sequence-wise image conditioning IDs
    :param ae: AutoEncoder for decoding previews
    :param torch_device: PyTorch device
    :param opts: Sampling options
    :param initial_layer_dropout: Initial layer dropout configuration
    """
    # this is ignored for schnell
    guidance_vec = (torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype) * 0.0) * 0.0
    current_layer_dropout = [initial_layer_dropout]
    use_previous_as_mask = [False]
    previous_step_tensor: list[Optional[Tensor]] = [None]  # Store previous step's tensor for masking
    cached_prediction: list[Optional[Tensor]] = [None]  # Cache prediction to avoid duplicate model calls
    cached_prediction_state: list[Optional[dict]] = [None]  # Cache state when prediction was generated
    vae_shift_offset: list[float] = [0.0]  # Offset to add to shift_factor in autoencoder decode
    vae_scale_offset: list[float] = [0.0]  # Offset to add to scale_factor in autoencoder decode

    def get_prediction(sample: Tensor, t_curr: float, guidance_val: float, layer_dropouts_val: Optional[list[int]]) -> Tensor:
        """Generate model prediction, reusing cached prediction if state hasn't changed.

        :param sample: Current sample tensor
        :param t_curr: Current timestep
        :param guidance_val: Guidance value
        :param layer_dropouts_val: Layer dropout configuration
        :returns: Model prediction
        """
        # Create a simple hash of the sample tensor for cache key (using first few values)
        # This is faster than hashing the entire tensor but should be sufficient for cache invalidation
        sample_hash = hash((sample.shape, float(sample[0, 0, 0].item()) if sample.numel() > 0 else 0))

        # Check if we can reuse cached prediction
        current_state = {
            "sample_hash": sample_hash,
            "t_curr": t_curr,
            "guidance": guidance_val,
            "layer_dropout": layer_dropouts_val,
        }

        if (
            cached_prediction[0] is not None
            and cached_prediction_state[0] is not None
            and cached_prediction_state[0]["sample_hash"] == current_state["sample_hash"]
            and cached_prediction_state[0]["t_curr"] == current_state["t_curr"]
            and cached_prediction_state[0]["guidance"] == current_state["guidance"]
            and cached_prediction_state[0]["layer_dropout"] == current_state["layer_dropout"]
        ):
            return cached_prediction[0]

        # Generate new prediction
        t_vec = torch.full((sample.shape[0],), t_curr, dtype=sample.dtype, device=sample.device)
        img_input = sample
        img_input_ids = img_ids

        if img_cond is not None:
            img_input = torch.cat((sample, img_cond), dim=-1)
        if img_cond_seq is not None:
            assert img_cond_seq_ids is not None, "You need to provide either both or neither of the sequence conditioning"
            img_input = torch.cat((img_input, img_cond_seq), dim=1)
            img_input_ids = torch.cat((img_input_ids, img_cond_seq_ids), dim=1)

        pred = model(
            img=img_input,
            img_ids=img_input_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            layer_dropouts=layer_dropouts_val,
        )

        if img_input_ids is not None:
            pred = pred[:, : sample.shape[1]]

        # Cache the prediction
        cached_prediction[0] = pred
        cached_prediction_state[0] = current_state

        return pred

    def denoise_step_fn(sample: Tensor, t_curr: float, t_prev: float, guidance_val: float) -> Tensor:
        """Single denoising step function for the controller."""

        # Get current layer_dropout from mutable container
        layer_dropouts = current_layer_dropout[0]

        # Get prediction (may reuse cached)
        pred = get_prediction(sample, t_curr, guidance_val, layer_dropouts)

        # Apply mask from previous step if enabled
        if use_previous_as_mask[0] and previous_step_tensor[0] is not None:
            # Normalize previous tensor to [0, 1] range for masking
            prev_tensor = previous_step_tensor[0]
            # Ensure shapes match
            if prev_tensor.shape == pred.shape:
                # Normalize previous tensor values to create a mask
                prev_min = prev_tensor.min()
                prev_max = prev_tensor.max()
                if prev_max > prev_min:
                    mask = (prev_tensor - prev_min) / (prev_max - prev_min)
                else:
                    mask = torch.ones_like(prev_tensor)
                # Apply mask to prediction: mask controls how much the prediction affects the result
                # Higher mask values = more effect from prediction, lower = less effect
                pred = pred * mask

        dt = t_prev - t_curr
        x0 = sample - t_curr * pred

        result = (1 - (t_curr + dt)) * x0 + (t_curr + dt) * torch.randn_like(x0)

        # Store current sample as previous for next step
        previous_step_tensor[0] = sample.clone()

        return result

    controller = ManualTimestepController(
        timesteps=timesteps,
        initial_sample=img,
        denoise_step_fn=denoise_step_fn,
        initial_guidance=guidance,
    )
    controller.set_layer_dropout(initial_layer_dropout)

    if opts is not None:
        controller.set_resolution(opts.width, opts.height)

    # Initialize current seed (use from opts if available, otherwise generate random)
    if opts is not None and opts.seed is not None:
        current_seed = int(opts.seed)
    else:
        current_seed = int(torch.randint(0, 2**31, (1,)).item())

    # Interactive loop
    while not controller.is_complete:
        state = controller.current_state
        step = state.timestep_index
        choice = input(": (l)ayer_dropout, (r)esolution, (s)eed, (b)uffer_mask, (g)uidance, (v)ae_shift, vae_s(c)ale, (e)dit, or advance with Enter: ").lower().strip()

        if choice == "":
            controller.step()
        elif choice == "g":
            try:
                new_guidance = float(input(f"Enter new guidance value (current: {state.guidance:.2f}): "))
                controller.set_guidance(new_guidance)
                # Invalidate cache since guidance changed
                cached_prediction[0] = None
                cached_prediction_state[0] = None
                print(f"Guidance set to {new_guidance:.2f}")
            except ValueError:
                print("Invalid guidance value, keeping current value")
        elif choice == "l":
            try:
                dropout_input = input("Enter layer indices to drop (comma-separated, or 'none' to clear): ").strip()
                if dropout_input.lower() == "none" or dropout_input == "":
                    layer_indices = None
                else:
                    layer_indices = [int(x.strip()) for x in dropout_input.split(",")]
                # Update immediately so it takes effect for the preview
                controller.set_layer_dropout(layer_indices)
                current_layer_dropout[0] = layer_indices
                # Invalidate cache since layer_dropout changed
                cached_prediction[0] = None
                cached_prediction_state[0] = None
                if layer_indices is None:
                    print("Layer dropout cleared")
                else:
                    print(f"Layer dropout set to: {layer_indices}")
            except ValueError:
                print("Invalid layer indices, keeping current value")
        elif choice == "r":
            try:
                print("\nPreferred resolutions:")
                for i, (w, h) in enumerate(PREFERRED_KONTEXT_RESOLUTIONS):
                    current_marker = ""
                    if state.width == w and state.height == h:
                        current_marker = " (current)"
                    print(f"  {i}: {w}x{h}{current_marker}")
                resolution_input = input(f"\nEnter resolution index (0-{len(PREFERRED_KONTEXT_RESOLUTIONS) - 1}) or 'custom' for custom: ").strip()
                if resolution_input.lower() == "custom":
                    width_input = input("Enter width: ").strip()
                    height_input = input("Enter height: ").strip()
                    new_width = int(width_input)
                    new_height = int(height_input)
                    controller.set_resolution(new_width, new_height)
                    print(f"Resolution set to: {new_width}x{new_height}")
                else:
                    resolution_idx = int(resolution_input)
                    if 0 <= resolution_idx < len(PREFERRED_KONTEXT_RESOLUTIONS):
                        new_width, new_height = PREFERRED_KONTEXT_RESOLUTIONS[resolution_idx]
                        controller.set_resolution(new_width, new_height)
                        print(f"Resolution set to: {new_width}x{new_height}")
                    else:
                        print("Invalid resolution index, keeping current value")
            except (ValueError, IndexError):
                print("Invalid resolution input, keeping current value")
        elif choice == "s":
            try:
                seed_input = input(f"Enter new seed number (current: {current_seed}, or press Enter for random): ").strip()
                if seed_input == "":
                    # Generate random seed
                    current_seed = int(torch.randint(0, 2**31, (1,)).item())
                else:
                    current_seed = int(seed_input)
                seed_planter(current_seed)
                print(f"Seed set to: {current_seed}")
            except ValueError:
                print("Invalid seed value, keeping current seed")
        elif choice == "b":
            try:
                use_previous_as_mask[0] = not use_previous_as_mask[0]
                print(f"Previous step tensor mask: {'ENABLED' if use_previous_as_mask[0] else 'DISABLED'}")
            except Exception as e:
                print(f"Error setting buffer options: {e}")
        elif choice == "v":
            try:
                if ae is None:
                    print("AutoEncoder not available, cannot set VAE shift")
                else:
                    shift_input = input(f"Enter VAE shift offset (current: {vae_shift_offset[0]:.4f}, or press Enter to reset to 0.0): ").strip()
                    if shift_input == "":
                        vae_shift_offset[0] = 0.0
                        print("VAE shift offset reset to 0.0")
                    else:
                        vae_shift_offset[0] = float(shift_input)
                        print(f"VAE shift offset set to: {vae_shift_offset[0]:.4f}")
            except ValueError:
                print("Invalid VAE shift value, keeping current value")
        elif choice == "c":
            try:
                if ae is None:
                    print("AutoEncoder not available, cannot set VAE scale")
                else:
                    scale_input = input(f"Enter VAE scale offset (current: {vae_scale_offset[0]:.4f}, or press Enter to reset to 0.0): ").strip()
                    if scale_input == "":
                        vae_scale_offset[0] = 0.0
                        print("VAE scale offset reset to 0.0")
                    else:
                        vae_scale_offset[0] = float(scale_input)
                        print(f"VAE scale offset set to: {vae_scale_offset[0]:.4f}")
            except ValueError:
                print("Invalid VAE scale value, keeping current value")
        elif choice == "e":
            print("Entering edit mode (use c/cont to exit)...")
            breakpoint()
        else:
            print("Invalid choice, please try again")

        print(f"\nStep {step}/{state.total_timesteps} @ noise level {state.current_timestep:.4f}")
        print(f"Guidance: {state.guidance:.2f}")
        print(f"Seed: {current_seed}")
        if state.width is not None and state.height is not None:
            print(f"Resolution: {state.width}x{state.height}")
        if state.layer_dropout:
            print(f"Layer dropout: {state.layer_dropout}")
        else:
            print("Layer dropout: None")
        print(f"Buffer mask: {'ON' if use_previous_as_mask[0] else 'OFF'}")
        if ae is not None:
            print(f"VAE shift offset: {vae_shift_offset[0]:.4f}")
            print(f"VAE scale offset: {vae_scale_offset[0]:.4f}")

        # Generate preview
        seed_planter(current_seed)
        if ae is not None and opts is not None:
            # Reuse cached prediction if available, otherwise generate it
            # This will be cached and reused in denoise_step_fn when advancing
            pred_preview = get_prediction(state.current_sample, state.current_timestep, state.guidance, current_layer_dropout[0])

            intermediate = state.current_sample - state.current_timestep * pred_preview
            intermediate = unpack(intermediate.float(), opts.height, opts.width)
            with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
                # Apply VAE shift offset by manually adjusting the decode operation
                if vae_shift_offset[0] != 0.0:
                    # Decode with offset: z = z / scale_factor + (shift_factor + offset)
                    z_adjusted = intermediate / (ae.scale_factor + vae_scale_offset[0]) + (ae.shift_factor + vae_shift_offset[0])
                    intermediate_image = ae.decoder(z_adjusted)
                else:
                    intermediate_image = ae.decode(intermediate)
                save_image_simple("preview.webp", intermediate_image)
                controller.store_state_in_chain(current_seed=current_seed)
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
