# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->
# adapted BFL Flux code from https://github.com/black-forest-labs/flux

import math
from dataclasses import dataclass
from typing import Callable, Optional
import gc
import torch
import time
from einops import rearrange, repeat
from torch import Tensor
from nnll.constants import ExtensionType
from nnll.save_generation import name_save_file_as, save_with_hyperchain
from nnll.console import nfo
from divisor.controller import ManualTimestepController, DenoisingState
from divisor.hardware import clear_cache, device, sync_torch
from divisor.flux_modules.autoencoder import AutoEncoder
from divisor.flux_modules.model import Flux
from divisor.flux_modules.text_embedder import HFEmbedder
from divisor.flux_modules.util import PREFERRED_KONTEXT_RESOLUTIONS

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
    del t5, clip
    clear_cache()
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
    state: DenoisingState,
    timesteps: list[float],  # sampling parameters
    img_cond: Tensor | None = None,  # extra img tokens (channel-wise)
    img_cond_seq: Tensor | None = None,  # extra img tokens (sequence-wise)
    img_cond_seq_ids: Tensor | None = None,
    ae: AutoEncoder | None = None,
    torch_device: torch.device = device,
    initial_layer_dropout: Optional[list[int]] = None,
):
    """Denoise using Flux model with optional ManualTimestepController.

    :param model: Flux model instance
    :param img: Initial noisy image tensor
    :param img_ids: Image position IDs tensor
    :param txt: Text embeddings tensor
    :param txt_ids: Text position IDs tensor
    :param vec: CLIP embeddings vector tensor
    :param state: DenoisingState containing current state parameters (guidance, width, height, seed, prompt, num_steps, etc.)
    :param timesteps: List of timestep values
    :param img_cond: Optional channel-wise image conditioning
    :param img_cond_seq: Optional sequence-wise image conditioning
    :param img_cond_seq_ids: Optional sequence-wise image conditioning IDs
    :param ae: AutoEncoder for decoding previews
    :param torch_device: PyTorch device
    :param initial_layer_dropout: Initial layer dropout configuration
    """
    # this is ignored for schnell
    guidance_vec = (torch.full((img.shape[0],), state.guidance, device=img.device, dtype=img.dtype) * 0.0) * 0.0
    current_layer_dropout = [initial_layer_dropout]
    previous_step_tensor: list[Optional[Tensor]] = [None]  # Store previous step's tensor for masking
    cached_prediction: list[Optional[Tensor]] = [None]  # Cache prediction to avoid duplicate model calls
    cached_prediction_state: list[Optional[dict]] = [None]  # Cache state when prediction was generated
    controller_ref: list[Optional["ManualTimestepController"]] = [None]  # Reference to controller for closure access

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

        # Get use_previous_as_mask from controller state if available
        use_mask = False
        if controller_ref[0] is not None:
            use_mask = controller_ref[0].use_previous_as_mask

        # Apply mask from previous step if enabled
        if use_mask and previous_step_tensor[0] is not None:
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
        initial_guidance=state.guidance,
    )
    controller_ref[0] = controller  # Store reference for closure access

    controller.set_layer_dropout(initial_layer_dropout)

    if state.width is not None and state.height is not None:
        controller.set_resolution(state.width, state.height)
    if state.seed is not None:
        controller.set_seed(state.seed)
    if state.prompt is not None:
        controller.set_prompt(state.prompt)
    if state.num_steps is not None:
        controller.set_num_steps(state.num_steps)
    controller.set_vae_shift_offset(state.vae_shift_offset)
    controller.set_vae_scale_offset(state.vae_scale_offset)
    controller.set_use_previous_as_mask(state.use_previous_as_mask)

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
                nfo(f"Guidance set to {new_guidance:.2f}")
            except ValueError:
                nfo("Invalid guidance value, keeping current value")
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
                    nfo("Layer dropout cleared")
                else:
                    nfo(f"Layer dropout set to: {layer_indices}")
            except ValueError:
                nfo("Invalid layer indices, keeping current value")
        elif choice == "r":
            try:
                nfo("\nPreferred resolutions:")
                for i, (w, h) in enumerate(PREFERRED_KONTEXT_RESOLUTIONS):
                    current_marker = ""
                    if state.width == w and state.height == h:
                        current_marker = " (current)"
                    nfo(f"  {i}: {w}x{h}{current_marker}")
                resolution_input = input(f"\nEnter resolution index (0-{len(PREFERRED_KONTEXT_RESOLUTIONS) - 1}) or 'custom' for custom: ").strip()
                if resolution_input.lower() == "custom":
                    width_input = input("Enter width: ").strip()
                    height_input = input("Enter height: ").strip()
                    new_width = int(width_input)
                    new_height = int(height_input)
                    controller.set_resolution(new_width, new_height)
                    nfo(f"Resolution set to: {new_width}x{new_height}")
                else:
                    resolution_idx = int(resolution_input)
                    if 0 <= resolution_idx < len(PREFERRED_KONTEXT_RESOLUTIONS):
                        new_width, new_height = PREFERRED_KONTEXT_RESOLUTIONS[resolution_idx]
                        controller.set_resolution(new_width, new_height)
                        nfo(f"Resolution set to: {new_width}x{new_height}")
                    else:
                        nfo("Invalid resolution index, keeping current value")
            except (ValueError, IndexError):
                nfo("Invalid resolution input, keeping current value")
        elif choice == "s":
            try:
                current_seed = state.seed if state.seed is not None else 0
                seed_input = input(f"Enter new seed number (current: {current_seed}, or press Enter for random): ").strip()
                if seed_input == "":
                    # Generate random seed
                    new_seed = int(torch.randint(0, 2**31, (1,)).item())
                else:
                    new_seed = int(seed_input)
                controller.set_seed(new_seed)
                seed_planter(new_seed)
                nfo(f"Seed set to: {new_seed}")
            except ValueError:
                nfo("Invalid seed value, keeping current seed")
        elif choice == "b":
            try:
                new_mask_value = not state.use_previous_as_mask
                controller.set_use_previous_as_mask(new_mask_value)
                nfo(f"Previous step tensor mask: {'ENABLED' if new_mask_value else 'DISABLED'}")
                # Refresh state to get updated values
                state = controller.current_state
            except Exception as e:
                nfo(f"Error setting buffer options: {e}")
        elif choice == "v":
            try:
                if ae is None:
                    nfo("AutoEncoder not available, cannot set VAE shift")
                else:
                    shift_input = input(f"Enter VAE shift offset (current: {state.vae_shift_offset:.4f}, or press Enter to reset to 0.0): ").strip()
                    if shift_input == "":
                        controller.set_vae_shift_offset(0.0)
                        nfo("VAE shift offset reset to 0.0")
                    else:
                        new_offset = float(shift_input)
                        controller.set_vae_shift_offset(new_offset)
                        nfo(f"VAE shift offset set to: {new_offset:.4f}")
                    # Refresh state to get updated values
                    state = controller.current_state
            except ValueError:
                nfo("Invalid VAE shift value, keeping current value")
        elif choice == "c":
            try:
                if ae is None:
                    nfo("AutoEncoder not available, cannot set VAE scale")
                else:
                    scale_input = input(f"Enter VAE scale offset (current: {state.vae_scale_offset:.4f}, or press Enter to reset to 0.0): ").strip()
                    if scale_input == "":
                        controller.set_vae_scale_offset(0.0)
                        nfo("VAE scale offset reset to 0.0")
                    else:
                        new_offset = float(scale_input)
                        controller.set_vae_scale_offset(new_offset)
                        nfo(f"VAE scale offset set to: {new_offset:.4f}")
                    # Refresh state to get updated values
                    state = controller.current_state
            except ValueError:
                nfo("Invalid VAE scale value, keeping current value")
        elif choice == "e":
            nfo("Entering edit mode (use c/cont to exit)...")
            breakpoint()
        else:
            nfo("Invalid choice, please try again")

        nfo(f"\nStep {step}/{state.total_timesteps} @ noise level {state.current_timestep:.4f}")
        nfo(f"Guidance: {state.guidance:.2f}")
        nfo(f"Seed: {state.seed}")
        if state.width is not None and state.height is not None:
            nfo(f"Resolution: {state.width}x{state.height}")
        if state.layer_dropout:
            nfo(f"Layer dropout: {state.layer_dropout}")
        else:
            nfo("Layer dropout: None")
        nfo(f"Buffer mask: {'ON' if state.use_previous_as_mask else 'OFF'}")
        if ae is not None:
            nfo(f"VAE shift offset: {state.vae_shift_offset:.4f}")
            nfo(f"VAE scale offset: {state.vae_scale_offset:.4f}")

        # Generate preview
        t0 = time.perf_counter()
        if state.seed is not None:
            seed_planter(state.seed)
        if ae is not None and state.width is not None and state.height is not None:
            # Reuse cached prediction if available, otherwise generate it
            # This will be cached and reused in denoise_step_fn when advancing
            pred_preview = get_prediction(state.current_sample, state.current_timestep, state.guidance, current_layer_dropout[0])

            intermediate = state.current_sample - state.current_timestep * pred_preview
            intermediate = unpack(intermediate.float(), state.height, state.width)

            sync_torch(device)
            t1 = time.perf_counter()

            nfo(f"Elapsed time: {t1 - t0:.1f}s")

            with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
                # Apply VAE shift/scale offset by manually adjusting the decode operation
                if state.vae_shift_offset != 0.0 or state.vae_scale_offset != 0.0:
                    # Decode with offset: z = z / (scale_factor + scale_offset) + (shift_factor + shift_offset)
                    z_adjusted = intermediate / (ae.scale_factor + state.vae_scale_offset) + (ae.shift_factor + state.vae_shift_offset)
                    intermediate_image = ae.decoder(z_adjusted)
                else:
                    intermediate_image = ae.decode(intermediate)
                file_path_named = name_save_file_as(ExtensionType.WEBP)
                if state.seed is not None:
                    controller.store_state_in_chain(current_seed=state.seed)
                save_with_hyperchain(file_path_named, intermediate_image, controller.hyperchain, ExtensionType.WEBP)
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
