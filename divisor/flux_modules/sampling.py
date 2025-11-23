# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->
# adapted BFL Flux code from https://github.com/black-forest-labs/flux

import math
from dataclasses import dataclass
from typing import Callable, Optional
import time
from einops import rearrange, repeat
import torch
from torch import Tensor
from nnll.constants import ExtensionType
from nnll.helpers import generate_valid_resolutions
from nnll.save_generation import name_save_file_as, save_with_hyperchain
from nnll.console import nfo
from nnll.init_gpu import device, sync_torch, clear_cache

from divisor.controller import ManualTimestepController, DenoisingState, rng, variation_rng
from divisor.flux_modules.autoencoder import AutoEncoder
from divisor.flux_modules.model import Flux
from divisor.flux_modules.text_embedder import HFEmbedder


@dataclass
class SamplingOptions:
    """Validate sampling parameters."""

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
    dtype: torch.dtype,
    seed: int,
    device: torch.device | None = None,
) -> Tensor:
    """Generate noise tensor.\n
    :param num_samples: Number of samples to generate
    :param height: Height of the image
    :param width: Width of the image
    :param device: Device to generate the noise on
    :param dtype: Data type of the noise
    :param seed: Seed for the random number generator
    :returns: Noise tensor"""
    # Get the generator's device to ensure compatibility
    generator_device = rng._torch_generator.device if rng._torch_generator is not None else torch.device("cpu")

    # Create tensor on generator's device first (required for MPS compatibility)
    noise = torch.randn(
        num_samples,
        16,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        dtype=dtype,
        generator=rng._torch_generator,
        device=generator_device,
    )

    # Move to target device if different
    if device is not None and generator_device != device:
        noise = noise.to(device)

    return noise


def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: str | list[str]) -> dict[str, Tensor]:
    """Prepare the text embeddings for the model.\n
    :param t5: T5 embedder
    :param clip: CLIP embedder
    :param img: Image tensor
    :param prompt: Prompt
    :returns: Dictionary of input tensors"""
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

    del clip, t5
    clear_cache()
    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }


def time_shift(mu: float, sigma: float, t: Tensor) -> Tensor:
    """Adjustable noise schedule. Compress or stretch any schedule to match a dynamic step sequence length.\n
    :param mu: Original schedule parameter.
    :param sigma: Original schedule parameter.
    :param t: Tensor of original timesteps in [0,1].
    :returns: Adjusted timestep tensor."""
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
    """Generate a schedule of timesteps.\n
    :param num_steps: Number of steps to generate
    :param image_seq_len: Length of the image sequence
    :param base_shift: Base shift value
    :param max_shift: Maximum shift value
    :param shift: Whether to shift the schedule
    :returns: List of timesteps"""
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def mix_noise(from_noise: Tensor, to_noise: Tensor, strength: float, variation_method: str = "linear") -> Tensor:
    """Mix two noise tensors using specified method.\n
    :param from_noise: Source noise tensor
    :param to_noise: Target noise tensor to mix towards
    :param strength: Mixing strength (0.0 to 1.0)
    :param variation_method: Mixing method ('linear' or 'slerp')
    :returns: Mixed noise tensor
    """
    to_noise = to_noise.to(from_noise.device)

    if variation_method == "slerp":
        # Spherical linear interpolation
        # Flatten for norm calculation (works with any tensor shape)
        from_flat = from_noise.flatten(start_dim=1)
        to_flat = to_noise.flatten(start_dim=1)

        from_norm = torch.norm(from_flat, dim=1, keepdim=True)
        to_norm = torch.norm(to_flat, dim=1, keepdim=True)

        # Normalize
        from_unit = from_flat / (from_norm + 1e-8)
        to_unit = to_flat / (to_norm + 1e-8)

        # Dot product for angle
        dot = (from_unit * to_unit).sum(dim=1, keepdim=True)
        dot = torch.clamp(dot, -1.0, 1.0)
        theta = torch.acos(dot)

        # Slerp formula
        sin_theta = torch.sin(theta)
        w1 = torch.sin((1 - strength) * theta) / (sin_theta + 1e-8)
        w2 = torch.sin(strength * theta) / (sin_theta + 1e-8)

        # Apply weights and reshape back
        mixed_flat = w1 * from_flat + w2 * to_flat
        mixed_noise = mixed_flat.reshape(from_noise.shape)
    else:
        # Linear interpolation
        mixed_noise = (1 - strength) * from_noise + strength * to_noise
        # Scale factor correction for variance preservation
        scale_factor = math.sqrt((1 - strength) ** 2 + strength**2)
        mixed_noise = mixed_noise / (scale_factor + 1e-8)

    return mixed_noise


def apply_variation_noise(
    latent_sample: Tensor,
    variation_seed: int | None,
    variation_strength: float,
    mask: Tensor | None = None,
    variation_method: str = "linear",
) -> Tensor:
    """Apply variation noise to the latent sample.\n
    :param latent_sample: Current sample tensor in 3D sequence format [batch, sequence, features]
    :param variation_seed: Seed for variation noise generation, or None to disable
    :param variation_strength: Strength of variation (0.0 to 1.0)
    :param mask: Optional mask tensor for selective application
    :param variation_method: Mixing method ('linear' or 'slerp')
    :returns: Sample with variation noise applied
    """
    if variation_seed is None or variation_strength == 0.0:
        return latent_sample

    # Set seed for variation noise generation
    if variation_seed is not None:
        variation_rng.next_seed(variation_seed)
    else:
        variation_seed = variation_rng.next_seed()

    # Get generator and its device
    variation_generator = variation_rng._torch_generator
    generator_device = variation_generator.device if variation_generator is not None else torch.device("cpu")

    # Generate variation noise matching the sample shape
    # Create on generator's device first (required for MPS compatibility)
    variation_noise = torch.randn(
        latent_sample.shape,
        dtype=latent_sample.dtype,
        layout=latent_sample.layout,
        generator=variation_generator,
        device=generator_device,
    )

    # Move to sample's device if different
    if generator_device != latent_sample.device:
        variation_noise = variation_noise.to(latent_sample.device)

    if mask is None:
        # Simple mixing without mask
        result = mix_noise(latent_sample, variation_noise, variation_strength, variation_method)
    else:
        # Apply mask: mask=1 uses mixed noise, mask=0 uses original
        mixed_noise_result = mix_noise(latent_sample, variation_noise, variation_strength, variation_method)
        result = (mask == 1).float() * mixed_noise_result + (mask == 0).float() * latent_sample

    return result


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
    ae: AutoEncoder,
    timesteps: list[float],  # sampling parameters
    img_cond: Tensor | None = None,  # extra img tokens (channel-wise)
    img_cond_seq: Tensor | None = None,  # extra img tokens (sequence-wise)
    img_cond_seq_ids: Tensor | None = None,
    torch_device: torch.device = device,
    initial_layer_dropout: Optional[list[int]] = None,
):
    """Denoise using Flux model with optional ManualTimestepController.\n
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
    :param initial_layer_dropout: Initial layer dropout configuration"""

    # this is ignored for schnell
    guidance_vec = (torch.full((img.shape[0],), state.guidance, device=img.device, dtype=img.dtype) * 0.0) * 0.0
    current_layer_dropout = [initial_layer_dropout]
    previous_step_tensor: list[Optional[Tensor]] = [None]  # Store previous step's tensor for masking
    cached_prediction: list[Optional[Tensor]] = [None]  # Cache prediction to avoid duplicate model calls
    cached_prediction_state: list[Optional[dict]] = [None]  # Cache state when prediction was generated
    controller_ref: list[Optional["ManualTimestepController"]] = [None]  # Reference to controller for closure access

    def clear_prediction_cache():
        """Empty the prediction cache.\n"""
        cached_prediction[0] = None
        cached_prediction_state[0] = None

    def get_prediction(sample: Tensor, t_curr: float, guidance_val: float, layer_dropouts_val: Optional[list[int]]) -> Tensor:
        """Generate model prediction, reusing cached prediction if state hasn't changed.\n
        :param sample: Current sample tensor
        :param t_curr: Current timestep
        :param guidance_val: Guidance value
        :param layer_dropouts_val: Layer dropout configuration
        :returns: Model prediction"""
        # Create a simple hash of the sample tensor for cache key (using first few values)
        # This is faster than hashing the entire tensor but should be sufficient for cache invalidation
        # Handle different tensor shapes safely
        if sample.numel() > 0:
            # Flatten and get first element for hash
            first_val = float(sample.flatten()[0].item())
        else:
            first_val = 0.0
        sample_hash = hash((sample.shape, first_val))

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
        """Single denoising step function for the controller.\n
        :param sample: Current sample tensor
        :param t_curr: Current timestep
        :param t_prev: Previous timestep
        :param guidance_val: Guidance value
        :returns: Model prediction"""

        layer_dropouts = current_layer_dropout[0]
        pred = get_prediction(sample, t_curr, guidance_val, layer_dropouts)
        use_mask = False
        if controller_ref[0] is not None:
            use_mask = controller_ref[0].use_previous_as_mask
        if use_mask and previous_step_tensor[0] is not None:
            prev_tensor = previous_step_tensor[0]
            if prev_tensor.shape == pred.shape:
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

        # Standard noise addition
        result = (1 - (t_curr + dt)) * x0 + (t_curr + dt) * torch.randn_like(x0)

        # Apply variation noise if enabled
        if controller_ref[0] is not None:
            state = controller_ref[0].current_state
            if state.variation_seed is not None and state.variation_strength > 0.0:
                result = apply_variation_noise(
                    latent_sample=result,
                    variation_seed=state.variation_seed,
                    variation_strength=state.variation_strength,
                    mask=None,
                    variation_method="linear",
                )

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
        file_path_named = name_save_file_as(ExtensionType.WEBP)
        state = controller.current_state
        step = state.timestep_index
        nfo(f"\nStep {step}/{state.total_timesteps} @ noise level {state.current_timestep:.4f}")
        nfo(f"[G]uidance: {state.guidance:.2f}")
        nfo(f"[S]eed: {state.seed}")
        if state.width is not None and state.height is not None:
            nfo(f"[R]esolution: {state.width}x{state.height}")
        if state.layer_dropout:
            nfo(f"[L]ayer dropout: {state.layer_dropout}")
        else:
            nfo("[L]ayer dropout: None")
        nfo(f"[B]uffer mask: {'ON' if state.use_previous_as_mask else 'OFF'}")
        if ae is not None:
            nfo(f"[V]AE shift offset: {state.vae_shift_offset:.4f}")
            nfo(f"[V]VAE scale offset: {state.vae_scale_offset:.4f}")
        if state.variation_seed is not None:
            nfo(f"[X]Variation seed: {state.variation_seed}, strength: {state.variation_strength:.3f}")
        else:
            nfo("[X]Variation: OFF")
        nfo(f"[D]eterministic: {'ON' if state.deterministic else 'OFF'}")
        choice = input(": [BDGLSRVX] advance with Enter: ").lower().strip()

        if choice == "":
            nfo("Advancing...")
            clear_prediction_cache()
            state = controller.current_state
            controller.step()
        elif choice == "g":
            try:
                new_guidance = float(input(f"Enter new guidance value (current: {state.guidance:.2f}): "))
                controller.set_guidance(new_guidance)
                # Invalidate cache since guidance changed
                clear_prediction_cache()
                state = controller.current_state
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
                state = controller.current_state
                current_layer_dropout[0] = controller.layer_dropout
                # Invalidate cache since layer_dropout changed
                clear_prediction_cache()
                if layer_indices is None:
                    nfo("Layer dropout cleared")
                else:
                    nfo(f"Layer dropout set to: {layer_indices}")
            except ValueError:
                nfo("Invalid layer indices, keeping current value")
        elif choice == "r":
            try:
                if state.width is None or state.height is None:
                    nfo("Cannot generate resolutions: width or height not set")
                else:
                    # Generate valid resolutions based on current resolution
                    valid_resolutions = generate_valid_resolutions(state.width, state.height)
                    nfo("\nValid resolutions (same patch count):")
                    for i, (w, h) in enumerate(valid_resolutions):
                        current_marker = ""
                        if state.width == w and state.height == h:
                            current_marker = " (current)"
                        nfo(f"  {i}: {w}x{h}{current_marker}")
                    resolution_input = input(f"\nEnter resolution index (0-{len(valid_resolutions) - 1}) or 'custom' for custom: ").strip()
                    if resolution_input.lower() == "custom":
                        width_input = input("Enter width: ").strip()
                        height_input = input("Enter height: ").strip()
                        new_width = int(width_input)
                        new_height = int(height_input)
                        controller.set_resolution(new_width, new_height)
                        clear_prediction_cache()
                        state = controller.current_state
                        nfo(f"Resolution set to: {new_width}x{new_height}")
                    else:
                        resolution_idx = int(resolution_input)
                        if 0 <= resolution_idx < len(valid_resolutions):
                            new_width, new_height = valid_resolutions[resolution_idx]
                            controller.set_resolution(new_width, new_height)
                            clear_prediction_cache()
                            state = controller.current_state
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
                    new_seed = rng.next_seed()
                else:
                    new_seed = rng.next_seed(int(seed_input))
                controller.set_seed(new_seed)
                clear_prediction_cache()
                state = controller.current_state
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
                    vae_input = input("\nChoose [S]hift or [C]scale: ").strip()
                    if vae_input.lower() == "c":
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
                        clear_prediction_cache()
                    elif vae_input.lower() == "s":
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
                        clear_prediction_cache()
            except ValueError:
                nfo("Invalid VAE scale value, keeping current value")
        elif choice == "x":
            try:
                var_input = input(
                    f"Variation (current integer seed: {state.variation_seed}, float strength: {state.variation_strength:.3f}. type a number, leave empty for random, or use 0.0 to disable): "
                ).strip()

                if not var_input or "." not in var_input:
                    # Try to parse as integer (seed)
                    try:
                        if var_input != "":
                            variation_seed = variation_rng.next_seed(int(var_input))
                        else:
                            variation_seed = variation_rng.next_seed()
                        controller.set_variation_seed(variation_seed)
                        clear_prediction_cache()
                        state = controller.current_state
                        nfo(f"Variation seed set to: {state.variation_seed}")
                    except ValueError:
                        nfo("Invalid integer seed value, keeping current value")
                else:
                    # Try to parse as float (strength)
                    try:
                        strength_value = float(var_input)
                        if strength_value < 0.0 or strength_value > 1.0:
                            state = controller.current_state
                            nfo("Variation strength must be between 0.0 and 1.0, keeping current value")
                        else:
                            controller.set_variation_strength(strength_value)
                            clear_prediction_cache()
                            state = controller.current_state
                            nfo(f"Variation strength set to: {strength_value:.3f}")
                    except ValueError:
                        nfo("Invalid float strength value, keeping current value")
            except (ValueError, KeyboardInterrupt):
                nfo("Invalid variation value, keeping current value")
        elif choice == "d":
            try:
                new_deterministic = not state.deterministic
                controller.set_deterministic(new_deterministic)
                clear_prediction_cache()
                state = controller.current_state
                nfo(f"Deterministic mode: {'ENABLED' if new_deterministic else 'DISABLED'}")
            except Exception as e:
                nfo(f"Error setting deterministic mode: {e}")
        elif choice == "e":
            nfo("Entering edit mode (use c/cont to exit)...")
            breakpoint()
            clear_prediction_cache()
        else:
            nfo("Invalid choice, please try again")

        # Generate preview
        t0 = time.perf_counter()
        if state.seed is not None:
            rng.next_seed(state.seed)
        else:
            state.seed = rng.next_seed()
        if ae is not None and state.width is not None and state.height is not None:
            # Reuse cached prediction if available, otherwise generate it
            # This will be cached and reused in denoise_step_fn when advancing
            pred_preview = get_prediction(state.current_sample, state.current_timestep, state.guidance, current_layer_dropout[0])

            intermediate = state.current_sample - state.current_timestep * pred_preview
            intermediate = unpack(intermediate.float(), state.height, state.width)

            sync_torch(device)
            t1 = time.perf_counter()

            nfo(f"Step time: {t1 - t0:.1f}s")

            with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
                # Apply VAE shift/scale offset by manually adjusting the decode operation
                if state.vae_shift_offset != 0.0 or state.vae_scale_offset != 0.0:
                    # Decode with offset: z = z / (scale_factor + scale_offset) + (shift_factor + shift_offset)
                    z_adjusted = intermediate / (ae.scale_factor + state.vae_scale_offset) + (ae.shift_factor + state.vae_shift_offset)
                    intermediate_image = ae.decoder(z_adjusted)
                else:
                    intermediate_image = ae.decode(intermediate)
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
