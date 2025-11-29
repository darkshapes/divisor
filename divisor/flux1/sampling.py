# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->
# adapted BFL Flux code from https://github.com/black-forest-labs/flux

import math
import time
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from einops import rearrange, repeat
from nnll.console import nfo
from nnll.constants import ExtensionType
from nnll.init_gpu import device, sync_torch
from nnll.save_generation import name_save_file_as, save_with_hyperchain
from torch import Tensor

from divisor.commands import process_choice
from divisor.controller import (
    DenoisingState,
    ManualTimestepController,
    rng,
    variation_rng,
)
from divisor.denoise_step import (
    create_clear_prediction_cache,
    create_recompute_text_embeddings,
    create_get_prediction,
    create_denoise_step_fn,
)
from divisor.flux1.autoencoder import AutoEncoder
from divisor.flux1.model import Flux
from divisor.flux1.text_embedder import HFEmbedder


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
    device: torch.device = device,
    initial_layer_dropout: Optional[list[int]] = None,
    t5: Optional[HFEmbedder] = None,  # T5 embedder for prompt changes
    clip: Optional[HFEmbedder] = None,  # CLIP embedder for prompt changes
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
    :param device: PyTorch device
    :param initial_layer_dropout: Initial layer dropout configuration
    :param t5: Optional T5 embedder for recomputing text embeddings when prompt changes
    :param clip: Optional CLIP embedder for recomputing text embeddings when prompt changes"""

    # this is ignored for schnell
    current_layer_dropout = [initial_layer_dropout]
    previous_step_tensor: list[Optional[Tensor]] = [None]  # Store previous step's tensor for masking
    cached_prediction: list[Optional[Tensor]] = [None]  # Cache prediction to avoid duplicate model calls
    cached_prediction_state: list[Optional[dict]] = [None]  # Cache state when prediction was generated
    controller_ref: list[Optional["ManualTimestepController"]] = [None]  # Reference to controller for closure access

    model_ref: list[Flux] = [model]
    # Ensure model is on the correct device (fixes meta device issue)
    target_device = img.device
    # Safely get model device, handling Mock objects in tests
    try:
        model_device = next(model.parameters()).device
    except (TypeError, StopIteration, AttributeError):
        # Fallback for Mock objects or models without parameters
        # Assume model is already on correct device if we can't determine it
        model_device = target_device
    if model_device != target_device:
        model_ref[0] = model.to_empty(device=target_device)

    # Store embeddings in mutable containers so they can be updated when prompt changes
    current_txt: list[Tensor] = [txt]
    current_txt_ids: list[Tensor] = [txt_ids]
    current_vec: list[Tensor] = [vec]
    current_prompt: list[Optional[str]] = [state.prompt]  # Track current prompt to detect changes

    clear_prediction_cache = create_clear_prediction_cache(cached_prediction, cached_prediction_state)

    recompute_text_embeddings = create_recompute_text_embeddings(  # formatting
        img, t5, clip, current_txt, current_txt_ids, current_vec, current_prompt, clear_prediction_cache, is_flux2=False
    )

    get_prediction = create_get_prediction(
        model_ref,
        img_ids,
        img,
        state,
        img_cond,
        img_cond_seq,
        img_cond_seq_ids,
        current_txt,
        current_txt_ids,
        current_vec,
        cached_prediction,
        cached_prediction_state,
    )

    denoise_step_fn = create_denoise_step_fn(  # formatting
        controller_ref, current_layer_dropout, previous_step_tensor, get_prediction
    )

    controller = ManualTimestepController(  # formatting
        timesteps=timesteps, initial_sample=img, denoise_step_fn=denoise_step_fn, initial_guidance=state.guidance
    )
    controller_ref[0] = controller  # Store reference for closure access

    # Use state.layer_dropout if available, otherwise fall back to initial_layer_dropout
    layer_dropout_to_set = state.layer_dropout if state.layer_dropout is not None else initial_layer_dropout
    controller.set_layer_dropout(layer_dropout_to_set)

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

        # Check if prompt changed and recompute embeddings if needed
        if state.prompt is not None and state.prompt != current_prompt[0]:
            if t5 is not None and clip is not None:
                recompute_text_embeddings(state.prompt)
            else:
                # If embedders not available, update current_prompt to avoid repeated checks
                current_prompt[0] = state.prompt

        state = process_choice(
            controller,
            state,
            clear_prediction_cache,
            current_layer_dropout,
            rng,
            variation_rng,
            ae,
            t5,
            clip,
            recompute_text_embeddings,
        )

        # Generate preview
        t0 = time.perf_counter()
        if state.seed is not None:
            rng.next_seed(state.seed)
        else:
            state.seed = rng.next_seed()
        if ae is not None and state.width is not None and state.height is not None:
            # Reuse cached prediction if available, otherwise generate it
            # This will be cached and reused in denoise_step_fn when advancing
            # Always use state.layer_dropout from controller to ensure consistency
            pred_preview = get_prediction(
                state.current_sample,
                state.current_timestep,
                state.guidance,
                state.layer_dropout,
            )

            intermediate = state.current_sample - state.current_timestep * pred_preview
            # Unpack requires float32, but we'll convert back to correct dtype after
            intermediate = unpack(intermediate.float(), state.height, state.width)

            sync_torch(device)
            t1 = time.perf_counter()

            nfo(f"Step time: {t1 - t0:.1f}s")

            if device.type == "cuda":
                context = torch.autocast(device_type=device.type, dtype=torch.bfloat16)
            else:
                from contextlib import nullcontext

                context = nullcontext()
            with context:
                # When autocast is disabled (MPS), ensure intermediate is in correct dtype for VAE
                if device.type != "cuda":
                    # Get VAE encoder dtype to ensure intermediate matches (bfloat16)
                    # Safely get encoder dtype, handling Mock objects in tests
                    try:
                        ae_dtype = next(ae.encoder.parameters()).dtype
                    except (TypeError, StopIteration, AttributeError):
                        # Fallback: use intermediate dtype if we can't get encoder dtype (for Mock objects in tests)
                        ae_dtype = intermediate.dtype
                    intermediate = intermediate.to(dtype=ae_dtype)

                # Apply VAE shift/scale offset by manually adjusting the decode operation
                if state.vae_shift_offset != 0.0 or state.vae_scale_offset != 0.0:
                    # Decode with offset: z = z / (scale_factor + scale_offset) + (shift_factor + shift_offset)
                    z_adjusted = intermediate / (ae.scale_factor + state.vae_scale_offset) + (ae.shift_factor + state.vae_shift_offset)
                    intermediate_image = ae.decoder(z_adjusted)
                else:
                    intermediate_image = ae.decode(intermediate)
                if state.seed is not None:
                    controller.store_state_in_chain(current_seed=state.seed)
                save_with_hyperchain(
                    file_path_named,
                    intermediate_image,
                    controller.hyperchain,
                    ExtensionType.WEBP,
                )

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
