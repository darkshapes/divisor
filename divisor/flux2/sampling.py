# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->
# adapted BFL Flux code from https://github.com/black-forest-labs/flux2

import math
import time
from typing import Optional

from PIL import Image
from einops import rearrange
from nnll.console import nfo
from nnll.constants import ExtensionType
from nnll.init_gpu import sync_torch
from nnll.save_generation import name_save_file_as, save_with_hyperchain
import torch
from torch import Tensor
import torchvision

from divisor.cli_menu import route_choices
from divisor.controller import ManualTimestepController, rng, variation_rng
from divisor.denoise_step import (
    create_clear_prediction_cache,
    create_denoise_step_fn,
    create_get_prediction,
    create_recompute_text_embeddings,
)
from divisor.flux2 import precision
from divisor.flux2.model import Flux2
from divisor.state import (
    DenoiseSettings,
    DenoiseSettingsFlux2,
    GetImagePredictionSettings,
    GetPredictionSettings,
)
from divisor.interaction_context import InteractionContext


def compress_time(t_ids: Tensor) -> Tensor:
    assert t_ids.ndim == 1
    t_ids_max = torch.max(t_ids)
    t_remap = torch.zeros((t_ids_max + 1,), device=t_ids.device, dtype=t_ids.dtype)  # type: ignore
    t_unique_sorted_ids = torch.unique(t_ids, sorted=True)
    t_remap[t_unique_sorted_ids] = torch.arange(len(t_unique_sorted_ids), device=t_ids.device, dtype=t_ids.dtype)
    t_ids_compressed = t_remap[t_ids]
    return t_ids_compressed


def scatter_ids(x: Tensor, x_ids: Tensor) -> list[Tensor]:
    """
    using position ids to scatter tokens into place
    """
    x_list = []
    t_coords = []
    for data, pos in zip(x, x_ids):
        _, ch = data.shape  # noqa: F841
        t_ids = pos[:, 0].to(torch.int64)
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)

        t_ids_cmpr = compress_time(t_ids)

        t = torch.max(t_ids_cmpr) + 1
        h = torch.max(h_ids) + 1
        w = torch.max(w_ids) + 1

        flat_ids = t_ids_cmpr * w * h + h_ids * w + w_ids

        out = torch.zeros((t * h * w, ch), device=data.device, dtype=data.dtype)  # type: ignore
        out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)

        x_list.append(rearrange(out, "(t h w) c -> 1 c t h w", t=t, h=h, w=w))
        t_coords.append(torch.unique(t_ids, sorted=True))
    return x_list


def encode_image_refs(ae, img_ctx: list[Image.Image]):
    scale = 10

    if len(img_ctx) > 1:
        limit_pixels = 1024**2
    elif len(img_ctx) == 1:
        limit_pixels = 2024**2
    else:
        limit_pixels = None

    if not img_ctx:
        return None, None

    img_ctx_prep = default_prep(img=img_ctx, limit_pixels=limit_pixels)
    if not isinstance(img_ctx_prep, list):
        img_ctx_prep = [img_ctx_prep]

    # Encode each reference image
    encoded_refs = []
    torch_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
    for img in img_ctx_prep:
        encoded = ae.encode(img[None].to(torch_device))[0]
        encoded_refs.append(encoded)

    # Create time offsets for each reference
    t_off = [scale + scale * t for t in torch.arange(0, len(encoded_refs))]
    t_off = [t.view(-1) for t in t_off]

    # Process with position IDs
    ref_tokens, ref_ids = listed_prc_img(encoded_refs, t_coord=t_off)

    # Concatenate all references along sequence dimension
    ref_tokens = torch.cat(ref_tokens, dim=0)  # (total_ref_tokens, C)
    ref_ids = torch.cat(ref_ids, dim=0)  # (total_ref_tokens, 4)

    # Add batch dimension
    ref_tokens = ref_tokens.unsqueeze(0)  # (1, total_ref_tokens, C)
    ref_ids = ref_ids.unsqueeze(0)  # (1, total_ref_tokens, 4)

    return ref_tokens.to(precision), ref_ids


def prc_txt(x: Tensor, t_coord: Tensor | None = None) -> tuple[Tensor, Tensor]:
    _l, _ = x.shape  # noqa: F841

    coords = {
        "t": torch.arange(1) if t_coord is None else t_coord,
        "h": torch.arange(1),  # dummy dimension
        "w": torch.arange(1),  # dummy dimension
        "l": torch.arange(_l),
    }
    x_ids = torch.cartesian_prod(coords["t"], coords["h"], coords["w"], coords["l"])
    return x, x_ids.to(x.device)


def batched_wrapper(fn):
    def batched_prc(x: Tensor, t_coord: Tensor | None = None) -> tuple[Tensor, Tensor]:
        results = []
        for i in range(len(x)):
            results.append(
                fn(
                    x[i],
                    t_coord[i] if t_coord is not None else None,
                )
            )
        x, x_ids = zip(*results)  # type: ignore
        return torch.stack(x), torch.stack(x_ids)  # type: ignore

    return batched_prc


def listed_wrapper(fn):
    def listed_prc(
        x: list[Tensor],
        t_coord: list[Tensor] | None = None,
    ) -> tuple[list[Tensor], list[Tensor]]:
        results = []
        for i in range(len(x)):
            results.append(
                fn(
                    x[i],
                    t_coord[i] if t_coord is not None else None,
                )
            )
        x, x_ids = zip(*results)  # type: ignore
        return list(x), list(x_ids)

    return listed_prc


def prc_img(x: Tensor, t_coord: Tensor | None = None) -> tuple[Tensor, Tensor]:
    _, h, w = x.shape  # noqa: F841
    x_coords = {
        "t": torch.arange(1) if t_coord is None else t_coord,
        "h": torch.arange(h),
        "w": torch.arange(w),
        "l": torch.arange(1),
    }
    x_ids = torch.cartesian_prod(x_coords["t"], x_coords["h"], x_coords["w"], x_coords["l"])
    x = rearrange(x, "c h w -> (h w) c")
    return x, x_ids.to(x.device)


listed_prc_img = listed_wrapper(prc_img)
batched_prc_img = batched_wrapper(prc_img)
batched_prc_txt = batched_wrapper(prc_txt)


def center_crop_to_multiple_of_x(img: Image.Image | list[Image.Image], x: int) -> Image.Image | list[Image.Image]:
    if isinstance(img, list):
        return [center_crop_to_multiple_of_x(_img, x) for _img in img]  # type: ignore

    w, h = img.size
    new_w = (w // x) * x
    new_h = (h // x) * x

    left = (w - new_w) // 2
    top = (h - new_h) // 2
    right = left + new_w
    bottom = top + new_h

    resized = img.crop((left, top, right, bottom))
    return resized


def cap_pixels(img: Image.Image | list[Image.Image], k):
    if isinstance(img, list):
        return [cap_pixels(_img, k) for _img in img]
    w, h = img.size
    pixel_count = w * h

    if pixel_count <= k:
        return img

    # Scaling factor to reduce total pixels below K
    scale = math.sqrt(k / pixel_count)
    new_w = int(w * scale)
    new_h = int(h * scale)

    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def cap_min_pixels(img: Image.Image | list[Image.Image], max_ar=8, min_sidelength=64):
    if isinstance(img, list):
        return [cap_min_pixels(_img, max_ar=max_ar, min_sidelength=min_sidelength) for _img in img]
    w, h = img.size
    if w < min_sidelength or h < min_sidelength:
        raise ValueError(f"Skipping due to minimal sidelength underschritten h {h} w {w}")
    if w / h > max_ar or h / w > max_ar:
        raise ValueError(f"Skipping due to maximal ar overschritten h {h} w {w}")
    return img


def to_rgb(img: Image.Image | list[Image.Image]):
    if isinstance(img, list):
        return [
            to_rgb(
                _img,
            )
            for _img in img
        ]
    return img.convert("RGB")


def default_images_prep(
    x: Image.Image | list[Image.Image],
) -> torch.Tensor | list[torch.Tensor]:
    if isinstance(x, list):
        return [default_images_prep(e) for e in x]  # type: ignore
    x_tensor = torchvision.transforms.ToTensor()(x)
    return 2 * x_tensor - 1


def default_prep(img: Image.Image | list[Image.Image], limit_pixels: int | None, ensure_multiple: int = 16) -> torch.Tensor | list[torch.Tensor]:
    img_rgb = to_rgb(img)
    img_min = cap_min_pixels(img_rgb)  # type: ignore
    if limit_pixels is not None:
        img_cap = cap_pixels(img_min, limit_pixels)  # type: ignore
    else:
        img_cap = img_min
    img_crop = center_crop_to_multiple_of_x(img_cap, ensure_multiple)  # type: ignore
    img_tensor = default_images_prep(img_crop)
    return img_tensor


def generalized_time_snr_shift(t: Tensor, mu: float, sigma: float) -> Tensor:
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_schedule(num_steps: int, image_seq_len: int) -> list[float]:
    mu = compute_empirical_mu(image_seq_len, num_steps)
    timesteps = torch.linspace(1, 0, num_steps + 1)
    timesteps = generalized_time_snr_shift(timesteps, mu, 1.0)
    return timesteps.tolist()


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)


def denoise(settings: DenoiseSettingsFlux2) -> Tensor:
    """Simple non-interactive denoising function for Flux2.\n
    :param settings: SimpleDenoiseSettingsFlux2 containing all denoising configuration parameters
    :returns: Denoised image tensor"""
    model = settings.model
    img = settings.img
    img_ids = settings.img_ids
    txt = settings.txt
    txt_ids = settings.txt_ids
    timesteps = settings.timesteps
    guidance = settings.guidance
    img_cond_seq = settings.img_cond_seq
    img_cond_seq_ids = settings.img_cond_seq_ids

    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        img_input = img
        img_input_ids = img_ids
        if img_cond_seq is not None:
            assert img_cond_seq_ids is not None, "You need to provide either both or neither of the sequence conditioning"
            img_input = torch.cat((img_input, img_cond_seq), dim=1)
            img_input_ids = torch.cat((img_input_ids, img_cond_seq_ids), dim=1)
        pred = model(
            x=img_input,
            x_ids=img_input_ids,
            timesteps=t_vec,
            ctx=txt,
            ctx_ids=txt_ids,
            guidance=guidance_vec,
        )
        if img_input_ids is not None:
            pred = pred[:, : img.shape[1]]

        img = img + (t_prev - t_curr) * pred

    return img


@torch.inference_mode()
def denoise_interactive(
    model: Flux2,
    settings: DenoiseSettings,
):
    """Interactive denoising using Flux2 model with optional ManualTimestepController.\n
    :param model: Flux2 model instance
    :param settings: DenoiseSettings containing all denoising configuration parameters"""

    # Extract settings for easier access
    img = settings.img
    img_ids = settings.img_ids
    txt = settings.txt
    txt_ids = settings.txt_ids
    state = settings.state
    ae = settings.ae
    timesteps = settings.timesteps
    img_cond_seq = settings.img_cond_seq
    img_cond_seq_ids = settings.img_cond_seq_ids
    from nnll.init_gpu import device as default_device

    denoise_device = settings.device if settings.device is not None else default_device
    initial_layer_dropout = settings.initial_layer_dropout
    text_embedder = settings.text_embedder

    # this is ignored for schnell
    current_layer_dropout = [initial_layer_dropout]
    previous_step_tensor: list[Optional[Tensor]] = [None]  # Store previous step's tensor for masking
    cached_prediction: list[Optional[Tensor]] = [None]  # Cache prediction to avoid duplicate model calls
    cached_prediction_state: list[Optional[dict]] = [None]  # Cache state when prediction was generated
    controller_ref: list[Optional["ManualTimestepController"]] = [None]  # Reference to controller for closure access

    model_ref: list[Flux2] = [model]
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
    # Flux2 uses ctx instead of txt, and doesn't have separate CLIP embeddings
    current_txt: list[Tensor] = [txt]
    current_txt_ids: list[Tensor] = [txt_ids]
    current_vec: list[Optional[Tensor]] = [None]  # Flux2 doesn't use CLIP embeddings
    current_prompt: list[Optional[str]] = [state.prompt]  # Track current prompt to detect changes

    clear_prediction_cache = create_clear_prediction_cache(cached_prediction, cached_prediction_state)

    recompute_text_embeddings = create_recompute_text_embeddings(
        img,
        None,  # t5 not used for Flux2
        None,  # clip not used for Flux2
        current_txt,
        current_txt_ids,
        current_vec,  # type: ignore
        current_prompt,
        clear_prediction_cache,
        is_flux2=True,
        text_embedder=text_embedder,
    )

    pred_set = GetPredictionSettings(
        model_ref=model_ref,
        state=state,
        current_txt=current_txt,
        current_txt_ids=current_txt_ids,
        current_vec=current_vec,  # type: ignore
        cached_prediction=cached_prediction,
        cached_prediction_state=cached_prediction_state,
    )
    img_set = GetImagePredictionSettings(
        img_ids=img_ids,
        img=img,
        img_cond=None,  # img_cond not used in Flux2 (only img_cond_seq)
        img_cond_seq=img_cond_seq,
        img_cond_seq_ids=img_cond_seq_ids,
    )
    get_prediction = create_get_prediction(pred_set, img_set)

    denoise_step_fn = create_denoise_step_fn(
        controller_ref,
        current_layer_dropout,
        previous_step_tensor,
        get_prediction,
    )

    controller = ManualTimestepController(
        timesteps=timesteps,
        initial_sample=img,
        denoise_step_fn=denoise_step_fn,
        initial_guidance=state.guidance,
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
            if text_embedder is not None:
                recompute_text_embeddings(state.prompt)
            else:
                # If embedder not available, update current_prompt to avoid repeated checks
                current_prompt[0] = state.prompt

        interaction_context = InteractionContext(
            clear_prediction_cache=clear_prediction_cache,
            rng=rng,
            variation_rng=variation_rng,
            ae=ae,
            recompute_text_embeddings=recompute_text_embeddings,
        )
        state = route_choices(
            controller,
            state,
            interaction_context,
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
            # Flux2 uses scatter_ids to convert back to spatial format
            # The intermediate is already in the correct format (sequence of tokens)
            # We need to scatter it back to spatial dimensions for VAE decoding
            scattered = scatter_ids(intermediate, img_ids)
            if len(scattered) > 0:
                intermediate_list = torch.cat(scattered).squeeze(2)
                # scatter_ids returns list of tensors with shape (1, C, T, H, W)
                # We need (1, C, H, W) for VAE, so we take the first time slice or squeeze
                intermediate = intermediate_list[0].squeeze(2)  # Remove time dimension if present
                if intermediate.dim() == 5:
                    intermediate = intermediate[:, :, 0, :, :]  # Take first time slice

            sync_torch(denoise_device)
            t1 = time.perf_counter()

            nfo(f"Step time: {t1 - t0:.1f}s")

            if denoise_device.type == "cuda":
                context = torch.autocast(device_type=denoise_device.type, dtype=torch.bfloat16)
            else:
                from contextlib import nullcontext

                context = nullcontext()
            with context:
                # When autocast is disabled (MPS), ensure intermediate is in correct dtype for VAE
                if denoise_device.type != "cuda":
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
                    z_adjusted = intermediate / (ae.scale_factor + state.vae_scale_offset) + (ae.shift_factor + state.vae_shift_offset)  # type: ignore
                    intermediate_image = ae.decode(z_adjusted).float()
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


def concatenate_images(
    images: list[Image.Image],
) -> Image.Image:
    """
    Concatenate a list of PIL images horizontally with center alignment and white background.
    """

    # If only one image, return a copy of it
    if len(images) == 1:
        return images[0].copy()

    # Convert all images to RGB if not already
    images = [img.convert("RGB") if img.mode != "RGB" else img for img in images]

    # Calculate dimensions for horizontal concatenation
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)

    # Create new image with white background
    background_color = (255, 255, 255)
    new_img = Image.new("RGB", (total_width, max_height), background_color)

    # Paste images with center alignment
    x_offset = 0
    for img in images:
        y_offset = (max_height - img.height) // 2
        new_img.paste(img, (x_offset, y_offset))
        x_offset += img.width

    return new_img
