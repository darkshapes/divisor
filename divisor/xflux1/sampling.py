# SPDX-License-Identifier:Apache-2.0
# original XFlux code from https://github.com/TencentARC/FluxKits

import time
from typing import Callable, Optional

import torch
from torch import Tensor
from nnll.console import nfo
from nnll.constants import ExtensionType
from nnll.init_gpu import device, sync_torch
from nnll.save_generation import name_save_file_as, save_with_hyperchain

from divisor.xflux1.model import XFlux
from divisor.xflux1.autoencoder import AutoEncoder
from divisor.flux1.text_embedder import HFEmbedder
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
    create_denoise_step_fn,
)
from divisor.flux1.sampling import unpack


def create_get_prediction_xflux1(
    model_ref: list[XFlux],
    img_ids: Tensor,
    img: Tensor,
    state: DenoisingState,
    current_txt: list[Tensor],
    current_txt_ids: list[Tensor],
    current_vec: list[Tensor],
    cached_prediction: list[Optional[Tensor]],
    cached_prediction_state: list[Optional[dict]],
    neg_pred_enabled: bool,
    current_neg_txt: list[Tensor] | None,
    current_neg_txt_ids: list[Tensor] | None,
    current_neg_vec: list[Tensor] | None,
    true_gs: float,
    timestep_to_start_cfg: int,
    image_proj: Tensor | None,
    neg_image_proj: Tensor | None,
    ip_scale: Tensor,
    neg_ip_scale: Tensor,
    current_timestep_index: list[int],
) -> Callable[[Tensor, float, float, Optional[list[int]]], Tensor]:
    """Create a function to generate model prediction with caching for XFlux1.\n
    :param model_ref: Mutable list containing model reference
    :param img_ids: Image position IDs tensor
    :param img: Image tensor for shape/device reference
    :param state: DenoisingState object
    :param current_txt: Mutable list containing current text embeddings
    :param current_txt_ids: Mutable list containing current text IDs
    :param current_vec: Mutable list containing current CLIP embeddings
    :param cached_prediction: Mutable list containing cached prediction
    :param cached_prediction_state: Mutable list containing cached prediction state
    :param neg_pred_enabled: Whether negative prompt is enabled
    :param current_neg_txt: Mutable list containing negative text embeddings
    :param current_neg_txt_ids: Mutable list containing negative text IDs
    :param current_neg_vec: Mutable list containing negative CLIP embeddings
    :param true_gs: True guidance scale for CFG
    :param timestep_to_start_cfg: Timestep index to start CFG
    :param image_proj: IP-Adapter image projection
    :param neg_image_proj: IP-Adapter negative image projection
    :param ip_scale: IP-Adapter scale
    :param neg_ip_scale: IP-Adapter negative scale
    :param current_timestep_index: Mutable list containing current timestep index
    :return: Function that generates predictions with caching"""

    def get_prediction(
        sample: Tensor,
        t_curr: float,
        guidance_val: float,
        layer_dropouts_val: Optional[list[int]],
    ) -> Tensor:
        """Generate model prediction, reusing cached prediction if state hasn't changed.\n
        :param sample: Current sample tensor
        :param t_curr: Current timestep
        :param guidance_val: Guidance value
        :param layer_dropouts_val: Layer dropout configuration
        :returns: Model prediction"""
        # Create a simple hash of the sample tensor for cache key
        if sample.numel() > 0:
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
        try:
            model_dtype = next(model_ref[0].parameters()).dtype
        except (TypeError, StopIteration, AttributeError):
            model_dtype = sample.dtype
        use_autocast = device.type == "cuda"

        if not use_autocast:
            sample = sample.to(dtype=model_dtype)

        t_vec = torch.full((sample.shape[0],), t_curr, dtype=sample.dtype, device=sample.device)
        guidance_vec = torch.full((img.shape[0],), guidance_val, device=img.device, dtype=img.dtype)

        if not use_autocast:
            img_input = sample.to(dtype=model_dtype)
            txt_input = current_txt[0].to(dtype=model_dtype)
            vec_input = current_vec[0].to(dtype=model_dtype)
            t_vec = t_vec.to(dtype=model_dtype)
            guidance_vec = guidance_vec.to(dtype=model_dtype)
        else:
            img_input = sample
            txt_input = current_txt[0]
            vec_input = current_vec[0]

        # Get current timestep index for CFG check
        timestep_index = current_timestep_index[0]

        # Generate positive prediction
        pred = model_ref[0](
            img=img_input,
            img_ids=img_ids,
            txt=txt_input,
            txt_ids=current_txt_ids[0],
            y=vec_input,
            timesteps=t_vec,
            guidance=guidance_vec,
            image_proj=image_proj,
            ip_scale=ip_scale,
        )

        # Apply CFG if enabled and past start timestep
        if neg_pred_enabled and all([current_neg_txt, current_neg_txt_ids, current_neg_vec]) and timestep_index >= timestep_to_start_cfg:
            if not use_autocast:
                neg_txt_input = current_neg_txt[0].to(dtype=model_dtype)  # type: ignore
                neg_vec_input = current_neg_vec[0].to(dtype=model_dtype)  # type: ignore
            else:
                neg_txt_input = current_neg_txt[0]  # type: ignore
                neg_vec_input = current_neg_vec[0]  # type: ignore

            neg_pred = model_ref[0](
                img=img_input,
                img_ids=img_ids,
                txt=neg_txt_input,
                txt_ids=current_neg_txt_ids[0],  # type: ignore
                y=neg_vec_input,
                timesteps=t_vec,
                guidance=guidance_vec,
                image_proj=neg_image_proj,
                ip_scale=neg_ip_scale,
            )
            pred = neg_pred + true_gs * (pred - neg_pred)

        # Cache the prediction
        cached_prediction[0] = pred
        cached_prediction_state[0] = current_state

        return pred

    return get_prediction


@torch.inference_mode()
def denoise(
    model: XFlux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    neg_txt: Tensor,
    neg_txt_ids: Tensor,
    neg_vec: Tensor,
    state: DenoisingState,
    ae: AutoEncoder,
    timesteps: list[float],
    # sampling parameters
    guidance: float = 4.0,
    true_gs: float = 1.0,
    timestep_to_start_cfg: int = 0,
    # ip-adapter parameters
    image_proj: Tensor | None = None,
    neg_image_proj: Tensor | None = None,
    ip_scale: Tensor = torch.tensor(1.0),
    neg_ip_scale: Tensor = torch.tensor(1.0),
    initial_layer_dropout: Optional[list[int]] = None,
    t5: Optional[HFEmbedder] = None,
    clip: Optional[HFEmbedder] = None,
):
    """Denoise using XFlux model with optional ManualTimestepController.\n
    :param model: XFlux model instance
    :param img: Initial noisy image tensor
    :param img_ids: Image position IDs tensor
    :param txt: Text embeddings tensor
    :param txt_ids: Text position IDs tensor
    :param vec: CLIP embeddings vector tensor
    :param neg_txt: Negative text embeddings tensor
    :param neg_txt_ids: Negative text position IDs tensor
    :param neg_vec: Negative CLIP embeddings vector tensor
    :param state: DenoisingState containing current state parameters
    :param timesteps: List of timestep values
    :param guidance: Guidance value
    :param true_gs: True guidance scale for CFG
    :param timestep_to_start_cfg: Timestep index to start CFG
    :param image_proj: IP-Adapter image projection
    :param neg_image_proj: IP-Adapter negative image projection
    :param ip_scale: IP-Adapter scale
    :param neg_ip_scale: IP-Adapter negative scale
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
    current_timestep_index: list[int] = [0]  # Track current timestep index for CFG

    model_ref: list[XFlux] = [model]
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
    neg_pred_enabled = all([neg_txt is not None, neg_txt_ids is not None, neg_vec is not None])
    if neg_pred_enabled:
        current_neg_txt: list[Tensor] = [neg_txt]  # type: ignore
        current_neg_txt_ids: list[Tensor] = [neg_txt_ids]  # type: ignore
        current_neg_vec: list[Tensor] = [neg_vec]  # type: ignore
    else:
        current_neg_txt: list[Tensor] | None = None
        current_neg_txt_ids: list[Tensor] | None = None
        current_neg_vec: list[Tensor] | None = None
    current_prompt: list[Optional[str]] = [state.prompt]  # Track current prompt to detect changes

    clear_prediction_cache = create_clear_prediction_cache(cached_prediction, cached_prediction_state)

    recompute_text_embeddings = create_recompute_text_embeddings(img, t5, clip, current_txt, current_txt_ids, current_vec, current_prompt, clear_prediction_cache, is_flux2=False)

    get_prediction = create_get_prediction_xflux1(
        model_ref,
        img_ids,
        img,
        state,
        current_txt,
        current_txt_ids,
        current_vec,
        cached_prediction,
        cached_prediction_state,
        neg_pred_enabled,
        current_neg_txt,  # type: ignore
        current_neg_txt_ids,  # type: ignore
        current_neg_vec,  # type: ignore
        true_gs,
        timestep_to_start_cfg,
        image_proj,
        neg_image_proj,
        ip_scale,
        neg_ip_scale,
        current_timestep_index,
    )

    denoise_step_fn = create_denoise_step_fn(controller_ref, current_layer_dropout, previous_step_tensor, get_prediction)

    controller = ManualTimestepController(timesteps=timesteps, initial_sample=img, denoise_step_fn=denoise_step_fn, initial_guidance=state.guidance)
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

        # Update timestep index for CFG check
        current_timestep_index[0] = state.timestep_index if hasattr(state, "timestep_index") else 0

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


def denoise_simple(model, img, img_ids, txt, txt_ids, vec, timesteps, guidance=4.0):
    device = list(model.parameters())[0].device
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
        )
        img = img + (t_prev - t_curr) * pred
    return img
