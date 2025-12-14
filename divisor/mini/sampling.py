# SPDX-License-Identifier:Apache-2.0
# original XFlux code from https://github.com/TencentARC/FluxKits

import time
from typing import Callable

from nnll.console import nfo
from nnll.constants import ExtensionType
from nnll.init_gpu import device, sync_torch
from nnll.save_generation import name_save_file_as, save_with_hyperchain
import torch
from torch import Tensor

from divisor.cli_menu import route_choices
from divisor.controller import ManualTimestepController, rng, variation_rng
from divisor.denoise_step import (
    create_clear_prediction_cache,
    create_denoise_step_fn,
    create_recompute_text_embeddings,
)
from divisor.flux1.sampling import prepare, unpack
from divisor.interaction_context import InteractionContext
from divisor.state import (
    AdditionalPredictionSettings,
    DenoiseSettings,
    GetImagePredictionSettings,
    GetPredictionSettings,
)
from divisor.mini.model import XFlux


def create_get_prediction_xflux1(
    pred_set: GetPredictionSettings,
    img_set: GetImagePredictionSettings,
    add_set: AdditionalPredictionSettings,
) -> Callable[[Tensor, float, float, list[int] | None], Tensor]:
    """Create a function to generate model prediction with caching for XFlux1.\n
    :param config: GetPredictionSettings containing all configuration parameters
    :return: Function that generates predictions with caching"""

    def get_prediction(
        sample: Tensor,
        t_curr: float,
        guidance_val: float,
        layer_dropouts_val: list[int] | None,
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
            pred_set.cached_prediction[0] is not None
            and pred_set.cached_prediction_state[0] is not None
            and pred_set.cached_prediction_state[0]["sample_hash"] == current_state["sample_hash"]
            and pred_set.cached_prediction_state[0]["t_curr"] == current_state["t_curr"]
            and pred_set.cached_prediction_state[0]["guidance"] == current_state["guidance"]
            and pred_set.cached_prediction_state[0]["layer_dropout"] == current_state["layer_dropout"]
        ):
            return pred_set.cached_prediction[0]

        # Generate new prediction
        try:
            model_dtype = next(pred_set.model_ref[0].parameters()).dtype
        except (TypeError, StopIteration, AttributeError):
            model_dtype = sample.dtype
        use_autocast = device.type == "cuda"

        if not use_autocast:
            sample = sample.to(dtype=model_dtype)

        t_vec = torch.full((sample.shape[0],), t_curr, dtype=sample.dtype, device=sample.device)
        guidance_vec = torch.full((img_set.img.shape[0],), guidance_val, device=img_set.img.device, dtype=img_set.img.dtype)

        if not use_autocast:
            img_input = sample.to(dtype=model_dtype)
            txt_input = pred_set.current_txt[0].to(dtype=model_dtype)
            vec_input = pred_set.current_vec[0].to(dtype=model_dtype)
            t_vec = t_vec.to(dtype=model_dtype)
            guidance_vec = guidance_vec.to(dtype=model_dtype)
        else:
            img_input = sample
            txt_input = pred_set.current_txt[0]
            vec_input = pred_set.current_vec[0]

        # Get current timestep index for CFG check
        timestep_index = add_set.current_timestep_index[0]

        kwargs = {}
        if "image_proj" in pred_set.model_ref[0].__dict__:
            kwargs = {"image_proj": img_set.image_proj}
        if "ip_scale" in pred_set.model_ref[0].__dict__:
            kwargs.setdefault("ip_scale", img_set.ip_scale)

        # Generate positive prediction
        pred = pred_set.model_ref[0](
            img=img_input,
            img_ids=img_set.img_ids,
            txt=txt_input,
            txt_ids=pred_set.current_txt_ids[0],
            y=vec_input,
            timesteps=t_vec,
            guidance=guidance_vec,
            **kwargs,
        )

        # Apply CFG if enabled and past start timestep
        if (
            pred_set.neg_pred_enabled
            and all([pred_set.current_neg_txt, pred_set.current_neg_txt_ids, pred_set.current_neg_vec])
            and timestep_index >= add_set.timestep_to_start_cfg
        ):
            if not use_autocast:
                neg_txt_input = pred_set.current_neg_txt[0].to(dtype=model_dtype)  # type: ignore
                neg_vec_input = pred_set.current_neg_vec[0].to(dtype=model_dtype)  # type: ignore
            else:
                neg_txt_input = pred_set.current_neg_txt[0]  # type: ignore
                neg_vec_input = pred_set.current_neg_vec[0]  # type: ignore

            neg_pred = pred_set.model_ref[0](
                img=img_input,
                img_ids=img_set.img_ids,
                txt=neg_txt_input,
                txt_ids=pred_set.current_neg_txt_ids[0],  # type: ignore
                y=neg_vec_input,
                timesteps=t_vec,
                guidance=guidance_vec,
                image_proj=img_set.neg_image_proj,
                ip_scale=img_set.neg_ip_scale,
            )
            pred = neg_pred + pred_set.true_gs * (pred - neg_pred)

        # Cache the prediction
        pred_set.cached_prediction[0] = pred
        pred_set.cached_prediction_state[0] = current_state

        return pred

    return get_prediction


@torch.inference_mode()
def denoise(
    model: XFlux,
    settings: DenoiseSettings,
):
    """Denoise using XFlux model with optional ManualTimestepController.\n
    :param model: XFlux model instance
    :param settings: DenoiseSettings containing all denoising configuration parameters"""

    # Extract settings for easier access
    img = settings.img
    img_ids = settings.img_ids
    txt = settings.txt
    txt_ids = settings.txt_ids
    vec = settings.vec
    neg_pred_enabled = settings.neg_pred_enabled
    neg_txt = settings.neg_txt
    neg_txt_ids = settings.neg_txt_ids
    neg_vec = settings.neg_vec
    state = settings.state
    ae = settings.ae
    timesteps = settings.timesteps
    true_gs = settings.true_gs
    timestep_to_start_cfg = settings.timestep_to_start_cfg
    image_proj = settings.image_proj
    neg_image_proj = settings.neg_image_proj
    ip_scale = settings.ip_scale if settings.ip_scale is not None else torch.tensor(1.0)
    neg_ip_scale = settings.neg_ip_scale if settings.neg_ip_scale is not None else torch.tensor(1.0)
    initial_layer_dropout = settings.initial_layer_dropout
    t5 = settings.t5
    clip = settings.clip

    # this is ignored for schnell
    current_layer_dropout = [initial_layer_dropout]
    previous_step_tensor: list[Tensor | None] = [None]  # Store previous step's tensor for masking
    cached_prediction: list[Tensor | None] = [None]  # Cache prediction to avoid duplicate model calls
    cached_prediction_state: list[dict | None] = [None]  # Cache state when prediction was generated
    controller_ref: list[ManualTimestepController | None] = [None]  # Reference to controller for closure access
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
    assert vec is not None, "vec (CLIP embeddings) is required for XFlux1"
    current_vec: list[Tensor] = [vec]

    # Handle negative prompts: XFlux1 supports negative prompts but they're optional
    # If neg_pred_enabled is True, negative prompts must be provided
    if neg_pred_enabled:
        if neg_txt is None or neg_txt_ids is None or neg_vec is None:
            # Generate empty negative prompts if not provided but neg_pred_enabled is True
            if t5 is not None and clip is not None:
                # Use empty string to generate embeddings
                neg_inp = prepare(t5, clip, img, prompt="")
                neg_txt = neg_inp["txt"]
                neg_txt_ids = neg_inp["txt_ids"]
                neg_vec = neg_inp["vec"]
            else:
                # If embedders not available, disable negative prompts
                neg_pred_enabled = False

    if neg_pred_enabled and neg_txt is not None and neg_txt_ids is not None and neg_vec is not None:
        current_neg_txt: list[Tensor] = [neg_txt]  # type: ignore
        current_neg_txt_ids: list[Tensor] = [neg_txt_ids]  # type: ignore
        current_neg_vec: list[Tensor] = [neg_vec]  # type: ignore
    else:
        current_neg_txt: list[Tensor] | None = None
        current_neg_txt_ids: list[Tensor] | None = None
        current_neg_vec: list[Tensor] | None = None
        neg_pred_enabled = False
    current_prompt: list[str | None] = [state.prompt]  # Track current prompt to detect changes

    clear_prediction_cache = create_clear_prediction_cache(cached_prediction, cached_prediction_state)

    recompute_text_embeddings = create_recompute_text_embeddings(img, t5, clip, current_txt, current_txt_ids, current_vec, current_prompt, clear_prediction_cache, is_flux2=False)

    pred_set = GetPredictionSettings(
        model_ref=model_ref,
        state=state,
        current_txt=current_txt,
        current_txt_ids=current_txt_ids,
        current_vec=current_vec,
        cached_prediction=cached_prediction,
        cached_prediction_state=cached_prediction_state,
        neg_pred_enabled=neg_pred_enabled,
        current_neg_txt=current_neg_txt,  # type: ignore
        current_neg_txt_ids=current_neg_txt_ids,  # type: ignore
        current_neg_vec=current_neg_vec,  # type: ignore
        true_gs=true_gs,
    )
    img_set = GetImagePredictionSettings(
        img_ids=img_ids,
        img=img,
        img_cond=None,
        img_cond_seq=None,
        img_cond_seq_ids=None,
        image_proj=image_proj,
        neg_image_proj=neg_image_proj,
        ip_scale=ip_scale,
        neg_ip_scale=neg_ip_scale,
    )
    add_set = AdditionalPredictionSettings(
        timestep_to_start_cfg=timestep_to_start_cfg,
        current_timestep_index=current_timestep_index,
    )
    get_prediction = create_get_prediction_xflux1(pred_set, img_set, add_set)

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

        interaction_context = InteractionContext(
            clear_prediction_cache=clear_prediction_cache,
            rng=rng,
            variation_rng=variation_rng,
            ae=ae,
            t5=t5,
            clip=clip,
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
            # Unpack requires float32, but we'll convert back to correct dtype after
            intermediate = unpack(intermediate.float(), state.height, state.width)

            from nnll.init_gpu import device as default_device

            sync_torch(default_device)
            t1 = time.perf_counter()

            nfo(f"Step time: {t1 - t0:.1f}s")

            if default_device.type == "cuda":
                context = torch.autocast(device_type=default_device.type, dtype=torch.bfloat16)
            else:
                from contextlib import nullcontext

                context = nullcontext()
            with context:
                # When autocast is disabled (MPS), ensure intermediate is in correct dtype for VAE
                if default_device.type != "cuda":
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
