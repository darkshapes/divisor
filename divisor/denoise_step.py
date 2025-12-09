# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Denoising step functions for interactive/manual controller-based denoising."""

from typing import Any, Callable, Optional

from einops import repeat
from nnll.init_gpu import device
import torch
from torch import Tensor

from divisor.state import GetImagePredictionSettings, GetPredictionSettings
from divisor.variant import apply_variation_noise

# Try to import model types for type checking
try:
    from divisor.flux1.model import Flux
    from divisor.flux2.model import Flux2
except ImportError:
    Flux = Any
    Flux2 = Any


def create_clear_prediction_cache(
    cached_prediction: list[Optional[Tensor]],
    cached_prediction_state: list[Optional[dict]],
) -> Callable[[], None]:
    """Create a function to clear the prediction cache.\n
    :param cached_prediction: Mutable list containing cached prediction
    :param cached_prediction_state: Mutable list containing cached prediction state
    :return: Function that clears the cache"""

    def clear_prediction_cache():
        """Empty the prediction cache.\n"""
        cached_prediction[0] = None
        cached_prediction_state[0] = None

    return clear_prediction_cache


def create_recompute_text_embeddings(
    img: Tensor,
    t5: Optional[Any],
    clip: Optional[Any],
    current_txt: list[Tensor],
    current_txt_ids: list[Tensor],
    current_vec: list[Tensor],
    current_prompt: list[Optional[str]],
    clear_prediction_cache: Callable[[], None],
    is_flux2: bool = False,
    text_embedder: Optional[Any] = None,
) -> Callable[[str], None]:
    """Create a function to recompute text embeddings when prompt changes.\n
    Supports both Flux1 (T5+CLIP) and Flux2 (Mistral) architectures.
    :param img: Image tensor for batch size reference
    :param t5: T5 embedder (Flux1 only, optional)
    :param clip: CLIP embedder (Flux1 only, optional)
    :param current_txt: Mutable list containing current text embeddings
    :param current_txt_ids: Mutable list containing current text IDs
    :param current_vec: Mutable list containing current CLIP embeddings (Flux1) or None (Flux2)
    :param current_prompt: Mutable list containing current prompt
    :param clear_prediction_cache: Function to clear prediction cache
    :param is_flux2: Whether this is for Flux2 model (uses different embedder)
    :param text_embedder: Text embedder for Flux2 (Mistral, optional)
    :return: Function that recomputes text embeddings"""

    def recompute_text_embeddings(prompt: str) -> None:
        """Recompute text embeddings when prompt changes.\n
        :param prompt: New prompt text"""
        bs = img.shape[0]
        prompt_list = [prompt] if isinstance(prompt, str) else prompt

        if is_flux2:
            # Flux2 uses Mistral embedder
            if text_embedder is None:
                return
            new_txt = text_embedder(prompt_list).to(img.device)
            if new_txt.shape[0] == 1 and bs > 1:
                new_txt = repeat(new_txt, "1 ... -> bs ...", bs=bs)
            # Flux2 uses 4D position IDs (t, h, w, l)
            # Generate IDs using the same approach as flux2/sampling.py
            try:
                from divisor.flux2.sampling import batched_prc_txt

                new_txt, new_txt_ids = batched_prc_txt(new_txt)
            except ImportError:
                # Fallback: create simple IDs if import fails
                # This matches the structure expected by Flux2
                _l = new_txt.shape[1]
                coords = {
                    "t": torch.arange(1),
                    "h": torch.arange(1),  # dummy dimension
                    "w": torch.arange(1),  # dummy dimension
                    "l": torch.arange(_l),
                }

                new_txt_ids = torch.cartesian_prod(coords["t"], coords["h"], coords["w"], coords["l"])
                if bs > 1:
                    new_txt_ids = new_txt_ids.unsqueeze(0).repeat(bs, 1, 1)
                new_txt_ids = new_txt_ids.to(new_txt.device)

            current_txt[0] = new_txt.to(img.device)
            current_txt_ids[0] = new_txt_ids.to(img.device)
            # Flux2 doesn't use separate CLIP embeddings
            if current_vec:
                current_vec[0] = None  # type: ignore
        else:
            # Flux1 uses T5 + CLIP
            if t5 is None or clip is None:
                return

            # Compute new embeddings
            new_txt = t5(prompt_list)
            if new_txt.shape[0] == 1 and bs > 1:
                new_txt = repeat(new_txt, "1 ... -> bs ...", bs=bs)
            new_txt_ids = torch.zeros(bs, new_txt.shape[1], 3)

            new_vec = clip(prompt_list)
            if new_vec.shape[0] == 1 and bs > 1:
                new_vec = repeat(new_vec, "1 ... -> bs ...", bs=bs)

            # Update embeddings and move to correct device
            current_txt[0] = new_txt.to(img.device)
            current_txt_ids[0] = new_txt_ids.to(img.device)
            current_vec[0] = new_vec.to(img.device)

        current_prompt[0] = prompt

        # Clear prediction cache since embeddings changed
        clear_prediction_cache()

    return recompute_text_embeddings


def _is_flux2_model(model: Any) -> bool:
    """Check if model is Flux2 type.\n
    :param model: Model instance to check
    :return: True if Flux2, False if Flux1"""
    try:
        from divisor.flux2.model import Flux2

        return isinstance(model, Flux2)
    except (ImportError, TypeError):
        # Fallback: check by class name or signature
        model_class_name = model.__class__.__name__
        return "Flux2" in model_class_name or "flux2" in model_class_name.lower()


def create_get_prediction(pred_set: GetPredictionSettings, img_set: GetImagePredictionSettings) -> Callable[[Tensor, float, float, Optional[list[int]]], Tensor]:
    """Create a function to generate model prediction with caching.\n
    :param config: GetPredictionSettings containing all configuration parameters
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
            pred_set.cached_prediction[0] is not None
            and pred_set.cached_prediction_state[0] is not None
            and pred_set.cached_prediction_state[0]["sample_hash"] == current_state["sample_hash"]
            and pred_set.cached_prediction_state[0]["t_curr"] == current_state["t_curr"]
            and pred_set.cached_prediction_state[0]["guidance"] == current_state["guidance"]
            and pred_set.cached_prediction_state[0]["layer_dropout"] == current_state["layer_dropout"]
        ):
            return pred_set.cached_prediction[0]

        # Generate new prediction
        # When autocast is disabled (MPS), ensure all inputs are in correct dtype (bfloat16)
        # Get model dtype to ensure inputs match
        # Safely get model dtype, handling Mock objects in tests
        try:
            model_dtype = next(pred_set.model_ref[0].parameters()).dtype
        except (TypeError, StopIteration, AttributeError):
            # Fallback: use sample dtype if we can't get model dtype (for Mock objects in tests)
            model_dtype = sample.dtype
        use_autocast = device.type == "cuda"

        # Ensure sample is in correct dtype before any operations
        if not use_autocast:
            sample = sample.to(dtype=model_dtype)

        t_vec = torch.full((sample.shape[0],), t_curr, dtype=sample.dtype, device=sample.device)
        img_input = sample
        img_input_ids = img_set.img_ids

        if img_set.img_cond is not None:
            # Ensure img_cond matches sample dtype before concatenation
            img_cond_converted = img_set.img_cond.to(dtype=model_dtype) if not use_autocast else img_set.img_cond
            img_input = torch.cat((sample, img_cond_converted), dim=-1)
        if img_set.img_cond_seq is not None:
            assert img_set.img_cond_seq_ids is not None, "You need to provide either both or neither of the sequence conditioning"
            # Ensure img_cond_seq matches dtype before concatenation
            img_cond_seq_converted = img_set.img_cond_seq.to(dtype=model_dtype) if not use_autocast else img_set.img_cond_seq
            img_input = torch.cat((img_input, img_cond_seq_converted), dim=1)
            img_input_ids = torch.cat((img_input_ids, img_set.img_cond_seq_ids), dim=1)

        # Determine model type and prepare inputs accordingly
        is_flux2 = _is_flux2_model(pred_set.model_ref[0])

        if is_flux2:
            # Flux2 model signature: model(x=..., x_ids=..., timesteps=..., ctx=..., ctx_ids=..., guidance=..., layer_dropouts=...)
            guidance_vec = torch.full((img_set.img.shape[0],), pred_set.state.guidance, device=img_set.img.device, dtype=img_set.img.dtype)

            if not use_autocast:
                # MPS: Convert all inputs to model dtype (bfloat16) before processing
                img_input = img_input.to(dtype=model_dtype)
                ctx_input = pred_set.current_txt[0].to(dtype=model_dtype)
                t_vec = t_vec.to(dtype=model_dtype)
                guidance_vec = guidance_vec.to(dtype=model_dtype)
            else:
                ctx_input = pred_set.current_txt[0]

            # Flux2 uses x, x_ids, ctx, ctx_ids instead of img, img_ids, txt, txt_ids, y
            pred = pred_set.model_ref[0](
                x=img_input,
                x_ids=img_input_ids,
                timesteps=t_vec,
                ctx=ctx_input,
                ctx_ids=pred_set.current_txt_ids[0],
                guidance=guidance_vec,
                layer_dropouts=layer_dropouts_val,
            )
        else:
            # Flux1 model signature: model(img=..., img_ids=..., txt=..., txt_ids=..., y=..., timesteps=..., guidance=..., layer_dropouts=...)
            guidance_vec = (torch.full((img_set.img.shape[0],), pred_set.state.guidance, device=img_set.img.device, dtype=img_set.img.dtype) * 0.0) * 0.0

            if not use_autocast:
                # MPS: Convert all inputs to model dtype (bfloat16) before processing
                img_input = img_input.to(dtype=model_dtype)

                if pred_set.neg_pred_enabled and all([pred_set.current_neg_txt, pred_set.current_neg_txt_ids, pred_set.current_neg_vec]):
                    txt_input = pred_set.current_neg_txt[0].to(dtype=model_dtype)  # type: ignore
                    vec_input = pred_set.current_neg_vec[0].to(dtype=model_dtype)  # type: ignore
                else:
                    txt_input = pred_set.current_txt[0].to(dtype=model_dtype)
                    vec_input = pred_set.current_vec[0].to(dtype=model_dtype)
                t_vec = t_vec.to(dtype=model_dtype)
                guidance_vec = guidance_vec.to(dtype=model_dtype)
            else:
                if pred_set.neg_pred_enabled and all([pred_set.current_neg_txt, pred_set.current_neg_txt_ids, pred_set.current_neg_vec]):
                    txt_input = pred_set.current_neg_txt[0]  # type: ignore
                    vec_input = pred_set.current_neg_vec[0]  # type: ignore
                else:
                    txt_input = pred_set.current_txt[0]
                    vec_input = pred_set.current_vec[0]

            # Use current embeddings (which may have been updated if prompt changed)
            pred = pred_set.model_ref[0](
                img=img_input,
                img_ids=img_input_ids,
                txt=txt_input,
                txt_ids=pred_set.current_txt_ids[0],
                y=vec_input,
                timesteps=t_vec,
                guidance=guidance_vec,
                layer_dropouts=layer_dropouts_val,
            )
            if pred_set.neg_pred_enabled and all([pred_set.current_neg_txt, pred_set.current_neg_txt_ids, pred_set.current_neg_vec]):
                neg_pred = pred_set.model_ref[0](
                    img=img_input,
                    img_ids=img_input_ids,
                    txt=pred_set.current_neg_txt[0],  # type: ignore
                    txt_ids=pred_set.current_neg_txt_ids[0],  # type: ignore
                    y=pred_set.current_neg_vec[0],  # type: ignore
                    timesteps=t_vec,
                    guidance=guidance_vec,
                    layer_dropouts=layer_dropouts_val,
                )
                pred = neg_pred + pred_set.true_gs * (pred - neg_pred)

        if img_input_ids is not None:
            pred = pred[:, : sample.shape[1]]

        # Cache the prediction
        pred_set.cached_prediction[0] = pred
        pred_set.cached_prediction_state[0] = current_state

        return pred

    return get_prediction


def create_denoise_step_fn(
    controller_ref: list[Optional[Any]],
    current_layer_dropout: list[Optional[list[int]]],
    previous_step_tensor: list[Optional[Tensor]],
    get_prediction: Callable[[Tensor, float, float, Optional[list[int]]], Tensor],
) -> Callable[[Tensor, float, float, float], Tensor]:
    """Create a denoising step function for the controller.\n
    :param controller_ref: Mutable list containing controller reference
    :param current_layer_dropout: Mutable list containing current layer dropout
    :param previous_step_tensor: Mutable list containing previous step tensor
    :param get_prediction: Function to get model prediction
    :return: Denoising step function"""

    def denoise_step_fn(sample: Tensor, t_curr: float, t_prev: float, guidance_val: float) -> Tensor:
        """Single denoising step function for the controller.\n
        :param sample: Current sample tensor
        :param t_curr: Current timestep
        :param t_prev: Previous timestep
        :param guidance_val: Guidance value
        :returns: Updated sample tensor"""

        if controller_ref[0] is not None:
            use_mask = controller_ref[0].use_previous_as_mask
            layer_dropouts = controller_ref[0].current_state.layer_dropout
        else:
            layer_dropouts = current_layer_dropout[0]
            use_mask = False
        pred = get_prediction(sample, t_curr, guidance_val, layer_dropouts)
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

        # Standard noise addition
        result = sample + dt * pred

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

    return denoise_step_fn
