# SPDX-License-Identifier: MIT
# Adapted from https://github.com/Gen-Verse/MMaDA

import torch
from divisor.mmada.modeling_mmada import MMadaModelLM, torch_dtype
from divisor.mmada.spec import get_merged_model_spec, ModelSpecDiffusers
from nnll.init_gpu import device

# VQ_MODEL = MAGVITv2().from_pretrained("showlab/magvitv2").to(DEVICE)


# Use a closure to encapsulate the cache (not a global variable)
def _make_load_model_with_cache():
    """Factory function that creates load_model with encapsulated cache."""
    _model_cache: dict[str, MMadaModelLM] = {}

    def load_model(
        mir_id: str,
        device: torch.device = device,
        compatibility_key: str | None = None,
        force_reload: bool = False,
    ) -> MMadaModelLM:
        """Load a MMaDA model with caching.

        :param mir_id: Model ID (e.g., "model.mldm.mmada")
        :param target_device: Device to load the model on
        :param compatibility_key: Optional compatibility key (e.g., "mixcot") to override repo_id and file_name
        :param force_reload: If True, bypass cache and reload the model
        :returns: Loaded MMaDA model
        """
        # Create cache key
        cache_key = f"{mir_id}:{compatibility_key}" if compatibility_key else mir_id

        # Return cached model if available and not forcing reload
        if not force_reload and cache_key in _model_cache:
            cached_model = _model_cache[cache_key]
            # Ensure model is on the correct device
            cached_device = next(cached_model.parameters()).device
            if cached_device != device:
                cached_model = cached_model.to(device)
            return cached_model

        spec: ModelSpecDiffusers = get_merged_model_spec(mir_id, compatibility_key=compatibility_key)
        spec.params.llm_model_path = spec.repo_id

        # Load the model using from_pretrained
        # The from_pretrained method will automatically store repo_id in the config
        model = MMadaModelLM.from_pretrained(spec.repo_id, trust_remote_code=True, torch_dtype=torch_dtype)  # type: ignore

        # Move to device and set to eval mode
        model = model.to(device).eval()

        # Cache the model
        _model_cache[cache_key] = model

        return model

    def clear_cache():
        """Clear the model cache."""
        _model_cache.clear()

    # Attach clear_cache to the function for external access if needed
    load_model.clear_cache = clear_cache  # type: ignore

    return load_model


# Create the actual load_model function
load_model = _make_load_model_with_cache()
