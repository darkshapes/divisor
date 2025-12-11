# SPDX-License-Identifier: MIT
# Adapted from https://github.com/Gen-Verse/MMaDA

import torch
from divisor.mmada.modeling_mmada import MMadaModelLM, torch_dtype
from divisor.mmada.spec import ModelSpecDiffusers, configs as mmada_configs
from divisor.spec import get_model_spec
from nnll.init_gpu import device


def load_diffusers_model(
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

    spec: ModelSpecDiffusers = get_model_spec(mir_id, configs=mmada_configs)
    spec.params.llm_model_path = spec.repo_id
    model = MMadaModelLM.from_pretrained(spec.repo_id, torch_dtype=torch_dtype)  # type: ignore
    model = model.to(device).eval()
    return model
