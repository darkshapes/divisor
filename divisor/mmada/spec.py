# SPDX-License-Identifier: MIT
# Adapted from https://github.com/Gen-Verse/MMaDA

from dataclasses import dataclass
from typing import Union

from huggingface_hub import constants

from divisor.mmada.modeling_mmada import MMadaConfig

cache_folder_named = constants.HF_HUB_CACHE


@dataclass
class CompatibilitySpecDiffusers:
    """Compatibility specification for alternative model variants."""

    repo_id: str


@dataclass
class InitialParams:
    """Default initialization parameters for MMaDA models."""

    steps: int
    gen_length: int
    block_length: int
    temperature: float
    cfg_scale: float
    remasking_strategy: str
    mask_id: int
    max_position_embeddings: int
    max_text_len: int


@dataclass
class ModelSpecDiffusers:
    """Model specification for MMaDA models."""

    repo_id: str
    params: MMadaConfig
    init: InitialParams


configs: dict[str, dict[str, ModelSpecDiffusers | CompatibilitySpecDiffusers]] = {
    "model.mldm.mmada": {
        "*": ModelSpecDiffusers(
            repo_id="Gen-Verse/MMaDA-8B-Base",
            init=InitialParams(
                steps=256,
                gen_length=512,
                block_length=128,
                temperature=1.0,
                cfg_scale=0.0,
                remasking_strategy="low_confidence",
                mask_id=126336,
                max_position_embeddings=2048,
                max_text_len=512,
            ),
            params=MMadaConfig(
                vocab_size=50257,
                llm_vocab_size=50257,
                llm_model_path="",
                codebook_size=8192,
                num_vq_tokens=1024,
                num_new_special_tokens=0,
            ),
        ),
        "mixcot": CompatibilitySpecDiffusers(
            repo_id="Gen-Verse/MMaDA-8B-MixCoT",
        ),
    },
}
