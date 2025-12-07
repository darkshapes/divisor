# SPDX-License-Identifier: MIT
# Adapted from https://github.com/Gen-Verse/MMaDA

from dataclasses import dataclass

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


configs = {
    "model.mldm.trado": {
        "*": ModelSpecDiffusers(
            repo_id="Gen-Verse/TraDo-4B-Instruct",
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
        "trado-4b": CompatibilitySpecDiffusers(
            repo_id="Gen-Verse/TraDo-4B-Instruct",
        ),
    },
}


def get_model_spec(mir_id: str, compatibility_key: str | None = None) -> ModelSpecDiffusers | CompatibilitySpecDiffusers | None:
    """Get a Diffusers ModelSpec or CompatibilitySpec for a given model ID.\n
    :param mir_id: Model ID (e.g., "model.mldm.mmada.8b-base")
    :param compatibility_key: Optional compatibility key. If None, returns base ModelSpec.
    :returns: ModelSpec if compatibility_key is None, CompatibilitySpec if provided and available, None if provided but not found
    """
    if mir_id not in configs:
        if compatibility_key is None:
            available = ", ".join(configs.keys())
            raise ValueError(f"Unknown model ID: {mir_id}. Available: {available}")
        return None

    config_dict = configs[mir_id]

    # If compatibility_key is provided, try to get compatibility spec
    if compatibility_key is not None:
        compat_spec = config_dict.get(compatibility_key)
        if compat_spec is None:
            return None
        return compat_spec
    else:
        # Otherwise, return base ModelSpec from "*" key
        if "*" not in config_dict:
            raise ValueError(f"Model {mir_id} does not have a base spec (missing '*' key)")

        base_spec = config_dict["*"]
        if not isinstance(base_spec, ModelSpecDiffusers):
            raise ValueError(f"Model {mir_id} base spec is not a ModelSpec")

        return base_spec


def get_merged_model_spec(mir_id: str, compatibility_key: str | None = None) -> ModelSpecDiffusers:
    """Get a Diffusers ModelSpec with compatibility overrides merged in.\n
    :param mir_id: Model ID (e.g., "model.mldm.mmada" or "model.mldm.mmada:mixcot")
    :param compatibility_key: Optional compatibility key (extracted from mir_id if it contains ':')
    :returns: ModelSpec with compatibility overrides applied
    """
    # Parse compatibility key from mir_id if it contains ':'
    if ":" in mir_id:
        base_mir_id, compat_key = mir_id.split(":", 1)
        compatibility_key = compat_key
        mir_id = base_mir_id

    base_spec = get_model_spec(mir_id)
    if not isinstance(base_spec, ModelSpecDiffusers):
        raise ValueError(f"Model {mir_id} does not have a base ModelSpec")

    if compatibility_key is None:
        return base_spec

    compat_spec = get_model_spec(mir_id, compatibility_key)
    if compat_spec is None:
        raise ValueError(f"Model {mir_id} does not have compatibility spec '{compatibility_key}'")

    if isinstance(compat_spec, CompatibilitySpecDiffusers):
        return ModelSpecDiffusers(
            repo_id=compat_spec.repo_id,
            init=base_spec.init,
            params=base_spec.params,
        )

    return base_spec
