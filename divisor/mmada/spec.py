# SPDX-License-Identifier: MIT
# Adapted from https://github.com/Gen-Verse/MMaDA

from dataclasses import dataclass
from huggingface_hub import constants

from divisor.spec import populate_model_choices
from divisor.mmada.modeling_mmada import MMadaConfig
from divisor.mmada.configuration_sdar import SDARConfig

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
class InitialSDARParams:
    mask_id: int
    gen_length: int
    block_length: int
    denoising_steps: int
    temperature: float
    top_k: int
    top_p: float
    remasking_strategy: str
    confidence_threshold: float


@dataclass
class ModelSpecDiffusers:
    """Model specification for MMaDA models."""

    repo_id: str
    params: MMadaConfig | SDARConfig
    init: InitialParams | InitialSDARParams


configs = [
    {
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
    },
    {
        "model.mldm.sdar": {
            "*": ModelSpecDiffusers(
                repo_id="exdysa/SDAR-1.7B-Chat",
                init=InitialSDARParams(
                    mask_id=151669,
                    gen_length=200,
                    block_length=4,
                    denoising_steps=4,
                    temperature=1.0,
                    top_k=0,
                    top_p=1.0,
                    remasking_strategy="low_confidence_dynamic",
                    confidence_threshold=0.9,
                ),
                params=SDARConfig(
                    vocab_size=151936,
                    hidden_size=4096,
                    intermediate_size=22016,
                    num_hidden_layers=32,
                    num_attention_heads=32,
                    num_key_value_heads=32,
                ),
            ),
            "trado": ModelSpecDiffusers(
                repo_id="exdysa/TraDo-4B-Instruct",
                init=InitialSDARParams(
                    mask_id=151669,
                    gen_length=200,
                    block_length=4,
                    denoising_steps=4,
                    temperature=1.0,
                    top_k=0,
                    top_p=1.0,
                    remasking_strategy="low_confidence_dynamic",
                    confidence_threshold=0.9,
                ),
                params=MMadaConfig(
                    vocab_size=151936,
                    hidden_size=4096,
                    intermediate_size=22016,
                    num_hidden_layers=32,
                    num_attention_heads=32,
                    num_key_value_heads=32,
                ),
            ),
        },
    },
]

# -*- coding: utf-8 -*-
"""
Configuration file
Contains commonly used parameters and settings
"""

# Generation related configuration
GENERATION_CONFIG = {
    "default_timesteps": 64,
    "default_temperature": 1.0,
    "default_cfg_scale": 4.0,
    "default_cfg_img": 4.0,
    "default_seq_len": 1024,
    "default_newline_every": 16,
    "remasking_strategy": "low_confidence",
}

# Image related configuration
IMAGE_CONFIG = {
    "default_height": 512,
    "default_width": 512,
    "max_height": 1024,
    "max_width": 1024,
}

# Special token IDs
SPECIAL_TOKENS = {
    "mask_token": 126336,
    "newline_token": 126084,
    "image_token_offset": 126356,
    "answer_start": 126354,
    "answer_end": 126355,
    "boi": 126349,  # begin of image
    "eoi": 126350,  # end of image
    "uncondition": 126351,
}

# Prompt templates
PROMPT_TEMPLATES = {
    "text_understanding": "You are a multimodal model that can process both text and images. Answer the following question based on the provided images. Analyze each image and combine relevant details to answer.",
    "image_generation": "Generate an image according to the text prompt.",
    "image_editing": "Generate an image applying the following editing instruction based on the original image.",
    "dense_prediction": "Perform dense prediction on the given images.",
    "control_generation": "Generate an image according to the text prompt and the given control image.",
    "subject_generation": "Generate an image according to the text prompt and the given object image.",
    "multi_view": "Generate a view-image based on the given image.",
    "style_transfer": "Transform the current image into the style of the provided image.",
}

# Edit type configuration
EDIT_TYPE_CONFIG = {
    "dense": {"canny": "canny edge map", "hed": "hed edge map", "depth": "depth map", "openpose": "pose estimation map"},
    "supported_types": [
        "canny_pred",
        "hed_pred",
        "depth_pred",
        "openpose_pred",
        "canny_control",
        "hed_control",
        "depth_control",
        "openpose_control",
        "subject_driven",
        "edit",
        "ref_transfer",
        "multi_view",
    ],
}
