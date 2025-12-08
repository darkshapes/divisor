# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from dataclasses import dataclass


@dataclass
class SDARParams:
    mask_id: int = 151669
    gen_length: int = 128
    block_length: int = 8
    denoising_steps: int = 8
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    remasking_strategy: str = "low_confidence_dynamic"
    confidence_threshold: float = 0.85
    stopping_criteria_idx: int | None = None


@dataclass
class ModelSpec:
    repo_id: str
    file_name: str
    params: SDARParams


configs = {
    "model.trado": {
        "*": ModelSpec(
            repo_id="exdysa/TraDo-4B-Instruct",
            file_name="model-00001-of-00002.safetensors",
            params=SDARParams(
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
        )
    },
    "model.sdar": {
        "*": ModelSpec(
            repo_id="exdysa/SDAR-1.7B-Chat",
            file_name="model.safetensors",
            params=SDARParams(
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
        )
    },
}
