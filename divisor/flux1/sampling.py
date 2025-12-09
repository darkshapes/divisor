# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->
# adapted BFL Flux code from https://github.com/black-forest-labs/flux

import math
import time
from typing import Callable, Optional

from einops import rearrange, repeat
from nnll.console import nfo
from nnll.constants import ExtensionType
from nnll.init_gpu import sync_torch
from nnll.save_generation import name_save_file_as, save_with_hyperchain
import torch
from torch import Tensor

from divisor.cli_menu import route_choices
from divisor.controller import ManualTimestepController, rng, variation_rng
from divisor.denoise_step import (
    create_clear_prediction_cache,
    create_denoise_step_fn,
    create_get_prediction,
    create_recompute_text_embeddings,
)
from divisor.flux1.model import Flux
from divisor.flux1.text_embedder import HFEmbedder
from divisor.state import (
    DenoiseSettings,
    GetImagePredictionSettings,
    GetPredictionSettings,
    InteractionContext,
)


def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: str | list[str]) -> dict[str, Tensor]:
    """Prepare the text embeddings for the model.\n
    :param t5: T1 embedder
    :param clip: CLIP embedder
    :param img: Image tensor
    :param prompt: Prompt
    :returns: Dictionary of input tensors"""
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, False):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        # ... (rest of function unchanged)
        pass

    # ... (rest of function unchanged)
    return {}


def time_shift(mu: float, sigma: float, t: Tensor) -> Tensor:
    """Adjustable noise schedule. Compress or stretch any schedule to match a dynamic step sequence length.\n
    :param mu: Original schedule parameter.
    :param sigma: Original schedule parameter.
    :returns: Adjusted timestep tensor."""
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15) -> Callable[[float], float]:
    """Linear function for schedule interpolation.\n
    :param x1: ...
    :return: ...
    """
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(num_steps: int, image_seq_len: int) -> list[float]:
    """Generate a schedule of timesteps for the denoising process.

    This is a placeholder implementation that returns a list of ``num_steps``
    linearly spaced values between ``0.0`` and ``1.0``.  Replace with the
    actual schedule logic required by the Flux model.

    :param num_steps: Number of diffusion steps.
    :param image_seq_len: Length of the image sequence (unused in placeholder).
    :returns: List of timestep values.
    """
    if num_steps <= 0:
        return []
    # Simple linear schedule as a fallback.
    return [i / (num_steps - 1) for i in range(num_steps)]


@torch.inference_mode()
def denoise(
    model: Flux,
    settings: DenoiseSettings,
):
    """Denoise using Flux model with optional ManualTimestepController.\n
    :param model: Flux model instance
    :param settings: DenoiseSettings containing all denoising configuration parameters"""

    # ... (setup code unchanged)

    route_processes = InteractionContext(
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
        route_processes,
    )

    # ... (rest of function unchanged)
    return controller.current_sample
