# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Noise generation functions for Flux models."""

import math
from typing import Any, Optional

from nnll.init_gpu import device
import torch
from torch import Tensor

from divisor.contents import get_dtype
from divisor.controller import rng


def get_noise(
    num_samples: int,
    height: int,
    width: int,
    seed: int,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | None = None,
    version_2: bool = False,
) -> Tensor:
    """Generate noise tensor for Flux models.\n
    :param num_samples: Number of samples to generate
    :param height: Height of the image
    :param width: Width of the image
    :param dtype: Data type of the noise
    :param seed: Seed for the random number generator
    :param device: Device to generate the noise on
    :param model_type: Model type - "flux1" or "flux2" (default: "flux1")
    :returns: Noise tensor with shape appropriate for the model type

    Flux1 shape: (num_samples, 16, 2 * ceil(height/16), 2 * ceil(width/16))
    Flux2 shape: (num_samples, 128, height // 16, width // 16)
    """

    generator_device = rng._torch_generator.device if rng._torch_generator is not None else torch.device("cpu")
    rng._torch_generator.manual_seed(seed)  # type: ignore # reset seed
    if version_2:
        shape = (num_samples, 128, height // 16, width // 16)
    else:
        shape = (num_samples, 16, 2 * math.ceil(height / 16), 2 * math.ceil(width / 16))
    noise = torch.randn(shape, dtype=dtype, generator=rng._torch_generator, device=generator_device)

    if device is not None and generator_device != device:
        noise = noise.to(device)
    return noise


def prepare_noise_for_model(
    height: int,
    width: int,
    seed: int,
    t5: Optional[Any] = None,
    clip: Optional[Any] = None,
    prompt: Optional[str] = None,
    num_samples: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | None = None,
    version_2: bool = False,
) -> Tensor:
    """Generate noise and convert to 3D format for model input.\n
    Generates 4D noise tensor and converts it from (batch, channels, height, width) format to\n
    (batch, sequence_length, features) format based on model type.\n
    :param height: Height of the image
    :param width: Width of the image
    :param seed: Seed for random number generation
    :param t5: Optional T5 embedder instance (required for Flux1/XFlux1)
    :param clip: Optional CLIP embedder instance (required for Flux1/XFlux1)
    :param prompt: Optional prompt string (required for Flux1/XFlux1)
    :param num_samples: Number of samples to generate (default: 1)
    :param dtype: Data type of the noise (default: torch.bfloat16)
    :param device: Device to generate the noise on (default: None)
    :param version_2: Whether to use Flux2 format (default: False)
    :returns: 3D tensor with shape (batch, sequence_length, features)"""

    noise_4d = get_noise(
        num_samples=num_samples,
        height=height,
        width=width,
        seed=seed,
        dtype=dtype,
        device=device,
        version_2=version_2,
    )

    if t5 is not None and clip is not None and prompt is not None:
        from divisor.flux1.sampling import prepare

        inp = prepare(t5, clip, noise_4d, prompt=prompt)
        return inp["img"]  # 3D format: (batch, sequence_length, features)
    else:
        from divisor.flux2.sampling import batched_prc_img

        noise_3d, _ = batched_prc_img(noise_4d)  # 4D -> 3D: Ignore x_ids as controller doesn't need them
        return noise_3d


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t, generator=None):
    noise = torch.zeros_like(t).uniform_(0, 1, generator=generator)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1, generator=None):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t, generator=generator)).argmax(dim=dim)


def add_gumbel_noise(logits, temperature):
    """
    Adds Gumbel noise to logits for stochastic sampling.
    Equivalent to argmax(logits + temperature * G) where G ~ Gumbel(0,1).
    This version is more numerically stable than a version involving exp() and division.
    """
    if abs(temperature) < 1e-9:  # Effectively zero temperature
        return logits

    max_device_precision = get_dtype(device, max_precision=True)
    logits = logits.to(max_device_precision)
    noise = torch.rand_like(logits, dtype=max_device_precision)
    # Standard Gumbel noise: -log(-log(U)), U ~ Uniform(0,1) Add small epsilon for numerical stability inside logs

    standard_gumbel_noise = -torch.log(-torch.log(noise + 1e-20) + 1e-20)
    return logits + temperature * standard_gumbel_noise


def alt_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    arXiv:2409.02908 low-precision Gumbel Max improves MDM perplexity score but reduces generation quality.
    Thus, we use float64... unless mps, in which case we must use float32
    """
    precision = get_dtype(device)
    if temperature == 0:
        return logits
    logits = logits.to(precision)
    noise = torch.rand_like(logits, dtype=precision)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise
