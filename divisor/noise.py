# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Noise generation functions for Flux models."""

import math
from typing import Any, Optional

import torch
from torch import Tensor

from divisor.controller import rng


def _get_noise_shape(
    num_samples: int,
    height: int,
    width: int,
    version_2: bool,
) -> tuple[int, int, int, int]:
    """Return the tensor shape for the requested Flux model.\n
    :param num_samples: Number of samples to generate
    :param height: Height of the image
    :param width: Width of the image
    :param version_2: ``True`` for Flux2, ``False`` for Flux1
    :returns: Shape tuple ``(batch_size, channels, height, width)`` appropriate for the model
    """
    if version_2:
        return (
            num_samples,
            128,
            height // 16,
            width // 16,
        )
    return (
        num_samples,
        16,
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
    )


def perlin_pyramid_noise(
    shape: tuple[int, int, int, int],
    dtype: torch.dtype,
    generator: torch.Generator,
    device: torch.device,
    mode: str = "bicubic",
    discount: float = 0.5,
    max_scale: int = 15,
    octaves: int = 4,
) -> Tensor:
    """High‑resolution “pyramid” Perlin‑style noise.    \n
    :param shape: Desired tensor shape
    :param dtype: Data type for the tensor
    :param generator: ``torch.Generator`` seeded by the caller
    :param device: Device to allocate for noise
    :param pyramid: Generate Perlin‑style noise with pyramid effect when ``True``
    :returns: Noise tensor on ``generator_device``"""
    batch_size, channels, height, width = shape
    original_height, original_width = height, width
    upsampler = torch.nn.Upsample(size=(original_height, original_width), mode=mode).to(device)

    # Base uniform field in the range [-1.73, +1.73] (≈ √3 * 2)
    noise = (torch.rand(shape, dtype=dtype, device=device, generator=generator) - 0.5) * 2 * 1.73
    for index in range(octaves):
        growth_factor = torch.rand(1, device=device, generator=generator).item() * 2 + 2  # → [2,4]

        reshaped_height = min(original_height * max_scale, int(height * (growth_factor**index)))
        reshaped_width = min(original_width * max_scale, int(width * (growth_factor**index)))
        reshaped_shape = (batch_size, channels, reshaped_height, reshaped_width)
        highres_noise = torch.randn(reshaped_shape, dtype=dtype, device=device, generator=generator)
        shrunk_noise_to_blend = upsampler(highres_noise)
        noise = noise + shrunk_noise_to_blend * (discount**index)
        if reshaped_height >= original_height * max_scale or reshaped_width >= original_width * max_scale:
            break

    # Normalise to unit variance (zero‑mean is already guaranteed by the symmetric construction)
    return noise / noise.std()


def get_noise(
    num_samples: int,
    height: int,
    width: int,
    seed: int,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | None = None,
    version_2: bool = False,
    perlin: bool = False,
) -> Tensor:
    """Generate noise tensor for Flux models.\n
    :param num_samples: Number of samples to generate
    :param height: Height of the image
    :param width: Width of the image
    :param dtype: Data type of the noise
    :param seed: Seed for the random number generator
    :param device: Device to generate the noise on
    :param version_2: ``True`` for Flux2 shape, ``False`` for Flux1 shape
    :param perlin: Generate Perlin‑style noise when ``True``
    :returns: Noise tensor with shape appropriate for the model type"""
    generator_device: torch.device = rng._torch_generator.device if rng._torch_generator is not None else torch.device("cpu")
    generator: torch.Generator = rng._torch_generator  # type: ignore[attr-defined]
    generator.manual_seed(seed)
    shape = _get_noise_shape(
        num_samples=num_samples,
        height=height,
        width=width,
        version_2=version_2,
    )
    if perlin:
        noise = perlin_pyramid_noise(
            shape=shape,
            dtype=dtype,
            generator=generator,
            device=generator_device,
        )
    else:
        noise = torch.randn(
            *shape,
            dtype=dtype,
            generator=generator,
            device=generator_device,
        )
    # Move to target device if different
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
    """Generate noise and convert to 3D format for model input.

    Generates 4D noise tensor and converts it from (batch, channels, height, width) format to
    (batch, sequence_length, features) format based on model type.

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
    :returns: 3D tensor with shape (batch, sequence_length, features)
    """
    # Generate 4D noise tensor
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
        # Flux1/XFlux1: Use prepare() to convert 4D noise to 3D format
        from divisor.flux1.sampling import prepare

        inp = prepare(t5, clip, noise_4d, prompt=prompt)
        return inp["img"]  # 3D format: (batch, sequence_length, features)
    else:
        # Flux2: Use batched_prc_img to convert 4D noise to 3D format
        from divisor.flux2.sampling import batched_prc_img

        # batched_prc_img expects (batch, channels, height, width) and returns (batch, sequence_length, features)
        noise_3d, _ = batched_prc_img(noise_4d)  # Ignore x_ids as controller doesn't need them
        return noise_3d
