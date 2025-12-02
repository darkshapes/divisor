# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Noise generation functions for Flux models."""

import math
from typing import Any, Optional

import torch
from torch import Tensor

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
    # Get the generator's device to ensure compatibility
    generator_device = rng._torch_generator.device if rng._torch_generator is not None else torch.device("cpu")

    if version_2:
        # Flux2: (num_samples, 128, height // 16, width // 16)
        shape = (
            num_samples,
            128,
            height // 16,
            width // 16,
        )
    else:
        # Flux1: (num_samples, 16, 2 * ceil(height/16), 2 * ceil(width/16))
        # allow for packing
        shape = (
            num_samples,
            16,
            2 * math.ceil(height / 16),
            2 * math.ceil(width / 16),
        )

    # Create tensor on generator's device first (required for MPS compatibility)
    noise = torch.randn(
        shape,
        dtype=dtype,
        generator=rng._torch_generator,
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
