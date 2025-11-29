# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Noise generation functions for Flux models."""

import math
from typing import Literal

import torch
from torch import Tensor

from divisor.controller import rng


def get_noise(
    num_samples: int,
    height: int,
    width: int,
    dtype: torch.dtype,
    seed: int,
    device: torch.device | None = None,
    version_2 : bool = False,
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

