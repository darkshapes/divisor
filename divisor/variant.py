# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""Variation noise functions for denoising process."""

import math

import torch
from torch import Tensor

from divisor.controller import variation_rng


def mix_noise(from_noise: Tensor, to_noise: Tensor, strength: float, variation_method: str = "linear") -> Tensor:
    """Mix two noise tensors using specified method.\n
    :param from_noise: Source noise tensor
    :param to_noise: Target noise tensor to mix towards
    :param strength: Mixing strength (0.0 to 1.0)
    :param variation_method: Mixing method ('linear' or 'slerp')
    :returns: Mixed noise tensor
    """
    to_noise = to_noise.to(from_noise.device)

    if variation_method == "slerp":
        # Spherical linear interpolation
        # Flatten for norm calculation (works with any tensor shape)
        from_flat = from_noise.flatten(start_dim=1)
        to_flat = to_noise.flatten(start_dim=1)

        from_norm = torch.norm(from_flat, dim=1, keepdim=True)
        to_norm = torch.norm(to_flat, dim=1, keepdim=True)

        # Normalize
        from_unit = from_flat / (from_norm + 1e-8)
        to_unit = to_flat / (to_norm + 1e-8)

        # Dot product for angle
        dot = (from_unit * to_unit).sum(dim=1, keepdim=True)
        dot = torch.clamp(dot, -1.0, 1.0)
        theta = torch.acos(dot)

        # Slerp formula
        sin_theta = torch.sin(theta)
        w1 = torch.sin((1 - strength) * theta) / (sin_theta + 1e-8)
        w2 = torch.sin(strength * theta) / (sin_theta + 1e-8)

        # Apply weights and reshape back
        mixed_flat = w1 * from_flat + w2 * to_flat
        mixed_noise = mixed_flat.reshape(from_noise.shape)
    else:
        # Linear interpolation
        mixed_noise = (1 - strength) * from_noise + strength * to_noise
        # Scale factor correction for variance preservation
        scale_factor = math.sqrt((1 - strength) ** 2 + strength**2)
        mixed_noise = mixed_noise / (scale_factor + 1e-8)

    return mixed_noise


def apply_variation_noise(
    latent_sample: Tensor,
    variation_seed: int | None,
    variation_strength: float,
    mask: Tensor | None = None,
    variation_method: str = "linear",
) -> Tensor:
    """Apply variation noise to the latent sample.\n
    :param latent_sample: Current sample tensor in 3D sequence format [batch, sequence, features]
    :param variation_seed: Seed for variation noise generation, or None to disable
    :param variation_strength: Strength of variation (0.0 to 1.0)
    :param mask: Optional mask tensor for selective application
    :param variation_method: Mixing method ('linear' or 'slerp')
    :returns: Sample with variation noise applied
    """
    if variation_seed is None or variation_strength == 0.0:
        return latent_sample

    # Set seed for variation noise generation
    if variation_seed is not None:
        variation_rng.next_seed(variation_seed)
    else:
        variation_seed = variation_rng.next_seed()

    # Get generator and its device
    variation_generator = variation_rng._torch_generator
    generator_device = variation_generator.device if variation_generator is not None else torch.device("cpu")

    # Generate variation noise matching the sample shape
    # Create on generator's device first (required for MPS compatibility)
    variation_noise = torch.randn(
        latent_sample.shape,
        dtype=latent_sample.dtype,
        layout=latent_sample.layout,
        generator=variation_generator,
        device=generator_device,
    )

    # Move to sample's device if different
    if generator_device != latent_sample.device:
        variation_noise = variation_noise.to(latent_sample.device)

    if mask is None:
        # Simple mixing without mask
        result = mix_noise(latent_sample, variation_noise, variation_strength, variation_method)
    else:
        # Apply mask: mask=1 uses mixed noise, mask=0 uses original
        mixed_noise_result = mix_noise(latent_sample, variation_noise, variation_strength, variation_method)
        result = (mask == 1).float() * mixed_noise_result + (mask == 0).float() * latent_sample

    return result
