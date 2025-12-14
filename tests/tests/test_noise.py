# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""
Tests for the ``perlin_pyramid_noise`` toggle in `divisor.noise`.
"""

import builtins
from unittest import mock

import pytest
import torch


def test_perlin_pyramid_flag_calls_helper(monkeypatch):
    """When ``pyramid=True`` the internal helper is invoked."""
    from divisor import noise

    perlin_pyramid_noise_spy = mock.Mock(wraps=noise.perlin_pyramid_noise)
    monkeypatch.setattr(noise, "perlin_pyramid_noise", perlin_pyramid_noise_spy)
    shape = (1, 16, 64, 64)  # Flux‑1 style (batch, ch, h, w)
    seed = 12345
    latent_shape = (1, 16, 64, 64)  # what the model expects
    out = noise.get_noise(
        num_samples=shape[0],
        height=shape[2],
        width=shape[3],
        seed=seed,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        version_2=False,
        perlin=True,  # flag OFF
    )

    perlin_pyramid_noise_spy.assert_called_once()
    assert isinstance(out, torch.Tensor)
    assert out.shape == shape
    assert out.dtype == torch.float32


def test_perlin_pyramid_flag_not_called_when_off(monkeypatch):
    """When ``pyramid=False`` the pyramid helper must stay untouched."""
    from divisor import noise

    perlin_pyramid_noise_spy = mock.Mock(wraps=noise.perlin_pyramid_noise)
    monkeypatch.setattr(noise, "perlin_pyramid_noise", perlin_pyramid_noise_spy)
    shape = (1, 16, 64, 64)  # a different size on purpose
    seed = 9876
    out = noise.get_noise(
        num_samples=shape[0],
        height=shape[2],
        width=shape[3],
        seed=seed,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        version_2=False,
        perlin=False,  # flag OFF
    )

    # Still, the function should return a *valid* noise tensor (the default
    # Gaussian path).  We only check shape/dtype here – the exact values are
    # random by design.
    perlin_pyramid_noise_spy.assert_not_called()
    assert out.shape == shape
    assert out.dtype == torch.bfloat16


def test_perlin_pyramid_statistics_differ(monkeypatch):
    """The pyramid version blends several octaves, therefore its variance\n
    (before the final normalisation) is typically larger than the variance\n
    of a single Gaussian field.  After normalisation both have unit variance,\n
    but the *raw* (pre‑normalisation) tensors differ noticeably."""
    from divisor import noise
    from divisor.noise import get_noise

    perlin_pyramid_noise_spy = mock.Mock(wraps=noise.perlin_pyramid_noise)
    monkeypatch.setattr(noise, "perlin_pyramid_noise", perlin_pyramid_noise_spy)
    shape = (1, 16, 128, 128)
    seed = 777

    torch.manual_seed(seed)
    gen_gauss = torch.Generator().manual_seed(seed)
    gauss = noise.get_noise(
        num_samples=shape[0],
        height=shape[2] * 16,
        width=shape[3] * 16,
        seed=seed,
        dtype=torch.float32,
        device=torch.device("cpu"),
        version_2=False,
        perlin=False,
    )

    torch.manual_seed(seed)
    gen_pyr = torch.Generator().manual_seed(seed)
    pyr = get_noise(
        num_samples=shape[0],
        height=shape[2] * 16,
        width=shape[3] * 16,
        seed=seed,
        dtype=torch.float32,
        device=torch.device("cpu"),
        version_2=False,
        perlin=True,
    )

    perlin_pyramid_noise_spy.assert_called_once()
    gauss_abs = gauss.abs().mean().item()
    pyr_abs = pyr.abs().mean().item()

    assert abs(gauss_abs - pyr_abs) > 0.05, "Pyramid noise should have a noticeably different absolute‑value distribution than plain Gaussian noise"
