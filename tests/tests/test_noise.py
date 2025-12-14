# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""
Tests for the ``perlin_pyramid_noise`` toggle in `divisor.noise`.
"""

import builtins
from unittest import mock

import pytest
import torch


def _dummy_pyramid_noise(
    shape: tuple[int, int, int, int],
    dtype: torch.dtype,
    generator: torch.Generator,
    device: torch.device,
    **kwargs,
) -> torch.Tensor:
    """
    Return a tensor filled with the constant value ``42``.
    The constant makes it trivial to assert that the function was used.
    """
    return torch.full(shape, 5, dtype=dtype, device=device)


def test_perlin_pyramid_flag_calls_helper(monkeypatch):
    """When ``pyramid=True`` the internal helper is invoked."""
    import divisor.noise as noise_mod

    monkeypatch.setattr(noise_mod, "perlin_pyramid_noise", _dummy_pyramid_noise)
    shape = (1, 16, 128, 128)  # Flux‑1 style (batch, ch, h, w)
    seed = 12345
    out = noise_mod.get_noise(
        num_samples=shape[0],
        height=shape[2] * 16,  # original image size before the 1/16 down‑sample
        width=shape[3] * 16,
        seed=seed,
        dtype=torch.float32,
        device=torch.device("cpu"),
        version_2=False,  # Flux‑1 shape
        perlin=True,  # <-- the flag we are testing
    )

    assert isinstance(out, torch.Tensor)
    assert out.shape == shape
    assert out.dtype == torch.float32
    assert torch.all(out == 5), "The dummy pyramid helper should fill the tensor with 5"


def test_perlin_pyramid_flag_not_called_when_off(monkeypatch):
    """When ``pyramid=False`` the pyramid helper must stay untouched."""
    import divisor.noise as noise_mod

    # Wrap the real helper with a spy so we can count calls.
    spy = mock.Mock(wraps=noise_mod.perlin_pyramid_noise)
    monkeypatch.setattr(noise_mod, "perlin_pyramid_noise", spy)

    shape = (2, 8, 32, 32)  # a different size on purpose
    seed = 9876
    out = noise_mod.get_noise(
        num_samples=shape[0],
        height=shape[2] * 16,
        width=shape[3] * 16,
        seed=seed,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        version_2=False,
        perlin=False,  # flag OFF
    )

    # The spy must not have been invoked.
    spy.assert_not_called()

    # Still, the function should return a *valid* noise tensor (the default
    # Gaussian path).  We only check shape/dtype here – the exact values are
    # random by design.
    assert out.shape == shape
    assert out.dtype == torch.bfloat16


def test_perlin_pyramid_statistics_differ():
    """The pyramid version blends several octaves, therefore its variance\n
    (before the final normalisation) is typically larger than the variance\n
    of a single Gaussian field.  After normalisation both have unit variance,\n
    but the *raw* (pre‑normalisation) tensors differ noticeably."""
    from divisor.noise import get_noise

    shape = (1, 16, 128, 128)
    seed = 777

    torch.manual_seed(seed)
    gen_gauss = torch.Generator().manual_seed(seed)
    gauss = get_noise(
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

    gauss_abs = gauss.abs().mean().item()
    pyr_abs = pyr.abs().mean().item()

    assert abs(gauss_abs - pyr_abs) > 0.05, "Pyramid noise should have a noticeably different absolute‑value distribution than plain Gaussian noise"
