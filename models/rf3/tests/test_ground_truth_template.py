"""Unit tests for the pure noise-scale helpers in rf3.data.ground_truth_template.

These back the noised-ground-truth distogram conditioning. The contracts pinned:

- ``af3_noise_scale_to_noise_level`` / ``af3_noise_level_to_noise_scale`` are
  inverses: level ``t`` and scale ``t^`` relate by ``t^ = 16 e^{1.5t - 1.2}``.
  The scale->level direction clamps the scale to a small ``eps`` so ``scale=0``
  yields a finite (large-negative) level rather than ``-inf``.
- ``wrap_probability_distribution`` folds samples into the requested support:
  modular wrapping into ``[lower, upper)`` when both bounds are finite, and a
  reflection about the single finite bound otherwise. Samples already inside the
  support are returned unchanged.
"""

import math

import pytest
import torch
from rf3.data.ground_truth_template import (
    af3_noise_level_to_noise_scale,
    af3_noise_scale_to_noise_level,
    wrap_probability_distribution,
)

# --- noise-scale <-> noise-level conversions --------------------------------


def test_level_to_scale_known_value():
    # t^ = 16 * exp(1.5*0 - 1.2) at t=0
    assert af3_noise_level_to_noise_scale(0.0).item() == pytest.approx(
        16.0 * math.exp(-1.2)
    )


def test_scale_to_level_is_inverse_of_level_to_scale():
    assert af3_noise_scale_to_noise_level(
        16.0 * math.exp(-1.2)
    ).item() == pytest.approx(0.0, abs=1e-6)


def test_scale_level_roundtrip():
    for scale in (1.0, 4.0, 16.0, 40.0):
        recovered = af3_noise_level_to_noise_scale(
            af3_noise_scale_to_noise_level(scale)
        ).item()
        assert recovered == pytest.approx(scale, rel=1e-5)


def test_scale_to_level_clamps_zero_to_finite():
    # log(0) would be -inf; the eps clamp keeps the level finite.
    level = af3_noise_scale_to_noise_level(0.0).item()
    assert math.isfinite(level)
    assert level < 0


def test_conversions_accept_tensor_input_elementwise():
    scales = torch.tensor([1.0, 4.0, 16.0])
    recovered = af3_noise_level_to_noise_scale(af3_noise_scale_to_noise_level(scales))
    assert recovered.shape == scales.shape
    assert torch.allclose(recovered, scales, rtol=1e-5)


# --- wrap_probability_distribution ------------------------------------------


def test_wrap_both_bounds_modular():
    out = wrap_probability_distribution(
        torch.tensor([2.0, 10.0, 12.5, -1.0]), lower=0.0, upper=10.0
    )
    # in-range unchanged; upper folds to lower (half-open); above/below wrap modulo 10.
    assert out.tolist() == pytest.approx([2.0, 0.0, 2.5, 9.0])


def test_wrap_lower_only_reflects():
    out = wrap_probability_distribution(
        torch.tensor([3.0, -3.0]), lower=0.0, upper=float("inf")
    )
    # >= lower unchanged; below lower reflects across it.
    assert out.tolist() == pytest.approx([3.0, 3.0])


def test_wrap_upper_only_reflects():
    out = wrap_probability_distribution(
        torch.tensor([2.0, 8.0]), lower=float("-inf"), upper=5.0
    )
    # <= upper unchanged; above upper reflects across it.
    assert out.tolist() == pytest.approx([2.0, 2.0])
