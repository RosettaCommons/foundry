"""Unit tests for rf3.util_module helpers.

``rbf`` expands distances into a radial-basis (Gaussian) feature vector over
``D_count`` evenly spaced centres ``D_min .. D_min + (D_count-1)*D_sigma``: the
feature at the centre nearest a distance peaks at 1.0 and falls off as a Gaussian
of width ``D_sigma`` (far centres underflow to 0). ``init_lecun_normal`` replaces a
module's weight with a truncated-normal sample (clamped to ±2 before scaling)
whose post-truncation standard deviation is the Lecun value ``sqrt(scale / fan_in)``.
"""

import math

import pytest
import torch
from rf3.util_module import init_lecun_normal, rbf

# Std of a standard normal truncated to [-2, 2]; the source divides by it so the
# scaled sample's std lands at sqrt(scale / fan_in) rather than below it.
_TRUNC_NORMAL_STD = 0.87962566103423978


# --- rbf --------------------------------------------------------------------


def test_rbf_appends_feature_dim():
    assert rbf(torch.rand(5)).shape == (5, 64)
    assert rbf(torch.rand(2, 3), D_count=16).shape == (2, 3, 16)


def test_rbf_gaussian_values_at_known_distance():
    # D == D_min lands exactly on centre 0; later centres are D_sigma apart, so the
    # features at D=0 are exp(-k**2) for k = 0, 1, 2, ...
    vals = rbf(torch.tensor([0.0]), D_min=0.0, D_count=64, D_sigma=0.5)[0]
    assert vals[0].item() == 1.0
    assert vals[1].item() == pytest.approx(math.exp(-1))
    assert vals[2].item() == pytest.approx(math.exp(-4))
    # The nearest centre is the argmax (D=0.5 -> centre 1), and values stay in [0, 1].
    assert int(rbf(torch.tensor([0.5]), D_sigma=0.5).argmax()) == 1
    full = rbf(torch.rand(20) * 30)
    assert (full >= 0).all() and (full <= 1).all()


# --- init_lecun_normal ------------------------------------------------------


def test_init_lecun_normal_returns_module_and_bounded_weight():
    torch.manual_seed(0)
    linear = torch.nn.Linear(128, 64, bias=False)
    result = init_lecun_normal(linear)
    stddev = math.sqrt(1.0 / 128) / _TRUNC_NORMAL_STD  # fan_in = 128
    assert result is linear
    assert isinstance(linear.weight, torch.nn.Parameter)
    assert linear.weight.shape == (64, 128)
    # The truncated normal is clamped to ±2 before the stddev scaling.
    assert linear.weight.abs().max().item() <= 2 * stddev


def test_init_lecun_normal_scales_to_lecun_stddev():
    torch.manual_seed(0)
    linear = torch.nn.Linear(512, 512, bias=False)
    init_lecun_normal(linear)
    # Lecun-normal: post-truncation std ~ sqrt(scale / fan_in).
    assert linear.weight.std().item() == pytest.approx(math.sqrt(1 / 512), rel=0.05)
