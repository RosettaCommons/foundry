"""Unit tests for rf3.model.layers.layer_utils shape helpers.

These are the structural building blocks the diffusion stack composes:

- ``MultiDimLinear`` is an ``nn.Linear`` whose output is reshaped from a flat
  ``prod(out_shape)`` vector back into ``x.shape[:-1] + out_shape``; its weight is
  re-initialised with Xavier-uniform (overriding ``nn.Linear``'s default).
- ``Transition`` is a SwiGLU feed-forward block: ``linear_3(silu(linear_1(LN(X))) *
  linear_2(LN(X)))``, all projections bias-free, output width equal to the input.
- ``AdaLN`` is adaptive layer-norm — affine-free LayerNorm of the content ``Ai``
  modulated by a sigmoid gain and a bias, both linear projections of LayerNorm'd
  conditioning ``Si``: ``sigmoid(W_g·LN(Si)) * LN_affine_free(Ai) + W_b·LN(Si)``.
- ``create_batch_dimension_if_not_present(n)`` decorates a function expecting an
  ``n``-dim batched arg so it also accepts an ``(n-1)``-dim unbatched arg, inserting a
  singleton batch dim before the call and stripping it from the result afterwards.
"""

import math

import pytest
import torch
import torch.nn.functional as F
from rf3.model.layers.layer_utils import (
    AdaLN,
    MultiDimLinear,
    Transition,
    create_batch_dimension_if_not_present,
)

# --- MultiDimLinear ---------------------------------------------------------


def test_multidim_linear_reshapes_output_to_out_shape():
    torch.manual_seed(0)
    layer = MultiDimLinear(8, (3, 4))
    # Leading dims of the input are preserved; the feature dim becomes out_shape.
    assert layer(torch.randn(2, 5, 8)).shape == (2, 5, 3, 4)
    assert layer(torch.randn(7, 8)).shape == (7, 3, 4)
    # Underlying Linear projects to the flattened width.
    assert layer.out_features == 12
    assert layer.weight.shape == (12, 8)


def test_multidim_linear_is_flat_linear_then_reshape():
    torch.manual_seed(0)
    layer = MultiDimLinear(8, (3, 4))
    x = torch.randn(2, 5, 8)
    expected = F.linear(x, layer.weight, layer.bias).reshape(2, 5, 3, 4)
    assert torch.allclose(layer(x), expected, atol=1e-6)


def test_multidim_linear_weight_is_xavier_bounded():
    torch.manual_seed(0)
    layer = MultiDimLinear(8, (3, 4))
    # Xavier-uniform draws from [-bound, bound], bound = sqrt(6 / (fan_in + fan_out)).
    bound = math.sqrt(6.0 / (8 + 12))
    assert layer.weight.abs().max().item() <= bound


# --- Transition -------------------------------------------------------------


def test_transition_preserves_channel_width():
    torch.manual_seed(0)
    block = Transition(n=2, c=6)
    assert block(torch.randn(2, 7, 6)).shape == (2, 7, 6)
    # Projections are bias-free and the hidden width is n*c.
    assert block.linear_1.bias is None
    assert block.linear_3.bias is None
    assert block.linear_1.weight.shape == (12, 6)
    assert block.linear_3.weight.shape == (6, 12)


def test_transition_matches_swiglu_gating():
    torch.manual_seed(0)
    block = Transition(n=2, c=6)
    x = torch.randn(2, 7, 6)
    ln = block.layer_norm_1(x)
    expected = block.linear_3(F.silu(block.linear_1(ln)) * block.linear_2(ln))
    assert torch.allclose(block(x), expected, atol=1e-6)


# --- AdaLN ------------------------------------------------------------------


def test_adaln_output_shape_and_affine_free_content_norm():
    block = AdaLN(c_a=6, c_s=4)
    out = block(torch.randn(2, 5, 6), torch.randn(2, 5, 4))
    assert out.shape == (2, 5, 6)
    # Content LayerNorm is affine-free; the conditioning LayerNorm drops its bias.
    assert block.ln_a.weight is None and block.ln_a.bias is None
    assert block.ln_s.bias is None


def test_adaln_matches_gain_bias_modulation():
    torch.manual_seed(0)
    block = AdaLN(c_a=6, c_s=4)
    Ai, Si = torch.randn(2, 5, 6), torch.randn(2, 5, 4)
    s = block.ln_s(Si)
    gain = block.to_gain(s)
    expected = gain * block.ln_a(Ai) + block.to_bias(s)
    assert torch.allclose(block(Ai, Si), expected, atol=1e-6)
    # The gain is a sigmoid, so modulation is bounded to (0, 1).
    assert (gain > 0).all() and (gain < 1).all()


# --- create_batch_dimension_if_not_present ----------------------------------


def test_batch_dim_inserted_and_stripped_for_unbatched_arg():
    seen = {}

    @create_batch_dimension_if_not_present(3)
    def double(z):
        seen["ndim"] = z.ndim
        return z * 2

    x = torch.randn(5, 8)
    out = double(x)
    # The wrapped function sees a 3-D arg, but the singleton batch dim is stripped back off.
    assert seen["ndim"] == 3
    assert out.shape == (5, 8)
    assert torch.equal(out, x * 2)


def test_batch_dim_passes_through_already_batched_arg():
    seen = {}

    @create_batch_dimension_if_not_present(3)
    def double(z):
        seen["ndim"] = z.ndim
        return z * 2

    x = torch.randn(2, 5, 8)
    out = double(x)
    assert seen["ndim"] == 3
    assert out.shape == (2, 5, 8)
    assert torch.equal(out, x * 2)


def test_batch_dim_rejects_wrong_rank():
    @create_batch_dimension_if_not_present(3)
    def identity(z):
        return z

    # ndim 1 is neither the batched (3) nor unbatched (2) rank.
    with pytest.raises(Exception, match="must have 2 or 3 dimensions"):
        identity(torch.randn(8))
