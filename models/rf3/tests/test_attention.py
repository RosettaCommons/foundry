"""Unit tests for rf3.model.layers.attention triangle-update blocks (vanilla path).

Both blocks map a pair representation ``[B, L, L, d_pair]`` to the same shape (for a
residual add) and, on CPU, run their vanilla PyTorch path (cuEquivariance is off).

- ``TriangleAttention`` zero-initialises its output projection ``to_out`` so the block is
  the identity (output 0) at the start of training; ``start_node`` switches the attention
  axis (rows vs. transposed columns) but preserves the output shape either way.
- ``TriangleMultiplication`` validates its ``direction`` (``"outgoing"``/``"incoming"``)
  and, when the cuEquivariance kernel is requested, requires ``d_pair == d_hidden``; the
  vanilla path lifts that constraint.
"""

import pytest
import torch
from rf3.model.layers.attention import TriangleAttention, TriangleMultiplication

# --- TriangleAttention ------------------------------------------------------


def test_triangle_attention_preserves_shape():
    pair = torch.randn(1, 6, 6, 8)
    for start_node in (True, False):
        layer = TriangleAttention(d_pair=8, n_head=2, d_hidden=4, start_node=start_node)
        assert layer(pair).shape == pair.shape


def test_triangle_attention_zero_initialized():
    torch.manual_seed(0)
    layer = TriangleAttention(d_pair=8, n_head=2, d_hidden=4)
    # to_out is zero-initialised so the residual add starts as the identity.
    assert bool((layer.to_out.weight == 0).all()) and bool(
        (layer.to_out.bias == 0).all()
    )
    assert bool((layer(torch.randn(1, 6, 6, 8)) == 0).all())


# --- TriangleMultiplication -------------------------------------------------


def test_triangle_multiplication_preserves_shape():
    pair = torch.randn(1, 6, 6, 8)
    for direction in ("outgoing", "incoming"):
        # Vanilla path lifts the d_pair == d_hidden cuEquivariance constraint.
        layer = TriangleMultiplication(
            d_pair=8, d_hidden=4, direction=direction, use_cuequivariance=False
        )
        assert layer(pair).shape == pair.shape


def test_triangle_multiplication_rejects_invalid_direction():
    with pytest.raises(ValueError, match="direction must be 'outgoing' or 'incoming'"):
        TriangleMultiplication(d_pair=8, direction="sideways", use_cuequivariance=False)


def test_triangle_multiplication_cuequivariance_requires_matching_dims():
    with pytest.raises(AssertionError, match="requires d_pair == d_hidden"):
        TriangleMultiplication(d_pair=8, d_hidden=4, use_cuequivariance=True)
