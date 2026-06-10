"""Unit tests for rf3.model.layers.outer_product.

``OuterProductMean`` / ``OuterProductMean_AF3`` turn an MSA embedding ``[B, N, L, c]``
into a pair representation ``[B, L, L, c_out]``: LayerNorm the MSA, project to a small
hidden width left/right, take the outer product of the two projections over the hidden
dims and average it across the ``N`` sequence rows (the ``/N`` and the ``einsum`` sum
over ``s`` together form the mean), then project to ``c_out``. ``OuterProductMean``
zero-initialises ``proj_out`` (so it is a no-op at init, the AF-style "start from zero"
trick); the AF3 variant does not.
"""

import torch
from rf3.model.layers.outer_product import OuterProductMean, OuterProductMean_AF3


def test_outer_product_mean_zero_initialized_output():
    torch.manual_seed(0)
    layer = OuterProductMean(d_msa=8, d_pair=5, d_hidden=4)
    out = layer(torch.randn(2, 3, 6, 8))  # [B, N, L, d_msa]
    assert out.shape == (2, 6, 6, 5)  # [B, L, L, d_pair]
    # proj_out is zero-initialised, so the block contributes nothing until trained.
    assert bool((out == 0).all())
    assert bool((layer.proj_left.bias == 0).all())
    assert bool((layer.proj_right.bias == 0).all())


def test_outer_product_af3_output_shape():
    torch.manual_seed(0)
    layer = OuterProductMean_AF3(c_msa_embed=8, c_outer_product=4, c_out=5)
    assert layer(torch.randn(2, 3, 6, 8)).shape == (2, 6, 6, 5)


def test_outer_product_af3_matches_mean_einsum():
    torch.manual_seed(0)
    layer = OuterProductMean_AF3(c_msa_embed=8, c_outer_product=4, c_out=5)
    msa = torch.randn(2, 3, 6, 8)
    B, N, L = msa.shape[:3]
    normed = layer.norm(msa)
    left = layer.proj_left(normed)
    right = layer.proj_right(normed) / float(N)
    expected = layer.proj_out(
        torch.einsum("bsli,bsmj->blmij", left, right).reshape(B, L, L, -1)
    )
    assert torch.allclose(layer(msa), expected, atol=1e-6)


def test_outer_product_mean_is_invariant_to_duplicate_rows():
    # The /N normalisation makes the block a true mean over sequence rows: N identical
    # rows give the same output as a single copy of that row.
    torch.manual_seed(0)
    layer = OuterProductMean_AF3(c_msa_embed=8, c_outer_product=4, c_out=5)
    row = torch.randn(2, 1, 6, 8)
    assert torch.allclose(layer(row), layer(row.repeat(1, 4, 1, 1)), atol=1e-6)
