"""Unit tests for rf3.model.layers.mlff.ConformerEmbeddingWeightedAverage.

The module compresses per-conformer atom-level embeddings ``[n_conformers, n_atom, d]``
into a single per-atom embedding ``[n_atom, c_atom]``: a shared MLP downcasts each
conformer's features to ``c_atompair``, the conformers are flattened into one vector per
atom, and a final (bias-free, zero-initialised) linear projects to ``c_atom``. The
zero-init makes the block a no-op at the start of training (output ≈ 0, for a clean
residual add). The forward also pins two input contracts: the conformer count must match
``n_conformers`` exactly, and an over-wide feature dim is truncated to
``atom_level_embedding_dim`` while an under-wide one is rejected.
"""

import pytest
import torch
from rf3.model.layers.mlff import ConformerEmbeddingWeightedAverage


def _layer():
    return ConformerEmbeddingWeightedAverage(
        atom_level_embedding_dim=16, c_atompair=4, c_atom=8, n_conformers=3
    )


def test_output_shape_and_zero_initialized():
    layer = _layer()
    out = layer(torch.randn(3, 5, 16))  # [n_conformers, n_atom, d]
    assert out.shape == (5, 8)  # [n_atom, c_atom]
    # The final projection is zero-initialised, so the block contributes nothing at init.
    assert bool((layer.conformers_to_atom_single_embedding[0].weight == 0).all())
    assert bool((out == 0).all())


def test_subsets_oversized_feature_dim():
    # A feature dim wider than atom_level_embedding_dim is truncated to it, not rejected.
    assert _layer()(torch.randn(3, 5, 24)).shape == (5, 8)


def test_rejects_undersized_feature_dim():
    with pytest.raises(ValueError, match="is less than the expected dimension"):
        _layer()(torch.randn(3, 5, 8))


def test_rejects_wrong_conformer_count():
    with pytest.raises(AssertionError, match="Number of conformers must be consistent"):
        _layer()(torch.randn(2, 5, 16))
