"""Unit tests for rf3.metrics.distogram pure helpers.

Covers the geometry/loss primitives and the ``ComparisonConfig`` value type:

- ``bin_distances`` turns coordinates into bucketized pairwise distances; the
  self-distance falls in bin 0, distances at/above ``max_distance`` (and NaN
  coordinates, which are coerced to a large value) land in the top bin
  ``n_bins``, and the result is symmetric.
- ``masked_distogram_cross_entropy_loss`` is the per-pair cross-entropy averaged
  over the masked entries (plus a 1e-4 floor); with uniform logits every cell is
  ``log(n_bins)``, so the mask selects which cells enter the average.
- ``ComparisonConfig`` is symmetric in ``token_a``/``token_b`` for equality,
  hashing, and set de-duplication; ``create_distogram_mask`` builds the
  token-pair boolean mask, honoring that symmetry and the intra/inter constraint.
"""

import math

import numpy as np
import pytest
import torch
from biotite.structure import AtomArray
from rf3.metrics.distogram import (
    ComparisonConfig,
    bin_distances,
    masked_distogram_cross_entropy_loss,
)

# --- bin_distances ----------------------------------------------------------


def test_bin_distances_diagonal_zero_and_symmetric():
    coords = torch.tensor([[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]]])
    bins = bin_distances(coords)

    assert bins.shape == (1, 3, 3)
    assert torch.equal(torch.diagonal(bins[0]), torch.zeros(3, dtype=bins.dtype))
    assert torch.equal(bins[0], bins[0].T)
    # Farther pair -> higher-or-equal bin.
    assert bins[0, 0, 2] >= bins[0, 0, 1]


def test_bin_distances_beyond_max_is_top_bin():
    n_bins = 64
    coords = torch.tensor([[[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]]])  # 100 > max (22)
    bins = bin_distances(coords, n_bins=n_bins)
    assert bins[0, 0, 1].item() == n_bins


def test_bin_distances_nan_coord_goes_to_top_bin():
    n_bins = 64
    coords = torch.tensor([[[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]]])
    bins = bin_distances(coords, n_bins=n_bins)
    # NaN distances are coerced to 9999 before bucketize -> top bin.
    assert bins[0, 0, 1].item() == n_bins


def test_bin_distances_batched_shape():
    coords = torch.arange(30, dtype=torch.float).reshape(2, 5, 3)
    bins = bin_distances(coords)
    assert bins.shape == (2, 5, 5)


# --- masked_distogram_cross_entropy_loss ------------------------------------


def test_masked_ce_uniform_logits_equals_log_nbins():
    D, I, n_bins = 1, 2, 4
    inp = torch.zeros(D, I, I, n_bins)  # uniform -> CE = log(n_bins) per cell
    target = torch.zeros(D, I, I, dtype=torch.long)
    out = masked_distogram_cross_entropy_loss(inp, target, torch.ones(I, I))

    assert out.shape == (D,)
    assert out.item() == pytest.approx(math.log(n_bins) + 1e-4, abs=1e-4)


def test_masked_ce_masking_changes_average():
    D, I, n_bins = 1, 2, 4
    inp = torch.zeros(D, I, I, n_bins)
    inp[0, 0, 0, 0] = 20.0  # cell (0,0): confident + correct -> CE ~ 0
    target = torch.zeros(D, I, I, dtype=torch.long)

    full = masked_distogram_cross_entropy_loss(inp, target, torch.ones(I, I))
    # Three uniform cells (log n_bins) + one ~0 cell, averaged over 4.
    assert full.item() == pytest.approx(3 * math.log(n_bins) / 4 + 1e-4, abs=1e-3)

    mask = torch.ones(I, I)
    mask[0, 0] = 0.0  # exclude the confident cell -> average over the 3 uniform cells
    excl = masked_distogram_cross_entropy_loss(inp, target, mask)
    assert excl.item() == pytest.approx(math.log(n_bins) + 1e-4, abs=1e-3)


def test_masked_ce_batched_shape():
    D, I, n_bins = 3, 2, 5
    inp = torch.zeros(D, I, I, n_bins)
    target = torch.zeros(D, I, I, dtype=torch.long)
    out = masked_distogram_cross_entropy_loss(inp, target, torch.ones(I, I))
    assert out.shape == (D,)


# --- ComparisonConfig -------------------------------------------------------


def test_comparison_config_symmetric_equality():
    a = ComparisonConfig("atomized", "non_atomized", "inter")
    b = ComparisonConfig("non_atomized", "atomized", "inter")
    assert a == b
    assert hash(a) == hash(b)


def test_comparison_config_relationship_distinguishes():
    assert ComparisonConfig("all", "all", "intra") != ComparisonConfig(
        "all", "all", "inter"
    )


def test_comparison_config_set_dedup():
    a = ComparisonConfig("atomized", "non_atomized", "inter")
    b = ComparisonConfig("non_atomized", "atomized", "inter")
    assert len({a, b}) == 1


def test_comparison_config_not_equal_to_other_type():
    assert ComparisonConfig("all", "all", "all") != "all_by_all"


def test_comparison_config_str():
    assert (
        str(ComparisonConfig("atomized", "non_atomized", "inter"))
        == "atomized_by_non_atomized_inter"
    )
    assert str(ComparisonConfig("all", "all", "all")) == "all_by_all"


# --- ComparisonConfig.create_distogram_mask ---------------------------------


def _token_rep_array() -> AtomArray:
    """Four token-representative atoms: 0,1 atomized in chain 0; 2,3 not, in chain 1."""
    arr = AtomArray(4)
    arr.coord = np.zeros((4, 3), dtype=np.float32)
    arr.set_annotation("atomize", np.array([True, True, False, False]))
    arr.set_annotation("pn_unit_iid", np.array([0, 0, 1, 1]))
    return arr


def test_create_mask_same_type_outer_product():
    mask = ComparisonConfig("atomized", "atomized", "all").create_distogram_mask(
        _token_rep_array()
    )
    assert mask.shape == (4, 4)
    assert mask[0, 1] and mask[1, 0]  # both atomized
    assert not mask[0, 2]  # atomized vs non-atomized
    assert not mask[2, 3]  # both non-atomized


def test_create_mask_cross_type_symmetric():
    mask = ComparisonConfig("atomized", "non_atomized", "all").create_distogram_mask(
        _token_rep_array()
    )
    assert mask[0, 2] and mask[2, 0]  # atomized <-> non-atomized, symmetric
    assert not mask[0, 1]  # both atomized
    assert not mask[2, 3]  # both non-atomized


def test_create_mask_intra_vs_inter():
    arr = _token_rep_array()
    intra = ComparisonConfig("all", "all", "intra").create_distogram_mask(arr)
    inter = ComparisonConfig("all", "all", "inter").create_distogram_mask(arr)
    assert intra[0, 1] and not intra[0, 2]  # same-chain pairs only
    assert inter[0, 2] and not inter[0, 1]  # cross-chain pairs only
