"""Unit tests for rf3.metrics.metric_utils.

Pure helpers behind RF3's confidence metrics (pLDDT / PAE / PDE). The non-obvious
contracts pinned here: `find_bin_midpoints` returns the `num_bins` centres of
equal bins spanning `[0, max_distance]`; `unbin_logits` takes the softmax
expectation over those midpoints, so a one-hot logit recovers its bin's
midpoint; the chainwise / interface mask builders turn a per-residue chain-label
array into intra-chain and cross-chain boolean masks; and the subsampled
mean / min reduce a batched matrix over a boolean pair mask, with the min
explicitly excluding unscored entries.
"""

import numpy as np
import torch
from rf3.metrics.metric_utils import (
    compute_mean_over_subsampled_pairs,
    compute_min_over_subsampled_pairs,
    create_chainwise_masks_1d,
    create_chainwise_masks_2d,
    create_interface_masks_2d,
    find_bin_midpoints,
    spread_batch_into_dictionary,
    unbin_logits,
)

# --- find_bin_midpoints -------------------------------------------------------


def test_find_bin_midpoints_are_equal_bin_centres():
    # 5 bins over [0, 10] -> centres at 1, 3, 5, 7, 9.
    mp = find_bin_midpoints(10.0, 5)
    assert torch.allclose(mp, torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0]))


def test_find_bin_midpoints_count_matches_num_bins():
    assert find_bin_midpoints(32.0, 64).shape == (64,)


# --- unbin_logits -------------------------------------------------------------


def test_unbin_logits_recovers_bin_midpoint():
    # A near-one-hot distribution on bin index 2 unbins to that bin's midpoint (5.0).
    num_bins = 5
    logits = torch.full((1, num_bins, 2, 2), -50.0)
    logits[:, 2] = 50.0
    out = unbin_logits(logits, 10.0, num_bins)
    assert out.shape == (1, 2, 2)
    assert torch.allclose(out, torch.full((1, 2, 2), 5.0), atol=1e-3)


# --- chainwise / interface masks ----------------------------------------------


def test_create_chainwise_masks_1d():
    masks = create_chainwise_masks_1d(np.array(["A", "A", "B"]))
    assert masks["A"].tolist() == [True, True, False]
    assert masks["B"].tolist() == [False, False, True]


def test_create_chainwise_masks_2d_is_intra_chain_outer_product():
    masks = create_chainwise_masks_2d(np.array(["A", "A", "B"]))
    assert masks["A"].int().tolist() == [[1, 1, 0], [1, 1, 0], [0, 0, 0]]
    assert masks["B"].int().tolist() == [[0, 0, 0], [0, 0, 0], [0, 0, 1]]


def test_create_interface_masks_2d_is_symmetric_cross_chain():
    masks = create_interface_masks_2d(np.array(["A", "A", "B"]))
    assert list(masks.keys()) == [("A", "B")]
    assert masks[("A", "B")].int().tolist() == [[0, 0, 1], [0, 0, 1], [1, 1, 0]]


# --- subsampled reductions ----------------------------------------------------


def test_compute_mean_over_subsampled_pairs():
    mat = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    pairs = torch.tensor([[True, False], [False, True]])
    # mean over the two scored (diagonal) entries: (1 + 4) / 2
    assert torch.allclose(
        compute_mean_over_subsampled_pairs(mat, pairs), torch.tensor([2.5]), atol=1e-4
    )


def test_compute_min_over_subsampled_pairs():
    mat = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    pairs = torch.tensor([[True, False], [False, True]])
    # min over the two scored (diagonal) entries: min(1, 4)
    assert compute_min_over_subsampled_pairs(mat, pairs).tolist() == [1.0]


def test_compute_min_excludes_unscored_entries():
    # The unscored off-diagonal holds the global min (0.0); masking must exclude it
    # so the result is the smallest *scored* entry (5.0), not 0.0.
    mat = torch.tensor([[[5.0, 0.0], [3.0, 9.0]]])
    pairs = torch.tensor([[True, False], [False, True]])
    assert compute_min_over_subsampled_pairs(mat, pairs).tolist() == [5.0]


# --- spread_batch_into_dictionary ---------------------------------------------


def test_spread_batch_into_dictionary():
    assert spread_batch_into_dictionary(torch.tensor([1.0, 2.0])) == {0: 1.0, 1: 2.0}
