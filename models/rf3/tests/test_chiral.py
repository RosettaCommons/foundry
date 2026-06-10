"""Unit tests for rf3.metrics.chiral.

``calc_chiral_metrics_masked`` is the pure tensor core behind RF3's chirality
metrics. The non-obvious contracts pinned here: a chiral center is the dihedral
of four atoms whose ideal angle is the 5th column of ``chirals``; correctness is
a *sign* match between the predicted and ideal dihedral; a center counts only if
all four of its atoms are unmasked; duplicate rows that share a first-atom index
(alternate orderings of the same center) are collapsed to the first occurrence;
and the empty cases (no chirals, or a fully-masked structure) return ``{}``.

``compute_chiral_metrics`` wraps that core: given predicted / ground-truth
``AtomArrayStack``s and (optionally) precomputed chiral features, it splits the
atoms into polymer vs non-polymer, drops ground-truth atoms with NaN
coordinates, and emits ``{category}_{n_chiral_centers,chiral_loss_mean,
percent_correct_chirality}`` keys only for categories that contain a scorable
center. Passing ``chiral_feats`` explicitly bypasses the rdkit/atomworks
feature generation, so the orchestration is exercised here on tiny fixtures.
"""

import math

import numpy as np
import pytest
import torch
from biotite.structure import AtomArrayStack
from rf3.kinematics import get_dih
from rf3.metrics.chiral import calc_chiral_metrics_masked, compute_chiral_metrics


def _chiral_center_coords(theta: float) -> list[list[float]]:
    """Four atoms (a, b, c, d) whose dihedral about the b->c axis is ~theta radians."""
    return [
        [0.0, 1.0, 0.0],  # a
        [0.0, 0.0, 0.0],  # b
        [1.0, 0.0, 0.0],  # c
        [1.0, math.cos(theta), math.sin(theta)],  # d
    ]


def test_no_chiral_centers_returns_empty():
    pred = torch.zeros((1, 4, 3))
    chirals = torch.zeros((0, 5))
    mask = torch.ones(4, dtype=torch.bool)
    assert calc_chiral_metrics_masked(pred, chirals, mask) == {}


def test_empty_mask_returns_empty():
    pred = torch.tensor([_chiral_center_coords(math.pi / 2)])
    chirals = torch.tensor([[0.0, 1.0, 2.0, 3.0, 1.0]])
    mask = torch.zeros(4, dtype=torch.bool)
    assert calc_chiral_metrics_masked(pred, chirals, mask) == {}


def test_correct_chirality_counts_and_loss():
    pred = torch.tensor([_chiral_center_coords(math.pi / 2)])  # dihedral ~ +pi/2
    ideal = 1.0  # positive -> same sign as the predicted dihedral
    chirals = torch.tensor([[0.0, 1.0, 2.0, 3.0, ideal]])
    mask = torch.ones(4, dtype=torch.bool)

    result = calc_chiral_metrics_masked(pred, chirals, mask)

    assert result["n_chiral_centers"].item() == 1
    assert result["percent_correct_chirality"].tolist() == [1.0]

    # chiral_loss_mean = (pred_dih - ideal)^2 summed over valid centers / mask.sum()
    pred_dih = get_dih(pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3])  # [B]
    expected = ((pred_dih - ideal) ** 2).item() / int(mask.sum())
    assert result["chiral_loss_mean"].item() == pytest.approx(expected, abs=1e-5)


def test_wrong_chirality_zero_percent():
    pred = torch.tensor([_chiral_center_coords(math.pi / 2)])  # dihedral ~ +pi/2
    chirals = torch.tensor(
        [[0.0, 1.0, 2.0, 3.0, -1.0]]
    )  # ideal negative -> sign mismatch
    mask = torch.ones(4, dtype=torch.bool)

    result = calc_chiral_metrics_masked(pred, chirals, mask)

    assert result["percent_correct_chirality"].tolist() == [0.0]
    assert result["n_chiral_centers"].item() == 1


def test_masked_atom_excludes_its_center():
    # Two independent centers (atoms 0-3 and 4-7); knock out one atom of the second.
    coords = _chiral_center_coords(math.pi / 2) + _chiral_center_coords(math.pi / 2)
    pred = torch.tensor([coords])  # (1, 8, 3)
    chirals = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0, 1.0],
            [4.0, 5.0, 6.0, 7.0, 1.0],
        ]
    )
    mask = torch.ones(8, dtype=torch.bool)
    mask[7] = False

    result = calc_chiral_metrics_masked(pred, chirals, mask)

    assert result["n_chiral_centers"].item() == 1


def test_duplicate_first_atom_counts_once():
    pred = torch.tensor([_chiral_center_coords(math.pi / 2)])  # (1, 4, 3)
    # Two rows share first-atom index 0 (alternate orderings) -> dedup keeps the first.
    chirals = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0, 1.0],
            [0.0, 2.0, 1.0, 3.0, 1.0],
        ]
    )
    mask = torch.ones(4, dtype=torch.bool)

    result = calc_chiral_metrics_masked(pred, chirals, mask)

    assert result["n_chiral_centers"].item() == 1


def test_batch_dimension_shapes():
    c = _chiral_center_coords(math.pi / 2)
    pred = torch.tensor([c, c])  # (2, 4, 3)
    chirals = torch.tensor([[0.0, 1.0, 2.0, 3.0, 1.0]])
    mask = torch.ones(4, dtype=torch.bool)

    result = calc_chiral_metrics_masked(pred, chirals, mask)

    assert result["chiral_loss_mean"].shape == (2,)
    assert result["percent_correct_chirality"].shape == (2,)


# --- compute_chiral_metrics -------------------------------------------------

# A single chiral center (atoms 0-3) whose predicted dihedral is +pi/2; ideal +1 -> a
# sign match, ideal -1 -> a mismatch.
_CENTER = [_chiral_center_coords(math.pi / 2)]  # (1 model, 4 atoms, 3)
_FEATS_CORRECT = torch.tensor([[0.0, 1.0, 2.0, 3.0, 1.0]])


def _stack(coords, is_polymer) -> AtomArrayStack:
    """AtomArrayStack from (D, L, 3) coords + a per-atom is_polymer flag list."""
    coord = np.asarray(coords, dtype=np.float32)
    stack = AtomArrayStack(depth=coord.shape[0], length=coord.shape[1])
    stack.coord = coord
    stack.set_annotation("is_polymer", np.asarray(is_polymer, dtype=bool))
    return stack


def test_compute_chiral_metrics_polymer_center_correct():
    stack = _stack(_CENTER, [True] * 4)
    out = compute_chiral_metrics(stack, stack, chiral_feats=_FEATS_CORRECT)

    assert out["polymer_n_chiral_centers"] == 1
    assert out["polymer_percent_correct_chirality"] == 1.0
    assert "polymer_chiral_loss_mean" in out
    # No non-polymer atoms -> that category is absent entirely.
    assert "non_polymer_n_chiral_centers" not in out


def test_compute_chiral_metrics_routes_to_non_polymer_category():
    stack = _stack(_CENTER, [False] * 4)
    out = compute_chiral_metrics(stack, stack, chiral_feats=_FEATS_CORRECT)

    assert out["non_polymer_n_chiral_centers"] == 1
    assert "polymer_n_chiral_centers" not in out


def test_compute_chiral_metrics_wrong_sign_is_zero_percent():
    stack = _stack(_CENTER, [True] * 4)
    feats_wrong = torch.tensor([[0.0, 1.0, 2.0, 3.0, -1.0]])  # ideal sign flipped
    out = compute_chiral_metrics(stack, stack, chiral_feats=feats_wrong)

    assert out["polymer_percent_correct_chirality"] == 0.0


def test_compute_chiral_metrics_nan_ground_truth_coord_drops_center():
    pred = _stack(_CENTER, [True] * 4)
    gt = _stack(_CENTER, [True] * 4)
    gt.coord[0, 3, :] = np.nan  # one center atom unresolved in the ground truth

    out = compute_chiral_metrics(pred, gt, chiral_feats=_FEATS_CORRECT)

    # The center has an unresolved atom -> no scorable center in either category.
    assert out == {}
