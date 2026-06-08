"""Unit tests for rf3.metrics.chiral.calc_chiral_metrics_masked.

The pure tensor core behind RF3's chirality metrics. The non-obvious contracts
pinned here: a chiral center is the dihedral of four atoms whose ideal angle is
the 5th column of ``chirals``; correctness is a *sign* match between the
predicted and ideal dihedral; a center counts only if all four of its atoms are
unmasked; duplicate rows that share a first-atom index (alternate orderings of
the same center) are collapsed to the first occurrence; and the empty cases
(no chirals, or a fully-masked structure) return ``{}``.
"""

import math

import pytest
import torch
from rf3.kinematics import get_dih
from rf3.metrics.chiral import calc_chiral_metrics_masked


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
