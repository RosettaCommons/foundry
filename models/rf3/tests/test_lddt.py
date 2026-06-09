"""Unit tests for rf3.metrics.lddt.calc_lddt.

``calc_lddt`` is the scientific core all the LDDT ``Metric`` subclasses delegate
to. For each model it scores the atom pairs whose ground-truth distance lies in
(0, cutoff), where both atoms are resolved (``crd_mask``) and the two atoms sit
in different tokens, then averages the fraction of those pairs whose predicted
distance is preserved within the standard 0.5 / 1.0 / 2.0 / 4.0 Å thresholds. A
perfect prediction scores 1.0; errors above 4 Å on every pair score 0.0; when no
pair survives the filters the score is 0.0 (0 / eps).
"""

import torch
from rf3.metrics.lddt import calc_lddt


def _coords(points: list[list[float]]) -> torch.Tensor:
    """One model with the given atom coordinates → shape (1, L, 3)."""
    return torch.tensor([points], dtype=torch.float32)


def test_perfect_prediction_scores_one():
    coords = _coords([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    tok = torch.arange(4)
    lddt = calc_lddt(coords, coords.clone(), mask, tok)
    assert lddt.shape == (1,)
    assert torch.allclose(lddt, torch.ones(1), atol=1e-4)


def test_large_error_scores_zero():
    gt = _coords([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    pred = gt * 10.0  # every pairwise distance off by far more than 4 Å
    mask = torch.ones(1, 4, dtype=torch.bool)
    tok = torch.arange(4)
    lddt = calc_lddt(pred, gt, mask, tok)
    assert torch.allclose(lddt, torch.zeros(1), atol=1e-4)


def test_unresolved_atoms_are_masked_out():
    gt = _coords([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    pred = gt.clone()
    pred[0, 3] = torch.tensor([100.0, 0, 0])  # atom 3 badly placed...
    # ...but atom 3 is unresolved, so all of its pairs are dropped from scoring.
    mask = torch.tensor([[True, True, True, False]])
    tok = torch.arange(4)
    lddt = calc_lddt(pred, gt, mask, tok)
    assert torch.allclose(lddt, torch.ones(1), atol=1e-4)


def test_same_token_pairs_excluded():
    gt = _coords([[0, 0, 0], [1, 0, 0]])
    pred = gt.clone()  # perfect
    mask = torch.ones(1, 2, dtype=torch.bool)
    # Distinct tokens: the single 1 Å pair is scored → 1.0.
    assert torch.allclose(
        calc_lddt(pred, gt, mask, torch.tensor([0, 1])), torch.ones(1), atol=1e-4
    )
    # Same token: the only pair is excluded → no valid pairs → 0.0.
    assert torch.allclose(
        calc_lddt(pred, gt, mask, torch.tensor([0, 0])), torch.zeros(1), atol=1e-4
    )


def test_distance_cutoff_excludes_far_pairs():
    gt = _coords([[0, 0, 0], [20, 0, 0]])  # 20 Å apart
    pred = gt.clone()
    mask = torch.ones(1, 2, dtype=torch.bool)
    tok = torch.tensor([0, 1])
    # Default cutoff 15 Å → pair is out of range → excluded → 0.0.
    assert torch.allclose(calc_lddt(pred, gt, mask, tok), torch.zeros(1), atol=1e-4)
    # Cutoff 30 Å → pair is in range and perfect → 1.0.
    assert torch.allclose(
        calc_lddt(pred, gt, mask, tok, distance_cutoff=30.0), torch.ones(1), atol=1e-4
    )


def test_batched_models():
    gt = _coords([[0, 0, 0], [1, 0, 0], [2, 0, 0]]).expand(2, 3, 3).contiguous()
    pred = gt.clone()
    pred[1] = pred[1] * 10.0  # second model is bad
    mask = torch.ones(2, 3, dtype=torch.bool)
    tok = torch.arange(3)
    lddt = calc_lddt(pred, gt, mask, tok)
    assert lddt.shape == (2,)
    assert torch.allclose(lddt[0], torch.tensor(1.0), atol=1e-4)
    assert torch.allclose(lddt[1], torch.tensor(0.0), atol=1e-4)
