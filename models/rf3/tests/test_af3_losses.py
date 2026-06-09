"""Unit tests for the rf3.loss.af3_losses scientific losses.

- ``distogram_loss`` bins ground-truth representative-atom distances and takes the
  masked cross-entropy of the predicted distogram against those bins: confident
  correct logits → ~0, uniform logits → log(n_classes), and the coordinate mask
  selects which token pairs enter the average.
- ``smoothed_lddt_loss`` is ``1 - <soft LDDT>``, where the four LDDT thresholds
  (0.5/1.0/2.0/4.0 Å) are replaced by sigmoids of (threshold - |Δdistance|). A
  perfect prediction therefore lands at the fixed value ``1 - mean(sigmoid(t))``
  rather than 0; error only increases it. Pairs are scored when the ground-truth
  distance is in (0, cutoff) — cutoff 30 Å if the first token is DNA/RNA, else
  15 Å — both atoms are resolved, and the two atoms are in different tokens.
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from rf3.loss.af3_losses import distogram_loss, smoothed_lddt_loss

# distogram_loss bins with torch.linspace(min, max, bins); bucketize adds a
# catch-all bin, so the target has bins + 1 = 65 classes.
_N_CLASSES = 65

# The soft-LDDT value of a perfect prediction: mean of the four threshold sigmoids
# (|Δ| ≈ 0). loss = 1 - that.
_PERFECT_LDDT_LOSS = 1 - 0.25 * torch.sigmoid(torch.tensor([0.5, 1.0, 2.0, 4.0])).sum()


def _binned(coords: torch.Tensor) -> torch.Tensor:
    """Mirror distogram_loss's internal binning of the pairwise distance map."""
    bins = torch.linspace(2, 22, 64)
    return torch.bucketize(torch.cdist(coords, coords), bins)


# --- distogram_loss ---------------------------------------------------------


def test_distogram_loss_confident_correct_is_near_zero():
    cce = nn.CrossEntropyLoss(reduction="none")
    coords = torch.tensor([[0, 0, 0], [5, 0, 0], [10, 0, 0]], dtype=torch.float32)
    # Sharp logits on the true bin for every pair → cross-entropy ≈ 0.
    pred = F.one_hot(_binned(coords), _N_CLASSES).float() * 20.0
    mask = torch.ones(3, dtype=torch.bool)
    loss = distogram_loss(pred, coords, mask, cce)
    assert loss.item() < 1e-3


def test_distogram_loss_uniform_logits_is_log_num_classes():
    cce = nn.CrossEntropyLoss(reduction="none")
    coords = torch.tensor([[0, 0, 0], [5, 0, 0], [10, 0, 0]], dtype=torch.float32)
    pred = torch.zeros(3, 3, _N_CLASSES)  # equal logits → uniform softmax
    mask = torch.ones(3, dtype=torch.bool)
    loss = distogram_loss(pred, coords, mask, cce)
    assert loss.item() == pytest.approx(math.log(_N_CLASSES), abs=1e-4)


def test_distogram_loss_mask_excludes_unresolved_atoms():
    cce = nn.CrossEntropyLoss(reduction="none")
    coords = torch.tensor([[0, 0, 0], [5, 0, 0], [10, 0, 0]], dtype=torch.float32)
    pred = F.one_hot(_binned(coords), _N_CLASSES).float() * 20.0
    # Corrupt every pair touching atom 2 (its rows/cols become uniform → wrong).
    pred[2, :, :] = 0.0
    pred[:, 2, :] = 0.0
    loss_all = distogram_loss(pred, coords, torch.ones(3, dtype=torch.bool), cce)
    loss_excl_2 = distogram_loss(pred, coords, torch.tensor([True, True, False]), cce)
    # Masking out the corrupted atom drops the error back to ~0.
    assert loss_excl_2.item() < loss_all.item()
    assert loss_excl_2.item() < 1e-3


# --- smoothed_lddt_loss -----------------------------------------------------


def _inputs(coords, tok_idx, is_dna=None, is_rna=None):
    X = torch.tensor([coords], dtype=torch.float32)  # one model, (1, L, 3)
    tok = torch.tensor(tok_idx)
    n_tok = int(tok.max()) + 1
    dna = (
        torch.tensor(is_dna)
        if is_dna is not None
        else torch.zeros(n_tok, dtype=torch.bool)
    )
    rna = (
        torch.tensor(is_rna)
        if is_rna is not None
        else torch.zeros(n_tok, dtype=torch.bool)
    )
    mask = torch.ones(1, X.shape[1], dtype=torch.bool)
    return X, tok, dna, rna, mask


def test_smoothed_lddt_perfect_prediction_is_the_soft_floor():
    X, tok, dna, rna, mask = _inputs(
        [[0, 0, 0], [3, 0, 0], [6, 0, 0], [9, 0, 0]], [0, 1, 2, 3]
    )
    loss = smoothed_lddt_loss(X, X.clone(), mask, dna, rna, tok)
    assert loss.shape == (1,)
    assert torch.allclose(loss, _PERFECT_LDDT_LOSS.reshape(1), atol=1e-3)


def test_smoothed_lddt_error_increases_loss():
    X, tok, dna, rna, mask = _inputs(
        [[0, 0, 0], [3, 0, 0], [6, 0, 0], [9, 0, 0]], [0, 1, 2, 3]
    )
    perfect = smoothed_lddt_loss(X, X.clone(), mask, dna, rna, tok)
    worse = smoothed_lddt_loss(X * 2.0, X, mask, dna, rna, tok)  # distances doubled
    assert worse.item() > perfect.item()


def test_smoothed_lddt_na_widens_distance_cutoff():
    # A single 20 Å pair (between the 15 Å protein cutoff and the 30 Å NA cutoff).
    coords = [[0, 0, 0], [20, 0, 0]]
    X, tok, _, rna, mask = _inputs(coords, [0, 1])
    # Protein: 20 Å is out of range → no scored pairs → soft LDDT 0 → loss 1.0.
    loss_protein = smoothed_lddt_loss(
        X, X.clone(), mask, torch.zeros(2, dtype=torch.bool), rna, tok
    )
    assert torch.allclose(loss_protein, torch.ones(1), atol=1e-4)
    # First token DNA: cutoff widens to 30 Å → the pair is scored (and perfect).
    loss_dna = smoothed_lddt_loss(
        X, X.clone(), mask, torch.tensor([True, False]), rna, tok
    )
    assert torch.allclose(loss_dna, _PERFECT_LDDT_LOSS.reshape(1), atol=1e-3)


def test_smoothed_lddt_excludes_same_token_pairs():
    coords = [[0, 0, 0], [3, 0, 0]]
    X = torch.tensor([coords], dtype=torch.float32)
    mask = torch.ones(1, 2, dtype=torch.bool)
    # Distinct tokens: the 3 Å pair is scored → perfect soft floor.
    loss_distinct = smoothed_lddt_loss(
        X,
        X.clone(),
        mask,
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
        torch.tensor([0, 1]),
    )
    assert torch.allclose(loss_distinct, _PERFECT_LDDT_LOSS.reshape(1), atol=1e-3)
    # Same token: the only pair is excluded → no scored pairs → loss 1.0.
    loss_same = smoothed_lddt_loss(
        X,
        X.clone(),
        mask,
        torch.zeros(1, dtype=torch.bool),
        torch.zeros(1, dtype=torch.bool),
        torch.tensor([0, 0]),
    )
    assert torch.allclose(loss_same, torch.ones(1), atol=1e-4)
