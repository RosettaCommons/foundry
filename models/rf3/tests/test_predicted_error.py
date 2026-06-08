"""Unit tests for rf3.metrics.predicted_error pure helpers.

Covers ``compute_ptm`` (the PAE -> predicted-TM reduction) and the thin
``ComputePTM`` Metric wrapper that reshapes the per-batch scores into a dict.

``compute_ptm`` takes a per-pair distribution over distance-error bins
``pae`` of shape ``[D, I, I, n_bins]``, softmaxes over the bins, weights each
bin by the TM term ``1 / (1 + (bin_center / d0)**2)`` (which is largest for the
nearest bin), averages over the columns selected by ``to_calculate``, and
returns the per-token maximum -> ``[D]``. So a distribution concentrated on the
nearest bin scores highest, a uniform one scores the mean weight, and one on
the farthest bin scores lowest; all scores lie in ``(0, 1]``.
"""

import torch
from rf3.metrics.predicted_error import ComputePTM, compute_ptm

D, I, N_BINS = 2, 5, 64


def _onehot_pae(bin_idx: int) -> torch.Tensor:
    """A `[D, I, I, N_BINS]` logit tensor whose softmax concentrates on `bin_idx`."""
    pae = torch.zeros(D, I, I, N_BINS)
    pae[..., bin_idx] = 50.0
    return pae


def test_compute_ptm_shape_and_bounds():
    # Deterministic, non-degenerate logits.
    pae = torch.arange(D * I * I * N_BINS, dtype=torch.float32).reshape(D, I, I, N_BINS)
    ptm = compute_ptm(pae, to_calculate=None)

    assert ptm.shape == (D,)
    assert (ptm > 0).all()
    assert (ptm <= 1.0).all()


def test_compute_ptm_orders_by_confidence():
    # Mass on the nearest bin -> highest TM weight; uniform -> mean weight;
    # mass on the farthest bin -> lowest weight.
    ptm_near = compute_ptm(_onehot_pae(0), to_calculate=None)
    ptm_uniform = compute_ptm(torch.zeros(D, I, I, N_BINS), to_calculate=None)
    ptm_far = compute_ptm(_onehot_pae(N_BINS - 1), to_calculate=None)

    assert (ptm_near > ptm_uniform).all()
    assert (ptm_uniform > ptm_far).all()


def test_compute_ptm_none_to_calculate_equals_all_ones():
    pae = _onehot_pae(3)
    ptm_default = compute_ptm(pae, to_calculate=None)
    ptm_explicit = compute_ptm(pae, to_calculate=torch.ones(I, I, dtype=torch.bool))

    assert torch.allclose(ptm_default, ptm_explicit)


def test_compute_ptm_subset_of_columns_changes_score():
    # Make one column (token 0) confident-near and the rest confident-far; then
    # restricting `to_calculate` to that column must score higher than averaging
    # over all columns.
    pae = _onehot_pae(N_BINS - 1)
    pae[:, :, 0, :] = 0.0
    pae[:, :, 0, 0] = 50.0  # column 0 -> nearest bin

    only_col0 = torch.zeros(I, I, dtype=torch.bool)
    only_col0[:, 0] = True

    ptm_col0 = compute_ptm(pae, to_calculate=only_col0)
    ptm_all = compute_ptm(pae, to_calculate=None)

    assert (ptm_col0 > ptm_all).all()


def test_compute_ptm_metric_returns_per_batch_dict():
    pae = _onehot_pae(0)
    # asym_id is unused by ComputePTM.compute (it forwards pae with to_calculate=None).
    out = ComputePTM().compute(pae=pae, asym_id=torch.zeros(I, dtype=torch.long))

    assert set(out) == {f"ptm_{i}" for i in range(D)}
    assert all(0.0 < v <= 1.0 for v in out.values())
