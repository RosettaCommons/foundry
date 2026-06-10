"""Unit tests for rf3.metrics.predicted_error pure helpers.

Covers ``compute_ptm`` (the PAE -> predicted-TM reduction), the thin
``ComputePTM`` Metric wrapper that reshapes the per-batch scores into a dict,
and ``ComputeIPTM`` which scores the *inter-chain* interfaces (overall, plus
protein-protein / protein-ligand / ligand-ligand sub-interfaces).

``compute_ptm`` takes a per-pair distribution over distance-error bins
``pae`` of shape ``[D, I, I, n_bins]``, softmaxes over the bins, weights each
bin by the TM term ``1 / (1 + (bin_center / d0)**2)`` (which is largest for the
nearest bin), averages over the columns selected by ``to_calculate``, and
returns the per-token maximum -> ``[D]``. So a distribution concentrated on the
nearest bin scores highest, a uniform one scores the mean weight, and one on
the farthest bin scores lowest; all scores lie in ``(0, 1]``. An empty
``to_calculate`` (no selected columns) averages nothing and scores 0.
"""

import pytest
import torch
from rf3.metrics.predicted_error import ComputeIPTM, ComputePTM, compute_ptm

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


# --- ComputeIPTM ------------------------------------------------------------

# Two chains of two tokens each; only the inter-chain pairs (asym 0 vs asym 1) are scored.
_TWO_CHAINS = torch.tensor([0, 0, 1, 1])
_IPTM_FAMILIES = (
    "iptm",
    "iptm_protein_protein",
    "iptm_protein_ligand",
    "iptm_ligand_ligand",
)


def _uniform_pae(n: int) -> torch.Tensor:
    """Flat `[D, n, n, N_BINS]` logits -> uniform per-pair bin distribution."""
    return torch.zeros(D, n, n, N_BINS)


def test_iptm_keys_cover_every_interface_type_per_model():
    out = ComputeIPTM().compute(
        pae=_uniform_pae(4), asym_id=_TWO_CHAINS, is_ligand=torch.tensor([0, 0, 1, 1])
    )

    assert set(out) == {f"{fam}_{i}" for fam in _IPTM_FAMILIES for i in range(D)}


def test_iptm_all_protein_zeroes_ligand_interfaces():
    # No ligand tokens -> the protein-ligand and ligand-ligand masks are empty (score 0),
    # and the protein-protein interface covers exactly the inter-chain pairs == overall iPTM.
    out = ComputeIPTM().compute(
        pae=_uniform_pae(4),
        asym_id=_TWO_CHAINS,
        is_ligand=torch.zeros(4, dtype=torch.long),
    )

    for i in range(D):
        assert out[f"iptm_protein_ligand_{i}"] == 0.0
        assert out[f"iptm_ligand_ligand_{i}"] == 0.0
        assert out[f"iptm_protein_protein_{i}"] == pytest.approx(out[f"iptm_{i}"])


def test_iptm_single_chain_scores_zero():
    # One chain -> no inter-chain pairs -> nothing selected -> every interface scores 0.
    out = ComputeIPTM().compute(
        pae=_uniform_pae(4),
        asym_id=torch.zeros(4, dtype=torch.long),
        is_ligand=torch.tensor([0, 0, 1, 1]),
    )

    assert all(v == 0.0 for v in out.values())


def test_iptm_values_in_unit_range():
    out = ComputeIPTM().compute(
        pae=_uniform_pae(4), asym_id=_TWO_CHAINS, is_ligand=torch.tensor([0, 1, 0, 1])
    )

    assert all(0.0 <= v <= 1.0 for v in out.values())
