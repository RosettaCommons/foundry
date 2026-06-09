"""Unit tests for rf3.utils.predicted_error pure helpers.

These pin the small, non-obvious building blocks of the AF3-style ranking logic
(filename suffixed ``_utils`` because ``test_predicted_error.py`` already covers
the sibling ``rf3.metrics.predicted_error`` module under pytest's prepend import
mode):

- ``_select_scored_units`` flattens the (chain, chain, type) interface specs and
  the (chain, type) unit specs into the scored-chain list, the
  ``"chain_a-chain_b"`` interface names, and the flat interface-chain list.
- ``_get_average_error_per_interface`` averages the two endpoint chains' per-batch
  error for each ``"chain_a-chain_b"`` interface.
- ``_get_lowest_error_indices`` is the per-key argmin over a dict of error tensors.
- ``get_mean_atomwise_plddt`` reshapes + unbins the pLDDT logits and averages over
  the real (non-padding) atoms; uniform logits give the mean of the bin midpoints.
"""

import torch
from rf3.chemical import NHEAVY
from rf3.metrics.metric_utils import find_bin_midpoints
from rf3.utils.predicted_error import (
    _get_average_error_per_interface,
    _get_lowest_error_indices,
    _select_scored_units,
    get_mean_atomwise_plddt,
)


def test_select_scored_units_flattens_specs():
    interfaces_to_score = [("A", "B", "protein-protein"), ("A", "C", "protein-ligand")]
    pn_units_to_score = [("A", "protein"), ("B", "protein")]
    scored_chains, interfaces, interface_chains = _select_scored_units(
        interfaces_to_score, pn_units_to_score
    )
    assert scored_chains == ["A", "B"]
    assert interfaces == ["A-B", "A-C"]
    assert interface_chains == ["A", "B", "A", "C"]


def test_get_average_error_per_interface_averages_endpoints():
    interface_errors = {
        "A": torch.tensor([2.0, 4.0]),
        "B": torch.tensor([4.0, 8.0]),
    }
    out = _get_average_error_per_interface(["A-B"], [], interface_errors)
    assert torch.allclose(out["A-B"], torch.tensor([3.0, 6.0]))


def test_get_lowest_error_indices_is_per_key_argmin():
    errors = {
        "A": torch.tensor([5.0, 1.0, 3.0]),
        "B": torch.tensor([0.0, 9.0, 9.0]),
    }
    out = _get_lowest_error_indices(errors)
    assert out["A"].item() == 1
    assert out["B"].item() == 0


def test_get_mean_atomwise_plddt_uniform_logits_is_midpoint_mean():
    n_token, n_bins, max_value = 3, 8, 1.0
    # Uniform logits → uniform softmax over bins → every atom's pLDDT is the mean
    # of the bin midpoints, so the masked mean over all-real atoms is that value.
    plddt_logits = torch.zeros(1, n_token, NHEAVY * n_bins)
    is_real_atom = torch.ones(n_token, NHEAVY, dtype=torch.bool)
    out = get_mean_atomwise_plddt(plddt_logits, is_real_atom, max_value)
    expected = find_bin_midpoints(max_value, n_bins).mean()
    assert out.shape == (1,)
    assert torch.allclose(out, expected.reshape(1), atol=1e-5)


def test_get_mean_atomwise_plddt_batch_shape_and_bounds():
    n_token, n_bins, max_value = 2, 4, 32.0
    plddt_logits = torch.rand(3, n_token, NHEAVY * n_bins)
    is_real_atom = torch.ones(n_token, NHEAVY, dtype=torch.bool)
    out = get_mean_atomwise_plddt(plddt_logits, is_real_atom, max_value)
    assert out.shape == (3,)
    # pLDDT is an expected value over midpoints in [0, max_value].
    assert torch.all(out >= 0) and torch.all(out <= max_value)
