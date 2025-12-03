from copy import deepcopy

import pytest
from atomworks.ml.utils.testing import cached_parse
from rf3.metrics.chiral import ChiralLoss


@pytest.mark.parametrize("pdb_id", ["5ocm", "1ivo"])
def test_chiral_metrics(pdb_id: str):
    # ... get the AtomArray
    ground_truth_atom_array = cached_parse(pdb_id, hydrogen_policy="remove")[
        "atom_array"
    ]
    predicted_atom_array = deepcopy(ground_truth_atom_array)

    chiral_loss = ChiralLoss()

    # Baseline
    perfect_output = chiral_loss.compute(
        predicted_atom_array_stack=predicted_atom_array,
        ground_truth_atom_array_stack=ground_truth_atom_array,
    )
    assert perfect_output["polymer_percent_correct_chirality"] == 1.0
    assert perfect_output["non_polymer_percent_correct_chirality"] == 1.0

    # (reflection to invert all stereocenters)
    predicted_atom_array.coord = -predicted_atom_array.coord

    # ... and recompute
    terrible_output = chiral_loss.compute(
        predicted_atom_array_stack=predicted_atom_array,
        ground_truth_atom_array_stack=ground_truth_atom_array,
    )

    assert terrible_output["polymer_percent_correct_chirality"] == 0.0
    assert terrible_output["non_polymer_percent_correct_chirality"] == 0.0

    # Compare the two outputs
    # (Perfect vs. terrible)
    assert (
        perfect_output["polymer_chiral_loss_mean"] * 10
        < terrible_output["polymer_chiral_loss_mean"]
    )
    assert (
        perfect_output["non_polymer_chiral_loss_mean"] * 10
        < terrible_output["non_polymer_chiral_loss_mean"]
    )
    # (Same number of chiral centers)
    assert (
        perfect_output["polymer_n_chiral_centers"]
        == terrible_output["polymer_n_chiral_centers"]
    )
    assert (
        perfect_output["non_polymer_n_chiral_centers"]
        == terrible_output["non_polymer_n_chiral_centers"]
    )
