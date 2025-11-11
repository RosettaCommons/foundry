import copy
import sys

import numpy as np
import pytest
from assertpy import assert_that
from rfd3.testing.testing_utils import (
    TEST_JSON_DATA,
    instantiate_example,
)
from rfd3.transforms.conditioning_utils import sample_subgraph_atoms


@pytest.mark.fast
@pytest.mark.parametrize("p_seed_furthest_from_o", [0.0, 1.0])
def test_subgraph_sampling(p_seed_furthest_from_o):
    example_name = "1nzy-1"
    args = TEST_JSON_DATA[example_name]
    input = instantiate_example(args, is_inference=False)

    atom_array = input["atom_array"]
    atom_array.set_annotation("occupancy", np.ones(atom_array.shape[0], dtype=bool))
    ligand_array = atom_array[atom_array.res_name == "BCA"]
    ligand_array.set_annotation(
        "is_protein", np.zeros(ligand_array.shape[0], dtype=bool)
    )
    assert_that(ligand_array.array_length()).is_greater_than(0)

    n = 10
    avg = 0
    for i in range(n):
        mask = sample_subgraph_atoms(
            copy.deepcopy(ligand_array),
            p_seed_furthest_from_o=0.0,
            n_bond_expectation=3,
            p_fix_all=0.0,
        )
        avg += mask.sum()
    avg /= n
    print(f"Average number of atoms in subgraph: {avg}")
    assert_that(avg).is_greater_than(0)


if __name__ == "__main__":
    pytest.main(sys.argv)
