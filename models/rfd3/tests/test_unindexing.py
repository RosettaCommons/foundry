import copy
import sys

import numpy as np
import pytest
from assertpy import assert_that
from rfd3.testing.testing_utils import (
    TEST_JSON_DATA,
    build_pipelines,
    instantiate_example,
    load_train_or_val_cfg,
)
from rfd3.trainer.trainer_utils import (
    process_unindexed_outputs,
)

unindexed_cfg = load_train_or_val_cfg("test-unindexed")
unindexed_cfg.datasets.global_transform_args.train_conditions.island.frequency = 1e10
unindexed_cfg.datasets.global_transform_args.train_conditions.island.p_unindex_motif_tokens = 1.0
unindexed_pipes = build_pipelines(composed_config=unindexed_cfg)


@pytest.mark.fast
@pytest.mark.parametrize("example", ["1qys-0"])
@pytest.mark.parametrize("is_inference", [False])
def test_unindexed_cleanup(example, is_inference):
    """
    Tests:
        1. Unique residue indices can be found during training (at least)
        2. Atom array cleanup yields the original atom array during training
    """

    # Grab test args:
    args = copy.deepcopy(TEST_JSON_DATA[example])
    input = instantiate_example(copy.deepcopy(args), is_inference=is_inference)

    # Pipe input
    example = unindexed_pipes[is_inference](input)
    atom_array = example["atom_array"]
    assert atom_array.is_motif_atom_unindexed.sum() > 0
    assert (
        np.isnan(atom_array.coord).any(-1).sum() == 0
    ), "Coords should not contain nans for this test"
    # ... Assert cleanup of unindexed tokens yields the identity in the output during training
    atom_array.coord = np.nan_to_num(atom_array.coord)
    atom_array.set_annotation(
        "src_component", np.arange(atom_array.array_length())
    )  # spoof inference label
    atom_array_cleaned, metadata = process_unindexed_outputs(atom_array)

    # from rfd3.testing.debug_utils import save_pipe_out

    # save_pipe_out(atom_array)
    # print("Metadata:", metadata)

    xyz_cleaned = np.nan_to_num(atom_array_cleaned.coord)
    xyz_diffused = np.nan_to_num(atom_array[~atom_array.is_motif_atom_unindexed].coord)
    np.testing.assert_allclose(
        xyz_cleaned,
        xyz_diffused,
        atol=1e-6,
    )

    # Assert no change in insertion RMSD if matching atom names or not:
    _, metadata_all_matched = process_unindexed_outputs(
        atom_array, match_atom_names=False
    )
    np.testing.assert_allclose(
        metadata["insertion_rmsd"],
        metadata_all_matched["insertion_rmsd"],
        atol=1e-6,
    )
    np.testing.assert_allclose(metadata["insertion_rmsd"], 0)
    np.testing.assert_allclose(metadata_all_matched["insertion_rmsd"], 0)

    is_central, is_ca = example["feats"]["is_central"], example["feats"]["is_ca"]
    assert_that(is_central.sum()).described_as("Number of central atoms").is_equal_to(
        is_ca.sum()
    )


if __name__ == "__main__":
    pytest.main(sys.argv)
