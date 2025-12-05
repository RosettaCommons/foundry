import copy
import sys

import numpy as np
import pytest
from assertpy import assert_that
from atomworks.ml.utils.token import get_af3_token_representative_masks
from biotite.structure import get_residue_starts
from rfd3.constants import ATOM14_ATOM_NAMES
from rfd3.testing.testing_utils import (
    TEST_JSON_DATA,
    build_pipelines,
    instantiate_example,
)
from rfd3.transforms.conditioning_base import (
    check_has_required_conditioning_annotations,
    get_motif_features,
)
from rfd3.transforms.virtual_atoms import map_to_association_scheme

pipes = build_pipelines("test-uncond")


@pytest.mark.fast
@pytest.mark.parametrize("example", ["monomer-100"])
@pytest.mark.parametrize("is_inference", [True, False])
def test_virtual_atom_naming(example, is_inference):
    # Instantiate test args:
    args = copy.deepcopy(TEST_JSON_DATA[example])
    xargs = args["extra"] if "extra" in args else args
    expected_diffused_tokens = xargs.pop("expected_diffused_tokens")

    input = instantiate_example(args, is_inference=is_inference)

    # Pipe input
    example = pipes[is_inference](input)
    atom_array = example["atom_array"]

    starts = get_residue_starts(atom_array, add_exclusive_stop=True)

    # Assert globals
    n_residues = (atom_array.is_residue[starts[:-1]]).sum()
    assert_that(n_residues).described_as("Number of tokens").is_equal_to(
        expected_diffused_tokens
    )
    assert_that(atom_array[atom_array.is_residue].array_length()).described_as(
        "Number of atoms"
    ).is_equal_to(n_residues * 14)

    # Loop over residues
    for i, (start, end) in enumerate(zip(starts[:-1], starts[1:])):
        atom_array_subset = atom_array[start:end]

        # Assert not all elements are VX
        assert_that(set(atom_array_subset.element.tolist())).described_as(
            "Element types"
        ).is_not_equal_to({"VX"})

        atom_names = atom_array_subset.atom_name.tolist()
        if is_inference:
            assert_that(atom_names).is_equal_to(ATOM14_ATOM_NAMES.tolist())
        else:
            # Since atom names are permuted, only check the sets are equal
            assert_that(set(atom_names)).is_equal_to(set(ATOM14_ATOM_NAMES.tolist()))


# @pytest.mark.fast
# @pytest.mark.parametrize("example", ["1qys-2"])
# @pytest.mark.parametrize("is_inference", [True])
# def test_indexing(example, is_inference):
#     '''
#     Sends a motif scaffolding problem through the pipeline and ensures
#     '''


@pytest.mark.fast
@pytest.mark.parametrize("example", ["1qys-0", "1qys-1", "af2-122", "af2-151"])
@pytest.mark.parametrize("is_inference", [True])
def test_unindexing(example, is_inference):
    """
    Sends a motif scaffolding- and tipatom scaffolding unindexed problem through
    the pipeline and ensures the output has the correct number of tokens recognized.
    """

    # Grab test args:
    args = copy.deepcopy(TEST_JSON_DATA[example])
    xargs = args["extra"] if "extra" in args else args
    expected_indexed_tokens = xargs["expected_indexed_tokens"]
    expected_unindexed_tokens = xargs["expected_unindexed_tokens"]
    expected_diffused_tokens = xargs["expected_diffused_tokens"]
    expected_fixed_atoms = (
        xargs["expected_fixed_atoms"] if "expected_fixed_atoms" in xargs else None
    )

    # Pipe input
    input = instantiate_example(args, is_inference=is_inference)
    atom_array_in = input["atom_array"].copy()
    n_unindexed_in = input["atom_array"].is_motif_atom_unindexed.sum()
    assert_that(n_unindexed_in).is_greater_than(
        0
    ) if expected_unindexed_tokens > 0 else assert_that(n_unindexed_in).is_equal_to(0)
    # Assert number of fixed atoms before pipeline
    if expected_fixed_atoms is not None:
        assert_that(n_unindexed_in).described_as(
            "Number of fixed motif atoms before pipeline"
        ).is_equal_to(expected_fixed_atoms)

    example = pipes[is_inference](input)

    atom_array = example["atom_array"]
    check_has_required_conditioning_annotations(atom_array)

    if expected_diffused_tokens is not None:
        n_diffused_tokens = len(
            get_residue_starts(atom_array[~atom_array.is_motif_atom_with_fixed_coord])
        )
        assert_that(n_diffused_tokens).described_as(
            "Number of diffused tokens after pipeline"
        ).is_equal_to(expected_diffused_tokens)

    # Get masks
    is_unindexed = atom_array.is_motif_atom_unindexed

    # Atom assertions general
    is_motif = get_motif_features(atom_array)["is_motif_atom"]
    assert_that(is_motif.sum()).described_as("Nonzero motif atoms").is_greater_than(0)
    assert_that(atom_array.is_motif_atom_with_fixed_coord.sum()).described_as(
        "Nonzero coordinate fixed atoms"
    ).is_greater_than(0)
    if expected_unindexed_tokens is not None and expected_unindexed_tokens > 0:
        assert_that(is_unindexed.sum()).described_as(
            "Nonzero unindexed atoms"
        ).is_greater_than(0)

    # Assert number of fixed atoms
    if expected_fixed_atoms is not None:
        n_fixed_atoms = is_motif.sum()
        assert_that(n_fixed_atoms).described_as(
            "Expected number of fixed atoms after pipeline"
        ).is_equal_to(expected_fixed_atoms)

    # Assert number of unindexed tokens
    n_indexed_tokens = len(get_residue_starts(atom_array[~is_unindexed]))
    n_unindexed_tokens = len(get_residue_starts(atom_array[is_unindexed]))
    assert_that(n_indexed_tokens).is_equal_to(expected_indexed_tokens)
    assert_that(n_unindexed_tokens).is_equal_to(expected_unindexed_tokens)
    print(
        f"Got {n_unindexed_tokens} unindexed tokens and {n_indexed_tokens} indexed tokens."
    )

    # Assert number of representative atoms
    n_unindexed_representative_atoms = get_af3_token_representative_masks(
        atom_array[is_unindexed]
    ).sum()
    n_indexed_representative_atoms = get_af3_token_representative_masks(
        atom_array[~is_unindexed]
    ).sum()
    assert_that(n_indexed_representative_atoms).is_equal_to(expected_indexed_tokens)
    assert_that(n_unindexed_representative_atoms).is_equal_to(expected_unindexed_tokens)
    print(
        f"Got {n_unindexed_representative_atoms} unindexed representative atoms and {n_indexed_representative_atoms} indexed representative atoms."
    )

    # Assert terminus assignments (Unindexed tokens must be neither)
    assert_that(atom_array[is_unindexed].is_C_terminus.sum()).is_equal_to(0)
    assert_that(atom_array[is_unindexed].is_N_terminus.sum()).is_equal_to(0)
    assert_that(atom_array[~is_unindexed].is_C_terminus.sum()).is_greater_than(0)
    assert_that(atom_array[~is_unindexed].is_N_terminus.sum()).is_greater_than(0)

    # Assert Atom names of diffused regions within ATOM14_ATOM_NAMES
    is_motif_atom = get_motif_features(atom_array)["is_motif_atom"]
    is_motif_atom_in = get_motif_features(atom_array_in)["is_motif_atom"]
    atom_names_diffused = atom_array[~is_motif_atom].atom_name
    atom_names_fixed = atom_array[is_motif_atom].atom_name
    assert_that(np.all(np.isin(atom_names_diffused, ATOM14_ATOM_NAMES))).is_true()
    assert_that(np.all(np.isin(atom_names_fixed, ATOM14_ATOM_NAMES))).is_true()

    # Assert input names have been transformed properly
    if "fixed_atoms" in args:
        atom_array_in = atom_array_in[is_motif_atom_in]
        atom_array_out = example["atom_array"]
        atom_array_out = atom_array_out[is_motif_atom]
        expected_names_out = np.concatenate(
            [
                map_to_association_scheme(atom_name, res_name, "atom14")
                for atom_name, res_name in zip(
                    atom_array_in.atom_name, atom_array_in.res_name
                )
            ]
        )
        names_out = atom_array_out.atom_name
        assert_that(expected_names_out.tolist()).is_equal_to(names_out.tolist())


@pytest.mark.fast
@pytest.mark.parametrize("example", ["1qys-0"])
@pytest.mark.parametrize("is_inference", [False])
def test_token_id_annotations(example, is_inference):
    """
    Tests:
        1. Unique residue indices can be found during training (at least)
        2. Atom array cleanup yields the original atom array during training
    """

    # Grab test args:
    args = copy.deepcopy(TEST_JSON_DATA[example])
    input = instantiate_example(copy.deepcopy(args), is_inference=is_inference)

    # Spoof zero occ region in input
    aa_in = input["atom_array"]
    middle_idx = aa_in.array_length() // 2
    aa_in.occupancy[middle_idx : (middle_idx + 20)] = 0
    input["atom_array"] = aa_in

    example = pipes[is_inference](input)
    aa_out = example["atom_array"]

    assert not (
        np.diff(aa_out.token_id) > 1
    ).any(), "Token ID has breaks > 1. Expected aranged token_id"


if __name__ == "__main__":
    pytest.main(sys.argv)
