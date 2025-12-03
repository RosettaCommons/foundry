import sys

import numpy as np
import pytest
from atomworks.io.utils.io_utils import load_any
from rfd3.inference.input_parsing import DesignInputSpecification
from rfd3.inference.parsing import InputSelection
from rfd3.testing.testing_utils import (
    TEST_JSON_DATA,
)

from foundry.utils.components import (
    fetch_mask_from_component,
    fetch_mask_from_idx,
    fetch_mask_from_name,
    get_name_mask,
    unravel_components,
)

# TEST 1 - test the selections
args = TEST_JSON_DATA["amidase_helix"]
file = args["input"]
ligand = args["ligand"]
atom_array_ref = load_any(file)[0]


def test_canonicalization(name="L:G", idx="Z9", atom_names="C1,N1,O1,C2,O2"):
    # Assert that the contig gets canonicalized to the same values
    canonical_components_0 = unravel_components("Z9", atom_array=atom_array_ref)
    canonical_components_1 = unravel_components("L:G", atom_array=atom_array_ref)

    assert (
        len(canonical_components_0) == 1
    ), f"0) Expected single canonical component but got {canonical_components_0}"
    assert (
        len(canonical_components_1) == 1
    ), f"1) Expected single canonical component but got {canonical_components_1}"
    assert (
        canonical_components_0[0] == idx
    ), f"0) Expected {idx} but got {canonical_components_0}"
    assert (
        canonical_components_1[0] == idx
    ), f"1) Expected {idx} but got {canonical_components_1}"

    # Assert same masks are made when creating from different routes
    mask_0 = fetch_mask_from_idx(idx, atom_array=atom_array_ref)
    mask_1 = fetch_mask_from_name(name, atom_array=atom_array_ref)
    mask_2 = fetch_mask_from_component(
        idx, atom_array=atom_array_ref
    )  # catch-all function should handle both
    mask_3 = fetch_mask_from_component(name, atom_array=atom_array_ref)
    assert (mask_0 == mask_2).all(), f"Expected {mask_0} but got {mask_2}"
    assert (mask_1 == mask_3).all(), f"Expected {mask_1} but got {mask_3}"
    assert (mask_0 == mask_1).all(), f"Expected {mask_0} but got {mask_1}"

    # Assert atom name selections work
    name_mask_0 = np.zeros(atom_array_ref.array_length(), dtype=bool)
    name_mask_0[mask_0] = get_name_mask(
        source_names=atom_array_ref.atom_name[mask_0],
        query_names=atom_names,
        source_resname=None,
    )

    sele = InputSelection.from_any({name: atom_names}, atom_array=atom_array_ref)
    name_mask_1 = sele.get_mask()

    assert (
        name_mask_0 == name_mask_1
    ).all(), f"Expected selection mask to be {name_mask_0} but got {name_mask_1}"


args = TEST_JSON_DATA["1j79_C2"]
file = args["input"]
atom_array_ref_sym = load_any(file)[0]


def test_ligand_canonicalization(ligand="ORO"):
    # Asking for this ligand should return all ligands
    mask_0 = fetch_mask_from_name(ligand, atom_array=atom_array_ref_sym)

    # as string
    sele = InputSelection.from_any(ligand, atom_array=atom_array_ref_sym)
    mask_1 = sele.get_mask()

    # as contig string
    ligand_idxs = mask_0
    res_ids, chain_ids = (
        atom_array_ref_sym.res_id[mask_0],
        atom_array_ref_sym.chain_id[mask_0],
    )
    ligand_idxs = ",".join(list(set([f"{c}{r}" for c, r in zip(chain_ids, res_ids)])))
    sele = InputSelection.from_any(ligand_idxs, atom_array=atom_array_ref_sym)
    mask_2 = sele.get_mask()

    # as dictionary of the indices
    sele = InputSelection.from_any({ligand_idxs: "ALL"}, atom_array=atom_array_ref_sym)
    mask_3 = sele.get_mask()

    # as dictionary of the name
    sele = InputSelection.from_any({ligand: "ALL"}, atom_array=atom_array_ref_sym)
    mask_4 = sele.get_mask()

    # assert all equal
    assert (mask_0 == mask_1).all(), f"Expected {mask_0} but got {mask_1}"
    assert (mask_0 == mask_2).all(), f"Expected {mask_0} but got {mask_2}"
    assert (mask_0 == mask_3).all(), f"Expected {mask_0} but got {mask_3}"
    assert (mask_0 == mask_4).all(), f"Expected {mask_0} but got {mask_4}"

    return


args = TEST_JSON_DATA["amidase_contiguous"]
file = args["input"]
atom_array_ref_unindexed = load_any(file)[0]


def test_unindexed_break():
    # Assert that unindexed selections with breaks work
    sele = InputSelection.from_any(args["unindex"], atom_array=atom_array_ref_unindexed)
    comps, breaks = DesignInputSpecification.break_unindexed(sele)


if __name__ == "__main__":
    pytest.main(sys.argv)
