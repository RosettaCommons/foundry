import numpy as np
import pytest
from biotite.structure import AtomArray

from validation.enzyme_tipatom.geometry import symmetry_aware_tip_rmsd
from validation.enzyme_tipatom.run import measure_insertion


def fixture():
    a = AtomArray(4)
    a.chain_id = np.array(["X", "X", "A", "A"])
    a.res_id = np.array([580, 580, 10, 10])
    a.res_name[:] = "CYS"
    a.atom_name = np.array(["slot4", "slot5", "slot4", "slot5"])
    a.coord = np.array([[0, 0, 0], [1, 0, 0], [0, 3, 0], [1, 3, 0]], dtype=float)
    a.set_annotation("gt_atom_name", np.array(["CB", "SG", "CB", "SG"]))
    a.set_annotation("src_component", np.array(["A580", "A580", "", ""]))
    a.set_annotation("is_motif_atom_unindexed", np.array([True, True, False, False]))
    return a


def test_tip_error_uses_physical_names_and_unfitted_preinsertion_positions():
    atoms = fixture()
    before = atoms.coord.copy()
    score = measure_insertion(
        atoms, {"diffused_index_map": {"A580": "A10"}}, {"CYS": ["CB", "SG"]}
    )
    assert score["tip_atom_rmsd"] == 3
    assert score["n_tip_atoms"] == 2
    np.testing.assert_array_equal(atoms.coord, before)


@pytest.mark.parametrize("corruption", ["missing", "nan"])
def test_bad_correspondence_is_not_silently_dropped(corruption):
    atoms = fixture()
    if corruption == "missing":
        atoms.atom_name[3] = "other"
    else:
        atoms.coord[3, 0] = np.nan
    with pytest.raises(ValueError):
        measure_insertion(
            atoms, {"diffused_index_map": {"A580": "A10"}}, {"CYS": ["CB", "SG"]}
        )


def test_only_equivalent_asp_oxygens_can_exchange():
    atoms = [
        {
            "source": "A1",
            "tip": True,
            "restype": "ASP",
            "atom": "OD1",
            "coord": [1, 0, 0],
            "target": [-1, 0, 0],
        },
        {
            "source": "A1",
            "tip": True,
            "restype": "ASP",
            "atom": "OD2",
            "coord": [-1, 0, 0],
            "target": [1, 0, 0],
        },
    ]
    assert symmetry_aware_tip_rmsd(atoms) == (0, ["A1"])
    for atom in atoms:
        atom["restype"] = "GLN"
    assert symmetry_aware_tip_rmsd(atoms) == (2, [])
