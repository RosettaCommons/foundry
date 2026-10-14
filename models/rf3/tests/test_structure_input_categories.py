"""PDB/CIF inference must not require optional conditioning categories."""

import numpy as np
import pytest
from biotite.structure import Atom, array
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFCategory, CIFFile, set_structure
from rf3.utils.inference import InferenceInput


def structure_file(tmp_path, suffix, conditioning=False):
    atoms = array(
        [
            Atom(
                [j * 3.8 + offset, 0.0, 0.0],
                atom_name=name,
                element=element,
                chain_id="A",
                res_id=j + 1,
                res_name="GLY",
            )
            for j in range(3)
            for name, element, offset in [
                ("N", "N", 0.0),
                ("CA", "C", 1.2),
                ("C", "C", 2.3),
                ("O", "O", 2.9),
            ]
        ]
    )
    path = tmp_path / ("protein." + suffix)
    if suffix == "pdb":
        output = PDBFile()
        output.set_structure(atoms)
    else:
        output = CIFFile()
        set_structure(output, atoms)
        if conditioning:
            output.block["template_selection"] = CIFCategory(
                {"template_selection": ["A/GLY/1"]}
            )
            output.block["ground_truth_conformer_selection"] = CIFCategory(
                {"ground_truth_conformer_selection": ["A/GLY/2"]}
            )
    output.write(path)
    return path


@pytest.mark.parametrize("suffix", ["pdb", "cif"])
@pytest.mark.parametrize("explicit", [False, True])
def test_plain_structure_accepts_absent_cif_categories_and_explicit_selections(
    tmp_path, suffix, explicit
):
    path = structure_file(tmp_path, suffix)
    kwargs = (
        {
            "template_selection": ["A/GLY/2"],
            "ground_truth_conformer_selection": ["A/GLY/3"],
        }
        if explicit
        else {}
    )
    result = InferenceInput.from_cif_path(path, add_missing_atoms=False, **kwargs)
    assert result.template_selection == kwargs.get("template_selection")
    assert result.ground_truth_conformer_selection == kwargs.get(
        "ground_truth_conformer_selection"
    )
    assert len(result.atom_array) == 12
    assert np.isfinite(result.atom_array.coord).all()
    if explicit:
        atoms = result.to_pipeline_input()["atom_array"]
        assert atoms.is_input_file_templated[atoms.res_id == 2].all()
        assert not atoms.is_input_file_templated[atoms.res_id == 1].any()


@pytest.mark.parametrize("override", [False, True])
def test_present_cif_categories_are_preserved_unless_explicitly_overridden(
    tmp_path, override
):
    path = structure_file(tmp_path, "cif", conditioning=True)
    kwargs = (
        {
            "template_selection": ["A/GLY/3"],
            "ground_truth_conformer_selection": ["A/GLY/1"],
        }
        if override
        else {}
    )
    result = InferenceInput.from_cif_path(path, add_missing_atoms=False, **kwargs)
    assert result.template_selection == (["A/GLY/3"] if override else ["A/GLY/1"])
    assert result.ground_truth_conformer_selection == (
        ["A/GLY/1"] if override else ["A/GLY/2"]
    )
