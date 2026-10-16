"""Portable whole-pipeline coverage for AtomWorks 3 and both MPNN models."""

import numpy as np
import pytest
import torch
from atomworks.io.tools.inference import components_to_atom_array
from mpnn.pipelines.mpnn import build_mpnn_transform_pipeline
from mpnn.utils.inference import MPNNInferenceInput


@pytest.mark.parametrize("model_type", ["protein_mpnn", "ligand_mpnn"])
@pytest.mark.parametrize("modifier", [None, "dna", "glycan", "cap"])
def test_mpnn_full_preprocessing_preserves_tokenization_and_design_scope(
    model_type, modifier
):
    components = [{"seq": "GNGG", "chain_id": "A", "chain_type": "POLYPEPTIDE(L)"}]
    bonds = []
    if modifier == "dna":
        components.append(
            {"seq": "ACGT", "chain_id": "B", "chain_type": "POLYDEOXYRIBONUCLEOTIDE"}
        )
    elif modifier == "glycan":
        components.append({"ccd_code": "NAG", "chain_id": "B"})
        bonds = [["A/ASN/2/ND2", "B/NAG/1/C1"]]
    elif modifier == "cap":
        components.append({"ccd_code": "ACE", "chain_id": "B"})
        bonds = [["A/GLY/1/N", "B/ACE/1/C"]]
    atoms = components_to_atom_array(components, bonds=bonds)
    # Sequence components intentionally have NaN coordinates; MPNN requires
    # a resolved backbone. Supply deterministic synthetic coordinates.
    atoms.coord = (
        np.random.default_rng(0).normal(size=(len(atoms), 3)).astype(np.float32)
    )
    atoms.set_annotation("occupancy", np.ones(len(atoms)))
    item = MPNNInferenceInput.from_atom_array_and_dict(
        atom_array=atoms, input_dict={"fixed_residues": ["A4"], "temperature": 0.15}
    )
    pipeline = build_mpnn_transform_pipeline(
        model_type=model_type, is_inference=True, device="cpu"
    )
    output = pipeline({**item.input_dict, "atom_array": item.atom_array})
    atoms = output["atom_array"]
    features = output["input_features"]
    modified_residue = {"glycan": 2, "cap": 1}.get(modifier)
    expected_residues = [r for r in [1, 2, 3, 4] if r != modified_residue]
    np.testing.assert_array_equal(
        np.unique(atoms.res_id[~atoms.atomize]), expected_residues
    )
    assert features["X"].shape[0] == len(expected_residues)
    assert not features["designed_residue_mask"][-1]
    torch.testing.assert_close(
        features["temperature"], torch.full((len(expected_residues),), 0.15)
    )
    if modifier:
        assert atoms.atomize[atoms.chain_id == "B"].all()
    if modified_residue:
        modified = (atoms.chain_id == "A") & (atoms.res_id == modified_residue)
        assert atoms.atomize[modified].all()
        assert atoms.is_covalent_modification[modified].all()
        assert set(atoms.pn_unit_iid[modified]) == set(
            atoms.pn_unit_iid[atoms.chain_id == "B"]
        )
    for value in features.values():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            assert torch.isfinite(value).all()


@pytest.mark.parametrize("has_charge", [False, True])
def test_missing_charge_is_inferred_from_ccd_without_mutating_caller(has_charge):
    atoms = components_to_atom_array(
        [
            {"seq": "DKAA", "chain_id": "A"},
            {"ccd_code": "NA", "chain_id": "B"},
            {"ccd_code": "CL", "chain_id": "C"},
        ]
    )
    if not has_charge:
        atoms.del_annotation("charge")
    original = atoms.copy()
    result = MPNNInferenceInput.from_atom_array_and_dict(atom_array=atoms).atom_array
    assert atoms.get_annotation_categories() == original.get_annotation_categories()
    for name in atoms.get_annotation_categories():
        np.testing.assert_array_equal(
            atoms.get_annotation(name), original.get_annotation(name)
        )
    np.testing.assert_array_equal(atoms.coord, original.coord)
    assert result.charge[result.res_name == "CL"].item() == -1
    assert result.charge[result.res_name == "NA"].item() == 1


def test_structure_path_input_uses_atomworks3_parser(tmp_path):
    from atomworks.io.utils.io_utils import to_cif_file

    atoms = components_to_atom_array([{"seq": "GLKE", "chain_id": "A"}])
    atoms.coord = (
        np.random.default_rng(0).normal(size=(len(atoms), 3)).astype(np.float32)
    )
    atoms.set_annotation("occupancy", np.ones(len(atoms)))
    path = tmp_path / "input.cif"
    to_cif_file(atoms, path)
    item = MPNNInferenceInput.from_atom_array_and_dict(
        input_dict={"structure_path": str(path)}
    )
    assert set(item.atom_array.res_name) == {"GLY", "LEU", "LYS", "GLU"}
    assert len(np.unique(item.atom_array.res_id)) == 4
