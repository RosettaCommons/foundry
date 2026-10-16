"""Portable integration checks for the supported AtomWorks 3 inference APIs."""

import importlib
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from biotite import structure as struc


@pytest.mark.parametrize("model", ["rfd3", "rfd3na"])
def test_model_features_match_atomworks221_reference(model):
    """All 41 model inputs match the pre-port pipeline, including mixed RNA/DNA.

    Fixtures: public Foundry 754d1caae, AtomWorks 2.2.1, Biotite 1.4.0,
    Torch 2.11.0, seed 123; same specification and pipeline arguments as below.
    Reference conformer generation is disabled to isolate preprocessing from
    RDKit version changes. The raw AtomArray bond graph is intentionally not a
    golden: AW3 retains CCD-inferred bonds on synthetic coordinates that AW2
    discarded. The actual model token-bond feature is included in this golden.
    """
    parsing = importlib.import_module(f"{model}.inference.input_parsing")
    transforms = importlib.import_module(f"{model}.transforms.pipelines")
    root = Path(__file__).parents[1]
    kwargs = (
        {"length": "4"}
        if model == "rfd3"
        else {
            "contig": "4R,/0,4D,/0,4",
            "length": "12",
            "input": str(root / "models/rfd3na/docs/input_pdbs/AMP.pdb"),
        }
    )
    numpy_state, random_state = np.random.get_state(), random.getstate()
    try:
        with torch.random.fork_rng(devices=[]):
            random.seed(123)
            np.random.seed(123)
            torch.manual_seed(123)
            pipeline = transforms.build_atom14_base_pipeline(
                is_inference=True,
                sigma_data=16.0,
                diffusion_batch_size=1,
                generate_conformers=False,
                provide_reference_conformer_when_unmasked=False,
                ground_truth_conformer_policy="ground_truth",
                use_element_for_atom_names_of_atomized_tokens=False,
                n_atoms_per_token=14,
                central_atom="CB",
                atom_1d_features={},
                token_1d_features={},
            )
            actual = pipeline(
                parsing.DesignInputSpecification(**kwargs).to_pipeline_input(
                    "comparison"
                )
            )
    finally:
        np.random.set_state(numpy_state)
        random.setstate(random_state)
    with np.load(
        root / f"tests/data/atomworks3/{model}_aw221_features.npz"
    ) as expected:
        assert set(actual["feats"]) == set(expected.files)
        for key, value in actual["feats"].items():
            if isinstance(value, torch.Tensor):
                value = value.cpu().numpy()
            np.testing.assert_array_equal(value, expected[key], err_msg=key)


@pytest.mark.parametrize("model", ["rfd3", "rfd3na"])
def test_design_input_preserves_conditioning_and_backbone(model):
    parsing = importlib.import_module(f"{model}.inference.input_parsing")
    spec = parsing.DesignInputSpecification(length="4")
    array = spec.build()
    if isinstance(array, tuple):
        array = array[0]
    before = array.copy()
    result = parsing.prepare_pipeline_input_from_atom_array(array)
    actual = result["atom_array"]
    np.testing.assert_equal(actual.atom_name, before.atom_name)
    np.testing.assert_equal(actual.res_id, before.res_id)
    np.testing.assert_equal(actual.coord, before.coord)
    for annotation in parsing.REQUIRED_INFERENCE_ANNOTATIONS:
        np.testing.assert_equal(
            actual.get_annotation(annotation), before.get_annotation(annotation)
        )
    assert len(np.unique(actual.res_id)) == 4


@pytest.mark.parametrize("model", ["rfd3", "rfd3na"])
@pytest.mark.parametrize("extra_fields", [None, [], "all"])
def test_motif_file_loads_with_atomworks3_parser(model, extra_fields):
    inference = importlib.import_module(f"{model}.utils.inference")
    path = Path(__file__).parents[1] / "models/rfd3/docs/input_pdbs/4zxb_cropped.pdb"
    options = {"extra_fields": extra_fields}
    result = inference.inference_load_(path, cif_parser_args=options)
    array = result["atom_array"]
    assert len(array) > 0
    assert np.isfinite(array.coord).all()
    assert set(array.chain_id) == {"E"}
    assert options == {"extra_fields": extra_fields}


def _residues(names, atom_names, *, chains=None, ids=None, chain_types=None):
    """Minimal, deliberately unpositioned motif atoms; no external CCD mirror."""
    from atomworks.enums import ChainType

    sizes = list(map(len, atom_names))
    array = struc.AtomArray(sum(sizes))
    array.res_name = np.repeat(names, sizes)
    array.res_id = np.repeat(ids or list(range(1, len(names) + 1)), sizes)
    array.chain_id = np.repeat(chains or ["A"] * len(names), sizes)
    array.atom_name = np.concatenate(atom_names)
    array.element = struc.infer_elements(array.atom_name)
    array.set_annotation(
        "chain_type",
        np.repeat(chain_types or [ChainType.POLYPEPTIDE_L] * len(names), sizes),
    )
    array.set_annotation("is_motif_atom_unindexed", np.zeros(len(array), dtype=bool))
    array.bonds = struc.BondList(len(array))
    return array


@pytest.mark.parametrize(
    ("names", "atoms", "expected"),
    [
        (
            ["ALA", "PTR", "GLY"],
            [["N", "CA", "C"], ["N", "CA", "C"], ["N", "CA", "C"]],
            {(2, 3), (5, 6)},
        ),
        (["ALA", "GLY"], [["N", "CA", "C"], ["N", "CA", "C"]], set()),
        (["ACE", "ALA"], [["CH3", "C", "O"], ["N", "CA", "C"]], {(1, 3)}),
        (["ALA", "NH2"], [["N", "CA", "C"], ["N"]], {(2, 3)}),
        (["PTR", "GLY"], [["N", "CA"], ["N", "CA", "C"]], set()),
    ],
)
def test_nonstandard_backbone_bonds_use_ccd_attachment_sites(names, atoms, expected):
    from rfd3.inference.input_parsing import (
        _add_backbone_bonds_for_nonstandard_residues,
    )

    array = _residues(names, atoms)
    before = array.copy()
    actual = _add_backbone_bonds_for_nonstandard_residues(array)
    bonds = actual.bonds.as_array()
    assert {tuple(pair) for pair in bonds[:, :2]} == expected
    assert np.all(bonds[:, 2] == struc.BondType.SINGLE)
    np.testing.assert_equal(actual.coord, before.coord)
    np.testing.assert_equal(actual.atom_name, before.atom_name)
    # Adding twice cannot duplicate or replace authored bonds.
    _add_backbone_bonds_for_nonstandard_residues(actual)
    np.testing.assert_equal(actual.bonds.as_array(), bonds)


@pytest.mark.parametrize("boundary", ["chain", "gap", "unindexed", "assembly"])
def test_nonstandard_backbone_bonds_respect_boundaries(boundary):
    from rfd3.inference.input_parsing import (
        _add_backbone_bonds_for_nonstandard_residues,
    )

    array = _residues(["PTR", "GLY"], [["N", "CA", "C"], ["N", "CA", "C"]])
    if boundary == "chain":
        array.chain_id[3:] = "B"
    elif boundary == "gap":
        array.res_id[3:] = 3
    elif boundary == "unindexed":
        array.is_motif_atom_unindexed[:3] = True
    else:
        array.set_annotation("transformation_id", np.array(["1"] * 3 + ["2"] * 3))
    actual = _add_backbone_bonds_for_nonstandard_residues(array)
    assert actual.bonds.get_bond_count() == 0


def test_nonstandard_nucleotide_backbone_bond():
    from atomworks.enums import ChainType
    from rfd3.inference.input_parsing import (
        _add_backbone_bonds_for_nonstandard_residues,
    )

    array = _residues(
        ["PSU", "A"],
        [["P", "O3'"], ["P", "O3'"]],
        chain_types=[ChainType.RNA] * 2,
    )
    actual = _add_backbone_bonds_for_nonstandard_residues(array)
    np.testing.assert_equal(actual.bonds.as_array(), [[1, 2, struc.BondType.SINGLE]])


def test_polymer_family_boundary_does_not_invent_crosslink():
    from atomworks.enums import ChainType
    from rfd3.inference.input_parsing import (
        _add_backbone_bonds_for_nonstandard_residues,
    )

    array = _residues(
        ["PTR", "A"],
        [["N", "CA", "C"], ["P", "O3'"]],
        chain_types=[ChainType.POLYPEPTIDE_L, ChainType.RNA],
    )
    assert (
        _add_backbone_bonds_for_nonstandard_residues(array).bonds.get_bond_count() == 0
    )


@pytest.mark.parametrize(
    "module",
    [
        "rfd3.engine",
        "rfd3na.engine",
        "mpnn.inference_engines.mpnn",
        "rf3.inference_engines.rf3",
    ],
)
def test_real_engine_import(module):
    importlib.import_module(module)
