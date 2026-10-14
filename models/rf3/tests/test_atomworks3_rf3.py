"""Portable RF3 preprocessing regressions using AtomWorks' bundled CCD."""

import numpy as np
import pytest
import torch
from atomworks.enums import GroundTruthConformerPolicy
from atomworks.io.tools.inference import components_to_atom_array
from atomworks.ml.utils.token import apply_segment_wise_2d, get_token_starts
from biotite.structure import AtomArray, BondList
from rf3.data.bond_features import AddRF3TokenBondFeatures
from rf3.data.pipelines import build_af3_transform_pipeline
from rf3.utils.inference import InferenceInput


def test_bond_features_keep_inference_topology_and_training_cutoff():
    atoms = AtomArray(6)
    atoms.chain_id[:] = "A"
    atoms.res_id = np.arange(6)
    atoms.coord[:, 0] = [0, 1, 2, 4.4, 8, np.nan]
    atoms.coord[:, 1:] = 0
    atoms.set_annotation("atomize", np.array([False, False, True, True, True, True]))
    atoms.set_annotation("is_polymer", ~atoms.atomize)
    atoms.bonds = BondList(
        6, np.array([[0, 1, 1], [1, 2, 1], [2, 3, 1], [3, 4, 1], [4, 5, 1]])
    )
    original_bonds = atoms.bonds.as_array().copy()
    for is_inference in (False, True):
        data = AddRF3TokenBondFeatures().forward(
            {"atom_array": atoms, "is_inference": is_inference}
        )
        bonds = data["feats"]["token_bonds"]
        assert not bonds[0, 1]  # Polymer-polymer edges are not conditioning.
        assert bonds[1, 2]
        assert bool(bonds[2, 3]) == is_inference  # Strict 2.4 A cutoff.
        assert bool(bonds[3, 4]) == is_inference
        assert not bonds[4, 5]
        assert not np.diag(bonds).any()
        np.testing.assert_array_equal(bonds, bonds.T)
        np.testing.assert_array_equal(atoms.bonds.as_array(), original_bonds)


@pytest.mark.parametrize("is_inference", [False, True])
@pytest.mark.parametrize("explicit_tokens", [False, True])
@pytest.mark.parametrize("edge_case", ["finite", "nan", "inf", "no_bonds"])
def test_bond_features_match_dense_reference(is_inference, explicit_tokens, edge_case):
    atoms = AtomArray(9)
    atoms.chain_id[:] = "A"
    atoms.res_id = np.array([1, 1, 1, 2, 2, 2, 3, 4, 5])
    atoms.coord = np.random.default_rng(17).normal(size=(9, 3)).astype(np.float32)
    atoms.set_annotation("atomize", np.array([False] * 6 + [True] * 3))
    if explicit_tokens:
        # Explicit token IDs, including nonzero/gapped labels, own the boundaries.
        atoms.set_annotation("token_id", np.array([10, 10, 20, 30, 30, 30, 40, 50, 60]))
    if edge_case in {"nan", "inf"}:
        atoms.coord[0, 0] = np.nan if edge_case == "nan" else np.inf
    edges = np.array([[0, 6, 1], [1, 6, 1], [2, 3, 1], [3, 4, 1], [6, 7, 1]])
    atoms.bonds = BondList(9, edges if edge_case != "no_bonds" else None)
    bonds = atoms.bonds.as_array()
    distances = np.full((len(atoms), len(atoms)), np.inf)
    lengths = np.linalg.norm(
        atoms.coord[bonds[:, 0]] - atoms.coord[bonds[:, 1]], axis=-1
    )
    distances[bonds[:, 0], bonds[:, 1]] = lengths
    distances[bonds[:, 1], bonds[:, 0]] = lengths
    starts = get_token_starts(atoms, add_exclusive_stop=True)
    expected = apply_segment_wise_2d(distances, starts, np.min) < (
        np.inf if is_inference else 2.4
    )
    np.fill_diagonal(expected, False)
    residues = ~atoms.atomize[starts[:-1]]
    expected[np.outer(residues, residues)] = False
    actual = AddRF3TokenBondFeatures().forward(
        {"atom_array": atoms, "is_inference": is_inference}
    )["feats"]["token_bonds"]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(atoms.bonds.as_array(), bonds)


def test_bond_features_large_structure_allocates_only_token_matrix(monkeypatch):
    atoms = AtomArray(50_000)
    atoms.chain_id[:] = "A"
    atoms.res_id = np.arange(len(atoms)) // 100
    atoms.coord[:, 0] = np.arange(len(atoms))
    atoms.coord[:, 1:] = 0
    atoms.set_annotation("atomize", np.arange(len(atoms)) == len(atoms) - 1)
    atoms.bonds = BondList(len(atoms), np.array([[len(atoms) - 2, len(atoms) - 1, 1]]))
    n_tokens = len(get_token_starts(atoms))
    full = np.full

    def bounded_full(shape, *args, **kwargs):
        # Fail before allocating the old ~20 GB atom matrix.
        assert np.prod(shape) <= n_tokens**2
        return full(shape, *args, **kwargs)

    monkeypatch.setattr(np, "full", bounded_full)
    actual = AddRF3TokenBondFeatures().forward(
        {"atom_array": atoms, "is_inference": True}
    )["feats"]["token_bonds"]
    assert actual.shape == (n_tokens, n_tokens)
    assert actual[-2, -1] and actual[-1, -2]
    assert actual.sum() == 2


def _pipeline():
    return build_af3_transform_pipeline(
        is_inference=True,
        protein_msa_dirs=[],
        rna_msa_dirs=[],
        residue_cache_dir=None,
        n_recycles=1,
        n_msa=1,
        max_msa_sequences=1,
        diffusion_batch_size=1,
    )


@pytest.mark.parametrize(
    "extra_components,bonds",
    [
        ([], []),
        (
            [{"seq": "ACGT", "chain_id": "B", "chain_type": "POLYDEOXYRIBONUCLEOTIDE"}],
            [],
        ),
        ([{"ccd_code": "NAG", "chain_id": "B"}], [["A/ASN/2/ND2", "B/NAG/1/C1"]]),
    ],
)
def test_json_to_model_features_preserves_components_and_covalent_bond(
    extra_components, bonds
):
    item = InferenceInput.from_json_dict(
        {
            "name": "portable",
            "components": [
                {"seq": "GNGG", "chain_id": "A", "chain_type": "POLYPEPTIDE(L)"},
                *extra_components,
            ],
            "bonds": bonds,
        }
    )
    original = item.atom_array
    assert set(original.chain_id) == {"A", *[c["chain_id"] for c in extra_components]}
    if bonds:
        # JSON components start far apart. Their explicit link is authoritative.
        left = np.flatnonzero(
            (original.chain_id == "A")
            & (original.res_id == 2)
            & (original.atom_name == "ND2")
        )[0]
        right = np.flatnonzero(
            (original.chain_id == "B") & (original.atom_name == "C1")
        )[0]
        assert original.bonds.adjacency_matrix()[left, right]
    output = _pipeline()(item.to_pipeline_input())
    assert output
    if bonds:
        atoms = output["atom_array"]
        modified = (atoms.chain_id == "A") & (atoms.res_id == 2)
        assert atoms.atomize[modified].all()
        assert atoms.is_covalent_modification[modified].all()
        # AW3 fixes the glycosylated amide nitrogen's formerly +1 charge.
        nitrogen = np.flatnonzero(modified & (atoms.atom_name == "ND2"))[0]
        carbon = np.flatnonzero((atoms.chain_id == "B") & (atoms.atom_name == "C1"))[0]
        assert output["feats"]["ref_charge"][nitrogen] == 0
        assert output["feats"]["token_bonds"][
            atoms.token_id[nitrogen], atoms.token_id[carbon]
        ]

    def check_finite(value):
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            assert torch.isfinite(value).all()
        elif isinstance(value, dict):
            for child in value.values():
                check_finite(child)

    check_finite(output)


def test_atom_array_input_preserves_template_selection():
    item = InferenceInput.from_json_dict(
        {
            "name": "annotations",
            "components": [
                {"seq": "GNGG", "chain_id": "A", "chain_type": "POLYPEPTIDE(L)"}
            ],
        }
    )
    item.atom_array.set_annotation(
        "is_input_file_templated", item.atom_array.res_id == 2
    )
    item.atom_array.set_annotation(
        "ground_truth_conformer_policy",
        np.where(
            item.atom_array.res_id == 2,
            GroundTruthConformerPolicy.REPLACE,
            GroundTruthConformerPolicy.IGNORE,
        ),
    )
    restored = InferenceInput.from_atom_array(
        item.atom_array, template_selection="A/GLY/3"
    )
    atoms = restored.to_pipeline_input()["atom_array"]
    np.testing.assert_array_equal(
        atoms.is_input_file_templated, np.isin(atoms.res_id, [2, 3])
    )
    assert np.all(
        atoms.ground_truth_conformer_policy[atoms.res_id == 2]
        == GroundTruthConformerPolicy.REPLACE
    )


def test_conditioning_survives_nonpolymer_author_residue_numbering():
    source = components_to_atom_array([{"ccd_code": "NAG", "chain_id": "L"}])
    source.set_annotation("auth_seq_id", np.full(len(source), 101))
    source.set_annotation("is_input_file_templated", np.ones(len(source), dtype=bool))
    source.set_annotation(
        "ground_truth_conformer_policy",
        np.full(len(source), GroundTruthConformerPolicy.REPLACE),
    )
    original_names = source.atom_name.copy()
    result = InferenceInput.from_atom_array(source).atom_array

    assert np.all(source.res_id == 1)  # Caller-owned identities stay unchanged.
    assert np.all(result.res_id == 101)
    original_atoms = np.isin(result.atom_name, original_names)
    assert original_atoms.sum() == len(source)
    assert (~original_atoms).any()  # CCD completion added unresolved hydrogens.
    np.testing.assert_array_equal(result.is_input_file_templated, original_atoms)
    np.testing.assert_array_equal(
        result.ground_truth_conformer_policy,
        np.where(
            original_atoms,
            GroundTruthConformerPolicy.REPLACE,
            GroundTruthConformerPolicy.IGNORE,
        ),
    )


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
    result = InferenceInput.from_atom_array(atoms).atom_array
    assert atoms.get_annotation_categories() == original.get_annotation_categories()
    for name in atoms.get_annotation_categories():
        np.testing.assert_array_equal(
            atoms.get_annotation(name), original.get_annotation(name)
        )
    np.testing.assert_array_equal(atoms.coord, original.coord)
    assert result.charge[result.res_name == "CL"].item() == -1
    assert result.charge[result.res_name == "NA"].item() == 1
