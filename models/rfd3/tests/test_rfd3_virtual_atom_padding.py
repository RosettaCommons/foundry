"""Vectorised virtual-atom padding against the previous per-token loop.

``PadTokensWithVirtualAtoms.forward`` used to build and insert the virtual atoms of
each token in a Python loop. The loop is kept here verbatim as the oracle, and the
vectorised transform must reproduce its output exactly: every annotation (order,
dtype, values), coordinates bit for bit, bonds, raised errors and NumPy RNG use.

PYTEST_DONT_REWRITE: the oracle's assertion messages must stay unmodified.
"""

import copy
import random
from pathlib import Path

import biotite.structure as struc
import numpy as np
import pytest
import torch
from atomworks.io.utils.atom_array_plus import AtomArrayPlus, insert_atoms
from atomworks.ml.utils.token import get_token_starts
from rfd3.constants import ATOM14_ATOM_NAMES, VIRTUAL_ATOM_ELEMENT_NAME
from rfd3.inference.input_parsing import DesignInputSpecification
from rfd3.transforms.pipelines import build_atom14_base_pipeline
from rfd3.transforms.util_transforms import (
    assert_single_representative,
    get_af3_token_representative_masks,
)
from rfd3.transforms.virtual_atoms import (
    PadTokensWithVirtualAtoms,
    map_to_association_scheme,
    permute_symmetric_atom_names_,
)

from foundry.common import exists

INPUT_PDBS = Path(__file__).resolve().parents[1] / "docs" / "input_pdbs"

# Inference transform arguments stored in the released RFD3 checkpoint, without
# the residue cache (a cluster path) and with a small diffusion batch.
PIPELINE_ARGS = dict(
    is_inference=True,
    return_atom_array=True,
    diffusion_batch_size=2,
    sigma_data=16,
    central_atom="CB",
    n_atoms_per_token=14,
    association_scheme="dense",
    center_option="diffuse",
    generate_conformers=True,
    generate_conformers_for_non_protein_only=True,
    provide_reference_conformer_when_unmasked=False,
    ground_truth_conformer_policy="IGNORE",
    provide_elements_for_unindexed_components=True,
    use_element_for_atom_names_of_atomized_tokens=True,
    residue_cache_dir=None,
    max_binder_length=170,
    atom_1d_features={
        "ref_atom_name_chars": 256,
        "ref_element": 128,
        "ref_charge": 1,
        "ref_mask": 1,
        "ref_is_motif_atom_with_fixed_coord": 1,
        "ref_is_motif_atom_unindexed": 1,
        "has_zero_occupancy": 1,
        "ref_pos": 3,
        "ref_atomwise_rasa": 3,
        "active_donor": 1,
        "active_acceptor": 1,
        "is_atom_level_hotspot": 1,
    },
    token_1d_features={
        "ref_motif_token_type": 3,
        "restype": 32,
        "ref_plddt": 1,
        "is_non_loopy": 1,
    },
)


def _pdb(name):
    return str(INPUT_PDBS / name)


SPECS = {
    "uncond_40": {"length": 40},
    "uncond_63": {"length": 63},
    # Indexed motifs with fixed sequence and coordinates.
    "motif_fixed": {
        "input": _pdb("7v11.pdb"),
        "contig": "6,A396-402,5,A425-427,4",
        "length": "25",
    },
    # Unfixed motif sequence: those residues are padded from their own CB, and
    # the motif tryptophans (A399, A401) already have 14 atoms (n_pad == 0).
    "motif_unfixed_seq": {
        "input": _pdb("7v11.pdb"),
        "contig": "6,A396-402,5,A425-427,4",
        "length": "25",
        "select_unfixed_sequence": "A398-401",
        "select_fixed_atoms": {"A396-397": "BKBN", "A425-427": "TIP"},
    },
    # Unindexed motifs next to a ligand.
    "unindexed_ligand": {
        "input": _pdb("7v11.pdb"),
        "ligand": "OQO",
        "unindex": "A431,A572-573",
        "length": "30",
        "select_fixed_atoms": {"A431": "TIP", "A572": "BKBN", "A573": "BKBN"},
        "allow_ligand_on_existing_chain": True,
    },
    # Atomized small molecule.
    "ligand": {
        "input": _pdb("IAI.pdb"),
        "ligand": "IAI",
        "length": "30",
        "select_fixed_atoms": {"IAI": ""},
    },
    # Nucleic-acid tokens are never padded.
    "dna": {
        "input": _pdb("1bna.pdb"),
        "contig": "A1-4,/0,B21-24,/0,20",
        "length": "28",
    },
}
E2E_SPECS = ["uncond_40", "uncond_63", "motif_fixed", "motif_unfixed_seq"]


def _reference_forward(self, data: dict) -> dict:
    """``PadTokensWithVirtualAtoms.forward`` before vectorisation (the oracle)."""
    atom_array = data["atom_array"]
    starts = get_token_starts(atom_array, add_exclusive_stop=True)
    token_starts = starts[:-1]
    token_level_array = atom_array[token_starts]
    is_motif_atom_with_fixed_seq = token_level_array.is_motif_atom_with_fixed_seq
    is_motif_token_unindexed = token_level_array.is_motif_atom_unindexed

    token_ids = np.unique(atom_array.token_id)
    assert len(token_ids) == len(
        is_motif_atom_with_fixed_seq
    ), "Token ids and token level array have different lengths!"

    is_residue = (
        token_level_array.is_protein & ~token_level_array.atomize
    ) | is_motif_token_unindexed
    is_paddable = is_residue & ~(
        is_motif_atom_with_fixed_seq | is_motif_token_unindexed
    )
    is_non_paddable_residue = is_residue & (
        is_motif_atom_with_fixed_seq | is_motif_token_unindexed
    )

    virtual_atoms_to_insert = []
    insert_positions = []

    for token_id, (start, end) in enumerate(zip(starts[:-1], starts[1:])):
        if is_paddable[token_id]:
            token = atom_array[start:end]
            n_pad = self.n_atoms_per_token - len(token)
            if n_pad > 0:
                mask = get_af3_token_representative_masks(
                    token, central_atom=self.atom_to_pad_from
                )
                assert_single_representative(token)

                pad_atoms = token[mask].copy()
                pad_atoms = (
                    pad_atoms[0]
                    if isinstance(pad_atoms, struc.AtomArray)
                    else pad_atoms
                )
                pad_atoms.element = VIRTUAL_ATOM_ELEMENT_NAME

                pad_array = struc.array([pad_atoms] * n_pad)

                occ = 1.0 if pad_atoms.occupancy.sum() > 0.0 else 0.0
                pad_array.occupancy = np.full(n_pad, occ)

                pad_array.is_motif_atom = np.zeros(n_pad, dtype=bool)

                if data["is_inference"]:
                    pad_array.is_motif_atom_with_fixed_coord = np.zeros(
                        n_pad, dtype=token.is_motif_atom_with_fixed_coord.dtype
                    )

                def _fix_multidimensional_annotations_in_pad_array(atomarray, padarray):
                    for annotation in atomarray.get_annotation_categories():
                        if len(atomarray.get_annotation(annotation).shape) > 1:
                            stacked = np.stack(
                                padarray.get_annotation(annotation)
                            ).astype(float)
                            padarray.del_annotation(annotation)
                            padarray.set_annotation(annotation, stacked)
                    return padarray

                pad_array = _fix_multidimensional_annotations_in_pad_array(
                    token, pad_array
                )

                virtual_atoms_to_insert.append(pad_array)
                insert_positions.append(end)

    if virtual_atoms_to_insert:
        atom_array_padded = insert_atoms(
            atom_array, virtual_atoms_to_insert, insert_positions
        )
    else:
        atom_array_padded = atom_array

    if "gt_atom_name" not in atom_array_padded.get_annotation_categories():
        atom_array_padded.set_annotation(
            "gt_atom_name", np.empty(len(atom_array_padded), dtype="U4")
        )

    starts_padded = get_token_starts(atom_array_padded, add_exclusive_stop=True)

    for token_id, (start, end) in enumerate(zip(starts_padded[:-1], starts_padded[1:])):
        if is_paddable[token_id]:
            if not data["is_inference"] and exists(self.association_scheme):
                atom_names = permute_symmetric_atom_names_(
                    ATOM14_ATOM_NAMES,
                    atom_array_padded.res_name[start],
                    association_map=self.association_map_,
                    symmetry_map=self.symmetry_map_,
                )
            else:
                atom_names = ATOM14_ATOM_NAMES
            atom_array_padded.atom_name[start:end] = atom_names
            atom_array_padded.get_annotation("gt_atom_name")[start:end] = atom_names

        elif is_non_paddable_residue[token_id]:
            atom_names, res_name = (
                atom_array_padded.atom_name[start:end],
                atom_array_padded.res_name[start],
            )
            atom_array_padded.get_annotation("gt_atom_name")[start:end] = atom_names
            atom_names = map_to_association_scheme(
                atom_names, res_name, scheme=self.association_scheme
            )
            atom_array_padded.atom_name[start:end] = atom_names
        else:
            atom_names = atom_array_padded.atom_name[start:end]
            atom_array_padded.get_annotation("gt_atom_name")[start:end] = atom_names

        assert {VIRTUAL_ATOM_ELEMENT_NAME} != set(
            atom_array_padded.element[start:end].tolist()
        ), (
            "Padded atoms should be virtual atoms, but found: "
            f"{set(atom_array_padded.element[start:end].tolist())}"
        )

    data["atom_array"] = atom_array_padded
    return data


def _seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def _pipeline_input(name):
    spec = DesignInputSpecification.safe_init(**copy.deepcopy(SPECS[name]))
    return spec.to_pipeline_input(example_id=name)


@pytest.fixture(scope="module")
def pipeline():
    return build_atom14_base_pipeline(**PIPELINE_ARGS)


@pytest.fixture(scope="module")
def pad_inputs(pipeline):
    """Data dicts as they arrive at ``PadTokensWithVirtualAtoms`` in the pipeline."""
    captured = {}
    original_forward = PadTokensWithVirtualAtoms.forward

    def capture(self, data):
        captured[data["example_id"]] = copy.deepcopy(data)
        return original_forward(self, data)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(PadTokensWithVirtualAtoms, "forward", capture)
        for name in SPECS:
            _seed_everything(0)
            pipeline(_pipeline_input(name))
    assert set(captured) == set(SPECS)
    return captured


def _assert_atom_arrays_identical(expected, actual):
    assert type(actual) is type(expected)
    assert actual.array_length() == expected.array_length()
    assert actual.get_annotation_categories() == expected.get_annotation_categories()
    for category in expected.get_annotation_categories():
        exp = expected.get_annotation(category)
        act = actual.get_annotation(category)
        assert act.dtype == exp.dtype, category
        assert act.shape == exp.shape, category
        if exp.dtype.kind == "f":
            assert np.array_equal(
                act.view(f"u{exp.dtype.itemsize}"), exp.view(f"u{exp.dtype.itemsize}")
            ), category
        else:
            assert np.array_equal(act, exp), category
    assert actual.coord.dtype == expected.coord.dtype
    assert np.array_equal(actual.coord.view(np.uint32), expected.coord.view(np.uint32))
    if expected.bonds is None:
        assert actual.bonds is None
    else:
        assert actual.bonds.get_atom_count() == expected.bonds.get_atom_count()
        assert np.array_equal(actual.bonds.as_array(), expected.bonds.as_array())
    assert (actual.box is None) == (expected.box is None)
    if expected.box is not None:
        assert np.array_equal(actual.box, expected.box)
    if isinstance(expected, AtomArrayPlus):
        assert actual._annot_2d.keys() == expected._annot_2d.keys()
        for name in expected._annot_2d:
            assert np.array_equal(
                actual._annot_2d[name].as_array(), expected._annot_2d[name].as_array()
            )


def _run(transform, forward, data, seed):
    data = copy.deepcopy(data)
    ids_before = {key: id(value) for key, value in data.items()}
    np.random.seed(seed)
    try:
        out = forward(transform, data)
    except Exception as error:  # noqa: BLE001 - errors are compared below
        return ("error", type(error), str(error), np.random.get_state()[1].copy())
    ids_after = {key: id(value) for key, value in out.items()}
    # The transform only replaces the atom array.
    assert out is data
    assert ids_after.keys() == ids_before.keys()
    assert {k for k in ids_before if ids_before[k] != ids_after[k]} <= {"atom_array"}
    return ("ok", out, np.random.get_state()[1].copy())


def _assert_same_behaviour(transform, data, seed=0):
    """Run oracle and vectorised transform on copies of ``data`` and compare.

    Returns the number of inserted atoms, or the exception type both raised.
    """
    expected = _run(transform, _reference_forward, data, seed)
    actual = _run(transform, PadTokensWithVirtualAtoms.forward, data, seed)
    assert actual[0] == expected[0]
    assert np.array_equal(actual[-1], expected[-1]), "NumPy RNG use differs"
    if expected[0] == "error":
        assert actual[1:3] == expected[1:3]
        return expected[1]
    _assert_atom_arrays_identical(expected[1]["atom_array"], actual[1]["atom_array"])
    return expected[1]["atom_array"].array_length() - len(data["atom_array"])


def _transform(**overrides):
    kwargs = dict(n_atoms_per_token=14, atom_to_pad_from="CB")
    kwargs["association_scheme"] = overrides.pop("association_scheme", "dense")
    kwargs.update(overrides)
    return PadTokensWithVirtualAtoms(**kwargs)


@pytest.mark.parametrize("name", list(SPECS))
@pytest.mark.parametrize("is_inference", [True, False])
def test_pipeline_inputs_match_reference(pad_inputs, name, is_inference):
    data = copy.deepcopy(pad_inputs[name])
    data["is_inference"] = is_inference
    n_inserted = _assert_same_behaviour(_transform(), data)
    if name != "dna":
        assert n_inserted > 0
    if name == "motif_unfixed_seq":
        # Some paddable tokens already have 14 atoms and receive no virtual atoms.
        atom_array = data["atom_array"]
        starts = get_token_starts(atom_array, add_exclusive_stop=True)
        tokens = atom_array[starts[:-1]]
        is_paddable = (
            tokens.is_protein
            & ~tokens.atomize
            & ~tokens.is_motif_atom_with_fixed_seq
            & ~tokens.is_motif_atom_unindexed
        )
        assert np.any(is_paddable & (np.diff(starts) >= 14))
        assert np.any(is_paddable & (np.diff(starts) < 14))


@pytest.mark.parametrize("is_inference", [True, False])
def test_symmetric_swaps_consume_rng_identically(pad_inputs, is_inference):
    transform = _transform()
    transform.symmetry_map_ = {"ALA": [[5, 6], [7, 8, 9]], "TRP": [[10, 11]]}
    for name in ("uncond_40", "motif_unfixed_seq"):
        data = copy.deepcopy(pad_inputs[name])
        data["is_inference"] = is_inference
        for seed in (0, 1):
            assert _assert_same_behaviour(transform, data, seed=seed) > 0


@pytest.mark.parametrize("n_atoms_per_token", [5, 20])
def test_other_atom_counts_fail_like_reference(pad_inputs, n_atoms_per_token):
    # Atom names are atom14 names, so other counts fail when names are assigned.
    for name in ("uncond_40", "motif_unfixed_seq"):
        transform = _transform(n_atoms_per_token=n_atoms_per_token)
        assert _assert_same_behaviour(transform, pad_inputs[name]) is ValueError


def test_without_association_scheme(pad_inputs):
    transform = _transform(association_scheme=None)
    for is_inference in (True, False):
        data = copy.deepcopy(pad_inputs["uncond_40"])
        data["is_inference"] = is_inference
        assert _assert_same_behaviour(transform, data) > 0
        # Fixed-sequence residues need a scheme: both versions raise the same error.
        data = copy.deepcopy(pad_inputs["motif_fixed"])
        data["is_inference"] = is_inference
        assert _assert_same_behaviour(transform, data) is ValueError


def test_pad_from_other_atom(pad_inputs):
    for name in ("uncond_40", "motif_unfixed_seq"):
        assert _assert_same_behaviour(
            _transform(atom_to_pad_from="CA"), pad_inputs[name]
        )

    # Several candidates (two atoms named CA): the first one is copied.
    data = copy.deepcopy(pad_inputs["uncond_40"])
    atom_array = data["atom_array"]
    atom_array.coord = np.arange(
        atom_array.array_length() * 3, dtype=np.float32
    ).reshape(-1, 3)
    start, end = _token_slices(atom_array)[6]
    o_idx = start + np.flatnonzero(atom_array.atom_name[start:end] == "O")[0]
    atom_array.atom_name[o_idx] = "CA"
    assert _assert_same_behaviour(_transform(atom_to_pad_from="CA"), data) > 0


def test_multidimensional_annotations_bonds_and_atom_array_plus(pad_inputs):
    data = copy.deepcopy(pad_inputs["motif_unfixed_seq"])
    atom_array = data["atom_array"]
    n = atom_array.array_length()
    atom_array.set_annotation("vec_int", np.arange(3 * n, dtype=np.int32).reshape(n, 3))
    atom_array.set_annotation(
        "vec_float", np.linspace(0, 1, 2 * n, dtype=np.float32).reshape(n, 2)
    )
    atom_array.occupancy[::7] = 0.0
    atom_array.bonds = struc.connect_via_residue_names(atom_array)
    assert atom_array.bonds.get_bond_count() > 0
    for is_inference in (True, False):
        data["is_inference"] = is_inference
        assert _assert_same_behaviour(_transform(), data) > 0

    data["atom_array"] = AtomArrayPlus.from_atom_array(atom_array)
    assert _assert_same_behaviour(_transform(), data) > 0


def _token_slices(atom_array):
    starts = get_token_starts(atom_array, add_exclusive_stop=True)
    return list(zip(starts[:-1], starts[1:]))


def test_invalid_tokens_raise_like_reference(pad_inputs):
    base = pad_inputs["uncond_40"]
    slices = _token_slices(base["atom_array"])

    # No representative atom: the CB of the third token is removed.
    data = copy.deepcopy(base)
    start, end = slices[2]
    keep = np.ones(data["atom_array"].array_length(), dtype=bool)
    keep[start:end] &= data["atom_array"].atom_name[start:end] != "CB"
    data["atom_array"] = data["atom_array"][keep]
    assert _assert_same_behaviour(_transform(), data) is AssertionError

    # Two representative atoms: the O of the fifth token is renamed to CB.
    data = copy.deepcopy(base)
    start, end = slices[4]
    o_idx = start + np.flatnonzero(data["atom_array"].atom_name[start:end] == "O")[0]
    data["atom_array"].atom_name[o_idx] = "CB"
    assert _assert_same_behaviour(_transform(), data) is AssertionError

    # Padding from CA, with CA missing but a single CB: indexing an empty selection.
    data = copy.deepcopy(base)
    start, end = slices[3]
    keep = np.ones(data["atom_array"].array_length(), dtype=bool)
    keep[start:end] &= data["atom_array"].atom_name[start:end] != "CA"
    data["atom_array"] = data["atom_array"][keep]
    assert _assert_same_behaviour(_transform(atom_to_pad_from="CA"), data) is IndexError


@pytest.mark.parametrize("name", E2E_SPECS)
def test_pipeline_features_match_reference(pipeline, name):
    def run(forward):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(PadTokensWithVirtualAtoms, "forward", forward)
            _seed_everything(123)
            return pipeline(_pipeline_input(name))

    expected = run(_reference_forward)
    actual = run(PadTokensWithVirtualAtoms.forward)

    assert actual["feats"].keys() == expected["feats"].keys()
    for key, exp in expected["feats"].items():
        act = actual["feats"][key]
        if isinstance(exp, torch.Tensor):
            assert act.dtype == exp.dtype, key
            assert act.shape == exp.shape, key
            assert torch.equal(act, exp) or (
                exp.is_floating_point()
                and torch.equal(act.isnan(), exp.isnan())
                and torch.equal(act.nan_to_num(), exp.nan_to_num())
            ), key
        else:
            assert type(act) is type(exp), key
            assert np.array_equal(np.asarray(act), np.asarray(exp)), key
    assert torch.equal(
        actual["coord_atom_lvl_to_be_noised"], expected["coord_atom_lvl_to_be_noised"]
    )
    _assert_atom_arrays_identical(expected["atom_array"], actual["atom_array"])
