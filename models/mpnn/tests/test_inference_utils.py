"""
Tests for MPNN inference utilities, containers, annotations, and CLI wiring.

This module covers:
- Helper functions and JSON/CLI utilities in 'mpnn.utils.inference'.
- The 'MPNNInferenceInput' and 'MPNNInferenceOutput' containers.
- Annotation behaviour on AtomArray.
- The top-level CLI entry point in 'mpnn.inference'.
"""

from __future__ import annotations

import argparse
import json
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from biotite.structure import AtomArray
from mpnn.transforms.feature_aggregation.token_encodings import (
    MPNN_TOKEN_ENCODING,
)
from mpnn.utils.inference import (
    MPNNInferenceInput,
    MPNNInferenceOutput,
    _absolute_path_or_none,
    build_arg_parser,
    cli_to_json,
    none_or_type,
    parse_json_like,
    parse_list_like,
    str2bool,
)

###############################################################################
# Helpers
###############################################################################


def _make_simple_atom_array(n_residues: int = 3) -> AtomArray:
    """
    Build a minimal AtomArray for testing MPNNInferenceInput utilities.

    The array has one atom per residue, with:
        - chain_id: 'A'
        - res_id: 1..n_residues
        - ins_code: '' (no insertion codes)
        - atom_name: 'CA'
        - res_name: cycling over a few standard residues
    """
    n_atoms = n_residues
    atom_array = AtomArray(n_atoms)

    # Required structural annotations.
    atom_array.chain_id = np.array(["A"] * n_atoms)
    atom_array.res_id = np.arange(1, n_atoms + 1, dtype=int)
    atom_array.ins_code = np.array([""] * n_atoms)
    atom_array.atom_name = np.array(["CA"] * n_atoms)
    res_names = ["ALA", "GLY", "SER"]
    atom_array.res_name = np.array(
        [res_names[i % len(res_names)] for i in range(n_atoms)]
    )
    atom_array.coord = np.zeros((n_atoms, 3), dtype=float)
    atom_array.occupancy = np.ones(n_atoms, dtype=float)

    # Simple non-atomized view: all False.
    atom_array.set_annotation("atomize", np.zeros(n_atoms, dtype=bool))

    return atom_array


def _make_simple_inference_output() -> MPNNInferenceOutput:
    atom_array = _make_simple_atom_array(2)
    input_dict = {
        "name": "test_name",
        "some_list": [1, 2, 3],
        "some_dict": {"a": 1},
        "none_val": None,
    }
    output_dict = {
        "batch_idx": 0,
        "design_idx": 1,
        "designed_sequence": "AG",
        "sequence_recovery": 0.5,
        "ligand_interface_sequence_recovery": None,
    }
    return MPNNInferenceOutput(
        atom_array=atom_array, input_dict=input_dict, output_dict=output_dict
    )


def _patch_token_helpers(monkeypatch) -> None:
    """
    Patch get_token_starts and spread_token_wise in mpnn.utils.inference
    to a simple 1-atom-per-token behaviour so we don't depend on the full
    atomworks implementation in tests.
    """

    def fake_get_token_starts(atom_array: AtomArray) -> np.ndarray:
        # One token per atom.
        return np.arange(atom_array.array_length(), dtype=int)

    def fake_spread_token_wise(
        atom_array: AtomArray, token_values: np.ndarray
    ) -> np.ndarray:
        # Since we define 1 atom per token in tests, we can just return
        # token_values directly for both 1D and 2D cases.
        return token_values

    monkeypatch.setattr("mpnn.utils.inference.get_token_starts", fake_get_token_starts)
    monkeypatch.setattr(
        "mpnn.utils.inference.spread_token_wise", fake_spread_token_wise
    )


###############################################################################
# Basic helper function tests
###############################################################################


@pytest.mark.parametrize(
    "value, expected",
    [
        ("True", True),
        ("1", True),
        ("False", False),
        ("0", False),
    ],
)
def test_str2bool_valid(value: str, expected: bool) -> None:
    assert str2bool(value) is expected


@pytest.mark.parametrize("value", ["yes", "no", "true ", ""])
def test_str2bool_invalid_raises(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _ = str2bool(value)


@pytest.mark.parametrize(
    "raw, type_fn, expected",
    [
        ("None", int, None),
        ("None", float, None),
        ("None", str, None),
        ("1", int, 1),
        ("1.5", float, 1.5),
        ("foo", str, "foo"),
    ],
)
def test_none_or_type_basic(raw: Any, type_fn, expected: Any) -> None:
    assert none_or_type(raw, type_fn) == expected


###############################################################################
# JSON-like parsing helpers
###############################################################################


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        ({"a": 1}, {"a": 1}),  # passthrough non-str
        ('{"a": 1}', {"a": 1}),
        ("[1, 2]", [1, 2]),
        ("{'a': 1}", {"a": 1}),  # python literal style
        ("3", 3),
        ("True", True),
        ("a,b,c", ["a", "b", "c"]),
        ("single", "single"),
    ],
)
def test_parse_json_like_various(value: Any, expected: Any) -> None:
    result = parse_json_like(value)
    assert result == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        ("[1, 2]", [1, 2]),
        ("1", [1]),
        ("a,b,c", ["a", "b", "c"]),
    ],
)
def test_parse_list_like(value: Any, expected: Any) -> None:
    assert parse_list_like(value) == expected


def test_absolute_path_or_none_basic(tmp_path: Path) -> None:
    assert _absolute_path_or_none(None) is None
    assert _absolute_path_or_none("") is None

    rel = "some_file.txt"
    abs_path = _absolute_path_or_none(rel)
    assert abs_path is not None
    assert Path(abs_path).is_absolute()

    home_like = str(tmp_path / "file.dat")
    # Already absolute -> unchanged (aside from normalization).
    assert Path(_absolute_path_or_none(home_like)) == Path(home_like)


###############################################################################
# Arg parser and CLI → JSON builder
###############################################################################


def test_build_arg_parser_smoke() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--model_type",
            "protein_mpnn",
            "--checkpoint_path",
            "/tmp/ckpt.pt",
            "--is_legacy_weights",
            "True",
            "--structure_path",
            "/tmp/structure.cif",
        ]
    )
    assert args.model_type == "protein_mpnn"
    assert args.checkpoint_path == "/tmp/ckpt.pt"
    assert args.is_legacy_weights is True
    assert args.structure_path == "/tmp/structure.cif"


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--fixed_residues", "A1", "--designed_residues", "A2"],
        ["--fixed_chains", "A", "--designed_chains", "B"],
    ],
)
def test_build_arg_parser_design_scope_mutual_exclusion(
    extra_args: list[str],
) -> None:
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        _ = parser.parse_args(
            [
                "--model_type",
                "protein_mpnn",
                "--checkpoint_path",
                "/tmp/ckpt.pt",
                "--is_legacy_weights",
                "True",
                "--structure_path",
                "/tmp/structure.cif",
            ]
            + extra_args
        )


def test_cli_to_json_requires_core_fields() -> None:
    """
    If config_json is not provided and any core field is missing/None,
    cli_to_json() should raise.
    """
    parser = build_arg_parser()
    args = parser.parse_args([])  # everything at default (mostly None)
    with pytest.raises(ValueError):
        _ = cli_to_json(args)


def test_cli_to_json_builds_single_input_and_parses_fields() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--model_type",
            "protein_mpnn",
            "--checkpoint_path",
            "/tmp/ckpt.pt",
            "--is_legacy_weights",
            "True",
            "--structure_path",
            "/tmp/structure.cif",
            "--out_directory",
            "/tmp/out",
            "--write_fasta",
            "False",
            "--write_structures",
            "True",
            "--seed",
            "123",
            "--batch_size",
            "4",
            "--number_of_batches",
            "2",
            "--remove_ccds",
            "A,B",
            "--remove_waters",
            "True",
            "--occupancy_threshold_sidechain",
            "0.5",
            "--occupancy_threshold_backbone",
            "0.6",
            "--undesired_res_names",
            '["UNK","HOH"]',
            "--structure_noise",
            "0.25",
            "--decode_type",
            "auto_regressive",
            "--causality_pattern",
            "conditional",
            "--initialize_sequence_embedding_with_ground_truth",
            "False",
            "--features_to_return",
            '{"input_features":["mask_for_loss"]}',
            "--atomize_side_chains",
            "False",
            "--fixed_residues",
            '["A1","A2"]',
            "--bias",
            '{"ALA": -1.0}',
            "--omit",
            '["UNK"]',
            "--temperature",
            "0.1",
            "--temperature_per_residue",
            '{"A1": 0.2}',
            "--symmetry_residues",
            '[["A1","A2"]]',
            "--symmetry_residues_weights",
            "[[1.0,0.5]]",
        ]
    )

    config = cli_to_json(args)

    # Top-level keys.
    assert config["model_type"] == "protein_mpnn"
    assert config["checkpoint_path"] == "/tmp/ckpt.pt"
    assert config["is_legacy_weights"] is True
    assert config["out_directory"] == "/tmp/out"
    assert config["write_fasta"] is False
    assert config["write_structures"] is True

    assert isinstance(config["inputs"], list) and len(config["inputs"]) == 1
    inp = config["inputs"][0]

    # Sampling / overrides.
    assert inp["seed"] == 123
    assert inp["batch_size"] == 4
    assert inp["number_of_batches"] == 2
    assert inp["remove_ccds"] == ["A", "B"]
    assert inp["remove_waters"] is True
    assert inp["occupancy_threshold_sidechain"] == 0.5
    assert inp["occupancy_threshold_backbone"] == 0.6
    assert inp["undesired_res_names"] == ["UNK", "HOH"]

    # Scalars and lists.
    assert inp["structure_noise"] == 0.25
    assert inp["decode_type"] == "auto_regressive"
    assert inp["causality_pattern"] == "conditional"
    assert inp["initialize_sequence_embedding_with_ground_truth"] is False
    assert inp["features_to_return"] == {"input_features": ["mask_for_loss"]}
    assert inp["atomize_side_chains"] is False
    assert inp["fixed_residues"] == ["A1", "A2"]
    assert inp["designed_residues"] is None
    assert inp["bias"] == {"ALA": -1.0}
    assert inp["omit"] == ["UNK"]
    assert inp["temperature"] == 0.1
    assert inp["temperature_per_residue"] == {"A1": 0.2}
    assert inp["symmetry_residues"] == [["A1", "A2"]]
    assert inp["symmetry_residues_weights"] == [[1.0, 0.5]]

    # Structure path and name are carried through; full absolutization happens
    # later in MPNNInferenceInput.post_process_inputs.
    assert inp["structure_path"] == "/tmp/structure.cif"
    assert "name" in inp  # may be None; defaults later


def test_cli_to_json_with_config_json_file(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    original = {
        "model_type": "ligand_mpnn",
        "checkpoint_path": "/ckpt.pt",
        "is_legacy_weights": False,
        "out_directory": "/out",
        "write_fasta": True,
        "write_structures": False,
        "inputs": [{"structure_path": "/str.cif"}],
    }
    config_path.write_text(json.dumps(original))

    parser = build_arg_parser()
    args = parser.parse_args(["--config_json", str(config_path)])

    config = cli_to_json(args)
    assert config == original


###############################################################################
# MPNNInferenceInput: ID parsing and masking
###############################################################################


@pytest.mark.parametrize(
    "id_str, kwargs, expected",
    [
        ("A", dict(res_num_required=False, res_num_allowed=True), ("A", None, None)),
        ("AB", dict(res_num_required=False, res_num_allowed=True), ("AB", None, None)),
        ("A35", dict(res_num_required=False, res_num_allowed=True), ("A", 35, None)),
        ("AB12C", dict(res_num_required=False, res_num_allowed=True), ("AB", 12, "C")),
    ],
)
def test_parse_id_valid(
    id_str: str, kwargs: dict[str, Any], expected: tuple[Any, ...]
) -> None:
    result = MPNNInferenceInput._parse_id(id_str, **kwargs)
    assert result == expected


@pytest.mark.parametrize("id_str", ["", "123", "A3.5", "A-1", "!", "A_10"])
def test_parse_id_invalid_raises(id_str: str) -> None:
    with pytest.raises(ValueError):
        _ = MPNNInferenceInput._parse_id(
            id_str, res_num_required=False, res_num_allowed=True
        )


def test_parse_id_res_num_required_and_disallowed() -> None:
    # Missing required residue number.
    with pytest.raises(ValueError):
        _ = MPNNInferenceInput._parse_id(
            "A", res_num_required=True, res_num_allowed=True
        )

    # Residue number present but disallowed.
    with pytest.raises(ValueError):
        _ = MPNNInferenceInput._parse_id(
            "A10", res_num_required=False, res_num_allowed=False
        )


def test_mask_from_ids_basic() -> None:
    atom_array = _make_simple_atom_array(n_residues=4)

    # Target a whole chain.
    mask_chain = MPNNInferenceInput._mask_from_ids(atom_array, ["A"])
    assert mask_chain.shape == (atom_array.array_length(),)
    assert np.all(mask_chain)

    # Target a specific residue by number.
    mask_res = MPNNInferenceInput._mask_from_ids(atom_array, ["A2"])
    assert mask_res.any()
    # One atom per residue -> exactly one entry for residue 2.
    assert mask_res.sum() == 1

    # Target a different residue just to ensure coverage.
    mask_res2 = MPNNInferenceInput._mask_from_ids(atom_array, ["A3"])
    assert mask_res2.any()


###############################################################################
# MPNNInferenceInput: validation and defaults
###############################################################################


def test_from_atom_array_and_dict_requires_structure_path_when_no_atom_array() -> None:
    # When atom_array is None and structure_path is not provided, a ValueError
    # should be raised during validation.
    input_dict: dict[str, Any] = {}
    with pytest.raises(ValueError):
        _ = MPNNInferenceInput.from_atom_array_and_dict(
            atom_array=None, input_dict=input_dict
        )


def test_from_atom_array_and_dict_applies_defaults_and_post_processing() -> None:
    atom_array = _make_simple_atom_array(n_residues=3)
    input_dict: dict[str, Any] = {
        # No structure_path: allowed because atom_array is provided.
        "batch_size": 5,
    }

    inference_input = MPNNInferenceInput.from_atom_array_and_dict(
        atom_array=atom_array, input_dict=input_dict
    )

    # batch_size should be preserved and repeat_sample_num derived from it.
    assert inference_input.input_dict["batch_size"] == 5
    assert inference_input.input_dict["repeat_sample_num"] == 5

    # Name should be set even if we did not provide one explicitly.
    assert inference_input.input_dict["name"] is not None

    # A few representative defaulted keys should be present.
    for key in (
        "fixed_residues",
        "designed_residues",
        "fixed_chains",
        "designed_chains",
        "temperature",
        "omit",
    ):
        assert key in inference_input.input_dict


def test_from_atom_array_and_dict_invalid_scalar_settings_raise() -> None:
    atom_array = _make_simple_atom_array()
    # Invalid decode_type.
    input_dict = {
        "decode_type": "invalid",
    }
    with pytest.raises(ValueError):
        _ = MPNNInferenceInput.from_atom_array_and_dict(
            atom_array=atom_array, input_dict=input_dict
        )


###############################################################################
# MPNNInferenceInput: annotation behaviour
###############################################################################


def test_design_scope_annotation_from_fixed_residues(monkeypatch) -> None:
    """
    If fixed_residues is provided, the design mask should be True for all
    residues except the fixed ones.
    """
    _patch_token_helpers(monkeypatch)

    atom_array = _make_simple_atom_array(n_residues=3)
    input_dict = {
        "fixed_residues": ["A2"],
    }

    inference_input = MPNNInferenceInput.from_atom_array_and_dict(
        atom_array=atom_array,
        input_dict=input_dict,
    )

    design_mask = inference_input.atom_array.get_annotation(
        "mpnn_designed_residue_mask"
    )
    assert design_mask.dtype == bool
    # Residues: 1, 2, 3 -> design all except residue 2.
    assert design_mask.tolist() == [True, False, True]


def test_temperature_annotation_global_and_per_residue(monkeypatch) -> None:
    """
    Global temperature should apply to all residues, with per-residue entries
    overriding the global value for specified residues.
    """
    _patch_token_helpers(monkeypatch)

    atom_array = _make_simple_atom_array(n_residues=3)
    input_dict = {
        "temperature": 0.1,
        "temperature_per_residue": {"A2": 0.2},
    }

    inference_input = MPNNInferenceInput.from_atom_array_and_dict(
        atom_array=atom_array,
        input_dict=input_dict,
    )

    temps = inference_input.atom_array.get_annotation("mpnn_temperature")
    assert temps.shape == (3,)
    np.testing.assert_allclose(temps, [0.1, 0.2, 0.1], rtol=1e-6)


def test_bias_annotation_combines_global_and_omit(monkeypatch) -> None:
    """
    Global bias and omission should be reflected in 'mpnn_bias':
    - ALA gets +1.0 everywhere.
    - GLY gets a large negative omit bias everywhere.
    """
    _patch_token_helpers(monkeypatch)

    atom_array = _make_simple_atom_array(n_residues=3)
    input_dict = {
        "bias": {"ALA": 1.0},
        "omit": ["GLY"],
    }

    inference_input = MPNNInferenceInput.from_atom_array_and_dict(
        atom_array=atom_array,
        input_dict=input_dict,
    )

    bias_arr = inference_input.atom_array.get_annotation("mpnn_bias")
    # shape: [n_atoms, vocab_size]
    assert bias_arr.shape[0] == 3

    token_to_idx = MPNN_TOKEN_ENCODING.token_to_idx
    idx_ala = token_to_idx["ALA"]
    idx_gly = token_to_idx["GLY"]

    # All residues: bias for ALA is 1.0
    np.testing.assert_allclose(bias_arr[:, idx_ala], 1.0, rtol=1e-6)

    # All residues: bias for GLY is a large negative omit bias (~ -1e8)
    # We don't hard-code the exact value, but ensure it's very negative.
    assert np.all(bias_arr[:, idx_gly] < -1e6)


def test_symmetry_annotation_residues_and_weights(monkeypatch) -> None:
    """
    Residue-based symmetry groups should get the same group ID, and
    weights should follow symmetry_residues_weights.
    """
    _patch_token_helpers(monkeypatch)

    atom_array = _make_simple_atom_array(n_residues=3)
    input_dict = {
        "symmetry_residues": [["A1", "A3"]],
        "symmetry_residues_weights": [[1.0, 0.5]],
    }

    inference_input = MPNNInferenceInput.from_atom_array_and_dict(
        atom_array=atom_array,
        input_dict=input_dict,
    )

    arr = inference_input.atom_array
    groups = arr.get_annotation("mpnn_symmetry_equivalence_group")
    weights = arr.get_annotation("mpnn_symmetry_weight")

    assert groups.shape == (3,)
    assert weights.shape == (3,)

    # Residues 1 and 3 share the same group; residue 2 is different.
    assert groups[0] == groups[2]
    assert groups[1] != groups[0]

    # Weights: residue 1 -> 1.0, residue 3 -> 0.5, residue 2 -> default 1.0
    np.testing.assert_allclose(weights, [1.0, 1.0, 0.5], rtol=1e-6)


def test_annotate_atom_array_preserves_existing_design_mask(monkeypatch) -> None:
    """
    If 'mpnn_designed_residue_mask' already exists on the atom array,
    design-scope fields in the input_dict should be ignored.
    """
    _patch_token_helpers(monkeypatch)

    atom_array = _make_simple_atom_array(n_residues=3)
    existing_mask = np.array([False, True, False], dtype=bool)
    atom_array.set_annotation("mpnn_designed_residue_mask", existing_mask)

    input_dict = {
        "fixed_residues": ["A2"],  # would normally change design mask
    }

    inference_input = MPNNInferenceInput.from_atom_array_and_dict(
        atom_array=atom_array,
        input_dict=input_dict,
    )

    new_mask = inference_input.atom_array.get_annotation("mpnn_designed_residue_mask")
    assert np.array_equal(new_mask, existing_mask)


###############################################################################
# MPNNInferenceOutput: extra categories and writing
###############################################################################


def test_build_extra_categories_basic() -> None:
    inference_output = _make_simple_inference_output()
    categories = inference_output._build_extra_categories()

    assert "mpnn_input" in categories
    assert "mpnn_output" in categories

    mpnn_input = categories["mpnn_input"]
    mpnn_output = categories["mpnn_output"]

    # Scalars should be stored as length-1 lists of the scalar.
    assert mpnn_output["batch_idx"] == [0]
    assert mpnn_output["design_idx"] == [1]

    # Lists/dicts/None become JSON strings.
    assert json.loads(mpnn_input["some_list"][0]) == [1, 2, 3]
    assert json.loads(mpnn_input["some_dict"][0]) == {"a": 1}
    assert mpnn_input["none_val"][0] == "null"


def test_write_structure_invokes_to_cif_file(monkeypatch, tmp_path: Path) -> None:
    inference_output = _make_simple_inference_output()

    called: dict[str, Any] = {}

    def fake_to_cif_file(
        atom_array, base_path, file_type, extra_fields, extra_categories
    ):
        called["atom_array"] = atom_array
        called["base_path"] = base_path
        called["file_type"] = file_type
        called["extra_fields"] = extra_fields
        called["extra_categories"] = extra_categories

    monkeypatch.setattr(
        "mpnn.utils.inference.to_cif_file",
        fake_to_cif_file,
    )

    base_path = tmp_path / "out"
    inference_output.write_structure(base_path=base_path, file_type="cif.gz")

    assert called["atom_array"] is inference_output.atom_array
    assert Path(called["base_path"]) == base_path
    assert called["file_type"] == "cif.gz"
    assert "mpnn_input" in called["extra_categories"]
    assert "mpnn_output" in called["extra_categories"]


def test_write_fasta_to_handle() -> None:
    inference_output = _make_simple_inference_output()

    handle = StringIO()
    inference_output.write_fasta(handle=handle)
    contents = handle.getvalue().strip().splitlines()

    assert len(contents) == 2
    header, sequence = contents

    assert header.startswith(">")
    # Name, batch_idx, design_idx should all appear in decorated name.
    assert "test_name_b0_d1" in header
    # Sequence recovery metric should be present.
    assert "sequence_recovery=0.5000" in header
    assert sequence == "AG"


def test_write_fasta_to_path(tmp_path: Path) -> None:
    inference_output = _make_simple_inference_output()
    base_path = tmp_path / "design"
    inference_output.write_fasta(base_path=base_path)

    fasta_path = base_path.with_suffix(".fa")
    assert fasta_path.is_file()
    text = fasta_path.read_text().strip().splitlines()
    assert len(text) == 2
    assert text[1] == "AG"


@pytest.mark.parametrize(
    "base_path, handle",
    [
        (None, None),
        (Path("x.fa"), StringIO()),
    ],
)
def test_write_fasta_argument_errors(
    base_path: Path | None, handle: StringIO | None
) -> None:
    inference_output = _make_simple_inference_output()
    with pytest.raises(ValueError):
        inference_output.write_fasta(base_path=base_path, handle=handle)


def test_write_fasta_requires_sequence() -> None:
    inference_output = _make_simple_inference_output()
    inference_output.output_dict["designed_sequence"] = ""
    with pytest.raises(ValueError):
        inference_output.write_fasta(handle=StringIO())
