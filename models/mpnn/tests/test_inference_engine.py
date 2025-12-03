"""
Tests for the MPNNInferenceEngine in 'mpnn.inference_engines.mpnn'.

These tests focus on:
- Configuration validation behaviour.
- Basic model/IO config post-processing.
- 'run()' control flow, RNG seeding, and batch handling.
- The '_write_outputs' helper and its side effects.

Engine-level and behavioral tests for MPNN inference.

This module also focuses on:
  - MPNNInferenceEngine end-to-end behavior using a real structure file.
  - Temperature behavior via entropy on decoder log-probs.
  - Design-scope annotations and fixed-residue identity.
  - Omit -> big negative bias + no omitted token in designed sequences.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch
from atomworks.constants import DICT_THREE_TO_ONE
from atomworks.io.utils.io_utils import to_cif_file
from atomworks.ml.utils.testing import cached_parse
from biotite.structure import AtomArray
from mpnn.inference_engines import mpnn as engine_mod
from mpnn.inference_engines.mpnn import MPNNInferenceEngine
from mpnn.model.mpnn import ProteinMPNN
from mpnn.pipelines.mpnn import build_mpnn_transform_pipeline
from mpnn.transforms.feature_aggregation.token_encodings import (
    MPNN_TOKEN_ENCODING,
)
from test_utils import (
    PDB_IDS,
    create_feature_collator,
    prepare_features,
)

###############################################################################
# Helpers (engine stub + simple AtomArray)
###############################################################################


def _make_engine_stub() -> engine_mod.MPNNInferenceEngine:
    """
    Construct an uninitialized engine instance suitable for testing internals.

    NOTE: This bypasses __init__, so any attribute that tests rely on
    must be populated here.
    """
    engine = object.__new__(engine_mod.MPNNInferenceEngine)
    engine.model_type = "protein_mpnn"
    engine.checkpoint_path = "/tmp/ckpt.pt"
    engine.is_legacy_weights = False
    engine.out_directory = None
    engine.write_fasta = False
    engine.write_structures = False
    engine.device = torch.device("cpu")
    engine.model = None
    engine.metrics = None
    engine.allowed_model_types = {"protein_mpnn", "ligand_mpnn"}
    return engine


def _make_simple_atom_array(n_residues: int = 3) -> AtomArray:
    """
    Minimal AtomArray for engine tests.

    One CA atom per residue, all in chain A with sequential residue IDs.
    """
    n_atoms = n_residues
    atom_array = AtomArray(n_atoms)

    atom_array.chain_id = np.array(["A"] * n_atoms)
    atom_array.res_id = np.arange(1, n_atoms + 1, dtype=int)
    atom_array.ins_code = np.array([""] * n_atoms)
    atom_array.atom_name = np.array(["CA"] * n_atoms)
    atom_array.res_name = np.array(["ALA"] * n_atoms)
    atom_array.coord = np.zeros((n_atoms, 3), dtype=float)
    atom_array.occupancy = np.ones(n_atoms, dtype=float)
    atom_array.set_annotation("atomize", np.zeros(n_atoms, dtype=bool))

    return atom_array


###############################################################################
# Engine configuration validation
###############################################################################


def test_validate_model_config_invalid_model_type() -> None:
    engine = _make_engine_stub()
    engine.model_type = "not_a_model"
    with pytest.raises(ValueError):
        engine._validate_model_config()


def test_validate_model_config_checkpoint_and_legacy_flags(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"dummy")

    engine = _make_engine_stub()
    engine.model_type = "protein_mpnn"
    engine.checkpoint_path = str(ckpt)
    engine.is_legacy_weights = True

    # Should not raise for valid configuration.
    engine._validate_model_config()

    # Non-string checkpoint_path.
    engine.checkpoint_path = 123  # type: ignore[assignment]
    with pytest.raises(TypeError):
        engine._validate_model_config()


def test_validate_output_config_types_and_required_outdir() -> None:
    engine = _make_engine_stub()

    # Non-string out_directory.
    engine.out_directory = 123  # type: ignore[assignment]
    with pytest.raises(TypeError):
        engine._validate_output_config()

    # Restore and test boolean flags.
    engine.out_directory = None
    engine.write_fasta = "yes"  # type: ignore[assignment]
    with pytest.raises(TypeError):
        engine._validate_output_config()

    engine.write_fasta = True
    engine.write_structures = False
    # With write_fasta True and out_directory None, ValueError expected.
    with pytest.raises(ValueError):
        engine._validate_output_config()


def test_post_process_engine_config_makes_paths_absolute(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"dummy")

    out_dir = tmp_path / "out"

    engine = _make_engine_stub()
    engine.checkpoint_path = str(ckpt)
    engine.out_directory = str(out_dir)

    engine._post_process_engine_config()

    assert Path(engine.checkpoint_path).is_absolute()
    assert Path(engine.out_directory).is_absolute()


###############################################################################
# run() cargument validation.
###############################################################################


def test_run_argument_validation() -> None:
    engine = _make_engine_stub()

    with pytest.raises(ValueError):
        _ = engine.run(input_dicts=None, atom_arrays=None)

    with pytest.raises(ValueError):
        _ = engine.run(
            input_dicts=[{}],
            atom_arrays=[_make_simple_atom_array(), _make_simple_atom_array()],
        )


###############################################################################
# Engine-level and behavioral tests for MPNN inference
###############################################################################


@pytest.fixture(scope="session")
def protein_checkpoint_path(tmp_path_factory) -> str:
    """
    Build a small ProteinMPNN checkpoint on disk and return its path.

    This ensures MPNNInferenceEngine exercises its normal non-legacy
    checkpoint-loading pathway.
    """
    tmp_dir = tmp_path_factory.mktemp("mpnn_engine_ckpt")
    ckpt_path = tmp_dir / "protein_mpnn_dummy.pt"

    model = ProteinMPNN()
    ckpt = {"model": model.state_dict()}
    torch.save(ckpt, ckpt_path)

    return str(ckpt_path)


@pytest.fixture(scope="session")
def structure_info(tmp_path_factory) -> Dict[str, Any]:
    """
    Create a realistic CIF structure file and expose both the path and the
    AtomArray that was written.

    We:
      - take the first PDB ID from PDB_IDS
      - run it through the standard MPNN pipeline to get an AtomArray
      - write that AtomArray to CIF

    The engine later re-parses this CIF from disk using its own utils.
    """
    pdb_id = PDB_IDS[0]

    # Use your existing cached_parse + pipeline to get the structure.
    data = cached_parse(pdb_id)
    pipeline = build_mpnn_transform_pipeline(
        model_type="protein_mpnn",
        is_inference=True,
    )
    pipeline_output = pipeline(data)
    atom_array = pipeline_output["atom_array"]

    tmp_dir = tmp_path_factory.mktemp("mpnn_engine_structure")
    cif_path = tmp_dir / f"{pdb_id}.cif"

    # Write without extra MPNN annotations; engine will rebuild them.
    to_cif_file(atom_array, cif_path, file_type="cif")

    return {
        "pdb_id": pdb_id,
        "path": str(cif_path),
        "atom_array": atom_array,
    }


@pytest.fixture(scope="session")
def engine(protein_checkpoint_path: str) -> MPNNInferenceEngine:
    """
    Engine configured for ProteinMPNN, CPU, and no output writing.
    """
    return MPNNInferenceEngine(
        model_type="protein_mpnn",
        checkpoint_path=protein_checkpoint_path,
        is_legacy_weights=False,
        out_directory=None,
        write_fasta=False,
        write_structures=False,
        device="cpu",
    )


###############################################################################
# Small helpers
###############################################################################


def _base_input_dict(structure_path: str, **overrides: Any) -> dict[str, Any]:
    """
    Minimal per-input dict; everything else is defaulted inside
    MPNNInferenceInput via the engine.

    Required:
        structure_path
    Optional (examples):
        seed, batch_size, number_of_batches, temperature, omit, ...
    """
    base: dict[str, Any] = {
        "structure_path": structure_path,
        "name": "engine_test",
    }
    base.update(overrides)
    return base


def _get_residue_keys(atom_array) -> List[Tuple[str, int, str, str]]:
    """
    Collect unique residue keys from an AtomArray as:
        (chain_id, res_id, ins_code, res_name)

    We take the first encountered atom for each residue.
    """
    mapping: Dict[Tuple[str, int, str], str] = {}
    chain_ids = atom_array.chain_id
    res_ids = getattr(atom_array, "res_id", None)
    ins_codes = getattr(atom_array, "ins_code", None)
    res_names = atom_array.res_name

    if res_ids is None:
        raise RuntimeError("AtomArray is missing 'res_id' annotation.")

    if ins_codes is None:
        # Treat missing ins_code as empty string.
        ins_codes = np.array([""] * atom_array.array_length(), dtype=object)

    for i in range(atom_array.array_length()):
        chain = str(chain_ids[i])
        res_id = int(res_ids[i])
        ins_code = str(ins_codes[i]) if ins_codes[i] is not None else ""
        res_name = str(res_names[i])

        key = (chain, res_id, ins_code)
        if key not in mapping:
            mapping[key] = res_name

    return [(c, r, ic, rn) for (c, r, ic), rn in mapping.items()]


def _res_id_string(chain_id: str, res_id: int, ins_code: str) -> str:
    """
    Build an ID string compatible with MPNNInferenceInput._parse_id, e.g.:
        'A35', 'A35B', 'AB12', 'AB12C'
    """
    if ins_code:
        return f"{chain_id}{res_id}{ins_code}"
    return f"{chain_id}{res_id}"


def _clone_nested(inp: dict[str, Any]) -> dict[str, Any]:
    """
    Deep-ish clone for network_input: tensors are cloned, nested dicts recurse.
    """
    out: dict[str, Any] = {}
    for k, v in inp.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.clone()
        elif isinstance(v, dict):
            out[k] = _clone_nested(v)
        else:
            out[k] = v
    return out


def _compute_entropy_from_output(network_output: Dict[str, Any]) -> torch.Tensor:
    """
    Compute per-position entropy from decoder probs/log_probs.
    Returns a tensor of shape (B, L).
    """
    decoder = network_output["decoder_features"]
    probs = decoder["probs"]  # (B, L, V)
    log_probs = decoder["log_probs"]  # (B, L, V)
    entropy = -(probs * log_probs).sum(dim=-1)  # (B, L)
    return entropy


###############################################################################
# Engine smoke + basic behavior
###############################################################################


def test_engine_run_smoke(
    engine: MPNNInferenceEngine,
    structure_info: Dict[str, Any],
) -> None:
    """
    Basic smoke test: engine.run() should produce outputs with
    reasonable metadata and atom arrays, using a real CIF file.
    """
    structure_path = structure_info["path"]

    input_dict = _base_input_dict(
        structure_path,
        seed=0,
        batch_size=2,
        number_of_batches=1,
        temperature=0.5,
    )

    outputs = engine.run(
        input_dicts=[input_dict],
        atom_arrays=None,
    )

    # batch_size * number_of_batches designs
    assert len(outputs) == 2

    for out in outputs:
        # Designed sequence present & non-empty.
        seq = out.output_dict.get("designed_sequence")
        assert isinstance(seq, str)
        assert len(seq) > 0

        # Metadata present.
        assert out.output_dict.get("batch_idx") is not None
        assert out.output_dict.get("design_idx") is not None
        assert out.output_dict.get("model_type") == "protein_mpnn"

        # AtomArray looks sane.
        atom_array = out.atom_array
        assert atom_array.coord.shape[0] > 0
        assert atom_array.coord.shape[1] == 3


###############################################################################
# Temperature behavior via entropy on log_probs
###############################################################################


@pytest.mark.parametrize("repeat_sample_num", [8])
def test_temperature_entropy_low_vs_high(
    engine: MPNNInferenceEngine,
    structure_info: Dict[str, Any],
    repeat_sample_num: int,
) -> None:
    """
    Lower temperature should produce lower-entropy (sharper) distributions
    than higher temperature, all else fixed.

    We use engine.model (same weights as the engine checkpoint) but drive it
    through the standard pipeline + collator so we can access decoder
    probs/log_probs.
    """
    pdb_id = structure_info["pdb_id"]

    # Build pipeline output and network_input as in other tests.
    data = cached_parse(pdb_id)
    pipeline = build_mpnn_transform_pipeline(
        model_type="protein_mpnn",
        is_inference=True,
    )
    pipeline_output = pipeline(data)

    prepare_features(
        pipeline_output["input_features"],
        repeat_sample_num=repeat_sample_num,
    )

    collator = create_feature_collator()
    base_input = collator([pipeline_output])

    # Sanity: temperature feature must exist.
    temp_tensor = base_input["input_features"].get("temperature")
    assert temp_tensor is not None

    # Fix RNG for determinism.
    torch.manual_seed(0)
    np.random.seed(0)

    cold_input = _clone_nested(base_input)
    hot_input = _clone_nested(base_input)

    cold_input["input_features"]["temperature"].fill_(0.01)
    hot_input["input_features"]["temperature"].fill_(2.0)

    with torch.no_grad():
        out_cold = engine.model(cold_input)
        out_hot = engine.model(hot_input)

    entropy_cold = _compute_entropy_from_output(out_cold).mean().item()
    entropy_hot = _compute_entropy_from_output(out_hot).mean().item()

    # Cold distributions should be sharper (lower entropy).
    assert entropy_cold < entropy_hot


###############################################################################
# Engine-level annotation behavior (design scope, temperature, omit)
###############################################################################


def test_engine_design_scope_annotation_and_fixed_residues(
    engine: MPNNInferenceEngine,
    structure_info: Dict[str, Any],
) -> None:
    """
    Use fixed_residues in the per-input dict and verify that:

      - mpnn_designed_residue_mask exists and is boolean.
      - At least some residues are fixed (mask False) and some designed (True).
      - Fixed residues retain the same residue type (res_name) as the input
        structure.

    We identify fixed residues based on actual residue IDs from the
    input AtomArray, then instruct the engine via fixed_residues IDs.
    """
    structure_path = structure_info["path"]
    base_atom_array = structure_info["atom_array"]

    residue_keys = _get_residue_keys(base_atom_array)
    if len(residue_keys) < 2:
        pytest.skip("Not enough residues to meaningfully test design scope.")

    # Choose a subset of residues to fix (but leave at least one residue
    # free to design).
    n_res = len(residue_keys)
    fixed_count = min(max(1, n_res // 2), n_res - 1)
    fixed_subset = residue_keys[:fixed_count]

    # Build mapping from residue key -> res_name for the base structure.
    base_residue_names: Dict[Tuple[str, int, str], str] = {}
    for chain_id, res_id, ins_code, res_name in residue_keys:
        base_residue_names[(chain_id, res_id, ins_code)] = res_name

    fixed_residue_ids = [
        _res_id_string(chain_id, res_id, ins_code)
        for (chain_id, res_id, ins_code, _res_name) in fixed_subset
    ]

    input_dict = _base_input_dict(
        structure_path,
        batch_size=1,
        number_of_batches=1,
        fixed_residues=fixed_residue_ids,
    )

    outputs = engine.run(
        input_dicts=[input_dict],
        atom_arrays=None,
    )
    assert len(outputs) == 1
    atom_array = outputs[0].atom_array

    # Annotation must exist.
    assert "mpnn_designed_residue_mask" in atom_array.get_annotation_categories()
    mask = atom_array.mpnn_designed_residue_mask

    assert mask.dtype == np.bool_
    assert mask.shape[0] == atom_array.array_length()

    # We expect at least one designed and at least one fixed atom.
    assert mask.any()
    assert (~mask).any()

    # Check that fixed residues retain their residue type and are marked
    # as not designed.
    chain_ids_new = atom_array.chain_id
    res_ids_new = getattr(atom_array, "res_id", None)
    ins_codes_new = getattr(atom_array, "ins_code", None)
    res_names_new = atom_array.res_name

    if res_ids_new is None:
        raise RuntimeError("Output AtomArray is missing 'res_id' annotation.")

    if ins_codes_new is None:
        ins_codes_new = np.array(
            ["" for _ in range(atom_array.array_length())], dtype=object
        )

    n_checked = 0
    for chain_id, res_id, ins_code, _ in fixed_subset:
        key = (chain_id, res_id, ins_code)
        base_resname = base_residue_names[key]

        # Find atoms corresponding to this residue in the new array.
        atom_mask = (chain_ids_new == chain_id) & (res_ids_new == res_id)
        if ins_code:
            atom_mask &= ins_codes_new == ins_code

        if not atom_mask.any():
            # If, for some reason, this residue disappeared (e.g. shenanigans
            # in assembly), just skip it.
            continue

        n_checked += 1

        # All atoms for this residue should be fixed (mask False).
        assert (~mask[atom_mask]).all()

        # Residue name should be identical to the base structure.
        new_resnames = res_names_new[atom_mask]
        assert all(str(rn) == base_resname for rn in new_resnames)

    # Sanity: we should have checked at least one fixed residue.
    assert n_checked > 0


def test_engine_temperature_annotation_from_inputs(
    engine: MPNNInferenceEngine,
    structure_info: Dict[str, Any],
) -> None:
    """
    Global + per-residue temperature settings in the per-input dict should
    produce mpnn_temperature on the output atom_array.

    We assert that:
      - the annotation exists
      - it is float32
      - global temperature is present
      - per-residue temperature overrides at least one residue
        (using a real residue ID from the structure).
    """
    structure_path = structure_info["path"]
    base_atom_array = structure_info["atom_array"]

    residue_keys = _get_residue_keys(base_atom_array)
    if not residue_keys:
        pytest.skip("No residues found to test temperature annotation.")

    # Pick a real residue from the structure.
    chain_id, res_id, ins_code, _ = residue_keys[0]
    res_id_str = _res_id_string(chain_id, res_id, ins_code)

    input_dict = _base_input_dict(
        structure_path,
        batch_size=1,
        number_of_batches=1,
        temperature=0.5,
        temperature_per_residue={res_id_str: 2.0},
    )

    outputs = engine.run(
        input_dicts=[input_dict],
        atom_arrays=None,
    )
    atom_array = outputs[0].atom_array

    assert "mpnn_temperature" in atom_array.get_annotation_categories()
    temps = atom_array.mpnn_temperature

    assert temps.dtype == np.float32
    assert temps.shape[0] == atom_array.array_length()

    # Check that both 0.5 and 2.0 appear somewhere in the temperature
    # annotation (we don't pin them to specific indices to avoid depending
    # on residue ordering).
    assert np.isclose(temps, 0.5).any()
    assert np.isclose(temps, 2.0).any()


def test_engine_omit_constructs_large_negative_bias_annotation(
    engine: MPNNInferenceEngine,
    structure_info: Dict[str, Any],
) -> None:
    """
    Using 'omit' in the per-input dict should construct a large
    negative bias for the omitted token in the atom-level mpnn_bias
    annotation on the engine's output atom_array.
    """
    structure_path = structure_info["path"]

    input_dict = _base_input_dict(
        structure_path,
        batch_size=1,
        number_of_batches=1,
        omit=["ALA"],
    )

    outputs = engine.run(
        input_dicts=[input_dict],
        atom_arrays=None,
    )
    atom_array = outputs[0].atom_array

    assert "mpnn_bias" in atom_array.get_annotation_categories()
    mpnn_bias = atom_array.mpnn_bias  # shape (n_atoms, vocab)

    ala_idx = MPNN_TOKEN_ENCODING.token_to_idx["ALA"]
    ala_bias = mpnn_bias[:, ala_idx]

    # Bias for omitted token should be a very large negative number everywhere.
    assert np.all(ala_bias < -1e6)


def test_engine_omit_prevents_token_in_designed_sequences(
    engine: MPNNInferenceEngine,
    structure_info: Dict[str, Any],
) -> None:
    """
    Strong omit-based bias should prevent the omitted residue from
    appearing in any designed sequences, as observed via engine outputs.
    """
    structure_path = structure_info["path"]
    ala_one_letter = DICT_THREE_TO_ONE["ALA"]

    input_dict = _base_input_dict(
        structure_path,
        omit=["ALA"],
        batch_size=16,
        number_of_batches=1,
        temperature=0.5,
        seed=0,
    )

    outputs = engine.run(
        input_dicts=[input_dict],
        atom_arrays=None,
    )

    for out in outputs:
        seq = out.output_dict["designed_sequence"]
        assert ala_one_letter not in seq
