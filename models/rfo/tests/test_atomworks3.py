"""Exercise RFO's real AtomWorks 3 selection, parsing and RF3 features on CPU."""

import importlib

import numpy as np
import pytest
import torch
from atomworks.io.tools.inference import components_to_atom_array
from atomworks.io.utils.io_utils import to_cif_file
from biotite.structure import AtomArray
from omegaconf import OmegaConf

ADAPTERS = ["af3", "af3_auto_detect_breaking", "af3_no_breaking_point"]


@pytest.fixture(params=ADAPTERS)
def adapter(request):
    return importlib.import_module(f"rfo.backprop.tools.{request.param}")


@pytest.fixture
def atoms():
    array = AtomArray(8)
    array.chain_id = np.array(["A"] * 4 + ["B"] * 4)
    array.res_id = np.array([1, 1, 2, 2] * 2)
    array.res_name[:] = "ALA"
    array.atom_name = np.array(["N", "CA"] * 4)
    array.coord = np.arange(24, dtype=np.float32).reshape(8, 3)
    array.set_annotation("is_polymer", np.ones(8, dtype=bool))
    return array


@pytest.mark.parametrize(
    "query, indices",
    [
        (None, []),
        ("B", [4, 5, 6, 7]),
        ("A, B", list(range(8))),
        ("A1-2", [0, 1, 2, 3]),
        ("A1-1, B2-2", [0, 1, 6, 7]),
        ("A/*/[1-2]/CA", [1, 3]),
        ("[A,B]/*/2/N", [2, 6]),
    ],
)
def test_template_selection(adapter, atoms, query, indices):
    original = atoms.coord.copy()
    selected = adapter.apply_template_selection(atoms, query)
    assert np.flatnonzero(selected.is_input_file_templated).tolist() == indices
    np.testing.assert_array_equal(selected.coord, original)


def test_template_query_does_not_silently_drop_missing_residues(adapter, atoms):
    with pytest.raises(ValueError):
        adapter.apply_template_selection(atoms, "A/*/99/CA")


def test_templates_reject_nonpolymer_atoms(adapter, atoms):
    atoms.is_polymer[0] = False
    with pytest.raises(AssertionError, match="polymer"):
        adapter.apply_template_selection(atoms, "A")


def test_templates_reject_missing_coordinates(adapter, atoms):
    atoms.coord[:] = np.nan
    with pytest.raises(AssertionError, match="NaN"):
        adapter.apply_template_selection(atoms, "A")


def inference_pipeline():
    from rfo.backprop.inference_pipeline.af3_inference_pipeline import (
        AF3InferencePipeline,
    )

    # Exercise the production initializer without loading a checkpoint/model.
    obj = AF3InferencePipeline.__new__(AF3InferencePipeline)
    obj.n_recycles = 1
    obj.diffusion_batch_size = 1
    obj.cfg = OmegaConf.create(
        {
            "datasets": {
                "val": {
                    "test": {
                        "dataset": {
                            "transform": {
                                "_target_": "rf3.data.pipelines.build_af3_transform_pipeline",
                                "is_inference": False,
                                "protein_msa_dirs": [],
                                "rna_msa_dirs": [],
                                "residue_cache_dir": "/missing/training-only/cache",
                                "add_cyclic_bonds": True,
                            }
                        }
                    }
                }
            }
        }
    )
    obj._initialize_pipeline()
    return obj.pipeline


def test_checkpoint_training_settings_are_overridden(monkeypatch):
    from rfo.backprop.inference_pipeline import af3_inference_pipeline as module

    seen = []
    monkeypatch.setattr(module.hydra.utils, "instantiate", seen.append)
    inference_pipeline()
    assert seen[0].residue_cache_dir is None
    assert seen[0].add_cyclic_bonds is False
    assert seen[0].is_inference is True
    assert seen[0].run_confidence_head is True


@pytest.mark.parametrize("seed", range(8))
def test_sequence_placeholders_do_not_create_cyclic_chains(tmp_path, seed):
    from rfo.backprop.tools.af3 import cif2input

    array, _ = components_to_atom_array(
        [
            {"seq": "AAAA", "chain_id": "A", "chain_type": "polypeptide(L)"},
            {"seq": "GGGG", "chain_id": "B", "chain_type": "polypeptide(L)"},
        ],
        return_components=True,
    )
    path = tmp_path / "linear.cif"
    to_cif_file(array, path)
    np.random.seed(seed)
    torch.manual_seed(seed)
    _, confidence, output = cif2input(
        path, inference_pipeline(), handle_chain_breaks=False
    )
    bonds = np.asarray(output["feats"]["token_bonds"])
    assert bonds.shape == (8, 8)
    assert not bonds[0, 3] and not bonds[3, 0]
    assert not bonds[4, 7] and not bonds[7, 4]
    assert not output["feats"].get("cyclic_asym_ids", [])
    assert "seq" in confidence and "frame_atom_idxs" in confidence
    assert torch.isfinite(output["coord_atom_lvl_to_be_noised"]).all()


@pytest.mark.parametrize("with_template", [False, True])
def test_target_first_input_retains_sorted_chain_contract(
    adapter, tmp_path, with_template
):
    # AW3 preserves file order; RFO reconstructs components in sorted order.
    # Different sequences detect a silent binder/target swap.
    array, _ = components_to_atom_array(
        [
            {"seq": "GGGG", "chain_id": "B", "chain_type": "polypeptide(L)"},
            {"seq": "AAAA", "chain_id": "A", "chain_type": "polypeptide(L)"},
        ],
        return_components=True,
    )
    path = tmp_path / "target_first.cif"
    to_cif_file(array, path)
    captured = []

    class Captured(Exception):
        pass

    def capture(data):
        captured.append(data["atom_array"].copy())
        raise Captured

    fn = adapter.cif2input_with_template if with_template else adapter.cif2input
    with pytest.raises(Captured):
        fn(path, capture)
    parsed = captured[0]
    assert list(dict.fromkeys(parsed.chain_id)) == ["A", "B"]
    assert set(parsed.res_name[parsed.chain_id == "A"]) == {"ALA"}
    assert set(parsed.res_name[parsed.chain_id == "B"]) == {"GLY"}


def test_target_first_tokens_match_sequence_reconstruction(tmp_path):
    from atomworks.ml.encoding_definitions import AF3SequenceEncoding
    from rfo.backprop.tools.af3 import cif2input, decode_restype, seq2cif

    array, _ = components_to_atom_array(
        [
            {"seq": "GGGG", "chain_id": "B", "chain_type": "polypeptide(L)"},
            {"seq": "AAAA", "chain_id": "A", "chain_type": "polypeptide(L)"},
        ],
        return_components=True,
    )
    path = tmp_path / "target_first.cif"
    to_cif_file(array, path)
    network, _, data = cif2input(path, inference_pipeline(), handle_chain_breaks=False)
    seq = decode_restype(network["f"]["restype"], AF3SequenceEncoding())
    reconstructed, components = seq2cif(
        seq,
        add_info={
            "chain_iid_token_level": data["ground_truth"]["chain_iid_token_lvl"],
            "msa_paths": None,
            "smiles": None,
        },
        protein_mask=network["f"]["is_protein"],
        out_file=tmp_path / "reconstructed.cif",
    )
    assert [(c["chain_id"], c["seq"]) for c in components] == [
        ("A", "AAAA"),
        ("B", "GGGG"),
    ]
    assert seq == "AAAAGGGG"
    rebuilt, _, _ = cif2input(
        reconstructed, inference_pipeline(), handle_chain_breaks=False
    )
    assert rebuilt["f"]["is_protein"].all()
    assert decode_restype(rebuilt["f"]["restype"], AF3SequenceEncoding()) == seq
