"""Exercise single-mutation selection and the external scoring subprocess."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from Bio.PDB import PDBIO, Atom, Chain, Model, PDBParser, Residue, Structure
from rfo.cycling.mpnn.mpnn_worker import (
    AMINO_ACIDS,
    chain_residues,
    redesign,
    select_mutation,
    settings,
)


@pytest.fixture
def complex_pdb(tmp_path):
    structure = Structure.Structure("complex")
    model = Model.Model(0)
    structure.add(model)
    for chain_id, rows in {
        "A": [(10, " ", "ALA"), (12, " ", "GLY"), (12, "B", "SER")],
        "B": [(7, " ", "TYR")],
        "L": [(1, " ", "LIG")],
    }.items():
        chain = Chain.Chain(chain_id)
        model.add(chain)
        for number, icode, name in rows:
            residue = Residue.Residue(
                ("H_LIG" if name == "LIG" else " ", number, icode), name, ""
            )
            chain.add(residue)
            for index, atom_name in enumerate(
                ["C1"] if name == "LIG" else ["N", "CA", "C", "O", "CB"]
            ):
                residue.add(
                    Atom.Atom(
                        atom_name,
                        np.array([index, number, 0.0]),
                        90.0,
                        1.0,
                        " ",
                        f"{atom_name:>4}",
                        index,
                        element=atom_name[0],
                    )
                )
    path = tmp_path / "input complex.pdb"
    writer = PDBIO()
    writer.set_structure(structure)
    writer.save(str(path))
    return path


@pytest.fixture
def upstream(tmp_path, monkeypatch):
    repo = tmp_path / "external MPNN"
    repo.mkdir()
    (repo / "model_params").mkdir()
    for checkpoint in ("proteinmpnn_v_48_020.pt", "ligandmpnn_v_32_010_25.pt"):
        torch.save(
            {"model_state_dict": {}, "num_edges": 48, "atom_context_num": 16},
            repo / "model_params" / checkpoint,
        )
    (repo / "run.py").touch()
    (repo / "data_utils.py").write_text("""
import torch
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
alphabet = "ACDEFGHIKLMNPQRSTVWYX"
def parse_PDB(path, device):
    residues = [r for r in PDBParser(QUIET=True).get_structure("s", path)[0].get_residues()
                if r.id[0] == " "]
    return dict(S=torch.tensor([alphabet.index(seq1(r.resname)) for r in residues], device=device),
                R_idx=torch.tensor([r.id[1] for r in residues], device=device),
                chain_letters=[r.parent.id for r in residues],
                mask=torch.ones(len(residues), device=device)), None, None, [r.id[2].strip() for r in residues], None

def featurize(protein, **kwargs):
    return {key: protein[key][None] for key in ("S", "mask", "chain_mask")}
""")
    (repo / "model_utils.py").write_text("""
import json, os
from pathlib import Path
import torch
from data_utils import alphabet
class ProteinMPNN:
    def __init__(self, **kwargs):
        assert "PYTHONPATH" not in os.environ
        self.settings = kwargs
    def to(self, device): return self
    def eval(self): return self
    def load_state_dict(self, state): pass
    def score(self, features, use_sequence):
        assert use_sequence is True
        order = torch.argsort((features["chain_mask"] + 0.0001) * features["randn"].abs())
        index = int(torch.nonzero(features["chain_mask"])[0, 1])
        assert torch.all(order[:, -1] == index)
        rows = [{"C": .99}, {"G": .79, "V": .20}, {"V": .80, "W": .10}, {"C": .99}]
        probs = []
        for row in rows:
            rest = (1 - sum(row.values())) / (len(alphabet) - len(row))
            probs.append([row.get(aa, rest) for aa in alphabet])
        invocation = {"model_type": self.settings["model_type"],
                      "batch_size": features["batch_size"], "use_sequence": use_sequence}
        Path("invocation.json").write_text(json.dumps(invocation))
        batched = torch.tensor(probs)[None].repeat(features["batch_size"], 1, 1)
        if features["batch_size"] > 1:
            # The first decoding order prefers A12V, but the mean prefers A12BV.
            for batch in range(features["batch_size"]):
                batched[batch, 1] = .01 / (len(alphabet) - 2)
                batched[batch, 1, alphabet.index("V")] = .95 if batch == 0 else .01
                batched[batch, 1, alphabet.index("G")] = .04 if batch == 0 else .98
        return {"log_probs": batched.log(), "decoding_order": order}
""")
    monkeypatch.setenv("MPNN_REPO", str(repo))
    monkeypatch.setenv("MPNN_PYTHON", sys.executable)
    monkeypatch.delenv("MPNN_SIF", raising=False)
    monkeypatch.delenv("MPNN_CKPT", raising=False)
    monkeypatch.setenv("PYTHONPATH", "/must/not/leak/foundry/models/mpnn")
    return repo


@pytest.mark.parametrize("model", ["protein_mpnn", "ligand_mpnn"])
def test_external_design_preserves_complex_and_maps_residue_ids(
    complex_pdb, upstream, tmp_path, monkeypatch, model
):
    monkeypatch.setenv("MPNN_MODEL_TYPE", model)
    sequence, output = redesign(
        str(complex_pdb), [0], str(tmp_path / "results with spaces"), 3, 0.25, 19
    )
    assert sequence == "AGV"
    assert sum(a != b for a, b in zip("AGS", sequence)) == 1
    args = json.loads((upstream / "invocation.json").read_text())
    spec = json.loads((Path(output).parent / "mpnn_spec.json").read_text())
    assert spec["designed_residues"] == ["A12", "A12B"]
    assert spec["seed"] == 19
    assert args["batch_size"] == 3
    assert args["model_type"] == model
    assert args["use_sequence"] is True
    scores = json.loads((Path(output).parent / "samples/scores.json").read_text())[
        "scores"
    ]
    assert scores[0]["probabilities"]["V"] == pytest.approx((0.95 + 0.01 + 0.01) / 3)
    result = PDBParser(QUIET=True).get_structure("out", output)[0]
    original = PDBParser(QUIET=True).get_structure("in", str(complex_pdb))[0]
    assert "CB" in result["A"][10]  # fixed side chain retained
    assert "CB" in result["A"][12]  # unfixed but unchosen side chain retained
    assert "CB" not in result["A"][(" ", 12, "B")]  # only mutation loses side chain
    for chain in ("B", "L"):
        for old, new in zip(original[chain].get_atoms(), result[chain].get_atoms()):
            assert old.name == new.name
            assert old.parent.resname == new.parent.resname
            np.testing.assert_array_equal(old.coord, new.coord)
    metadata = json.loads((Path(output).parent / "mpnn_result.json").read_text())
    assert metadata["sequence"] == "AGV"
    assert metadata["mutation"]["residue_id"] == "A12B"
    assert metadata["mutation"]["from_aa"] == "S"
    assert metadata["mutation"]["to_aa"] == "V"
    assert metadata["mutation"]["probability"] == pytest.approx(0.8)
    assert Path(metadata["structure_pdb"]).is_file()


def test_all_fixed_skips_upstream(complex_pdb, upstream, tmp_path):
    seq, output = redesign(
        str(complex_pdb), [0, 1, 2], str(tmp_path / "fixed"), 1, 0.1, 1
    )
    assert seq == "AGS"
    assert Path(output).read_bytes() == complex_pdb.read_bytes()
    assert not (upstream / "invocation.json").exists()
    assert (
        json.loads(Path(output).with_name("mpnn_result.json").read_text())["mutation"]
        is None
    )


def test_missing_install_has_setup_message(monkeypatch):
    monkeypatch.delenv("MPNN_REPO", raising=False)
    with pytest.raises(ValueError, match="installation.md"):
        settings()


def test_missing_weights_fail_before_launch(upstream):
    (upstream / "model_params" / "proteinmpnn_v_48_020.pt").unlink()
    with pytest.raises(FileNotFoundError, match="get_model_params"):
        settings()


def test_invalid_mask_rejected(complex_pdb, tmp_path):
    with pytest.raises(ValueError, match="outside"):
        redesign(str(complex_pdb), [3], str(tmp_path / "invalid"), 1, 0.1, 1)


def test_missing_chain_rejected(complex_pdb, tmp_path):
    with pytest.raises(ValueError, match="missing"):
        redesign(str(complex_pdb), [], str(tmp_path / "missing"), 1, 0.1, 1, "Z")


def test_success_without_output_cannot_reuse_stale_result(
    complex_pdb, upstream, tmp_path, monkeypatch
):
    from rfo.cycling.mpnn import mpnn_worker

    out = tmp_path / "stale"
    out.mkdir()
    (out / "mpnn_result.json").write_text('{"sequence":"STALE"}')
    monkeypatch.setattr(mpnn_worker.subprocess, "run", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="no single-residue scores"):
        redesign(str(complex_pdb), [], str(out), 1, 0.1, 1)
    assert not (out / "mpnn_result.json").exists()


def test_redesigned_complex_is_used_by_the_next_cycle(
    complex_pdb, upstream, tmp_path, monkeypatch
):
    import argparse

    from rfo.cli import DEFAULTS
    from rfo.cycling import cycle

    seen_sequences = []

    def structure_step(template, structure, *args, **kwargs):
        seen_sequences.append(cycle.get_chain_sequence(structure, "A"))
        return {"method": "Boltz"}, structure, [0]

    monkeypatch.setattr(cycle, "validate_dependencies", lambda args: settings())
    monkeypatch.setattr(cycle, "run_boltz_branch", structure_step)
    output = tmp_path / "cycling"
    params = dict(
        DEFAULTS,
        input_structure=str(complex_pdb),
        out_dir=str(output),
        total_cycles=2,
        backprop_fraction=0.0,
    )
    cycle.run_loop(argparse.Namespace(**params))
    assert seen_sequences == ["AGS", "AGV"]
    records = json.loads((output / "input complex_records.json").read_text())
    assert [r["mpnn_sequence"] for r in records] == ["AGV", "AVV"]
    assert (output / "inputs" / "input complex_2.pdb").is_file()


def probability_rows():
    return [
        dict(
            residue_id=residue_id,
            native_aa=aa,
            probabilities=dict.fromkeys(AMINO_ACIDS, 0.01),
        )
        for residue_id, aa in [("A10", "A"), ("A12", "G"), ("A12B", "S")]
    ]


def test_selection_excludes_fixed_native_and_unknown(complex_pdb):
    residues = chain_residues(
        PDBParser(QUIET=True).get_structure("s", complex_pdb), "A"
    )
    rows = probability_rows()
    rows[0]["probabilities"]["W"] = 0.99  # fixed position
    rows[1]["probabilities"].update(G=0.98, V=0.6, X=0.999)
    rows[2]["probabilities"]["W"] = 0.7
    chosen = select_mutation(rows[::-1], residues, [1, 2], "A")
    assert chosen["residue_id"] == "A12B"
    assert chosen["to_aa"] == "W"
    assert chosen["probability"] == 0.7


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1, None])
def test_invalid_scores_rejected(complex_pdb, bad):
    residues = chain_residues(
        PDBParser(QUIET=True).get_structure("s", complex_pdb), "A"
    )
    rows = probability_rows()
    rows[2]["probabilities"]["W"] = bad
    with pytest.raises(ValueError, match="Invalid MPNN probability"):
        select_mutation(rows, residues, [1, 2], "A")


def test_score_sequence_and_numbering_must_match(complex_pdb):
    residues = chain_residues(
        PDBParser(QUIET=True).get_structure("s", complex_pdb), "A"
    )
    rows = probability_rows()
    rows[2]["residue_id"] = "A13"
    with pytest.raises(ValueError, match="do not match"):
        select_mutation(rows, residues, [1, 2], "A")
    rows[2].update(residue_id="A12B", native_aa="A")
    with pytest.raises(ValueError, match="do not match"):
        select_mutation(rows, residues, [1, 2], "A")


def test_probability_ties_are_deterministic(complex_pdb):
    residues = chain_residues(
        PDBParser(QUIET=True).get_structure("s", complex_pdb), "A"
    )
    chosen = select_mutation(probability_rows()[::-1], residues, [1, 2], "A")
    assert chosen["residue_id"] == "A12"
    assert chosen["to_aa"] == "A"


def test_apptainer_binds_scoring_worker(complex_pdb, upstream, tmp_path, monkeypatch):
    from rfo.cycling.mpnn import mpnn_worker

    image = tmp_path / "mpnn image.sif"
    image.touch()
    monkeypatch.setenv("MPNN_SIF", str(image))
    monkeypatch.setattr(mpnn_worker.shutil, "which", lambda name: "/bin/apptainer")
    original_run = mpnn_worker.subprocess.run

    def container_run(command, **kwargs):
        assert command[:4] == ["apptainer", "exec", "--nv", "-B"]
        assert str(Path(mpnn_worker.__file__).parent) in command[4].split(",")
        assert command[5] == str(image)
        return original_run(command[6:], **kwargs)

    monkeypatch.setattr(mpnn_worker.subprocess, "run", container_run)
    sequence, _ = redesign(complex_pdb, [0], tmp_path / "out", 1, 0.1, 7)
    assert sequence == "AGV"
