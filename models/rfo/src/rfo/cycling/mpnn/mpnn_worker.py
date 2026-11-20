"""Adapter for a separately installed https://github.com/dauparas/LigandMPNN.

No Foundry MPNN imports are used. The external interpreter may live in its own
Conda/venv environment, or in an optional MPNN_SIF Apptainer image.
"""

from __future__ import annotations

import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from Bio.PDB import PDBIO, PDBParser
from Bio.SeqUtils import seq1, seq3

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

CHECKPOINTS = {
    "protein_mpnn": "proteinmpnn_v_48_020.pt",
    "ligand_mpnn": "ligandmpnn_v_32_010_25.pt",
}


def settings():
    """Resolve and validate the user's external MPNN installation."""
    repo = os.environ.get("MPNN_REPO", "")
    if not repo:
        raise ValueError(
            "Set MPNN_REPO to your upstream LigandMPNN checkout and MPNN_PYTHON "
            "to its environment's Python. See models/rfo/docs/installation.md."
        )
    repo = Path(repo).expanduser().resolve()
    if not (repo / "run.py").is_file():
        raise FileNotFoundError(f"MPNN_REPO must contain LigandMPNN run.py: {repo}")
    for module in ("model_utils.py", "data_utils.py"):
        if not (repo / module).is_file():
            raise FileNotFoundError(f"MPNN_REPO must contain upstream {module}: {repo}")
    model = os.environ.get("MPNN_MODEL_TYPE", "protein_mpnn")
    if model not in CHECKPOINTS:
        raise ValueError(f"MPNN_MODEL_TYPE must be one of {tuple(CHECKPOINTS)}")
    checkpoint = Path(
        os.environ.get("MPNN_CKPT") or repo / "model_params" / CHECKPOINTS[model]
    )
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"MPNN weights missing: {checkpoint}. Run LigandMPNN's "
            "get_model_params.sh or set MPNN_CKPT."
        )
    sif = os.environ.get("MPNN_SIF", "")
    python = os.environ.get("MPNN_PYTHON") or ("python" if sif else sys.executable)
    if sif:
        sif = str(Path(sif).expanduser().resolve())
        if not Path(sif).is_file():
            raise FileNotFoundError(f"MPNN_SIF does not exist: {sif}")
        if not shutil.which("apptainer"):
            raise FileNotFoundError("MPNN_SIF requires apptainer on PATH.")
    elif not shutil.which(python):
        raise FileNotFoundError(f"MPNN_PYTHON is not executable: {python}")
    return repo, model, checkpoint, python, sif


def chain_residues(structure, chain_id):
    model = next(structure.get_models())
    if chain_id not in model:
        raise ValueError(f"Designed chain {chain_id!r} is missing from the structure.")
    residues = [res for res in model[chain_id] if res.id[0] == " "]
    if not residues:
        raise ValueError(f"Designed chain {chain_id!r} has no protein residues.")
    for res in residues:
        if seq1(res.resname) not in "ACDEFGHIKLMNPQRSTVWY" or not all(
            a in res for a in ("N", "CA", "C", "O")
        ):
            raise ValueError(
                f"MPNN needs standard amino acids with N/CA/C/O in the designed "
                f"chain; check {chain_id}{res.id[1]}{res.id[2].strip()}."
            )
    return residues


def select_mutation(scores, residues, positions, designed_chain):
    """Choose one non-native AA by raw probability; ties follow chain/AA order."""
    rows = {row["residue_id"]: row for row in scores}
    if len(rows) != len(scores):
        raise ValueError("MPNN scores contain duplicate residue IDs.")
    best = None
    for i in positions:
        residue = residues[i]
        residue_id = f"{designed_chain}{residue.id[1]}{residue.id[2].strip()}"
        native = seq1(residue.resname)
        row = rows.get(residue_id)
        if row is None or row["native_aa"] != native:
            raise ValueError(f"MPNN scores do not match input residue {residue_id}.")
        probs = row["probabilities"]
        for aa in AMINO_ACIDS:
            value = probs.get(aa)
            if value is None or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid MPNN probability for {residue_id}/{aa}.")
            if aa != native and (best is None or value > best["probability"]):
                best = dict(
                    position_0idx=i,
                    residue_id=residue_id,
                    from_aa=native,
                    to_aa=aa,
                    probability=value,
                )
    if best is None or best["probability"] <= 0:
        raise ValueError("MPNN returned no positive-probability substitution.")
    return best


def redesign(
    structure_file,
    fixed_residues_0idx,
    out_dir,
    num_seqs,
    temperature,
    seed,
    designed_chain="A",
):
    """Return (chain sequence, PDB path) after one maximum-likelihood mutation.

    Masks are zero-based positions in the designed protein chain. Translate
    them to actual PDB residue IDs (including gaps and insertion codes). Keep
    all other chains, ligands and fixed atoms from the input complex intact.
    """
    if num_seqs < 1 or temperature <= 0:
        raise ValueError("MPNN num_seqs and temperature must be positive.")
    source = Path(structure_file).resolve()
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "mpnn_result.json"
    result_path.unlink(missing_ok=True)
    structure = PDBParser(QUIET=True).get_structure("input", str(source))
    residues = chain_residues(structure, designed_chain)
    fixed = set(fixed_residues_0idx)
    if any(i < 0 or i >= len(residues) for i in fixed):
        raise ValueError("MPNN fixed residue position is outside the designed chain.")
    positions = [i for i in range(len(residues)) if i not in fixed]
    ids = [
        f"{designed_chain}{residues[i].id[1]}{residues[i].id[2].strip()}"
        for i in positions
    ]
    design_path = out / "mpnn_design.pdb"
    spec = dict(
        structure_path=str(source),
        designed_chain=designed_chain,
        designed_residues=ids,
        fixed_residues_0idx=sorted(fixed),
        num_seqs=num_seqs,
        temperature=temperature,
        seed=seed,
    )

    mutation = None

    # No eligible substitution: retain the original complex without scoring.
    if not positions:
        shutil.copy2(source, design_path)
    else:
        repo, model, checkpoint, python, sif = settings()
        spec.update(
            model_type=model, checkpoint_path=str(checkpoint), mpnn_repo=str(repo)
        )
        # Score in the user's MPNN environment; the host needs no torch import.
        # Fresh artifacts prevent failed/retried steps from reusing old scores.
        worker = Path(__file__).with_name("score_worker.py").resolve()
        with tempfile.TemporaryDirectory(prefix="mpnn_", dir=out) as temp_dir:
            score_path = Path(temp_dir) / "scores.json"
            spec_path = Path(temp_dir) / "spec.json"
            spec_path.write_text(json.dumps(spec))
            command = [python, str(worker), str(spec_path), str(score_path)]
            if sif:
                binds = [
                    str(repo),
                    str(worker.parent),
                    str(source.parent),
                    str(out),
                    str(checkpoint.parent),
                ]
                extra = os.environ.get("MPNN_BIND_PATHS", "")
                if extra:
                    binds.append(extra)
                command = [
                    "apptainer",
                    "exec",
                    "--nv",
                    "-B",
                    ",".join(binds),
                    sif,
                    *command,
                ]
            print("\nMPNN cmd:\n" + shlex.join(command), flush=True)
            # Avoid leaking Foundry's PYTHONPATH into the external environment.
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            try:
                subprocess.run(command, cwd=repo, env=env, check=True)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"External MPNN failed with exit code {exc.returncode}"
                ) from exc
            if not score_path.is_file():
                raise RuntimeError("MPNN produced no single-residue scores.")
            scores = json.loads(score_path.read_text())["scores"]
            mutation = select_mutation(scores, residues, positions, designed_chain)
            residue = residues[mutation["position_0idx"]]
            residue.resname = seq3(mutation["to_aa"]).upper()
            # Only the mutated residue loses chemically invalid old side chains.
            for atom in list(residue):
                if atom.name not in {"N", "CA", "C", "O", "OXT"}:
                    residue.detach_child(atom.id)
            writer = PDBIO()
            writer.set_structure(structure)
            writer.save(str(design_path))
            # Preserve scoring inputs/probabilities for inspection.
            samples = out / "samples"
            if samples.exists():
                shutil.rmtree(samples)
            shutil.copytree(temp_dir, samples)
    sequence = "".join(seq1(res.resname) for res in residues)
    (out / "mpnn_spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    result_path.write_text(
        json.dumps(
            {
                "sequence": sequence,
                "structure_pdb": str(design_path),
                "mutation": mutation,
            },
            indent=2,
        )
        + "\n"
    )
    return sequence, str(design_path)
