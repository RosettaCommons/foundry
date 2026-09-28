"""Adapter for a separately installed https://github.com/dauparas/LigandMPNN.

No Foundry MPNN imports are used. The external interpreter may live in its own
Conda/venv environment, or in an optional MPNN_SIF Apptainer image.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from Bio.PDB import PDBIO, PDBParser
from Bio.SeqUtils import seq1

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


def redesign(
    structure_file,
    fixed_residues_0idx,
    out_dir,
    num_seqs,
    temperature,
    seed,
    designed_chain="A",
):
    """Return (chain sequence, PDB path), selecting the first sampled design.

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

    # Upstream treats an empty redesign list as "design all". Explicitly skip
    # sampling when the entire binder is fixed, retaining the original file.
    if not positions:
        shutil.copy2(source, design_path)
    else:
        repo, model, checkpoint, python, sif = settings()
        spec.update(
            model_type=model, checkpoint_path=str(checkpoint), mpnn_repo=str(repo)
        )
        # A fresh output directory prevents a failed/retried step from reading
        # a previous sample. Keep the first sample as the stable handoff file.
        with tempfile.TemporaryDirectory(prefix="mpnn_", dir=out) as temp_dir:
            command = [
                python,
                str(repo / "run.py"),
                "--pdb_path",
                str(source),
                "--out_folder",
                temp_dir,
                "--model_type",
                model,
                f"--checkpoint_{model}",
                str(checkpoint),
                "--chains_to_design",
                designed_chain,
                "--redesigned_residues",
                " ".join(ids),
                "--batch_size",
                "1",
                "--number_of_batches",
                str(num_seqs),
                "--temperature",
                str(temperature),
                "--seed",
                str(seed),
                "--zero_indexed",
                "0",
            ]
            if sif:
                binds = [
                    str(repo),
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
            # Upstream LigandMPNN parses --zero_indexed as `type=str`, so any
            # non-empty string is truthy and the suffix is always "_0" regardless
            # of the flag value. Accept the 0-indexed name and fall back to
            # 1-indexed for older LigandMPNN builds.
            backbones = Path(temp_dir) / "backbones"
            candidate = backbones / f"{source.stem}_0.pdb"
            if not candidate.is_file():
                legacy = backbones / f"{source.stem}_1.pdb"
                if legacy.is_file():
                    candidate = legacy
                else:
                    raise RuntimeError(
                        f"MPNN produced no first design: {candidate} (or legacy {legacy})"
                    )
            sampled = PDBParser(QUIET=True).get_structure("sample", str(candidate))
            new_residues = chain_residues(sampled, designed_chain)
            if [r.id for r in new_residues] != [r.id for r in residues]:
                raise ValueError(
                    "MPNN output changed the designed chain's residue IDs."
                )
            for i, (old, new) in enumerate(zip(residues, new_residues)):
                if i in fixed and old.resname != new.resname:
                    raise ValueError(
                        f"MPNN changed a fixed residue: {designed_chain}{old.id[1]}"
                    )
                if old.resname != new.resname:
                    old.resname = new.resname
                    # Old side chains are chemically invalid after mutation.
                    for atom in list(old):
                        if atom.name not in {"N", "CA", "C", "O", "OXT"}:
                            old.detach_child(atom.id)
            writer = PDBIO()
            writer.set_structure(structure)
            writer.save(str(design_path))
            # Retain upstream samples and FASTA for inspection / best-of-N work.
            samples = out / "samples"
            if samples.exists():
                shutil.rmtree(samples)
            shutil.copytree(temp_dir, samples)
    sequence = "".join(seq1(res.resname) for res in residues)
    (out / "mpnn_spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    result_path.write_text(
        json.dumps({"sequence": sequence, "structure_pdb": str(design_path)}, indent=2)
        + "\n"
    )
    return sequence, str(design_path)
