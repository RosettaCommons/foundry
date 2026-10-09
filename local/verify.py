#!/usr/bin/env python3
"""Run real checkpoint inference on one device, retaining logs and verified outputs."""

import argparse
import gzip
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from biotite.structure import get_residue_count
from biotite.structure.io import pdbx

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--device", choices=["cpu", "mps"], required=True)
p.add_argument(
    "--quick",
    action="store_true",
    help="10 diffusion steps and 1 RF3 recycle; execution test only",
)
args = p.parse_args()
if args.device == "mps" and not torch.backends.mps.is_available():
    raise RuntimeError(
        "MPS is unavailable in this process; run in a normal macOS terminal or with GPU access."
    )
out = ROOT / "local" / "results" / (args.device + ("-quick" if args.quick else "-full"))
if out.exists():
    out = out / time.strftime("%Y%m%d-%H%M%S")
out.mkdir(parents=True, exist_ok=False)
env = dict(
    os.environ, FOUNDRY_DEVICE=args.device, OMP_NUM_THREADS="4", PYTHONUNBUFFERED="1"
)
report = {
    "device": args.device,
    "torch": torch.__version__,
    "quick": args.quick,
    "runs": {},
}


def run(model, options):
    log = out / f"{model}.log"
    command = [str(ROOT / ".venv" / "bin" / model), *options]
    print(f"Running {model} on {args.device}; log: {log}", flush=True)
    start = time.monotonic()
    with log.open("w") as f:
        result = subprocess.run(
            command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=1800
        )
    elapsed = round(time.monotonic() - start, 2)
    if result.returncode:
        raise RuntimeError(f"{model} failed; see {log}\n{log.read_text()[-8000:]}")
    report["runs"][model] = {"seconds": elapsed, "command": command, "log": str(log)}
    print(f"{model} exited successfully after {elapsed}s", flush=True)


def validate_cif(path, expected_residues, allow_missing_oxt=False):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        structure = pdbx.get_structure(
            pdbx.CIFFile.read(f), model=1, extra_fields=["occupancy"]
        )
    missing = ~np.isfinite(structure.coord).all(axis=1)
    expected_missing = (structure.atom_name == "OXT") & (structure.occupancy == 0)
    if missing.any():
        if not allow_missing_oxt or not np.all(expected_missing[missing]):
            raise ValueError(f"Unexpected non-finite coordinates in {path}")
        print(
            f"NOTE: {path} has {missing.sum()} unplaced OXT atom(s), occupancy=0; retained in output.",
            flush=True,
        )
    if get_residue_count(structure) != expected_residues:
        raise ValueError(f"Unexpected residue count in {path}")
    return {
        "file": str(path),
        "atoms": len(structure),
        "residues": expected_residues,
        "finite_coordinates": bool(not missing.any()),
        "unplaced_terminal_oxygen_atoms": int(missing.sum()),
    }


run(
    "rfd3",
    [
        "design",
        "inputs=null",
        "+specification.length=32",
        f'out_dir={out / "rfd3"}',
        "diffusion_batch_size=1",
        "n_batches=1",
        "skip_existing=false",
        f"inference_sampler.num_timesteps={10 if args.quick else 200}",
        "seed=42",
    ],
)
backbone = out / "rfd3" / "_0_model_0.cif.gz"
report["runs"]["rfd3"]["validation"] = validate_cif(backbone, 32)
run(
    "mpnn",
    [
        "--model_type",
        "protein_mpnn",
        "--is_legacy_weights",
        "True",
        "--checkpoint_path",
        str(ROOT / "local/checkpoints/proteinmpnn_v_48_020.pt"),
        "--structure_path",
        str(backbone),
        "--out_directory",
        str(out / "mpnn"),
        "--seed",
        "42",
    ],
)
fasta = next((out / "mpnn").glob("*.fa"))
seq = "".join(
    line.strip() for line in fasta.read_text().splitlines() if not line.startswith(">")
)
if len(seq) != 32 or not set(seq) <= set("ACDEFGHIKLMNPQRSTVWY"):
    raise ValueError(f"Invalid ProteinMPNN sequence: {seq!r}")
report["runs"]["mpnn"]["sequence"] = seq
report["runs"]["mpnn"]["validation"] = validate_cif(
    next((out / "mpnn").glob("*.cif")), 32, allow_missing_oxt=True
)
run(
    "rf3",
    [
        "fold",
        f'inputs={ROOT / "local/inputs/rf3.json"}',
        f'out_dir={out / "rf3"}',
        f"n_recycles={1 if args.quick else 10}",
        f"num_steps={10 if args.quick else 50}",
        "diffusion_batch_size=1",
        "seed=42",
        "early_stopping_plddt_threshold=0.0",
    ],
)
rf3_out = out / "rf3" / "local_peptide"
report["runs"]["rf3"]["validation"] = validate_cif(
    rf3_out / "local_peptide_model.cif", 32
)
summary = json.loads((rf3_out / "local_peptide_summary_confidences.json").read_text())
plddt = summary["overall_plddt"]
if not np.isfinite(plddt) or not 0 <= plddt <= 1:
    raise ValueError(f"Invalid RF3 pLDDT: {plddt}")
report["runs"]["rf3"]["overall_plddt"] = plddt
report_path = out / "verification.json"
report_path.write_text(json.dumps(report, indent=2) + "\n")
print(
    f"PASS: all three models completed and outputs validated. {report_path}", flush=True
)
