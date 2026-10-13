"""Run the public enzyme tutorial and capture insertion error before cleanup."""

import argparse
import hashlib
import importlib
import importlib.util
import json
import random
import re
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


def measure_insertion(atoms, metadata, tip_definitions):
    """Score physical tip names using the model's assigned residue/slot mapping."""
    guide = atoms[atoms.is_motif_atom_unindexed]
    scaffold = atoms[~atoms.is_motif_atom_unindexed]
    replacements, rows = [], []
    for component, destination in metadata["diffused_index_map"].items():
        token = guide[guide.src_component == component]
        match = re.fullmatch(r"(.+?)(-?\d+)", destination)
        if not match or len(token) == 0:
            raise ValueError(f"Invalid mapping: {component} -> {destination}")
        chain, resid = match.group(1), int(match.group(2))
        residue = scaffold[(scaffold.chain_id == chain) & (scaffold.res_id == resid)]
        restype = str(token.res_name[0])
        tips = tip_definitions.get(restype) or []
        errors = []
        for atom in token:
            # atom_name is an atom14 slot label here; gt_atom_name is chemical identity.
            name = str(atom.gt_atom_name)
            candidates = residue[residue.atom_name == atom.atom_name]
            if len(candidates) != 1:
                raise ValueError(f"Ambiguous/missing slot: {component}/{name}")
            coord = candidates.coord[0].astype(float)
            target = atom.coord.astype(float)
            if not np.isfinite([coord, target]).all():
                raise ValueError("Nonfinite insertion coordinates")
            replacements.append(
                {
                    "chain": chain,
                    "resid": resid,
                    "atom": name,
                    "coord": coord.tolist(),
                    "target": target.tolist(),
                    "source": component,
                    "restype": restype,
                    "tip": name in tips,
                }
            )
            if name in tips:
                errors.append(float(np.sum((coord - target) ** 2)))
        if errors:
            rows.append(
                {
                    "source": component,
                    "destination": destination,
                    "restype": restype,
                    "n_atoms": len(errors),
                    "rmsd": float(np.sqrt(np.mean(errors))),
                    "squared_errors": errors,
                }
            )
    errors = [v for row in rows for v in row["squared_errors"]]
    if not errors:
        raise ValueError("No constrained tip atoms found")
    return {
        "tip_atom_rmsd": float(np.sqrt(np.mean(errors))),
        "mean_residue_tip_rmsd": float(np.mean([r["rmsd"] for r in rows])),
        "n_tip_atoms": len(errors),
        "by_residue": rows,
        "replacements": replacements,
    }


def write_pdb(atoms, path):
    from biotite.structure.io.pdb import PDBFile

    atoms = atoms.copy()
    atoms.bonds = None
    atoms.chain_id = np.array([str(chain)[0] for chain in atoms.chain_id])
    atoms.set_annotation("occupancy", np.ones(len(atoms)))
    atoms.set_annotation("b_factor", np.zeros(len(atoms)))
    file = PDBFile()
    file.set_structure(atoms)
    file.write(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pooling-source", type=Path)
    parser.add_argument("--reference-mask-dir", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44])
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if args.pooling_source:
        spec = importlib.util.spec_from_file_location(
            "enzyme_pooling", args.pooling_source
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        import foundry.utils.torch as utils

        utils.scatter_mean = module.scatter_mean
    from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine
    from rfd3.trainer.trainer_utils import process_unindexed_outputs

    from foundry.constants import TIP_BY_RESTYPE

    trainer_module = importlib.import_module("rfd3.trainer.rfd3")
    records = []

    def capture(atoms, **kwargs):
        _, metadata = process_unindexed_outputs(
            atoms, insert_guideposts=False, verbose=True
        )
        metrics = measure_insertion(atoms, metadata, TIP_BY_RESTYPE)
        assert metrics["n_tip_atoms"] == 15, metrics
        expected = atoms[atoms.is_motif_atom_unindexed | atoms.is_ligand].copy()
        expected.atom_name = expected.gt_atom_name
        records.append((metrics, metadata, expected))
        return process_unindexed_outputs(atoms, **kwargs)

    config = RFD3InferenceConfig(
        ckpt_path=str(args.checkpoint),
        diffusion_batch_size=args.batch_size,
        compile_model=args.arm.startswith("padded"),
        seed=args.seeds[0],
        prevalidate_inputs=True,
        skip_existing=False,
    )
    if hasattr(config, "inference_num_workers"):
        config.inference_num_workers = 0
    engine = RFD3InferenceEngine(**config)
    engine.initialize()
    mask_audit = []
    if args.reference_mask_dir:
        original_forward = engine._model_forward_batch

        def restore_reference_mask(examples):
            saved = torch.load(
                args.reference_mask_dir / f"s{engine.seed}.pt", weights_only=False
            )
            for example in examples:
                changed = []
                for key, value in example["feats"].items():
                    other = saved["feats"][key]
                    if isinstance(value, torch.Tensor):
                        same = torch.equal(value.cpu(), other.cpu())
                    else:
                        same = np.array_equal(value, other)
                    if not same:
                        changed.append(key)
                if changed != ["ref_mask"]:
                    raise ValueError(
                        f"Diagnostic expects only ref_mask to differ: {changed}"
                    )
                example["feats"]["ref_mask"] = saved["feats"]["ref_mask"].clone()
                mask_audit.append({"seed": engine.seed, "changed_features": changed})
            return original_forward(examples)

        engine._model_forward_batch = restore_reference_mask
    args.root.mkdir(parents=True, exist_ok=True)
    source_root = Path(trainer_module.__file__).resolve().parents[5]
    provenance = {
        "arm": args.arm,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "source": str(source_root),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
        ).strip(),
        "pooling_sha256": hashlib.sha256(
            (
                args.pooling_source or source_root / "src/foundry/utils/torch.py"
            ).read_bytes()
        ).hexdigest(),
        "config": {
            "compile_model": config.compile_model,
            "batch_size": args.batch_size,
            "sampler_overrides": {},
            "kernel": getattr(config, "inference_kernel_backend", "legacy"),
        },
    }
    (args.root / "provenance.json").write_text(json.dumps(provenance, indent=2))
    for seed in args.seeds:
        records.clear()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        engine.seed = seed
        engine.trainer.seed = seed
        with patch.object(trainer_module, "process_unindexed_outputs", capture):
            results = engine.run(inputs=str(args.input), n_batches=1, out_dir=None)
        outputs = [output for group in results.values() for output in group]
        assert len(outputs) == len(records) == args.batch_size
        for index, (output, (metrics, metadata, expected)) in enumerate(
            zip(outputs, records)
        ):
            stem = f"s{seed}_d{index}"
            directory = args.root / stem
            directory.mkdir(exist_ok=True)
            output.dump(directory)
            raw = output.atom_array.copy()
            for replacement in metrics["replacements"]:
                mask = (
                    (raw.chain_id == replacement["chain"])
                    & (raw.res_id == replacement["resid"])
                    & (raw.atom_name == replacement["atom"])
                )
                if mask.sum() != 1:
                    raise ValueError(f"Missing final output atom: {replacement}")
                raw.coord[mask] = replacement["coord"]
            write_pdb(raw, directory / "preinsert.pdb")
            write_pdb(expected, directory / "guideposts.pdb")
            write_pdb(output.atom_array, directory / "cleaned.pdb")
            metrics.update(
                {
                    "arm": args.arm,
                    "seed": seed,
                    "sample": index,
                    "insertion_rmsd": metadata["insertion_rmsd"],
                    "insertion_by_residue": metadata["insertion_rmsd_by_token"],
                    "diffused_index_map": metadata["diffused_index_map"],
                    "engine_metrics": output.metadata["metrics"],
                    "protein_length": int(
                        np.sum((raw.atom_name == "CA") & raw.is_protein)
                    ),
                }
            )
            (directory / "scores.json").write_text(json.dumps(metrics, indent=2))
            print(
                json.dumps(
                    {
                        key: metrics[key]
                        for key in [
                            "arm",
                            "seed",
                            "sample",
                            "protein_length",
                            "insertion_rmsd",
                            "tip_atom_rmsd",
                        ]
                    }
                ),
                flush=True,
            )
    (args.root / "COMPLETE").write_text("complete\n")
    if mask_audit:
        (args.root / "mask_audit.json").write_text(json.dumps(mask_audit, indent=2))


if __name__ == "__main__":
    main()
