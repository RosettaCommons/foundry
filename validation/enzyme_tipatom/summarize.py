"""Summarize every planned design without filtering on its insertion score."""

import csv
import json
from pathlib import Path

import numpy as np
from geometry import symmetry_aware_tip_rmsd


def main():
    here = Path(__file__).resolve().parent
    root = here / "results"
    plan = json.loads((here / "plan.json").read_text())
    rows, missing = [], []
    for arm in plan["arms"]:
        for seed in plan["seeds"]:
            for sample in range(plan["diffusion_batch_size"]):
                path = root / arm / f"s{seed}_d{sample}" / "scores.json"
                if not path.exists():
                    missing.append(str(path))
                    continue
                result = json.loads(path.read_text())
                numeric = [result["tip_atom_rmsd"], result["insertion_rmsd"]]
                assert np.isfinite(numeric).all()
                assert result["n_tip_atoms"] == 15 and result["protein_length"] == 150
                assert (
                    abs(
                        result["insertion_rmsd"]
                        - result["engine_metrics"]["insertion_rmsd"]
                    )
                    < 1e-6
                )
                row = {
                    key: result[key]
                    for key in [
                        "arm",
                        "seed",
                        "sample",
                        "tip_atom_rmsd",
                        "mean_residue_tip_rmsd",
                        "insertion_rmsd",
                    ]
                }
                row["tip_below_0.5"] = result["tip_atom_rmsd"] < 0.5
                symmetry_rmsd, swaps = symmetry_aware_tip_rmsd(result["replacements"])
                row["symmetry_aware_tip_rmsd"] = symmetry_rmsd
                row["asp_oxygen_swaps"] = ";".join(swaps)
                for residue in result["by_residue"]:
                    row[residue["source"] + "_tip_rmsd"] = residue["rmsd"]
                rows.append(row)
    (root / "coverage.json").write_text(
        json.dumps({"expected": 32, "found": len(rows), "missing": missing}, indent=2)
    )
    if rows:
        with (root / "scores.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if missing:
        raise ValueError(f"Incomplete experiment: {len(missing)} missing outputs")
    summary = {"arms": {}, "contrasts": {}, "plan": plan}
    for arm in plan["arms"]:
        scores = [row for row in rows if row["arm"] == arm]
        tips = np.array([row["tip_atom_rmsd"] for row in scores])
        native = np.array([row["insertion_rmsd"] for row in scores])
        symmetric = np.array([row["symmetry_aware_tip_rmsd"] for row in scores])
        summary["arms"][arm] = {
            "n": len(scores),
            "tip_mean": float(tips.mean()),
            "tip_median": float(np.median(tips)),
            "tip_min": float(tips.min()),
            "tip_max": float(tips.max()),
            "tip_below_0.5": int((tips < 0.5).sum()),
            "builtin_mean": float(native.mean()),
            "symmetry_aware_tip_mean": float(symmetric.mean()),
            "symmetry_aware_tip_below_0.5": int((symmetric < 0.5).sum()),
        }
    for old, new in zip(plan["arms"][:-1], plan["arms"][1:]):
        seed_deltas = []
        for seed in plan["seeds"]:
            old_mean = np.mean(
                [
                    r["tip_atom_rmsd"]
                    for r in rows
                    if r["arm"] == old and r["seed"] == seed
                ]
            )
            new_mean = np.mean(
                [
                    r["tip_atom_rmsd"]
                    for r in rows
                    if r["arm"] == new and r["seed"] == seed
                ]
            )
            seed_deltas.append(float(new_mean - old_mean))
        summary["contrasts"][new + "_minus_" + old] = {
            "seed_mean_deltas": seed_deltas,
            "mean_delta": float(np.mean(seed_deltas)),
            "interpretation": "Four seed groups only; descriptive, not a significance or noninferiority test.",
        }
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["arms"], indent=2))


if __name__ == "__main__":
    main()
