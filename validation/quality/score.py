"""Score all planned designs, including failures, without treating siblings as replicates."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from biotite.structure.io import pdbx


def aligned_rmsd(x, y):
    assert x.shape == y.shape and x.ndim == 2 and x.shape[1] == 3
    assert np.isfinite(x).all() and np.isfinite(y).all()
    x = x.astype(np.float64) - x.mean(0)
    y = y.astype(np.float64) - y.mean(0)
    u, _, vt = np.linalg.svd(x.T @ y)
    if np.linalg.det(u @ vt) < 0:
        u[:, -1] *= -1
    return float(np.sqrt(np.mean(np.sum((x @ u @ vt - y) ** 2, axis=1))))


def interval(values, clusters=None):
    values = np.array(values, dtype=float)
    assert np.isfinite(values).all()
    if clusters is not None:
        # Same generation seeds recur across lengths. Treat their shared streams
        # as clusters as a conservative sensitivity analysis (only four clusters).
        groups = sorted(set(clusters))
        values = np.array([values[np.array(clusters) == key].mean() for key in groups])
    rng = np.random.default_rng(914)
    means = values[rng.integers(0, len(values), (10000, len(values)))].mean(1)
    return [float(np.quantile(means, 0.05)), float(np.quantile(means, 0.95))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    plan = json.loads((Path(__file__).parent / "plan.json").read_text())
    sequences = json.loads((root / "sequences.json").read_text())
    assert len(sequences) == len(plan["arms"]) * len(plan["lengths"]) * len(plan["seeds"]) * 2
    rows = []
    for item in sequences:
        name = item["name"]
        directory = root / "refolds" / name
        coords_file = directory / f"{name}_model.cif"
        summary_file = directory / f"{name}_summary_confidences.json"
        row = {key: item[key] for key in ("name", "arm", "case", "seed", "sequence")}
        row["length"] = len(item["sequence"])
        if not coords_file.exists() or not summary_file.exists():
            row.update({"failed": True, "pass": False, "rmsd": None, "plddt": None})
        else:
            atoms = pdbx.get_structure(pdbx.CIFFile.read(coords_file), model=1)
            ca = atoms.coord[atoms.atom_name == "CA"]
            assert ca.shape == (row["length"], 3), name
            assert np.isfinite(atoms.coord).all(), name
            summary = json.loads(summary_file.read_text())
            assert 0 <= summary["overall_plddt"] <= 1, name
            reference = np.load(root / item["arm"] / f'{item["case"]}_ca.npy')
            rmsd = aligned_rmsd(reference, ca)
            row.update({"failed": False, "rmsd": rmsd, "plddt": summary["overall_plddt"],
                "ptm": summary["ptm"], "has_clash": summary["has_clash"],
                "pass": rmsd < 2.0 and summary["overall_plddt"] >= 0.8})
        rows.append(row)
    assert not any(row["failed"] for row in rows), "Incomplete pilot: failures retained but no noninferiority inference allowed"
    arms = {}
    by_case = {}
    for arm in plan["arms"]:
        arm_rows = [r for r in rows if r["arm"] == arm]
        cases = {}
        for length in plan["lengths"]:
            for seed in plan["seeds"]:
                case = f"L{length}_s{seed}"
                siblings = [r for r in arm_rows if r["case"] == case]
                assert len(siblings) == 2
                cases[case] = {"success": float(any(r["pass"] for r in siblings)),
                    "best_rmsd": min(r["rmsd"] for r in siblings),
                    "best_plddt": max(r["plddt"] for r in siblings), "seed": seed, "length": length}
        by_case[arm] = cases
        arms[arm] = {"n_backbones": len(cases), "n_sequences": len(arm_rows),
            "backbones_pass": int(sum(r["success"] for r in cases.values())),
            "sequences_pass": sum(r["pass"] for r in arm_rows),
            "backbone_pass_fraction": float(np.mean([r["success"] for r in cases.values()])),
            "mean_best_rmsd": float(np.mean([r["best_rmsd"] for r in cases.values()])),
            "mean_best_plddt": float(np.mean([r["best_plddt"] for r in cases.values()]))}
    contrasts = {}
    for candidate, baseline in plan["contrasts"]:
        pair = {}
        keys = sorted(by_case[baseline])
        seeds = [by_case[baseline][k]["seed"] for k in keys]
        for metric, margin, direction in [("success", -0.1, "lower"), ("best_plddt", -0.03, "lower"), ("best_rmsd", 0.25, "upper")]:
            delta = [by_case[candidate][k][metric] - by_case[baseline][k][metric] for k in keys]
            ci = interval(delta)
            cluster_ci = interval(delta, seeds)
            pair[metric] = {"mean_delta": float(np.mean(delta)), "paired90ci": ci,
                "seed_cluster90ci": cluster_ci, "margin": margin,
                "exploratory_gate": (min(ci[0], cluster_ci[0]) >= margin if direction == "lower" else max(ci[1], cluster_ci[1]) <= margin)}
        pair["aligned_generated_ca_rmsds"] = [aligned_rmsd(
            np.load(root / candidate / f"{k}_ca.npy"), np.load(root / baseline / f"{k}_ca.npy")) for k in keys]
        pair["sequences_identical"] = sum(
            next(r["sequence"] for r in rows if r["arm"] == candidate and r["case"] == k and r["name"].endswith(f"_d{d}")) ==
            next(r["sequence"] for r in rows if r["arm"] == baseline and r["case"] == k and r["name"].endswith(f"_d{d}"))
            for k in keys for d in range(2))
        contrasts[f"{candidate}_minus_{baseline}"] = pair
    report = {"plan": plan, "arms": arms, "contrasts": contrasts, "per_backbone": by_case,
        "interpretation": "Exploratory pilot only. Bootstrap has limited coverage with 16 cases/four recurring seed clusters; does not establish population-wide equivalence or experimental function."}
    (root / "quality_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (root / "quality_sequences.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"arms": arms, "contrasts": contrasts}, indent=2))


if __name__ == "__main__":
    main()
