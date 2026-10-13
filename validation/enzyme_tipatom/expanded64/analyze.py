"""Read native RFD3 JSON scores and bootstrap paired seed groups."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE / "results"
LABELS = [
    "Original pooling, AW2",
    "Fixed pooling, AW2",
    "Fixed + #397, AW2",
    "Fixed + #397, AW3",
]


def describe(values, indices):
    per_seed = values.mean(axis=1)
    pass_per_seed = (values < 1).mean(axis=1)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(values.max()),
        "below_1": int((values < 1).sum()),
        "mean_ci95": np.quantile(
            per_seed[indices].mean(axis=1), [0.025, 0.975]
        ).tolist(),
        "pass_fraction_ci95": np.quantile(
            pass_per_seed[indices].mean(axis=1), [0.025, 0.975]
        ).tolist(),
    }


def main():
    plan = json.loads((HERE / "plan.json").read_text())
    # Verify that all retained artifacts are unchanged.
    for item in json.loads((HERE / "retained_manifest.json").read_text()):
        assert (
            hashlib.sha256((HERE / item["path"]).read_bytes()).hexdigest()
            == item["sha256"]
        )
    rows, matrices = [], {}
    for arm in plan["arms"]:
        assert (ROOT / arm / "COMPLETE").exists(), arm
        expected = {f"s{s}_d{d}" for s in plan["seeds"] for d in range(2)}
        assert {p.name for p in (ROOT / arm).glob("s*_d*")} == expected
        values = []
        for seed in plan["seeds"]:
            seed_values = []
            for sample in range(2):
                directory = ROOT / arm / f"s{seed}_d{sample}"
                files = [p for p in directory.glob("*.json") if p.name != "scores.json"]
                assert len(files) == 1, files
                score = json.loads(files[0].read_text())["metrics"]["insertion_rmsd"]
                assert np.isfinite(score) and score >= 0
                rows.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "sample": sample,
                        "insertion_rmsd": score,
                        "below_1": score < 1,
                        "retained": seed in plan["retained_seeds"],
                        "native_json": str(files[0].relative_to(HERE)),
                    }
                )
                seed_values.append(score)
            values.append(seed_values)
        matrices[arm] = np.array(values)
    rng = np.random.default_rng(20261013)
    indices = rng.integers(0, 32, size=(20000, 32))
    new_indices = rng.integers(0, 28, size=(20000, 28))
    summary = {
        "metric": plan["metric"],
        "threshold": 1.0,
        "arms": {},
        "contrasts": {},
        "coverage": {"expected": 256, "found": len(rows)},
        "bootstrap": plan["analysis"],
    }
    for arm, values in matrices.items():
        summary["arms"][arm] = {
            "all64": describe(values, indices),
            "new56": describe(values[4:], new_indices),
            "outliers": [r for r in rows if r["arm"] == arm and not r["below_1"]],
        }
    for old, new in zip(plan["arms"][:-1], plan["arms"][1:]):
        a, b = matrices[old], matrices[new]
        contrasts = {}
        for key, delta in {
            "mean_rmsd": (b - a).mean(axis=1),
            "pass_fraction": ((b < 1).astype(float) - (a < 1)).mean(axis=1),
        }.items():
            contrasts[key] = {
                "difference": float(delta.mean()),
                "ci95": np.quantile(
                    delta[indices].mean(axis=1), [0.025, 0.975]
                ).tolist(),
            }
        summary["contrasts"][new + "_minus_" + old] = contrasts
    evidence = HERE / "evidence"
    evidence.mkdir(exist_ok=True)
    with (evidence / "scores.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (evidence / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (evidence / "new_provenance.json").write_text(
        json.dumps(
            {
                arm: json.loads((ROOT / arm / "provenance.json").read_text())
                for arm in plan["arms"]
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
