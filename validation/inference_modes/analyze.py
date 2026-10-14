"""Require complete checkpoint runs in both presets and read native JSON scores."""

import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    plan = json.loads((HERE / "plan.json").read_text())
    rows, arrays, summary = [], {}, {}
    for mode in plan["modes"]:
        root = HERE / "results" / mode
        assert (root / "COMPLETE").exists(), mode
        values = []
        for seed in plan["seeds"]:
            pair = []
            for sample in range(2):
                directory = root / f"s{seed}_d{sample}"
                paths = [p for p in directory.glob("*.json") if p.name != "scores.json"]
                assert len(paths) == 1
                data = json.loads(paths[0].read_text())
                settings = data["inference_settings"]
                assert settings["mode"] == mode
                assert settings["compile_model"] == (mode == "fast")
                assert settings["execution_path"] == (
                    "padded" if mode == "fast" else "legacy"
                )
                assert settings["pooling"] == (
                    "fp32_mean" if mode == "fast" else "legacy_index_reduce"
                )
                score = data["metrics"]["insertion_rmsd"]
                assert np.isfinite(score)
                rows.append(
                    {
                        "mode": mode,
                        "seed": seed,
                        "sample": sample,
                        "insertion_rmsd": score,
                        "below_1": score < 1,
                        "native_json": str(paths[0].relative_to(HERE)),
                    }
                )
                pair.append(score)
            values.append(pair)
        arrays[mode] = np.asarray(values)
        flat = arrays[mode].ravel()
        summary[mode] = {
            "n": len(flat),
            "mean": float(flat.mean()),
            "median": float(np.median(flat)),
            "max": float(flat.max()),
            "below_1": int((flat < 1).sum()),
            "settings": settings,
        }
    baseline = json.loads(
        (HERE.parent / "enzyme_tipatom/expanded64/evidence/summary.json").read_text()
    )
    summary["prior_fast_before_precision_fix"] = baseline["arms"]["padded_aw3"]["all64"]
    rng = np.random.default_rng(20261014)
    indices = rng.integers(0, len(plan["seeds"]), size=(20000, len(plan["seeds"])))
    contrasts = {}
    for key, a, b in [
        ("insertion_rmsd", arrays["fast"], arrays["legacy"]),
        (
            "pass_fraction",
            (arrays["fast"] < 1).astype(float),
            (arrays["legacy"] < 1).astype(float),
        ),
    ]:
        delta = (a - b).mean(axis=1)
        contrasts[key] = {
            "fast_minus_legacy": float(delta.mean()),
            "ci95": np.quantile(delta[indices].mean(axis=1), [0.025, 0.975]).tolist(),
        }
    summary["contrasts"] = contrasts
    evidence = HERE / "evidence"
    evidence.mkdir(exist_ok=True)
    with (evidence / "scores.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (evidence / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (evidence / "provenance.json").write_text(
        json.dumps(
            {
                mode: json.loads(
                    (HERE / "results" / mode / "provenance.json").read_text()
                )
                for mode in plan["modes"]
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
