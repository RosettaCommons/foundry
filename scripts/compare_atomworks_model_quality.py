"""Summarize paired artifacts against preregistered thresholds."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from biotite.structure import rmsd, superimpose


def tensors(value, prefix=""):
    if isinstance(value, torch.Tensor):
        yield prefix, value.detach().cpu()
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from tensors(v, prefix + "/" + str(k))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            yield from tensors(v, prefix + "/" + str(i))


def compare_pt(a, b, network_only=False):
    aa = torch.load(a, weights_only=False, map_location="cpu")
    bb = torch.load(b, weights_only=False, map_location="cpu")
    if network_only:
        aa = aa["network_output"]
        bb = bb["network_output"]
    x = dict(tensors(aa))
    y = dict(tensors(bb))
    bad = [] if x else [dict(reason="no tensors")]
    exact = 0
    maximum = 0.0
    for key in sorted(x.keys() | y.keys()):
        if key not in x or key not in y:
            bad.append(dict(key=key, reason="missing"))
            continue
        v, w = x[key], y[key]
        if v.dtype != w.dtype:
            bad.append(dict(key=key, reason="dtype"))
            continue
        if v.shape != w.shape:
            bad.append(dict(key=key, reason="shape"))
            continue
        exact += int(torch.equal(v, w))
        if v.is_floating_point():
            if not (torch.isfinite(v).all() and torch.isfinite(w).all()):
                bad.append(dict(key=key, reason="nonfinite"))
            delta = (v.float() - w.float()).abs()
            finite = delta[torch.isfinite(delta)]
            d = float(finite.max()) if finite.numel() else 0
            maximum = max(maximum, d)
            if not torch.allclose(v, w, atol=1e-5, rtol=1e-5, equal_nan=True):
                bad.append(dict(key=key, max_abs=d))
        elif not torch.equal(v, w):
            bad.append(
                dict(key=key, reason="discrete", n_different=int((v != w).sum()))
            )
    return dict(
        tensors=len(x),
        bitwise_equal=exact,
        max_abs=maximum,
        pass_threshold=not bad,
        failures=bad,
    )


def fixed_residues_preserved(input_path, output_path):
    features = torch.load(input_path, weights_only=False, map_location="cpu")["input"][
        "input_features"
    ]
    sampled = torch.load(output_path, weights_only=False, map_location="cpu")[
        "decoder_features"
    ]["S_sampled"]
    fixed = ~features["designed_residue_mask"].bool().expand_as(sampled)
    return bool(
        fixed.any()
        and torch.equal(sampled[fixed], features["S"].expand_as(sampled)[fixed])
    )


def quality_gates_pass(summary):
    return all(
        summary.get(key) is True
        for key in ("native_quality_gate", "rf3_fixed_all_pass", "mpnn_all_pass")
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--rf3-pairs", type=int, default=18)
    p.add_argument(
        "--require-pass",
        action="store_true",
        help="Exit nonzero if numerical or quality gates fail; seeded structural equivalence is reported separately.",
    )
    args = p.parse_args()
    root = args.root
    out = dict(rf3_native=[], rf3_fixed_features=[], mpnn=[])
    for old in sorted((root / "aw2-rf3").glob("*/seed*/metrics.json")):
        rel = old.relative_to(root / "aw2-rf3")
        new = root / "aw3-rf3" / rel
        if not new.exists():
            continue
        a = json.loads(old.read_text())
        b = json.loads(new.read_text())
        x = torch.load(old.with_name("atoms.pt"), weights_only=False)
        y = torch.load(new.with_name("atoms.pt"), weights_only=False)
        ca = x.atom_name == "CA"
        cb = y.atom_name == "CA"
        v = x.coord[ca]
        w = y.coord[cb]
        aligned, _ = superimpose(v, w)
        ca_rmsd = float(rmsd(v, aligned))
        dx = float(b["summary"]["overall_plddt"]) - float(a["summary"]["overall_plddt"])
        ga, gb = a["geometry"], b["geometry"]
        out["rf3_native"].append(
            dict(
                case=rel.parts[0],
                seed=rel.parts[1],
                aw2=a,
                aw3=b,
                ca_rmsd=ca_rmsd,
                plddt_delta=dx,
                finite=gb["finite"],
                same_atoms=ga["n_atoms"] == gb["n_atoms"]
                and all(
                    np.array_equal(x.get_annotation(key), y.get_annotation(key))
                    for key in ("chain_id", "res_id", "res_name", "atom_name")
                ),
                geometry_outlier_change=gb["covalent_outlier_fraction"]
                - ga["covalent_outlier_fraction"],
                severe_clash_change=gb["severe_nonbonded_clashes"]
                - ga["severe_nonbonded_clashes"],
            )
        )
        fixed = root / "aw3-rf3-replay" / rel.with_name("network.pt")
        if fixed.exists():
            out["rf3_fixed_features"].append(
                dict(
                    case=rel.parts[0],
                    seed=rel.parts[1],
                    **compare_pt(old.with_name("network.pt"), fixed, True),
                )
            )
    for old in sorted((root / "aw2-mpnn").glob("*/*/seed*/network.pt")):
        rel = old.relative_to(root / "aw2-mpnn")
        new = root / "aw3-mpnn" / rel
        if not new.exists():
            continue
        output = compare_pt(old, new)
        inputs = compare_pt(old.with_name("input.pt"), new.with_name("input.pt"))
        # RNG state arrays are captured too; report input feature tensors separately below.
        a = json.loads(old.with_name("metrics.json").read_text())
        b = json.loads(new.with_name("metrics.json").read_text())
        out["mpnn"].append(
            dict(
                model=rel.parts[0],
                case=rel.parts[1],
                seed=rel.parts[2],
                inputs=inputs,
                outputs=output,
                fixed_residues_preserved=all(
                    fixed_residues_preserved(path.with_name("input.pt"), path)
                    for path in (old, new)
                ),
                aw2=a,
                aw3=b,
            )
        )
    native = out["rf3_native"]
    diffs = [row["plddt_delta"] for row in native]
    case_confidence = {
        case: float(np.mean([r["plddt_delta"] for r in native if r["case"] == case]))
        for case in {r["case"] for r in native}
    }
    native_complete = len(native) == args.rf3_pairs
    native_quality = (
        native_complete
        and np.median(diffs) >= -0.05
        and min(case_confidence.values()) >= -0.10
        and all(
            r["finite"]
            and r["same_atoms"]
            and r["geometry_outlier_change"] <= 0.10
            and r["severe_clash_change"] <= 0
            and r["aw3"]["geometry"].get("template_ca_rmsd", 0) <= 1.0
            for r in native
        )
    )
    out["summary"] = dict(
        native_quality_gate=bool(native_quality),
        case_mean_plddt_deltas=case_confidence,
        native_structural_equivalence=bool(
            native_complete and all(r["ca_rmsd"] <= 2.0 for r in native)
        ),
        rf3_native_pairs=len(native),
        rf3_fixed_pairs=len(out["rf3_fixed_features"]),
        mpnn_pairs=len(out["mpnn"]),
        median_plddt_delta=float(np.median(diffs)) if diffs else None,
        rf3_fixed_all_pass=len(out["rf3_fixed_features"]) == args.rf3_pairs
        and all(x["pass_threshold"] for x in out["rf3_fixed_features"]),
        mpnn_all_pass=len(out["mpnn"]) == 36
        and all(
            x["outputs"]["pass_threshold"] and x["fixed_residues_preserved"]
            for x in out["mpnn"]
        ),
    )
    (root / "comparison.json").write_text(json.dumps(out, indent=2, allow_nan=True))
    print(json.dumps(out["summary"], indent=2))
    for x in native:
        print(
            x["case"],
            x["seed"],
            "ca_rmsd",
            x["ca_rmsd"],
            "plddt_delta",
            x["plddt_delta"],
            "bond_outlier_change",
            x["geometry_outlier_change"],
            "clash_delta",
            x["severe_clash_change"],
        )

    if args.require_pass and not quality_gates_pass(out["summary"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
