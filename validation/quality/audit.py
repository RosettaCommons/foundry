"""Audit benchmark completeness, identity and provenance before running the registered scorer."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from biotite.sequence import ProteinSequence
from biotite.structure.io import pdbx


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(root, plan, cases):
    """Persist all planned outputs, including missing cases, before any assertion."""
    records = []
    for arm in plan["arms"]:
        for case in cases:
            paths = [
                root / arm / f"{case}{suffix}"
                for suffix in (".json", ".pdb", "_ca.npy")
            ]
            records.append(
                {
                    "stage": "generate",
                    "arm": arm,
                    "case": case,
                    "missing": [
                        str(p.relative_to(root)) for p in paths if not p.is_file()
                    ],
                }
            )
            for draw in range(plan["mpnn_sequences_per_backbone"]):
                name = f"{arm}_{case}_d{draw}"
                paths = [
                    root / "refolds" / name / f"{name}{suffix}"
                    for suffix in ("_model.cif", "_summary_confidences.json")
                ]
                records.append(
                    {
                        "stage": "refold",
                        "arm": arm,
                        "case": case,
                        "draw": draw,
                        "missing": [
                            str(p.relative_to(root)) for p in paths if not p.is_file()
                        ],
                    }
                )
    result = {
        "records": records,
        "incomplete_records": sum(bool(r["missing"]) for r in records),
    }
    (root / "quality_inventory.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def audit(root):
    study = Path(__file__).resolve().parent
    plan = json.loads((study / "plan.json").read_text())
    provenance = json.loads((study / "provenance.json").read_text())
    hashes = {name: digest(study / name) for name in provenance["scripts"]}
    assert hashes == provenance["scripts"], "Registered script bytes changed"
    cases = {
        f"L{length}_s{seed}": (length, seed)
        for length in plan["lengths"]
        for seed in plan["seeds"]
    }
    completeness = inventory(root, plan, cases)
    assert (
        completeness["incomplete_records"] == 0
    ), "Incomplete pilot; see quality_inventory.json"
    expected = {
        f"{arm}_{case}_d{draw}"
        for arm in plan["arms"]
        for case in cases
        for draw in range(plan["mpnn_sequences_per_backbone"])
    }
    sequences = json.loads((root / "sequences.json").read_text())
    assert len(sequences) == len(expected)
    assert {item["name"] for item in sequences} == expected
    entries = []
    for item in sequences:
        length, seed = cases[item["case"]]
        assert item["seed"] == seed and len(item["sequence"]) == length
        assert item["components"] == [
            {"chain_id": "A", "seq": item["sequence"], "chain_type": "polypeptide(l)"}
        ]
        path = root / "refolds" / item["name"] / (item["name"] + "_model.cif")
        summary = path.with_name(item["name"] + "_summary_confidences.json")
        assert path.is_file() and summary.is_file(), item["name"]
        atoms = pdbx.get_structure(pdbx.CIFFile.read(path), model=1)
        ca = atoms[atoms.atom_name == "CA"]
        assert set(ca.chain_id) == {"A"}, item["name"]
        assert np.isfinite(atoms.coord).all(), item["name"]
        actual = "".join(
            ProteinSequence.convert_letter_3to1(name) for name in ca.res_name
        )
        assert actual == item["sequence"], item["name"]
        assert len(set(zip(ca.chain_id, ca.res_id, ca.ins_code))) == length
        values = json.loads(summary.read_text())
        assert all(np.isfinite(values[key]) for key in ("overall_plddt", "ptm"))
        assert 0 <= values["overall_plddt"] <= 1 and 0 <= values["ptm"] <= 1
        entries.append(
            {
                "name": item["name"],
                "model_sha256": digest(path),
                "summary_sha256": digest(summary),
            }
        )
    for arm in plan["arms"]:
        for case, (length, seed) in cases.items():
            metadata = json.loads((root / arm / f"{case}.json").read_text())
            assert (
                metadata["arm"],
                metadata["case"],
                metadata["length"],
                metadata["seed"],
            ) == (arm, case, length, seed)
            reference = np.load(root / arm / f"{case}_ca.npy")
            assert reference.shape == (length, 3) and np.isfinite(reference).all()
    changes = {
        case: json.loads(
            (root / "padded_aw3" / f"{case}_feature_changes.json").read_text()
        )
        for case in cases
    }
    assert all(value == [] for value in changes.values()), changes
    inputs = {case: digest(root / "matched_inputs" / f"{case}.pt") for case in cases}
    result = {
        "passed": True,
        "n_cases_per_arm": len(cases),
        "n_refolds": len(entries),
        "registered_script_hashes": hashes,
        "matched_input_sha256": inputs,
        "aw3_feature_changes": changes,
        "refold_identity_and_finite_checks": entries,
    }
    (root / "quality_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = audit(args.root.resolve())
    print(
        json.dumps(
            {key: result[key] for key in ("passed", "n_cases_per_arm", "n_refolds")}
        )
    )
