"""Bitwise comparison of RFD3 designs across run directories.

Compares, for every design in a run directory, the atom coordinates and the residue
sequence read from the ``*.cif.gz`` output, plus the numeric metrics in the sibling
``*.json``. Coordinates are read from the decompressed CIF because gzip embeds a
timestamp, so the raw ``.cif.gz`` bytes differ even for identical structures.

    compare_runs.py equal  DIR_REF DIR [DIR ...]   exit 1 unless every DIR matches DIR_REF
    compare_runs.py differ DIR_REF DIR [DIR ...]   exit 1 if any DIR matches DIR_REF
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
from biotite.structure.io import pdbx


class Design(NamedTuple):
    coord: np.ndarray
    seq: tuple[str, ...]
    metrics: dict[str, float]


def _flatten_numbers(obj: object, prefix: str = "") -> dict[str, float]:
    """Flatten the numeric leaves of a nested JSON object into ``{dotted.key: value}``."""
    if isinstance(obj, bool):
        return {}
    if isinstance(obj, (int, float)):
        return {prefix: float(obj)}
    if isinstance(obj, dict):
        items = obj.items()
    elif isinstance(obj, list):
        items = enumerate(obj)
    else:
        return {}
    out: dict[str, float] = {}
    for key, value in items:
        out.update(_flatten_numbers(value, f"{prefix}.{key}" if prefix else str(key)))
    return out


def load_designs(run_dir: str) -> list[Design]:
    cifs = sorted(Path(run_dir).rglob("*.cif.gz"))
    if not cifs:
        raise SystemExit(f"no *.cif.gz under {run_dir}")
    designs = []
    for cif in cifs:
        with gzip.open(cif, "rt") as fh:
            atoms = pdbx.get_structure(pdbx.CIFFile.read(fh), model=1)
        seq = tuple(
            dict.fromkeys(
                f"{c}{i}{n}"
                for c, i, n in zip(atoms.chain_id, atoms.res_id, atoms.res_name)
            )
        )
        json_path = cif.with_name(cif.name.removesuffix(".cif.gz") + ".json")
        metrics = (
            _flatten_numbers(json.loads(json_path.read_text()))
            if json_path.exists()
            else {}
        )
        designs.append(Design(atoms.coord, seq, metrics))
    return designs


def diff(a: Design, b: Design) -> tuple[bool, str]:
    seq_equal = a.seq == b.seq
    metrics_equal = a.metrics == b.metrics
    if a.coord.shape != b.coord.shape:
        return False, (
            f"coord shape {a.coord.shape} vs {b.coord.shape} "
            f"seq_equal={seq_equal} metrics_equal={metrics_equal}"
        )
    coord_equal = bool(np.array_equal(a.coord, b.coord, equal_nan=True))
    max_dx = float(np.nanmax(np.abs(a.coord - b.coord)))
    return (
        coord_equal and seq_equal and metrics_equal,
        f"coord_equal={coord_equal} (max|dx|={max_dx:.3g}A) "
        f"seq_equal={seq_equal} metrics_equal={metrics_equal}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mode", choices=["equal", "differ"])
    parser.add_argument("ref_dir")
    parser.add_argument("dirs", nargs="+")
    args = parser.parse_args()

    ref = load_designs(args.ref_dir)
    ok = True
    for run_dir in args.dirs:
        other = load_designs(run_dir)
        if len(other) != len(ref):
            raise SystemExit(f"{run_dir}: {len(other)} designs, expected {len(ref)}")
        results = [diff(a, b) for a, b in zip(ref, other)]
        identical = all(same for same, _ in results)
        good = identical if args.mode == "equal" else not identical
        ok &= good
        print(
            f"  {'PASS' if good else 'FAIL'} [{args.mode}] "
            f"{Path(args.ref_dir).name} vs {Path(run_dir).name}: identical={identical}"
        )
        for idx, (_, info) in enumerate(results):
            print(f"       design {idx}: {info}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
