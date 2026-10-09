"""Freeze tracked runtime source, including local edits, without host secrets/caches."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=root
    ).decode().split("\0")
    files = {}
    for name in tracked:
        if not (
            name.startswith(("src/", "models/"))
            or name in {"pyproject.toml", "README.md", "LICENSE.md", ".project-root"}
        ):
            continue
        source = root / name
        if not source.is_file():
            continue
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        files[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest = {
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "description": "Working-tree content of tracked runtime files, including staged local changes",
        "diff_sha256": hashlib.sha256(subprocess.check_output(
            ["git", "diff", "HEAD", "--", "src", "models", "pyproject.toml"], cwd=root
        )).hexdigest(),
        "files_sha256": files,
        "excluded": [".git", ".env", "weights", "outs", "compile caches"],
    }
    (destination / "IMAGE_SOURCE.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Snapshot: {len(files)} files at {destination}")


if __name__ == "__main__":
    main()
