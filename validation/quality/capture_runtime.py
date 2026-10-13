"""Capture local source equivalence, environment versions, and checkpoint hashes."""

import hashlib
import json
import os
import subprocess
from pathlib import Path


def sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def main():
    study = Path(__file__).resolve().parent
    sources = {}
    for name, variable, revision in [
        ("native_aw2", "NATIVE_SOURCE", "8bffbcb9bec7eb8c3be0813f87d33c98faa0f5d2"),
        ("padded_aw2", "PADDED_SOURCE", "754d1caae5a74b35a3bfcf68169e459474b7dbc4"),
        ("padded_aw3", "AW3_SOURCE", "d0474f0f467793d7dd7151399499470fe9475efc"),
    ]:
        source = Path(os.environ[variable])
        paths = git(
            source, "diff", "--name-only", revision, "--", "src", "models"
        ).splitlines()
        # The benchmark checkout can contain extra tests/docs, but production
        # source bytes must match the registered revisions.
        runtime_changes = [
            p for p in paths if "/src/" in f"/{p}" or p.startswith("src/")
        ]
        assert not runtime_changes, (name, runtime_changes)
        sources[name] = {
            "registered_revision": revision,
            "actual_head": git(source, "rev-parse", "HEAD"),
            "runtime_diff": runtime_changes,
            "nonruntime_model_changes": paths,
        }
    helper = Path(os.environ["POOL_SOURCE"])
    helper_repo = helper.parents[3]
    expected = subprocess.check_output(
        ["git", "-C", str(helper_repo), "show", "aa97ded:src/foundry/utils/torch.py"]
    )
    assert helper.read_bytes() == expected
    sources["pooling_helper"] = {
        "registered_revision": git(helper_repo, "rev-parse", "aa97ded"),
        "actual_head": git(helper_repo, "rev-parse", "HEAD"),
        "sha256": sha256(helper),
        "runtime_identical": True,
    }
    environment_script = """
import importlib.metadata as m, json, platform, torch
names = ['atomworks','biotite','numpy','scipy','pandas','torch','lightning','hydra-core','rdkit','triton','numba','cuequivariance-torch','cuequivariance-ops-torch-cu12','cuequivariance-ops-cu12']
versions = {}
for name in names:
    try: versions[name] = m.version(name)
    except m.PackageNotFoundError: versions[name] = None
print(json.dumps({'python': platform.python_version(), 'packages': versions, 'torch_cuda': torch.version.cuda}))
"""
    environments = {
        name: json.loads(
            subprocess.check_output(
                [os.environ[var], "-c", environment_script], text=True
            )
        )
        for name, var in [("aw2", "AW2_PY"), ("aw3", "AW3_PY")]
    }
    checkpoints = {}
    for name, variable in [
        ("rfd3", "RFD3_CKPT"),
        ("mpnn", "MPNN_CKPT"),
        ("rf3", "RF3_CKPT"),
    ]:
        path = Path(os.environ[variable])
        checkpoints[name] = {
            "filename": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    output = study / "evidence" / "runtime_provenance.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "sources": sources,
                "environments": environments,
                "checkpoints": checkpoints,
            },
            indent=2,
        )
        + "\n"
    )
    print(output)


if __name__ == "__main__":
    main()
