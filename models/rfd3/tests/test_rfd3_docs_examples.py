"""The ligand-bearing example specifications shipped in ``models/rfd3/docs/examples``
must build.

Since #261, ``DesignInputSpecification`` rejects a ligand that shares a chain with the
built structure, or several ligand residues on one chain, unless
``allow_ligand_on_existing_chain`` is set. The docs tell users to run these files
verbatim (e.g. ``rfd3 design inputs=./enzyme_design.json``), so they must keep
building when the input validation changes.
"""

import json
from pathlib import Path

import pytest
from rfd3.inference.input_parsing import DesignInputSpecification

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "docs" / "examples"


def _ligand_examples():
    for json_path in sorted(EXAMPLES_DIR.glob("*.json")):
        with open(json_path) as f:
            specs = json.load(f)
        for name, spec in specs.items():
            if spec.get("ligand"):
                yield pytest.param(json_path, name, id=f"{json_path.name}::{name}")


@pytest.mark.parametrize("json_path, name", list(_ligand_examples()))
def test_docs_ligand_example_builds(json_path, name, monkeypatch):
    with open(json_path) as f:
        spec = json.load(f)[name]
    # Example inputs are relative to the JSON file, as when running the documented
    # command from the examples directory.
    monkeypatch.chdir(json_path.parent)
    atom_array, _ = DesignInputSpecification(**spec).build(return_metadata=True)
    assert atom_array.array_length() > 0
