"""RFO template selection uses the AtomWorks 3 query API."""

import importlib

import numpy as np
import pytest
from biotite.structure import AtomArray


@pytest.mark.parametrize("module", ["af3", "af3_auto_detect_breaking"])
@pytest.mark.parametrize(
    "selection, expected",
    [
        (None, [False, False, False]),
        ("A", [True, True, False]),
        ("A/*/2", [False, True, False]),
    ],
)
def test_template_selection(module, selection, expected):
    tool = importlib.import_module(f"rfo.backprop.tools.{module}")
    atoms = AtomArray(3)
    atoms.chain_id = ["A", "A", "B"]
    atoms.res_id = [1, 2, 1]
    atoms.res_name = ["ALA"] * 3
    atoms.atom_name = ["CA"] * 3
    atoms.coord = np.zeros((3, 3), dtype=np.float32)
    atoms.set_annotation("is_polymer", np.ones(3, dtype=bool))
    tool.apply_template_selection(atoms, selection)
    np.testing.assert_array_equal(atoms.is_input_file_templated, expected)
