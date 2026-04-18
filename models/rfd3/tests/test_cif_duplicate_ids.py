"""Regression tests for duplicate CIF atom_id (#148).

PadTokensWithVirtualAtoms copies the central atom (CB) including its
atom_id annotation.  After virtual atoms are removed, sidechain atoms
retain CB's duplicated atom_id.  The CIF writer uses these values for
_atom_site.id, producing duplicate IDs that violate the mmCIF spec.
"""

import tempfile
from pathlib import Path

import numpy as np
from atomworks.io.utils.io_utils import to_cif_file
from biotite.structure import AtomArray
from rfd3.engine import RFD3Output
from rfd3.trainer.trainer_utils import (
    _cleanup_virtual_atoms_and_assign_atom_name_elements,
)


def _make_alanine_array_with_duplicate_atom_ids(n_residues=4):
    """Build a protein AtomArray where every atom in each residue carries
    the same atom_id as CB — the corruption caused by virtual-atom padding."""
    names = ["N", "CA", "C", "O", "CB"]
    n_atoms = n_residues * len(names)
    atoms = AtomArray(n_atoms)

    for i in range(n_residues):
        for j, name in enumerate(names):
            idx = i * len(names) + j
            atoms.chain_id[idx] = "A"
            atoms.res_id[idx] = i + 1
            atoms.res_name[idx] = "ALA"
            atoms.atom_name[idx] = name
            atoms.element[idx] = name[0]
            atoms.coord[idx] = [float(i * 4), float(j * 1.5), 0.0]

    # Simulate the bug: every atom in a residue shares CB's atom_id
    atom_ids = np.array(
        [i * len(names) + 4 for i in range(n_residues) for _ in range(len(names))]
    )
    atoms.set_annotation("atom_id", atom_ids)

    # Annotations required by _cleanup_virtual_atoms_and_assign_atom_name_elements
    atoms.set_annotation("is_motif_atom_with_fixed_seq", np.ones(n_atoms, dtype=bool))
    atoms.set_annotation("is_motif_atom_unindexed", np.zeros(n_atoms, dtype=bool))
    atoms.set_annotation("gt_atom_name", atoms.atom_name.copy())

    return atoms


def test_cleanup_strips_atom_id():
    """_cleanup_virtual_atoms_and_assign_atom_name_elements must remove
    the atom_id annotation so the CIF writer generates fresh IDs."""
    atoms = _make_alanine_array_with_duplicate_atom_ids()

    # Precondition: atom_id exists and has duplicates
    assert "atom_id" in atoms.get_annotation_categories()
    assert len(set(atoms.atom_id)) < len(atoms.atom_id)

    result = _cleanup_virtual_atoms_and_assign_atom_name_elements(atoms)
    assert "atom_id" not in result.get_annotation_categories()


def test_cif_output_has_unique_ids():
    """CIF _atom_site.id values must be unique after _strip_atom_id."""
    atoms = _make_alanine_array_with_duplicate_atom_ids()
    atoms = RFD3Output._strip_atom_id(atoms)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test"
        to_cif_file(atoms, out_path, file_type="cif")

        cif_text = Path(f"{out_path}.cif").read_text()

        # Parse _atom_site loop to find the id column
        lines = cif_text.splitlines()
        col_names = []
        data_start = None
        for i, line in enumerate(lines):
            if line.strip().startswith("_atom_site."):
                col_names.append(line.strip().split()[0])
            elif col_names and not line.strip().startswith("_") and line.strip():
                data_start = i
                break

        assert "_atom_site.id" in col_names, "missing _atom_site.id column"
        id_col = col_names.index("_atom_site.id")

        ids = []
        for line in lines[data_start:]:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("loop_"):
                break
            parts = stripped.split()
            if len(parts) > id_col:
                ids.append(parts[id_col])

        assert len(ids) > 0, "no atom records found in CIF output"
        assert len(ids) == len(set(ids)), f"duplicate _atom_site.id values: {ids}"
