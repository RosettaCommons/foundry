from pathlib import Path

import numpy as np
from beartype import beartype
from jaxtyping import Bool, Shaped
from rfd3.inference.input_parsing import DesignInputSpecification

PDB_CONTENT = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 10.00           N  
ATOM      2  CA  ALA A   1       1.500   0.000   0.000  1.00 10.00           C  
ATOM      3  C   ALA A   1       2.000   1.500   0.000  1.00 10.00           C  
ATOM      4  O   ALA A   1       2.000   2.500   0.000  1.00 10.00           O  
HETATM    5  C1  LIG B   1       5.000   0.000   0.000  1.00 10.00           C  
HETATM    6  O1  LIG B   1       6.200   0.000   0.000  1.00 10.00           O  
HETATM    7  N1  LIG B   1       5.000   1.200   0.000  1.00 10.00           N  
HETATM    8  C2  LIG B   1       6.200   1.200   0.000  1.00 10.00           C  
TER
END
"""


@beartype
def _ligand_fixed_lookup(
    atom_names: Shaped[np.ndarray, "n"],
    fixed_mask: Bool[np.ndarray, "n"],
) -> dict[str, bool]:
    """Map ligand atom names to fixed-coordinate flags."""
    return {
        str(name): bool(is_fixed)
        for name, is_fixed in zip(atom_names.tolist(), fixed_mask.tolist())
    }


def test_partial_ligand_fixed_atoms_respected(tmp_path: Path) -> None:
    pdb_path = tmp_path / "ligand.pdb"
    pdb_path.write_text(PDB_CONTENT)

    spec = DesignInputSpecification.safe_init(
        input=pdb_path,
        length=1,
        ligand="LIG",
        select_fixed_atoms={"LIG": "C1,O1"},
    )
    atom_array = spec.build(return_metadata=False)

    ligand_mask = atom_array.res_name == "LIG"
    assert ligand_mask.any(), "Expected ligand atoms in output atom array."

    ligand_names = atom_array.atom_name[ligand_mask]
    fixed_mask = atom_array.is_motif_atom_with_fixed_coord[ligand_mask].astype(bool)
    fixed_lookup = _ligand_fixed_lookup(ligand_names, fixed_mask)

    assert set(fixed_lookup.keys()) == {"C1", "O1", "N1", "C2"}
    assert fixed_lookup["C1"]
    assert fixed_lookup["O1"]
    assert not fixed_lookup["N1"]
    assert not fixed_lookup["C2"]
