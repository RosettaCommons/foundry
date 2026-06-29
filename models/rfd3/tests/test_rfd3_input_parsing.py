"""Unit tests for the pure backbone-bond helpers in ``rfd3.inference.input_parsing``.

These three helpers decide which inter-residue atom pairs count as a polymer
backbone link when restoring bonds during input accumulation:

- ``_polymer_link_atoms_for_residue`` maps a residue name to its canonical link
  atom-name pair(s): ``C``–``N`` for standard amino acids, ``O3'``/``O3*``–``P``
  for standard DNA/RNA, and *no* links for anything else (PTMs, ligands, ``UNK``).
- ``_is_standard_polymer_backbone_bond`` is the strict check: same chain, adjacent
  ``res_id``, and the atom pair is a canonical link shared by *both* residues — so a
  bond touching a non-standard residue is rejected (its link set is empty).
- ``_is_polymer_backbone_like`` is the broader check: same chain, adjacent
  ``res_id``, and the atom pair is one of the canonical backbone pairs regardless of
  whether either residue is standard (covers PTMs like PTR/SEP).

The callers use ``standard(...) or backbone_like(...)``; the distinguishing case is a
``C``–``N`` bond between a standard and a non-standard residue, where ``standard`` is
False but ``backbone_like`` is True.

Named ``test_rfd3_input_parsing`` to avoid a pytest basename clash under the suite's
prepend import mode.
"""

import biotite.structure as struc
import numpy as np
from rfd3.inference.input_parsing import (
    _is_polymer_backbone_like,
    _is_standard_polymer_backbone_bond,
    _polymer_link_atoms_for_residue,
)


def _atom(chain_id: str, res_id: int, res_name: str, atom_name: str) -> struc.Atom:
    return struc.Atom(
        np.zeros(3, dtype=np.float32),
        chain_id=chain_id,
        res_id=res_id,
        res_name=res_name,
        atom_name=atom_name,
    )


# --- _polymer_link_atoms_for_residue ----------------------------------------


def test_link_atoms_standard_amino_acid():
    assert _polymer_link_atoms_for_residue("ALA") == {frozenset({"C", "N"})}
    assert _polymer_link_atoms_for_residue("GLY") == {frozenset({"C", "N"})}


def test_link_atoms_standard_dna_and_rna():
    expected = {frozenset({"O3'", "P"}), frozenset({"O3*", "P"})}
    assert _polymer_link_atoms_for_residue("DA") == expected  # DNA
    assert _polymer_link_atoms_for_residue("A") == expected  # RNA


def test_link_atoms_nonstandard_residue_has_no_links():
    assert _polymer_link_atoms_for_residue("PTR") == set()  # phosphotyrosine PTM
    assert _polymer_link_atoms_for_residue("UNK") == set()
    assert _polymer_link_atoms_for_residue("LIG") == set()


# --- _is_standard_polymer_backbone_bond -------------------------------------


def test_standard_bond_peptide_link_true():
    a = _atom("A", 1, "ALA", "C")
    b = _atom("A", 2, "GLY", "N")
    assert _is_standard_polymer_backbone_bond(a, b) is True


def test_standard_bond_order_independent():
    # res_id ordering reversed — abs() difference still 1.
    a = _atom("A", 2, "GLY", "N")
    b = _atom("A", 1, "ALA", "C")
    assert _is_standard_polymer_backbone_bond(a, b) is True


def test_standard_bond_different_chain_false():
    a = _atom("A", 1, "ALA", "C")
    b = _atom("B", 2, "GLY", "N")
    assert _is_standard_polymer_backbone_bond(a, b) is False


def test_standard_bond_non_adjacent_resid_false():
    a = _atom("A", 1, "ALA", "C")
    b = _atom("A", 3, "GLY", "N")
    assert _is_standard_polymer_backbone_bond(a, b) is False


def test_standard_bond_wrong_atom_pair_false():
    # Adjacent, same chain, both standard AA — but CA–N is not a canonical link.
    a = _atom("A", 1, "ALA", "CA")
    b = _atom("A", 2, "GLY", "N")
    assert _is_standard_polymer_backbone_bond(a, b) is False


def test_standard_bond_nucleic_link_true():
    a = _atom("A", 1, "DA", "O3'")
    b = _atom("A", 2, "DC", "P")
    assert _is_standard_polymer_backbone_bond(a, b) is True


def test_standard_bond_rejects_nonstandard_residue():
    # C–N pair, adjacent, same chain — but PTR contributes no canonical links,
    # so the shared-pair intersection is empty and the strict check is False.
    a = _atom("A", 1, "PTR", "C")
    b = _atom("A", 2, "ALA", "N")
    assert _is_standard_polymer_backbone_bond(a, b) is False


# --- _is_polymer_backbone_like ----------------------------------------------


def test_backbone_like_accepts_nonstandard_residue():
    # Same case the strict check rejects: the broad check accepts a C–N bond
    # touching a PTM residue. This is why callers OR the two checks together.
    a = _atom("A", 1, "PTR", "C")
    b = _atom("A", 2, "ALA", "N")
    assert _is_polymer_backbone_like(a, b) is True


def test_backbone_like_legacy_o3_star_pair():
    a = _atom("A", 1, "DA", "O3*")  # legacy O3* naming
    b = _atom("A", 2, "DC", "P")
    assert _is_polymer_backbone_like(a, b) is True


def test_backbone_like_different_chain_false():
    a = _atom("A", 1, "ALA", "C")
    b = _atom("B", 2, "ALA", "N")
    assert _is_polymer_backbone_like(a, b) is False


def test_backbone_like_non_adjacent_resid_false():
    a = _atom("A", 1, "ALA", "C")
    b = _atom("A", 5, "ALA", "N")
    assert _is_polymer_backbone_like(a, b) is False


def test_backbone_like_wrong_atom_pair_false():
    a = _atom("A", 1, "ALA", "CA")
    b = _atom("A", 2, "ALA", "C")
    assert _is_polymer_backbone_like(a, b) is False
