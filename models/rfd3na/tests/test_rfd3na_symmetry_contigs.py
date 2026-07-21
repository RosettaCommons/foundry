"""Unit tests for the contig helpers in ``rfd3na.inference.symmetry.contigs``.

- ``expand_contig_to_resid_from_string("A1-5")`` expands an inclusive ``"<chain><start>-<end>"``
  range into per-residue ids ``["A1", ..., "A5"]``.
- ``expand_contig_unsym_motif(names)`` expands any range entries (those containing ``"-"``)
  and appends the expansions after the non-range entries, which are kept as-is.
- ``get_unsym_motif_mask(atom_array, names)`` builds a boolean atom mask, matching each name
  against ``res_name`` (and ``src_component`` when present) or, for a ``"<chain><resid>"``
  token like ``"A2"``, against the chain/res_id index.
"""

import numpy as np
from biotite.structure import Atom, array
from rfd3na.inference.symmetry.contigs import (
    expand_contig_to_resid_from_string,
    expand_contig_unsym_motif,
    get_unsym_motif_mask,
)

# --- expand_contig_to_resid_from_string -------------------------------------


def test_expand_range_is_inclusive():
    assert expand_contig_to_resid_from_string("A1-5") == ["A1", "A2", "A3", "A4", "A5"]


def test_expand_single_residue_range():
    assert expand_contig_to_resid_from_string("B3-3") == ["B3"]


def test_expand_preserves_chain_letter():
    assert expand_contig_to_resid_from_string("C10-12") == ["C10", "C11", "C12"]


# --- expand_contig_unsym_motif ----------------------------------------------


def test_expand_unsym_keeps_non_range_and_appends_expansions():
    # Non-range names stay first; the expanded range is appended afterwards.
    assert expand_contig_unsym_motif(["A1-3", "LIG"]) == ["LIG", "A1", "A2", "A3"]


def test_expand_unsym_without_ranges_is_unchanged():
    assert expand_contig_unsym_motif(["LIG", "HEM"]) == ["LIG", "HEM"]


def test_expand_unsym_all_ranges():
    assert expand_contig_unsym_motif(["A1-2", "B5-6"]) == ["A1", "A2", "B5", "B6"]


# --- get_unsym_motif_mask ---------------------------------------------------


def _atom_array():
    atoms = [
        Atom([0, 0, 0], chain_id="A", res_id=i + 1, res_name=rn, atom_name="CA")
        for i, rn in enumerate(["GLY", "LIG", "ALA"])
    ]
    return array(atoms)


def test_mask_matches_res_name():
    mask = get_unsym_motif_mask(_atom_array(), ["LIG"])
    assert mask.tolist() == [False, True, False]


def test_mask_matches_chain_resid_index():
    # "A2" -> chain A, res_id 2 -> the second atom.
    mask = get_unsym_motif_mask(_atom_array(), ["A2"])
    assert mask.tolist() == [False, True, False]


def test_mask_all_false_when_no_match():
    mask = get_unsym_motif_mask(_atom_array(), ["ZZZ"])
    assert mask.tolist() == [False, False, False]


def test_mask_ors_multiple_names():
    mask = get_unsym_motif_mask(_atom_array(), ["GLY", "ALA"])
    assert mask.tolist() == [True, False, True]
    assert mask.dtype == np.bool_
