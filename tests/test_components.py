"""Unit tests for foundry.utils.components contig/component parsing.

These string parsers turn user-facing contig specifications (e.g. "A14-15,A16",
"5-10,A20-21") into the component/break/free-residue structures the rfd3/rfd3na
inference paths consume. Their grammar is non-obvious from the signatures — leading
chain letter, optional ranges, the "/0" chain-break token, and the R/D/P diffusion
suffixes — so the tests below pin the documented behaviour and the validation errors
on small inputs. `get_name_mask` is the pure atom-name selector (ALL/BKBN/explicit).
"""

import random

import numpy as np
import pytest

from foundry.utils.components import (
    ComponentValidationError,
    extract_pn_unit_info,
    get_design_pattern_with_constraints,
    get_motif_components_and_breaks,
    get_name_mask,
    split_contig,
)

# --- split_contig ----------------------------------------------------------------------


def test_split_contig_parses_chain_and_index():
    assert split_contig("A20") == ["A", 20]
    assert split_contig("B0") == ["B", 0]


def test_split_contig_rejects_negative_index():
    with pytest.raises(ComponentValidationError):
        split_contig("A-5")


def test_split_contig_rejects_malformed():
    with pytest.raises(ComponentValidationError):
        split_contig("AB")


# --- extract_pn_unit_info --------------------------------------------------------------


def test_extract_pn_unit_info_range():
    assert extract_pn_unit_info("A20-21") == ("A", 20, 21)


def test_extract_pn_unit_info_single_residue_duplicates_bound():
    assert extract_pn_unit_info("Z5") == ("Z", 5, 5)


def test_extract_pn_unit_info_rejects_missing_chain():
    with pytest.raises(ComponentValidationError):
        extract_pn_unit_info("123")


# --- get_motif_components_and_breaks ---------------------------------------------------


def test_motif_breaks_all_single_residues_break_between_each():
    # Documented example: each comma-separated residue is its own component, all broken.
    components, breaks = get_motif_components_and_breaks("A14,A15,A16")
    assert components == ["A14", "A15", "A16"]
    assert breaks == [True, True, True]


def test_motif_breaks_range_keeps_interior_glued():
    # Documented example: a range stays glued (break only at its start).
    components, breaks = get_motif_components_and_breaks("A14-15,A16")
    assert components == ["A14", "A15", "A16"]
    assert breaks == [True, False, True]


def test_motif_breaks_index_all_drops_internal_breaks():
    components, breaks = get_motif_components_and_breaks("A14,A15,A16", index_all=True)
    assert components == ["A14", "A15", "A16"]
    assert breaks == [False, False, False]


def test_motif_breaks_chain_break_token_has_none_break():
    components, breaks = get_motif_components_and_breaks("A14,/0")
    assert components == ["A14", "/0"]
    assert breaks == [True, None]


def test_motif_breaks_rejects_partial_unindexing_range():
    with pytest.raises(ComponentValidationError):
        get_motif_components_and_breaks("A14,5-6")


# --- get_design_pattern_with_constraints -----------------------------------------------


def test_design_pattern_free_then_fixed_motif():
    random.seed(0)
    # "5-5" is a fixed-width free segment (-> "5P"); "A20-21" expands to two fixed residues.
    assert get_design_pattern_with_constraints("5-5,A20-21") == ["5P", "A20", "A21"]


def test_design_pattern_rna_suffix_preserved():
    random.seed(0)
    # A trailing "R"/"D" marks a non-fixed RNA/DNA segment and the suffix is carried through.
    assert get_design_pattern_with_constraints("3-3R,A5") == ["3R", "A5"]


def test_design_pattern_chain_break_token_passthrough():
    random.seed(0)
    assert get_design_pattern_with_constraints("A1,/0,A2") == ["A1", "/0", "A2"]


def test_design_pattern_threads_total_length():
    random.seed(0)
    # length="6" minus the 1 motif residue leaves exactly 5 free -> the "5-5" segment.
    assert get_design_pattern_with_constraints("5-5,A20", length="6") == ["5P", "A20"]


def test_design_pattern_raises_when_length_infeasible():
    random.seed(0)
    # 12 total - 1 motif = 11 free required, but the segment caps at 10.
    with pytest.raises(ComponentValidationError):
        get_design_pattern_with_constraints("5-10,A20", length="12")


# --- get_name_mask ---------------------------------------------------------------------


def _atom_names(*names):
    return np.array(names)


def test_get_name_mask_all():
    names = _atom_names("N", "CA", "C", "O", "CB")
    assert get_name_mask(names, "ALL").tolist() == [True] * 5


def test_get_name_mask_backbone_excludes_cb():
    names = _atom_names("N", "CA", "C", "O", "CB")
    assert get_name_mask(names, "BKBN").tolist() == [True, True, True, True, False]


def test_get_name_mask_explicit_comma_string():
    names = _atom_names("N", "CA", "C", "O", "CB")
    assert get_name_mask(names, "N,CA").tolist() == [True, True, False, False, False]


def test_get_name_mask_accepts_list_of_names():
    names = _atom_names("N", "CA", "C", "O", "CB")
    assert get_name_mask(names, ["N", "CA"]).tolist() == [
        True,
        True,
        False,
        False,
        False,
    ]


def test_get_name_mask_empty_string_selects_nothing():
    names = _atom_names("N", "CA", "C")
    assert get_name_mask(names, "").tolist() == [False, False, False]


def test_get_name_mask_rejects_duplicate_names():
    names = _atom_names("N", "CA", "C")
    with pytest.raises(ComponentValidationError):
        get_name_mask(names, "N,N")


def test_get_name_mask_rejects_missing_names():
    names = _atom_names("N", "CA", "C")
    with pytest.raises(ComponentValidationError):
        get_name_mask(names, "XYZ")


def test_get_name_mask_tip_requires_resname():
    names = _atom_names("N", "CA", "C")
    with pytest.raises(ComponentValidationError):
        get_name_mask(names, "TIP", source_resname=None)


def test_get_name_mask_rejects_non_multiple_atom_count():
    # Two N's but one CA: the match count (3) is not a multiple of the requested names (2).
    names = _atom_names("N", "N", "CA")
    with pytest.raises(ComponentValidationError):
        get_name_mask(names, "N,CA")
