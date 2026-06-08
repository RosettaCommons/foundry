"""Unit tests for rfd3.inference.symmetry.contigs.

The contig helpers expand compact motif specifications into explicit per-residue
labels. `expand_contig_to_resid_from_string` reads a single-character chain id
followed by an inclusive `start-end` residue range (e.g. "A1-5" -> A1..A5);
`expand_contig_unsym_motif` expands the range entries in a mixed list while
keeping the plain (dash-free) names. Both are pure string logic, pinned here.
"""

from rfd3.inference.symmetry.contigs import (
    expand_contig_to_resid_from_string,
    expand_contig_unsym_motif,
)

# --- expand_contig_to_resid_from_string ---------------------------------------


def test_expand_contig_basic_range():
    assert expand_contig_to_resid_from_string("A1-5") == ["A1", "A2", "A3", "A4", "A5"]


def test_expand_contig_is_inclusive_of_endpoints():
    assert expand_contig_to_resid_from_string("B10-12") == ["B10", "B11", "B12"]


def test_expand_contig_single_residue_range():
    assert expand_contig_to_resid_from_string("C7-7") == ["C7"]


# --- expand_contig_unsym_motif ------------------------------------------------


def test_expand_unsym_motif_expands_ranges_and_keeps_plain_names():
    # plain (dash-free) names are kept first, expanded ranges appended after.
    result = expand_contig_unsym_motif(["A1-3", "LIG"])
    assert result == ["LIG", "A1", "A2", "A3"]


def test_expand_unsym_motif_without_ranges_is_unchanged():
    assert expand_contig_unsym_motif(["LIG", "GLY"]) == ["LIG", "GLY"]


def test_expand_unsym_motif_only_ranges():
    assert expand_contig_unsym_motif(["A1-2"]) == ["A1", "A2"]
