"""Format-contract tests for the MSA (.a3m) input rf3 expects at inference.

These are the checkpoint-free "Layer 1" of the MSA test suite: they pin the
*format* an MSA file must follow and how ``atomworks`` parses it, so this file
doubles as executable documentation of what a valid rf3 MSA looks like. The
end-to-end fold behaviour (feeding these files through ``rf3 fold`` for
monomers, homodimers, and heteromers) lives in the checkpoint-gated
``tests/integration/test_msa_fold.py`` "Layer 2".

rf3 consumes ColabFold-style **a3m** files, referenced per chain via a
component's ``msa_path`` (JSON input) or a ``_msa_paths_by_chain_id`` header
(CIF input). ``atomworks`` parses them with
``parse_a3m(path) -> (msa, ins, tax_ids)``:

- ``msa``  — ``(N, L)`` byte array; row 0 is the query, rows 1.. are homologs.
- ``ins``  — ``(N, L)`` insertion counts: the number of lowercase (insertion)
  residues that sat immediately to the *left* of each aligned column before
  they were stripped. The query row is all zeros.
- ``tax_ids`` — ``(N,)`` taxonomy IDs parsed from ``TaxID=<n>`` in each header.
  Row 0 is forced to the sentinel ``"query"``; a header with no ``TaxID=`` maps
  to the empty string. These IDs are the key that
  ``PairAndMergePolymerMSAs`` uses to pair homologs across chains for
  multimer/heteromer inputs, so getting them right is what makes multi-chain
  MSAs meaningful.

Format rules exercised here:
- everything before the first ``>`` header line is ignored;
- uppercase letters and ``-`` are aligned columns (gaps included) — all rows
  therefore share the query's column count ``L``;
- lowercase letters are insertions relative to the query and are removed from
  the alignment (but counted in ``ins``).
"""

import numpy as np
import pytest
from atomworks.ml.transforms.msa._msa_loading_utils import parse_a3m

# A 7-column query alignment used across the tests below.
QUERY_SEQ = "GLKAADE"


def _write_minimal_a3m(path):
    """Write a tiny, self-documenting a3m and return its path.

    Four sequences (query + 3 homologs) chosen to exercise every parsing rule:

    - row 0 ``query``            — no insertions, tax id forced to ``"query"``;
    - row 1 identical homolog    — carries an explicit ``TaxID=101``;
    - row 2 with two insertions  — the lowercase ``kk`` sits left of column 2
                                   and is stripped, contributing ``ins[2] == 2``;
    - row 3 with a gap + no TaxID — ``-`` is a real aligned column (kept), and
                                   the header lacks ``TaxID=`` so the id is ``""``.

    All rows collapse to the query's 7 aligned columns.
    """
    path.write_text(
        # A leading comment line before the first '>' must be ignored by the parser.
        "#a3m minimal fixture\n"
        f">query\n{QUERY_SEQ}\n"
        f">UniRef100_AAA Example protein TaxID=101 RepID=AAA_TEST\n{QUERY_SEQ}\n"
        ">UniRef100_BBB Example protein TaxID=202 RepID=BBB_TEST\nGLkkKAADE\n"
        ">plain_header_without_taxid\nG-KAADE\n"
    )
    return path


@pytest.fixture
def minimal_a3m(tmp_path):
    return _write_minimal_a3m(tmp_path / "minimal.a3m")


def test_parse_a3m_shape_and_query_row(minimal_a3m):
    """The parsed MSA is (N, L) with the query as row 0 and uniform width."""
    msa, ins, _tax_ids = parse_a3m(str(minimal_a3m))

    assert msa.shape == (4, len(QUERY_SEQ)), "expected 4 sequences x 7 columns"
    assert ins.shape == msa.shape, "insertion array must match the MSA shape"
    # Row 0 is the query verbatim (bytes joined back into a string).
    assert b"".join(msa[0]).decode() == QUERY_SEQ
    # Every row shares the query's column count (a3m alignment invariant).
    assert all(len(row) == len(QUERY_SEQ) for row in msa)


def test_parse_a3m_strips_insertions_but_keeps_gaps(minimal_a3m):
    """Lowercase insertions are removed and counted; ``-`` gaps stay as columns."""
    msa, ins, _tax_ids = parse_a3m(str(minimal_a3m))

    # Query row has no insertions.
    assert ins[0].tolist() == [0] * len(QUERY_SEQ)

    # Row 2 ("GLkkKAADE"): the two lowercase residues sit left of column 2, so
    # they are stripped from the alignment and recorded as ins[2] == 2.
    assert b"".join(msa[2]).decode() == QUERY_SEQ
    assert ins[2].tolist() == [0, 0, 2, 0, 0, 0, 0]

    # Row 3 ("G-KAADE"): the gap is a genuine aligned column, not an insertion.
    assert b"".join(msa[3]).decode() == "G-KAADE"
    assert ins[3].tolist() == [0] * len(QUERY_SEQ)


def test_parse_a3m_extracts_tax_ids_for_pairing(minimal_a3m):
    """Taxonomy IDs are parsed from headers and drive cross-chain MSA pairing."""
    _msa, _ins, tax_ids = parse_a3m(str(minimal_a3m))

    # Row 0 is always the reserved query sentinel; TaxID= values are parsed from
    # the UniRef headers; a header without TaxID= yields the empty string.
    assert tax_ids.tolist() == ["query", "101", "202", ""]


def test_parse_a3m_query_only_has_depth_one(tmp_path):
    """A query-only a3m (no homologs) parses to depth 1 rather than crashing.

    This is the degenerate "no MSA found" shape a chain gets when only its own
    sequence is available, mirroring OpenFold3's empty-input handling test.
    """
    query_only = tmp_path / "query_only.a3m"
    query_only.write_text(f">query\n{QUERY_SEQ}\n")

    msa, ins, tax_ids = parse_a3m(str(query_only))

    assert msa.shape == (1, len(QUERY_SEQ))
    assert np.array_equal(ins, np.zeros((1, len(QUERY_SEQ))))
    assert tax_ids.tolist() == ["query"]
