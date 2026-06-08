"""Unit tests for the pure helpers in rf3.inference_engines.rf3.

- ``compute_ranking_score`` = ``0.8 * ipTM + 0.2 * pTM - 100 * has_clash``; when
  ipTM is ``None`` it falls back to pTM (so a single-chain score is just pTM),
  and a missing pTM counts as 0.
- ``dump_json_compact_arrays`` writes indented JSON but collapses each
  innermost array onto a single line, while remaining valid JSON.
"""

import io
import json

from rf3.inference_engines.rf3 import (
    compute_ranking_score,
    dump_json_compact_arrays,
)

# --- compute_ranking_score --------------------------------------------------


def test_ranking_score_combines_iptm_and_ptm():
    assert compute_ranking_score(iptm=1.0, ptm=1.0, has_clash=False) == 1.0
    assert compute_ranking_score(iptm=0.5, ptm=0.25, has_clash=False) == 0.45


def test_ranking_score_clash_penalty():
    assert compute_ranking_score(iptm=1.0, ptm=1.0, has_clash=True) == -99.0


def test_ranking_score_missing_iptm_falls_back_to_ptm():
    # Single-chain: ipTM is None, so 0.8*pTM + 0.2*pTM == pTM.
    assert compute_ranking_score(iptm=None, ptm=0.7, has_clash=False) == 0.7


def test_ranking_score_missing_ptm_counts_as_zero():
    assert compute_ranking_score(iptm=0.6, ptm=None, has_clash=False) == 0.8 * 0.6


def test_ranking_score_all_missing_is_zero():
    assert compute_ranking_score(iptm=None, ptm=None, has_clash=False) == 0.0
    assert compute_ranking_score(iptm=None, ptm=None, has_clash=True) == -100.0


# --- dump_json_compact_arrays -----------------------------------------------


def _dump(obj: dict) -> str:
    buf = io.StringIO()
    dump_json_compact_arrays(obj, buf)
    return buf.getvalue()


def test_compact_arrays_collapses_flat_array_onto_one_line():
    out = _dump({"vals": [1, 2, 3], "meta": {"n": 3}})

    assert "[1,2,3]" in out
    # Still valid JSON that round-trips to the input.
    assert json.loads(out) == {"vals": [1, 2, 3], "meta": {"n": 3}}


def test_compact_arrays_roundtrips_nested_and_empty():
    obj = {"matrix": [[1, 2], [3, 4]], "empty": [], "scalar": 5}
    out = _dump(obj)

    assert json.loads(out) == obj
    # The innermost rows collapse even when the outer array stays multi-line.
    assert "[1,2]" in out and "[3,4]" in out


def test_compact_arrays_object_without_arrays_is_unchanged_semantically():
    obj = {"a": {"b": {"c": 1}}, "d": "text"}
    out = _dump(obj)

    assert json.loads(out) == obj
