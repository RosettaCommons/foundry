"""Unit tests for the pure helpers in rf3.utils.inference.

- ``extract_example_id_from_path`` derives an example id from a file path by stripping a
  known extension. It tries the known extensions longest-first, so a compound suffix like
  ``.cif.gz`` is stripped whole (``prot.cif.gz`` -> ``prot``) rather than via the stdlib
  ``Path.stem`` fallback, which would only drop the final ``.gz`` (-> ``prot.cif``). Unknown
  extensions fall back to ``Path.stem`` (drops just the last suffix).
- ``_resolve_override`` implements CLI-override-vs-source-value priority: a non-None override
  always wins (and warns when it shadows a truthy source value); otherwise the source value is
  used.
- ``extract_example_ids_from_json`` reads a JSON file holding a list of example dicts and
  returns their ``name`` fields.

Named ``test_rf3_inference`` to avoid a pytest basename clash under the suite's prepend
import mode.
"""

import json
import logging
from pathlib import Path

import pytest
from rf3.utils.inference import (
    _resolve_override,
    extract_example_id_from_path,
    extract_example_ids_from_json,
)

# --- extract_example_id_from_path -------------------------------------------


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("1abc.cif", "1abc"),
        ("1abc.bcif", "1abc"),
        ("x.pdb", "x"),
        ("x.pdb.gz", "x"),
        ("prot.bcif.gz", "prot"),
        ("data.json", "data"),
    ],
)
def test_strips_known_extensions(filename, expected):
    assert extract_example_id_from_path(Path(filename)) == expected


def test_strips_compound_cif_gz_whole_not_stem_only():
    # The longest-first match strips `.cif.gz` entirely; a plain `Path.stem` would
    # only drop `.gz` and leave `1abc.cif`.
    assert extract_example_id_from_path(Path("1abc.cif.gz")) == "1abc"


def test_uses_filename_not_full_path():
    assert extract_example_id_from_path(Path("/some/dir/1abc.cif")) == "1abc"


def test_unknown_extension_falls_back_to_stem():
    assert extract_example_id_from_path(Path("weird.xyz")) == "weird"


def test_unknown_multi_dot_extension_strips_only_last_suffix():
    # Stem fallback drops only the final suffix, so the inner dot is preserved.
    assert extract_example_id_from_path(Path("multi.part.xyz")) == "multi.part"


# --- _resolve_override ------------------------------------------------------


def test_override_wins_over_source():
    assert _resolve_override(["a"], ["b"], "template_selection", "ex") == ["a"]


def test_override_used_when_source_missing():
    assert _resolve_override(["a"], None, "template_selection", "ex") == ["a"]


def test_source_used_when_no_override():
    assert _resolve_override(None, ["b"], "template_selection", "ex") == ["b"]


def test_returns_none_when_neither_present():
    assert _resolve_override(None, None, "template_selection", "ex") is None


def test_override_shadowing_truthy_source_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="rf3.utils.inference"):
        _resolve_override(["a"], ["b"], "template_selection", "ex")
    assert any("template_selection" in r.message for r in caplog.records)


def test_override_over_falsy_source_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING, logger="rf3.utils.inference"):
        result = _resolve_override(["a"], [], "template_selection", "ex")
    assert result == ["a"]
    assert caplog.records == []


# --- extract_example_ids_from_json ------------------------------------------


def test_extracts_names_from_json_list(tmp_path):
    path = tmp_path / "examples.json"
    path.write_text(json.dumps([{"name": "ex1"}, {"name": "ex2"}]))
    assert extract_example_ids_from_json(path) == ["ex1", "ex2"]
