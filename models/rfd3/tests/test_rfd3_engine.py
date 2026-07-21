"""Unit tests for ``rfd3.engine.normalize_inputs``.

``normalize_inputs`` is the pure input-normaliser that ``process_input`` relies on to
turn the engine's flexible ``inputs`` argument into a uniform list before the spec loop.
Its contract:

- ``None`` or an empty list -> ``[None]`` (the dummy-input sentinel used for motif-only
  design),
- a comma-separated string -> the split list of paths,
- a single-path string -> a one-element list,
- a list -> returned unchanged,
- anything else -> ``ValueError``.

Pinning this contract guards the simplification in ``process_input``, which trusts
``normalize_inputs`` to have already done all of the list/comma/empty normalisation.

Named ``test_rfd3_engine`` to avoid a pytest basename clash under the suite's prepend
import mode.
"""

import pytest
from rfd3.engine import normalize_inputs


def test_none_becomes_dummy_input():
    assert normalize_inputs(None) == [None]


def test_empty_list_becomes_dummy_input():
    assert normalize_inputs([]) == [None]


def test_single_path_string_is_wrapped():
    assert normalize_inputs("a.json") == ["a.json"]


def test_comma_separated_string_is_split():
    assert normalize_inputs("a.json,b.json,c.cif") == ["a.json", "b.json", "c.cif"]


def test_list_is_returned_unchanged():
    paths = ["a.json", "b.json"]
    result = normalize_inputs(paths)
    assert result == ["a.json", "b.json"]


def test_invalid_type_raises_value_error():
    with pytest.raises(ValueError):
        normalize_inputs(5)  # type: ignore[arg-type]
