"""CPU-only tests for the pure parsing helpers in ``mpnn.utils.inference``.

These cover the small, data-free functions (CLI/JSON parsing, path resolution) that
run in the generic gate. The heavier structure-building parts of the module are
exercised by the cluster-coupled suite (``test_inference_utils.py``, held out of CI).
"""

import argparse

import pytest
from mpnn.utils.inference import (
    _absolute_path_or_none,
    none_or_type,
    parse_json_like,
    parse_list_like,
    str2bool,
)


@pytest.mark.parametrize(
    ("value", "expected"), [("True", True), ("1", True), ("False", False), ("0", False)]
)
def test_str2bool_valid(value: str, expected: bool):
    assert str2bool(value) is expected


@pytest.mark.parametrize("value", ["true", "false", "yes", "", "2"])
def test_str2bool_invalid_raises(value: str):
    with pytest.raises(argparse.ArgumentTypeError):
        str2bool(value)


def test_none_or_type_sentinel_and_cast():
    assert none_or_type("None", int) is None
    assert none_or_type("5", int) == 5
    assert none_or_type("1.5", float) == 1.5


def test_parse_json_like_passthrough_non_strings():
    assert parse_json_like(None) is None
    assert parse_json_like(42) == 42
    assert parse_json_like([1, 2]) == [1, 2]


def test_parse_json_like_json_and_literals():
    assert parse_json_like('{"k": 1}') == {"k": 1}
    assert parse_json_like("[1, 2, 3]") == [1, 2, 3]
    assert parse_json_like("5") == 5


def test_parse_json_like_comma_separated_and_plain():
    assert parse_json_like("a, b ,c") == ["a", "b", "c"]
    assert parse_json_like("hello") == "hello"


def test_parse_list_like():
    assert parse_list_like(None) is None
    assert parse_list_like("[1, 2]") == [1, 2]
    assert parse_list_like("a,b") == ["a", "b"]
    # A single scalar is wrapped into a singleton list.
    assert parse_list_like("5") == [5]


def test_absolute_path_or_none():
    assert _absolute_path_or_none(None) is None
    assert _absolute_path_or_none("") is None
    resolved = _absolute_path_or_none("some/rel/path")
    assert resolved is not None
    assert resolved.endswith("some/rel/path")
    # Result is absolute.
    assert resolved.startswith("/")
