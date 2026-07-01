"""Unit tests for the pure helper ``rfd3na.metrics.metrics_utils._flatten_dict``.

Recursively flattens a nested dict, joining key paths with a separator
(``{"a": {"b": 1}} -> {"a.b": 1}``). Only ``dict`` values recurse; everything else
(including lists/tuples) is treated as a leaf.
"""

from rfd3na.metrics.metrics_utils import _flatten_dict


def test_flattens_one_level():
    assert _flatten_dict({"a": {"b": 1, "c": 2}}) == {"a.b": 1, "a.c": 2}


def test_already_flat_is_unchanged():
    assert _flatten_dict({"x": 1, "y": 2}) == {"x": 1, "y": 2}


def test_flattens_deep_nesting():
    assert _flatten_dict({"a": {"b": {"c": 3}}}) == {"a.b.c": 3}


def test_custom_separator():
    assert _flatten_dict({"a": {"b": 1}}, sep="/") == {"a/b": 1}


def test_empty_dict():
    assert _flatten_dict({}) == {}


def test_non_dict_values_are_leaves():
    # Lists are not recursed into — they are kept as leaf values.
    assert _flatten_dict({"a": [1, 2], "b": {"c": 3}}) == {"a": [1, 2], "b.c": 3}
