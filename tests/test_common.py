"""Unit tests for foundry.common.

These are small, pervasively-used helpers with no prior coverage. The contracts
worth pinning are the ones where behaviour is easy to get subtly wrong:
`exists`/`default` treat *only* `None` as absent (so `0`/`""`/`[]` are present),
`run_once` fires its wrapped function exactly once per process, `concat_dicts`
zips same-keyed dicts into lists, the `listmap` family materialises a list, and
`ensure_dtype` is a no-op (same object) when the dtype already matches.
"""

import pytest
import torch

from foundry.common import (
    at_least_one_exists,
    concat_dicts,
    default,
    do_nothing,
    ensure_dtype,
    exactly_one_exists,
    exists,
    listmap,
    listmap_with_idx,
    run_once,
)


def test_run_once_executes_only_once():
    calls = []

    @run_once
    def record() -> str:
        calls.append(1)
        return "ran"

    assert record() == "ran"
    assert record() is None  # second call short-circuits
    assert record() is None
    assert calls == [1]


def test_do_nothing_returns_none_for_any_args():
    assert do_nothing() is None
    assert do_nothing(1, 2, key="value") is None


def test_exists_treats_only_none_as_absent():
    assert exists(None) is False
    assert exists(0) is True
    assert exists("") is True
    assert exists([]) is True
    assert exists(False) is True


def test_default_falls_back_only_on_none():
    assert default(None, 5) == 5
    assert default(3, 5) == 3
    assert default(0, 5) == 0  # 0 exists, so it is kept


def test_exactly_one_exists():
    assert exactly_one_exists(1, None) is True
    assert exactly_one_exists(1) is True
    assert exactly_one_exists(1, 2) is False
    assert exactly_one_exists(None, None) is False


def test_at_least_one_exists():
    assert at_least_one_exists(None, 1) is True
    assert at_least_one_exists(None, None) is False
    assert at_least_one_exists() is False


def test_concat_dicts_zips_same_keys_into_lists():
    assert concat_dicts({"a": 1, "b": 2}, {"a": 3, "b": 4}) == {
        "a": [1, 3],
        "b": [2, 4],
    }


def test_concat_dicts_single_dict_wraps_values():
    assert concat_dicts({"a": 1, "b": 2}) == {"a": [1], "b": [2]}


def test_listmap_applies_and_materialises():
    assert listmap(lambda x: x + 1, [1, 2, 3]) == [2, 3, 4]
    assert listmap(str, (i for i in range(3))) == ["0", "1", "2"]  # consumes iterables
    assert listmap(lambda x: x, []) == []


def test_listmap_with_idx_passes_index_and_value():
    assert listmap_with_idx(lambda i, x: f"{i}_{x}", ["a", "b", "c"]) == [
        "0_a",
        "1_b",
        "2_c",
    ]


def test_ensure_dtype_noop_when_already_matching():
    """Matching dtype returns the same tensor object (no copy)."""
    t = torch.ones(3, dtype=torch.float32)
    assert ensure_dtype(t, torch.float32) is t


def test_ensure_dtype_converts_when_mismatched():
    t = torch.ones(3, dtype=torch.float32)
    out = ensure_dtype(t, torch.float64)
    assert out.dtype == torch.float64
    assert torch.allclose(out, t.double())


if __name__ == "__main__":
    pytest.main(["-v", __file__])
