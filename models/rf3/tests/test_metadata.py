"""Unit tests for rf3.metrics.metadata.ExtraInfo.

``ExtraInfo.compute`` copies entries out of the dataloader's ``extra_info`` dict
into the metrics dict, but only those whose value is *both* hashable and
JSON-serializable (so it survives downstream metric aggregation and logging).
The non-obvious contracts pinned here: unhashable containers (``list``/``dict``)
are dropped while a hashable ``tuple`` is kept; a hashable-but-not-serializable
object is dropped; and ``keys_to_store`` narrows the copied keys to an allowlist.
"""

from rf3.metrics.metadata import ExtraInfo


def test_keeps_basic_scalar_types():
    info = {"i": 1, "s": "x", "f": 1.5, "b": True, "n": None}
    assert ExtraInfo().compute(extra_info=info) == info


def test_drops_unhashable_containers():
    result = ExtraInfo().compute(extra_info={"keep": 1, "lst": [1, 2], "dct": {"k": 1}})
    assert result == {"keep": 1}


def test_keeps_hashable_tuple():
    # A tuple is hashable and JSON-serializes to an array, so it is retained.
    assert ExtraInfo().compute(extra_info={"t": (1, 2)}) == {"t": (1, 2)}


def test_drops_hashable_but_non_serializable():
    # A bare object() is hashable (id-based) but not JSON-serializable -> dropped.
    result = ExtraInfo().compute(extra_info={"keep": "ok", "obj": object()})
    assert result == {"keep": "ok"}


def test_keys_to_store_restricts_to_allowlist():
    info = {"a": 1, "b": 2}
    assert ExtraInfo(keys_to_store=["a"]).compute(extra_info=info) == {"a": 1}


def test_empty_extra_info_returns_empty():
    assert ExtraInfo().compute(extra_info={}) == {}
