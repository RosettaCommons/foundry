"""Unit tests for foundry.metrics.metric.

`Metric` + `MetricManager` are the introspection machinery every model's
validation metrics ride on. The contracts worth pinning are non-obvious from
the signatures: `Metric.__init__` rejects required/prohibited tag conflicts;
`compute_from_kwargs` either forwards kwargs verbatim (no mapping) or pulls each
compute argument from a nested-key path, treating `optional_kwargs` as
present-only; and `MetricManager.__call__` extracts `example_id`, prefixes each
result key with the metric name, and skips metrics whose tag requirements the
batch does not satisfy.
"""

from functools import cached_property

import pytest

from foundry.metrics.metric import Metric, MetricManager


class _SumMetric(Metric):
    """No key mapping: receives kwargs verbatim; absorbs extras via **kwargs."""

    def compute(self, x, y, **kwargs):
        return {"value": x + y}


class _MappedMetric(Metric):
    """Pulls compute args from nested key paths in the incoming kwargs."""

    @cached_property
    def kwargs_to_compute_args(self):
        return {"x": ("a", "b"), "y": ("c",)}

    def compute(self, x, y):
        return {"value": x + y}


class _OptionalMetric(Metric):
    @cached_property
    def kwargs_to_compute_args(self):
        return {"x": ("x",), "opt": ("opt",)}

    @property
    def optional_kwargs(self):
        return frozenset(["opt"])

    def compute(self, x, opt="default"):
        return {"x": x, "opt": opt}


class _ListMetric(Metric):
    def compute(self, **kwargs):
        return [{"row": 1}, {"row": 2}]


class _BoomMetric(Metric):
    def compute(self, **kwargs):
        raise ValueError("boom")


# --- Metric base -----------------------------------------------------------


def test_tag_conflict_raises():
    with pytest.raises(ValueError, match="disjoint"):
        _SumMetric(required_tags_all=["a"], prohibited_tags=["a"])


def test_required_compute_args_read_from_signature():
    assert _MappedMetric().required_compute_args == frozenset({"x", "y"})


# --- compute_from_kwargs ---------------------------------------------------


def test_compute_from_kwargs_passes_through_without_mapping():
    assert _SumMetric().compute_from_kwargs(x=1, y=2) == {"value": 3}


def test_compute_from_kwargs_remaps_nested_keys():
    result = _MappedMetric().compute_from_kwargs(a={"b": 10}, c=5)
    assert result == {"value": 15}


def test_compute_from_kwargs_optional_absent_uses_default():
    assert _OptionalMetric().compute_from_kwargs(x=1) == {"x": 1, "opt": "default"}


def test_compute_from_kwargs_optional_present_is_passed():
    assert _OptionalMetric().compute_from_kwargs(x=1, opt=99) == {"x": 1, "opt": 99}


# --- MetricManager ---------------------------------------------------------


def test_manager_prefixes_keys_and_defaults_example_id_to_none():
    manager = MetricManager({"sum": _SumMetric()})
    assert manager(x=1, y=2) == {"example_id": None, "sum.value": 3}


def test_manager_extracts_example_id_from_extra_info():
    manager = MetricManager({"sum": _SumMetric()})
    result = manager(x=1, y=2, extra_info={"example_id": "abc"})
    assert result["example_id"] == "abc"
    assert result["sum.value"] == 3


def test_manager_required_tags_all_must_all_be_present():
    manager = MetricManager({"sum": _SumMetric(required_tags_all=["needed"])})
    missing = manager(x=1, y=2, extra_info={"metrics_tags": ["other"]})
    assert "sum.value" not in missing
    present = manager(x=1, y=2, extra_info={"metrics_tags": ["needed", "other"]})
    assert present["sum.value"] == 3


def test_manager_required_tags_any_needs_one():
    manager = MetricManager({"sum": _SumMetric(required_tags_any=["p", "q"])})
    missing = manager(x=1, y=2, extra_info={"metrics_tags": ["other"]})
    assert "sum.value" not in missing
    present = manager(x=1, y=2, extra_info={"metrics_tags": ["q"]})
    assert present["sum.value"] == 3


def test_manager_prohibited_tags_block_computation():
    manager = MetricManager({"sum": _SumMetric(prohibited_tags=["skip"])})
    blocked = manager(x=1, y=2, extra_info={"metrics_tags": ["skip"]})
    assert "sum.value" not in blocked


def test_manager_list_result_stored_under_metric_name():
    manager = MetricManager({"rows": _ListMetric()})
    result = manager(anything=1)
    assert result["rows"] == [{"row": 1}, {"row": 2}]


def test_manager_swallows_failure_when_raise_errors_false():
    manager = MetricManager({"boom": _BoomMetric()}, raise_errors=False)
    result = manager(x=1)  # the failing metric is skipped, no exception
    assert "boom" not in result
    assert result == {"example_id": None}


def test_manager_propagates_failure_when_raise_errors_true():
    manager = MetricManager({"boom": _BoomMetric()}, raise_errors=True)
    with pytest.raises(ValueError, match="boom"):
        manager(x=1)


def test_from_metrics_accepts_list_of_tuples():
    manager = MetricManager.from_metrics([("sum", _SumMetric())])
    assert manager(x=1, y=2)["sum.value"] == 3


def test_from_metrics_rejects_non_metric():
    with pytest.raises(TypeError, match="must be a Metric"):
        MetricManager.from_metrics({"bad": object()})


if __name__ == "__main__":
    pytest.main(["-v", __file__])
