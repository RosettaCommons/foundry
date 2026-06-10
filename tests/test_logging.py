"""Unit tests for the pure helpers in foundry.utils.logging.

The display/configuration functions in this module are side-effecting glue
(Rich console output, warning filters, logger levels). The two pieces with a
non-obvious, environment-independent contract are pinned here:

- ``CachedDataFilter`` suppresses a specific atomworks log line by substring.
- ``condense_count_columns_of_grouped_df`` collapses the repeated per-metric
  ``count`` columns of a grouped (MultiIndex-column) DataFrame into one ``Count``
  column — but only when the count is identical across metrics in every row,
  and only for a MultiIndex frame with both ``count`` and ``mean`` sub-levels.
"""

import logging

import pandas as pd
import pytest

from foundry.utils.logging import (
    CachedDataFilter,
    condense_count_columns_of_grouped_df,
)


def _record(msg: str) -> logging.LogRecord:
    return logging.LogRecord("test", logging.INFO, __file__, 1, msg, None, None)


def test_cached_data_filter_suppresses_cached_data_message():
    assert (
        CachedDataFilter().filter(_record("Cached data not found at /tmp/x")) is False
    )


def test_cached_data_filter_keeps_unrelated_message():
    assert CachedDataFilter().filter(_record("Loaded 12 structures")) is True


def _grouped(rows: list[list[float]]) -> pd.DataFrame:
    """Frame with MultiIndex columns (metric, {count,mean}) for two metrics."""
    cols = pd.MultiIndex.from_tuples(
        [("a", "count"), ("a", "mean"), ("b", "count"), ("b", "mean")]
    )
    return pd.DataFrame(rows, columns=cols)


def test_condense_returns_non_multiindex_frame_unchanged():
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    assert condense_count_columns_of_grouped_df(df) is df


def test_condense_collapses_consistent_counts():
    df = _grouped([[5, 1.0, 5, 2.0], [3, 0.5, 3, 1.5]])
    result = condense_count_columns_of_grouped_df(df)

    assert list(result.columns) == ["a (mean)", "b (mean)", "Count"]
    assert result["Count"].tolist() == [5, 3]
    assert result["a (mean)"].tolist() == [1.0, 0.5]
    assert result["b (mean)"].tolist() == [2.0, 1.5]


def test_condense_leaves_frame_when_counts_disagree_within_a_row():
    """Row 0's metrics have counts 5 vs 6, so the frame is returned untouched."""
    df = _grouped([[5, 1.0, 6, 2.0]])
    assert condense_count_columns_of_grouped_df(df) is df


def test_condense_leaves_frame_without_a_count_sublevel():
    """MultiIndex columns lacking a 'count' level raise KeyError -> returned as-is."""
    cols = pd.MultiIndex.from_tuples([("a", "total"), ("a", "mean")])
    df = pd.DataFrame([[5, 1.0]], columns=cols)
    assert condense_count_columns_of_grouped_df(df) is df


if __name__ == "__main__":
    pytest.main(["-v", __file__])
