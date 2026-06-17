"""Unit tests for the pure logic in foundry.callbacks.

Most of the callbacks package is side-effecting training/validation glue (Rich
console output, fabric logging, forward-hook registration, matplotlib plotting)
and is intentionally not unit-tested. The one piece with a non-obvious,
environment-independent contract is pinned here:

``StoreValidationMetricsInDFCallback._load_and_concatenate_csvs`` merges the
per-rank validation CSVs written for one epoch, de-duplicating rows by the
``example_id``/``dataset`` pair (the same example may be validated on more than
one rank) and skipping empty rank files.
"""

import pandas as pd
import pytest

from foundry.callbacks.metrics_logging import StoreValidationMetricsInDFCallback


def _write_rank_csv(save_dir, rank: int, epoch: int, rows: list[dict]) -> None:
    path = save_dir / f"validation_output_rank_{rank}_epoch_{epoch}.csv"
    pd.DataFrame(rows, columns=["example_id", "dataset", "lddt"]).to_csv(
        path, index=False
    )


def _callback(tmp_path) -> StoreValidationMetricsInDFCallback:
    return StoreValidationMetricsInDFCallback(save_dir=tmp_path)


def test_single_rank_returns_all_rows_without_temp_key(tmp_path):
    _write_rank_csv(
        tmp_path,
        rank=0,
        epoch=3,
        rows=[
            {"example_id": "e1", "dataset": "d1", "lddt": 0.9},
            {"example_id": "e2", "dataset": "d1", "lddt": 0.8},
        ],
    )

    merged = _callback(tmp_path)._load_and_concatenate_csvs(epoch=3)

    assert sorted(merged["example_id"]) == ["e1", "e2"]
    assert "_example_key" not in merged.columns


def test_duplicate_example_across_ranks_is_deduplicated(tmp_path):
    _write_rank_csv(
        tmp_path,
        rank=0,
        epoch=1,
        rows=[
            {"example_id": "e1", "dataset": "d1", "lddt": 0.9},
            {"example_id": "e2", "dataset": "d1", "lddt": 0.8},
        ],
    )
    _write_rank_csv(
        tmp_path,
        rank=1,
        epoch=1,
        rows=[
            {"example_id": "e2", "dataset": "d1", "lddt": 0.8},
            {"example_id": "e3", "dataset": "d1", "lddt": 0.7},
        ],
    )

    merged = _callback(tmp_path)._load_and_concatenate_csvs(epoch=1)

    # e2 appears on both ranks but is kept once; e1 and e3 once each.
    assert sorted(merged["example_id"]) == ["e1", "e2", "e3"]


def test_same_example_different_dataset_is_kept(tmp_path):
    """De-duplication is keyed on example_id AND dataset, not example_id alone."""
    _write_rank_csv(
        tmp_path,
        rank=0,
        epoch=2,
        rows=[
            {"example_id": "e1", "dataset": "d1", "lddt": 0.9},
            {"example_id": "e1", "dataset": "d2", "lddt": 0.5},
        ],
    )

    merged = _callback(tmp_path)._load_and_concatenate_csvs(epoch=2)

    assert len(merged) == 2
    assert sorted(merged["dataset"]) == ["d1", "d2"]


def test_empty_rank_csv_is_skipped(tmp_path):
    """A rank that validated no examples writes a header-only (empty) CSV."""
    _write_rank_csv(
        tmp_path,
        rank=0,
        epoch=5,
        rows=[{"example_id": "e1", "dataset": "d1", "lddt": 0.9}],
    )
    _write_rank_csv(tmp_path, rank=1, epoch=5, rows=[])

    merged = _callback(tmp_path)._load_and_concatenate_csvs(epoch=5)

    assert merged["example_id"].tolist() == ["e1"]


if __name__ == "__main__":
    pytest.main(["-v", __file__])
