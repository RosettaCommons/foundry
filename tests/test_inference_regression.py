#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../scripts/shebang/modelhub_exec.sh" "$0" "$@"'

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import TEST_DATA_DIR
from hydra import compose, initialize
from hydra.utils import instantiate

from atomworks.ml.utils.rng import (
    create_rng_state_from_seeds,
    rng_state,
)


def compare_csv_files(
    predicted_file: Path, baseline_file: Path, tolerance: float = 1e-3
):
    """Compare CSV files with numerical tolerance for floating-point values."""
    predicted_df = pd.read_csv(predicted_file)
    baseline_df = pd.read_csv(baseline_file)

    # Check shape
    assert (
        predicted_df.shape == baseline_df.shape
    ), f"Shape mismatch in {predicted_file.name}: {predicted_df.shape} vs {baseline_df.shape}"

    # Check column names (order-independent)
    predicted_cols = set(predicted_df.columns)
    baseline_cols = set(baseline_df.columns)
    assert (
        predicted_cols == baseline_cols
    ), f"Column mismatch in {predicted_file.name}: {predicted_cols} vs {baseline_cols}"

    # Compare values with tolerance for numeric columns
    for col in predicted_df.columns:
        print(predicted_df[col])
        print(baseline_df[col])
        if predicted_df[col].dtype in ["float64", "float32", "int64", "int32"]:
            # Numeric comparison with tolerance
            diff = np.abs(predicted_df[col] - baseline_df[col])
            max_diff = diff.max()
            assert (
                max_diff <= tolerance
            ), f"Numerical difference {max_diff} exceeds tolerance {tolerance} in column {col} of {predicted_file.name}. Predicted: {predicted_df[col]}, Baseline: {baseline_df[col]}"
        else:
            # Exact comparison for non-numeric
            assert predicted_df[col].equals(
                baseline_df[col]
            ), f"Non-numeric content mismatch in column {col} of {predicted_file.name}"


@pytest.mark.gpu
def test_inference_regression():
    print("GPU available: ", torch.cuda.is_available())
    # inputs = "/home/ncorley/projects/modelhub_dev/tests/data/5vht_from_file.cif"
    inputs = TEST_DATA_DIR / "5vht_from_file.cif"
    data_dir = TEST_DATA_DIR / "inference_regression_tests" / "5vht_from_file"

    with (
        initialize(config_path="../configs"),
        tempfile.TemporaryDirectory() as temp_dir,
        rng_state(create_rng_state_from_seeds(1, 1, 1)),
    ):
        # Predict and save the results to the temp_dir
        cfg = compose(
            config_name="inference",
            overrides=[
                "inference_engine=af3",
                f"inputs={inputs}",
                "annotate_b_factor_with_plddt=true",
                "one_model_per_file=false",
                f"out_dir={temp_dir}",
            ],
        )

        inference_engine = instantiate(
            cfg, temp_dir=temp_dir, _convert_="partial", _recursive_=False
        )
        inference_engine.trainer.fabric.launch()
        inference_engine.eval()

        # Compare the results to the baseline
        # (CSV files with confidence outputs)
        for file in data_dir.glob("*.csv"):
            predicted_file = Path(temp_dir) / file.name
            compare_csv_files(predicted_file, file, tolerance=1e-3)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
