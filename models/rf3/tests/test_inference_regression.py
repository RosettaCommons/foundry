#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../../../.ipd/shebang/rf3_exec.sh" "$0" "$@"'

import tempfile
from pathlib import Path

import biotite.structure as struc
import numpy as np
import pandas as pd
import pytest
from atomworks.io.utils.io_utils import load_any
from atomworks.ml.utils.rng import (
    create_rng_state_from_seeds,
    rng_state,
)
from conftest import TEST_DATA_DIR
from hydra import compose, initialize
from hydra.utils import instantiate

import modelhub


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
        if predicted_df[col].dtype in ["float64", "float32", "int64", "int32"]:
            # Numeric comparison with tolerance
            diff = np.abs(predicted_df[col] - baseline_df[col])
            max_diff = diff.max()
            assert (
                max_diff <= tolerance
            ), f"Numerical difference {max_diff} exceeds tolerance {tolerance} in column {col} of {predicted_file.name}"
        else:
            # Exact comparison for non-numeric
            assert predicted_df[col].equals(
                baseline_df[col]
            ), f"Non-numeric content mismatch in column {col} of {predicted_file.name}"


def compare_structures(
    predicted_file: Path, baseline_file: Path, tolerance: float = 0.2
):
    """Compare structures with RMSD after alignment.

    Args:
        predicted_file: Path to predicted structure file.
        baseline_file: Path to baseline structure file.
        tolerance: Maximum allowed RMSD in Angstroms. Defaults to 0.2.
    """
    # Load structures using atomworks
    predicted = load_any(predicted_file)
    baseline = load_any(baseline_file)

    def _compare_atom_arrays_rmsds(
        baseline: struc.AtomArray, predicted: struc.AtomArray
    ):
        """Helper to compare two AtomArray objects."""
        # Superimpose predicted onto baseline and calculate RMSD
        superimposed, _ = struc.superimpose(baseline, predicted)
        rmsd_value = struc.rmsd(baseline, superimposed)

        assert (
            rmsd_value < tolerance
        ), f"RMSD {rmsd_value:.4f} Å exceeds tolerance {tolerance} Å in {predicted_file.name}"

    # Handle AtomArrayStack (multiple models) vs AtomArray (single model)
    if isinstance(predicted, struc.AtomArrayStack):
        # Compare each model in the stack
        for i in range(len(predicted)):
            pred_model = predicted[i]
            base_model = baseline[i]
            _compare_atom_arrays_rmsds(base_model, pred_model)

    else:
        _compare_atom_arrays_rmsds(baseline, predicted)


@pytest.mark.gpu
@pytest.mark.parametrize("use_cueq", [False, True])
def test_inference_regression(use_cueq, monkeypatch):
    # Monkeypatch the global flag to control cuEquivariance usage
    monkeypatch.setattr(modelhub, "SHOULD_USE_CUEQUIVARIANCE", use_cueq)

    inputs = TEST_DATA_DIR / "5vht_from_file.cif"
    example_id = "5vht_from_file"
    baseline_dir = TEST_DATA_DIR / "inference_regression_tests" / example_id

    with (
        initialize(config_path="../configs"),
        tempfile.TemporaryDirectory() as temp_dir,
        rng_state(create_rng_state_from_seeds(1, 1, 1)),
    ):
        # Predict and save the results to the temp_dir
        cfg = compose(
            config_name="inference",
            overrides=[
                "inference_engine=rf3",
                f"inputs={inputs}",
                "annotate_b_factor_with_plddt=true",
                "one_model_per_file=false",
                f"out_dir={temp_dir}",
            ],
        )

        inference_engine = instantiate(
            cfg, temp_dir=temp_dir, _convert_="partial", _recursive_=False
        )
        inference_engine.eval()

        # Outputs are now nested in a subdirectory named after the example_id
        predicted_dir = Path(temp_dir) / example_id

        # Compare the CSV files (metrics)
        for baseline_csv in baseline_dir.glob("*metrics.csv"):
            predicted_csv = predicted_dir / baseline_csv.name
            compare_csv_files(predicted_csv, baseline_csv, tolerance=2e-3)

        # Compare the structure files
        for baseline_structure in baseline_dir.glob("*.cif.gz"):
            predicted_structure = predicted_dir / baseline_structure.name
            compare_structures(predicted_structure, baseline_structure, tolerance=0.2)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
