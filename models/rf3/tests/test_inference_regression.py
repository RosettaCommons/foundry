#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../../../.ipd/shebang/rf3_exec.sh" "$0" "$@"'

import tempfile
from pathlib import Path

import biotite.structure as struc
import numpy as np
import pandas as pd
import pytest
from atomworks.io.parser import STANDARD_PARSER_ARGS, parse
from atomworks.io.utils.io_utils import load_any
from atomworks.ml.transforms.filters import remove_protein_terminal_oxygen
from atomworks.ml.utils.rng import (
    create_rng_state_from_seeds,
    rng_state,
)
from conftest import TEST_DATA_DIR
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
from rf3.utils.inference import InferenceInput

RUN_PARAM_KEYS = {
    "inputs",
    "out_dir",
    "dump_predictions",
    "dump_trajectories",
    "one_model_per_file",
    "annotate_b_factor_with_plddt",
    "sharding_pattern",
    "skip_existing",
    "template_selection",
    "ground_truth_conformer_selection",
    "cyclic_chains",
}
"""Run parameters that should be passed to engine.run(), not __init__."""


def assert_similar_csv_files(
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
            mean_pred = predicted_df[col].mean()
            mean_base = baseline_df[col].mean()
            assert abs(mean_pred - mean_base) <= tolerance, (
                f"Numerical difference {abs(mean_pred - mean_base)} exceeds tolerance {tolerance} in column {col} of {predicted_file.name} "
                f"(predicted_mean={mean_pred}, baseline_mean={mean_base})"
            )
        else:
            # Exact comparison for non-numeric
            assert predicted_df[col].equals(
                baseline_df[col]
            ), f"Non-numeric content mismatch in column {col} of {predicted_file.name}"


def compare_structures(
    predicted: struc.AtomArray | struc.AtomArrayStack | list[struc.AtomArray],
    baseline: struc.AtomArray | struc.AtomArrayStack | list[struc.AtomArray],
) -> list[float]:
    """Compare structures with RMSD after alignment.

    Args:
        predicted: Predicted structure(s).
        baseline: Baseline structure(s).

    Returns:
        List of RMSD values (one per model).
    """

    def _compare_single(
        baseline_array: struc.AtomArray,
        predicted_array: struc.AtomArray,
    ) -> float:
        """Compare two AtomArray objects and return RMSD."""
        assert isinstance(baseline_array, struc.AtomArray)
        assert isinstance(predicted_array, struc.AtomArray)
        assert (
            len(baseline_array) == len(predicted_array)
        ), f"Atom count mismatch: baseline {len(baseline_array)} vs predicted {len(predicted_array)}"

        # Mask: only consider atoms resolved in both baseline and predicted arrays
        # Check if occupancy exists, otherwise assume all atoms are resolved
        baseline_mask = np.ones(len(baseline_array), dtype=bool)
        if (
            hasattr(baseline_array, "occupancy")
            and baseline_array.occupancy is not None
        ):
            baseline_mask = baseline_array.occupancy == 1

        predicted_mask = np.ones(len(predicted_array), dtype=bool)
        if (
            hasattr(predicted_array, "occupancy")
            and predicted_array.occupancy is not None
        ):
            predicted_mask = predicted_array.occupancy == 1

        resolved_mask = baseline_mask & predicted_mask
        baseline_array = baseline_array[resolved_mask]
        predicted_array = predicted_array[resolved_mask]

        # Superimpose and calculate RMSD
        superimposed, _ = struc.superimpose(baseline_array, predicted_array)
        rmsd_value = struc.rmsd(baseline_array, superimposed)
        return rmsd_value

    # Convert to common format: list of AtomArrays
    def _to_atom_array_list(structures):
        """Convert various structure formats to a list of AtomArray objects."""
        if isinstance(structures, struc.AtomArrayStack):
            return [structures[i] for i in range(len(structures))]
        elif isinstance(structures, list):
            # If it's a list of AtomArrayStacks, ensure they all are length 1, and downcast to AtomArrays
            if all(isinstance(s, struc.AtomArrayStack) for s in structures):
                assert all(
                    len(s) == 1 for s in structures
                ), "All AtomArrayStacks in the list must contain exactly one structure."
                return [s[0] for s in structures]
            # Assert list of AtomArrays
            assert all(
                isinstance(s, struc.AtomArray) for s in structures
            ), "All elements in the list must be AtomArray objects."
            return structures
        else:
            # Single AtomArray - wrap in list
            return [structures]

    predicted_list = _to_atom_array_list(predicted)
    baseline_list = _to_atom_array_list(baseline)

    # If baseline is a single structure, replicate to match predicted length
    if len(baseline_list) == 1 and len(predicted_list) > 1:
        baseline_list = baseline_list * len(predicted_list)

    # Compare each pair
    return [
        _compare_single(base, pred) for base, pred in zip(baseline_list, predicted_list)
    ]


@pytest.mark.gpu
@pytest.mark.parametrize(
    "example_id,rmsd_tolerance,csv_tolerance",
    [
        ("5vht_from_file", 0.1, 0.01),
        ("8vkf_from_file", 0.1, 0.01),
    ],
)
def test_inference_regression(example_id, rmsd_tolerance, csv_tolerance):
    """Test RF3 inference by comparing predictions to ground truth structure and baseline metrics."""

    input_file = TEST_DATA_DIR / f"{example_id}.cif"
    baseline_dir = TEST_DATA_DIR / "inference_regression_tests" / example_id

    with (
        initialize(config_path="../configs"),
        tempfile.TemporaryDirectory() as temp_dir,
        rng_state(create_rng_state_from_seeds(1, 1, 1)),
    ):
        # Load ground truth structure from input CIF
        ground_truth_full = remove_protein_terminal_oxygen(
            parse(input_file, **STANDARD_PARSER_ARGS)["assemblies"]["1"][0]
        )
        # Filter to heavy atoms only (no hydrogen)
        ground_truth = ground_truth_full[ground_truth_full.element != "H"]

        # Predict and save the results to the temp_dir
        cfg = compose(
            config_name="inference",
            overrides=[
                "inference_engine=rf3",
                f"inputs={input_file}",
                "annotate_b_factor_with_plddt=true",
                "one_model_per_file=false",
                f"out_dir={temp_dir}",
            ],
        )

        # Separate config into init params and run params
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        run_params = {k: v for k, v in cfg_dict.items() if k in RUN_PARAM_KEYS}
        init_cfg_dict = {k: v for k, v in cfg_dict.items() if k not in RUN_PARAM_KEYS}
        init_cfg = OmegaConf.create(init_cfg_dict)

        # Instantiate engine (only __init__ params)
        inference_engine = instantiate(init_cfg, _convert_="partial", _recursive_=False)

        # Run inference with the new API
        inference_engine.run(**run_params)

        # Outputs are now nested in a subdirectory named after the example_id
        predicted_dir = Path(temp_dir) / example_id

        # Compare CSV metrics to baseline
        baseline_metrics_csv = baseline_dir / f"{example_id}_metrics.csv"
        predicted_metrics_csv = predicted_dir / f"{example_id}_metrics.csv"
        assert_similar_csv_files(
            predicted_metrics_csv, baseline_metrics_csv, tolerance=csv_tolerance
        )

        # Compare predicted structures to ground truth
        # Load all predicted structures
        predicted_files = sorted(predicted_dir.glob("*.cif.gz"))
        predicted_structures = [load_any(f) for f in predicted_files]
        mean_predicted_rmsd = np.mean(
            compare_structures(predicted_structures, ground_truth)
        )

        # Load all baseline structures
        baseline_files = sorted(baseline_dir.glob(f"{example_id}_model_*.cif.gz"))
        baseline_structures = [load_any(f) for f in baseline_files]
        mean_baseline_rmsd = np.mean(
            compare_structures(baseline_structures, ground_truth)
        )

        # Assert mean RMSD difference between baseline and predicted < threshold
        rmsd_difference = abs(mean_predicted_rmsd - mean_baseline_rmsd)
        assert (
            rmsd_difference < rmsd_tolerance
        ), f"Mean RMSD difference {rmsd_difference:.4f}Å exceeds {rmsd_tolerance}Å tolerance for {example_id}"


@pytest.mark.gpu
@pytest.mark.parametrize(
    "example_id,rmsd_tolerance,csv_tolerance",
    [
        ("5vht_from_file", 0.1, 0.01),
        ("8vkf_from_file", 0.1, 0.01),
    ],
)
def test_inference_regression_in_memory(example_id, rmsd_tolerance, csv_tolerance):
    """Test in-memory inference against baseline predictions."""

    inputs_path = TEST_DATA_DIR / f"{example_id}.cif"
    baseline_dir = TEST_DATA_DIR / "inference_regression_tests" / example_id

    with (
        initialize(config_path="../configs"),
        tempfile.TemporaryDirectory() as temp_dir,
        rng_state(create_rng_state_from_seeds(1, 1, 1)),
    ):
        # Load ground truth structure from input CIF
        ground_truth_full = remove_protein_terminal_oxygen(
            parse(inputs_path, **STANDARD_PARSER_ARGS)["assemblies"]["1"][0]
        )

        # Filter to heavy atoms only (no hydrogen)
        ground_truth = ground_truth_full[ground_truth_full.element != "H"]

        # Create InferenceInput from CIF file
        inference_input = InferenceInput.from_cif_path(
            inputs_path, example_id=example_id
        )

        # Load config
        cfg = compose(
            config_name="inference",
            overrides=[
                "inference_engine=rf3",
            ],
        )

        # Separate config into init params and run params
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        init_cfg_dict = {k: v for k, v in cfg_dict.items() if k not in RUN_PARAM_KEYS}
        init_cfg = OmegaConf.create(init_cfg_dict)

        # Initialize engine
        engine = instantiate(init_cfg, _convert_="partial", _recursive_=False)

        # Run inference in-memory
        results = engine.run(
            inputs=inference_input,
            out_dir=None,  # Return in-memory
            annotate_b_factor_with_plddt=True,
        )

        # Extract results for this example
        result = results[example_id]

        # Compare CSV metrics to baseline
        # (Write predicted metrics to temp CSV for comparison)
        predicted_metrics_df = pd.DataFrame([result["metrics"]])
        predicted_metrics_csv = Path(temp_dir) / f"{example_id}_metrics.csv"
        predicted_metrics_df.to_csv(predicted_metrics_csv, index=False)
        baseline_metrics_csv = baseline_dir / f"{example_id}_metrics.csv"
        assert_similar_csv_files(
            predicted_metrics_csv, baseline_metrics_csv, tolerance=csv_tolerance
        )

        # Compare predicted structures to ground truth
        predicted_structures = result["predicted_structures"]
        mean_predicted_rmsd = np.mean(
            compare_structures(predicted_structures, ground_truth)
        )

        # Load baseline structures and compare to ground truth
        baseline_structures = []
        for i in range(len(predicted_structures)):
            baseline_file = baseline_dir / f"{example_id}_model_{i}.cif.gz"
            baseline_structures.append(load_any(baseline_file))
        mean_baseline_rmsd = np.mean(
            compare_structures(baseline_structures, ground_truth)
        )

        # Assert mean RMSD difference < threshold
        rmsd_difference = abs(mean_predicted_rmsd - mean_baseline_rmsd)
        assert (
            rmsd_difference < rmsd_tolerance
        ), f"Mean RMSD difference {rmsd_difference:.4f}Å exceeds {rmsd_tolerance}Å tolerance for {example_id}"

if __name__ == "__main__":
    pytest.main([__file__])
