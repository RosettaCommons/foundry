"""Regression tests for the atom14 base pipeline to ensure consistent behavior.

To regenerate regression data:
1. Uncomment the save lines in the respective test functions
2. Run the test to generate new .pkl files
3. Re-comment the save lines
"""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pytest
from atomworks.io.utils.testing import assert_same_atom_array
from atomworks.ml.utils.rng import create_rng_state_from_seeds, rng_state
from rfd3.testing.testing_utils import (
    TEST_JSON_DATA,
    assert_tensor_or_array_equal,
    build_pipelines,
    instantiate_example,
    load_train_or_val_cfg,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for a pipeline test case."""

    name: str
    example_names: List[str]
    inference_modes: List[bool]
    pipeline: Dict[bool, object]


# Define pipeline configurations
base_cfg = load_train_or_val_cfg("pretrain")
unindexed_cfg = load_train_or_val_cfg("test-unindexed")
unindexed_cfg.datasets.global_transform_args.train_conditions.island.frequency = 1e10
unindexed_cfg.datasets.global_transform_args.train_conditions.island.p_unindex_motif_tokens = 1.0

PIPELINE_CONFIGS = [
    PipelineConfig(
        name="pretrain",
        example_names=["1qys-0", "1p5d-0"],
        inference_modes=[False, True],
        pipeline=build_pipelines(composed_config=base_cfg),
    ),
    PipelineConfig(
        name="test-unindexed",
        example_names=["af2-122"],
        inference_modes=[False],  # Only test training mode where unindexing happens
        pipeline=build_pipelines(composed_config=unindexed_cfg),
    ),
]


def _make_test_identifier(
    example_name: str, is_inference: bool, config_name: str
) -> str:
    """Generate a test identifier for parametrized tests and file paths."""
    mode = "inference" if is_inference else "train"
    return f"{example_name}-{mode}-{config_name}"


def _get_regression_data_path(
    example_name: str, is_inference: bool, config_name: str
) -> Path:
    """Get the path for regression test data based on the example name, inference mode, and config."""
    regression_dir = Path(__file__).parent / "regression_test_data"
    regression_dir.mkdir(parents=True, exist_ok=True)

    # Use shared identifier logic for consistent naming
    identifier = _make_test_identifier(example_name, is_inference, config_name)
    # Convert to file-friendly format (replace hyphens with underscores)
    file_name = identifier.replace("-", "_")
    return regression_dir / f"{file_name}.pkl"


def _run_pipeline_test(
    example_name: str, is_inference: bool, config: PipelineConfig
) -> dict:
    """Run a single pipeline test and return the result."""
    args = TEST_JSON_DATA[example_name].copy()

    # Run pipeline with fixed random seed for reproducibility
    seed = 42
    with rng_state(
        create_rng_state_from_seeds(np_seed=seed, torch_seed=seed, py_seed=seed)
    ):
        input_data = instantiate_example(args, is_inference=is_inference)
        input_data["example_id"] = example_name
        pipeline = config.pipeline
        result = pipeline[is_inference](input_data)

    assert result is not None, "Pipeline should return a result"
    return result


def _assert_pipeline_results_equal(
    result: dict, expected: dict, example_name: str, mode: str
):
    """Assert that two pipeline results are equal."""
    # Check that both have the same keys
    assert set(result.keys()) == set(
        expected.keys()
    ), f"Keys don't match for {example_name} in {mode} mode"

    # Check atom array if present
    assert "atom_array" in result, "Atom array not found in result"
    assert_same_atom_array(
        result["atom_array"],
        expected["atom_array"],
        compare_coords=True,
        compare_bonds=True,
        # (All annotation categories present in the expected atom array are compared)
        annotations_to_compare=expected["atom_array"].get_annotation_categories(),
    )

    # Check features
    assert "feats" in result, "Features not found in result"
    _assert_features_equal(result["feats"], expected["feats"], example_name, mode)


def _assert_features_equal(
    feats: dict, expected_feats: dict, example_name: str, mode: str
):
    """Assert that feature dictionaries are equal, with new features being a superset of old features."""
    # Check that all expected feature keys are present in the new features
    missing_keys = set(expected_feats.keys()) - set(feats.keys())
    assert (
        not missing_keys
    ), f"Missing feature keys {missing_keys} for {example_name} in {mode} mode"

    # Only check features that were in the expected results (allows for new features)
    for key in expected_feats.keys():
        feat = feats[key]
        expected_feat = expected_feats[key]

        # Check shapes
        assert (
            feat.shape == expected_feat.shape
        ), f"Feature {key} shape mismatch for {example_name} in {mode} mode: {feat.shape} vs {expected_feat.shape}"

        # Check values with tolerance
        assert_tensor_or_array_equal(
            feat,
            expected_feat,
            f"Feature {key} values don't match for {example_name} in {mode} mode",
        )


@pytest.mark.slow
@pytest.mark.parametrize(
    "example_name,is_inference,config",
    [
        pytest.param(
            example_name,
            is_inference,
            config,
            id=_make_test_identifier(example_name, is_inference, config.name),
        )
        for config in PIPELINE_CONFIGS
        for example_name in config.example_names
        for is_inference in config.inference_modes
    ],
)
def test_atom14_pipeline_regression(
    example_name: str, is_inference: bool, config: PipelineConfig
):
    """Test the atom14 pipeline against stored regression results for various configurations."""

    # Run the pipeline test
    result = _run_pipeline_test(example_name, is_inference, config)

    # Get regression data path using shared logic
    regression_path = _get_regression_data_path(example_name, is_inference, config.name)

    # # Uncomment the following lines to create/update the regression data
    # with regression_path.open("wb") as f:
    #     pickle.dump(result, f)
    #     logger.info(f"Saved regression data to {regression_path}")

    # Load expected result
    with regression_path.open("rb") as f:
        expected_result = pickle.load(f)

    # Compare results
    config_desc = f" ({config.name})" if config.name != "pretrain" else ""
    mode_description = f"{'inference' if is_inference else 'training'}{config_desc}"
    if "specification" in result:
        expected_result["specification"] = result["specification"]

    _assert_pipeline_results_equal(
        result, expected_result, example_name, mode_description
    )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
