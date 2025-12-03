#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../../../.ipd/shebang/rf3_exec.sh" "$0" "$@"'

import json
import tempfile
from pathlib import Path

import biotite.structure as struc
import numpy as np
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


def compute_rmsd(
    predicted: struc.AtomArray | struc.AtomArrayStack,
    baseline: struc.AtomArray | struc.AtomArrayStack,
) -> float:
    """Compute RMSD between two structures after superposition."""
    # Handle AtomArrayStack - extract first model
    if isinstance(predicted, struc.AtomArrayStack):
        predicted = predicted[0]
    if isinstance(baseline, struc.AtomArrayStack):
        baseline = baseline[0]

    # Mask: only consider atoms resolved in both
    baseline_mask = np.ones(len(baseline), dtype=bool)
    if hasattr(baseline, "occupancy") and baseline.occupancy is not None:
        baseline_mask = baseline.occupancy == 1

    predicted_mask = np.ones(len(predicted), dtype=bool)
    if hasattr(predicted, "occupancy") and predicted.occupancy is not None:
        predicted_mask = predicted.occupancy == 1

    resolved_mask = baseline_mask & predicted_mask
    baseline = baseline[resolved_mask]
    predicted = predicted[resolved_mask]

    superimposed, _ = struc.superimpose(baseline, predicted)
    return struc.rmsd(baseline, superimposed)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "example_id",
    ["5vht_from_file", "8vkf_from_file"],
)
def test_inference_regression(example_id):
    """Test RF3 inference comparing predictions to baseline structure and confidence metrics."""

    input_file = TEST_DATA_DIR / f"{example_id}.cif"
    baseline_dir = TEST_DATA_DIR / "inference_regression_tests" / example_id

    with (
        initialize(config_path="../configs", version_base="1.3"),
        tempfile.TemporaryDirectory() as temp_dir,
        rng_state(create_rng_state_from_seeds(1, 1, 1)),
    ):
        # Load ground truth structure from input CIF
        ground_truth_full = remove_protein_terminal_oxygen(
            parse(input_file, **STANDARD_PARSER_ARGS)["assemblies"]["1"][0]
        )
        ground_truth = ground_truth_full[ground_truth_full.element != "H"]

        # Run inference
        cfg = compose(
            config_name="inference",
            overrides=[
                "inference_engine=rf3",
                f"inputs={input_file}",
                "annotate_b_factor_with_plddt=true",
                f"out_dir={temp_dir}",
            ],
        )

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        run_params = {k: v for k, v in cfg_dict.items() if k in RUN_PARAM_KEYS}
        init_cfg_dict = {k: v for k, v in cfg_dict.items() if k not in RUN_PARAM_KEYS}
        init_cfg = OmegaConf.create(init_cfg_dict)

        inference_engine = instantiate(init_cfg, _convert_="partial", _recursive_=False)
        inference_engine.run(**run_params)

        predicted_dir = Path(temp_dir) / example_id

        # Compare predicted structure to ground truth
        predicted_model = load_any(predicted_dir / f"{example_id}_model.cif")
        baseline_model = load_any(baseline_dir / f"{example_id}_model.cif")

        predicted_rmsd = compute_rmsd(predicted_model, ground_truth)
        baseline_rmsd = compute_rmsd(baseline_model, ground_truth)

        rmsd_tolerance = 0.3
        rmsd_difference = abs(predicted_rmsd - baseline_rmsd)
        assert (
            rmsd_difference < rmsd_tolerance
        ), f"RMSD difference {rmsd_difference:.4f}Å exceeds {rmsd_tolerance}Å tolerance"

        # Compare confidence metrics to baseline
        with open(predicted_dir / f"{example_id}_summary_confidences.json") as f:
            predicted_conf = json.load(f)
        with open(baseline_dir / f"{example_id}_summary_confidences.json") as f:
            baseline_conf = json.load(f)

        confidence_tolerance = 0.02
        for key in ["overall_plddt", "ptm", "iptm"]:
            pred_val = predicted_conf.get(key)
            base_val = baseline_conf.get(key)
            if pred_val is not None and base_val is not None:
                diff = abs(pred_val - base_val)
                assert diff <= confidence_tolerance, (
                    f"{key} difference {diff:.4f} exceeds tolerance {confidence_tolerance} "
                    f"(predicted={pred_val:.4f}, baseline={base_val:.4f})"
                )


if __name__ == "__main__":
    pytest.main([__file__])
