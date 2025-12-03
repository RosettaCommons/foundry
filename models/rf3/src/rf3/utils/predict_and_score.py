"""Utility to run RF3 predictions and then a set of metrics on those predictions."""

from pathlib import Path

import biotite.structure as struc
from atomworks.ml.transforms.filters import remove_protein_terminal_oxygen
from atomworks.ml.utils.rng import create_rng_state_from_seeds, rng_state
from beartype.typing import Any
from rf3.inference_engines.rf3 import RF3InferenceEngine
from rf3.utils.inference import InferenceInput

from foundry.metrics.metric import MetricManager
from foundry.utils.ddp import RankedLogger

ranked_logger = RankedLogger(__name__, rank_zero_only=True)


def _clean_atom_array_for_rf3(atom_array: struc.AtomArray) -> struc.AtomArray:
    """Preprocess atom array by removing terminal oxygen and hydrogens."""
    original_count = len(atom_array)

    # Remove terminal oxygen atoms
    atom_array = remove_protein_terminal_oxygen(atom_array)
    if len(atom_array) < original_count:
        ranked_logger.warning(
            f"Removed {original_count - len(atom_array)} terminal oxygen atoms. "
            f"Atom count changed from {original_count} to {len(atom_array)}."
        )
        original_count = len(atom_array)

    # Filter to heavy atoms only (no hydrogen)
    atom_array = atom_array[atom_array.element != "H"]
    if len(atom_array) < original_count:
        ranked_logger.warning(
            f"Removed {original_count - len(atom_array)} hydrogen atoms. "
            f"Atom count changed from {original_count} to {len(atom_array)}."
        )

    return atom_array


def predict_and_score_with_rf3(
    atom_arrays: list[struc.AtomArray],
    ckpt_path: str | Path,
    metrics=None,
    n_recycles: int = 10,
    diffusion_batch_size: int = 5,
    num_steps: int = 50,
    example_ids: list[str] | None = None,
    annotate_b_factor_with_plddt: bool = True,
    rng_seed: int = 1,
) -> dict[str, dict[str, Any]]:
    """Predict structures with RF3 and evaluate against inputs.

    Metrics are computed using the RF3 inference engine's internal trainer,
    which automatically handles symmetry resolution during validation.

    Args:
        atom_arrays: List of input structures (ground truth).
        ckpt_path: Path to RF3 checkpoint file.
        metrics: Metrics to compute. Can be:
            - Dict mapping names to Metric objects
            - List of (name, Metric) tuples
            - None (no metrics)
        n_recycles: Number of recycles. Defaults to ``10``.
        diffusion_batch_size: Number of structures per input. Defaults to ``5``.
        num_steps: Number of diffusion steps. Defaults to ``50``.
        example_ids: Optional IDs for each structure. Defaults to "example_0", "example_1", etc.
        annotate_b_factor_with_plddt: Whether to write pLDDT to B-factor. Defaults to ``True``.
        rng_seed: RNG seed for reproducibility. Defaults to ``1``.

    Returns:
        Dict mapping example_id to::

            {
                "predicted_structures": list[AtomArray] | AtomArrayStack,
                "metrics": dict[str, float],
            }

    Example:
        ```python
        metrics = [
            ("all_atom_lddt", AllAtomLDDT()),
            ("by_type_lddt", ByTypeLDDT()),
        ]

        results = predict_and_score_with_rf3(
            atom_arrays=structures,
            ckpt_path="rf3_latest.pt",
            metrics=metrics,
        )
        ```
    """
    # Generate example IDs if not provided
    if example_ids is None:
        example_ids = [f"example_{i}" for i in range(len(atom_arrays))]

    # Preprocess atom arrays (remove terminal oxygen and hydrogens so that atom counts match)
    ranked_logger.info("Preprocessing atom arrays...")
    preprocessed_arrays = [_clean_atom_array_for_rf3(arr.copy()) for arr in atom_arrays]

    # Convert metrics to MetricManager if provided
    if metrics is not None:
        metric_manager = MetricManager.from_metrics(metrics)
    else:
        # (Prediction only, no metrics)
        metric_manager = None

    # Initialize RF3 engine (one-time) with custom metrics
    ranked_logger.info("Initializing RF3 inference engine...")
    inference_engine = RF3InferenceEngine(
        ckpt_path=ckpt_path,
        n_recycles=n_recycles,
        diffusion_batch_size=diffusion_batch_size,
        num_steps=num_steps,
        seed=None,  # We'll use external RNG state (set below)
        metrics_cfg=metric_manager,  # Pass MetricManager to engine
    )

    with rng_state(create_rng_state_from_seeds(rng_seed, rng_seed, rng_seed)):
        results = {}

        # Loop over each example
        for example_id, ground_truth_array in zip(example_ids, preprocessed_arrays):
            ranked_logger.info(f"Predicting structure for {example_id}...")

            # Create InferenceInput from AtomArray
            inference_input = InferenceInput.from_atom_array(
                ground_truth_array, example_id=example_id
            )

            # Run inference in-memory
            # The engine's trainer.validation_step() automatically:
            # 1. Runs inference
            # 2. Applies symmetry resolution
            # 3. Computes configured metrics
            inference_results = inference_engine.run(
                inputs=inference_input,
                out_dir=None,  # Return in-memory
                annotate_b_factor_with_plddt=annotate_b_factor_with_plddt,
            )

            # Extract results for this example
            result = inference_results[example_id]

            # Check for early stopping
            if result.get("early_stopped", False):
                ranked_logger.warning(
                    f"Early stopping triggered for {example_id} "
                    f"(mean pLDDT = {result.get('mean_plddt', 'N/A'):.2f})"
                )
                results[example_id] = {
                    "predicted_structures": None,
                    "metrics": result.get("metrics", {}),
                    "early_stopped": True,
                }
                continue

            # Store results
            results[example_id] = {
                "predicted_structures": result["predicted_structures"],
                "metrics": result.get("metrics", {}),
            }

    return results
