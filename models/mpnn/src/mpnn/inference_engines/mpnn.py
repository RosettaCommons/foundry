import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from atomworks.constants import (
    DICT_THREE_TO_ONE,
    PROTEIN_BACKBONE_ATOM_NAMES,
    UNKNOWN_AA,
)
from atomworks.ml.utils.token import get_token_starts, spread_token_wise
from biotite.structure import AtomArray
from mpnn.collate.feature_collator import FeatureCollator
from mpnn.metrics.sequence_recovery import (
    InterfaceSequenceRecovery,
    SequenceRecovery,
)
from mpnn.model.mpnn import LigandMPNN, ProteinMPNN
from mpnn.pipelines.mpnn import build_mpnn_transform_pipeline
from mpnn.transforms.feature_aggregation.token_encodings import MPNN_TOKEN_ENCODING
from mpnn.utils.inference import (
    MPNN_GLOBAL_INFERENCE_DEFAULTS,
    MPNNInferenceInput,
    MPNNInferenceOutput,
    _absolute_path_or_none,
)
from mpnn.utils.weights import load_legacy_weights

from foundry.inference_engines.checkpoint_registry import REGISTERED_CHECKPOINTS
from foundry.metrics.metric import MetricManager
from foundry.utils.ddp import RankedLogger

ranked_logger = RankedLogger(__name__, rank_zero_only=True)


class MPNNInferenceEngine:
    """Inference engine for ProteinMPNN/LigandMPNN."""

    def __init__(
        self,
        *,
        model_type: str = MPNN_GLOBAL_INFERENCE_DEFAULTS["model_type"],
        checkpoint_path: str = MPNN_GLOBAL_INFERENCE_DEFAULTS["checkpoint_path"],
        is_legacy_weights: bool = MPNN_GLOBAL_INFERENCE_DEFAULTS["is_legacy_weights"],
        out_directory: str | None = MPNN_GLOBAL_INFERENCE_DEFAULTS["out_directory"],
        write_fasta: bool = MPNN_GLOBAL_INFERENCE_DEFAULTS["write_fasta"],
        write_structures: bool = MPNN_GLOBAL_INFERENCE_DEFAULTS["write_structures"],
        device: str | torch.device | None = None,
    ):
        # Store raw configuration
        self.model_type = model_type
        self.is_legacy_weights = is_legacy_weights
        self.out_directory = out_directory
        self.write_fasta = write_fasta
        self.write_structures = write_structures

        # allow null for checkpoint path when foundry-installed
        # TODO: Currently this assumes the model type is the key in the registered path. Rework needed
        self.checkpoint_path = (
            str(
                REGISTERED_CHECKPOINTS[
                    self.model_type.replace("_", "")
                ].get_default_path()
            )
            if not checkpoint_path
            else checkpoint_path
        )

        # Determine the device.
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set up allowed model types.
        self.allowed_model_types = {"protein_mpnn", "ligand_mpnn"}

        # Validate the user configuration.
        self._validate_all()

        # Post-process the configuration (making absolute paths, etc).
        self._post_process_engine_config()

        # Build and load the model.
        self.model = self._build_and_load_model().to(self.device)

        # Construct metrics manager.
        self.metrics = self._build_metrics_manager()

    def _validate_model_config(self) -> None:
        """Validate model-type and checkpoint-related configuration."""
        # Model type.
        if self.model_type not in self.allowed_model_types:
            raise ValueError(
                f"model_type must be one of {self.allowed_model_types}; "
                f"got {self.model_type!r}"
            )

        # Checkpoint path.
        if not isinstance(self.checkpoint_path, str):
            raise TypeError("checkpoint_path must be a string path.")

        # Check that the checkpoint path exists.
        ckpt_path = Path(_absolute_path_or_none(self.checkpoint_path))
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"checkpoint_path does not exist: {self.checkpoint_path}"
            )

        # Legacy-weight flag.
        if not isinstance(self.is_legacy_weights, bool):
            raise TypeError("is_legacy_weights must be a bool.")

    def _validate_output_config(self) -> None:
        """Validate output-directory and writing-related configuration."""
        # Output directory.
        if self.out_directory is not None:
            # Must be a string.
            if not isinstance(self.out_directory, str):
                raise TypeError("out_directory must be a string when provided.")

        # Boolean writing flags.
        for name in ("write_fasta", "write_structures"):
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool.")

            # If asked to write outputs, out_directory must be set.
            if value and self.out_directory is None:
                raise ValueError(f"{name} is True, but out_directory is not set.")

    def _validate_all(self) -> None:
        """Run validation on the user-specified engine config variables."""
        # Validate the model configuration.
        self._validate_model_config()

        # Validate the output configuration.
        self._validate_output_config()

    def _post_process_engine_config(self) -> None:
        """Normalize paths into absolute paths."""
        # Make checkpoint path absolute.
        self.checkpoint_path = _absolute_path_or_none(self.checkpoint_path)

        # Make output directory absolute.
        if self.out_directory is not None:
            self.out_directory = _absolute_path_or_none(self.out_directory)

    def _build_and_load_model(self) -> torch.nn.Module:
        # Load model architecture.
        if self.model_type == "protein_mpnn":
            model = ProteinMPNN()
        elif self.model_type == "ligand_mpnn":
            model = LigandMPNN()
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        # Load weights.
        if self.is_legacy_weights:
            ranked_logger.info("Loading legacy MPNN weights.")
            load_legacy_weights(model, self.checkpoint_path)
        else:
            ranked_logger.info("Loading MPNN weights.")

            # Load the checkpoint.
            checkpoint = torch.load(
                self.checkpoint_path, map_location="cpu", weights_only=False
            )

            # Check that checkpoint is a dict.
            if not isinstance(checkpoint, dict) or "model" not in checkpoint:
                raise TypeError("Expected checkpoint to be a dict with a 'model' key.")

            state_dict = checkpoint["model"]

            model.load_state_dict(state_dict, strict=True)

        # Set model to eval mode.
        model.eval()

        return model

    def _build_metrics_manager(self) -> MetricManager:
        """Build the metrics manager for inference."""

        # Construct metrics dict.
        metrics: dict[str, Any] = {
            "sequence_recovery": SequenceRecovery(return_per_example_metrics=True),
        }
        if self.model_type == "ligand_mpnn":
            metrics["interface_sequence_recovery"] = InterfaceSequenceRecovery(
                return_per_example_metrics=True
            )

        # Construct the MetricManager.
        metric_manager = MetricManager.from_metrics(metrics, raise_errors=True)

        return metric_manager

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(
        self,
        *,
        input_dicts: list[dict[str, Any]] | None = None,
        atom_arrays: list[AtomArray] | None = None,
    ) -> list[MPNNInferenceOutput]:
        """Run inference and return a list of MPNNInferenceOutput objects.

        Parameters
        ----------
        input_dicts:
            Optional list of per-input JSON-like dictionaries (one per
            input). If None, 'atom_arrays' must be provided.
        atom_arrays:
            Optional list of externally provided AtomArray objects. If given,
            must align one-to-one with 'input_dicts'. If None, 'input_dicts'
            must be sufficient to resolve structures internally.

        Returns
        -------
        list[MPNNInferenceOutput]
            A flat list of per-design MPNNInferenceOutput objects. Writing
            of CIF/FASTA outputs is handled internally based on engine-level
            configuration.
        """
        if input_dicts is None and atom_arrays is None:
            raise ValueError(
                "At least one of 'input_dicts' or 'atom_arrays' must be provided."
            )
        if atom_arrays is not None and input_dicts is not None:
            if len(atom_arrays) != len(input_dicts):
                raise ValueError(
                    "'atom_arrays' and 'input_dicts' must have the same length."
                )

        # Determine the number of inputs.
        num_inputs = len(input_dicts) if input_dicts is not None else len(atom_arrays)
        results: list[MPNNInferenceOutput] = []
        for input_idx in range(num_inputs):
            # Construct the per-input MPNNInferenceInput.
            inference_input = MPNNInferenceInput.from_atom_array_and_dict(
                atom_array=atom_arrays[input_idx] if atom_arrays is not None else None,
                input_dict=input_dicts[input_idx] if input_dicts is not None else None,
            )

            # Optional per-input RNG seeding for deterministic sampling across
            # batches. Initialize the seed at the beginning of the batches.
            seed = inference_input.input_dict["seed"]
            if seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

            # Run the batches for this input.
            for batch_idx in range(inference_input.input_dict["number_of_batches"]):
                ranked_logger.info(
                    f"Running MPNN inference for input {input_idx}, "
                    f"batch {batch_idx}..."
                )

                # Run a single batch.
                result = self._run_batch(
                    atom_array=inference_input.atom_array,
                    input_dict=inference_input.input_dict,
                    batch_idx=batch_idx,
                )
                results.extend(result)

        # Write outputs if requested.
        self._write_outputs(results)

        return results

    def _run_batch(
        self,
        atom_array: AtomArray,
        input_dict: dict[str, Any],
        batch_idx: int | None = None,
    ) -> list[MPNNInferenceOutput]:
        """
        Run a single batch (possibly multiple designs) through the pipeline.

        This function:
            - builds the transform pipeline based on 'input_dict',
            - runs the pipeline and collator,
            - executes the model forward pass,
            - decodes sequences and applies them to the pipeline output
              AtomArray,
            - constructs 'MPNNInferenceOutput' objects
        """
        # Overriding of default pipeline args from input_dict.
        pipeline_args = dict()
        if input_dict["occupancy_threshold_sidechain"] is not None:
            pipeline_args["occupancy_threshold_sidechain"] = input_dict[
                "occupancy_threshold_sidechain"
            ]
        if input_dict["occupancy_threshold_backbone"] is not None:
            pipeline_args["occupancy_threshold_backbone"] = input_dict[
                "occupancy_threshold_backbone"
            ]
        if input_dict["undesired_res_names"] is not None:
            pipeline_args["undesired_res_names"] = input_dict["undesired_res_names"]

        # Construct the pipeline.
        pipeline = build_mpnn_transform_pipeline(
            model_type=self.model_type,
            is_inference=True,
            minimal_return=True,
            device=self.device,
            **pipeline_args,
        )

        # Construct the collator.
        collator = FeatureCollator()

        # Data dict for pipeline: atom_array plus scalar user-settings.
        data: dict[str, Any] = {
            "atom_array": atom_array.copy(),
            # Scalar user settings.
            "structure_noise": input_dict["structure_noise"],
            "decode_type": input_dict["decode_type"],
            "causality_pattern": input_dict["causality_pattern"],
            "initialize_sequence_embedding_with_ground_truth": input_dict[
                "initialize_sequence_embedding_with_ground_truth"
            ],
            "atomize_side_chains": input_dict["atomize_side_chains"],
            "repeat_sample_num": input_dict["repeat_sample_num"],
            "features_to_return": input_dict["features_to_return"],
        }

        # Run the pipeline.
        pipeline_output = pipeline(data)

        # Construct the collated network input.
        network_input = collator([pipeline_output])

        # Run the model forward pass.
        with torch.no_grad():
            network_output = self.model(network_input)

        # Compute metrics once per batch.
        metrics_output = self.metrics(
            network_input=network_input,
            network_output=network_output,
            extra_info={},
        )

        # Extract the sampled sequences.
        # S_sampled: [B = batch_size, L = sequence length]
        S_sampled = (
            network_output["decoder_features"]["S_sampled"].detach().cpu().numpy()
        )
        B, L = S_sampled.shape
        if B != input_dict["batch_size"]:
            raise ValueError(
                "Mismatch between network output batch size and input_dict batch_size."
            )

        # Extract the metrics.
        sequence_recovery_per_design = (
            metrics_output["sequence_recovery.sequence_recovery_per_example_sampled"]
            .detach()
            .cpu()
            .numpy()
        )
        if self.model_type == "ligand_mpnn":
            interface_sequence_recovery_per_design = (
                metrics_output[
                    "interface_sequence_recovery.interface_sequence_recovery_per_example_sampled"
                ]
                .detach()
                .cpu()
                .numpy()
            )
        else:
            interface_sequence_recovery_per_design = None

        # Grab the index to token mapping from the model.
        idx_to_token = MPNN_TOKEN_ENCODING.idx_to_token

        # Construct the output objects.
        outputs: list[MPNNInferenceOutput] = []
        for design_idx in range(input_dict["batch_size"]):
            # Per design, copy the atom array.
            design_atom_array = pipeline_output["atom_array"].copy()

            # Grab the non-atomized atom and token level arrays. This mimics
            # the logic in the pipeline for token level extraction, so it
            # should lead to a one-to-one mapping between decoded tokens and
            # non-atomized residues.
            design_non_atomized_array = design_atom_array[~design_atom_array.atomize]
            design_non_atomized_token_starts = get_token_starts(
                design_non_atomized_array
            )
            design_non_atomized_token_level = design_non_atomized_array[
                design_non_atomized_token_starts
            ]

            # Create the res_name array for the design.
            designed_resnames = np.array(
                [idx_to_token[int(token_idx)] for token_idx in S_sampled[design_idx]],
                dtype=design_atom_array.res_name.dtype,
            )

            # Sanity check: decoded sequence length must match number of
            # non-atomized tokens.
            if len(design_non_atomized_token_level) != len(designed_resnames):
                raise ValueError(
                    "Mismatch between number of non-atomized tokens and "
                    "decoded sequence length."
                )

            # Spread token-level residue names back to atom level, but only
            # over the non-atomized subset.
            designed_resnames_atom = spread_token_wise(
                design_non_atomized_array,
                designed_resnames,
            )

            # Create a full res_name array.
            full_resnames = design_atom_array.res_name.copy()
            full_resnames[~design_atom_array.atomize] = designed_resnames_atom

            # Overwrite with designed residue names.
            design_atom_array.set_annotation("res_name", full_resnames)

            # We need to remove any non-atomized residue atoms that no
            # longer belong (i.e. old side chain atoms). We want to keep any
            # atom that is atomized, any atom that is a backbone atom, and
            # any atom that was fixed.
            design_is_backbone_atom = np.isin(
                design_atom_array.atom_name,
                PROTEIN_BACKBONE_ATOM_NAMES,
            )
            if (
                "mpnn_designed_residue_mask"
                in design_atom_array.get_annotation_categories()
            ):
                design_is_fixed_atom = ~design_atom_array.mpnn_designed_residue_mask
            else:
                design_is_fixed_atom = np.zeros(len(design_atom_array), dtype=bool)
            design_atom_array = design_atom_array[
                design_atom_array.atomize
                | design_is_backbone_atom
                | design_is_fixed_atom
            ]

            # Construct one letter sequence and recovery metrics for
            # output dict.
            one_letter_seq = "".join(
                [
                    DICT_THREE_TO_ONE.get(res_name, DICT_THREE_TO_ONE[UNKNOWN_AA])
                    for res_name in designed_resnames
                ]
            )
            sequence_recovery = float(sequence_recovery_per_design[design_idx])
            if interface_sequence_recovery_per_design is not None:
                ligand_interface_sequence_recovery = float(
                    interface_sequence_recovery_per_design[design_idx]
                )
            else:
                ligand_interface_sequence_recovery = None

            # Build the output dict.
            output_dict = {
                "batch_idx": batch_idx,
                "design_idx": design_idx,
                "designed_sequence": one_letter_seq,
                "sequence_recovery": sequence_recovery,
                "ligand_interface_sequence_recovery": (
                    ligand_interface_sequence_recovery
                ),
                "model_type": self.model_type,
                "checkpoint_path": self.checkpoint_path,
                "is_legacy_weights": self.is_legacy_weights,
            }

            outputs.append(
                MPNNInferenceOutput(
                    atom_array=design_atom_array,
                    output_dict=output_dict,
                    input_dict=copy.deepcopy(input_dict),
                )
            )

        return outputs

    def _write_outputs(self, results: list[MPNNInferenceOutput]) -> None:
        """Write CIF and/or FASTA outputs based on engine-level settings."""
        out_directory = self.out_directory

        # If no output directory and writing requested, raise error.
        if not out_directory and (self.write_fasta or self.write_structures):
            raise ValueError(
                "Output directory is not set, but writing of outputs was requested."
            )
        elif not out_directory:
            # Nothing to do.
            return

        # Make the output directory if it does not exist.
        out_dir_path = Path(out_directory)
        out_dir_path.mkdir(parents=True, exist_ok=True)

        if self.write_structures:
            # One CIF per design.
            for idx, result in enumerate(results):
                name = result.input_dict["name"]
                batch_idx = result.output_dict["batch_idx"]
                design_idx = result.output_dict["design_idx"]

                # Can't write without a name.
                if name is None:
                    raise ValueError(
                        f"Cannot write structure for result {idx}: 'name' is "
                        "not set in input_dict."
                    )

                # Construct the output file path.
                file_stem = f"{name}_b{batch_idx}_d{design_idx}"
                base_path = out_dir_path / file_stem

                # Use the MPNNInferenceOutput helper for writing.
                result.write_structure(
                    base_path=base_path,
                )

        # Write FASTA outputs if requested, one per input name.
        if self.write_fasta:
            # Group results by input name.
            grouped: dict[str, list[MPNNInferenceOutput]] = {}
            for result in results:
                name = result.input_dict["name"]

                # Can't write without a name.
                if name is None:
                    raise ValueError(
                        "Cannot write FASTA output: 'name' is not set in input_dict."
                    )

                if name not in grouped:
                    grouped[name] = []

                grouped[name].append(result)

            # Write one FASTA file per input name.
            for name, group in grouped.items():
                fasta_path = out_dir_path / f"{name}.fa"
                # Append mode so that multiple runs can accumulate designs.
                with fasta_path.open("a") as handle:
                    for result in group:
                        result.write_fasta(handle=handle)
