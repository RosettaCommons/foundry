import functools
from os import PathLike
from pathlib import Path

from atomworks.ml.example_id import parse_example_id
from beartype.typing import Any, Literal
from rfd3.utils.io import (
    dump_metadata,
    dump_structures,
    dump_trajectories,
)

from foundry.callbacks.callback import BaseCallback
from foundry.utils.ddp import RankedLogger

ranked_logger = RankedLogger(__name__, rank_zero_only=True)


class DumpValidationStructuresCallback(BaseCallback):
    """Dump predicted structures and/or diffusion trajectories during validation"""

    def __init__(
        self,
        save_dir: PathLike,
        dump_predictions: bool = False,
        dump_prediction_metadata_json: bool = True,
        one_model_per_file: bool = False,
        dump_trajectories: bool = False,
        dump_denoised_trajectories_only: bool = False,
        dump_every_n: int = 1,
        align_trajectories: bool = True,
        verbose: bool = True,
        extra_fields: list[str] | Literal["all"] = [],
    ):
        """
        Args:
            dump_predictions: Whether to dump structures (CIF files) after validation batches.
            one_model_per_file: If True, write each structure within a diffusion batch to its own CIF files. If False,
                include each structure within a diffusion batch as a separate model within one CIF file.
            dump_trajectories: Whether to dump denoising trajectories after validation batches.
            dump_every_n: Dump validation structures every n epochs.
            extra_fields: List of extra annotations to include in the saved CIF file.
        """
        super().__init__()
        self.save_dir = Path(save_dir)
        self.dump_predictions = dump_predictions
        self.dump_prediction_metadata_json = dump_prediction_metadata_json
        self.dump_trajectories = dump_trajectories
        self.one_model_per_file = one_model_per_file
        self.dump_denoised_trajectories_only = dump_denoised_trajectories_only
        self.dump_every_n = dump_every_n
        self.align_trajectories = align_trajectories
        self.verbose = verbose
        self.extra_fields = extra_fields

    def _build_path_from_example_id(
        self,
        example_id,
        dir: str,
        extra: str = "",
        epoch: str = None,
        dataset_name: str = "",
    ) -> Path:
        """Helper function to build a path from a training or validation example_id."""
        try:
            # ... try to extract the PDB ID and assembly ID from the example ID
            parsed_id = parse_example_id(example_id)
            identifier = f"{parsed_id['pdb_id']}_{parsed_id['assembly_id']}"
        except (KeyError, ValueError):
            # ... if parsing fails, fall back to the original example ID
            identifier = example_id

        # ... parse the example_id into a dictionary of components
        epoch_str = "epoch_{}".format(epoch) if epoch else ""
        return (
            self.save_dir / dir / f"{epoch_str}" / dataset_name / f"{identifier}{extra}"
        )

    def on_validation_batch_end(
        self,
        *,
        trainer,
        outputs: dict,
        batch: Any,
        dataset_name: str,
        **_,
    ):
        if (not self.dump_predictions) and (not self.dump_trajectories):
            return  # Nothing to do
        current_epoch = trainer.state["current_epoch"]
        if current_epoch % self.dump_every_n != 0:
            ranked_logger.debug(
                f"Skipping validation batch dump at step {current_epoch} (not every {self.dump_every_n} epochs)"
            )
            return

        assert (
            "network_output" in outputs
        ), "Validation outputs must contain `network_output` to dump structures!"

        network_output = outputs["network_output"]
        atom_array_stack = outputs["predicted_atom_array_stack"]
        prediction_metadata = outputs["prediction_metadata"]
        example = batch[0]  # Assume batch size = 1

        _build_path_from_example_id = functools.partial(
            self._build_path_from_example_id,
            example_id=example["example_id"],
            epoch=current_epoch,
            dataset_name=dataset_name,
        )

        if self.dump_predictions:
            dump_structures(
                atom_arrays=atom_array_stack,
                base_path=(f := _build_path_from_example_id(dir="predictions")),
                one_model_per_file=self.one_model_per_file,
                extra_fields=self.extra_fields,
            )
            if self.verbose:
                ranked_logger.info(f"Dumped validation predictions to {f}")

        if self.dump_prediction_metadata_json and prediction_metadata:
            dump_metadata(
                prediction_metadata=prediction_metadata,
                base_path=(f := _build_path_from_example_id(dir="predictions")),
                one_model_per_file=self.one_model_per_file,
            )
            if self.verbose:
                ranked_logger.info(f"Dumped validation metadata to {f}")

        if self.dump_trajectories:
            dump_trajectories(
                trajectory_list=network_output["X_denoised_L_traj"],
                atom_array=example["atom_array"],
                base_path=(
                    f := _build_path_from_example_id(
                        dir="trajectories", extra="_denoised"
                    )
                ),
                align_structures=self.align_trajectories,
            )
            if not self.dump_denoised_trajectories_only:
                dump_trajectories(
                    trajectory_list=network_output["X_noisy_L_traj"],
                    atom_array=example["atom_array"],
                    base_path=_build_path_from_example_id(
                        dir="trajectories", extra="_noisy"
                    ),
                    align_structures=self.align_trajectories,
                )

            if self.verbose:
                ranked_logger.info(f"Dumped validation trajectories to {f}")
