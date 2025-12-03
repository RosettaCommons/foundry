from os import PathLike
from pathlib import Path

from atomworks.ml.example_id import parse_example_id
from beartype.typing import Any
from rf3.utils.io import (
    build_stack_from_atom_array_and_batched_coords,
    dump_structures,
    dump_trajectories,
)

from foundry.callbacks.callback import BaseCallback


class DumpValidationStructuresCallback(BaseCallback):
    """Dump predicted structures and/or diffusion trajectories during validation"""

    def __init__(
        self,
        save_dir: PathLike,
        dump_predictions: bool = False,
        one_model_per_file: bool = False,
        dump_trajectories: bool = False,
        compress_outputs: bool = False,
    ):
        """
        Args:
            dump_predictions: Whether to dump structures (CIF files) after validation batches.
            one_model_per_file: If True, write each structure within a diffusion batch to its own CIF files. If False,
                include each structure within a diffusion batch as a separate model within one CIF file.
            dump_trajectories: Whether to dump denoising trajectories after validation batches.
            compress_outputs: Whether to gzip output files. Defaults to ``False``.
        """
        super().__init__()
        self.save_dir = Path(save_dir)
        self.dump_predictions = dump_predictions
        self.dump_trajectories = dump_trajectories
        self.one_model_per_file = one_model_per_file
        self.compress_outputs = compress_outputs

    def on_validation_batch_end(
        self,
        trainer,
        outputs: dict,
        batch: Any,
        dataset_name: str | None,
        **_,
    ):
        if (not self.dump_predictions) and (not self.dump_trajectories):
            return  # Nothing to do

        assert (
            "network_output" in outputs
        ), "Validation outputs must contain `network_output` to dump structures!"

        network_output = outputs["network_output"]
        example = batch[0]  # Assume batch size = 1

        try:
            # ... try to extract the PDB ID and assembly ID from the example ID
            parsed_id = parse_example_id(example["example_id"])
            identifier = f"{parsed_id['pdb_id']}_{parsed_id['assembly_id']}"
        except (KeyError, ValueError):
            # ... if parsing fails, fall back to the original example ID
            identifier = example["example_id"]

        def _build_path_from_example_id(dir: str, extra: str = "") -> Path:
            """Helper function to build a path from a training or validation example_id."""
            path = self.save_dir / dir / f"epoch_{trainer.state['current_epoch']}"

            path = path / dataset_name

            return path / f"{identifier}{extra}"

        # Determine file type based on compression setting
        file_type = "cif.gz" if self.compress_outputs else "cif"

        if self.dump_predictions:
            atom_array_stack = build_stack_from_atom_array_and_batched_coords(
                network_output["X_L"], example["atom_array"]
            )
            dump_structures(
                atom_arrays=atom_array_stack,
                base_path=_build_path_from_example_id("predictions"),
                one_model_per_file=self.one_model_per_file,
                file_type=file_type,
            )

        if self.dump_trajectories:
            dump_trajectories(
                trajectory_list=network_output["X_denoised_L_traj"],
                atom_array=example["atom_array"],
                base_path=_build_path_from_example_id("trajectories", "_denoised"),
                file_type=file_type,
            )
            dump_trajectories(
                trajectory_list=network_output["X_noisy_L_traj"],
                atom_array=example["atom_array"],
                base_path=_build_path_from_example_id("trajectories", "_noisy"),
                file_type=file_type,
            )
