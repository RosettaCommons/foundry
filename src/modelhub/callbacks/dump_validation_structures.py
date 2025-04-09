from beartype.typing import Any
from modelhub.callbacks.base import BaseCallback
from modelhub.utils.io import (
    dump_structures,
    dump_trajectories,
    build_stack_from_atom_array_and_batched_coords,
)
from datahub.common import parse_example_id
from pathlib import Path
from os import PathLike
import functools
from modelhub.utils.ddp import RankedLogger  # noqa

ranked_logger = RankedLogger(__name__, rank_zero_only=True)


class DumpValidationStructuresCallback(BaseCallback):
    """Dump predicted structures and/or diffusion trajectories during validation"""

    def __init__(
        self,
        save_dir: PathLike,
        dump_predictions: bool = False,
        one_model_per_file: bool = False,
        dump_trajectories: bool = False,
        dump_denoised_trajectories_only: bool = False,
        dump_every_n: int = 1,
    ):
        """
        Args:
            dump_predictions: Whether to dump structures (CIF files) after validation batches.
            one_model_per_file: If True, write each structure within a diffusion batch to its own CIF files. If False,
                include each structure within a diffusion batch as a separate model within one CIF file.
            dump_trajectories: Whether to dump denoising trajectories after validation batches.
        """
        super().__init__()
        self.save_dir = Path(save_dir)
        self.dump_predictions = dump_predictions
        self.dump_trajectories = dump_trajectories
        self.one_model_per_file = one_model_per_file
        self.dump_denoised_trajectories_only = dump_denoised_trajectories_only
        self.dump_every_n = dump_every_n
    
    def _build_path_from_example_id(self, example_id, dir: str, extra: str = "", epoch: str = None, dataset_name: str = '') -> Path:

        try:
            # ... try to extract the PDB ID and assembly ID from the example ID
            parsed_id = parse_example_id(example_id)
            identifier = f"{parsed_id['pdb_id']}_{parsed_id['assembly_id']}"
        except (KeyError, ValueError):
            # ... if parsing fails, fall back to the original example ID
            identifier = example_id

        # ... parse the example_id into a dictionary of components
        epoch_str = 'epoch_{}'.format(epoch) if epoch else ''
        """Helper function to build a path from a training or validation example_id."""
        return (
            self.save_dir
            / dir
            / f"{epoch_str}"
            / dataset_name
            / f"{identifier}{extra}"
        )

    def on_validation_batch_end(
        self,
        *,
        outputs: dict,
        trainer: Any,
        batch: Any,
        dataset_name: str,
        **kwargs,
    ):
        if (not self.dump_predictions) and (not self.dump_trajectories):
            return  # Nothing to do
        
        if trainer.state["global_step"] % self.dump_every_n != 0:
            ranked_logger.debug(f"Skipping validation batch dump at step {trainer.state['global_step']} (not every {self.dump_every_n} steps)")
            return

        assert (
            "network_output" in outputs
        ), "Validation outputs must contain `network_output` to dump structures!"

        network_output = outputs["network_output"]
        example = batch[0]  # Assume batch size = 1

        _build_path_from_example_id = functools.partial(self._build_path_from_example_id,
            example_id=example["example_id"],
            epoch=trainer.state['current_epoch'],
            dataset_name=dataset_name,
        )

        if self.dump_predictions:
            atom_array_stack = build_stack_from_atom_array_and_batched_coords(
                network_output["X_L"], example["atom_array"]
            )
            f=_build_path_from_example_id(dir="predictions")
            dump_structures(
                atom_arrays=atom_array_stack,
                base_path=f,
                one_model_per_file=self.one_model_per_file,
            )
            ranked_logger.info(f'Dumped validation predictions to {f}')
    
        if self.dump_trajectories:
            dump_trajectories(
                trajectory_list=network_output["X_denoised_L_traj"],
                atom_array=example["atom_array"],
                base_path=_build_path_from_example_id(dir="trajectories", extra="_denoised"),
            )
            if not self.dump_denoised_trajectories_only:
                dump_trajectories(
                    trajectory_list=network_output["X_noisy_L_traj"],
                    atom_array=example["atom_array"],
                    base_path=_build_path_from_example_id(dir="trajectories", extra="_noisy"),
                )