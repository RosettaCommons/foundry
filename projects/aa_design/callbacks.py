
import functools
from pathlib import Path

import pandas as pd
import torch
from beartype.typing import Any

from modelhub.alignment import weighted_rigid_align
from modelhub.callbacks.base import BaseCallback
from modelhub.callbacks.dump_validation_structures import (
    DumpValidationStructuresCallback,
)
from modelhub.utils.ddp import RankedLogger  # noqa
from modelhub.utils.io import (
    build_stack_from_atom_array_and_batched_coords,
    dump_structures,
    dump_trajectories,
)
from modelhub.utils.logging import print_df_as_table

ranked_logger = RankedLogger(__name__, rank_zero_only=True)


class DumpDesignStructuresValidationCallback(DumpValidationStructuresCallback):
    """Handles designs where example id may not be formatted as a conventional PDB ID."""

    def _build_path_from_example_id(self, example_id, dir: str, extra: str = "", epoch: str = None, dataset_name: str = '') -> Path:
        """Helper function to build a path from a training or validation example_id."""
        f = (
            self.save_dir / dir / f"epoch_{epoch}" / dataset_name / f"{example_id.replace('-', '_')}{extra}"
        )
        return f

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

            # Alignment of trajectories to original motif
            trajectories_list = network_output["X_denoised_L_traj"]
            coord_atom_lvl_to_be_noised = example['coord_atom_lvl_to_be_noised']
            is_motif_atom_with_fixed_pos = example['feats']['is_motif_atom_with_fixed_pos']
            if (coord_atom_lvl_to_be_noised is not None and 
                is_motif_atom_with_fixed_pos is not None and
                torch.any(is_motif_atom_with_fixed_pos) 
            ):
                for step in range(len(trajectories_list)):
                    trajectories_list[step] = weighted_rigid_align(
                        X_L=coord_atom_lvl_to_be_noised,  # target
                        X_gt_L=trajectories_list[step],  # mobile
                        X_exists_L=is_motif_atom_with_fixed_pos,  # mask for target
                    )

            dump_trajectories(
                trajectory_list=trajectories_list,
                atom_array=example["atom_array"],
                base_path=_build_path_from_example_id(dir="trajectories", extra="_denoised"),
                align_structures=False
            )
            if not self.dump_denoised_trajectories_only:
                dump_trajectories(
                    trajectory_list=network_output["X_noisy_L_traj"],
                    atom_array=example["atom_array"],
                    base_path=_build_path_from_example_id(dir="trajectories", extra="_noisy"),
                    align_structures=False
                )

class LogDesignValidationMetricsCallback(BaseCallback):
    def on_validation_epoch_end(self, trainer: Any):
        # Only log metrics to disk if this is the global zero rank
        if not trainer.fabric.is_global_zero:
            return

        assert hasattr(trainer, "validation_results_path"), "Results path not found! Ensure that StoreValidationMetricsInDFCallback is called first."
        df = pd.read_csv(trainer.validation_results_path)

        # ... filter to most recent epoch, drop epoch column
        df = df[df["epoch"] == df["epoch"].max()]
        df.drop(columns=["epoch"], inplace=True)

        for dataset in df["dataset"].unique():
            dataset_df = df[df["dataset"] == dataset].copy()
            dataset_df.drop(columns=["dataset"], inplace=True)

            print(f"\n+{' ' + dataset + ' ':-^150}+\n")

            remaining_cols = [col for col in dataset_df.columns if col not in ['example_id']]
            remaining_df = dataset_df[remaining_cols].copy()
            remaining_df = remaining_df.dropna(how='all')
            numeric_cols = remaining_df.select_dtypes(include="number").columns

            # Compute means and non-NaN counts for numeric columns
            final_means = remaining_df[numeric_cols].mean()
            non_nan_counts = remaining_df[numeric_cols].count()

            # Convert the Series to a DataFrame and add the count as a new column
            final_means_df = final_means.to_frame(name="mean")
            final_means_df["Count"] = non_nan_counts

            print_df_as_table(
                final_means_df.reset_index(),
                f"{dataset} — {trainer.state['current_epoch']} — Design Validation Metrics",
            )
            if trainer.fabric:
                trainer.fabric.log_dict({
                    f"val/{dataset}/{col}": final_means[col] for col in numeric_cols
                }, step=trainer.state["current_epoch"])

                if len(dataset_df['example_id'].unique()) <= 25:
                    for eid, df_ in dataset_df.groupby('example_id'):
                        df_ = df_[numeric_cols].mean()
                        trainer.fabric.log_dict({
                            f"val/{dataset}/{col}/{eid}": df_[col] for col in numeric_cols
                        }, step=trainer.state["current_epoch"])
