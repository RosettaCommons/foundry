import os
from copy import deepcopy
from pathlib import Path

import pandas as pd
from atomworks.ml.utils import nested_dict
from beartype.typing import Any, Literal
from omegaconf import ListConfig

from foundry.callbacks.callback import BaseCallback
from foundry.utils.ddp import RankedLogger

ranked_logger = RankedLogger(__name__, rank_zero_only=True)


class StoreValidationMetricsInDFCallback(BaseCallback):
    """Saves the validation outputs in a DataFrame for each rank and concatenates them at the end of the validation epoch."""

    def __init__(
        self,
        save_dir: os.PathLike,
        metrics_to_save: list[str] | Literal["all"] = "all",
    ):
        self.save_dir = Path(save_dir)
        self.metrics_to_save = metrics_to_save

    def _save_dataframe_for_rank(self, rank: int, epoch: int):
        """Saves per-GPU output dataframe of metrics to a rank-specific CSV."""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        file_path = self.save_dir / f"validation_output_rank_{rank}_epoch_{epoch}.csv"

        # Flush explicitly to ensure the file is written to disk
        with open(file_path, "w") as f:
            self.per_gpu_outputs_df.to_csv(f, index=False)
            f.flush()
            os.fsync(f.fileno())

        ranked_logger.info(
            f"Saved validation outputs to {file_path} for rank {rank}, epoch {epoch}"
        )

    def on_validation_epoch_start(self, trainer):
        self.per_gpu_outputs_df = pd.DataFrame()

    def on_validation_batch_end(
        self,
        trainer,
        outputs: dict,
        batch: Any,
        batch_idx: int,
        num_batches: int,
        dataset_name: str | None = None,
    ):
        """Build a flattened DataFrame from the metrics output and accumulate with the prior batches"""
        assert "metrics_output" in outputs, "Validation outputs must contain metrics."
        metrics_output = deepcopy(outputs["metrics_output"])

        # ... assemble a flat DataFrame from the metrics output
        example_id = metrics_output.pop("example_id")
        metrics_as_list_of_dicts = []

        # ... remove metrics that are not in the save list
        if self.metrics_to_save != "all" and isinstance(
            self.metrics_to_save, list | ListConfig
        ):
            metrics_output = {
                k: v
                for k, v in metrics_output.items()
                if any(k.startswith(prefix) for prefix in self.metrics_to_save)
            }

        def _build_row_from_flattened_dict(
            dict_to_flatten: dict, prefix: str, example_id: str
        ):
            """Helper function to build a DataFrame row"""
            flattened_dict = nested_dict.flatten(dict_to_flatten, fuse_keys=".")
            row_data = {"example_id": example_id}
            for sub_k, sub_v in flattened_dict.items():
                # Convert lists to tuples so that they are hashable
                if isinstance(sub_v, list):
                    sub_v = tuple(sub_v)
                row_data[f"{prefix}.{sub_k}"] = sub_v
            return row_data

        scalar_metrics = {"example_id": example_id}
        for key, value in metrics_output.items():
            if isinstance(value, dict):
                # Flatten once for this dict => 1 row.
                metrics_as_list_of_dicts.append(
                    _build_row_from_flattened_dict(value, key, example_id)
                )
            elif isinstance(value, list) and all(isinstance(x, dict) for x in value):
                # Flatten each dict in the list => multiple rows.
                for subdict in value:
                    metrics_as_list_of_dicts.append(
                        _build_row_from_flattened_dict(subdict, key, example_id)
                    )
            else:
                # Scalar (string, float, int, or list that isn't list-of-dicts)
                assert key not in scalar_metrics, f"Duplicate key: {key}"
                scalar_metrics[key] = value

        metrics_as_list_of_dicts.append(scalar_metrics)

        # ... convert the list of dicts to a DataFrame and add epoch and dataset columns
        batch_df = pd.DataFrame(metrics_as_list_of_dicts)
        batch_df["epoch"] = trainer.state["current_epoch"]
        batch_df["dataset"] = dataset_name

        # Assert no duplicate rows
        assert (
            batch_df.duplicated().sum() == 0
        ), "Duplicate rows found in the metrics DataFrame!"

        # Accumulate into the per-rank DataFrame
        self.per_gpu_outputs_df = pd.concat(
            [self.per_gpu_outputs_df, batch_df], ignore_index=True
        )

        ranked_logger.info(
            f"Validation Progress: {100 * (batch_idx + 1) / num_batches:.0f}% for {dataset_name}"
        )

    def on_validation_epoch_end(self, trainer):
        """Aggregate and log the validation metrics at the end of the epoch.

        Each rank writes out its partial CSV. Then rank 0 aggregates them, logs grouped metrics by dataset,
        and appends them to a master file containing data from all epochs.
        """

        #  ... write out partial CSV for this rank
        rank = trainer.fabric.global_rank
        epoch = trainer.state["current_epoch"]
        self._save_dataframe_for_rank(rank, epoch)

        # Synchronize all processes
        ranked_logger.info(
            "Synchronizing all processes before concatenating DataFrames..."
        )
        trainer.fabric.barrier()

        # Only rank 0 loads and concatenates the DataFrames
        ranked_logger.info("Loading and concatenating DataFrames...")
        if trainer.fabric.is_global_zero:
            # ... load all partial CSVs
            merged_df = self._load_and_concatenate_csvs(epoch)

            # ... append to master CSV for all epochs
            master_path = self.save_dir / "validation_output_all_epochs.csv"
            if master_path.exists():
                old_df = pd.read_csv(master_path)
                merged_df = pd.concat(
                    [old_df, merged_df], ignore_index=True, sort=False
                )
            merged_df.to_csv(master_path, index=False)
            ranked_logger.info(f"Appended epoch={epoch} results to {master_path}")

            # Store the path to the master CSV in the Trainer
            trainer.validation_results_path = master_path

            # Cleanup
            self._cleanup_temp_files()

    def _load_and_concatenate_csvs(self, epoch: int) -> pd.DataFrame:
        """Load rank-specific CSVs for the given epoch and concatenate them without duplicating examples."""
        pattern = f"validation_output_rank_*_epoch_{epoch}.csv"
        files = list(self.save_dir.glob(pattern))

        # Track which example_id + dataset combinations we've already seen
        seen_examples = set()
        final_dataframes = []

        for f in files:
            try:
                df = pd.read_csv(f)

                # Create a filter for rows with new example_id + dataset combinations
                if not df.empty:
                    # Create a unique identifier for each example_id + dataset combination
                    df["_example_key"] = (
                        df["example_id"].astype(str) + "|" + df["dataset"].astype(str)
                    )

                    # Filter out rows with example_id + dataset combinations we've already seen
                    new_examples_mask = ~df["_example_key"].isin(seen_examples)

                    # If there are any new examples, add them to our final list
                    if new_examples_mask.any():
                        new_examples_df = df[new_examples_mask].copy()

                        # Update our set of seen examples
                        seen_examples.update(new_examples_df["_example_key"].tolist())

                        # Remove the temporary column before adding to final list
                        new_examples_df.drop("_example_key", axis=1, inplace=True)
                        final_dataframes.append(new_examples_df)

            except pd.errors.EmptyDataError:
                ranked_logger.warning(f"Skipping empty CSV: {f}")

        # Concatenate dataframes, filling missing columns with NaN
        return pd.concat(final_dataframes, axis=0, ignore_index=True, sort=False)

    def _cleanup_temp_files(self):
        """Remove temporary files used to store individual rank outputs."""
        all_files = list(self.save_dir.rglob("validation_output_rank_*_epoch_*.csv"))
        for file in all_files:
            try:
                file.unlink()  # Remove the file
            except Exception as e:
                ranked_logger.warning(f"Failed to delete file {file}: {e}")
