import os
import subprocess
from os import PathLike
from pathlib import Path

import pandas as pd
from beartype.typing import Any

from modelhub.callbacks.callback import BaseCallback
from modelhub.utils.ddp import RankedLogger
from modelhub.utils.logging import print_df_as_table

ranked_logger = RankedLogger(__name__, rank_zero_only=True)


class LogPipelinesResultsCallback(BaseCallback):
    """Checks for and logs the results of any pipelines runs."""

    def __init__(self, metrics_save_dir: PathLike):
        """
        Args:
            metrics_save_dir (PathLike): Directory where the pipelines validation metrics are saved.
        """
        super().__init__()
        self.path_to_benchmark = Path(metrics_save_dir) / "aa_design_validation.csv"
        self.last_benchmarked_epoch = None

    def on_train_epoch_end(self, *, trainer: Any, **kwargs):
        if not trainer.fabric.is_global_zero:
            return
        if not self.path_to_benchmark.exists():
            ranked_logger.info(
                f"Validation benchmark file not found at {self.path_to_benchmark}."
            )
            return

        # Load the benchmark file
        benchmark_df = pd.read_csv(self.path_to_benchmark)
        if self.last_benchmarked_epoch is not None:
            new_benchmark_rows = benchmark_df[
                benchmark_df["epoch"] > self.last_benchmarked_epoch
            ]
        else:
            new_benchmark_rows = benchmark_df

        # Sort for clarity of logging
        new_benchmark_rows = new_benchmark_rows.sort_values(by="epoch", ascending=True)

        # Record which rows have already been logged
        self.last_benchmarked_epoch = benchmark_df["epoch"].max()

        # Log to terminal
        ranked_logger.info(
            "Found new results from pipeline runs launched during earlier validation epoch(s)."
        )

        # Log with fabric
        if trainer.fabric:
            numeric_cols = new_benchmark_rows.select_dtypes(include="number").columns
            for _, row in new_benchmark_rows.iterrows():
                for col in numeric_cols:
                    trainer.fabric.log_dict(
                        {f"val/{row['dataset']}/{col}": row[col]},
                        step=row["epoch"],
                    )


class RunPipelinesCallback(BaseCallback):
    """Runs the designs output during validation through a computational design pipeline, using the `pipelines` repo."""

    def __init__(
        self,
        pipelines_config_path: PathLike,
        pipelines_script_path: PathLike,
        save_dir: PathLike,
        run_every_n_epochs: int = 10,
    ):
        """
        Args:
            pipelines_config_path (PathLike): Path to the pipelines config file.
            pipelines_script_path (PathLike): Path to the pipeline.py entry-point script.
            save_dir (PathLike): Base directory where validation results are saved.
            run_every_n_epochs (int): Frequency of running the pipelines.
        """
        super().__init__()
        self.run_every_n_epochs = run_every_n_epochs
        self.save_dir = Path(save_dir)
        self.pipelines_config_dir = Path(pipelines_config_path).parent
        self.pipelines_config_name = Path(pipelines_config_path).stem
        self.pipelines_script_path = Path(pipelines_script_path)

    def on_validation_epoch_end(self, *, trainer: Any, **kwargs):
        # Only run pipelines if this is the global zero rank
        if not trainer.fabric.is_global_zero:
            return

        # Only run pipelines at the frequency specified
        current_epoch = trainer.state["current_epoch"]
        if current_epoch % self.run_every_n_epochs != 0:
            return

        preds_parent_dir = (
            self.save_dir / "val_structures" / "predictions" / f"epoch_{current_epoch}"
        )

        if not preds_parent_dir.exists():
            ranked_logger.warning(
                f"Predictions directory not found at {preds_parent_dir}. Please ensure that the RunPipelinesCallback is "
                f"only run on validation epochs where the DumpDesignStructuresValidationCallback is also run."
            )
            return

        for preds_dir in preds_parent_dir.iterdir():
            # Get relevant paths
            dataset_name = preds_dir.name
            model_name = self.save_dir.parent.name
            rundir = (
                self.save_dir / "pipelines" / f"epoch_{current_epoch}" / dataset_name
            )
            rundir.mkdir(parents=True, exist_ok=True)

            log_path = rundir / "initial_pipelines_slurm_submission.log"

            # Assemble the subprocess command
            pipelines_cmd = (
                f"{self.pipelines_script_path} --config-path={self.pipelines_config_dir} --config-name={self.pipelines_config_name} "
                f"rundir={rundir} link_from_dir.dir={preds_dir} update_benchmark.additional_columns.dataset={dataset_name} "
                f"update_benchmark.additional_columns.epoch={current_epoch} update_benchmark.additional_columns.model={model_name}"
            )

            cmd_sbatch = (
                f'sbatch --wrap "{pipelines_cmd}" -p cpu -c 1 --mem "8g" -J val_pipeline_epoch_{current_epoch} '
                f"-o {log_path} --export PYTHONPATH={os.environ['PYTHONPATH']}"
            )

            print(f"Running the following command: {cmd_sbatch}")
            subprocess.run(cmd_sbatch, shell=True)


class LogDesignValidationMetricsCallback(BaseCallback):
    def on_validation_epoch_end(self, trainer: Any):
        # Only log metrics to disk if this is the global zero rank
        if not trainer.fabric.is_global_zero:
            return

        assert hasattr(
            trainer, "validation_results_path"
        ), "Results path not found! Ensure that StoreValidationMetricsInDFCallback is called first."
        df = pd.read_csv(trainer.validation_results_path)

        # ... filter to most recent epoch, drop epoch column
        df = df[df["epoch"] == df["epoch"].max()]
        df.drop(columns=["epoch"], inplace=True)

        for dataset in df["dataset"].unique():
            dataset_df = df[df["dataset"] == dataset].copy()
            dataset_df.drop(columns=["dataset"], inplace=True)

            print(f"\n+{' ' + dataset + ' ':-^150}+\n")

            remaining_cols = [
                col for col in dataset_df.columns if col not in ["example_id"]
            ]
            remaining_df = dataset_df[remaining_cols].copy()
            remaining_df = remaining_df.dropna(how="all")
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
                trainer.fabric.log_dict(
                    {f"val/{dataset}/{col}": final_means[col] for col in numeric_cols},
                    step=trainer.state["current_epoch"],
                )

                if len(dataset_df["example_id"].unique()) <= 25:
                    for eid, df_ in dataset_df.groupby("example_id"):
                        df_ = df_[numeric_cols].mean()
                        trainer.fabric.log_dict(
                            {
                                f"val/{dataset}/{col}/{eid}": df_[col]
                                for col in numeric_cols
                            },
                            step=trainer.state["current_epoch"],
                        )


class LogTrainVariableCallback(BaseCallback):
    """Callback to log specified training variables during training."""

    def __init__(self, log_every_n: int = 10, variables: list[str] = None):
        """
        Args:
            log_every_n (int): Log stats every N batches
            variables (list[str]): List of variable names to track and log
        """
        super().__init__()
        self.log_every_n = log_every_n
        self.stats_to_log = variables or []

    def on_train_batch_end(self, batch: Any, batch_idx: int, trainer: Any, **_):
        # Initialize accumulators for each stat
        stat_totals = {stat: 0 for stat in self.stats_to_log}
        valid_samples = 0

        for sample in batch:
            logs = sample.get("log_dict", {})
            # log the information that are in stats_to_log
            for stat in self.stats_to_log:
                if stat in logs:
                    stat_totals[stat] += logs[stat]
                    valid_samples += 1

        if valid_samples > 0 and batch_idx % self.log_every_n == 0:
            # Calculate averages and prepare logging dict
            log_dict = {
                f"train/{stat}_avg": stat_totals[stat] / valid_samples
                for stat in self.stats_to_log
            }

            trainer.fabric.log_dict(
                log_dict,
                step=trainer.state["current_epoch"],
            )
