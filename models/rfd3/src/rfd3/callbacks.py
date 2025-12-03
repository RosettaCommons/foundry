import pandas as pd
from beartype.typing import Any

from foundry.callbacks.callback import BaseCallback
from foundry.utils.ddp import RankedLogger
from foundry.utils.logging import print_df_as_table

ranked_logger = RankedLogger(__name__, rank_zero_only=True)


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
