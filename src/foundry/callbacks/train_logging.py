import time
from collections import defaultdict

import pandas as pd
from atomworks.ml.example_id import parse_example_id
from beartype.typing import Any
from lightning.fabric.wrappers import (
    _FabricOptimizer,
)
from rf3.utils.loss import convert_batched_losses_to_list_of_dicts, mean_losses
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from torchmetrics.aggregation import MeanMetric

from foundry.callbacks.callback import BaseCallback
from foundry.utils.ddp import RankedLogger
from foundry.utils.logging import (
    print_df_as_table,
    print_model_parameters,
    safe_print,
    table_from_df,
)


class LogModelParametersCallback(BaseCallback):
    """Print a table of the total and trainable parameters of the model at the start of training."""

    def on_fit_start(self, trainer):
        print_model_parameters(trainer.state["model"])


class PrintExampleIDBeforeForwardPassCallback(BaseCallback):
    """Print the example ID for each rank at the start of the forward pass for each batch.

    WARNING: Spams the console. Use only for debugging purposes.
    """

    def __init__(self, rank_zero_only: bool = True):
        self.logger = RankedLogger(__name__, rank_zero_only=rank_zero_only)

    def on_train_batch_start(self, trainer, batch: Any, batch_idx: int):
        example_id = batch[0]["example_id"]

        # Prepare the formatted strings with colors
        rank_info = f"[grey]<Rank {trainer.fabric.global_rank}>[/grey]"
        epoch_batch_info = (
            f"[blue]Epoch {trainer.state['current_epoch']} Batch {batch_idx}[/blue]"
        )
        example_id_info = f"[bold yellow]Example ID: {example_id}[/bold yellow]"

        safe_print(
            f"{rank_info} {epoch_batch_info} - {example_id_info}",
            logger=self.logger,
        )


class LogDatasetSamplingRatiosCallback(BaseCallback):
    """Monitor the sampling ratios of the datasets and log after each epoch."""

    def on_fit_start(self, trainer):
        self.dataset_sampling_counts = defaultdict(int)

    def on_train_batch_start(self, trainer, batch, batch_idx):
        example_id = batch[0]["example_id"]

        if trainer.fabric.is_global_zero:
            dataset_string = "/".join(parse_example_id(example_id)["datasets"])
            self.dataset_sampling_counts[dataset_string] += 1

    def on_train_epoch_end(self, trainer):
        if trainer.fabric.is_global_zero:
            total_samples = sum(self.dataset_sampling_counts.values())

            data = {
                "Dataset": list(self.dataset_sampling_counts.keys()),
                "Count": list(self.dataset_sampling_counts.values()),
                "Percentage": [
                    f"{(count / total_samples) * 100:.2f}%"
                    for count in self.dataset_sampling_counts.values()
                ],
            }

            print_df_as_table(
                df=pd.DataFrame(data),
                title=f"Epoch {trainer.state['current_epoch']}: Dataset Sampling Ratios",
            )

            # Reset the counts for the next epoch
            self.dataset_sampling_counts.clear()


class LogLearningRateCallback(BaseCallback):
    """Monitor the learning rate of the optimizer

    Args:
        log_every_n: Log the learning rate every n optimizer steps.
    """

    def __init__(self, log_every_n: int):
        self.log_every_n = log_every_n

    def optimizer_step(self, trainer, optimizer: _FabricOptimizer):
        # Get the current global step
        current_step = trainer.state["global_step"]

        # Log the learning rate only every `log_every_n` steps
        if current_step % self.log_every_n == 0:
            trainer.fabric.log(
                "train/learning_rate",
                optimizer.param_groups[0]["lr"],
                step=current_step,
            )


class LogAF3TrainingLossesCallback(BaseCallback):
    """Log the primary model losses for AF3.

    Includes:
        - The mean training losses every `log_every_n` batches
        - The mean training losses at the end of each epoch
        - The time taken to complete each epoch
        - (Optionally) The full batch losses for each structure in the diffusion batch

    Args:
        log_every_n (int): Print the training loss after every n batches.
    """

    def __init__(
        self,
        log_full_batch_losses: bool = False,
        log_every_n: int = 10,
    ):
        """
        Args:
            log_full_batch_losses(bool): Log losses for every structure within the diffusion batch.
            log_every_n (int): Print the training loss after every n batches.
            console_width (int): Width of the console for printing.
        """
        self.log_every_n = log_every_n
        self.log_full_batch_losses = log_full_batch_losses

        self.start_time = None
        self.logger = RankedLogger(__name__, rank_zero_only=True)

        # This dict will store key -> MeanMetric() for each loss
        self.loss_trackers = {}

    def on_train_epoch_start(self, trainer):
        # Record the start time of the epoch
        self.start_time = time.time()

    def on_train_batch_end(self, trainer, outputs: Any, batch: Any, batch_idx: int):
        mean_loss_dict = {}
        if "loss_dict" in outputs:
            mean_loss_dict.update(mean_losses(outputs["loss_dict"]))

        for key, val in mean_loss_dict.items():
            if key not in self.loss_trackers:
                self.loss_trackers[key] = trainer.fabric.to_device(MeanMetric())
            self.loss_trackers[key].update(val)

        if trainer.fabric.is_global_zero and batch_idx % self.log_every_n == 0:
            # ... log losses for each structure in the batch
            if self.log_full_batch_losses:
                full_batch_loss_dicts = convert_batched_losses_to_list_of_dicts(
                    outputs["loss_dict"]
                )
                for loss_dict in full_batch_loss_dicts:
                    loss_dict = {
                        f"train/per_structure/{k}": v for k, v in loss_dict.items()
                    }
                    trainer.fabric.log_dict(
                        loss_dict, step=trainer.state["global_step"]
                    )

            # ... log losses meaned across the batch
            # (Prepend "train/batch_mean" to the keys in the loss dictionary)
            mean_loss_dict_for_logging = {
                f"train/batch_mean/{k}": v for k, v in mean_loss_dict.items()
            }
            trainer.fabric.log_dict(
                mean_loss_dict_for_logging, step=trainer.state["global_step"]
            )

            # ... print the mean losses in a table
            df_losses = pd.DataFrame(
                {
                    "Train Loss Name": [
                        k.replace("_", " ").title() for k in mean_loss_dict.keys()
                    ],
                    "Value": [v for v in mean_loss_dict.values()],
                }
            )
            table = table_from_df(df_losses, title="Training Losses")

            # (percentage of batch count)
            percentage_complete = (batch_idx / trainer.n_batches_per_epoch) * 100

            # Simple progress bar using Unicode blocks
            progress_bar_length = 10  # Length of the progress bar
            filled_length = int(progress_bar_length * percentage_complete // 100)
            progress_bar = "█" * filled_length + "░" * (
                progress_bar_length - filled_length
            )
            percentage_str = f"[bold magenta]{percentage_complete:.2f}%[/bold magenta]"

            # Create a panel for the epoch and batch info with a progress bar
            epoch_batch_info = (
                f"[grey]<Rank {trainer.fabric.global_rank}>[/grey] "
                f"Epoch {trainer.state['current_epoch']} Batch {batch_idx} "
                f"[{progress_bar}] {percentage_str}"
            )

            epoch_batch_panel = Panel(
                epoch_batch_info,
                border_style="bold blue",
            )

            # Create a panel for the example ID
            example_id = batch[0]["example_id"]
            example_id_str = f"[bold yellow]{example_id}[/bold yellow]"
            example_id_panel = Panel(
                example_id_str,
                border_style="bold green",
            )

            # Combine all components vertically
            combined_content = Group(epoch_batch_panel, example_id_panel, table)

            safe_print(combined_content)

    def on_train_epoch_end(self, trainer):
        # Gather final epoch means (must be run on all ranks)
        final_means = {
            k: tracker.compute().item() for k, tracker in self.loss_trackers.items()
        }

        # Calculate elapsed time and number of batches (from the total_loss tracker, if available)
        elapsed_time = time.time() - self.start_time
        num_batches = (
            self.loss_trackers["total_loss"].update_count
            if "total_loss" in self.loss_trackers
            else trainer.n_batches_per_epoch
        )

        if trainer.fabric.is_global_zero:
            # Create a summary table
            table = Table(
                title=f"Epoch {trainer.state['current_epoch']} Summary",
                show_header=False,
                header_style="bold magenta",
            )
            table.add_column("Loss Name", style="bold cyan", justify="left")
            table.add_column("Value", style="green", justify="right")

            for k, v in final_means.items():
                table.add_row(f"<Train> Mean {k}", f"{v:.4f}")

            table.add_section()
            table.add_row("Total Optimizer Steps", str(trainer.state["global_step"]))
            table.add_row("Number of Batches", str(num_batches))
            table.add_row("Elapsed Time (s)", f"{elapsed_time:.2f}")
            table.add_row(
                "Mean Time per Batch (s)", f"{elapsed_time / num_batches:.2f}"
            )

            safe_print(table)

        # Log these final epoch means (prepend "train/per_epoch_" to each key)
        trainer.fabric.log_dict(
            {f"train/per_epoch_{k}": v for k, v in final_means.items()},
            step=trainer.state["current_epoch"],
        )

        # Reset the trackers for the next epoch
        for metric in self.loss_trackers.values():
            metric.reset()
