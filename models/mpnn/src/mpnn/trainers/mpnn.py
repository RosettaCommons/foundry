import torch
from beartype.typing import Any
from lightning_utilities import apply_to_collection
from mpnn.loss.nll_loss import LabelSmoothedNLLLoss
from mpnn.metrics.nll import NLL, InterfaceNLL
from mpnn.metrics.sequence_recovery import (
    InterfaceSequenceRecovery,
    SequenceRecovery,
)
from mpnn.model.mpnn import LigandMPNN, ProteinMPNN
from omegaconf import DictConfig

from foundry.metrics.metric import MetricManager
from foundry.trainers.fabric import FabricTrainer
from foundry.utils.ddp import RankedLogger
from foundry.utils.torch import assert_no_nans

ranked_logger = RankedLogger(__name__, rank_zero_only=True)


class MPNNTrainer(FabricTrainer):
    """Standard Trainer for MPNN-style models"""

    def __init__(
        self,
        *,
        model_type: str,
        loss: DictConfig | dict | None = None,
        metrics: DictConfig | dict | None = None,
        **kwargs,
    ):
        """
        See `FabricTrainer` for the additional initialization arguments.

        Args:
            model_type (str): Type of model to use ("protein_mpnn" or
                "ligand_mpnn").
            loss (DictConfig | dict | None): Configuration for the loss
                function. If None, default parameters will be used.
            metrics (DictConfig | dict | None): Configuration for the metrics.
                Ignored - metrics are hard-coded.
        """
        super().__init__(**kwargs)

        self.model_type = model_type

        # Metrics
        metrics = {
            "nll": NLL(),
            "sequence_recovery": SequenceRecovery(),
        }
        if self.model_type == "ligand_mpnn":
            metrics["interface_nll"] = InterfaceNLL()
            metrics["interface_sequence_recovery"] = InterfaceSequenceRecovery()
        self.metrics = MetricManager(metrics)

        # Loss
        loss_params = loss if loss else {}
        self.loss = LabelSmoothedNLLLoss(**loss_params)

    def construct_model(self):
        """Construct the model with hard-coded parameters."""
        with self.fabric.init_module():
            ranked_logger.info(f"Instantiating {self.model_type} model...")

            # Hard-coded model selection
            if self.model_type == "protein_mpnn":
                model = ProteinMPNN()
            elif self.model_type == "ligand_mpnn":
                model = LigandMPNN()
            else:
                raise ValueError(f"Invalid model type: {self.model_type}")

            # Initialize model weights
            model.apply(model.init_weights)

        self.initialize_or_update_trainer_state({"model": model})

    def training_step(
        self,
        batch: Any,
        batch_idx: int,
        is_accumulating: bool,
    ) -> None:
        """
        Training step, running forward and backward passes.

        Args:
            batch (Any): The current batch; can be of any form.
            batch_idx (int): The index of the current batch.
            is_accumulating (bool): Whether we are accumulating gradients
                (i.e., not yet calling optimizer.step()). If this is the case,
                we should skip the synchronization during the backward pass.

        Returns:
            None; we call `loss.backward()` directly, and store the outputs in
                `self._current_train_return`.
        """
        model = self.state["model"]
        assert model.training, "Model must be training!"

        network_input = batch

        with self.fabric.no_backward_sync(model, enabled=is_accumulating):
            # Forward pass
            network_output = model.forward(network_input)
            assert_no_nans(
                network_output["decoder_features"],
                msg="network_output['decoder_features'] "
                + f"for batch_idx: {batch_idx}",
            )

            total_loss, loss_dict = self.loss(
                network_input=batch,
                network_output=network_output,
                loss_input={},
            )

            # Backward pass
            self.fabric.backward(total_loss)

            # Optionally compute training metrics
            train_return = {"total_loss": total_loss, "loss_dict": loss_dict}

            # Store the outputs without gradients for use in logging,
            # callbacks, learning rate schedulers, etc.
            self._current_train_return = apply_to_collection(
                train_return,
                dtype=torch.Tensor,
                function=lambda x: x.detach(),
            )

    def validation_step(
        self,
        batch: Any,
        batch_idx: int,
        compute_metrics: bool = True,
    ) -> dict:
        """
        Validation step, running forward pass and computing validation
        metrics.

        Args:
            batch (Any): The current batch; can be of any form.
            batch_idx (int): The index of the current batch.
            compute_metrics (bool): Whether to compute metrics. If False, we
                will not compute metrics, and the output will be None. Set to
                False during the inference pipeline, where we need the network
                output but cannot compute metrics (since we do not have the
                ground truth).

        Returns:
            dict: Output dictionary containing the validation metrics and
                network output.
        """
        model = self.state["model"]
        assert not model.training, "Model must be in evaluation mode during validation!"

        network_input = batch

        # Forward pass
        network_output = model.forward(network_input)

        assert_no_nans(
            network_output["decoder_features"],
            msg="network_output['decoder_features'] " + f"for batch_idx: {batch_idx}",
        )

        metrics_output = {}
        if compute_metrics:
            # Compute all metrics using MetricManager
            metrics_output = self.metrics(
                network_input=batch,
                network_output=network_output,
                extra_info={},
            )

            # Avoid gradients in stored values to prevent memory leaks
            if metrics_output:
                metrics_output = apply_to_collection(
                    metrics_output, torch.Tensor, lambda x: x.detach()
                )

        network_output = apply_to_collection(
            network_output, torch.Tensor, lambda x: x.detach()
        )

        validation_return = {
            "metrics_output": metrics_output,
            "network_output": network_output,
        }

        return validation_return
