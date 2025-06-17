from abc import ABC

from beartype.typing import Any
from lightning.fabric.wrappers import (
    _FabricOptimizer,
)
from torch import nn


class BaseCallback(ABC):
    """Abstract base class used to build new callbacks.

    Where possible, use names consistent with PyTorch Lightning's callback names (see references below).
    Note that if using any callbacks directly within a Model, they must also adhere to this schema.

    References:
        - Pytorch Lightning Hooks (https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#hooks)
        - Callbacks Flow (https://pytorch-lightning.readthedocs.io/en/0.10.0/callbacks.html#callbacks)
    """

    # Epoch loops
    def on_fit_start(self, trainer: Any | None = None, model: nn.Module = None):
        """Called at the start of the training"""
        pass

    def on_fit_end(self, trainer: Any | None = None):
        """Called at the end of the training"""
        pass

    # Training loop
    def on_train_epoch_start(self, trainer: Any | None = None):
        """Called at the start of each training epoch"""
        pass

    def on_after_train_loader_iter(self, trainer: Any | None = None, **kwargs):
        """Called after 'iter(train_loader)' is called, but before the first batch is yielded"""
        pass

    def on_before_train_loader_next(self, trainer: Any | None = None, **kwargs):
        """Called after each batch is yielded from the train_loader 'next(train_iter)' call"""
        pass

    def on_train_batch_start(
        self, batch: Any, batch_idx: int, trainer: Any | None = None
    ):
        """Called at the start of each training batch"""
        pass

    def on_train_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int, trainer: Any | None = None
    ):
        """Called after each training batch, but before the optimizer.step"""
        pass

    def on_before_optimizer_step(
        self, optimizer: _FabricOptimizer, trainer: Any | None = None
    ):
        """Called before each optimizer.step"""
        pass

    def optimizer_step(self, optimizer: _FabricOptimizer, trainer: Any | None = None):
        """Called after each optimizer.step"""
        pass

    def on_train_epoch_end(self, trainer: Any | None = None):
        """Called at the end of each training epoch"""
        pass

    # Validation loop
    def on_validation_epoch_start(self, trainer: Any | None = None):
        """Called at the start of each validation epoch"""
        pass

    def on_validation_batch_start(
        self,
        batch: Any,
        batch_idx: int,
        num_batches: int,
        trainer: Any | None = None,
        dataset_name: str | None = None,
    ):
        """Called at the start of each validation batch"""
        pass

    def on_validation_batch_end(
        self,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        num_batches: int,
        trainer: Any | None = None,
        dataset_name: str | None = None,
    ):
        """Called after each validation batch"""
        pass

    def on_validation_epoch_end(self, trainer: Any | None = None):
        """Called at the end of each validation epoch"""
        pass

    # Saving and Loading
    def on_save_checkpoint(self, state: dict[str, Any], trainer: Any | None = None):
        """Called when saving a checkpoint"""
        pass
