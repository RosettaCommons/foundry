from abc import ABC

from beartype.typing import Any
from lightning.fabric.wrappers import (
    _FabricOptimizer,
)


class BaseCallback(ABC):
    """Abstract base class used to build new callbacks.

    Callbacks receive the trainer as the first argument to all hook methods, following
    PyTorch Lightning's convention. This allows callbacks to access trainer.state,
    trainer.fabric, etc.

    NOTE: on_after_optimizer_step is called internally by Fabric and does NOT receive trainer.
    Use on_before_optimizer_step for logic that requires trainer access.

    Where possible, use names consistent with PyTorch Lightning's callback names (see references below).
    Note that if using any callbacks directly within a Model, they must also adhere to this schema.

    References:
        - Pytorch Lightning Hooks (https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#hooks)
        - Callbacks Flow (https://pytorch-lightning.readthedocs.io/en/0.10.0/callbacks.html#callbacks)
    """

    # Epoch loops
    def on_fit_start(self, trainer: Any):
        """Called at the start of the training"""
        pass

    def on_fit_end(self, trainer: Any):
        """Called at the end of the training"""
        pass

    # Training loop
    def on_train_epoch_start(self, trainer: Any):
        """Called at the start of each training epoch"""
        pass

    def on_after_train_loader_iter(self, trainer: Any, **kwargs):
        """Called after 'iter(train_loader)' is called, but before the first batch is yielded"""
        pass

    def on_before_train_loader_next(self, trainer: Any, **kwargs):
        """Called after each batch is yielded from the train_loader 'next(train_iter)' call"""
        pass

    def on_train_batch_start(self, trainer: Any, batch: Any, batch_idx: int):
        """Called at the start of each training batch"""
        pass

    def on_train_batch_end(
        self, trainer: Any, outputs: Any, batch: Any, batch_idx: int
    ):
        """Called after each training batch, but before the optimizer.step"""
        pass

    def on_before_optimizer_step(self, trainer: Any, optimizer: _FabricOptimizer):
        """Called before each optimizer.step"""
        pass

    def on_after_optimizer_step(self, optimizer: _FabricOptimizer, **kwargs):
        """Called after each optimizer.step.

        NOTE: This hook is called internally by Fabric when optimizer.step() executes.
        Trainer is NOT available here. Use optimizer_step for logic requiring trainer.
        """
        pass

    def optimizer_step(self, trainer: Any, optimizer: _FabricOptimizer):
        """Called after optimizer.step completes. Unlike on_after_optimizer_step,
        this hook is called explicitly by the trainer and receives trainer access.
        """
        pass

    def on_train_epoch_end(self, trainer: Any):
        """Called at the end of each training epoch"""
        pass

    # Validation loop
    def on_validation_epoch_start(self, trainer: Any):
        """Called at the start of each validation epoch"""
        pass

    def on_validation_batch_start(
        self,
        trainer: Any,
        batch: Any,
        batch_idx: int,
        num_batches: int,
        dataset_name: str | None = None,
    ):
        """Called at the start of each validation batch"""
        pass

    def on_validation_batch_end(
        self,
        trainer: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        num_batches: int,
        dataset_name: str | None = None,
    ):
        """Called after each validation batch"""
        pass

    def on_validation_epoch_end(self, trainer: Any):
        """Called at the end of each validation epoch"""
        pass

    # Saving and Loading
    def on_save_checkpoint(self, trainer: Any, state: dict[str, Any]):
        """Called when saving a checkpoint"""
        pass
