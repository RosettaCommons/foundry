from abc import ABC

from beartype.typing import Any
from lightning.fabric.wrappers import (
    _FabricOptimizer,
)


class BaseCallback(ABC):
    """Abstract base class used to build new callbacks.

    Callbacks have access to the trainer via ``self.trainer``, which is automatically
    injected by the FabricTrainer during initialization.

    Where possible, use names consistent with PyTorch Lightning's callback names (see references below).
    Note that if using any callbacks directly within a Model, they must also adhere to this schema.

    References:
        - Pytorch Lightning Hooks (https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#hooks)
        - Callbacks Flow (https://pytorch-lightning.readthedocs.io/en/0.10.0/callbacks.html#callbacks)
    """

    # Epoch loops
    def on_fit_start(self):
        """Called at the start of the training"""
        pass

    def on_fit_end(self):
        """Called at the end of the training"""
        pass

    # Training loop
    def on_train_epoch_start(self):
        """Called at the start of each training epoch"""
        pass

    def on_after_train_loader_iter(self, **kwargs):
        """Called after 'iter(train_loader)' is called, but before the first batch is yielded"""
        pass

    def on_before_train_loader_next(self, **kwargs):
        """Called after each batch is yielded from the train_loader 'next(train_iter)' call"""
        pass

    def on_train_batch_start(self, batch: Any, batch_idx: int):
        """Called at the start of each training batch"""
        pass

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int):
        """Called after each training batch, but before the optimizer.step"""
        pass

    def on_before_optimizer_step(self, optimizer: _FabricOptimizer):
        """Called before each optimizer.step"""
        pass

    def on_after_optimizer_step(self, optimizer: _FabricOptimizer, **kwargs):
        """Called after each optimizer.step"""
        pass

    def on_train_epoch_end(self):
        """Called at the end of each training epoch"""
        pass

    # Validation loop
    def on_validation_epoch_start(self):
        """Called at the start of each validation epoch"""
        pass

    def on_validation_batch_start(
        self,
        batch: Any,
        batch_idx: int,
        num_batches: int,
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
        dataset_name: str | None = None,
    ):
        """Called after each validation batch"""
        pass

    def on_validation_epoch_end(self):
        """Called at the end of each validation epoch"""
        pass

    # Saving and Loading
    def on_save_checkpoint(self, state: dict[str, Any]):
        """Called when saving a checkpoint"""
        pass
