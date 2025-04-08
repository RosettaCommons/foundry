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
        - Calbacks Flow (https://pytorch-lightning.readthedocs.io/en/0.10.0/callbacks.html#callbacks)
    """

    # Epoch loops
    def on_fit_start(self, trainer: Any | None = None, model: nn.Module = None):
        pass

    def on_fit_end(self, trainer: Any | None = None):
        pass

    # Training loop
    def on_train_epoch_start(self, trainer: Any | None = None):
        pass

    def on_train_batch_start(
        self, batch: Any, batch_idx: int, trainer: Any | None = None
    ):
        pass

    def on_before_optimizer_step(
        self, optimizer: _FabricOptimizer, trainer: Any | None = None
    ):
        pass

    def optimizer_step(self, optimizer: _FabricOptimizer, trainer: Any | None = None):
        pass

    def on_train_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int, trainer: Any | None = None
    ):
        pass

    def on_train_epoch_end(self, trainer: Any | None = None):
        pass

    # Validation loop
    def on_validation_epoch_start(self, trainer: Any | None = None):
        pass

    def on_validation_batch_start(
        self,
        batch: Any,
        batch_idx: int,
        num_batches: int,
        trainer: Any | None = None,
        dataset_name: str | None = None,
    ):
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
        pass

    def on_validation_epoch_end(self, trainer: Any | None = None):
        pass

    # Saving and Loading
    def on_save_checkpoint(self, state: dict[str, Any], trainer: Any | None = None):
        pass
