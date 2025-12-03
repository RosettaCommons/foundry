"""Generic training harness built atop PyTorch Lightning Fabric.

In addition to standard harness features (gradient accumulation, mixed precision, etc.), includes native support for EMA.

References:
    - Pytorch Lightning Trainer Example (https://github.com/Lightning-AI/pytorch-lightning/blob/master/examples/fabric/build_your_own_trainer/trainer.py)
    - Lightning Hydra Template (https://github.com/ashleve/lightning-hydra-template)
"""

import math
import time
from abc import ABC, abstractmethod
from datetime import timedelta
from pathlib import Path
from typing import cast

import hydra
import lightning as L
import numpy as np
import torch
from beartype.typing import Any, Mapping
from lightning.fabric.accelerators import Accelerator
from lightning.fabric.loggers import Logger
from lightning.fabric.strategies import DDPStrategy, Strategy
from lightning.fabric.wrappers import (
    _FabricDataLoader,
    _FabricModule,
    _FabricOptimizer,
)

from foundry.callbacks.callback import BaseCallback
from foundry.training.EMA import EMA
from foundry.training.schedulers import SchedulerConfig
from foundry.utils.ddp import RankedLogger

ranked_logger = RankedLogger(__name__, rank_zero_only=True)
logger = RankedLogger(__name__, rank_zero_only=False)


class FabricTrainer(ABC):
    def __init__(
        self,
        *,
        accelerator: str | Accelerator = "auto",
        strategy: str | Strategy = "ddp",
        devices_per_node: list[int] | int | str = "auto",
        num_nodes: int = 1,
        precision: str | int = "32-true",
        callbacks: BaseCallback | list[BaseCallback] | None = None,
        loggers: Logger | list[Logger] | None = None,
        max_epochs: int = 1000,
        grad_accum_steps: int = 1,
        validate_every_n_epochs: int = 1,
        n_examples_per_epoch: int = 24_000,
        output_dir: Path | str | None = None,
        checkpoint_every_n_epochs: int = 1,
        clip_grad_max_norm: float | None = None,
        error_if_grad_nonfinite: bool = True,
        limit_train_batches: int | float = float("inf"),
        limit_val_batches: int | float = float("inf"),
        prevalidate: bool = False,
        nccl_timeout: int = 3200,
        skip_optimizer_loading: bool = False,
    ) -> None:
        """Base Trainer class built around Lightning Fabric.

        Args:
            accelerator: The hardware to run on. See (1) for details. Possible choices are:
                ``"cpu"``, ``"cuda"``, ``"mps"``, ``"gpu"``, ``"tpu"``, ``"auto"``.
            strategy: Strategy for how to run across multiple devices. See (1) for details. Possible choices are:
                ``"dp"``, ``"ddp"``, ``"ddp_spawn"``, ``"deepspeed"``, ``"fsdp"``.
            devices_per_node: Number of devices to train on per machine (``int``), which GPUs to train on (``list`` or ``str``), or ``"auto"``.
                See (1) for details.
                EXAMPLE: If you run on 2 nodes with 8 GPUs each, you would set ``devices_per_node=8``, not ``16``.
            num_nodes: Number of machines (nodes) for distributed training (default: 1). See (1) for details.
            precision: Double precision (``"64"``), full precision (``"32"``), half precision AMP (``"16-mixed"``),
                or bfloat16 precision AMP (``"bf16-mixed"``). See (2) for details.
            callbacks: A single callback or a list of callbacks, each inheriting the BaseCallback Abstract Base Class.
            loggers:  A single logger or a list of loggers. See (3) for details.
            max_epochs: Maximum number of epochs to train for (default: 1000).
            grad_accum_steps: Number of batches to process before calling optimizer.step() (default: 1). See (4) for details on gradient accumulation in Fabric.
            validate_every_n_epochs: Number of epochs between validation runs (default: 1).
            n_examples_per_epoch: Number of examples to sample per epoch, across all GPUs. E.g., number of distinct examples that will
                be "seen" by the model in a single epoch. If smaller than the the number implied by the dataloader, we will
                alert a warning and use the smaller number.
            output_dir: Directory to save checkpoints, metrics, intermediate validation strructures, etc. (default: None).
            checkpoint_every_n_epochs: Number of epochs between saving checkpoints (default: 1).
            clip_grad_max_norm: Maximum gradient norm to clip to (default: None). If None, no gradient clipping is performed.
            error_if_grad_nonfinite: Whether to raise an error in gradient clipping if gradients are non-finite (default: True).
            limit_train_batches: Limit on the number of training batches per epoch (default: float("inf")).
                Helpful for debugging; should NOT be used when training production models.
            limit_val_batches: Limit on the number of validation batches per epoch (default: float("inf")).
                Helpful for debugging; should NOT be used when training production models.
            prevalidate: Whether to run validation before training starts (default: False).
            nccl_timeout: Timeout for NCCL operations (default: 3200). Only used with DDP strategy.

        References:
            (1) Fabric Arguments (https://lightning.ai/docs/fabric/stable/api/fabric_args.html)
            (2) Fabric Precision Documentation (https://lightning.ai/docs/fabric/stable/fundamentals/precision.html)
            (3) Fabric Loggers (https://lightning.ai/docs/fabric/2.4.0/api/loggers.html)
            (4) Efficient Gradient Accumulation (https://lightning.ai/docs/fabric/2.4.0/advanced/gradient_accumulation.html)
        """
        # DDP strategy requires a manual timeout higher than the default
        if strategy == "ddp":
            strategy = DDPStrategy(timeout=timedelta(seconds=nccl_timeout))

        # See (1) for initialization arguments for Fabric()
        self.fabric = L.Fabric(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices_per_node,
            num_nodes=num_nodes,
            precision=precision,
            callbacks=callbacks,
            loggers=loggers,
        )

        # Training
        self.clip_grad_max_norm = clip_grad_max_norm
        self.grad_accum_steps = grad_accum_steps
        self.error_if_grad_nonfinite = error_if_grad_nonfinite

        # Stopping
        self.max_epochs = max_epochs
        self.should_stop = False
        self.n_examples_per_epoch = n_examples_per_epoch
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches

        # Validation
        self.validate_every_n_epochs = validate_every_n_epochs
        self.prevalidate = prevalidate

        # Checkpoints
        self.output_dir = Path(output_dir) if output_dir else None
        self.checkpoint_every_n_epochs = checkpoint_every_n_epochs
        self.skip_optimizer_loading = skip_optimizer_loading

    def initialize_or_update_trainer_state(
        self,
        updates: dict,
    ):
        """Initialize or update the state dictionary for the trainer.

        State keys:
            model: The model to train.
            optimizer: The optimizer to use with the model. May be None for validation/inference.
            scheduler_cfg: Learning rate SchedulerConfig (e.g., a LRScheduler with intervals/frequency). May be None for validation/inference or if no scheduler is used.
            global_step: Global optimizer step; used by W&B logger, learning rate schedulers, etc. Default is 0.
            current_epoch: Global epoch counter; used for validation, learning rate schedulers, checkpointing, etc. Default is 0.
            train_cfg: The training configuration dictionary. Used for reinitializing the trainer with the same configuration
                (for training or for inference). Default is an empty dictionary.
        """
        # Default values for the state
        default_state = {
            "model": None,
            "optimizer": None,
            "scheduler_cfg": None,
            "global_step": 0,
            "current_epoch": 0,
            "train_cfg": {},
        }

        # Initialize self.state with default values if it doesn't exist
        if not hasattr(self, "state"):
            self.state = default_state.copy()
        else:
            # Ensure existing state has all default keys
            for key, value in default_state.items():
                self.state.setdefault(key, value)

        # Merge the updates into the existing state
        self.state.update(updates)

    def construct_optimizer(self) -> None:
        """Instantiate the optimizer(s)

        We provide a default implementation that instantiates the optimizer(s) from the Hydra configuration.
        More complex models (e.g., GANs) may require custom implementations.
        """
        assert (
            "model" in self.state and hasattr(self.state["model"], "parameters")
        ), "Model not found in state dictionary! You must call `construct_model()` before constructing the optimizer."

        if self.state["train_cfg"].model.optimizer:
            # ... instantiate the optimizer
            optimizer = hydra.utils.instantiate(
                self.state["train_cfg"].model.optimizer,
                params=self.state["model"].parameters(),
            )
            self.initialize_or_update_trainer_state({"optimizer": optimizer})

    def construct_scheduler(self) -> None:
        """Instantiate the learning rate scheduler(s)

        Like optimizers, we provided a default implementation that instantiates the scheduler(s) from the Hydra configuration.
        More complex models (e.g., GANs) may require custom implementations.
        """
        assert (
            "optimizer" in self.state and self.state["optimizer"]
        ), "Optimizer not found in state dictionary! You must call `construct_optimizer()` before constructing the scheduler."

        # ...  instantiate the LR scheduler(s)
        lr_scheduler = (
            hydra.utils.instantiate(
                self.state["train_cfg"].model.lr_scheduler,
                optimizer=self.state["optimizer"],
            )
            if self.state["train_cfg"].model.lr_scheduler
            else None
        )

        if lr_scheduler:
            # We assume "interval = step" and "frequency = 1" for the default scheduler; custom implementations may override this method
            scheduler_cfg = SchedulerConfig(
                scheduler=lr_scheduler,
                interval="step",
                frequency=1,
            )
            self.initialize_or_update_trainer_state({"scheduler_cfg": scheduler_cfg})

    def construct_model(self):
        """Instantiate the model, updating the trainer state in-place.

        This method must set the "model" key in the state dictionary using `self.initialize_or_update_trainer_state()`.
        For an example, see the `construct_model` method in the `AF3Trainer`
        """
        raise NotImplementedError

    def setup_model_optimizers_and_schedulers(self) -> None:
        """Setup the model, optimizer(s), and scheduler(s) with Fabric.

        Note that we must call this method after constructing (instantiating) the model, optimizer(s), and scheduler(s).
        For details on multi-model and multi-optimizer setups, see: https://lightning.ai/docs/fabric/2.2.3/advanced/multiple_setup.html
        """
        assert self.state[
            "model"
        ], "You must construct the model before setting up the model, optimizer, and scheduler."
        model = self.state["model"]
        optimizer = self.state["optimizer"]

        # ... setup the model and optimizer
        if optimizer:
            model, optimizer = self.fabric.setup(model, optimizer)
        else:
            model = self.fabric.setup(model)

        # ... update the state dictionary (we avoid updating the state dictionary in-place, which is an anti-pattern)
        self.initialize_or_update_trainer_state(
            {
                "model": model,
                "optimizer": optimizer,
            }
        )

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loaders: dict[str, torch.utils.data.DataLoader] | None = None,
        ckpt_path: Path | str | None = None,
    ) -> None:
        """Main entry point for training a model.

        Args:
            train_loader: Dataloader for training. Must have an iterable returning batches.
            val_loaders: Dictionary of dataloaders for validation. The keys are the names of the loaders, and the values are the loaders themselves.
            ckpt_path: Path to either:
                (a) A previous checkpoint directory from which to resume training from. In this case, we will automatically load
                    the latest checkpoint using `self.get_latest_checkpoint()`.
                (b) A specific checkpoint file to load. In this case, we will load the checkpoint from the specified file.
                If None, no checkpoint is loaded, and the model will be trained from scratch.
        """
        assert (
            hasattr(self, "state") and "model" in self.state
        ), "Model not found in state dictionary! You must call `instantiate_model()` before running fit()."

        # (If we don't have enough examples to sample, we will log a warning and use the smaller number)
        if len(train_loader) * self.fabric.world_size < self.n_examples_per_epoch:
            ranked_logger.warning(
                f"Number of examples per epoch ({self.n_examples_per_epoch}) exceeds the number of examples in the loader: "
                f"({len(train_loader) * self.fabric.world_size}). Using the latter."
            )
            self.n_examples_per_epoch = len(train_loader) * self.fabric.world_size
        self.n_batches_per_epoch = math.ceil(
            self.n_examples_per_epoch / self.fabric.world_size
        )

        # ... setup training and validation dataloaders with Fabric
        train_loader = self.fabric.setup_dataloaders(
            # Our sampler is already distributed, so we don't need to wrap with a DistributedSampler
            train_loader,
            use_distributed_sampler=False,
        )

        if val_loaders is not None:
            for key, loader in val_loaders.items():
                val_loaders[key] = self.fabric.setup_dataloaders(
                    loader, use_distributed_sampler=False
                )

        self.setup_model_optimizers_and_schedulers()

        if ckpt_path is not None:
            ckpt_path = Path(ckpt_path)
            if ckpt_path.is_dir():
                # If given a directory, load the latest checkpoint from the directory
                ranked_logger.info(
                    f"Loading latest checkpoint within the directory {ckpt_path}..."
                )
                self.load_checkpoint(self.get_latest_checkpoint(ckpt_path))
            else:
                # If given a specific checkpoint file, load that checkpoint
                self.load_checkpoint(ckpt_path)

            # Increment the global epoch (e.g., if we loaded a checkpoint from [the end of] epoch 5, we should start training at epoch 6)
            self.state["current_epoch"] += 1
            # Stopping conditions
            if (
                self.max_epochs is not None
                and self.state["current_epoch"] >= self.max_epochs
            ):
                self.should_stop = True
        else:
            ranked_logger.info("No checkpoint provided; training from scratch.")

        # Set the _num_iter_calls internal attribute of the wrapped loader to the current epoch
        # (NOTE: This addresses a bug in Lightning Fabric, where there the iter() method calls the `_set_sampler_epoch()` method,
        # relying on the _num_iter_calls attribute to determine the current epoch)
        train_loader._num_iter_calls = self.state["current_epoch"]

        self.fabric.call("on_fit_start", trainer=self, model=self.state["model"])

        # Prevalidate
        if self.prevalidate and val_loaders:
            # Temporarily decrement the current epoch, since we haven't done any training this epoch
            self.state["current_epoch"] -= 1  # (Will be -1 if training from scratch)
            ranked_logger.info(
                f"Prevalidating with epoch {self.state['current_epoch']} before training; to avoid this behavior, set `prevalidate=False` in the Trainer config."
            )
            self.validation_loop(
                val_loaders=val_loaders,
                limit_batches=self.limit_val_batches,
            )
            self.state["current_epoch"] += 1  # (Restore the current epoch)

        while not self.should_stop:
            # ... train for one epoch
            ranked_logger.info(
                f"\n+ Starting epoch {self.state['current_epoch']}/{self.max_epochs - 1}\n"
                f"+ Total examples per epoch (across all GPU): {self.n_examples_per_epoch}\n"
                f"+ Examples per GPU (batches per epoch): {self.n_batches_per_epoch}\n"
                f"+ Gradient accumulation steps: {self.grad_accum_steps}\n"
                f"+ Expected optimizer steps per epoch: {(self.n_batches_per_epoch // self.grad_accum_steps if self.grad_accum_steps > 0 else 0)}\n"
            )

            self.train_loop(
                train_loader=train_loader,
                limit_batches=self.limit_train_batches,
            )

            ranked_logger.info(f"Finished epoch {self.state['current_epoch']}!")

            # ... validate, if we're at the validation interval
            if self.should_validate and val_loaders:
                ranked_logger.info(
                    f"Starting validation for epoch {self.state['current_epoch']}!"
                )
                self.validation_loop(
                    val_loaders=val_loaders,
                    limit_batches=self.limit_val_batches,
                )

            # ... step the scheduler, if we're adjusting the learning rate at the epoch-level
            self.step_scheduler(
                level="epoch", current_value=self.state["current_epoch"]
            )

            # ... save checkpoint, if we've reached the checkpoint interval
            if self.state["current_epoch"] % self.checkpoint_every_n_epochs == 0:
                self.save_checkpoint()

            # ... increment the epoch
            self.state["current_epoch"] += 1

            # Stopping conditions
            if (
                self.max_epochs is not None
                and self.state["current_epoch"] >= self.max_epochs
            ):
                self.should_stop = True

        # Reset for next `fit()` call
        self.should_stop = False

        self.fabric.call("on_fit_end", trainer=self)

    def train_loop(
        self,
        *,
        train_loader: _FabricDataLoader,
        limit_batches: int | float = float("inf"),
    ):
        """Train model for a single epoch.

        Args:
            train_loader: Dataloader for training.
            limit_batches: Limit on the batches during this training epoch. If greater than the number of batches in the
                `train_loader`, this argument has no effect. Helpful for debugging; should NOT be used when training production models.
        """
        self.fabric.call("on_train_epoch_start", trainer=self)

        assert self.state["model"].training

        # NOTE: When we call iter(), Fabric calls the `set_sampler_epoch()` method on the sampler behind the scenes, so we don't need to call it explicitly
        train_iter = iter(train_loader)
        self.fabric.call("on_after_train_loader_iter", trainer=self)

        t_dataloaders = []
        t_trains = []
        t_step_end = time.time()
        for batch_idx in range(len(train_loader)):
            # (End epoch if stopping training completely or maximum desired batches for this epoch reached)
            if self.should_stop or batch_idx >= limit_batches:
                break

            batch = next(train_iter)

            self.fabric.call(
                "on_train_batch_start", batch=batch, batch_idx=batch_idx, trainer=self
            )
            # Optimizer should step if we've accumulated the desired number of gradients
            should_optimizer_step = (batch_idx + 1) % self.grad_accum_steps == 0

            t_step_start = time.time()
            t_dataloader = t_step_start - t_step_end
            t_dataloaders.append(t_dataloader)

            self.training_step(
                batch=batch,
                batch_idx=batch_idx,
                is_accumulating=not should_optimizer_step,
            )
            t_step_end = time.time()
            t_train = t_step_end - t_step_start
            t_trains.append(t_train)

            if t_train > 100 or t_dataloader > 100:
                ranked_logger.warning(
                    f"Training step took {t_train:.3f} seconds, dataloader took {t_dataloader:.3f} seconds. "
                    "This may indicate a performance issue with the data loading or training step."
                )
                # Uncomment these lines to dump the slow example to disk for debugging
                # fout = f"local/debug/batch_{batch_idx}_slow_example.pt"
                # ranked_logger.info("Dumping batch example to disk: {}".format(fout))
                # torch.save(batch, fout)

            if batch_idx % 100 == 0:
                logger.info(
                    f"Dataloading times for batch {batch_idx} - Avg time: {np.mean(t_dataloaders):.3f} (s, Wall), Max: {np.max(t_dataloaders):.3f} (s, Wall)",
                    rank=0,
                )
                logger.info(
                    f"Finished training step in {t_train} (s, Wall), Avg time: {np.mean(t_trains):.3f} (s, Wall), Max: {np.max(t_trains):.3f} (s, Wall)",
                    rank=0,
                )
                t_trains = []
                t_dataloaders = []

            if should_optimizer_step:
                self.fabric.call(
                    "on_before_optimizer_step",
                    optimizer=self.state["optimizer"],
                    trainer=self,
                )

                # ... step the optimizer, clipping gradients and updating EMA parameters if applicable
                self.step_optimizer()

                self.fabric.call(
                    "optimizer_step", optimizer=self.state["optimizer"], trainer=self
                )

            self.fabric.call(
                "on_train_batch_end",
                outputs=self._current_train_return,
                batch=batch,
                batch_idx=batch_idx,
                trainer=self,
            )

            if should_optimizer_step:
                # ... step the scheduler, if we're adjusting the learning rate at the optimizer step-level
                self.step_scheduler(
                    level="step", current_value=self.state["global_step"]
                )

            # ... increment the global step, if optimizer stepped
            # NOTE: Each node maintains its own global step
            self.state["global_step"] += int(should_optimizer_step)

        self.fabric.call("on_train_epoch_end", trainer=self)

    def validation_loop(
        self,
        *,
        val_loaders: dict[str, _FabricDataLoader],
        limit_batches: int | float = float("inf"),
    ):
        """Run validation loop for a single validation epoch.

        Args:
            val_loader: Dictionary of Dataloaders (more precisely, _FabricDataLoader) for validation.
            limit_batches: Limit on the batches during this validation epoch. If greater than the number of batches in the
                `val_loader`, this argument has no effect. Helpful for debugging; should NOT be used for production.
        """
        # ... set model to evaluation mode
        self.state["model"].eval()

        with torch.no_grad():
            # ... assert we're in evaluation mode
            assert not self.state["model"].training

            self.fabric.call("on_validation_epoch_start", trainer=self)

            # ... iterate over all validation loaders
            for val_loader_name, val_loader in val_loaders.items():
                if (
                    hasattr(val_loader, "eval_every_n")
                    and self.state["global_step"] > 0
                    and self.state["global_step"] % val_loader.eval_every_n != 0
                ):
                    ranked_logger.info(
                        f"Skipping validation on dataset: {val_loader_name}, with {len(val_loader)} batches, with world_size={self.fabric.world_size}."
                    )
                    continue
                else:
                    ranked_logger.info(
                        f"Running validation on dataset: {val_loader_name}, with {len(val_loader)} batches, with world_size={self.fabric.world_size}."
                    )

                for batch_idx, batch in enumerate(val_loader):
                    # ... end validation epoch if stopping training completely or maximum desired batches for this epoch reached
                    if self.should_stop or batch_idx >= limit_batches:
                        break

                    self.fabric.call(
                        "on_validation_batch_start",
                        batch=batch,
                        batch_idx=batch_idx,
                        num_batches=len(val_loader),
                        trainer=self,
                        dataset_name=val_loader_name,
                    )

                    validation_result = self.validation_step(
                        batch=batch,
                        batch_idx=batch_idx,
                    )

                    self.fabric.call(
                        "on_validation_batch_end",
                        outputs=validation_result,
                        batch=batch,
                        batch_idx=batch_idx,
                        num_batches=len(val_loader),
                        dataset_name=val_loader_name,
                        current_epoch=self.state["current_epoch"],
                    )

            self.fabric.call("on_validation_epoch_end", trainer=self)

            # ... reset the model to training mode
            self.state["model"].train()

    @abstractmethod
    def training_step(
        self,
        batch: Any,
        batch_idx: int,
        is_accumulating: bool,
    ) -> None:
        """Training step, running forward and backward passes.

        Args:
            batch: The current batch; can be of any form.
            batch_idx: The index of the current batch.
            is_accumulating: Whether we are accumulating gradients (i.e., not yet calling optimizer.step()).
                If this is the case, we should skip the synchronization during the backward pass.

        Returns:
            torch.Tensor | Mapping[str, Any]: The loss tensor or a dictionary containing the loss tensor.
        """
        pass

    @abstractmethod
    def validation_step(
        self,
        batch: Any,
        batch_idx: int,
        val_loader_name: str | None = None,
    ) -> dict:
        """Validation step, running forward pass.

        Args:
            batch: The current batch; can be of any form.
            batch_idx: The index of the current batch (within that validation loader).
            val_loader_name: The name of the validation loader, if applicable.

        Returns:
            dict: A dictionary containing the output of the designated validation metrics.
        """
        pass

    def validate(
        self,
        val_loaders: dict,
        ckpt_path: Path | str,
    ) -> None:
        """Validate a model using the given dataloaders and checkpoint.

        Args:
            model: The PyTorch model to validate.
            val_loaders: A dictionary of dataloaders for validation, where keys are names and values are dataloaders.
            ckpt_path: Path to a specific checkpoint file to load. If None, the model will be validated as is.
        """
        assert (
            hasattr(self, "state") and "model" in self.state
        ), "Model not found in state dictionary! You must call `instantiate_model()` before running validate()."

        self.setup_model_optimizers_and_schedulers()

        self.load_checkpoint(ckpt_path)

        # Setup validation dataloaders with Fabric
        for key, loader in val_loaders.items():
            val_loaders[key] = self.fabric.setup_dataloaders(
                loader, use_distributed_sampler=False
            )

        # Run the validation loop
        self.state["model"].eval()
        self.validation_loop(
            val_loaders=val_loaders, limit_batches=self.limit_val_batches
        )

    def step_optimizer(self):
        """Step the optimizer.

        This method must be called only when the optimizer is stepped (i.e., after accumulating the desired number of gradients).

        We then perform following steps:
            1. Clip gradients, if applicable.
            2. Step the optimizer.
            3. Zero the gradients.
            4. Update the EMA parameters, if applicable.
        """
        assert "optimizer" in self.state and isinstance(
            self.state["optimizer"], _FabricOptimizer
        )
        assert "model" in self.state and isinstance(
            self.state["model"], _FabricModule | EMA
        )

        optimizer = self.state["optimizer"]
        model = self.state["model"]

        # ... clip gradients, if applicable
        if self.clip_grad_max_norm is not None:
            self.fabric.clip_gradients(
                module=model,
                optimizer=optimizer,
                max_norm=self.clip_grad_max_norm,
                error_if_nonfinite=self.error_if_grad_nonfinite,  ## Ultimately, should find the root cause of nonfinite gradients!
            )

        # ... step the optimizer
        optimizer.step()

        # ... zero gradients
        optimizer.zero_grad()

        # ... update EMA parameters, if applicable
        if hasattr(model, "update"):
            model.update()

    def step_scheduler(
        self,
        level,  #: Literal["epoch", "step"],
        current_value: int,
    ):
        """Step the learning rate scheduler.

        Args:
            level: The level at which to step the scheduler. Either "epoch" or "step".
            current_value: The current epoch or step value.
        """
        # (No scheduler)
        if "scheduler_cfg" not in self.state or self.state["scheduler_cfg"] is None:
            return
        else:
            scheduler_cfg = self.state["scheduler_cfg"]

        # (Wrong interval; e.g., we adjust learning rate every epoch, but we are stepping at the step level)
        if scheduler_cfg.interval != level:
            return

        # (Right interval, but wrong frequency)
        if current_value % cast(int, scheduler_cfg.frequency) != 0:
            return

        # ... step the scheduler
        scheduler_cfg.scheduler.step()

    def save_checkpoint(self) -> None:
        """Saves a checkpoint with current state to `self.output_dir/ckpt`.

        If no output directory is specified, then no checkpoint is saved.
        """
        # No checkpoint directory; skip saving
        if not self.output_dir:
            ranked_logger.warning(
                "No output directory specified; skipping model checkpointing of state dictionary."
            )
            return

        # (Provide a hook to modify the state before saving)
        self.fabric.call("on_save_checkpoint", state=self.state, trainer=self)

        # ... construct the checkpoint file path using Path
        checkpoint_file = (
            self.output_dir / "ckpt" / f"epoch-{self.state['current_epoch']:04d}.ckpt"
        )

        # NOTE: Fabric's `save()` will call the `state_dict()` method on the model, optimizer, and scheduler_cfg
        self.fabric.save(checkpoint_file, self.state)
        ranked_logger.info(f"Saved checkpoint to: {checkpoint_file}")

    def _load_optimizer(self, ckpt: Mapping) -> None:
        """Loads the optimizer state from the checkpoint."""
        if (
            "optimizer" in ckpt
            and self.state["optimizer"]
            and not self.skip_optimizer_loading
        ):
            self.state["optimizer"].load_state_dict(ckpt["optimizer"])
        else:
            ranked_logger.warning("Skipping optimizer loading...")

    def _load_scheduler(self, ckpt: Mapping) -> None:
        """Loads the learning rate scheduler state from the checkpoint."""
        if "scheduler_cfg" in ckpt and self.state["scheduler_cfg"]:
            self.state["scheduler_cfg"].load_state_dict(ckpt["scheduler_cfg"])
        else:
            ranked_logger.warning("Skipping scheduler loading...")

    def _load_model(self, ckpt: Mapping) -> None:
        """Loads the model state from the checkpoint, handling EMA and size mismatches."""

        def _subset_state_dict_to_valid_params(
            current_dict: Mapping, ckpt_dict: Mapping, log_prefix: str = ""
        ) -> dict:
            """Subset checkpoint to parameters with matching sizes, warn on mismatches."""
            valid_state_dict = {}
            for key, ckpt_tensor in ckpt_dict.items():
                if key not in current_dict:
                    continue  # Let strict=False handle missing keys

                if ckpt_tensor.size() != current_dict[key].size():
                    ranked_logger.warning(
                        f"{log_prefix}Size mismatch for '{key}': "
                        f"model size {tuple(current_dict[key].size())} vs "
                        f"checkpoint size {tuple(ckpt_tensor.size())}. "
                        "Skipping this parameter."
                    )
                else:
                    valid_state_dict[key] = ckpt_tensor

            return valid_state_dict

        # ... load the model, subsetting to parameters with matching sizes
        model = self.state["model"]
        model.load_state_dict(
            _subset_state_dict_to_valid_params(model.state_dict(), ckpt["model"]),
            strict=False,
        )

    def load_checkpoint(
        self, ckpt_path: Path | str, is_inference: bool = False
    ) -> None:
        """Loads a checkpoint from the specified path."""
        # ... load the checkpoint (replaces the state dictionary in-place)
        ranked_logger.info(f"Loading checkpoint from: {ckpt_path}...")
        ckpt = self.fabric.load(ckpt_path)

        try:
            # ... optimize, scheduler, model
            if not is_inference:
                self._load_optimizer(ckpt)
                self._load_scheduler(ckpt)
            self._load_model(ckpt)

            # ... stateless keys
            # (We do not want to load the `train_cfg` in this instance, as it may contain different configurations)
            keys_to_ignore = {"model", "optimizer", "scheduler_cfg", "train_cfg"}
            self.state.update(
                {
                    key: value
                    for key, value in ckpt.items()
                    if key not in keys_to_ignore and key in self.state
                }
            )

            # Log warnings for missing and extra keys
            state_keys = set(self.state) - keys_to_ignore
            ckpt_keys = set(ckpt) - keys_to_ignore

            if missing := state_keys - ckpt_keys:
                ranked_logger.warning(
                    f"Keys found in STATE but not CKPT: {sorted(missing)}"
                )
            if extra := ckpt_keys - state_keys:
                ranked_logger.warning(
                    f"Keys found in CKPT but not STATE: {sorted(extra)}"
                )

            ranked_logger.info(
                f"Loaded checkpoint. Current epoch: {self.state['current_epoch']}, global step: {self.state['global_step']}"
            )
        except Exception as e:
            ranked_logger.error(
                f"Error loading checkpoint: {e}. Trying to load with legacy settings..."
            )
            self.load_legacy_checkpoint(ckpt)

    def load_legacy_checkpoint(self, ckpt: dict) -> dict:
        # TODO: Remove when no longer needed
        """Backwards-compatibility function to checkpoints with legacy state formats"""
        new_model_state = {}
        prefixes = {key.split(".")[0] for key in ckpt["final_state_dict"].keys()}

        if "model" not in prefixes:
            # (Model-only checkpoints from training, without confidence head)
            model_state_dict = {
                f"model.{k}": v for k, v in ckpt["final_state_dict"].items()
            }
            shadow_state_dict = {
                f"shadow.{k}": v for k, v in ckpt["model_state_dict"].items()
            }
            full_state_dict = {**model_state_dict, **shadow_state_dict}

        elif "confidence" in prefixes:
            # (Checkpoints with confidence head)
            ranked_logger.info("Detected confidence module in checkpoint...")

            # ... replace confidence head keys with model and shadow prefixes
            model_state_dict = {
                f"model.confidence_head{key[len('confidence') :]}"
                if key.startswith("confidence")
                else key: value
                for key, value in ckpt["final_state_dict"].items()
            }

            shadow_state_dict = {
                (
                    f"shadow.confidence_head{key[len('confidence') :]}"
                    if key.startswith("confidence")
                    else f"shadow{key[len('model') :]}"
                    if key.startswith("model")
                    else key
                ): value
                for key, value in ckpt["model_state_dict"].items()
            }
            full_state_dict = {**model_state_dict, **shadow_state_dict}
        else:
            raise ValueError("Unknown checkpoint format")

        # ... check shapes (we only load matching shapes to support fine-tuning or adding channels)
        state_dict = self.state["model"].state_dict()
        for param in state_dict:
            if param not in full_state_dict:
                ranked_logger.error(f"missing: {param}")
            elif full_state_dict[param].shape == state_dict[param].shape:
                new_model_state[param] = full_state_dict[param]
            else:
                ranked_logger.error(
                    f"wrong size: {param} {full_state_dict[param].shape} {state_dict[param].shape}"
                )

        # ... update the state
        self.state["model"].load_state_dict(new_model_state, strict=False)
        self.state["current_epoch"] = ckpt["epoch"]

        ranked_logger.info(
            f"Loaded internal AF3 clone checkpoint into model. Current epoch: {self.state['current_epoch']}, global step: {self.state['global_step']}"
        )

    @staticmethod
    def get_latest_checkpoint(ckpt_load_dir: Path) -> Path:
        """Returns the latest checkpoint file from the given directory.

        Assumes that checkpoints are stored with filenames such that a standard string-based
        sort will correctly order them by creation time (e.g., with epoch numbers, or timestamps).

        Args:
            ckpt_load_dir (Path): The directory to search for checkpoint files.

        Returns:
            Path: The path to the latest checkpoint file, or None if no checkpoints are found
            or if the directory does not exist.
        """
        if not ckpt_load_dir.is_dir():
            return None

        # List all files in the directory and sort them
        items = sorted(ckpt_load_dir.iterdir())

        # Return the last item in the sorted list, if any
        return items[-1] if items else None

    @property
    def should_validate(self) -> bool:
        """Whether to currently run validation."""
        return self.state["current_epoch"] % self.validate_every_n_epochs == 0
