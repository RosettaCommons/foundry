"""Fabric-based trainer for LightningModules and PyTorch models.

This trainer works with both LightningModules (like Proteina) that have their own
training_step/validation_step methods, and regular PyTorch models.
"""

import lightning as L
import torch
from beartype.typing import Any

from modelhub.trainers.fabric import FabricTrainer
from modelhub.training.EMA import EMA
from modelhub.utils.ddp import RankedLogger

ranked_logger = RankedLogger(__name__, rank_zero_only=True)


class ProteinaTrainer(FabricTrainer):
    """Fabric-based trainer that works with both LightningModules and PyTorch models.

    For LightningModules (like Proteina): delegates to the module's training_step,
    validation_step, and configure_optimizers methods. Injects the trainer instance
    so the module can access trainer.world_size and other properties.

    For PyTorch models: subclasses must implement training_step and validation_step.

    Supports logging frequency throttling via log_every_n_steps parameter.

    Usage with LightningModule:
        trainer = ProteinaTrainer(
            accelerator="gpu",
            devices_per_node=8,
            num_nodes=1,
            precision="bf16-mixed",
            max_epochs=100,
            output_dir="./outputs",
            grad_accum_steps=1,
            clip_grad_max_norm=1.0,
        )

        # Initialize with config
        trainer.initialize_or_update_trainer_state({"train_cfg": cfg_exp})

        # Build model and optimizer
        trainer.construct_model()
        trainer.construct_optimizer()

        # Train (using FabricTrainer's fit method)
        trainer.fit(train_loader, val_loaders)

        # Or run inference
        predictions = trainer.predict(model, dataloader)
    """

    def __init__(self, *, log_every_n_steps: int = 1, **kwargs):
        """Initialize ProteinaTrainer with optional logging frequency control.

        Args:
            log_every_n_steps: Log metrics every N optimizer steps (default: 1, i.e., every step)
            **kwargs: All other arguments passed to FabricTrainer
        """
        super().__init__(**kwargs)
        self.log_every_n_steps = log_every_n_steps

    @property
    def world_size(self):
        """Expose world_size for compatibility with LightningModule."""
        return self.fabric.world_size

    @property
    def global_step(self):
        """Expose global_step for compatibility with LightningModule."""
        return self.state["global_step"]

    def log(self, name: str, value: float, step: int | None = None):
        """Wrapper for fabric.log() that automatically uses global_step and respects log_every_n_steps."""
        if step is None:
            step = self.state["global_step"]

        # Only log if we're at the logging interval
        if step % self.log_every_n_steps == 0:
            self.fabric.log(name, value, step=step)

    def log_dict(self, metrics: dict, step: int | None = None):
        """Wrapper for fabric.log_dict() that automatically uses global_step and respects log_every_n_steps."""
        if step is None:
            step = self.state["global_step"]

        # Only log if we're at the logging interval
        if step % self.log_every_n_steps == 0:
            self.fabric.log_dict(metrics, step=step)

    def construct_model(self):
        """Construct the Proteina LightningModule.

        Proteina is instantiated with its config and handles its own
        internal setup (autoencoder, flow matcher, etc.).
        """
        with self.fabric.init_module():
            ranked_logger.info("Instantiating Proteina model...")

            # Import here to avoid circular dependencies
            try:
                from proteinfoundation.proteina import Proteina
            except ImportError:
                raise ImportError(
                    "Cannot import Proteina. Make sure proteinfoundation is installed "
                    "as an editable package: pip install -e projects/proteinfoundation"
                )

            # Get config - Proteina expects the full experiment config
            cfg_exp = self.state["train_cfg"]

            # Instantiate Proteina (it's a LightningModule)
            # Note: Proteina will wrap its internal self.nn in EMA if cfg_exp.ema.decay > 0
            model = Proteina(
                cfg_exp=cfg_exp,
                store_dir=self.output_dir,
                autoencoder_ckpt_path=cfg_exp.get("autoencoder_ckpt_path", None),
            )

        self.initialize_or_update_trainer_state({"model": model})

    def construct_optimizer(self) -> None:
        """Construct optimizer, using LightningModule's method if available."""
        assert "model" in self.state and self.state["model"] is not None, (
            "Model not found! Call construct_model() first."
        )

        model = self.state["model"]

        # Check if model is a LightningModule with configure_optimizers
        if isinstance(model, L.LightningModule) and hasattr(
            model, "configure_optimizers"
        ):
            optimizer = model.configure_optimizers()
            self.initialize_or_update_trainer_state({"optimizer": optimizer})
        else:
            # Fall back to default Hydra-based optimizer construction
            super().construct_optimizer()

    def setup_model_optimizers_and_schedulers(self) -> None:
        """Setup model, optimizer, and scheduler. Injects trainer into LightningModules."""
        # Call parent setup
        super().setup_model_optimizers_and_schedulers()

        # If model is a LightningModule, inject trainer reference
        model = self.state["model"]
        if isinstance(model, L.LightningModule):
            model.trainer = self
            ranked_logger.info(
                "Injected FabricTrainer into LightningModule for compatibility"
            )

    def training_step(
        self,
        batch: Any,
        batch_idx: int,
        is_accumulating: bool,
    ) -> None:
        """Training step that works with both LightningModules and PyTorch models.

        For LightningModules: delegates to model.training_step()
        For PyTorch models: raises NotImplementedError (subclass must implement)
        """
        model = self.state["model"]
        assert model.training, "Model must be in training mode!"

        # Check if this is a LightningModule with training_step method
        if hasattr(model, "training_step"):
            # Delegate to LightningModule's training_step
            with self.fabric.no_backward_sync(model, enabled=is_accumulating):
                loss = model.training_step(batch, batch_idx)
                self.fabric.backward(loss)
                self._current_train_return = {"loss": loss.detach()}
        else:
            # For regular PyTorch models, subclass must implement this
            raise NotImplementedError(
                "training_step must be implemented for PyTorch models. "
                "ProteinaTrainer is designed for LightningModules (like Proteina)."
            )

    def validation_step(
        self,
        batch: Any,
        batch_idx: int,
        val_loader_name: str | None = None,
    ) -> dict:
        """Validation step that works with both LightningModules and PyTorch models.

        For LightningModules: delegates to model.validation_step()
        For PyTorch models: raises NotImplementedError (subclass must implement)
        """
        model = self.state["model"]
        assert not model.training, "Model must be in evaluation mode during validation!"

        # Check if model is a LightningModule with validation_step method
        if hasattr(model, "validation_step"):
            # Delegate to LightningModule's validation_step
            # Note: If model.nn is EMA-wrapped, it automatically uses shadow weights during eval
            model.validation_step(batch, batch_idx)
            return {}
        else:
            # For regular PyTorch models, subclass must implement this
            raise NotImplementedError(
                "validation_step must be implemented for PyTorch models. "
                "ProteinaTrainer is designed for LightningModules (like Proteina)."
            )

    def save_checkpoint(self) -> None:
        """Saves checkpoints with current state.

        If model.nn is EMA-wrapped, saves two checkpoints:
        - Regular: Full state with both model.nn and model.nn.shadow (for resuming training)
        - EMA-only: Shadow weights only (lighter, for inference/evaluation)
        """
        if not self.output_dir:
            ranked_logger.warning(
                "No output directory specified; skipping checkpointing."
            )
            return

        # Provide hook to modify state before saving
        # Note: Pass as positional arg to work with both BaseCallback and LightningModule hooks
        self.fabric.call("on_save_checkpoint", self.state)

        # Determine checkpoint naming (includes both epoch and step)
        checkpoint_file = (
            self.output_dir
            / "ckpt"
            / f"epoch-{self.state['current_epoch']:04d}-step-{self.state['global_step']:08d}.ckpt"
        )

        # Save full checkpoint (includes both nn.model and nn.shadow if EMA)
        self.fabric.save(checkpoint_file, self.state)
        ranked_logger.info(f"Saved full checkpoint to: {checkpoint_file}")

        # If using EMA (check if model.nn is EMA-wrapped), also save shadow-only checkpoint
        if isinstance(self.state["model"].nn, EMA):
            ema_checkpoint_file = (
                self.output_dir
                / "ckpt"
                / f"epoch-{self.state['current_epoch']:04d}-step-{self.state['global_step']:08d}-EMA.ckpt"
            )

            # Get model state dict and swap nn.model.* with nn.shadow.* (EMA weights)
            full_state_dict = self.state["model"].state_dict()
            ema_state_dict = {}

            for key, value in full_state_dict.items():
                if key.startswith("nn.model."):
                    # Replace with shadow weights
                    shadow_key = key.replace("nn.model.", "nn.shadow.")
                    ema_state_dict[key] = full_state_dict.get(shadow_key, value)
                elif not key.startswith("nn.shadow."):
                    # Keep non-shadow keys (autoencoder, fm, etc.)
                    ema_state_dict[key] = value

            # Lightweight EMA checkpoint (no optimizer/scheduler)
            ema_state = {
                "model": ema_state_dict,
                "current_epoch": self.state["current_epoch"],
                "global_step": self.state["global_step"],
            }

            self.fabric.save(ema_checkpoint_file, ema_state)
            ranked_logger.info(f"Saved EMA-only checkpoint to: {ema_checkpoint_file}")

    def predict(self, model, dataloader):
        """Run prediction/inference for LightningModules.

        Runs the model's predict_step without Fabric distributed features,
        mimicking L.Trainer.predict() behavior.

        Args:
            model: LightningModule with predict_step method
            dataloader: DataLoader for generation

        Returns:
            List of predictions from model.predict_step
        """
        if not (
            isinstance(model, L.LightningModule) and hasattr(model, "predict_step")
        ):
            raise NotImplementedError(
                "predict() requires a LightningModule with predict_step method"
            )

        model.eval()
        all_predictions = []

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Inject trainer if not already set (needed for predict_step)
        if not hasattr(model, "trainer") or model.trainer is None:
            model.trainer = self

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # When model.nn is EMA-wrapped and in eval mode, it uses shadow weights automatically
                predictions = model.predict_step(batch, batch_idx)
                all_predictions.append(predictions)

        return all_predictions