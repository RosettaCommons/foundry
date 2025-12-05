import logging
import os
from os import PathLike
from pathlib import Path
from typing import Any, Dict

import hydra
import torch
from biotite.structure import AtomArray
from lightning.fabric import seed_everything
from omegaconf import OmegaConf

from foundry.inference_engines.checkpoint_registry import REGISTERED_CHECKPOINTS
from foundry.utils.ddp import RankedLogger, set_accelerator_based_on_availability
from foundry.utils.logging import (
    configure_minimal_inference_logging,
    print_config_tree,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
ranked_logger = RankedLogger(__name__, rank_zero_only=True)


def merge(cfg, overrides: dict):
    return OmegaConf.merge(cfg, OmegaConf.create(overrides))


class BaseInferenceEngine:
    """
    Base inference engine.
    Separates model setup (expensive, once) from inference (can run multiple times).
    """

    def __init__(
        self,
        ckpt_path: PathLike,
        num_nodes: int = 1,
        devices_per_node: int = 1,
        # Config overrides
        transform_overrides={},
        inference_sampler_overrides={},
        trainer_overrides={},
        # Debug
        verbose: bool = False,
        seed: int | None = None,
    ):
        """Initialize inference engine and load model.

        Model config is loaded from checkpoint and overridden with parameters provided here.

        Args:
          ckpt_path: Path to model checkpoint.
          seed: Random seed. If None, uses external RNG state. Defaults to ``None``.
          num_nodes: Number of nodes for distributed inference. Defaults to ``1``.
          devices_per_node: Number of devices per node. Defaults to ``1``.
          verbose: If True, show detailed logging and config trees. Defaults to ``False``.
        """
        if not verbose:
            configure_minimal_inference_logging()

        # Set attrs
        self.initialized_ = False
        self.trainer = None
        self.pipeline = None
        self.verbose = verbose

        # Resolve checkpoint path
        if "." not in str(ckpt_path):
            # Assume registered model
            name = str(ckpt_path)
            assert (
                name in REGISTERED_CHECKPOINTS
            ), "Checkpoint provided not and not in registered checkpoints"
            ckpt = REGISTERED_CHECKPOINTS[name]

            ckpt_path = ckpt.get_default_path()
            ranked_logger.info(
                "Using checkpoint from default installation directory, got: {}".format(
                    str(ckpt_path)
                )
            )
            assert os.path.exists(
                ckpt_path
            ), "Invalid checkpoint: {}. And could not find checkpoint in default installation location: {}".format(
                name, ckpt_path
            )
        self.ckpt_path = Path(ckpt_path).resolve()

        # Set random seed (only if seed is not None)
        if seed is not None:
            ranked_logger.info(f"Seeding everything with seed={seed}...")
            seed_everything(seed, workers=True, verbose=True)
        else:
            ranked_logger.info("Seed is None - using external RNG state")
        self.seed = seed

        # Stored for later;
        self.transform_overrides = transform_overrides
        self.overrides: dict[str, Any] = {}

        base_overrides = {
            "trainer.seed": seed,
            "trainer.metrics": {},
            "trainer.loss": None,
            "trainer.num_nodes": num_nodes,
            "trainer.devices_per_node": devices_per_node,
        }
        for key, value in base_overrides.items():
            self._assign_override(key, value)

        for key, value in trainer_overrides.items():
            self._assign_override(f"trainer.{key}", value)

        for key, value in inference_sampler_overrides.items():
            self._assign_override(f"model.net.inference_sampler.{key}", value)

    ###################################################################################
    # Required subclasss methods
    ###################################################################################

    def initialize(self):
        if self.initialized_:
            return getattr(self, "cfg", None)

        # Load checkpoint and config
        ranked_logger.info(
            f"Loading checkpoint from {Path(self.ckpt_path).resolve()}..."
        )
        checkpoint = torch.load(self.ckpt_path, "cpu", weights_only=False)
        cfg = self._override_checkpoint_config(checkpoint["train_cfg"])

        # Load pipeline first before trainer/model
        self._construct_pipeline(cfg)
        self._construct_trainer(cfg, checkpoint=checkpoint)

        ranked_logger.info("Model loaded and ready for inference.")
        self.initialized_ = True
        return cfg

    def run(
        self,
        inputs: (
            Dict[str, dict] | AtomArray | list[AtomArray] | PathLike | list[PathLike]
        ),
        *_,
    ) -> dict[str, dict] | None:
        self.initialize()
        raise NotImplementedError(
            "Subclasses must implement inference logic in `run` method."
        )

    ###################################################################################
    # Util methods
    ###################################################################################

    def _override_checkpoint_config(self, cfg):
        cfg = merge(cfg, self.overrides)
        cfg = set_accelerator_based_on_availability(cfg)
        return cfg

    def _construct_trainer(self, cfg, checkpoint=None):
        """
        Sets attr self.trainer
        """
        # Instantiate trainer
        ranked_logger.info("Instantiating trainer...")
        if self.verbose:
            print_config_tree(
                cfg.trainer, resolve=True, title="INFERENCE TRAINER CONFIGURATION"
            )
        trainer = hydra.utils.instantiate(
            cfg.trainer,
            _convert_="partial",
            _recursive_=False,
        )

        # Setup model
        ranked_logger.info("Setting up model...")
        trainer.fabric.launch()
        trainer.initialize_or_update_trainer_state(
            {"train_cfg": cfg}
        )  # config from training stores net params
        trainer.construct_model()

        ranked_logger.info("Loading model weights from checkpoint...")
        trainer.load_checkpoint(checkpoint=checkpoint or self.ckpt_path)

        # Ensure optimizer isn't loaded
        trainer.state["optimizer"] = None
        trainer.state["train_cfg"].model.optimizer = None
        trainer.setup_model_optimizers_and_schedulers()
        trainer.state["model"].eval()
        self.trainer = trainer

    def _assign_override(self, dotted_key: str, value: Any) -> None:
        """Assign ``value`` into ``self.overrides`` using a dotted path."""
        target = self.overrides
        keys = dotted_key.split(".")
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                target[key] = {}
            target = target[key]
        target[keys[-1]] = value

    def _construct_pipeline(self, cfg):
        """
        Sets attr self.pipeline
        """
        # Construct pipeline
        ranked_logger.info("Building Transform pipeline...")
        first_val_dataset_key, first_val_dataset = next(iter(cfg.datasets.val.items()))
        ranked_logger.info(
            f"Using settings from validation dataset: {first_val_dataset_key}."
        )
        transform = first_val_dataset.dataset.transform
        transform = merge(transform, self.transform_overrides)

        if self.verbose:
            print_config_tree(
                transform,
                resolve=True,
                title="INFERENCE TRANSFORM PIPELINE",
            )

        self.pipeline = hydra.utils.instantiate(transform)

    # aliases for run
    def forward(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    # for use as a context manager: e.g. `with BaseInferenceEngine(...) as engine:` to automatically cleanup
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.trainer = None
        self.pipeline = None
        self.initialized_ = False
