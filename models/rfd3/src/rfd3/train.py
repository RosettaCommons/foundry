#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../../../../.ipd/shebang/rfd3_exec.sh" "$0" "$@"'

import logging
import os

import hydra
import rootutils
from dotenv import load_dotenv
from omegaconf import DictConfig

from foundry.utils.logging import suppress_warnings
from foundry.utils.weights import CheckpointConfig

# Setup root dir and environment variables (more info: https://github.com/ashleve/rootutils)
# NOTE: Sets the `PROJECT_ROOT` environment variable to the root directory of the project (where `.project-root` is located)
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

load_dotenv(override=True)

_config_path = os.path.join(os.environ["PROJECT_ROOT"], "models/rfd3/configs")

_spawning_process_logger = logging.getLogger(__name__)


@hydra.main(config_path=_config_path, config_name="train", version_base="1.3")
def train(cfg: DictConfig) -> None:
    # ==============================================================================
    # Import dependencies and resolve Hydra configuration
    # ==============================================================================

    _spawning_process_logger.info("Importing dependencies...")

    # Lazy imports to make config generation fast
    import torch
    from lightning.fabric import seed_everything
    from lightning.fabric.loggers import Logger

    # If training on DIGS L40, set precision of matrix multiplication to balance speed and accuracy
    # Reference: https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    torch.set_float32_matmul_precision("medium")

    from foundry.callbacks.callback import BaseCallback  # noqa
    from foundry.utils.instantiators import instantiate_loggers, instantiate_callbacks  # noqa
    from foundry.utils.logging import (
        print_config_tree,
        log_hyperparameters_with_all_loggers,
    )  # noqa
    from foundry.utils.ddp import RankedLogger  # noqa
    from foundry.utils.ddp import is_rank_zero, set_accelerator_based_on_availability  # noqa
    from foundry.utils.datasets import (
        recursively_instantiate_datasets_and_samplers,
        assemble_distributed_loader,
        subset_dataset_to_example_ids,
        assemble_val_loader_dict,
    )  # noqa

    set_accelerator_based_on_availability(cfg)

    ranked_logger = RankedLogger(__name__, rank_zero_only=True)
    _spawning_process_logger.info("Completed dependency imports ...")

    # ... print the configuration tree (NOTE: Only prints for rank 0)
    print_config_tree(cfg, resolve=True)

    # ==============================================================================
    # Logging and Callback instantiation
    # ==============================================================================

    # Reduce the logging level for all dataset and sampler loggers (unless rank 0)
    # We will still see messages from Rank 0; they are identical, since all ranks load and sample from the same datasets
    if not is_rank_zero():
        dataset_logger = logging.getLogger("datasets")
        sampler_logger = logging.getLogger("atomworks.ml.samplers")
        dataset_logger.setLevel(logging.WARNING)
        sampler_logger.setLevel(logging.ERROR)

    # ... seed everything (NOTE: By setting `workers=True`, we ensure that the dataloaders are seeded as well)
    # (`PL_GLOBAL_SEED` environment varaible will be passed to the spawned subprocessed; e.g., through `ddp_spawn` backend)
    if cfg.get("seed"):
        ranked_logger.info(f"Seeding everything with seed={cfg.seed}...")
        seed_everything(cfg.seed, workers=True, verbose=True)
    else:
        ranked_logger.warning("No seed provided - Not seeding anything!")

    ranked_logger.info("Instantiating loggers...")
    loggers: list[Logger] = instantiate_loggers(cfg.get("logger"))

    ranked_logger.info("Instantiating callbacks...")
    callbacks: list[BaseCallback] = instantiate_callbacks(cfg.get("callbacks"))

    # ==============================================================================
    # Trainer and model instantiation
    # ==============================================================================

    # ... instantiate the trainer
    ranked_logger.info("Instantiating trainer...")
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        loggers=loggers or None,
        callbacks=callbacks or None,
        _convert_="partial",
        _recursive_=False,
    )
    # (Store the Hydra configuration in the trainer state)
    trainer.initialize_or_update_trainer_state({"train_cfg": cfg})

    # ... spawn processes for distributed training
    # (We spawn here, rather than within `fit`, so we can use Fabric's `init_module` to efficiently initialize the model on the appropriate device)
    ranked_logger.info(
        f"Spawning {trainer.fabric.world_size} processes from {trainer.fabric.global_rank}..."
    )
    trainer.fabric.launch()

    # ... construct the model
    trainer.construct_model()

    # ... construct the optimizer and schedule (which requires the model to be constructed)
    trainer.construct_optimizer()
    trainer.construct_scheduler()

    # ==============================================================================
    # Dataset instantiation
    # ==============================================================================

    # Number of examples per epoch (accross all GPUs)
    # (We must sample this many indices from our sampler)
    n_examples_per_epoch = cfg.trainer.n_examples_per_epoch

    # ... build the train dataset
    assert (
        "train" in cfg.datasets and cfg.datasets.train
    ), "No 'train' dataloader configuration provided! If only performing validation, use `validate.py` instead."
    dataset_and_sampler = recursively_instantiate_datasets_and_samplers(
        cfg.datasets.train
    )
    train_dataset, train_sampler = (
        dataset_and_sampler["dataset"],
        dataset_and_sampler["sampler"],
    )

    # ... compose the train loader
    if "subset_to_example_ids" in cfg.datasets:
        # Backdoor for debugging and overfitting: subset the dataset to a specific set of example IDs
        train_dataset = subset_dataset_to_example_ids(
            train_dataset, cfg.datasets.subset_to_example_ids
        )
        train_sampler = None  # Sampler is no longer valid, since we are using a subset of the dataset

    train_loader = assemble_distributed_loader(
        dataset=train_dataset,
        sampler=train_sampler,
        rank=trainer.fabric.global_rank,
        world_size=trainer.fabric.world_size,
        n_examples_per_epoch=n_examples_per_epoch,
        loader_cfg=cfg.dataloader["train"],
    )

    # ... compose the validation loader(s)
    if "val" in cfg.datasets and cfg.datasets.val:
        val_loaders = assemble_val_loader_dict(
            cfg=cfg.datasets.val,
            rank=trainer.fabric.global_rank,
            world_size=trainer.fabric.world_size,
            loader_cfg=cfg.dataloader["val"],
        )
    else:
        ranked_logger.warning("No validation datasets provided! Skipping validation...")
        val_loaders = None

    ranked_logger.info("Logging hyperparameters...")
    log_hyperparameters_with_all_loggers(
        trainer=trainer, cfg=cfg, model=trainer.state["model"]
    )

    # ... load the checkpoint configuration
    ckpt_config = None
    if "ckpt_config" in cfg and cfg.ckpt_config:
        ckpt_config = hydra.utils.instantiate(cfg.ckpt_config)
    elif "ckpt_path" in cfg and cfg.ckpt_path:
        # Just a checkpoint path
        if cfg.ckpt_path is not None:
            ckpt_config = CheckpointConfig(path=cfg.ckpt_path)

    # ... train the model
    ranked_logger.info("Training model...")

    with suppress_warnings():
        trainer.fit(
            train_loader=train_loader, val_loaders=val_loaders, ckpt_config=ckpt_config
        )


if __name__ == "__main__":
    train()
