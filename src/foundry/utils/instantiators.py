import hydra
from lightning.fabric.loggers import Logger
from omegaconf import DictConfig

from foundry.callbacks.callback import BaseCallback


def _can_be_instantiated(cfg: DictConfig) -> bool:
    """Checks if a config can be instantiated."""
    return isinstance(cfg, DictConfig) and "_target_" in cfg


class InstantiationError(ValueError):
    """Raised when a config cannot be instantiated."""

    pass


def instantiate_callbacks(callbacks_cfg: DictConfig | None) -> list[BaseCallback]:
    """Instantiates callbacks from config.

    Args:
        callbacks_cfg: A DictConfig object containing callback configurations.

    Returns:
        A list of instantiated callbacks.

    Reference:
        - Lightning Hydra Template (https://github.com/ashleve/lightning-hydra-template/blob/main/src/utils/instantiators.py#L36)
    """
    callbacks: list[BaseCallback] = []

    if not callbacks_cfg:
        return callbacks

    for _, cb_conf in callbacks_cfg.items():
        if _can_be_instantiated(cb_conf):
            callbacks.append(hydra.utils.instantiate(cb_conf))
        else:
            raise InstantiationError(
                f"Skipping callback <{cb_conf}> - Not a DictConfig with `_target_` key! Please provide a valid `_target_` for instantiation."
            )

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig | None) -> list[Logger]:
    """Instantiates loggers from config.

    Args:
        logger_cfg: A DictConfig object containing logger configurations.

    Return:
        A list of instantiated loggers.

    Reference:
        - Lightning Hydra Template (https://github.com/ashleve/lightning-hydra-template/blob/main/src/utils/instantiators.py#L36)
    """
    loggers: list[Logger] = []

    if not logger_cfg:
        return loggers

    for _, lg_conf in logger_cfg.items():
        if _can_be_instantiated(lg_conf):
            loggers.append(hydra.utils.instantiate(lg_conf))
        else:
            raise InstantiationError(
                f"Skipping logger <{lg_conf}> - Not a DictConfig with `_target_` key! Please provide a valid `_target_` for instantiation."
            )

    return loggers
