import logging

import torch
from beartype.typing import Any
from lightning_fabric.utilities import rank_zero_only
from lightning_utilities.core.rank_zero import rank_prefixed_message
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def get_current_rank() -> int:
    """Returns the rank of the current process."""
    return getattr(rank_zero_only, "rank", None)


def is_rank_zero() -> bool:
    """Returns whether the current process is rank zero."""
    return get_current_rank() == 0


def set_accelerator_based_on_availability(cfg: dict | DictConfig):
    """Set training accelerator to CPU if no GPUs are available.

    Args:
        cfg: Hydra object with trainer settings "accelerator", "devices_per_node", and "num_nodes".

    Returns:
        None; modifies the input `cfg` object in place.
    """
    if not torch.cuda.is_available():
        logger.error(
            "No GPUs available - Setting accelerator to 'cpu'. Are you sure you are using the correct configs?"
        )
        assert "trainer" in cfg, "Configuration object must have a 'trainer' key."
        for key in ["accelerator", "devices_per_node", "num_nodes"]:
            assert (
                key in cfg.trainer
            ), f"Configuration object must have a 'trainer.{key}' key."

        # Override accelerator settings
        cfg.trainer.accelerator = "cpu"
        cfg.trainer.devices_per_node = 1
        cfg.trainer.num_nodes = 1
    else:
        cfg.trainer.accelerator = "gpu"
    return cfg


class RankedLogger(logging.LoggerAdapter):
    """A multi-GPU-friendly python command line logger.

    Modified from https://github.com/ashleve/lightning-hydra-template/blob/main/src/utils/pylogger.py
    """

    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = False,
        extra: Any | None = None,
    ) -> None:
        """Initializes a multi-GPU-friendly python command line logger that logs on all processes
        with their rank prefixed in the log message.

        :param name: The name of the logger. Default is ``__name__``.
        :param rank_zero_only: Whether to force all logs to only occur on the rank zero process. Default is `False`.
        :param extra: (Optional) A dict-like object which provides contextual information. See `logging.LoggerAdapter`.
        """
        logger = logging.getLogger(name)
        super().__init__(logger=logger, extra=extra)
        self.rank_zero_only = rank_zero_only

    def log(
        self, level: int, msg: str, rank: int | None = None, *args, **kwargs
    ) -> None:
        """
        Delegate a log call to the underlying logger, after prefixing its message with the rank
        of the process it's being logged from. If `'rank'` is provided, then the log will only
        occur on that rank/process.

        Args:
            level (int): The level to log at. Look at `logging.__init__.py` for more information.
            msg (str): The message to log.
            rank (Optional[int]): The rank to log at.
            args: Additional args to pass to the underlying logging function.
            kwargs: Any additional keyword args to pass to the underlying logging function.
        """
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            current_rank = getattr(rank_zero_only, "rank", None)
            if current_rank is None:
                raise RuntimeError(
                    "The `rank_zero_only.rank` needs to be set before use"
                )
            msg = rank_prefixed_message(msg, current_rank)
            if self.rank_zero_only:
                if current_rank == 0:
                    self.logger.log(level, msg, *args, **kwargs)
            else:
                if rank is None:
                    self.logger.log(level, msg, *args, **kwargs)
                elif current_rank == rank:
                    self.logger.log(level, msg, *args, **kwargs)
