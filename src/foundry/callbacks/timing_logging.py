import pandas as pd
from beartype.typing import Any
from lightning.fabric.utilities.rank_zero import rank_zero_only
from lightning.fabric.wrappers import (
    _FabricOptimizer,
)

from foundry.callbacks.callback import BaseCallback
from foundry.utils.logging import print_df_as_table
from foundry.utils.torch import Timers


class TimingCallback(BaseCallback):
    """Fabric callback to print timing metrics.

    The hooks that the base declares with explicit positional params
    (``on_train_batch_start``/``on_train_batch_end``/``on_before_optimizer_step``)
    are overridden here with ``**kwargs`` because Fabric always dispatches hooks by
    keyword (``fabric.call(name, trainer=..., batch=..., ...)``), so the unused
    arguments are simply absorbed. mypy flags the narrower signature as an
    incompatible override; the ``# type: ignore[override]`` documents that this is
    intentional and safe given the keyword-only dispatch.
    """

    def __init__(self, log_every_n: int = 100) -> None:
        super().__init__()
        self.log_every_n = log_every_n
        self.timers = Timers()
        self.n_steps_since_last_log = 0

    @rank_zero_only
    def on_train_epoch_start(self, trainer: Any, **kwargs: Any) -> None:
        self.timers.start("train_loader_iter")

    @rank_zero_only
    def on_after_train_loader_iter(self, trainer: Any, **kwargs: Any) -> None:
        self.timers.stop("train_loader_iter")

    @rank_zero_only
    def on_before_train_loader_next(self, trainer: Any, **kwargs: Any) -> None:
        self.timers.start("train_step", "train_loader_next")

    @rank_zero_only
    def on_train_batch_start(self, trainer: Any, **kwargs: Any) -> None:  # type: ignore[override]
        self.timers.start("forward_loss_backward")
        self.timers.stop("train_loader_next")

    @rank_zero_only
    def on_train_batch_end(self, trainer: Any, **kwargs: Any) -> None:  # type: ignore[override]
        self.timers.stop("forward_loss_backward")
        self.timers.stop("train_step")

    @rank_zero_only
    def on_before_optimizer_step(self, trainer: Any, **kwargs: Any) -> None:  # type: ignore[override]
        self.timers.start("optimizer_step")

    @rank_zero_only
    def on_after_optimizer_step(
        self, optimizer: _FabricOptimizer, **kwargs: Any
    ) -> None:
        self.timers.stop("optimizer_step")

    @rank_zero_only
    def optimizer_step(self, trainer: Any, optimizer: _FabricOptimizer) -> None:
        step = trainer.state["global_step"]
        self.n_steps_since_last_log += 1
        if step % self.log_every_n == 0:
            timings = self.timers.elapsed(*self.timers.timers.keys(), reset=True)
            timings = {
                f"timings/{k}": v / self.n_steps_since_last_log
                for k, v in timings.items()
            }
            trainer.fabric.log_dict(timings, step=step)
            if trainer.fabric.is_global_zero:
                self._print_timings(timings)

    def _print_timings(self, timings: dict[str, float]) -> None:
        df = pd.DataFrame(timings.items(), columns=["Step", "Time (s)"])
        print_df_as_table(
            df, title=f"Timing stats (over {self.n_steps_since_last_log} steps)"
        )
        self.n_steps_since_last_log = 0
