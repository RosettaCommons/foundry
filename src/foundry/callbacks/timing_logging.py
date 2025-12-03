import pandas as pd
from lightning.fabric.utilities.rank_zero import rank_zero_only

from foundry.callbacks.callback import BaseCallback
from foundry.utils.logging import print_df_as_table
from foundry.utils.torch import Timers


class TimingCallback(BaseCallback):
    """Fabric callback to print timing metrics."""

    def __init__(self, log_every_n: int = 100):
        super().__init__()
        self.log_every_n = log_every_n
        self.timers = Timers()
        self.n_steps_since_last_log = 0

    @rank_zero_only
    def on_train_epoch_start(self, trainer, **kwargs):
        self.timers.start("train_loader_iter")

    @rank_zero_only
    def on_after_train_loader_iter(self, trainer, **kwargs):
        self.timers.stop("train_loader_iter")

    @rank_zero_only
    def on_before_train_loader_next(self, trainer, **kwargs):
        self.timers.start("train_step", "train_loader_next")

    @rank_zero_only
    def on_train_batch_start(self, trainer, **kwargs):
        self.timers.start("forward_loss_backward")
        self.timers.stop("train_loader_next")

    @rank_zero_only
    def on_train_batch_end(self, trainer, **kwargs):
        self.timers.stop("forward_loss_backward")
        self.timers.stop("train_step")

    @rank_zero_only
    def on_before_optimizer_step(self, trainer, **kwargs):
        self.timers.start("optimizer_step")

    @rank_zero_only
    def on_after_optimizer_step(self, optimizer, **kwargs):
        self.timers.stop("optimizer_step")

    @rank_zero_only
    def optimizer_step(self, trainer, optimizer):
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

    def _print_timings(self, timings: dict[str, float]):
        df = pd.DataFrame(timings.items(), columns=["Step", "Time (s)"])
        print_df_as_table(
            df, title=f"Timing stats (over {self.n_steps_since_last_log} steps)"
        )
        self.n_steps_since_last_log = 0
