from dataclasses import dataclass

from torch.optim.lr_scheduler import LRScheduler, _LRScheduler
from torch.optim.optimizer import Optimizer


class AF3Scheduler(_LRScheduler):
    """Implements a two-phase learning rate schedule a-la AF-3:
        1. The base learning rate is 1.8 · 10^−3, which is linearly increased from 0 over the first 1,000 steps.
        2. The learning rate is then decreased by a factor of 0.95 every 50,000 steps.

    From the AF-3 Supplement, Section 5.4:
    >  "For training we use the Adam optimizer with parameters β1 = 0.9, β2 = 0.95, ϵ = 10^−8. The base learning rate
        is 1.8 · 10^−3, which is linearly increased from 0 over the first 1,000 steps. The learning rate is then decreased
        by a factor of 0.95 every 5 · 10^4 steps."

    References:
        - AF-3 Supplement
    """

    def __init__(
        self,
        optimizer: Optimizer,
        base_lr: float = 1.8e-3,
        warmup_steps: int = 1000,
        decay_factor: float = 0.95,
        decay_steps: int = 50000,
        last_epoch: int = -1,
    ) -> None:
        """Initializes a new instance of AF3LRScheduler.

        Note that the "last_epoch" value is incremented every time we call `scheduler.step()`
        method; we name it "epoch" to follow the PyTorch convention.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            base_lr (float): The base learning rate after warmup (which will then be decayed).
            warmup_steps (int): Number of steps for linear warmup.
            decay_factor (float): Factor by which the learning rate is multiplied every decay_steps.
            decay_steps (int): Number of steps between each decay.
            last_epoch (int): The index of the last epoch. Default: -1.
        """
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.decay_factor = decay_factor
        self.decay_steps = decay_steps
        super(AF3Scheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [
                self.base_lr * (self.last_epoch / self.warmup_steps)
                for _ in self.optimizer.param_groups
            ]
        else:
            # Decay after warmup
            num_decays = (self.last_epoch - self.warmup_steps) // self.decay_steps
            return [
                self.base_lr * (self.decay_factor**num_decays)
                for _ in self.optimizer.param_groups
            ]


@dataclass
class SchedulerConfig:
    """Flexible configuration for a learning rate scheduler.

    Modeled on the PyTorch Lightning scheduler configuration.

    Attributes:
        scheduler (LRScheduler): The learning rate scheduler instance. Must inherit from `torch.optim.lr_scheduler.LRScheduler`.
        interval (str): The interval at which to apply the scheduler, typically "epoch" or "step". Defaults to "step".
        frequency (int): The frequency of applying the scheduler. For example, a frequency of 1 means the scheduler is applied every epoch. Defaults to 1.
    """

    scheduler: LRScheduler = None
    interval: str = "step"
    frequency: int = 1

    def state_dict(self) -> dict:
        return {
            "scheduler": self.scheduler.state_dict(),
            "interval": self.interval,
            "frequency": self.frequency,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.scheduler.load_state_dict(state_dict["scheduler"])
        self.interval = state_dict["interval"]
        self.frequency = state_dict["frequency"]
