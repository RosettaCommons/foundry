"""Unit tests for foundry.training.schedulers.

`AF3Scheduler` implements the AF-3 two-phase learning-rate schedule: a linear
warmup from 0 to `base_lr` over `warmup_steps`, then a geometric decay by
`decay_factor` every `decay_steps`. The tests pin those phase boundaries on a
small, exactly-computable schedule. `SchedulerConfig` is a thin Lightning-style
config wrapper whose state-dict round-trip is also pinned.
"""

import pytest
import torch
from torch.optim import SGD

from foundry.training.schedulers import AF3Scheduler, SchedulerConfig

# Small, exact schedule: base_lr=1.0 keeps the expected LR values trivial.
_KW = dict(base_lr=1.0, warmup_steps=10, decay_factor=0.5, decay_steps=20)


def _single_param_optimizer() -> SGD:
    return SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)


def _lr_after_steps(n: int) -> float:
    """LR reported after advancing a fresh AF3Scheduler `n` times."""
    opt = _single_param_optimizer()
    scheduler = AF3Scheduler(opt, **_KW)
    for _ in range(n):
        opt.step()  # documented order: optimizer before scheduler
        scheduler.step()
    return scheduler.get_last_lr()[0]


def test_initial_lr_is_zero():
    """Construction steps once to last_epoch=0, the start of warmup (LR 0)."""
    opt = _single_param_optimizer()
    scheduler = AF3Scheduler(opt, **_KW)
    assert scheduler.get_last_lr()[0] == pytest.approx(0.0)


def test_linear_warmup_midpoint():
    # last_epoch=5, warmup_steps=10 -> 1.0 * 5/10
    assert _lr_after_steps(5) == pytest.approx(0.5)


def test_lr_reaches_base_at_end_of_warmup():
    # last_epoch=10: warmup is exclusive (10 < 10 is False), so decay branch
    # with num_decays=0 -> base_lr * 0.5**0 = 1.0
    assert _lr_after_steps(10) == pytest.approx(1.0)


def test_geometric_decay_after_warmup():
    # last_epoch=30 -> num_decays=(30-10)//20=1 -> 1.0 * 0.5
    assert _lr_after_steps(30) == pytest.approx(0.5)
    # last_epoch=50 -> num_decays=(50-10)//20=2 -> 1.0 * 0.25
    assert _lr_after_steps(50) == pytest.approx(0.25)


def test_all_param_groups_share_one_lr():
    """get_lr emits the same value for every param group."""
    p1 = torch.nn.Parameter(torch.zeros(1))
    p2 = torch.nn.Parameter(torch.zeros(1))
    opt = SGD([{"params": [p1]}, {"params": [p2]}], lr=1.0)
    scheduler = AF3Scheduler(opt, **_KW)
    for _ in range(5):
        opt.step()
        scheduler.step()
    lrs = scheduler.get_last_lr()
    assert len(lrs) == 2
    assert lrs[0] == lrs[1] == pytest.approx(0.5)


def test_scheduler_config_state_dict_roundtrip():
    """load_state_dict restores interval, frequency, and the wrapped scheduler."""
    opt = _single_param_optimizer()
    scheduler = AF3Scheduler(opt, **_KW)
    cfg = SchedulerConfig(scheduler=scheduler, interval="epoch", frequency=3)
    for _ in range(7):
        opt.step()
        scheduler.step()
    state = cfg.state_dict()
    assert state["interval"] == "epoch"
    assert state["frequency"] == 3

    opt2 = _single_param_optimizer()
    fresh = AF3Scheduler(opt2, **_KW)
    cfg2 = SchedulerConfig(scheduler=fresh, interval="step", frequency=1)
    cfg2.load_state_dict(state)

    assert cfg2.interval == "epoch"
    assert cfg2.frequency == 3
    assert cfg2.scheduler.last_epoch == scheduler.last_epoch
    assert cfg2.scheduler.get_last_lr() == scheduler.get_last_lr()


if __name__ == "__main__":
    pytest.main(["-v", __file__])
