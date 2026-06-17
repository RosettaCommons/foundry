"""Unit tests for foundry.training.EMA.

`EMA` keeps a shadow copy of a model whose parameters track the live model via
the exponential-moving-average update
``shadow -= (1 - decay) * (shadow - param)``. The contracts worth pinning are
numeric and behavioural: the update applies that exact formula, only touches
parameters that require grad, copies buffers verbatim (not EMA'd), refuses to
run outside training mode, and `forward` dispatches to the live model while
training and to the shadow while evaluating.
"""

import pytest
import torch
import torch.nn as nn

from foundry.training.EMA import EMA


class _TinyModel(nn.Module):
    """Minimal module with two parameters and a buffer for exercising EMA."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, 3))
        self.bias = nn.Parameter(torch.zeros(2))
        self.register_buffer("counter", torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.t() + self.bias


def test_shadow_is_detached_at_init():
    """The shadow starts as a copy whose parameters are off the autograd graph."""
    ema = EMA(_TinyModel(), decay=0.9)
    assert all(not p.requires_grad for p in ema.shadow.parameters())
    assert all(p.requires_grad for p in ema.model.parameters())


def test_update_applies_ema_formula():
    """shadow moves toward the live param by exactly (1 - decay) of the gap."""
    model = _TinyModel()
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema = EMA(model, decay=0.9)  # shadow.weight captured at 1.0
    with torch.no_grad():
        model.weight.fill_(2.0)

    ema.train()
    ema.update()

    # 1.0 - (1 - 0.9) * (1.0 - 2.0) = 1.0 + 0.1 = 1.1
    assert torch.allclose(ema.shadow.weight, torch.full((2, 3), 1.1))


def test_update_skips_frozen_params():
    """A parameter with requires_grad=False is left untouched by the update."""
    model = _TinyModel()
    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.fill_(1.0)
    model.bias.requires_grad_(False)
    ema = EMA(model, decay=0.5)
    with torch.no_grad():
        model.weight.fill_(3.0)
        model.bias.fill_(3.0)

    ema.train()
    ema.update()

    # weight is trainable: 1.0 - 0.5 * (1.0 - 3.0) = 2.0
    assert torch.allclose(ema.shadow.weight, torch.full((2, 3), 2.0))
    # bias is frozen: unchanged from its captured value
    assert torch.allclose(ema.shadow.bias, torch.full((2,), 1.0))


def test_update_copies_buffers_verbatim():
    """Buffers are copied, not exponentially averaged."""
    model = _TinyModel()
    ema = EMA(model, decay=0.5)  # shadow.counter captured at 0.0
    with torch.no_grad():
        model.counter.fill_(7.0)

    ema.train()
    ema.update()

    # A copy gives 7.0; an EMA with decay 0.5 from 0.0 would give 3.5.
    assert torch.allclose(ema.shadow.counter, torch.full((1,), 7.0))


def test_update_raises_outside_training():
    ema = EMA(_TinyModel(), decay=0.9)
    ema.eval()
    with pytest.raises(RuntimeError, match="during training"):
        ema.update()


def test_forward_dispatches_model_in_train_shadow_in_eval():
    """Training routes to the live model; evaluation routes to the shadow."""
    model = _TinyModel()
    ema = EMA(model, decay=0.9)
    with torch.no_grad():
        ema.shadow.bias.fill_(5.0)  # make the shadow differ from the model
    x = torch.zeros(4, 3)

    ema.train()
    assert torch.allclose(ema(x), torch.zeros(4, 2))  # model bias is 0

    ema.eval()
    assert torch.allclose(ema(x), torch.full((4, 2), 5.0))  # shadow bias is 5


if __name__ == "__main__":
    pytest.main(["-v", __file__])
