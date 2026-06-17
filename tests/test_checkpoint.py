"""Unit tests for foundry.training.checkpoint.

`create_custom_forward` adapts a kwargs-taking callable into the positional-only
shape `torch.utils.checkpoint` requires, by binding the kwargs into a closure.
`activation_checkpointing` decorates a function so it runs through gradient
checkpointing when grad is enabled and calls through directly otherwise. The
tests pin the kwarg binding, both branches, and that gradients still flow (and
match the non-checkpointed result) on the checkpointed path.
"""

import pytest
import torch

from foundry.training.checkpoint import activation_checkpointing, create_custom_forward


def test_create_custom_forward_binds_fixed_kwargs():
    """Bound kwargs are supplied to the wrapped callable on each call."""
    forward = create_custom_forward(lambda a, b: a + b, b=10)
    assert forward(5) == 15


def test_create_custom_forward_forwards_positional_inputs():
    """All positional inputs pass through in order, alongside the fixed kwargs."""
    forward = create_custom_forward(lambda a, b, c: (a, b, c), c=3)
    assert forward(1, 2) == (1, 2, 3)


def test_activation_checkpointing_no_grad_calls_directly():
    """With grad disabled the decorator just calls the function."""

    @activation_checkpointing
    def double(x: torch.Tensor) -> torch.Tensor:
        return x * 2

    with torch.no_grad():
        out = double(torch.tensor([1.0, 2.0]))
    assert torch.allclose(out, torch.tensor([2.0, 4.0]))


def test_activation_checkpointing_grad_enabled_matches_and_backprops():
    """The checkpointed path returns the same value and propagates gradients."""

    def square_sum(x: torch.Tensor) -> torch.Tensor:
        return (x**2).sum()

    checkpointed = activation_checkpointing(square_sum)
    x = torch.tensor([3.0], requires_grad=True)
    out = checkpointed(x)
    out.backward()

    assert torch.allclose(out, torch.tensor(9.0))
    assert torch.allclose(x.grad, torch.tensor([6.0]))  # d/dx x^2 = 2x


def test_activation_checkpointing_forwards_kwargs_through_checkpoint():
    """Keyword arguments reach the function via the checkpointed path."""

    def scale_sum(x: torch.Tensor, *, scale: float) -> torch.Tensor:
        return (x * scale).sum()

    checkpointed = activation_checkpointing(scale_sum)
    x = torch.tensor([2.0], requires_grad=True)
    out = checkpointed(x, scale=4.0)
    out.backward()

    assert torch.allclose(out, torch.tensor(8.0))
    assert torch.allclose(x.grad, torch.tensor([4.0]))


if __name__ == "__main__":
    pytest.main(["-v", __file__])
