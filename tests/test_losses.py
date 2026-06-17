"""Unit tests for foundry.metrics.losses.

`Loss` aggregates a set of child loss functions: its `forward` sums their scalar
losses, merges their per-loss dicts, and records the (detached) running total
under `total_loss` while still returning the grad-carrying sum. The child losses
are normally Hydra-instantiated; here we set `to_compute` directly with stubs to
exercise the aggregation logic without a config.
"""

import pytest
import torch

from foundry.metrics.losses import Loss


def _stub_loss(value: float, extra: dict, requires_grad: bool = False):
    """A child loss returning a fixed scalar tensor and a fixed loss dict."""
    tensor = torch.tensor(value, requires_grad=requires_grad)

    def loss_fn(network_input, network_output, loss_input):
        return tensor, dict(extra)

    return loss_fn


def test_empty_loss_has_no_children():
    assert Loss().to_compute == []


def test_forward_sums_children_and_merges_dicts():
    loss = Loss()
    loss.to_compute = [
        _stub_loss(1.0, {"a": 10}),
        _stub_loss(2.0, {"b": 20}),
    ]

    total, loss_dict = loss({}, {}, {})

    assert torch.allclose(total, torch.tensor(3.0))
    assert loss_dict["a"] == 10
    assert loss_dict["b"] == 20
    assert torch.allclose(loss_dict["total_loss"], torch.tensor(3.0))


def test_forward_total_loss_is_detached_but_returned_loss_keeps_grad():
    loss = Loss()
    loss.to_compute = [
        _stub_loss(2.0, {}, requires_grad=True),
        _stub_loss(3.0, {}, requires_grad=True),
    ]

    total, loss_dict = loss({}, {}, {})

    # The returned aggregate still carries grad for the backward pass...
    assert total.requires_grad
    # ...while the logged copy is detached.
    assert not loss_dict["total_loss"].requires_grad
    assert torch.allclose(loss_dict["total_loss"], torch.tensor(5.0))


if __name__ == "__main__":
    pytest.main(["-v", __file__])
