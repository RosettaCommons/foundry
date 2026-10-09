"""Compare native attention with the retained independent einsum implementation."""

import copy
import os

import pytest
import torch
from rf3.model.layers.attention import TriangleAttention


def layer(device, start, channels, dtype=torch.float32):
    torch.manual_seed(21)
    module = TriangleAttention(
        channels, n_head=4, d_hidden=16, start_node=start, attention_backend="vanilla"
    ).to(device=device, dtype=dtype)
    # The default zero projection would make an incorrect attention path pass.
    with torch.no_grad():
        module.to_out.weight.normal_(std=0.1)
        module.to_out.bias.normal_(std=0.1)
    return module.eval()


@pytest.mark.parametrize("start", [True, False])
@pytest.mark.parametrize("channels", [64, 128])
@pytest.mark.parametrize("batch,length", [(1, 1), (2, 7), (1, 33)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sdpa_matches_vanilla(start, channels, batch, length, dtype):
    ref = layer("cpu", start, channels, dtype)
    fast = copy.deepcopy(ref)
    fast.attention_backend = "sdpa"
    x = torch.randn(batch, length, length, channels, dtype=dtype)
    # Non-contiguous inputs and asymmetric pairs exercise ending-node bias layout.
    x = x.transpose(1, 2)
    with torch.no_grad():
        torch.testing.assert_close(fast(x), ref(x), atol=3e-6, rtol=3e-6)
    assert ref.state_dict().keys() == fast.state_dict().keys()


@pytest.mark.parametrize("start", [True, False])
def test_explicit_sdpa_preserves_gradients(start):
    ref = layer("cpu", start, 64, torch.float64).train()
    fast = copy.deepcopy(ref)
    fast.attention_backend = "sdpa"
    x = torch.randn(2, 5, 5, 64, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    ref(x).square().sum().backward()
    fast(y).square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=1e-9, rtol=1e-9)
    for a, b in zip(ref.parameters(), fast.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-9, rtol=1e-9)


def test_auto_keeps_cpu_path(monkeypatch):
    module = layer("cpu", True, 64)
    module.attention_backend = "auto"
    monkeypatch.setattr(
        module, "_forward_sdpa", lambda *a: pytest.fail("Unexpected SDPA")
    )
    module(torch.randn(1, 3, 3, 64))


def test_invalid_backend():
    with pytest.raises(ValueError, match="attention_backend"):
        TriangleAttention(64, attention_backend="unknown")


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("RF3_RUN_MPS_TESTS") != "1" or not torch.backends.mps.is_available(),
    reason="Set RF3_RUN_MPS_TESTS=1 on a Metal host",
)
@pytest.mark.parametrize("start", [True, False])
@pytest.mark.parametrize("channels", [64, 128])
def test_auto_mps_parity(start, channels):
    ref = layer("mps", start, channels)
    fast = copy.deepcopy(ref)
    fast.attention_backend = "auto"
    x = torch.randn(2, 33, 33, channels, device="mps")
    with torch.no_grad():
        torch.testing.assert_close(fast(x), ref(x), atol=2e-5, rtol=2e-5)
