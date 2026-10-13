"""Pooling semantics, precision and reproducibility independent of model weights."""

import pytest
import torch

from foundry.utils.torch import scatter_mean

DEVICES = (
    ["cpu"]
    + (["cuda"] if torch.cuda.is_available() else [])
    + (["mps"] if torch.backends.mps.is_available() else [])
)


def reference(destination, dim, index, source):
    # Compute the mathematical mean of the already-quantized values, excluding self.
    parts = []
    for i in range(destination.shape[dim]):
        selected = source.index_select(dim, (index == i).nonzero().flatten())
        parts.append(
            selected.cpu()
            .double()
            .mean(dim)
            .to(device=source.device, dtype=source.dtype)
            if selected.shape[dim]
            else destination.select(dim, i)
        )
    return torch.stack(parts, dim)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_mean_semantics(device, dtype, dim):
    if device == "mps" and dtype == torch.float64:
        pytest.skip("MPS has no float64")
    torch.manual_seed(123)
    # Transpose produces non-contiguous inputs and tests arbitrary batch axes.
    source = (
        torch.randn(3, 7, 2, device=device, dtype=dtype).transpose(0, 2).movedim(1, dim)
    )
    dest = (
        torch.randn(3, 5, 2, device=device, dtype=dtype).transpose(0, 2).movedim(1, dim)
    )
    index = torch.tensor([3, 0, 3, 4, 0, 3, 4], device=device, dtype=torch.int32)
    before = dest.clone()
    expected = reference(dest, dim, index, source)
    result = scatter_mean(dest, dim, index, source)
    torch.testing.assert_close(result, expected)
    assert torch.equal(dest, before)
    assert result.dtype == dtype
    torch.testing.assert_close(
        result.select(dim, 1), dest.select(dim, 1), rtol=0, atol=0
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("n", [14, 23, 64, 300, 3000])
def test_low_precision_mean_uses_exact_counts(device, dtype, n):
    # At large n the old CUDA BF16 reduction returns 2.0 instead of ~0.5.
    source = torch.linspace(-1, 2, n, device=device, dtype=dtype).reshape(n, 1)
    index = torch.zeros(n, device=device, dtype=torch.long)
    result = scatter_mean(
        torch.full((1, 1), 100, device=device, dtype=dtype), 0, index, source
    )
    expected = source.float().mean(0, keepdim=True).to(dtype)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_empty_input_and_destination_gradients(device):
    dest = torch.randn(2, 4, 3, device=device, requires_grad=True)
    source = torch.empty(2, 0, 3, device=device, requires_grad=True)
    result = scatter_mean(
        dest, 1, torch.empty(0, device=device, dtype=torch.long), source
    )
    torch.testing.assert_close(result, dest, rtol=0, atol=0)
    result.sum().backward()
    assert torch.equal(dest.grad, torch.ones_like(dest))
    assert source.grad.shape == source.shape


@pytest.mark.parametrize("device", [d for d in DEVICES if d != "mps"])
def test_first_and_second_derivatives(device):
    dest = torch.randn(2, 4, 2, device=device, dtype=torch.double, requires_grad=True)
    source = torch.randn(2, 5, 2, device=device, dtype=torch.double, requires_grad=True)
    index = torch.tensor([2, 0, 2, 0, 2], device=device)
    fn = lambda d, s: scatter_mean(d, 1, index, s)
    assert torch.autograd.gradcheck(fn, (dest, source))
    assert torch.autograd.gradgradcheck(fn, (dest, source))
    fn(dest, source).sum().backward()
    expected = torch.tensor([0, 1, 0, 1], device=device).view(1, 4, 1).expand_as(dest)
    torch.testing.assert_close(dest.grad, expected.double())
    expected_src = (
        torch.tensor(
            [1 / 3, 1 / 2, 1 / 3, 1 / 2, 1 / 3], device=device, dtype=torch.double
        )
        .view(1, 5, 1)
        .expand_as(source)
    )
    torch.testing.assert_close(source.grad, expected_src)


@pytest.mark.parametrize(
    "index",
    [
        torch.tensor([-1, 0]),
        torch.tensor([0, 3]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([[0, 1]]),
    ],
)
def test_invalid_indices_rejected(index):
    with pytest.raises((RuntimeError, IndexError, ValueError)):
        scatter_mean(torch.zeros(3, 2), 0, index, torch.ones(2, 2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_repeatability_with_and_without_global_determinism():
    original = torch.are_deterministic_algorithms_enabled()
    original_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.manual_seed(789)
        source = torch.randn(
            2, 2800, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        index = torch.randperm(2800, device="cuda") % 200
        dest = torch.zeros(2, 200, 64, device="cuda", dtype=source.dtype)
        outputs, grads = [], []
        for strict in [False, False, True, True]:
            torch.use_deterministic_algorithms(strict)
            result = scatter_mean(dest, 1, index, source)
            grad = torch.autograd.grad(result.square().float().sum(), source)[0]
            assert torch.are_deterministic_algorithms_enabled() == strict
            outputs.append(result.detach())
            grads.append(grad)
        assert all(torch.equal(outputs[0], x) for x in outputs[1:])
        assert all(torch.equal(grads[0], x) for x in grads[1:])
    finally:
        torch.use_deterministic_algorithms(original, warn_only=original_warn)


@pytest.mark.parametrize("device", DEVICES)
def test_empty_destination(device):
    source = torch.empty(2, 0, 3, device=device, requires_grad=True)
    destination = torch.empty_like(source, requires_grad=True)
    output = scatter_mean(
        destination, 1, torch.empty(0, device=device, dtype=torch.long), source
    )
    assert output.shape == source.shape
    output.sum().backward()
    assert source.grad.shape == source.shape
    assert destination.grad.shape == destination.shape
