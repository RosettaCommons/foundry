"""Exact CUDA arithmetic, dispatch boundaries and unmodified training semantics."""

import copy
from contextlib import nullcontext

import pytest
import torch
from rf3.model.layers.inference_kernels import (
    ExactLayerNorm,
    set_exact_kernels,
    validated_stack,
)
from torch import nn


def assert_bits(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert torch.equal(
        actual.contiguous().reshape(-1).view(torch.uint8),
        expected.contiguous().reshape(-1).view(torch.uint8),
    )


def test_install_preserves_checkpoint_parameters_aliases_and_switch():
    norm = nn.LayerNorm(128)
    model = nn.ModuleDict({"a": norm, "b": norm, "small": nn.LayerNorm(32)}).eval()
    keys = list(model.state_dict())
    parameters = list(model.parameters())
    set_exact_kernels(model, True)
    assert isinstance(model["a"], ExactLayerNorm)
    assert model["a"] is model["b"]
    assert type(model["small"]) is nn.LayerNorm
    assert all(a is b for a, b in zip(parameters, model.parameters()))
    assert keys == list(model.state_dict())
    set_exact_kernels(model, False)
    assert not model["a"].exact_enabled
    set_exact_kernels(model, True)
    assert model["a"].exact_enabled
    assert not model["a"].training


@pytest.mark.parametrize("training", [False, True])
def test_cpu_forward_and_backward(training):
    ref = nn.Sequential(nn.LayerNorm(128)).train(training)
    actual = copy.deepcopy(ref)
    set_exact_kernels(actual, True)
    x = torch.randn(13, 128, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    a, b = actual(x), ref(y)
    assert_bits(a, b)
    a.square().sum().backward()
    b.square().sum().backward()
    assert_bits(x.grad, y.grad)
    for p, q in zip(actual.parameters(), ref.parameters()):
        assert_bits(p.grad, q.grad)


def test_hook_and_custom_subclass_retained():
    class CustomNorm(nn.LayerNorm):
        pass

    norm = nn.LayerNorm(128)
    norm.register_forward_hook(lambda m, a, out: out + 1)
    model = nn.Sequential(norm, CustomNorm(128))
    set_exact_kernels(model, True)
    assert model[0] is norm
    assert type(model[1]) is CustomNorm


def test_unknown_stack_falls_back(monkeypatch):
    monkeypatch.setattr(torch, "__version__", "99.0.0")
    validated_stack.cache_clear()
    assert not validated_stack(0)
    validated_stack.cache_clear()


@pytest.fixture
def cuda_stack():
    if not torch.cuda.is_available() or not validated_stack(
        torch.cuda.current_device()
    ):
        pytest.skip("Requires the validated H200 / Torch 2.11 cu128 stack")


@pytest.mark.gpu
@pytest.mark.parametrize("width", [64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "layout", ["contiguous", "transpose", "column_stride", "expand"]
)
@pytest.mark.parametrize("affine", ["both", "weight", "none"])
def test_cuda_layer_norm_bits(cuda_stack, monkeypatch, width, dtype, layout, affine):
    from rf3.model.kernels import exact_layer_norm

    torch.manual_seed(37)
    norm = (
        nn.LayerNorm(width, elementwise_affine=affine != "none", bias=affine == "both")
        .cuda()
        .eval()
    )
    with torch.no_grad():
        for p in norm.parameters():
            p.normal_()
    model = nn.Sequential(copy.deepcopy(norm))
    set_exact_kernels(model, True)
    x = torch.randn(384, 384, width, device="cuda", dtype=dtype)
    if layout == "transpose":
        x = x.transpose(0, 1)
    elif layout == "column_stride":
        x = x.transpose(1, 2).contiguous().transpose(1, 2)
    elif layout == "expand":
        x = x[:1].expand(384, -1, -1)
    launches = []
    original = exact_layer_norm.layer_norm

    def counted(*a, **kw):
        out = original(*a, **kw)
        launches.append(True)
        return out

    monkeypatch.setattr(exact_layer_norm, "layer_norm", counted)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        assert_bits(model(x), norm(x))
    assert launches, "test passed by falling back to native"


@pytest.mark.gpu
@pytest.mark.parametrize("width", [64, 128, 256])
@pytest.mark.parametrize(
    "kind", ["constant", "near_constant", "dynamic_range", "signed_zero"]
)
@pytest.mark.parametrize("eps", [1e-5, 1e-3])
def test_cuda_adversarial_values(cuda_stack, width, kind, eps):
    torch.manual_seed(19)
    ref = nn.LayerNorm(width, eps=eps).cuda().eval()
    model = nn.Sequential(copy.deepcopy(ref))
    set_exact_kernels(model, True)
    x = torch.randn(131075, width, device="cuda")  # partial last block
    if kind == "constant":
        x.fill_(2.0)
    elif kind == "near_constant":
        x = 1000 + x * 1e-4
    elif kind == "dynamic_range":
        x *= torch.logspace(-6, 6, width, device="cuda")
    else:
        x = torch.copysign(torch.zeros_like(x), x)
    with torch.no_grad():
        assert_bits(model(x), ref(x))


@pytest.mark.gpu
@pytest.mark.parametrize(
    "reason",
    [
        "disabled",
        "training",
        "grad",
        "small",
        "plain_bf16",
        "bf16_parameters",
        "misaligned_weight",
        "misaligned_input",
    ],
)
def test_cuda_fallbacks(cuda_stack, monkeypatch, reason):
    from rf3.model.kernels import exact_layer_norm

    ref = nn.LayerNorm(128).cuda().eval()
    model = nn.Sequential(copy.deepcopy(ref))
    set_exact_kernels(model, True)
    x = torch.randn(131072, 128, device="cuda")
    if reason == "disabled":
        set_exact_kernels(model, False)
    elif reason == "training":
        model.train()
    elif reason == "small":
        x = x[:20]
    elif reason in ("plain_bf16", "bf16_parameters"):
        x = x.bfloat16()
        if reason == "bf16_parameters":
            model.bfloat16()
            ref.bfloat16()
    elif reason == "misaligned_weight":
        weight = nn.Parameter(torch.randn(129, device="cuda")[1:])
        ref.weight = weight
        model[0].weight = weight
    elif reason == "misaligned_input":
        x = torch.randn(x.numel() + 1, device="cuda")[1:].view_as(x)

    # Check actual launches, allowing Unsupported to be raised during planning.
    def forbidden(*a, **kw):
        raise AssertionError("native fallback launched a CUDA kernel")

    monkeypatch.setattr(exact_layer_norm._Launcher, "launch", forbidden)
    with nullcontext() if reason == "grad" else torch.no_grad():
        if reason == "plain_bf16":
            # This CUDA ATen path rejects mixed parameters without autocast.
            with pytest.raises(RuntimeError) as expected:
                ref(x)
            with pytest.raises(RuntimeError) as actual:
                model(x)
            assert str(actual.value) == str(expected.value)
        else:
            assert_bits(model(x), ref(x))


@pytest.mark.gpu
def test_cuda_nondefault_stream(cuda_stack):
    model = nn.Sequential(nn.LayerNorm(128)).cuda().eval()
    ref = copy.deepcopy(model)
    set_exact_kernels(model, True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.no_grad():
        x = torch.randn(131072, 128, device="cuda")
        a, b = model(x), ref(x)
    stream.synchronize()
    assert_bits(a, b)
