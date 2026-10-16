"""Independent einsum oracles for dense trunk and diffusion token attention."""

import copy
import os

import pytest
import torch
from rf3.model.layers.af3_diffusion_transformer import AttentionPairBiasDiffusion
from rf3.model.layers.pairformer_layers import AttentionPairBiasPairformerDeepspeed


def check(kind, leading, dtype, device="cpu", gradients=False, pair_leading=()):
    torch.manual_seed(9)
    if kind == "pairformer":
        ref = AttentionPairBiasPairformerDeepspeed(48, 24, 8, 4)
    else:
        ref = AttentionPairBiasDiffusion(48, 24, 8, 4, kind == "normalized")
    ref = ref.to(device=device, dtype=dtype).eval()
    ref.dense_attention_backend = "vanilla"
    # Exercise all output/gating parameters, including zero-initialized AdaLN.
    with torch.no_grad():
        for p in ref.parameters():
            if p.count_nonzero() == 0:
                p.normal_(std=0.1)
    fast = copy.deepcopy(ref)
    fast.dense_attention_backend = "sdpa"
    a = torch.randn(
        *leading, 9, 48, device=device, dtype=dtype, requires_grad=gradients
    )
    s = (
        None
        if kind == "pairformer"
        else torch.randn(*leading, 9, 24, device=device, dtype=dtype)
    )
    # Pair bias is shared across diffusion batches; asymmetric, noncontiguous.
    z = torch.randn(*pair_leading, 9, 9, 8, device=device, dtype=dtype).transpose(
        -3, -2
    )
    beta = torch.randn(9, 9, device=device, dtype=dtype) if s is None else None
    x = ref(a, s, z, beta)
    y = fast(a, s, z, beta)
    torch.testing.assert_close(y, x, atol=2e-5 if device == "mps" else 2e-6, rtol=2e-5)
    assert x.abs().max() > 0
    assert ref.state_dict().keys() == fast.state_dict().keys()
    if gradients:
        x.square().sum().backward()
        grad = a.grad.clone()
        a.grad = None
        y.square().sum().backward()
        torch.testing.assert_close(a.grad, grad, atol=1e-9, rtol=1e-9)
        for p, q in zip(ref.parameters(), fast.parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("kind", ["pairformer", "diffusion", "normalized"])
@pytest.mark.parametrize("leading", [(), (1,), (3,), (2, 3)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dense_parity(kind, leading, dtype):
    with torch.no_grad():
        check(kind, leading, dtype)


@pytest.mark.parametrize("kind", ["pairformer", "diffusion", "normalized"])
def test_dense_gradients(kind):
    check(kind, (2,), torch.float64, gradients=True)


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("RF3_RUN_MPS_TESTS") != "1" or not torch.backends.mps.is_available(),
    reason="Requires opt-in Metal host",
)
@pytest.mark.parametrize("kind", ["pairformer", "diffusion", "normalized"])
@pytest.mark.parametrize("leading", [(), (3,), (2, 3)])
def test_dense_mps(kind, leading):
    with torch.no_grad():
        check(kind, leading, torch.float32, "mps")


def test_atom_path_unchanged(monkeypatch):
    m = AttentionPairBiasDiffusion(16, 16, 8, 4, True, dense_attention_backend="sdpa")
    sentinel = torch.tensor(19.0)
    monkeypatch.setattr(m, "atom_attention", lambda *a: sentinel)
    assert (
        m(
            torch.randn(1, 4, 16),
            torch.randn(1, 4, 16),
            torch.randn(4, 4, 8),
            torch.ones(4, 4),
        )
        is sentinel
    )


@pytest.mark.parametrize("kind", ["pairformer", "normalized"])
@pytest.mark.parametrize(
    "leading,pair_leading", [((), (1,)), ((), (2,)), ((3,), (2, 1))]
)
def test_bias_adds_leading_batch_axes(kind, leading, pair_leading):
    with torch.no_grad():
        check(kind, leading, torch.float32, pair_leading=pair_leading)


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("RF3_RUN_MPS_TESTS") != "1" or not torch.backends.mps.is_available(),
    reason="Requires opt-in Metal host",
)
def test_confidence_head_bias_broadcast_mps():
    with torch.no_grad():
        check("pairformer", (), torch.float32, "mps", pair_leading=(1,))
