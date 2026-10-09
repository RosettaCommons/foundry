"""Custom kernels preserve padded example/sample axes and fullgraph capture."""

import pytest
import torch
from rfd3.model.kernel_ops import padded_attention
from rfd3.model.layers.batched import initialize, masked_softmax
from rfd3.model.layers.layer_utils import Transition
from rfd3_batch_fixtures import example, tiny_model
from test_rfd3_batch_model import prepared

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sparse_slots_masks_duplicates_and_compilation(dtype):
    torch.manual_seed(10)
    b, d, n, c, h, keys = 3, 2, 19, 32, 2, 13
    q, k = [torch.randn(b, d, n, c, device="cuda", dtype=dtype) for _ in range(2)]
    v = torch.randn_like(q, dtype=torch.bfloat16)
    gate = torch.sigmoid(torch.randn_like(v))
    # Strided bias, duplicate and unsorted slots, different masks for each B,D.
    bias = torch.randn(b, d, h, n, keys, device="cuda", dtype=v.dtype).movedim(2, -1)
    idx = torch.randint(n, (b, d, n, keys), device="cuda")
    valid = torch.rand(b, d, n, keys, device="cuda") > 0.3
    valid[1, :, 8:] = False
    valid[2] = False  # entirely padded example
    bi = torch.arange(b, device="cuda")[:, None, None, None]
    di = torch.arange(d, device="cuda")[None, :, None, None]
    kg = k[bi, di, idx].unflatten(-1, (h, c // h))
    vg = v[bi, di, idx].unflatten(-1, (h, c // h))
    scores = (q.unflatten(-1, (h, c // h)).unsqueeze(-3) * kg).sum(-1)
    scores = scores / (c // h) ** 0.5 + bias
    p = masked_softmax(scores.transpose(-1, -2), valid.unsqueeze(-2)).transpose(-1, -2)
    expected = (p[..., None] * vg).sum(-3).flatten(-2) * gate
    compiled = torch.compile(padded_attention, fullgraph=True)
    args = (q, k, v, bias, idx, valid, gate, h)
    with torch.no_grad():
        actual = padded_attention(*args)
        captured = compiled(*args)
    tolerance = 0.006 if dtype == torch.bfloat16 else 2e-6
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(captured, actual, atol=0, rtol=0)
    assert (actual[2] == 0).all() and torch.isfinite(actual).all()
    torch.library.opcheck(padded_attention, args)


def test_compiled_transition_uses_current_weights():
    layer = Transition(n=2, c=128).cuda().eval()
    layer.inference_kernel_backend = "triton"
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return torch._inductor.compile(graph, inputs)

    compiled = torch.compile(layer, backend=backend, fullgraph=True)
    x = torch.randn(2, 3, 17, 128, device="cuda")
    # Mutating weights within an autocast region requires disabling PyTorch's
    # eager cast cache, independently of the compiled operator's behavior.
    with (
        torch.no_grad(),
        torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False),
    ):
        first = compiled(x)
        for _ in range(2):
            layer.linear_3.weight.add_(0.02)
            layer.inference_kernel_backend = "torch"
            expected = layer(x)
            layer.inference_kernel_backend = "triton"
            actual = compiled(x)
            torch.testing.assert_close(actual, expected, atol=0.012, rtol=0.025)
        assert not torch.equal(first, actual)
    assert len(graphs) == 1
    assert "rfd3.transition" in graphs[0].code


@pytest.mark.parametrize("bf16", [False, True])
def test_fullgraph_mixed_examples_and_dummy_padding(bf16):
    torch.set_num_threads(4)
    model = tiny_model(bf16=bf16).cuda()
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return torch._inductor.compile(graph, inputs)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        f, x = prepared([example(d=2), example((1, 2), d=2, seed=9)], model, b=3)
        f = {key: value.cuda() for key, value in f.items()}
        x = x.cuda().masked_fill(~f["atom_valid"][:, None, :, None], float("nan"))
        init = initialize(model.token_initializer, f)
        t = torch.ones(3, 2, device="cuda")
        original = model.diffusion_module.forward_batched
        expected = original(x, t, f, **init)
        for module in model.modules():
            module.inference_kernel_backend = "triton"
        compiled = torch.compile(original, backend=backend, fullgraph=True)
        actual = compiled(x, t, f, **init)
        for key in ("X_L", "sequence_logits_I"):
            tolerance = 0.025 if bf16 else 3e-5
            torch.testing.assert_close(
                actual[key], expected[key], atol=tolerance, rtol=tolerance
            )
            assert (actual[key][2] == 0).all()
        changed = x.clone()
        changed[0, 1] += 10
        changed[1] += 100
        second = compiled(changed, t, f, **init)
        torch.testing.assert_close(second["X_L"][0, 0], actual["X_L"][0, 0])
        assert not torch.equal(second["X_L"][0, 1], actual["X_L"][0, 1])
    assert len(graphs) == 1
    assert "rfd3.padded_attention" in graphs[0].code
