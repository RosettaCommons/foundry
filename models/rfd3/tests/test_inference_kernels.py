"""CUDA arithmetic and cache lifetime regressions for optional RFD3 kernels."""

import pytest
import torch
from rfd3.model import inference_acceleration as accel
from rfd3.model.layers.attention import dense_sdpa_pairbias_attention
from rfd3.model.layers.block_utils import build_valid_mask, group_atoms, ungroup_atoms
from rfd3.model.layers.blocks import (
    CompactStreamingDecoder,
    LocalAtomTransformer,
    LocalTokenTransformer,
)
from rfd3.model.layers.layer_utils import Transition


def test_rollout_cache_scope_and_conditioning():
    layer = torch.nn.Linear(4, 3)
    x, ref = torch.randn(2, 4), torch.randn(2, 4)
    with torch.no_grad():
        with accel.rollout_cache(constants=(x,)):
            first = accel.cached_projection(layer, x)
            assert accel.cached_projection(layer, x) is first
            assert accel.is_constant(x) and not accel.is_constant(ref)
            torch.testing.assert_close(accel.cached_projection(layer, ref), layer(ref))
        assert not accel.is_constant(x)
        layer.weight.add_(1)
        with accel.rollout_cache():
            second = accel.cached_projection(layer, x)
            torch.testing.assert_close(second, layer(x))
            assert not torch.equal(first, second)
        with pytest.raises(RuntimeError), accel.rollout_cache(constants=(x,)):
            raise RuntimeError("sampling failed")
        assert not accel.is_constant(x)


def test_training_uses_autograd_even_with_kernel_backend():
    layer = Transition(n=2, c=128)
    layer.inference_kernel_backend = "triton"
    x = torch.randn(3, 128, requires_grad=True)
    with accel.rollout_cache():
        layer(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None for p in layer.parameters())


def test_atom_layout_cache_matches_original_and_expires():
    # Uneven token sizes, gaps in token IDs, and a non-contiguous activation.
    mapping = torch.tensor([0, 0, 2, 2, 2, 5])
    x = torch.randn(2, 7, 6).transpose(1, 2)
    mask = build_valid_mask(mapping)
    padded = ungroup_atoms(x, mask)
    with torch.no_grad(), accel.rollout_cache():
        cached = build_valid_mask(mapping)
        assert build_valid_mask(mapping) is cached
        torch.testing.assert_close(cached, mask)
        actual = ungroup_atoms(x, cached)
        torch.testing.assert_close(actual, padded, atol=0, rtol=0)
        torch.testing.assert_close(group_atoms(actual, cached), x, atol=0, rtol=0)
        # Another CFG/input mapping of the same length must have its own layout.
        other = build_valid_mask(torch.tensor([0, 1, 1, 2, 2, 2]))
        assert not torch.equal(cached, other)
    with torch.no_grad(), accel.rollout_cache():
        assert build_valid_mask(mapping) is not cached
    with accel.rollout_cache():
        assert not accel.indexing_enabled()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("backend", ["torch", "triton"])
def test_atom_graphs_refresh_inputs_and_separate_fixed_layouts(backend):
    torch.manual_seed(91)
    block = {
        "n_head": 4,
        "kq_norm": True,
        "dropout": 0.0,
        "no_residual_connection_between_attention_and_transition": False,
    }
    cross = {"n_head": 4, "c_model": 128, "dropout": 0.0, "kq_norm": True}
    encoder = LocalAtomTransformer(128, 128, 16, block, 1).cuda().eval()
    decoder = (
        CompactStreamingDecoder(
            c_atom=128,
            c_atompair=16,
            c_token=64,
            c_s=32,
            c_tokenpair=16,
            atom_transformer_block=block,
            upcast={
                "method": "cross_attention",
                "n_split": 2,
                "cross_attention_block": cross,
            },
            downcast={"method": "cross_attention", "cross_attention_block": cross},
            n_blocks=2,
        )
        .cuda()
        .eval()
    )
    for root in (encoder, decoder):
        for module in root.modules():
            module.inference_kernel_backend = backend
    p = torch.randn(6, 6, 16, device="cuda")
    p_ref = torch.randn_like(p)
    mappings = [
        torch.tensor(m, device="cuda")
        for m in (
            [0, 0, 1, 2, 2, 2],
            [0, 1, 1, 1, 2, 2],
        )
    ]
    previous = []
    with (
        torch.no_grad(),
        torch.autocast("cuda", dtype=torch.bfloat16),
        accel.rollout_cache(constants=(p, p_ref)),
    ):
        for i in range(4):
            q, c = [torch.randn(2, 6, 128, device="cuda") for _ in range(2)]
            a = torch.randn(2, 3, 64, device="cuda")
            s = torch.randn(2, 3, 32, device="cuda")
            z = torch.randn(2, 3, 3, 16, device="cuda")
            indices = (
                torch.randint(6, (2, 6, 4), device="cuda")
                .sort(-1)
                .values.to(torch.int32)
            )
            pair = p if i < 3 else p_ref
            tok_idx = mappings[i % 2]
            encoder.inference_cuda_graph = decoder.inference_cuda_graph = False
            # Exercise the original mask/scatter/gather path for the reference.
            with accel.rollout_cache(constants=(pair,), atom_layout=False):
                expected_q = encoder(q, c, pair, indices=indices)
                expected = decoder(a, s, z, expected_q, c, pair, tok_idx, indices)
            encoder.inference_cuda_graph = decoder.inference_cuda_graph = True
            actual_q = encoder(q, c, pair, indices=indices)
            actual = decoder(a, s, z, actual_q, c, pair, tok_idx, indices)
            torch.testing.assert_close(actual_q, expected_q, atol=0, rtol=0)
            torch.testing.assert_close(actual[:2], expected[:2], atol=0, rtol=0)
            assert actual[2] == {}
            for old, saved in previous:
                torch.testing.assert_close(old, saved, atol=0, rtol=0)
            previous.append((actual[:2], tuple(t.clone() for t in actual[:2])))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "batch,shared,keys", [(1, True, 128), (2, True, 37), (2, False, 128)]
)
@pytest.mark.parametrize("q_dtype", [torch.float32, torch.bfloat16])
def test_gather_matches_dense_attention(batch, shared, keys, q_dtype):
    torch.manual_seed(42)
    length, channels, heads = 139, 128, 4
    q, k = [
        torch.randn(batch, length, channels, device="cuda", dtype=q_dtype)
        for _ in range(2)
    ]
    v = torch.randn_like(q, dtype=torch.bfloat16)
    gate = torch.sigmoid(torch.randn_like(v))
    # Non-contiguous pair-bias layout and duplicate indices exercise set semantics.
    b = torch.randn(
        1 if shared else batch, heads, length, length, device="cuda", dtype=v.dtype
    ).movedim(1, -1)
    idx = (
        torch.randint(
            length,
            (1 if shared else batch, length, keys),
            device="cuda",
            dtype=torch.int32,
        )
        .sort(-1)
        .values
    )
    with torch.no_grad():
        actual = accel.gather_attention(q, k, v, b, idx, heads, gate)
        expected = dense_sdpa_pairbias_attention(q, k, v, b, idx, heads, gate)
        torch.testing.assert_close(actual, expected, atol=0.012, rtol=0.025)
        torch.testing.assert_close(
            actual, accel.gather_attention(q, k, v, b, idx, heads, gate), atol=0, rtol=0
        )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "channels,n,rows",
    [(128, 2, 97), (128, 4, 4096), (256, 2, 129), (128, 3, 35), (128, 2, 0)],
)
def test_transition_matches_pytorch(channels, n, rows):
    torch.manual_seed(42)
    layer = Transition(n=n, c=channels).cuda().eval()
    x = torch.randn(rows, channels, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = layer(x)
        layer.inference_kernel_backend = "triton"
        with accel.rollout_cache():
            actual = layer(x)
        torch.testing.assert_close(actual, expected, atol=0.003, rtol=0.02)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_transition_strides_hidden_tail_and_weight_refresh():
    torch.manual_seed(14)
    layer = Transition(n=2, c=128).cuda().eval()
    hidden = 259
    layer.linear_1 = torch.nn.Linear(128, hidden, bias=False).cuda()
    layer.linear_2 = torch.nn.Linear(128, hidden, bias=False).cuda()
    layer.linear_3 = torch.nn.Linear(hidden, 128, bias=False).cuda()
    # Both a batched strided activation and non-contiguous input weights.
    layer.linear_1.weight = torch.nn.Parameter(
        layer.linear_1.weight.detach().T.contiguous().T
    )
    x = torch.randn(2, 128, 37, device="cuda").transpose(1, 2)
    keys = set(layer.state_dict())
    with (
        torch.no_grad(),
        torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False),
    ):
        for _ in range(2):
            layer.inference_kernel_backend = "torch"
            expected = layer(x)
            layer.inference_kernel_backend = "triton"
            with accel.rollout_cache():
                actual = layer(x)
            torch.testing.assert_close(actual, expected, atol=0.003, rtol=0.02)
            layer.linear_1.weight.add_(0.05)
        assert set(layer.state_dict()) == keys


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_transition_graph_replay_with_new_inputs():
    layer = Transition(n=2, c=128).cuda().eval()
    layer.inference_kernel_backend = "triton"

    def call(x, *, full):
        return layer(x)

    with (
        torch.no_grad(),
        torch.autocast("cuda", dtype=torch.bfloat16),
        accel.rollout_cache(),
    ):
        saved = []
        for _ in range(2):
            x = torch.randn(39, 128, device="cuda")
            expected = layer(x)
            actual = accel.graph_call(layer, call, (x,), full=True)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            saved.append((actual, actual.clone()))
        for actual, copy in saved:
            torch.testing.assert_close(actual, copy, atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_graph_refreshes_inputs_and_preserves_previous_outputs():
    layer = torch.nn.Linear(32, 32).cuda().eval()

    def call(x, *, full):
        return layer(x).sin()

    x = torch.randn(5, 32, device="cuda")
    y = torch.randn_like(x)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        with accel.rollout_cache():
            a = accel.graph_call(layer, call, (x,), full=True)
            a_copy = a.clone()
            b = accel.graph_call(layer, call, (y,), full=True)
            torch.testing.assert_close(a, a_copy, atol=0, rtol=0)
            torch.testing.assert_close(a, call(x, full=True), atol=0, rtol=0)
            torch.testing.assert_close(b, call(y, full=True), atol=0, rtol=0)
            shorter = y[:3]
            c = accel.graph_call(layer, call, (shorter,), full=True)
            torch.testing.assert_close(c, call(shorter, full=True), atol=0, rtol=0)
        with accel.rollout_cache():
            layer.weight.add_(0.1)
            # Invalidate PyTorch's own eager autocast cache after a weight edit.
            torch.clear_autocast_cache()
            c = accel.graph_call(layer, call, (x,), full=True)
            torch.testing.assert_close(c, call(x, full=True), atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_token_stack_graph_matches_eager_with_changed_conditioning_and_topology():
    torch.manual_seed(8)
    stack = (
        LocalTokenTransformer(
            c_token=64,
            c_s=32,
            c_tokenpair=16,
            n_block=2,
            diffusion_transformer_block={
                "n_head": 4,
                "kq_norm": True,
                "dropout": 0.0,
                "no_residual_connection_between_attention_and_transition": False,
            },
        )
        .cuda()
        .eval()
    )
    state_keys = set(stack.state_dict())
    with (
        torch.no_grad(),
        torch.autocast("cuda", dtype=torch.bfloat16),
        accel.rollout_cache(),
    ):
        for _ in range(2):
            a = torch.randn(2, 17, 64, device="cuda")
            s = torch.randn(2, 17, 32, device="cuda")
            z = torch.randn(2, 17, 17, 16, device="cuda")
            indices = torch.randint(17, (2, 17, 8), device="cuda")
            expected = stack._forward_blocks(a, s, z, indices, full=True)
            actual = accel.graph_call(
                stack, stack._forward_blocks, (a, s, z, indices), full=True
            )
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert set(stack.state_dict()) == state_keys
