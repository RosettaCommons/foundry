"""Dense SDPA preserves padding, topology, B/D isolation and legacy results."""

import copy
import os

import pytest
import torch
from rfd3.model.layers.attention import LocalAttentionPairBias
from rfd3.model.layers.batched import initialize, local_attention
from rfd3_batch_fixtures import example, tiny_model
from test_rfd3_batch_model import prepared


def set_backend(model, backend):
    for m in model.modules():
        if hasattr(m, "dense_attention_backend"):
            m.dense_attention_backend = backend


def check_model(device, bf16=False):
    model = tiny_model(bf16=bf16).to(device)
    ref = copy.deepcopy(model)
    set_backend(ref, "vanilla")
    set_backend(model, "sdpa")
    exs = [example(d=3), example((1, 2), d=3, seed=9)]
    with torch.no_grad():
        f, x = prepared(exs, model, b=3)
        f = {k: v.to(device) for k, v in f.items()}
        x = x.to(device).masked_fill(~f["atom_valid"][:, None, :, None], float("nan"))
        t = torch.rand(3, 3, device=device) + 0.5

        def run(m):
            return m.diffusion_module.forward_batched(
                x, t, f, **initialize(m.token_initializer, f), n_recycle=2
            )

        expected, actual = run(ref), run(model)
        for key in ("X_L", "sequence_logits_I"):
            torch.testing.assert_close(
                actual[key],
                expected[key],
                atol=0.02 if bf16 else 3e-5,
                rtol=0.02 if bf16 else 3e-5,
            )
            assert torch.isfinite(actual[key]).all()
            assert (actual[key][2] == 0).all()
        # Legacy full-attention mask uses the same neighbor topology.
        for b, ex in enumerate(exs):
            lf = {k: v.to(device) for k, v in copy.deepcopy(ex["feats"]).items()}
            legacy = ref.diffusion_module(
                ex["coord_atom_lvl_to_be_noised"].to(device),
                t[b],
                lf,
                **ref.token_initializer(lf),
                n_recycle=2,
            )
            n = len(ex["feats"]["restype"])
            torch.testing.assert_close(
                actual["sequence_logits_I"][b, :, :n],
                legacy["sequence_logits_I"],
                atol=0.03 if bf16 else 3e-5,
                rtol=0.03 if bf16 else 3e-5,
            )


@pytest.mark.parametrize("bf16", [False, True])
def test_full_model_parity(bf16):
    check_model("cpu", bf16)


def test_sparse_topology_duplicate_padding_keys_and_gradients():
    torch.manual_seed(4)
    m = LocalAttentionPairBias(24, None, 8, 2).double().eval()
    valid = torch.tensor([[True, True, True, False], [False] * 4])
    q = torch.randn(2, 3, 4, 24, dtype=torch.float64, requires_grad=True)
    z = torch.randn(2, 1, 4, 4, 8, dtype=torch.float64, requires_grad=True)
    # Invalid duplicate key 0 must not erase the first real slot.
    ix = torch.zeros(2, 3, 4, 2, dtype=torch.long)
    allowed = torch.zeros_like(ix, dtype=torch.bool)
    allowed[0, :, :3, 0] = True
    ref = local_attention(m, q, None, z, ix, allowed, valid, full=True)
    fast = local_attention(m, q, None, z, ix, allowed, valid, sdpa=True)
    torch.testing.assert_close(fast, ref, atol=1e-10, rtol=1e-10)
    assert torch.isfinite(fast).all() and (fast[1] == 0).all()
    a = torch.autograd.grad(ref.square().sum(), (q, z), retain_graph=True)
    b = torch.autograd.grad(fast.square().sum(), (q, z))
    for got, want in zip(b, a):
        torch.testing.assert_close(got, want, atol=1e-10, rtol=1e-10)
        assert torch.isfinite(got).all() and (got[1] == 0).all()


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("RFD3_RUN_MPS_COMPILE_TESTS") != "1"
    or not torch.backends.mps.is_available(),
    reason="Requires opt-in Metal host",
)
def test_full_model_mps():
    check_model("mps")


def test_legacy_token_topology_and_sparse_atom_dispatch(monkeypatch):
    import rfd3.model.layers.attention as attention

    torch.manual_seed(10)
    m = LocalAttentionPairBias(24, None, 8, 2).eval()
    q = torch.randn(3, 7, 24)
    z = torch.randn(3, 7, 7, 8)
    # Deliberately excludes most pairs, including asymmetric neighborhoods.
    ix = torch.stack([torch.arange(7), torch.arange(7).roll(1)], -1)[None].expand(
        3, -1, -1
    )
    m.dense_attention_backend = "vanilla"
    with torch.no_grad():
        expected = m(q, None, z, indices=ix, full=True)
        m.dense_attention_backend = "sdpa"
        torch.testing.assert_close(
            m(q, None, z, indices=ix, full=True), expected, atol=2e-6, rtol=2e-6
        )
        monkeypatch.setattr(
            attention,
            "dense_pair_attention",
            lambda *a, **k: pytest.fail("Sparse atom dispatch changed"),
        )
        sparse = m(q, None, z, indices=ix, full=False)
        torch.testing.assert_close(sparse, expected, atol=2e-6, rtol=2e-6)


def test_engine_dense_backend_reaches_all_copies(monkeypatch):
    from types import SimpleNamespace

    from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine

    from foundry.inference_engines.base import BaseInferenceEngine

    obj = RFD3InferenceEngine(**RFD3InferenceConfig(dense_attention_backend="sdpa"))
    copies = torch.nn.ModuleList([tiny_model(), tiny_model()])
    obj.trainer = SimpleNamespace(state={"model": copies})
    monkeypatch.setattr(BaseInferenceEngine, "initialize", lambda self: None)
    obj.initialize()
    assert all(
        m.dense_attention_backend == "sdpa"
        for m in copies.modules()
        if hasattr(m, "dense_attention_backend")
    )
    with pytest.raises(ValueError, match="dense_attention_backend"):
        RFD3InferenceEngine(**RFD3InferenceConfig(dense_attention_backend="invalid"))


@pytest.mark.parametrize("d", [1, 3])
@pytest.mark.parametrize("bf16", [False, True])
def test_gather_before_projection_matches_dense_projection(d, bf16):
    from unittest.mock import patch

    # Independent reference projects every dense pair before selecting neighbors.
    with patch.dict(os.environ, {"RFD3_LOW_MEMORY_MODE": "0"}):
        m = tiny_model(bf16=bf16)
    with torch.no_grad():
        f, x = prepared([example(d=d), example((1, 2), d=d, seed=9)], m, b=3)
        init = initialize(m.token_initializer, f)
        pairs = init["P_LL"]
        assert isinstance(pairs, torch.Tensor)
        from rfd3.model.layers.batched import select_neighbors

        dm = m.diffusion_module
        indices, neighbor_valid = select_neighbors(
            x, f["atom_valid"], f["atom_sequence_mask"], dm.n_attn_keys
        )
        b, _, n, k = indices.shape
        gathered = torch.gather(
            pairs.expand(b, d, n, n, pairs.shape[-1]),
            3,
            indices[..., None].expand(b, d, n, k, pairs.shape[-1]),
        )
        for block in [*dm.encoder.blocks, *dm.decoder.atom_transformer]:
            layer = block.attention_pair_bias
            args = (
                layer,
                init["Q_L_init"].expand(b, d, -1, -1),
                init["C_L"],
            )
            old = local_attention(
                *args, pairs, indices, neighbor_valid, f["atom_valid"]
            )
            new = local_attention(
                *args, gathered, indices, neighbor_valid, f["atom_valid"], gathered=True
            )
            torch.testing.assert_close(new, old, atol=2e-6, rtol=2e-6)
