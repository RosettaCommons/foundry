import copy

import pytest
import torch
from rfd3.inference.batching import collate_examples, prepare_attention
from rfd3.model.layers.batched import (
    downcast,
    initialize,
    local_attention,
    select_neighbors,
    upcast,
)
from rfd3.model.layers.block_utils import create_attention_indices
from rfd3_batch_fixtures import example, tiny_model


def inputs():
    examples = [example((2, 1, 3)), example((1, 2), seed=8)]
    batch = collate_examples(examples, atom_capacity=9, token_capacity=5)
    f = prepare_attention(
        batch["f"], atom_keys=4, atom_neighbors=1, token_keys=32, token_neighbors=8
    )
    return examples, f, batch["coords"]


def test_neighbors_match_legacy_and_ignore_padding():
    examples, f, x = inputs()
    indices, valid = select_neighbors(x, f["atom_valid"], f["atom_sequence_mask"], 4)
    for b, ex in enumerate(examples):
        n = len(ex["feats"]["atom_to_token_map"])
        expected = create_attention_indices(ex["feats"], 4, 1, X_L=x[b, :, :n])
        torch.testing.assert_close(indices[b, :, :n, : min(n, 4)], expected)
        assert not valid[b, :, n:].any()
        assert not valid[b, :, :, min(n, 4) :].any()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_precomputed_neighbors_follow_moving_coordinates_and_empty_rows(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    examples = [example(), example((1, 2), seed=8)]
    batch = collate_examples(
        examples, atom_capacity=9, token_capacity=5, batch_capacity=3
    )
    f = prepare_attention(
        batch["f"], atom_keys=4, atom_neighbors=1, token_keys=4, token_neighbors=1
    )
    f = {k: v.to(device) for k, v in f.items()}
    cached_select = (
        torch.compile(select_neighbors, fullgraph=True)
        if device == "cuda" else select_neighbors
    )
    for level in ("atom", "token"):
        valid = f[f"{level}_valid"]
        for seed in (1, 2):
            x = torch.randn(
                valid.shape[0], 2, valid.shape[1], 3,
                generator=torch.Generator().manual_seed(seed),
            ).to(device)
            expected = select_neighbors(x, valid, f[f"{level}_sequence_mask"], 4)
            actual = cached_select(
                x, valid, f[f"{level}_sequence_mask"], 4,
                f[f"{level}_forced_neighbors"],
            )
            for got, ref in zip(actual, expected):
                torch.testing.assert_close(got, ref, rtol=0, atol=0)
            assert not actual[1][-1].any()


def test_neighbors_preserve_chain_and_unindexing_policy_at_longer_lengths():
    examples = [example((1, 2, 3) * 8), example((2, 1) * 11, seed=8)]
    for ex in examples:
        f = ex["feats"]
        n = len(f["restype"])
        f["asym_id"] = torch.arange(n) % 3
        f["unindexing_pair_mask"][1:4, 8:12] = True
    batch = collate_examples(examples, atom_capacity=64, token_capacity=32)
    f = prepare_attention(
        batch["f"], atom_keys=12, atom_neighbors=1, token_keys=6, token_neighbors=2
    )
    idx, mask = select_neighbors(
        batch["coords"], f["atom_valid"], f["atom_sequence_mask"], 12
    )
    for b, ex in enumerate(examples):
        n = ex["coord_atom_lvl_to_be_noised"].shape[-2]
        expected = create_attention_indices(
            ex["feats"], 12, 1, X_L=ex["coord_atom_lvl_to_be_noised"]
        )
        torch.testing.assert_close(idx[b, :, :n], expected)
        assert not mask[b, :, n:].any()


@pytest.mark.parametrize("method", ["mean", "cross_attention"])
def test_downcast_matches_legacy_per_example(method):
    from rfd3.model.layers.blocks import Downcast

    examples, f, _ = inputs()
    torch.manual_seed(32)
    layer = Downcast(
        c_atom=4,
        c_token=8,
        method=method,
        cross_attention_block=dict(c_model=8, n_head=2),
    ).eval()
    q, a = torch.randn(2, 3, 9, 4), torch.randn(2, 3, 5, 8)
    with torch.no_grad():
        actual = downcast(layer, q, a, None, f)
        for b, ex in enumerate(examples):
            tok = ex["feats"]["atom_to_token_map"]
            i = len(ex["feats"]["restype"])
            expected = layer(q[b, :, : len(tok)], a[b, :, :i], tok_idx=tok)
            torch.testing.assert_close(actual[b, :, :i], expected)
            assert (actual[b, :, i:] == 0).all()


@pytest.mark.parametrize("method", ["broadcast", "cross_attention"])
def test_upcast_matches_legacy_per_example(method):
    from rfd3.model.layers.blocks import Upcast

    examples, f, _ = inputs()
    torch.manual_seed(33)
    layer = Upcast(
        c_atom=4,
        c_token=8,
        method=method,
        n_split=2,
        cross_attention_block=dict(c_model=8, n_head=2),
    ).eval()
    q, a = torch.randn(2, 3, 9, 4), torch.randn(2, 3, 5, 8)
    with torch.no_grad():
        actual = upcast(layer, q, a, f)
        for b, ex in enumerate(examples):
            tok = ex["feats"]["atom_to_token_map"]
            i = len(ex["feats"]["restype"])
            expected = layer(q[b, :, : len(tok)], a[b, :, :i], tok_idx=tok)
            torch.testing.assert_close(actual[b, :, : len(tok)], expected)
            assert (actual[b, :, len(tok) :] == 0).all()


@pytest.mark.parametrize("full", [False, True])
def test_attention_matches_legacy_and_empty_queries(full):
    from rfd3.model.layers.attention import LocalAttentionPairBias

    examples, f, x = inputs()
    layer = LocalAttentionPairBias(c_a=8, c_s=8, c_pair=4, n_head=2).eval()
    q, s, z = (
        torch.randn(2, 3, 9, 8),
        torch.randn(2, 3, 9, 8),
        torch.randn(2, 1, 9, 9, 4),
    )
    indices, valid = select_neighbors(x, f["atom_valid"], f["atom_sequence_mask"], 4)
    with torch.no_grad():
        actual = local_attention(
            layer, q, s, z, indices, valid, f["atom_valid"], full=full
        )
        for b, ex in enumerate(examples):
            n = len(ex["feats"]["atom_to_token_map"])
            # Legacy full path requires explicitly diffusion-batched pair bias.
            pairs = z[b, :, :n, :n].expand(3, -1, -1, -1) if full else z[b, 0, :n, :n]
            expected = layer(
                q[b, :, :n],
                s[b, :, :n],
                pairs,
                indices=indices[b, :, :n, : min(n, 4)],
                full=full,
            )
            torch.testing.assert_close(actual[b, :, :n], expected, atol=2e-6, rtol=2e-5)
            assert (actual[b, :, n:] == 0).all()


def test_initializer_matches_legacy_without_changing_checkpoint_keys():
    examples, f, _ = inputs()
    model = tiny_model()
    state = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        actual = initialize(model.token_initializer, f)
        for b, ex in enumerate(examples):
            expected = model.token_initializer(copy.deepcopy(ex["feats"]))
            l, i = len(ex["feats"]["atom_to_token_map"]), len(ex["feats"]["restype"])
            for key, v in expected.items():
                n = l if key in ("Q_L_init", "C_L", "P_LL") else i
                got = actual[key][b, 0, :n]
                if key in ("P_LL", "Z_II"):
                    got = got[:, :n]
                torch.testing.assert_close(got, v, atol=2e-5, rtol=2e-5, msg=key)
    assert model.state_dict().keys() == state.keys()
    model.load_state_dict(state, strict=True)


def test_legacy_pair_only_block_still_supports_zero_single_channels():
    from rfd3.model.layers.pairformer_layers import PairformerBlock

    block = PairformerBlock(c_s=0, c_z=8, attention_pair_bias={"n_head": 2}).eval()
    with torch.no_grad():
        single, pairs = block(None, torch.randn(3, 3, 8))
    assert single is None and torch.isfinite(pairs).all()
