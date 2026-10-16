import copy

import pytest
import torch
from rfd3.inference.batching import collate_examples, prepare_attention
from rfd3.model.layers.batched import initialize
from rfd3_batch_fixtures import example, tiny_model


def prepared(examples, model, l=10, i=6, b=None):
    batch = collate_examples(
        examples, atom_capacity=l, token_capacity=i, batch_capacity=b
    )
    dm = model.diffusion_module
    f = prepare_attention(
        batch["f"],
        atom_keys=dm.n_attn_keys,
        atom_neighbors=dm.n_attn_seq_neighbours,
        token_keys=dm.diffusion_transformer.n_keys,
        token_neighbors=dm.diffusion_transformer.n_local_tokens,
    )
    return f, batch["coords"]


@pytest.mark.parametrize("d,recycles", [(1, 1), (3, 1), (3, 2)])
@pytest.mark.parametrize("bf16", [False, True])
def test_denoiser_matches_independent_legacy_examples(d, recycles, bf16):
    model = tiny_model(bf16=bf16)
    examples = [example(d=d), example((1, 2, 1, 1), d=d, seed=8)]
    examples[1]["feats"]["is_motif_atom_with_fixed_coord"][0] = True
    examples[1]["feats"]["is_motif_token_with_fully_fixed_coord"][0] = True
    with torch.no_grad():
        f, x = prepared(examples, model)
        init = initialize(model.token_initializer, f)
        t = torch.linspace(0.5, 2, d).expand(2, d)
        out = model.diffusion_module.forward_batched(
            x, t, f, **init, n_recycle=recycles
        )
        for b, ex in enumerate(examples):
            lf = copy.deepcopy(ex["feats"])
            expected = model.diffusion_module(
                ex["coord_atom_lvl_to_be_noised"],
                t[b],
                lf,
                **model.token_initializer(lf),
                n_recycle=recycles,
            )
            for key in ("X_L", "sequence_logits_I"):
                n = len(ex["feats"]["atom_to_token_map" if key == "X_L" else "restype"])
                tolerance = 0.025 if bf16 else 2e-5
                torch.testing.assert_close(
                    out[key][b, :, :n],
                    expected[key],
                    atol=tolerance,
                    rtol=tolerance,
                    msg=key,
                )
                assert (out[key][b, :, n:] == 0).all()
        assert torch.isfinite(out["X_L"]).all()


@pytest.mark.parametrize("backend_name", ["vanilla", "sdpa"])
def test_padding_capacity_peers_and_dummy_rows_cannot_change_predictions(backend_name):
    model = tiny_model()
    for m in model.modules():
        if hasattr(m, "dense_attention_backend"):
            m.dense_attention_backend = backend_name
    ex = example(d=3)
    with torch.no_grad():

        def run(examples, l, i, b):
            f, x = prepared(examples, model, l=l, i=i, b=b)
            init = initialize(model.token_initializer, f)
            x = x.masked_fill(~f["atom_valid"][:, None, :, None], float("nan"))
            return model.diffusion_module.forward_batched(
                x, torch.ones(b, 3), f, **init
            )

        alone = run([ex], 6, 3, 1)
        together = run([example((1, 2), seed=9), ex], 11, 7, 3)
        for key, n in [("X_L", 6), ("sequence_logits_I", 3)]:
            torch.testing.assert_close(
                together[key][1, :, :n], alone[key][0], atol=2e-5, rtol=2e-5
            )
            assert (together[key][2] == 0).all()


def test_diffusion_and_example_axes_are_independent():
    model = tiny_model()
    with torch.no_grad():
        f, x = prepared([example(), example((1, 2), seed=9)], model)
        init = initialize(model.token_initializer, f)
        t = torch.ones(2, 3)
        first = model.diffusion_module.forward_batched(x, t, f, **init)
        changed = x.clone()
        changed[0, 1] += torch.randn_like(changed[0, 1]) * 10
        changed[1] += 100
        second = model.diffusion_module.forward_batched(changed, t, f, **init)
        torch.testing.assert_close(first["X_L"][0, 0], second["X_L"][0, 0])
        torch.testing.assert_close(first["X_L"][0, 2], second["X_L"][0, 2])
        assert not torch.allclose(first["X_L"][0, 1], second["X_L"][0, 1])
