import copy
import os
from unittest.mock import patch

import torch
from rfd3.model.layers.batched import initialize, select_neighbors, sparse_pairs
from rfd3_batch_fixtures import example, tiny_model
from test_rfd3_batch_model import prepared


def test_low_memory_pairs_and_denoiser_match_their_own_legacy_path():
    with patch.dict(os.environ, {"RFD3_LOW_MEMORY_MODE": "1"}):
        model = tiny_model()
        examples = [example(), example((1, 2), seed=8)]
        examples[0]["feats"]["is_motif_atom_with_fixed_coord"][:3] = True
        examples[0]["feats"]["is_motif_atom_with_fixed_seq"][:3] = True
        with torch.no_grad():
            f, x = prepared(examples, model)
            init = initialize(model.token_initializer, f)
            idx, valid = select_neighbors(
                x, f["atom_valid"], f["atom_sequence_mask"], 4
            )
            actual_pairs = sparse_pairs(init["P_LL"], f, idx, valid)
            out = model.diffusion_module.forward_batched(x, torch.ones(2, 3), f, **init)
            for b, ex in enumerate(examples):
                lf = copy.deepcopy(ex["feats"])
                original = model.token_initializer(lf)
                embedder = original.pop("chunked_pairwise_embedder")
                n = len(lf["atom_to_token_map"])
                expected_pairs = embedder.forward_chunked(
                    f=lf,
                    indices=idx[b, :, :n, : min(n, 4)],
                    C_L=original["C_L"],
                    Z_init_II=original["Z_II"],
                    tok_idx=lf["atom_to_token_map"],
                )
                torch.testing.assert_close(
                    actual_pairs[b, :, :n, : min(n, 4)],
                    expected_pairs,
                    atol=2e-5,
                    rtol=2e-5,
                )
                expected = model.diffusion_module(
                    ex["coord_atom_lvl_to_be_noised"],
                    torch.ones(3),
                    lf,
                    P_LL=None,
                    chunked_pairwise_embedder=embedder,
                    initializer_outputs=original,
                    **original,
                )
                torch.testing.assert_close(
                    out["X_L"][b, :, :n], expected["X_L"], atol=2e-5, rtol=2e-5
                )


def test_low_memory_reference_initialization_does_not_overwrite_conditional_cache():
    with patch.dict(os.environ, {"RFD3_LOW_MEMORY_MODE": "1"}):
        model = tiny_model()
    with torch.no_grad():
        f, x = prepared([example(), example((1, 2), seed=8)], model)
        conditional = initialize(model.token_initializer, f)
        idx, valid = select_neighbors(x, f["atom_valid"], f["atom_sequence_mask"], 4)
        before = sparse_pairs(conditional["P_LL"], f, idx, valid)
        ref = {k: v.clone() for k, v in f.items()}
        ref["ref_atomwise_rasa"].zero_()
        initialize(model.token_initializer, ref)
        after = sparse_pairs(conditional["P_LL"], f, idx, valid)
        torch.testing.assert_close(after, before)
