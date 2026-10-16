import pytest
import torch
from rfd3.inference.batching import collate_examples, unpad_output
from rfd3.model.layers.batched import (
    gather_nodes,
    group_atoms,
    masked_softmax,
    pool_atoms,
    pool_pairs,
    ungroup_atoms,
)
from rfd3_batch_fixtures import example


def test_collation_and_unpad_preserve_each_example():
    examples = [example(), example((1, 2), seed=8)]
    batch = collate_examples(
        examples, atom_capacity=9, token_capacity=5, batch_capacity=3
    )
    f = batch["f"]
    assert batch["coords"].shape == (3, 3, 9, 3)
    assert f["atom_valid"].sum(1).tolist() == [6, 3, 0]
    assert f["token_valid"].sum(1).tolist() == [3, 2, 0]
    for b, ex in enumerate(examples):
        out = unpad_output(
            {"X_L": batch["coords"], "sequence_indices_I": torch.zeros(3, 3, 5)}, b, ex
        )
        torch.testing.assert_close(out["X_L"], ex["coord_atom_lvl_to_be_noised"])
        assert out["sequence_indices_I"].shape == (3, len(ex["feats"]["restype"]))
        assert ex["feats"]["ref_atom_name_chars"].ndim == 3  # no mutation


@pytest.mark.parametrize(
    "failure",
    ["unknown", "bad_map", "bad_rep", "empty", "small_capacity", "different_d"],
)
def test_collation_rejects_ambiguous_or_invalid_inputs(failure):
    ex = example()
    kwargs, examples = {}, [ex]
    if failure == "unknown":
        ex["feats"]["mystery"] = torch.ones(6)
    if failure == "bad_map":
        ex["feats"]["atom_to_token_map"][0] = 10
    if failure == "bad_rep":
        ex["feats"]["is_ca"][:] = False
    if failure == "empty":
        examples = []
    if failure == "small_capacity":
        kwargs["atom_capacity"] = 1
    if failure == "different_d":
        examples.append(example(d=2))
    with pytest.raises(ValueError):
        collate_examples(examples, **kwargs)


def test_group_pool_and_gather_match_independent_loops():
    examples = [example(), example((1, 2), seed=8)]
    batch = collate_examples(
        examples, atom_capacity=8, token_capacity=4, batch_capacity=3
    )
    f, x = batch["f"], batch["coords"]
    # Padding storage must not leak even if a caller puts NaNs there.
    x = x.masked_fill(~f["atom_valid"][:, None, :, None], float("nan"))
    grouped = group_atoms(x, f)
    restored = ungroup_atoms(grouped, f)
    means = pool_atoms(x, f)
    for b, ex in enumerate(examples):
        l = len(ex["feats"]["atom_to_token_map"])
        torch.testing.assert_close(restored[b, :, :l], x[b, :, :l])
        for i in range(len(ex["feats"]["restype"])):
            select = ex["feats"]["atom_to_token_map"] == i
            torch.testing.assert_close(means[b, :, i], x[b, :, :l][:, select].mean(-2))
    assert torch.isfinite(restored).all() and torch.isfinite(means).all()
    gathered = gather_nodes(x.nan_to_num(), f["representative_atom"])
    for b, ex in enumerate(examples):
        torch.testing.assert_close(
            gathered[b, :, : len(ex["feats"]["restype"])],
            ex["coord_atom_lvl_to_be_noised"][:, ex["feats"]["is_ca"]],
        )


def test_pair_pool_uses_per_example_valid_counts():
    examples = [example(), example((1, 2), seed=8)]
    f = collate_examples(examples, atom_capacity=8, token_capacity=4)["f"]
    pairs = torch.randn(2, 1, 8, 8, 3)
    out = pool_pairs(pairs, f)
    for b, ex in enumerate(examples):
        tok = ex["feats"]["atom_to_token_map"]
        for i in range(len(ex["feats"]["restype"])):
            for j in range(len(ex["feats"]["restype"])):
                expected = pairs[b, 0, : len(tok), : len(tok)][tok == i][
                    :, tok == j
                ].mean((0, 1))
                torch.testing.assert_close(out[b, 0, i, j], expected)
    assert (out[:, :, 3] == 0).all()


def test_masked_softmax_has_zero_empty_rows_and_no_padding_weight():
    logits = torch.tensor([[2.0, 50.0, 4.0], [1.0, 2.0, 3.0]])
    mask = torch.tensor([[True, False, True], [False, False, False]])
    p = masked_softmax(logits, mask)
    torch.testing.assert_close(p[0, [0, 2]], torch.softmax(logits[0, [0, 2]], -1))
    assert p[0, 1] == 0 and (p[1] == 0).all() and torch.isfinite(p).all()
