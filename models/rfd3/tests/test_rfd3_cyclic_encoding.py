"""Portable checks for cyclic offsets, encoding channels, and guidance."""

import pytest
import torch
from rfd3.model.cfg_utils import strip_f
from rfd3.model.layers.blocks import (
    RelativePositionEncodingWithIndexRemoval,
    _cyclic_residue_offsets,
)


@pytest.mark.parametrize("length", [1, 2, 5, 10, 12, 70])
def test_signed_offsets(length):
    indices = torch.arange(length)
    offsets = indices[:, None] - indices[None, :]
    actual = _cyclic_residue_offsets(
        offsets, torch.zeros_like(indices), indices, torch.tensor([0])
    )
    # Stable minimum independently expresses the shortest-path/tie contract.
    expected = torch.tensor(
        [
            [min((i - j, i - j + length, i - j - length), key=abs) for j in indices]
            for i in indices
        ]
    )
    assert torch.equal(actual, expected)
    assert torch.equal(actual, -actual.T)
    assert not actual.diagonal().any()
    if length > 2:
        assert actual[0, -1] == 1
        assert actual[-1, 0] == -1
    if length % 2 == 0:
        assert actual[length // 2, 0] == length // 2
        assert actual[0, length // 2] == -length // 2


def test_multiple_tokens_per_residue_do_not_inflate_cyclic_length():
    residue_index = torch.tensor([0, 1, 1, 2, 3])
    offsets = residue_index[:, None] - residue_index[None, :]
    actual = _cyclic_residue_offsets(
        offsets,
        torch.zeros_like(residue_index),
        residue_index,
        torch.tensor([0]),
    )
    expected = torch.tensor(
        [
            [min((i - j, i - j + 4, i - j - 4), key=abs) for j in residue_index]
            for i in residue_index
        ]
    )
    assert torch.equal(actual, expected)


def _features(length, context=True):
    # Nonzero cyclic ID; context includes gapped numbering and atomized residues.
    prefix, suffix = ([0, 0], [8, 8, 9]) if context else ([], [])
    chains = prefix + [3] * length + suffix
    residues = ([2, 40] if context else []) + list(range(21, 21 + length))
    residues += [7, 7, 1] if context else []
    n = len(chains)
    return {
        "asym_id": torch.tensor(chains),
        "entity_id": torch.tensor([0 if c in (0, 8) else c for c in chains]),
        "sym_id": torch.tensor([1 if c == 8 else 0 for c in chains]),
        "residue_index": torch.tensor(residues),
        "token_index": torch.arange(n),
        "unindexing_pair_mask": torch.zeros(n, n, dtype=torch.bool),
    }


def _expected_encoding(f, cyclic):
    """Construct all bins independently using scalar pair rules (r_max=4, s_max=2)."""
    chains, residues, tokens, entities, copies = (
        f[key].tolist()
        for key in ("asym_id", "residue_index", "token_index", "entity_id", "sym_id")
    )
    length = chains.count(3)
    n = len(chains)
    encoding = torch.zeros(n, n, 29)
    for i in range(n):
        for j in range(n):
            same_chain = chains[i] == chains[j]
            same_entity = entities[i] == entities[j]
            offset = residues[i] - residues[j]
            if cyclic and chains[i] == chains[j] == 3:
                offset = min((offset, offset + length, offset - length), key=abs)
            residue_bin = min(8, max(0, offset + 4)) if same_chain else 9
            token_bin = (
                min(8, max(0, tokens[i] - tokens[j] + 4))
                if same_chain and residues[i] == residues[j]
                else 9
            )
            if f["unindexing_pair_mask"][i, j]:
                residue_bin = token_bin = 10
            chain_bin = min(4, max(0, copies[i] - copies[j] + 2)) if same_entity else 5
            encoding[i, j, residue_bin] = 1
            encoding[i, j, 11 + token_bin] = 1
            encoding[i, j, 22] = same_entity
            encoding[i, j, 23 + chain_bin] = 1
    return encoding


@pytest.mark.parametrize("length", [1, 2, 5, 10, 12, 70])
def test_rpe_bins_isolation_defaults_and_no_mutation(length):
    f = _features(length)
    # Synthetic selected-pair override verifies unknown bins take precedence.
    f["unindexing_pair_mask"][2, length + 1] = True
    original = {key: value.clone() for key, value in f.items()}
    rpe = RelativePositionEncodingWithIndexRemoval(r_max=4, s_max=2, c_z=7)
    captured = []
    hook = rpe.linear.register_forward_pre_hook(
        lambda _, args: captured.append(args[0])
    )
    linear = rpe(f)
    empty = rpe({**f, "cyclic_asym_ids": torch.empty(0, dtype=torch.int64)})
    cyclic = rpe({**f, "cyclic_asym_ids": torch.tensor([3])})
    hook.remove()
    assert torch.equal(linear, empty)
    assert torch.equal(captured[0], _expected_encoding(f, cyclic=False))
    assert torch.equal(captured[0], captured[1])
    assert torch.equal(captured[2], _expected_encoding(f, cyclic=True))
    selected = f["asym_id"] == 3
    outside = ~(selected[:, None] & selected[None, :])
    assert torch.equal(linear[outside], cyclic[outside])
    assert torch.equal(captured[0][..., 11:], captured[2][..., 11:])
    assert all(torch.equal(f[key], value) for key, value in original.items())


def test_checkpoint_schema_and_strict_loading():
    rpe = RelativePositionEncodingWithIndexRemoval(r_max=4, s_max=2, c_z=7)
    assert {key: tuple(value.shape) for key, value in rpe.state_dict().items()} == {
        "linear.weight": (7, 29)
    }
    rpe.load_state_dict({"linear.weight": torch.randn(7, 29)}, strict=True)


@pytest.mark.parametrize("length", [1, 12])
@pytest.mark.parametrize("unindexed_context", [False, True])
def test_guidance_preserves_cyclic_peptide(length, unindexed_context):
    f = _features(length, context=False)
    if unindexed_context:
        for key in ("asym_id", "entity_id", "sym_id", "residue_index", "token_index"):
            f[key] = torch.cat([f[key], torch.tensor([8, 9])])
        n = length + 2
        f["unindexing_pair_mask"] = torch.zeros(n, n, dtype=torch.bool)
    n = len(f["asym_id"])
    f["is_motif_token_unindexed"] = torch.arange(n) >= length
    f["is_motif_atom_unindexed"] = f["is_motif_token_unindexed"].repeat_interleave(4)
    f["cyclic_asym_ids"] = torch.tensor([3])
    f["is_hotspot"] = torch.ones(n)
    stripped = strip_f(f, cfg_features=["is_hotspot"])
    assert torch.equal(stripped["cyclic_asym_ids"], f["cyclic_asym_ids"])
    assert len(stripped["asym_id"]) == length
    assert not stripped["is_hotspot"].any()
    rpe = RelativePositionEncodingWithIndexRemoval(r_max=4, s_max=2, c_z=7)
    captured = []
    hook = rpe.linear.register_forward_pre_hook(
        lambda _, args: captured.append(args[0])
    )
    full = rpe(f)[:length, :length]
    cropped = rpe(stripped)
    hook.remove()
    assert torch.equal(captured[0][:length, :length], captured[1])
    assert torch.allclose(full, cropped, rtol=1e-6, atol=1e-7)
