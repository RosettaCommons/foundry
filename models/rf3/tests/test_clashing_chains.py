"""Unit tests for rf3.metrics.clashing_chains.CountClashingChains.

``compute`` flags, per diffusion-batch model, whether any pair of *polymer*
``pn_unit`` chains clashes: a pair clashes when the count of inter-chain atom
distances below 1.1 A exceeds 100, or exceeds half the larger chain's atom
count. Non-polymer chains (judged by each chain's first atom) are skipped. The
result is a ``{"has_clash_<i>": 0|1}`` dict, one key per model.
"""

import numpy as np
import torch
from biotite.structure import AtomArrayStack
from rf3.metrics.clashing_chains import CountClashingChains


def _make_stack(
    coords: list[list[float]],
    pn_unit_id: list[str],
    is_polymer: list[bool],
) -> AtomArrayStack:
    """Single-model stack from per-atom xyz / pn_unit / is_polymer lists."""
    n = len(coords)
    stack = AtomArrayStack(depth=1, length=n)
    stack.coord = np.array([coords], dtype=np.float32)  # [1, n, 3]
    stack.set_annotation("pn_unit_id", np.array(pn_unit_id))
    stack.set_annotation("is_polymer", np.array(is_polymer, dtype=bool))
    return stack


# X_L is only read for its batch dimension (`D = X_L.shape[0]`); a [1, *, *] tensor
# matches the single-model stacks below.
X_L = torch.zeros(1, 1, 3)


def test_far_apart_polymer_chains_do_not_clash():
    stack = _make_stack(
        coords=[[0, 0, 0], [1, 0, 0], [100, 0, 0], [101, 0, 0]],
        pn_unit_id=["A", "A", "B", "B"],
        is_polymer=[True, True, True, True],
    )
    out = CountClashingChains().compute(X_L=X_L, predicted_atom_array_stack=stack)

    assert out == {"has_clash_0": 0}


def test_overlapping_polymer_chains_clash():
    # Chain B sits on top of chain A: 3 of 4 inter-chain distances are < 1.1 A,
    # which is > 0.5 * max(chain sizes) -> clash.
    stack = _make_stack(
        coords=[[0, 0, 0], [1, 0, 0], [0.5, 0, 0], [1.5, 0, 0]],
        pn_unit_id=["A", "A", "B", "B"],
        is_polymer=[True, True, True, True],
    )
    out = CountClashingChains().compute(X_L=X_L, predicted_atom_array_stack=stack)

    assert out == {"has_clash_0": 1}


def test_non_polymer_chain_pair_is_skipped():
    # Same overlapping geometry, but chain B is non-polymer -> the pair is skipped.
    stack = _make_stack(
        coords=[[0, 0, 0], [1, 0, 0], [0.5, 0, 0], [1.5, 0, 0]],
        pn_unit_id=["A", "A", "B", "B"],
        is_polymer=[True, True, False, False],
    )
    out = CountClashingChains().compute(X_L=X_L, predicted_atom_array_stack=stack)

    assert out == {"has_clash_0": 0}


def test_single_chain_has_no_pairs():
    # One pn_unit -> no chain pairs to compare -> no clash.
    stack = _make_stack(
        coords=[[0, 0, 0], [0.5, 0, 0], [1.0, 0, 0]],
        pn_unit_id=["A", "A", "A"],
        is_polymer=[True, True, True],
    )
    out = CountClashingChains().compute(X_L=X_L, predicted_atom_array_stack=stack)

    assert out == {"has_clash_0": 0}
