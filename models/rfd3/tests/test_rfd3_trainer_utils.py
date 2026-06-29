"""Unit tests for the pure helpers in ``rfd3.trainer.trainer_utils``.

- ``_reorder_dict`` returns an ``OrderedDict`` with a fixed key ordering: the
  ``first_keys`` (``task``, ``diffused_index_map``) first, then every other key in
  insertion order, then the ``last_keys`` (``metrics``, ``specification``,
  ``inference_sampler``) last — each group keeping its canonical order, absent keys
  skipped.
- ``_remap_outputs`` scatters each batch row's coordinates by a per-row index map:
  ``xyz[i, mapping[i][j]] = xyz[i][j]`` (so element ``j`` lands at position
  ``mapping[i][j]``), in place, returning the same tensor.

Named ``test_rfd3_trainer_utils`` to avoid a pytest basename clash under the suite's
prepend import mode.
"""

from collections import OrderedDict

import torch
from rfd3.trainer.trainer_utils import _remap_outputs, _reorder_dict

# --- _reorder_dict ----------------------------------------------------------


def test_reorder_dict_orders_first_middle_last():
    d = {
        "specification": 1,
        "foo": 2,
        "task": 3,
        "metrics": 4,
        "diffused_index_map": 5,
        "bar": 6,
    }
    assert list(_reorder_dict(d)) == [
        "task",
        "diffused_index_map",
        "foo",
        "bar",
        "metrics",
        "specification",
    ]


def test_reorder_dict_last_keys_in_canonical_order():
    d = {"inference_sampler": 1, "specification": 2, "metrics": 3, "x": 4}
    assert list(_reorder_dict(d)) == [
        "x",
        "metrics",
        "specification",
        "inference_sampler",
    ]


def test_reorder_dict_skips_absent_keys_and_preserves_middle_order():
    d = {"a": 1, "b": 2, "c": 3}
    assert list(_reorder_dict(d)) == ["a", "b", "c"]


def test_reorder_dict_returns_ordereddict_with_values_preserved():
    d = {"task": 10, "foo": 20, "metrics": 30}
    result = _reorder_dict(d)
    assert isinstance(result, OrderedDict)
    assert result == d


# --- _remap_outputs ---------------------------------------------------------


def test_remap_outputs_identity_is_noop():
    xyz = torch.tensor([[[1.0, 1, 1], [2, 2, 2], [3, 3, 3]]])
    mapping = torch.tensor([[0, 1, 2]])
    result = _remap_outputs(xyz.clone(), mapping)
    assert torch.equal(result, xyz)


def test_remap_outputs_applies_scatter_permutation():
    # mapping[0] = [2, 0, 1] places row j at position mapping[0][j]:
    # new[2]=old[0], new[0]=old[1], new[1]=old[2].
    xyz = torch.tensor([[[1.0, 1, 1], [2, 2, 2], [3, 3, 3]]])
    mapping = torch.tensor([[2, 0, 1]])
    result = _remap_outputs(xyz, mapping)
    expected = torch.tensor([[[2.0, 2, 2], [3, 3, 3], [1, 1, 1]]])
    assert torch.equal(result, expected)


def test_remap_outputs_is_independent_per_batch_entry():
    xyz = torch.tensor(
        [
            [[1.0, 1, 1], [2, 2, 2]],
            [[5.0, 5, 5], [6, 6, 6]],
        ]
    )
    mapping = torch.tensor([[1, 0], [0, 1]])  # swap row 0; identity row 1
    result = _remap_outputs(xyz, mapping)
    expected = torch.tensor(
        [
            [[2.0, 2, 2], [1, 1, 1]],
            [[5.0, 5, 5], [6, 6, 6]],
        ]
    )
    assert torch.equal(result, expected)


def test_remap_outputs_mutates_and_returns_same_tensor():
    xyz = torch.tensor([[[1.0, 1, 1], [2, 2, 2]]])
    result = _remap_outputs(xyz, torch.tensor([[1, 0]]))
    assert result is xyz
