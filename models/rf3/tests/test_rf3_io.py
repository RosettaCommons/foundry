"""Unit tests for the pure helpers in rf3.utils.io.

- ``get_sharded_output_path`` builds an output directory for an example, optionally
  fanning it out into hash-derived subdirectories. With no (or a falsy) sharding
  pattern it returns ``base_dir / example_id``; with a pattern like ``/0:2/2:4/`` it
  inserts directories sliced from ``hash_sequence(example_id)`` before the id.
- ``build_stack_from_atom_array_and_batched_coords`` broadcasts a single AtomArray
  across a batch of coordinate sets into an AtomArrayStack (the diffusion batch
  becomes the models), accepting numpy or torch coords. When the array spans more
  than one ``transformation_id`` it disambiguates by suffixing each ``chain_id``
  with its ``transformation_id`` (so bond annotations stay unique).

Named ``test_rf3_io`` to avoid a pytest basename clash with any future top-level
``test_io`` under the suite's prepend import mode.
"""

import numpy as np
import torch
from atomworks.ml.utils.misc import hash_sequence
from biotite.structure import AtomArray
from rf3.utils.io import (
    build_stack_from_atom_array_and_batched_coords,
    get_sharded_output_path,
)

# --- get_sharded_output_path ------------------------------------------------


def test_no_sharding_pattern_returns_base_dir_over_id(tmp_path):
    assert get_sharded_output_path("entry_1", tmp_path) == tmp_path / "entry_1"


def test_empty_sharding_pattern_is_treated_as_no_sharding(tmp_path):
    assert get_sharded_output_path("entry_1", tmp_path, "") == tmp_path / "entry_1"


def test_sharding_pattern_inserts_hash_sliced_dirs(tmp_path):
    h = hash_sequence("entry_1")
    result = get_sharded_output_path("entry_1", tmp_path, "/0:2/2:4/")
    assert result == tmp_path / h[0:2] / h[2:4] / "entry_1"


# --- build_stack_from_atom_array_and_batched_coords -------------------------


def _atom_array(n: int, chain_ids: list[str]) -> AtomArray:
    arr = AtomArray(n)
    arr.coord = np.zeros((n, 3), dtype=np.float32)
    arr.set_annotation("chain_id", np.array(chain_ids))
    return arr


def test_builds_one_model_per_batch_entry_and_assigns_coords():
    arr = _atom_array(3, ["A", "A", "B"])
    coords = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)
    stack = build_stack_from_atom_array_and_batched_coords(coords, arr)
    assert stack.shape == (2, 3)
    assert np.array_equal(stack.coord, coords)


def test_accepts_torch_coords():
    arr = _atom_array(2, ["A", "A"])
    coords = torch.arange(2 * 2 * 3, dtype=torch.float32).reshape(2, 2, 3)
    stack = build_stack_from_atom_array_and_batched_coords(coords, arr)
    assert stack.shape == (2, 2)
    assert np.array_equal(stack.coord, coords.numpy())


def test_chain_id_unchanged_without_transformation_id():
    arr = _atom_array(3, ["A", "A", "B"])
    stack = build_stack_from_atom_array_and_batched_coords(
        np.zeros((2, 3, 3), dtype=np.float32), arr
    )
    assert stack.chain_id.tolist() == ["A", "A", "B"]


def test_chain_id_suffixed_when_multiple_transformation_ids():
    arr = _atom_array(2, ["A", "A"])
    arr.set_annotation("transformation_id", np.array(["1", "2"]))
    stack = build_stack_from_atom_array_and_batched_coords(
        np.zeros((2, 2, 3), dtype=np.float32), arr
    )
    assert stack.chain_id.tolist() == ["A1", "A2"]


def test_chain_id_unchanged_with_single_transformation_id():
    arr = _atom_array(2, ["A", "A"])
    arr.set_annotation("transformation_id", np.array(["1", "1"]))
    stack = build_stack_from_atom_array_and_batched_coords(
        np.zeros((2, 2, 3), dtype=np.float32), arr
    )
    assert stack.chain_id.tolist() == ["A", "A"]
