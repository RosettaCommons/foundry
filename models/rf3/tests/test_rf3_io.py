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
import pytest
import torch
from atomworks.ml.utils.misc import hash_sequence
from biotite.structure import Atom, AtomArray, BondList, array
from biotite.structure.io.pdbx import CIFFile, get_structure
from rf3.utils.io import (
    build_stack_from_atom_array_and_batched_coords,
    dump_trajectories,
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


# Real CIF round trips: sample and time axes must never be swapped.


@pytest.mark.parametrize("final_first", [False, True])
@pytest.mark.parametrize("as_numpy", [False, True])
@pytest.mark.parametrize("align", [False, True])
def test_trajectory_cif_order_samples_and_input_immutability(
    tmp_path, final_first, as_numpy, align
):
    atoms = array(
        [
            Atom(
                [0, 0, 0],
                chain_id="A",
                res_id=1,
                res_name="GLY",
                atom_name=name,
                element=name[0],
            )
            for name in ("N", "CA", "C", "O")
        ]
    )
    atoms.bonds = BondList(len(atoms))
    rng = np.random.default_rng(42)
    frames = [
        torch.tensor(rng.normal(size=(2, 4, 3)), dtype=torch.float32) for _ in range(3)
    ]
    if as_numpy:
        frames = [frame.numpy() for frame in frames]
    before = [np.asarray(frame).copy() for frame in frames]
    dump_trajectories(
        frames,
        atoms,
        tmp_path / "traj",
        align_structures=align,
        final_frame_first=final_first,
        file_type="cif",
    )
    for sample in range(2):
        stack = get_structure(CIFFile.read(tmp_path / f"traj_model_{sample}.cif"))
        assert stack.shape == (3, 4)
        expected = before[::-1] if final_first else before
        if not align:
            np.testing.assert_allclose(
                stack.coord, np.stack([f[sample] for f in expected]), atol=0.001
            )
        np.testing.assert_allclose(
            stack.coord[0 if final_first else -1], before[-1][sample], atol=0.001
        )
    for original, frame in zip(before, frames):
        np.testing.assert_array_equal(original, frame)


@pytest.mark.parametrize(
    "frames", [[], [torch.zeros(2, 3)], [torch.zeros(2, 4, 3), torch.zeros(1, 4, 3)]]
)
def test_invalid_trajectory_fails_before_writing(tmp_path, frames):
    with pytest.raises(ValueError):
        dump_trajectories(frames, AtomArray(4), tmp_path / "bad")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("separate", [False, True])
def test_dump_structure_stack_uses_atomworks3_model_list(tmp_path, separate):
    from rf3.utils.io import dump_structures

    atoms = array(
        [
            Atom(
                [0, 0, 0],
                chain_id="A",
                res_id=1,
                res_name="GLY",
                atom_name="CA",
                element="C",
            )
        ]
    )
    atoms.bonds = BondList(1)
    coords = np.arange(6, dtype=np.float32).reshape(2, 1, 3)
    structures = build_stack_from_atom_array_and_batched_coords(coords, atoms)
    dump_structures(
        structures, tmp_path / "model", one_model_per_file=separate, file_type="cif"
    )
    if separate:
        for i in range(2):
            restored = get_structure(
                CIFFile.read(tmp_path / f"model_model_{i}.cif"), model=1
            )
            np.testing.assert_allclose(restored.coord, coords[i])
    else:
        restored = get_structure(CIFFile.read(tmp_path / "model.cif"))
        np.testing.assert_allclose(restored.coord, coords)
