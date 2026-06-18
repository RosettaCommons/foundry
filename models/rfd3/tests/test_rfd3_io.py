"""Unit tests for the pure helpers in ``rfd3.utils.io``.

- ``create_example_id_extractor`` returns a closure that strips a known structure-file
  extension from a path's basename and returns the last remaining dot-separated part as
  the example id (e.g. ``"run.dataset.entry_1.cif.gz" -> "entry_1"``); the multi-part
  ``$``-anchored extensions disambiguate so the result is independent of set order.
  ``extract_example_id_from_path`` is the one-shot wrapper.
- ``find_files_with_extension`` returns the matching files directly inside a directory
  (a single level — *not* recursive, despite the docstring), or the path itself when it
  is a file with a supported suffix.
- ``build_stack_from_atom_array_and_batched_coords`` broadcasts a single AtomArray across
  a batch of coordinate sets into an AtomArrayStack (the diffusion batch becomes the
  models), accepting numpy or torch coords. When the array spans more than one
  ``transformation_id`` it suffixes each ``chain_id`` with its ``transformation_id`` so
  bond annotations stay unique.

Named ``test_rfd3_io`` to avoid a pytest basename clash with rf3's ``test_rf3_io`` (and
any future top-level ``test_io``) under the suite's prepend import mode.
"""

import numpy as np
import torch
from biotite.structure import AtomArray
from rfd3.utils.io import (
    CIF_LIKE_EXTENSIONS,
    build_stack_from_atom_array_and_batched_coords,
    create_example_id_extractor,
    extract_example_id_from_path,
    find_files_with_extension,
)

# --- create_example_id_extractor / extract_example_id_from_path -------------


def test_extractor_strips_extension_and_returns_last_dotted_part():
    extractor = create_example_id_extractor({".cif", ".cif.gz"})
    assert extractor("run.dataset.entry_1.cif.gz") == "entry_1"


def test_extractor_uses_default_cif_like_extensions():
    extractor = create_example_id_extractor()
    assert extractor("foo.bar.cif") == "bar"


def test_extractor_simple_basename_without_extra_dots():
    extractor = create_example_id_extractor({".cif"})
    assert extractor("name.cif") == "name"


def test_extractor_strips_directory_components():
    extractor = create_example_id_extractor({".cif"})
    assert extractor("/some/dir/a.b.id.cif") == "id"


def test_extract_example_id_from_path_matches_extractor():
    assert extract_example_id_from_path("x.y.cif", CIF_LIKE_EXTENSIONS) == "y"


# --- find_files_with_extension ----------------------------------------------


def test_finds_files_matching_extensions_in_dir(tmp_path):
    (tmp_path / "a.cif").touch()
    (tmp_path / "b.json").touch()
    (tmp_path / "c.txt").touch()
    found = find_files_with_extension(tmp_path, [".cif", ".json"])
    assert {p.name for p in found} == {"a.cif", "b.json"}


def test_single_file_with_supported_extension_returned(tmp_path):
    f = tmp_path / "single.cif"
    f.touch()
    assert find_files_with_extension(f, [".cif"]) == [f]


def test_single_file_with_unsupported_extension_returns_empty(tmp_path):
    f = tmp_path / "single.txt"
    f.touch()
    assert find_files_with_extension(f, [".cif"]) == []


def test_nonexistent_path_returns_empty(tmp_path):
    assert find_files_with_extension(tmp_path / "missing", [".cif"]) == []


def test_does_not_recurse_into_subdirectories(tmp_path):
    """Pins the actual single-level glob behaviour (the docstring says 'Recursively')."""
    (tmp_path / "top.cif").touch()
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "deep.cif").touch()
    found = find_files_with_extension(tmp_path, [".cif"])
    assert {p.name for p in found} == {"top.cif"}


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
