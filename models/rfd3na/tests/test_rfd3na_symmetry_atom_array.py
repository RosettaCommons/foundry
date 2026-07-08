"""Unit tests for the pure atom-array helpers in ``rfd3na.inference.symmetry.atom_array``.

These annotate / rewrite a biotite ``AtomArray`` while building a symmetric assembly:

- ``reset_chain_ids`` — remaps the (sorted, unique) chain ids to consecutive characters
  starting at ``start_id`` and mirrors the result into ``pn_unit_iid``. The sole caller
  passes ``start_id="a"`` on uppercase input, so the new lowercase range never overlaps the
  old ids; the tests pin that production path (the overlapping-range collision is filed to
  the roadmap, not exercised here).
- ``reannotate_chain_ids`` — shifts every chain-id character by ``offset * multiplier`` (so
  ``multiplier=0`` is an identity) and mirrors it into ``pn_unit_iid``.
- ``add_sym_annotations`` — marks every atom as the asymmetric unit (``is_sym_asu``) and
  stamps the symmetry id onto each atom.
- ``apply_symmetry_to_atomarray_coord`` — applies a ``(R, T)`` frame as ``coord @ R.T + T``.
- ``get_2d_annotation_categories`` / ``add_2d_entity_annotations`` /
  ``reannotate_2d_conditions`` — round-trip the per-atom 2D-conditioning entity ids to and
  from the boolean ``*2d_condition*`` annotations.
"""

from types import SimpleNamespace

import numpy as np
from biotite.structure import AtomArray
from rfd3na.inference.symmetry.atom_array import (
    add_2d_entity_annotations,
    add_sym_annotations,
    apply_symmetry_to_atomarray_coord,
    get_2d_annotation_categories,
    reannotate_2d_conditions,
    reannotate_chain_ids,
    reset_chain_ids,
)


def _atom_array(chain_ids: list[str]) -> AtomArray:
    n = len(chain_ids)
    arr = AtomArray(n)
    arr.coord = np.zeros((n, 3), dtype=np.float32)
    arr.set_annotation("chain_id", np.array(chain_ids))
    return arr


# --- reset_chain_ids --------------------------------------------------------


def test_reset_chain_ids_remaps_from_start_char():
    arr = _atom_array(["A", "A", "B"])
    reset_chain_ids(arr, "a")
    assert arr.chain_id.tolist() == ["a", "a", "b"]


def test_reset_chain_ids_mirrors_into_pn_unit_iid():
    arr = _atom_array(["A", "B", "C"])
    reset_chain_ids(arr, "a")
    assert arr.pn_unit_iid.tolist() == arr.chain_id.tolist() == ["a", "b", "c"]


def test_reset_chain_ids_single_chain():
    arr = _atom_array(["X", "X"])
    reset_chain_ids(arr, "a")
    assert arr.chain_id.tolist() == ["a", "a"]


def test_reset_chain_ids_follows_sorted_unique_order():
    # The mapping is keyed on np.unique (sorted), independent of atom order.
    arr = _atom_array(["B", "A"])
    reset_chain_ids(arr, "a")
    assert arr.chain_id.tolist() == ["b", "a"]


# --- reannotate_chain_ids ---------------------------------------------------


def test_reannotate_chain_ids_multiplier_zero_is_identity():
    arr = _atom_array(["A", "B"])
    reannotate_chain_ids(arr, offset=5)  # multiplier defaults to 0 -> shift is 0
    assert arr.chain_id.tolist() == ["A", "B"]


def test_reannotate_chain_ids_shifts_by_offset_times_multiplier():
    arr = _atom_array(["A", "B"])
    reannotate_chain_ids(arr, offset=2, multiplier=1)  # shift +2: A->C, B->D
    assert arr.chain_id.tolist() == ["C", "D"]


def test_reannotate_chain_ids_mirrors_into_pn_unit_iid():
    arr = _atom_array(["A", "A"])
    reannotate_chain_ids(arr, offset=1, multiplier=2)  # shift +2
    assert arr.pn_unit_iid.tolist() == arr.chain_id.tolist() == ["C", "C"]


# --- add_sym_annotations ----------------------------------------------------


def test_add_sym_annotations_marks_all_asu_and_stamps_id():
    arr = _atom_array(["A", "A", "A"])
    add_sym_annotations(arr, SimpleNamespace(id="C3"))
    assert arr.is_sym_asu.tolist() == [True, True, True]
    assert arr.symmetry_id.tolist() == ["C3", "C3", "C3"]


def test_add_sym_annotations_returns_same_array():
    arr = _atom_array(["A"])
    assert add_sym_annotations(arr, SimpleNamespace(id="D2")) is arr


# --- apply_symmetry_to_atomarray_coord --------------------------------------


def test_apply_symmetry_rotates_coords():
    arr = _atom_array(["A"])
    arr.coord = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    rot_z_90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    apply_symmetry_to_atomarray_coord(arr, (rot_z_90, np.zeros(3, dtype=np.float32)))
    assert np.allclose(arr.coord, [[0.0, 1.0, 0.0]])


def test_apply_symmetry_adds_translation():
    arr = _atom_array(["A", "A"])
    translation = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    apply_symmetry_to_atomarray_coord(arr, (np.eye(3, dtype=np.float32), translation))
    assert np.allclose(arr.coord, [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])


def test_apply_symmetry_identity_frame_is_noop():
    arr = _atom_array(["A"])
    arr.coord = np.array([[4.0, 5.0, 6.0]], dtype=np.float32)
    apply_symmetry_to_atomarray_coord(
        arr, (np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
    )
    assert np.allclose(arr.coord, [[4.0, 5.0, 6.0]])


# --- get_2d_annotation_categories -------------------------------------------


def test_get_2d_categories_filters_and_sorts():
    arr = _atom_array(["A", "A"])
    arr.set_annotation("2d_condition_1", np.array([1, 0]))
    arr.set_annotation("2d_condition_0", np.array([0, 1]))
    arr.set_annotation("chain_type", np.array([0, 0]))
    assert get_2d_annotation_categories(arr) == ["2d_condition_0", "2d_condition_1"]


def test_get_2d_categories_empty_when_none_present():
    arr = _atom_array(["A"])
    assert get_2d_annotation_categories(arr) == []


# --- add_2d_entity_annotations ----------------------------------------------


def test_add_2d_entity_annotations_assigns_per_category_ids():
    arr = _atom_array(["A", "A", "B"])
    arr.set_annotation("2d_condition_0", np.array([0, 1, 0]))
    arr.set_annotation("2d_condition_1", np.array([1, 0, 0]))
    add_2d_entity_annotations(arr)
    # sorted category '2d_condition_0' -> entity 1 (atom 1); '2d_condition_1' -> 2 (atom 0)
    assert arr._2d_entity_id.tolist() == [2, 1, 0]


# --- reannotate_2d_conditions -----------------------------------------------


def test_reannotate_2d_conditions_round_trips_add():
    arr = _atom_array(["A", "A", "B"])
    arr.set_annotation("2d_condition_0", np.array([0, 1, 0]))
    arr.set_annotation("2d_condition_1", np.array([1, 0, 0]))
    add_2d_entity_annotations(arr)
    reannotate_2d_conditions(arr)
    assert arr.get_annotation("2d_condition_0").tolist() == [False, True, False]
    assert arr.get_annotation("2d_condition_1").tolist() == [True, False, False]


def test_reannotate_2d_conditions_deletes_entity_id():
    arr = _atom_array(["A", "A"])
    arr.set_annotation("2d_condition_0", np.array([1, 0]))
    add_2d_entity_annotations(arr)
    reannotate_2d_conditions(arr)
    assert "_2d_entity_id" not in arr.get_annotation_categories()


def test_reannotate_2d_conditions_synthesizes_extra_categories():
    # More non-zero entity ids than existing 2d_condition categories -> derived names.
    arr = _atom_array(["A", "A", "B"])
    arr.set_annotation("2d_condition_0", np.array([0, 0, 0]))
    arr.set_annotation("_2d_entity_id", np.array([1, 2, 0]))
    reannotate_2d_conditions(arr)
    assert get_2d_annotation_categories(arr) == ["2d_condition_0", "2d_condition_0_1"]
