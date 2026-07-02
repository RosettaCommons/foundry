"""Unit tests for the validation and symmetry-check helpers in ``rfd3na.inference.symmetry.checks``.

These guard a symmetric-assembly input before frame computation. The first group each raises a
clear error (or, for ``check_max_rmsds``, only warns) when its contract is violated:

- ``check_valid_multiplicity`` — the per-entity chain counts must share a common multiplicity
  > 1 (every entity's count divisible by the minimum); multiplicity 1 or a non-divisible
  count raises.
- ``check_valid_subunit_size`` — every chain of an entity must span the same number of atoms.
- ``check_min_atoms_to_align`` — the reference entity needs at least ``MIN_ATOMS_ALIGN`` atoms.
- ``check_max_transforms`` — at most ``MAX_TRANSFORMS`` chains.
- ``check_input_frames_match_symmetry_frames`` — computed vs original frame counts must match.
- ``check_max_rmsds`` — warns (does *not* raise) when any RMSD exceeds ``RMSD_CUT``.
- ``find_optimal_rotation`` — Kabsch rotation (numpy) aligning two equal-length point sets;
  translation-invariant, forces a proper rotation (det +1) on reflections, truncates unequal
  lengths to the shorter, and returns ``None`` for fewer than 3 points.
- ``check_atom_array_is_symmetric`` — structural gate over a multi-chain ``AtomArray``:
  ``False`` on a per-chain atom-count or atom-name mismatch, ``True`` for a well-formed
  assembly. Its final "aligns with the symmetry frames" step only requires >= 3 alignable
  atoms — it does *not* verify the chains are truly symmetric — so a matching-shape but
  geometrically asymmetric input also returns ``True`` (filed to the roadmap).
"""

import numpy as np
import pytest
from biotite.structure import AtomArray
from rfd3na.inference.symmetry.checks import (
    MAX_TRANSFORMS,
    MIN_ATOMS_ALIGN,
    check_atom_array_is_symmetric,
    check_input_frames_match_symmetry_frames,
    check_max_rmsds,
    check_max_transforms,
    check_min_atoms_to_align,
    check_valid_multiplicity,
    check_valid_subunit_size,
    find_optimal_rotation,
)


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# --- check_valid_multiplicity -----------------------------------------------


def test_multiplicity_ok_for_uniform_dimer():
    check_valid_multiplicity({0: ["A_1", "B_1"], 1: ["A_2", "B_2"]})


def test_multiplicity_one_raises():
    with pytest.raises(ValueError, match="no possible symmetry"):
        check_valid_multiplicity({0: ["A_1"]})


def test_multiplicity_non_divisible_raises():
    # min multiplicity is 2, but the first entity has 3 chains (3 % 2 != 0).
    with pytest.raises(ValueError, match="multiplicity"):
        check_valid_multiplicity({0: ["A_1", "B_1", "C_1"], 1: ["A_2", "B_2"]})


# --- check_valid_subunit_size -----------------------------------------------


def test_subunit_size_ok_when_chains_equal_length():
    pn_unit_id = np.array(["A_1", "A_1", "B_1", "B_1"])
    check_valid_subunit_size({0: ["A_1", "B_1"]}, pn_unit_id)


def test_subunit_size_mismatch_raises():
    pn_unit_id = np.array(["A_1", "A_1", "B_1"])  # A_1 has 2 atoms, B_1 has 1
    with pytest.raises(ValueError, match="Size mismatch"):
        check_valid_subunit_size({0: ["A_1", "B_1"]}, pn_unit_id)


# --- check_min_atoms_to_align -----------------------------------------------


def test_min_atoms_ok_at_threshold():
    check_min_atoms_to_align({0: MIN_ATOMS_ALIGN}, 0)


def test_min_atoms_below_threshold_raises():
    with pytest.raises(ValueError, match="Not enough atoms"):
        check_min_atoms_to_align({0: MIN_ATOMS_ALIGN - 1}, 0)


# --- check_max_transforms ---------------------------------------------------


def test_max_transforms_ok_at_limit():
    check_max_transforms(list(range(MAX_TRANSFORMS)))


def test_max_transforms_over_limit_raises():
    with pytest.raises(ValueError, match="exceeds the max"):
        check_max_transforms(list(range(MAX_TRANSFORMS + 1)))


# --- check_input_frames_match_symmetry_frames -------------------------------


def test_frame_counts_match_ok():
    check_input_frames_match_symmetry_frames([1, 2, 3], ["a", "b", "c"], {})


def test_frame_counts_mismatch_raises():
    with pytest.raises(AssertionError):
        check_input_frames_match_symmetry_frames([1, 2], ["a", "b", "c"], {})


# --- check_max_rmsds --------------------------------------------------------


def test_max_rmsds_within_cut_does_not_raise():
    check_max_rmsds({"A": 0.5, "B": 0.9})


def test_max_rmsds_over_cut_warns_without_raising():
    # Contract: an oversized RMSD is a soft warning, not a hard failure.
    check_max_rmsds({"A": 5.0})


# --- find_optimal_rotation --------------------------------------------------

# Five points in general position (not all coplanar) so Kabsch is well-conditioned.
_CLOUD = np.array(
    [[1.0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=float
)


def test_find_rotation_identity_for_equal_coords():
    R = find_optimal_rotation(_CLOUD, _CLOUD)
    assert np.allclose(R, np.eye(3), atol=1e-6)


def test_find_rotation_recovers_known_rotation():
    r_true = _rot_z(np.pi / 6)
    R = find_optimal_rotation(_CLOUD, _CLOUD @ r_true.T)
    assert np.allclose(R, r_true, atol=1e-6)


def test_find_rotation_is_translation_invariant():
    # Kabsch removes both centroids first, so a pure shift maps to the identity.
    R = find_optimal_rotation(_CLOUD, _CLOUD + np.array([5.0, -3.0, 2.0]))
    assert np.allclose(R, np.eye(3), atol=1e-6)


def test_find_rotation_forces_proper_rotation_on_reflection():
    # A pure reflection (z negated) has no proper-rotation alignment; the det
    # correction flips the sign so the result is a rotation (det +1), not a reflection.
    R = find_optimal_rotation(_CLOUD, _CLOUD * np.array([1.0, 1.0, -1.0]))
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)


def test_find_rotation_returns_none_below_three_points():
    assert find_optimal_rotation(_CLOUD[:2], _CLOUD[:2]) is None


def test_find_rotation_truncates_to_shorter_length():
    # Unequal lengths are truncated to the shorter set (here 3), then aligned.
    r_true = _rot_z(np.pi / 4)
    R = find_optimal_rotation(_CLOUD, _CLOUD[:3] @ r_true.T)
    assert np.allclose(R, r_true, atol=1e-6)


def test_find_rotation_subsamples_above_max_points():
    # With len > max_points the points are randomly subsampled; for a rigid transform
    # any subsample recovers the same rotation (seeded here for a deterministic pick).
    np.random.seed(0)
    cloud = np.random.rand(8, 3)
    r_true = _rot_z(np.pi / 5)
    R = find_optimal_rotation(cloud, cloud @ r_true.T, max_points=4)
    assert np.allclose(R, r_true, atol=1e-6)


# --- check_atom_array_is_symmetric ------------------------------------------

# A 4-atom asymmetric unit and the z-180 rotation that maps it to a C2 partner.
_ASU = np.array([[1.0, 0, 0], [0, 1, 0], [0, 0, 2], [1, 1, 1]], dtype=float)
_ASU_NAMES = ["N", "CA", "C", "O"]
_R180_Z = np.array([[-1.0, 0, 0], [0, -1, 0], [0, 0, 1]])


def _sym_atom_array(
    chain_ids: list[str],
    atom_names: list[str],
    coord: np.ndarray,
    *,
    hetero: list[bool] | None = None,
    symmetry_id: str = "C2",
) -> AtomArray:
    n = len(chain_ids)
    arr = AtomArray(n)
    arr.coord = np.asarray(coord, dtype=np.float32)
    arr.set_annotation("chain_id", np.array(chain_ids))
    arr.set_annotation("atom_name", np.array(atom_names))
    arr.set_annotation("hetero", np.array([False] * n if hetero is None else hetero))
    arr.set_annotation("symmetry_id", np.array([symmetry_id] * n))
    return arr


def test_symmetric_c2_input_returns_true():
    coord = np.vstack([_ASU, _ASU @ _R180_Z.T])
    arr = _sym_atom_array(["A"] * 4 + ["B"] * 4, _ASU_NAMES * 2, coord)
    assert check_atom_array_is_symmetric(arr) is True


def test_matching_shape_but_asymmetric_still_returns_true():
    # Documents the weak alignment step: a merely translated (non-C2) second chain
    # matches the atom count and names and has >= 3 alignable atoms, so it passes even
    # though it is not related to the first chain by the symmetry frames.
    coord = np.vstack([_ASU, _ASU + np.array([5.0, 5.0, 5.0])])
    arr = _sym_atom_array(["A"] * 4 + ["B"] * 4, _ASU_NAMES * 2, coord)
    assert check_atom_array_is_symmetric(arr) is True


def test_mismatched_atom_counts_returns_false():
    coord = np.vstack([_ASU, _ASU[:3]])
    arr = _sym_atom_array(["A"] * 4 + ["B"] * 3, _ASU_NAMES + _ASU_NAMES[:3], coord)
    assert check_atom_array_is_symmetric(arr) is False


def test_mismatched_atom_names_returns_false():
    coord = np.vstack([_ASU, _ASU @ _R180_Z.T])
    arr = _sym_atom_array(
        ["A"] * 4 + ["B"] * 4, _ASU_NAMES + ["N", "CA", "C", "X"], coord
    )
    assert check_atom_array_is_symmetric(arr) is False


def test_all_hetero_returns_true():
    # Removing hetero atoms empties the array; the "no protein chains" path returns True.
    arr = _sym_atom_array(["A"] * 4, _ASU_NAMES, _ASU, hetero=[True] * 4)
    assert check_atom_array_is_symmetric(arr) is True
