"""Unit tests for rfd3.inference.symmetry.frames.

These pure functions build and manipulate the symmetry frames that drive RFD3's
symmetric-assembly generation: the cyclic / dihedral rotation sets, the
frame <-> (rotation, translation) conversions used in the symmetry loss, and the
Kabsch alignment that recovers a transform from two coordinate sets. Their
contracts — a frame is an `(R, t)` pair; `Cn` is `n` proper rotations about z;
`Dn` is `2n`; the framecoord conversion round-trips; `_align` recovers an exact
rigid transform — are not obvious from the signatures, so the tests pin them on
small CPU inputs.

One sharp edge is pinned deliberately: `is_valid_rotation_matrix` checks only
orthogonality (`R @ R.T == I`), not `det(R) == +1`, so it accepts reflections
(see the roadmap finding on tightening it).
"""

import numpy as np
import pytest
import torch
from rfd3.inference.symmetry.frames import (
    RTs_to_framecoords,
    _align,
    _rms,
    decompose_symmetry_frame,
    framecoords_to_RTs,
    get_cyclic_frames,
    get_dihedral_frames,
    get_symmetry_frames_from_symmetry_id,
    is_valid_rotation_matrix,
    pack_vector,
    unpack_vector,
)

# --- is_valid_rotation_matrix -------------------------------------------------


def test_identity_is_valid_rotation():
    assert is_valid_rotation_matrix(np.eye(3))


def test_proper_rotation_is_valid():
    R = get_cyclic_frames(4)[1][0]  # 90 deg about z
    assert is_valid_rotation_matrix(R)


def test_non_orthogonal_matrix_is_invalid():
    assert not is_valid_rotation_matrix(2 * np.eye(3))


def test_reflection_passes_orthogonality_only_check():
    """`is_valid_rotation_matrix` constrains orthogonality, not determinant.

    A reflection (det -1) is orthogonal, so it is accepted even though it is not
    a proper rotation. Pinned to document the actual contract; see the roadmap
    finding on tightening this to also require det == +1.
    """
    reflection = np.diag([1.0, 1.0, -1.0])
    assert np.isclose(np.linalg.det(reflection), -1.0)
    assert is_valid_rotation_matrix(reflection)


# --- get_cyclic_frames --------------------------------------------------------


def test_cyclic_frame_count_and_zero_translation():
    frames = get_cyclic_frames(3)
    assert len(frames) == 3
    for _, t in frames:
        assert np.array_equal(t, np.zeros(3))


def test_cyclic_first_frame_is_identity():
    R, _ = get_cyclic_frames(6)[0]
    assert np.allclose(R, np.eye(3))


def test_cyclic_frames_are_proper_rotations():
    for R, _ in get_cyclic_frames(5):
        assert is_valid_rotation_matrix(R)
        assert np.isclose(np.linalg.det(R), 1.0)


def test_cyclic_frame_rotates_about_z_by_expected_angle():
    # order 4, index 1 -> 90 deg CCW about z: e_x -> e_y, z fixed.
    R, _ = get_cyclic_frames(4)[1]
    assert np.allclose(R @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)
    assert np.allclose(R @ np.array([0.0, 0.0, 1.0]), [0.0, 0.0, 1.0])


def test_cyclic_generator_has_order_n():
    # applying the unit rotation `order` times returns to identity.
    order = 7
    R = get_cyclic_frames(order)[1][0]
    assert np.allclose(np.linalg.matrix_power(R, order), np.eye(3), atol=1e-9)


# --- get_dihedral_frames ------------------------------------------------------


def test_dihedral_frame_count_is_double_order():
    assert len(get_dihedral_frames(3)) == 6


def test_dihedral_frames_are_proper_rotations():
    # both the rotation frames and the flipped frames are proper rotations.
    for R, t in get_dihedral_frames(4):
        assert np.array_equal(t, np.zeros(3))
        assert is_valid_rotation_matrix(R)
        assert np.isclose(np.linalg.det(R), 1.0)


def test_dihedral_even_frames_match_cyclic():
    order = 3
    dihedral = get_dihedral_frames(order)
    cyclic = get_cyclic_frames(order)
    for i in range(order):
        assert np.allclose(dihedral[2 * i][0], cyclic[i][0])


# --- get_symmetry_frames_from_symmetry_id -------------------------------------


def test_symmetry_id_cyclic():
    frames = get_symmetry_frames_from_symmetry_id("C2")
    assert len(frames) == 2
    assert all(is_valid_rotation_matrix(R) for R, _ in frames)


def test_symmetry_id_dihedral():
    assert len(get_symmetry_frames_from_symmetry_id("D2")) == 4


def test_symmetry_id_is_case_insensitive():
    assert len(get_symmetry_frames_from_symmetry_id("c3")) == 3
    assert len(get_symmetry_frames_from_symmetry_id("d3")) == 6


def test_symmetry_id_unsupported_raises():
    with pytest.raises(ValueError, match="not supported"):
        get_symmetry_frames_from_symmetry_id("X9")


# --- RTs_to_framecoords <-> framecoords_to_RTs --------------------------------


def test_framecoord_roundtrip_recovers_rotation_and_translation():
    R = torch.tensor(get_cyclic_frames(5)[1][0], dtype=torch.float64)
    t = torch.tensor([3.0, -2.0, 5.0], dtype=torch.float64)
    Ori, X, Y = RTs_to_framecoords(R, t, sig=1.0)
    R_rec, T_rec = framecoords_to_RTs(Ori, X, Y)
    assert torch.allclose(R_rec, R, atol=1e-5)
    assert torch.allclose(T_rec, t, atol=1e-5)


def test_RTs_to_framecoords_accepts_numpy_and_returns_torch():
    R = get_cyclic_frames(4)[1][0]  # numpy
    t = np.array([1.0, 2.0, 3.0])
    Ori, X, Y = RTs_to_framecoords(R, t)
    assert isinstance(Ori, torch.Tensor)
    assert isinstance(X, torch.Tensor)
    # Ori is the translation; X/Y sit one unit along the first two rotation rows.
    assert torch.allclose(Ori, torch.from_numpy(t))


# --- pack_vector / unpack_vector ----------------------------------------------


def test_pack_unpack_roundtrip_preserves_values_and_dtype():
    v = np.array([1.5, -2.0, 3.25], dtype=np.float64)
    packed = pack_vector(v)
    assert packed.shape == (1,)
    unpacked = unpack_vector(packed)
    assert unpacked.shape == (1, 3)
    assert np.array_equal(unpacked[0], v)
    assert unpacked.dtype == v.dtype


def test_pack_vector_preserves_integer_dtype():
    v = np.array([1, 2, 3], dtype=np.int32)
    assert unpack_vector(pack_vector(v)).dtype == np.int32


# --- _align / _rms (Kabsch) ---------------------------------------------------


def test_align_recovers_known_rigid_transform():
    rng = np.random.default_rng(0)
    X_moving = rng.normal(size=(8, 3))
    R_true = get_cyclic_frames(4)[1][0]  # 90 deg about z
    centroid = np.array([10.0, -3.0, 2.0])
    X_fixed = (X_moving - X_moving.mean(axis=0)) @ R_true.T + centroid

    u_moving, R, u_fixed = _align(X_fixed, X_moving)
    assert is_valid_rotation_matrix(R)
    assert np.allclose(R, R_true, atol=1e-6)
    assert np.allclose(u_fixed, centroid, atol=1e-6)
    # the recovered transform aligns moving onto fixed with ~zero RMSD.
    assert _rms(X_fixed, X_moving, u_moving, R, u_fixed) < 1e-6


def test_align_identical_point_sets_is_identity():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(6, 3))
    _, R, _ = _align(X, X)
    assert np.allclose(R, np.eye(3), atol=1e-6)


# --- decompose_symmetry_frame -------------------------------------------------


def test_decompose_symmetry_frame_origin_is_translation():
    R = get_cyclic_frames(4)[1][0]
    T = np.array([1.0, 2.0, 3.0])
    Ori, _X, _Y = decompose_symmetry_frame((R, T))
    # each returned value is a packed (1,) structured array; the origin is T.
    assert np.allclose(unpack_vector(Ori)[0], T, atol=1e-6)
