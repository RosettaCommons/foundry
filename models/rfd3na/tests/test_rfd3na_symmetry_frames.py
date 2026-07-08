"""Unit tests for the pure geometry helpers in ``rfd3na.inference.symmetry.frames``.

Covered:

- ``get_cyclic_frames(order)`` / ``get_dihedral_frames(order)`` — the Cn / Dn frame
  generators. Cn yields ``order`` proper rotations about z (angles ``2*pi*i/order``); Dn
  yields ``2*order`` proper rotations (each Cn rotation plus a 180-degree flip about an
  in-plane axis — ``2*u*u^T - I``, which is itself a proper rotation, so every Dn frame
  has ``det == +1``).
- ``is_valid_rotation_matrix(R)`` — an orthogonality-only check (``R @ R.T == I``); it
  therefore *accepts reflections* (det -1) despite the name. Pinned below.
- ``_align(X_fixed, X_moving)`` — Kabsch superposition. Returns
  ``(mean_moving, R, mean_fixed)`` such that ``R @ (X_moving - mean_moving) + mean_fixed``
  reproduces ``X_fixed``; ``R`` is the moving->fixed rotation (i.e. the inverse of the
  rotation that was applied to build ``X_moving``). Supports numpy and torch inputs.
- ``_rms(...)`` — RMSD of ``R @ (X_moving - t_pre) + t_post`` against ``X_fixed``,
  normalised by the number of points.
- ``RTs_to_framecoords`` / ``framecoords_to_RTs`` — round-trip between a rotation+origin
  and three "virtual frame" points (Gram-Schmidt), used by the symmetry loss.
- ``pack_vector`` / ``unpack_vector`` — structured-array (de)serialisation of a 3-vector.

The torch ``_align`` path is exercised in float32 (the production dtype): float64 input
trips a ``torch.eye`` dtype footgun in the module (filed in .ai/roadmap.md), so f32 pins
the reachable behaviour.
"""

import numpy as np
import torch
from rfd3na.inference.symmetry.frames import (
    RTs_to_framecoords,
    _align,
    _rms,
    framecoords_to_RTs,
    get_cyclic_frames,
    get_dihedral_frames,
    is_valid_rotation_matrix,
    pack_vector,
    unpack_vector,
)


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# --- is_valid_rotation_matrix -----------------------------------------------


def test_identity_is_valid_rotation():
    assert is_valid_rotation_matrix(np.eye(3))


def test_proper_rotation_is_valid():
    assert is_valid_rotation_matrix(_rot_z(0.9))


def test_non_orthogonal_matrix_rejected():
    assert not is_valid_rotation_matrix(np.array([[2.0, 0, 0], [0, 1, 0], [0, 0, 1]]))


def test_reflection_passes_orthogonality_only_check():
    """Pins the known behaviour: the check is orthogonality-only, so a det -1
    reflection is (incorrectly, per the name) accepted. See .ai/roadmap.md finding."""
    reflection = np.diag([1.0, 1.0, -1.0])
    assert np.isclose(np.linalg.det(reflection), -1.0)
    assert is_valid_rotation_matrix(reflection)


# --- get_cyclic_frames ------------------------------------------------------


def test_cyclic_frame_count_matches_order():
    assert len(get_cyclic_frames(5)) == 5


def test_cyclic_frames_are_proper_rotations_with_zero_translation():
    for R, t in get_cyclic_frames(6):
        assert is_valid_rotation_matrix(R)
        assert np.isclose(np.linalg.det(R), 1.0)
        assert np.array_equal(t, np.array([0, 0, 0]))


def test_cyclic_first_frame_is_identity_and_angles_are_evenly_spaced():
    frames = get_cyclic_frames(4)
    assert np.allclose(frames[0][0], np.eye(3))
    # C4: the i-th frame is a rotation by 2*pi*i/4 about z.
    assert np.allclose(frames[1][0], _rot_z(np.pi / 2))
    assert np.allclose(frames[2][0], _rot_z(np.pi))


# --- get_dihedral_frames ----------------------------------------------------


def test_dihedral_frame_count_is_twice_order():
    assert len(get_dihedral_frames(3)) == 6


def test_dihedral_frames_are_all_proper_rotations():
    # The 180-degree flip 2*u*u^T - I is a proper rotation, so R @ flip stays det +1.
    for R, t in get_dihedral_frames(4):
        assert is_valid_rotation_matrix(R)
        assert np.isclose(np.linalg.det(R), 1.0)
        assert np.array_equal(t, np.array([0, 0, 0]))


# --- _align (Kabsch) --------------------------------------------------------


def _make_pair(rng, dtype=np.float64):
    x = rng.normal(size=(8, 3)).astype(dtype)
    r_applied = _rot_z(0.7).astype(dtype)
    t_applied = np.array([1.0, -2.0, 3.0], dtype=dtype)
    x_moving = (r_applied @ x.T).T + t_applied
    return x, x_moving, r_applied


def test_align_numpy_recovers_superposition():
    rng = np.random.default_rng(0)
    x_fixed, x_moving, r_applied = _make_pair(rng)
    mean_moving, r, mean_fixed = _align(x_fixed, x_moving)
    x_aligned = (
        np.einsum("ij,bj->bi", r, x_moving - mean_moving[None]) + mean_fixed[None]
    )
    rms = np.sqrt(np.sum((x_aligned - x_fixed) ** 2) / x_fixed.shape[0])
    assert rms < 1e-6
    assert is_valid_rotation_matrix(r)
    # _align returns the moving->fixed rotation, i.e. the inverse of what was applied.
    assert np.allclose(r, r_applied.T, atol=1e-6)


def test_align_torch_float32_recovers_superposition():
    rng = np.random.default_rng(1)
    x_fixed_np, x_moving_np, _ = _make_pair(rng, dtype=np.float32)
    x_fixed = torch.from_numpy(x_fixed_np)
    x_moving = torch.from_numpy(x_moving_np)
    mean_moving, r, mean_fixed = _align(x_fixed, x_moving)
    x_aligned = (
        torch.einsum("ij,bj->bi", r, x_moving - mean_moving[None]) + mean_fixed[None]
    )
    rms = torch.sqrt(((x_aligned - x_fixed) ** 2).sum() / x_fixed.shape[0])
    assert rms < 1e-4


# --- _rms -------------------------------------------------------------------


def test_rms_zero_for_identical_coords():
    x = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    assert _rms(x, x.copy(), np.zeros(3), np.eye(3), np.zeros(3)) == 0.0


def test_rms_reflects_post_translation():
    x_fixed = np.array([[0.0, 0.0, 0.0]])
    x_moving = np.array([[1.0, 0.0, 0.0]])
    # identity R, no pre-translation: aligned == x_moving + t_post.
    assert _rms(x_fixed, x_moving, np.zeros(3), np.eye(3), np.zeros(3)) == 1.0
    assert (
        _rms(x_fixed, x_moving, np.array([1.0, 0.0, 0.0]), np.eye(3), np.zeros(3))
        == 0.0
    )


# --- RTs_to_framecoords / framecoords_to_RTs --------------------------------


def test_framecoords_round_trip_recovers_rotation():
    r = _rot_z(0.3)
    origin = np.array([0.0, 0.0, 0.0])
    ori, x, y = RTs_to_framecoords(r, origin)
    r_recovered, t_recovered = framecoords_to_RTs(ori, x, y)
    # Gram-Schmidt reconstructs the frame from its first two (orthonormal) rows.
    assert np.allclose(r_recovered.numpy()[:2], r[:2], atol=1e-4)
    assert is_valid_rotation_matrix(r_recovered.numpy())
    assert np.allclose(t_recovered.numpy(), origin, atol=1e-6)


# --- pack_vector / unpack_vector --------------------------------------------


def test_pack_unpack_round_trip():
    v = np.array([1.5, -2.0, 3.25])
    assert np.array_equal(unpack_vector(pack_vector(v))[0], v)
