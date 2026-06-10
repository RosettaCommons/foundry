"""Unit tests for the symmetry-resolution geometry in ``rf3.loss.af3_losses``.

Both helpers below are the load-bearing machinery behind ``SubunitSymmetryResolution``
and ``ResidueSymmetryResolution`` (used by ``rf3.symmetry.resolve`` /
``rf3.trainers.rf3``), which re-label the ground-truth coordinates to the symmetry
copy / automorphism that best matches the prediction before the loss is taken.

- ``SubunitSymmetryResolution._rms_align`` is a batched Kabsch fit. Given predicted
  coordinates ``X_fixed`` (``Nbatch x L x 3``) and candidate native copies ``X_moving``
  (``Nambig x L x 3``) it returns ``(u_moving, R, u_fixed)`` such that
  ``(X_moving - u_moving) @ R + u_fixed`` lands the native on the prediction — the exact
  transform ``_resolve_subunits`` then applies to the native centres of mass. The SVD is
  sign-corrected so ``R`` is always a proper rotation (``det = +1``), never a reflection.
- ``ResidueSymmetryResolution._get_best`` picks, per model in the batch, the atom
  automorphism (a permutation of a set of interchangeable atom indices) whose
  intra-structure distance pattern best matches the prediction, then rewrites the native
  coordinates / mask at those positions to that permutation.

All tests run in float32 (production dtype); ``_rms_align`` builds its sign-correction
matrix with an un-typed ``torch.eye`` that only matches float32 inputs (see the roadmap).
"""

import pytest
import torch
from rf3.loss.af3_losses import ResidueSymmetryResolution, SubunitSymmetryResolution

# A non-degenerate, non-coplanar point cloud to align (L = 5).
_POINTS = torch.tensor(
    [
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 3.0],
        [1.0, 1.0, 1.0],
        [-1.0, 2.0, -1.0],
    ]
)


def _rotation_z(theta: float) -> torch.Tensor:
    """Proper rotation (det +1) about the z-axis by ``theta`` radians."""
    c, s = torch.cos(torch.tensor(theta)), torch.sin(torch.tensor(theta))
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# --- SubunitSymmetryResolution._rms_align -----------------------------------


def test_rms_align_recovers_a_rigid_transform():
    # X_moving is X_fixed pushed through a known rotation + translation; the returned
    # transform must land it back on X_fixed.
    fixed = _POINTS[None]  # (Nbatch=1, L, 3)
    moving = (_POINTS @ _rotation_z(0.7) + torch.tensor([2.0, -1.0, 3.0]))[None]

    u_moving, R, u_fixed = SubunitSymmetryResolution()._rms_align(fixed, moving)

    aligned = (moving[0] - u_moving[0, 0]) @ R[0, 0] + u_fixed[0, 0]
    assert torch.allclose(aligned, _POINTS, atol=1e-4)
    # Sign-corrected SVD → a proper rotation, not a reflection.
    assert torch.linalg.det(R[0, 0]).item() == pytest.approx(1.0, abs=1e-4)


def test_rms_align_identity_when_moving_equals_fixed():
    fixed = _POINTS[None]
    u_moving, R, u_fixed = SubunitSymmetryResolution()._rms_align(fixed, _POINTS[None])

    assert torch.allclose(R[0, 0], torch.eye(3), atol=1e-4)
    assert torch.allclose(u_moving[0, 0], u_fixed[0, 0], atol=1e-4)
    aligned = (_POINTS - u_moving[0, 0]) @ R[0, 0] + u_fixed[0, 0]
    assert torch.allclose(aligned, _POINTS, atol=1e-4)


def test_rms_align_corrects_reflection_to_proper_rotation():
    # A mirror image cannot be rotated onto the original; the sign correction must still
    # return a proper rotation (det +1) rather than the optimal-but-improper reflection.
    fixed = _POINTS[None]
    reflected = (_POINTS @ torch.diag(torch.tensor([1.0, 1.0, -1.0])))[None]

    _, R, _ = SubunitSymmetryResolution()._rms_align(fixed, reflected)

    assert torch.linalg.det(R[0, 0]).item() == pytest.approx(1.0, abs=1e-4)


def test_rms_align_output_shapes_broadcast_over_ambig_and_batch():
    # u_moving is per-ambiguity, u_fixed is per-batch, R is the full cross product — the
    # broadcast-ready shapes _resolve_subunits relies on.
    fixed = _POINTS[None].repeat(3, 1, 1)  # Nbatch = 3
    moving = _POINTS[None].repeat(2, 1, 1)  # Nambig = 2

    u_moving, R, u_fixed = SubunitSymmetryResolution()._rms_align(fixed, moving)

    assert u_moving.shape == (2, 1, 3)
    assert R.shape == (2, 3, 3, 3)
    assert u_fixed.shape == (1, 3, 3)


# --- ResidueSymmetryResolution._get_best ------------------------------------


def _get_best_inputs(pred_sym, native_sym, context):
    """Build (x_pred, x_native, x_native_mask, a_i) for a 1-model, 3-atom case.

    Atoms 0 and 1 are interchangeable (the automorphism set); atom 2 is fixed context
    whose distances to atoms 0/1 break the tie. ``a_i`` offers the identity ordering
    ``[0, 1]`` and the swap ``[1, 0]``.
    """
    x_pred = torch.tensor([[*pred_sym, context]])  # (1, 3, 3)
    x_native = torch.tensor([[*native_sym, context]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    a_i = torch.tensor([[0, 1], [1, 0]])
    return x_pred, x_native, mask, a_i


def test_get_best_selects_the_swap_that_matches_prediction():
    # Native has atoms 0/1 swapped relative to the prediction; _get_best must undo it so
    # the native's interchangeable atoms line up with the prediction's arrangement.
    pred_sym = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    native_sym = [[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]  # swapped
    context = [0.0, 3.0, 0.0]
    x_pred, x_native, mask, a_i = _get_best_inputs(pred_sym, native_sym, context)

    out_native, _ = ResidueSymmetryResolution()._get_best(x_pred, x_native, mask, a_i)

    assert torch.allclose(out_native[0, 0], torch.tensor([0.0, 0.0, 0.0]))
    assert torch.allclose(out_native[0, 1], torch.tensor([2.0, 0.0, 0.0]))


def test_get_best_leaves_already_matching_native_unchanged():
    # Native already matches the prediction → the identity ordering wins → no rewrite.
    sym = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    context = [0.0, 3.0, 0.0]
    x_pred, x_native, mask, a_i = _get_best_inputs(sym, sym, context)
    before = x_native.clone()

    out_native, _ = ResidueSymmetryResolution()._get_best(x_pred, x_native, mask, a_i)

    assert torch.allclose(out_native, before)
