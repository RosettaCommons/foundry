"""Unit tests for foundry.utils.alignment.

`weighted_rigid_align` is the SE(3) Kabsch alignment used across rf3/rfd3/rfd3na
to align predicted/ground-truth coordinates before computing losses and during
sampling. Its contract (recover an exact rigid transform, ignore points the
mask/weights exclude from the fit, detach the output) is not obvious from the
signature, so the tests below pin it on representative CPU inputs.

Most tests use float32 to match production call sites; one float64 test guards
that the det-correction matrix follows the input dtype rather than defaulting to
float32 (which used to make float64 inputs raise).
"""

import pytest
import torch

from foundry.utils.alignment import get_rmsd, weighted_rigid_align


def _rotation_about_z(angle: float, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Proper rotation (det +1) about the z-axis, as a [3, 3] matrix in `dtype`."""
    a = torch.tensor(angle, dtype=dtype)
    c, s = torch.cos(a), torch.sin(a)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=dtype)


def test_identity_alignment_is_noop():
    """Aligning a structure onto itself returns the structure unchanged."""
    torch.manual_seed(0)
    X = torch.randn(1, 16, 3)
    aligned = weighted_rigid_align(X, X)
    assert torch.allclose(aligned, X, atol=1e-4)


def test_recovers_known_rigid_transform():
    """A rigid image of X aligns back onto X regardless of the rotation/translation."""
    torch.manual_seed(0)
    X = torch.randn(1, 16, 3)
    R = _rotation_about_z(1.0)
    t = torch.tensor([3.0, -2.0, 5.0])
    X_gt = X @ R.T + t  # an exact rigid image of X

    aligned = weighted_rigid_align(X, X_gt)
    assert torch.allclose(aligned, X, atol=1e-4)


def test_recovers_pure_translation():
    """Translation-only ground truth aligns back exactly."""
    torch.manual_seed(0)
    X = torch.randn(1, 16, 3)
    X_gt = X + torch.tensor([10.0, -4.0, 0.5])
    aligned = weighted_rigid_align(X, X_gt)
    assert torch.allclose(aligned, X, atol=1e-4)


def test_float64_inputs_supported():
    """float64 coordinates align in float64 — det-correction matrix follows the input dtype."""
    torch.manual_seed(0)
    X = torch.randn(1, 16, 3, dtype=torch.float64)
    R = _rotation_about_z(1.0, dtype=torch.float64)
    t = torch.tensor([3.0, -2.0, 5.0], dtype=torch.float64)
    X_gt = X @ R.T + t  # an exact rigid image of X, in float64

    aligned = weighted_rigid_align(X, X_gt)
    assert aligned.dtype == torch.float64
    assert torch.allclose(aligned, X, atol=1e-10)


def test_output_is_detached_and_canonicalized():
    """Output is detached and a bare [L, 3] input is promoted to [1, L, 3]."""
    torch.manual_seed(0)
    X = torch.randn(16, 3, requires_grad=True)
    X_gt = (X @ _rotation_about_z(0.7).T).detach()

    aligned = weighted_rigid_align(X, X_gt)
    assert aligned.shape == (1, 16, 3)
    assert not aligned.requires_grad


def test_x_exists_mask_excludes_points_from_fit():
    """Points marked absent must not influence the recovered transform."""
    torch.manual_seed(0)
    L = 16
    X = torch.randn(1, L, 3)
    R = _rotation_about_z(1.0)
    t = torch.tensor([1.0, 2.0, 3.0])
    X_gt = (X @ R.T + t).clone()

    exists = torch.ones(L, dtype=torch.bool)
    exists[-3:] = False
    X_gt[:, ~exists] = 1e3  # corrupt the absent points

    aligned = weighted_rigid_align(X, X_gt, exists)
    # The fit ignored the corrupted points, so the present ones recover exactly.
    assert torch.allclose(aligned[:, exists], X[:, exists], atol=1e-4)


def test_zero_weight_points_excluded_from_fit():
    """Zero-weighted points are excluded from the fit, like an absent-point mask."""
    torch.manual_seed(0)
    L = 16
    X = torch.randn(1, L, 3)
    R = _rotation_about_z(1.0)
    X_gt = (X @ R.T).clone()
    X_gt[:, -3:] = 1e3  # corrupt the points we will zero-weight

    w = torch.ones(1, L)
    w[:, -3:] = 0.0
    aligned = weighted_rigid_align(X, X_gt, w_L=w)
    assert torch.allclose(aligned[:, :-3], X[:, :-3], atol=1e-4)


def test_uniform_weights_match_default():
    """Explicit all-ones weights produce the same result as the default (None)."""
    torch.manual_seed(0)
    X = torch.randn(1, 16, 3)
    X_gt = X @ _rotation_about_z(0.4).T
    default = weighted_rigid_align(X, X_gt)
    ones = weighted_rigid_align(X, X_gt, None, torch.ones(1, 16))
    assert torch.allclose(default, ones, atol=1e-5)


def test_x_exists_must_be_boolean():
    """A non-boolean mask is rejected to avoid a silent mis-alignment."""
    X = torch.randn(1, 8, 3)
    with pytest.raises(AssertionError, match="boolean mask"):
        weighted_rigid_align(X, X, torch.ones(8))


def test_get_rmsd_identical_returns_sqrt_eps():
    """RMSD of identical coordinates is sqrt(eps), not 0 (eps lives under the sqrt)."""
    torch.manual_seed(0)
    X = torch.randn(4, 10, 3)
    rmsd = get_rmsd(X, X)
    assert torch.allclose(rmsd, torch.full_like(rmsd, 0.01))  # sqrt(1e-4)


def test_get_rmsd_constant_offset():
    """A constant per-atom offset v gives RMSD sqrt(|v|^2 + eps)."""
    torch.manual_seed(0)
    X = torch.randn(2, 10, 3)
    v = torch.tensor([1.0, 2.0, 2.0])  # |v|^2 = 9
    rmsd = get_rmsd(X, X + v)
    expected = torch.full_like(rmsd, (9.0 + 1e-4) ** 0.5)
    assert torch.allclose(rmsd, expected, atol=1e-4)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
