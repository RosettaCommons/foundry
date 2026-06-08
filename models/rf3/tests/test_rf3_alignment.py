"""Unit tests for rf3.alignment.

`weighted_rigid_align` is RF3's SE(3) Kabsch alignment: it rigidly superimposes
the ground-truth coordinates onto the reference before the diffusion loss is
computed, so the loss is invariant to the ground truth's pose. Its contract is
not obvious from the body: it recovers an exact rigid transform, excludes points
the `X_exists_L` mask or zero weights remove from the fit, and returns a detached
tensor (the aligned target is treated as a constant w.r.t. autograd). Unlike the
foundry copy, every argument is required and inputs must already carry a batch
dim. `get_rmsd` keeps `eps` *inside* the sqrt, so identical inputs give
sqrt(eps), not 0. The contracts are pinned here on small float32 CPU inputs,
matching production call sites.
"""

import pytest
import torch
from rf3.alignment import get_rmsd, weighted_rigid_align


def _rotation_about_z(angle: float) -> torch.Tensor:
    """Proper rotation (det +1) about the z-axis, as a [3, 3] float32 matrix."""
    a = torch.tensor(angle)
    c, s = torch.cos(a), torch.sin(a)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _all_present(length: int) -> torch.Tensor:
    return torch.ones(length, dtype=torch.bool)


def _uniform_weights(batch: int, length: int) -> torch.Tensor:
    return torch.ones(batch, length)


def test_identity_alignment_is_noop():
    """Aligning a structure onto itself returns the structure unchanged."""
    torch.manual_seed(0)
    X = torch.randn(1, 16, 3)
    aligned = weighted_rigid_align(X, X, _all_present(16), _uniform_weights(1, 16))
    assert aligned.shape == (1, 16, 3)
    assert torch.allclose(aligned, X, atol=1e-4)


def test_recovers_known_rigid_transform():
    """A rigid image of X aligns back onto X regardless of the rotation/translation."""
    torch.manual_seed(0)
    X = torch.randn(1, 16, 3)
    R = _rotation_about_z(1.0)
    t = torch.tensor([3.0, -2.0, 5.0])
    X_gt = X @ R.T + t  # an exact rigid image of X

    aligned = weighted_rigid_align(X, X_gt, _all_present(16), _uniform_weights(1, 16))
    assert torch.allclose(aligned, X, atol=1e-4)


def test_recovers_pure_translation():
    """Translation-only ground truth aligns back exactly."""
    torch.manual_seed(0)
    X = torch.randn(1, 16, 3)
    X_gt = X + torch.tensor([10.0, -4.0, 0.5])
    aligned = weighted_rigid_align(X, X_gt, _all_present(16), _uniform_weights(1, 16))
    assert torch.allclose(aligned, X, atol=1e-4)


def test_x_exists_mask_excludes_points_from_fit():
    """Points marked absent must not influence the recovered transform."""
    torch.manual_seed(0)
    L = 16
    X = torch.randn(1, L, 3)
    R = _rotation_about_z(1.0)
    t = torch.tensor([1.0, 2.0, 3.0])
    X_gt = (X @ R.T + t).clone()

    exists = _all_present(L)
    exists[-3:] = False
    X_gt[:, ~exists] = 1e3  # corrupt the absent points

    aligned = weighted_rigid_align(X, X_gt, exists, _uniform_weights(1, L))
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

    w = _uniform_weights(1, L)
    w[:, -3:] = 0.0
    aligned = weighted_rigid_align(X, X_gt, _all_present(L), w)
    assert torch.allclose(aligned[:, :-3], X[:, :-3], atol=1e-4)


def test_output_is_detached():
    """The aligned target is detached, so it carries no grad even from a grad input."""
    torch.manual_seed(0)
    X = torch.randn(1, 16, 3, requires_grad=True)
    X_gt = (X @ _rotation_about_z(0.7).T).detach()

    aligned = weighted_rigid_align(X, X_gt, _all_present(16), _uniform_weights(1, 16))
    assert not aligned.requires_grad


def test_x_exists_must_be_boolean():
    """A non-boolean mask is rejected to avoid a silent mis-alignment."""
    X = torch.randn(1, 8, 3)
    with pytest.raises(AssertionError, match="boolean mask"):
        weighted_rigid_align(X, X, torch.ones(8), _uniform_weights(1, 8))


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
