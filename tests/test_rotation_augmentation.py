"""Unit tests for foundry.utils.rotation_augmentation.

These helpers apply the random SE(3) augmentation used during rf3 sampling.
The contracts worth pinning are geometric: `uniform_random_rotation` must emit
proper rotations, `centre` removes the global centroid of the present atoms and
zeros the absent ones, and the augmentation is rigid (distance-preserving).
Inputs follow the production shapes: coordinates are [D, L, 3] and the
existence mask is [D, L].
"""

import pytest
import torch

from foundry.utils.rotation_augmentation import (
    centre,
    centre_random_augmentation,
    get_random_augmentation,
    uniform_random_rotation,
)


def _pairwise_distances(x: torch.Tensor) -> torch.Tensor:
    """Per-batch pairwise distance matrices for [D, L, 3] coordinates."""
    return torch.cdist(x, x)


def test_uniform_random_rotation_shape():
    rotations = uniform_random_rotation((5,))
    assert rotations.shape == (5, 3, 3)


def test_uniform_random_rotation_is_proper_rotation():
    """Sampled matrices are orthogonal with determinant +1 (no reflections)."""
    torch.manual_seed(0)
    n = 8
    rotations = uniform_random_rotation((n,))
    identity = torch.eye(3).expand(n, 3, 3)
    assert torch.allclose(rotations @ rotations.transpose(-1, -2), identity, atol=1e-5)
    assert torch.allclose(torch.linalg.det(rotations), torch.ones(n), atol=1e-5)


def test_centre_removes_global_centroid_of_present_atoms():
    """Present atoms are shifted by the centroid taken over all present atoms."""
    torch.manual_seed(0)
    X = torch.randn(2, 12, 3)
    mask = torch.ones(2, 12, dtype=torch.bool)
    mask[1, 0] = False  # one absent atom

    centred = centre(X, mask)
    expected_present = X[mask] - X[mask].mean(dim=0)
    assert torch.allclose(centred[mask], expected_present, atol=1e-6)
    assert torch.allclose(centred[mask].mean(dim=0), torch.zeros(3), atol=1e-5)


def test_centre_zeros_absent_atoms():
    torch.manual_seed(0)
    X = torch.randn(2, 12, 3)
    mask = torch.ones(2, 12, dtype=torch.bool)
    mask[0, 3:6] = False

    centred = centre(X, mask)
    assert torch.all(centred[~mask] == 0.0)


def test_centre_does_not_mutate_input():
    """centre clones, so the caller's tensor is left untouched."""
    torch.manual_seed(0)
    X = torch.randn(2, 12, 3)
    original = X.clone()
    centre(X, torch.ones(2, 12, dtype=torch.bool))
    assert torch.equal(X, original)


def test_get_random_augmentation_preserves_distances():
    """Augmentation is rigid: intra-structure distances are unchanged."""
    torch.manual_seed(0)
    X = torch.randn(3, 12, 3)
    augmented = get_random_augmentation(X, s_trans=2.0)
    assert augmented.shape == X.shape
    assert torch.allclose(
        _pairwise_distances(X), _pairwise_distances(augmented), atol=1e-4
    )


def test_get_random_augmentation_zero_translation_keeps_centroid_rotating():
    """With s_trans=0 the centroid only rotates, so its distance to origin holds."""
    torch.manual_seed(0)
    X = torch.randn(3, 12, 3)
    augmented = get_random_augmentation(X, s_trans=0.0)
    before = X.mean(dim=1).norm(dim=-1)
    after = augmented.mean(dim=1).norm(dim=-1)
    assert torch.allclose(before, after, atol=1e-4)


def test_centre_random_augmentation_preserves_present_distances():
    """The composed centre+augment step stays rigid over the present atoms."""
    torch.manual_seed(0)
    X = torch.randn(3, 12, 3)
    mask = torch.ones(3, 12, dtype=torch.bool)

    result = centre_random_augmentation(X, mask, s_trans=1.0)
    assert result.shape == X.shape
    # centre then a rigid transform preserves all pairwise distances.
    centred = centre(X, mask)
    assert torch.allclose(
        _pairwise_distances(centred), _pairwise_distances(result), atol=1e-4
    )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
