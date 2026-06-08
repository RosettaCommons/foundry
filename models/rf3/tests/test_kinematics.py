"""Unit tests for rf3.kinematics.

Most of this RF2AA-derived module is unused (its header flags it for deletion);
`get_dih` is the live one — it computes the dihedral angle around the b->c axis
for quadruples (a, b, c, d) and feeds RF3's chirality metric. The non-obvious
part is the sign and range (atan2-based, so in (-pi, pi]); it is pinned here on a
canonical setup where the dihedral equals a known rotation angle about b->c.
"""

import math

import torch
from rf3.kinematics import get_dih


def _dih_for_angle(theta: float) -> torch.Tensor:
    """Dihedral of (a, b, c, d) where d is rotated by `theta` about the b->c axis.

    With b at the origin, c on +x and a on +y, the projections of b->a and c->d
    onto the plane normal to b->c sit at 0 and `theta`, so the dihedral is exactly
    `theta`.
    """
    a = torch.tensor([[[0.0, 1.0, 0.0]]])
    b = torch.tensor([[[0.0, 0.0, 0.0]]])
    c = torch.tensor([[[1.0, 0.0, 0.0]]])
    d = torch.tensor([[[1.0, math.cos(theta), math.sin(theta)]]])
    return get_dih(a, b, c, d)


def test_get_dih_cis_is_zero():
    assert torch.allclose(_dih_for_angle(0.0), torch.zeros(1, 1), atol=1e-3)


def test_get_dih_quarter_turn():
    expected = torch.full((1, 1), math.pi / 2)
    assert torch.allclose(_dih_for_angle(math.pi / 2), expected, atol=1e-3)


def test_get_dih_negative_quarter_turn():
    expected = torch.full((1, 1), -math.pi / 2)
    assert torch.allclose(_dih_for_angle(-math.pi / 2), expected, atol=1e-3)


def test_get_dih_trans_is_pi():
    expected = torch.full((1, 1), math.pi)
    assert torch.allclose(_dih_for_angle(math.pi), expected, atol=1e-3)


def test_get_dih_preserves_batch_shape():
    torch.manual_seed(0)
    a, b, c, d = (torch.randn(2, 5, 3) for _ in range(4))
    assert get_dih(a, b, c, d).shape == (2, 5)


if __name__ == "__main__":
    import pytest

    pytest.main(["-v", __file__])
