"""Unit tests for foundry.utils.rigid.

`Rotation` and `Rigid` are the OpenFold-derived SE(3) frame primitives used across
the models for backbone/atom frames and IPA-style structure updates. Their contracts
(quaternion <-> rotation-matrix round-trips, composition order, inversion, and the
homogeneous/7-vector tensor encodings) are not obvious from the signatures, so the
tests below pin them on small CPU inputs.

Both classes force float32 internally, so reference values are computed in float32 and
compared with a loose tolerance (the quaternion path goes through torch.linalg.eigh).
"""

import pytest
import torch

from foundry.utils.rigid import (
    Rigid,
    Rotation,
    identity_quats,
    identity_rot_mats,
    identity_trans,
    invert_quat,
    invert_rot_mat,
    quat_multiply,
    quat_to_rot,
    rot_matmul,
    rot_to_quat,
    rot_vec_mul,
)

ATOL = 1e-4


def _rodrigues(axis: torch.Tensor, angle: float) -> torch.Tensor:
    """Reference proper rotation about `axis` by `angle` (radians), independent of rigid.py."""
    u = axis / torch.linalg.norm(axis)
    a = torch.tensor(angle, dtype=torch.float32)
    K = torch.tensor(
        [
            [0.0, -u[2], u[1]],
            [u[2], 0.0, -u[0]],
            [-u[1], u[0], 0.0],
        ]
    )
    eye = torch.eye(3)
    return eye + torch.sin(a) * K + (1 - torch.cos(a)) * (K @ K)


def _axis_angle_quat(axis: torch.Tensor, angle: float) -> torch.Tensor:
    """Unit quaternion (w, x, y, z) for a rotation about `axis` by `angle` (radians)."""
    u = axis / torch.linalg.norm(axis)
    half = torch.tensor(angle / 2.0, dtype=torch.float32)
    return torch.cat([torch.cos(half).reshape(1), torch.sin(half) * u])


def _random_rotations(n: int, seed: int) -> torch.Tensor:
    """`n` random proper rotation matrices [n, 3, 3] via QR with a determinant fix-up."""
    torch.manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(n, 3, 3))
    # Multiplying one column by det(q) = +-1 forces det -> +1 (a proper rotation).
    det = torch.linalg.det(q)
    q[..., -1] = q[..., -1] * det.unsqueeze(-1)
    return q


# --- module-level helpers --------------------------------------------------------------


def test_rot_matmul_matches_matrix_product():
    a = _random_rotations(4, 0)
    b = _random_rotations(4, 1)
    assert torch.allclose(rot_matmul(a, b), a @ b, atol=ATOL)


def test_rot_vec_mul_matches_matvec():
    r = _random_rotations(5, 2)
    torch.manual_seed(3)
    v = torch.randn(5, 3)
    expected = torch.einsum("...ij,...j->...i", r, v)
    assert torch.allclose(rot_vec_mul(r, v), expected, atol=ATOL)


def test_quat_to_rot_identity_quat_is_identity_matrix():
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
    assert torch.allclose(quat_to_rot(quat), torch.eye(3), atol=ATOL)


def test_quat_to_rot_matches_axis_angle():
    axis = torch.tensor([0.3, -0.7, 0.5])
    angle = 1.1
    quat = _axis_angle_quat(axis, angle)
    assert torch.allclose(quat_to_rot(quat), _rodrigues(axis, angle), atol=ATOL)


def test_rot_to_quat_roundtrips_through_quat_to_rot():
    r = _random_rotations(6, 4)
    recovered = quat_to_rot(rot_to_quat(r))
    assert torch.allclose(recovered, r, atol=ATOL)


def test_rot_to_quat_rejects_bad_shape():
    with pytest.raises(ValueError):
        rot_to_quat(torch.randn(3, 2))


def test_quat_multiply_composes_rotations():
    q1 = _axis_angle_quat(torch.tensor([0.0, 0.0, 1.0]), 0.6)
    q2 = _axis_angle_quat(torch.tensor([1.0, 0.0, 0.0]), -0.9)
    composed = quat_to_rot(quat_multiply(q1, q2))
    assert torch.allclose(composed, quat_to_rot(q1) @ quat_to_rot(q2), atol=ATOL)


def test_invert_rot_mat_is_transpose_and_true_inverse():
    r = _random_rotations(3, 5)
    assert torch.allclose(invert_rot_mat(r), r.transpose(-1, -2), atol=ATOL)
    prod = rot_matmul(r, invert_rot_mat(r))
    assert torch.allclose(prod, torch.eye(3).expand(3, 3, 3), atol=ATOL)


def test_invert_quat_yields_identity_quaternion():
    q = _axis_angle_quat(torch.tensor([0.2, 0.5, -0.4]), 0.8)
    prod = quat_multiply(q, invert_quat(q))
    assert torch.allclose(prod, torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=ATOL)


def test_identity_helpers():
    assert torch.allclose(identity_rot_mats((2,)), torch.eye(3).expand(2, 3, 3))
    quats = identity_quats((2,))
    assert torch.allclose(quats, torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(2, 4))
    assert torch.count_nonzero(identity_trans((2,))) == 0


# --- Rotation --------------------------------------------------------------------------


def test_rotation_quat_and_matrix_formats_agree():
    quat = _axis_angle_quat(torch.tensor([0.1, 0.2, 0.9]), 1.3)
    from_quat = Rotation(quats=quat)
    from_mat = Rotation(rot_mats=quat_to_rot(quat))
    # get_rot_mats() of the quat-format object matches the matrix it encodes...
    assert torch.allclose(from_quat.get_rot_mats(), from_mat.get_rot_mats(), atol=ATOL)
    # ...and get_quats() of the matrix-format object round-trips back to the same rotation.
    assert torch.allclose(
        quat_to_rot(from_mat.get_quats()), from_mat.get_rot_mats(), atol=ATOL
    )


def test_rotation_requires_exactly_one_input():
    with pytest.raises(ValueError):
        Rotation()
    with pytest.raises(ValueError):
        Rotation(rot_mats=torch.eye(3), quats=torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_rotation_apply_and_invert_apply_roundtrip():
    rot = Rotation(rot_mats=_random_rotations(4, 6))
    torch.manual_seed(7)
    pts = torch.randn(4, 3)
    assert torch.allclose(
        rot.apply(pts), rot_vec_mul(rot.get_rot_mats(), pts), atol=ATOL
    )
    assert torch.allclose(rot.invert_apply(rot.apply(pts)), pts, atol=ATOL)


def test_rotation_compose_r_matches_sequential_apply():
    r1 = Rotation(rot_mats=_random_rotations(3, 8))
    r2 = Rotation(rot_mats=_random_rotations(3, 9))
    torch.manual_seed(10)
    pts = torch.randn(3, 3)
    composed = r1.compose_r(r2).apply(pts)
    assert torch.allclose(composed, r1.apply(r2.apply(pts)), atol=ATOL)


def test_rotation_compose_q_matches_compose_r():
    q1 = Rotation(quats=_axis_angle_quat(torch.tensor([0.0, 1.0, 0.0]), 0.7))
    q2 = Rotation(quats=_axis_angle_quat(torch.tensor([1.0, 0.0, 1.0]), -0.5))
    via_q = q1.compose_q(q2).get_rot_mats()
    via_r = q1.compose_r(q2).get_rot_mats()
    assert torch.allclose(via_q, via_r, atol=ATOL)


def test_rotation_invert_matrix_and_quat_formats():
    torch.manual_seed(11)
    pts = torch.randn(2, 3)
    rot_mat = Rotation(rot_mats=_random_rotations(2, 12))
    assert torch.allclose(rot_mat.invert().apply(rot_mat.apply(pts)), pts, atol=ATOL)
    rot_quat = Rotation(quats=_axis_angle_quat(torch.tensor([0.4, 0.4, 0.8]), 1.0))
    single = torch.randn(3)
    assert torch.allclose(
        rot_quat.invert().apply(rot_quat.apply(single)), single, atol=ATOL
    )


@pytest.mark.parametrize("fmt", ["quat", "rot_mat"])
def test_rotation_identity_apply_is_noop(fmt):
    rot = Rotation.identity((5,), fmt=fmt)
    torch.manual_seed(13)
    pts = torch.randn(5, 3)
    assert torch.allclose(rot.apply(pts), pts, atol=ATOL)


def test_rotation_getitem_slices_virtual_shape():
    mats = _random_rotations(4, 14)
    rot = Rotation(rot_mats=mats)
    sliced = rot[1:3]
    assert sliced.shape == (2,)
    assert torch.allclose(sliced.get_rot_mats(), mats[1:3], atol=ATOL)


def test_rotation_mul_by_mask_zeroes_entries():
    mats = _random_rotations(3, 15)
    rot = Rotation(rot_mats=mats)
    mask = torch.tensor([1.0, 0.0, 1.0])
    masked = (rot * mask).get_rot_mats()
    assert torch.allclose(masked, mats * mask[..., None, None], atol=ATOL)


def test_get_rotvec_returns_axis_times_angle():
    axis = torch.tensor([0.0, 0.0, 1.0])
    angle = 1.2
    rot = Rotation(quats=_axis_angle_quat(axis, angle))
    assert torch.allclose(rot.get_rotvec(), axis * angle, atol=1e-3)
    identity = Rotation.identity((), fmt="quat")
    assert torch.allclose(identity.get_rotvec(), torch.zeros(3), atol=ATOL)


# --- Rigid -----------------------------------------------------------------------------


def test_rigid_apply_is_rotate_then_translate():
    rot = Rotation(rot_mats=_random_rotations(4, 16))
    trans = torch.randn(4, 3)
    rigid = Rigid(rot, trans)
    torch.manual_seed(17)
    pts = torch.randn(4, 3)
    expected = rot.apply(pts) + trans
    assert torch.allclose(rigid.apply(pts), expected, atol=ATOL)


def test_rigid_invert_apply_roundtrip():
    rigid = Rigid(Rotation(rot_mats=_random_rotations(3, 18)), torch.randn(3, 3))
    torch.manual_seed(19)
    pts = torch.randn(3, 3)
    assert torch.allclose(rigid.invert_apply(rigid.apply(pts)), pts, atol=ATOL)


def test_rigid_compose_matches_sequential_apply():
    t1 = Rigid(Rotation(rot_mats=_random_rotations(2, 20)), torch.randn(2, 3))
    t2 = Rigid(Rotation(rot_mats=_random_rotations(2, 21)), torch.randn(2, 3))
    torch.manual_seed(22)
    pts = torch.randn(2, 3)
    assert torch.allclose(t1.compose(t2).apply(pts), t1.apply(t2.apply(pts)), atol=ATOL)


def test_rigid_invert_inverts_apply():
    rigid = Rigid(Rotation(rot_mats=_random_rotations(2, 23)), torch.randn(2, 3))
    torch.manual_seed(24)
    pts = torch.randn(2, 3)
    assert torch.allclose(rigid.invert().apply(rigid.apply(pts)), pts, atol=ATOL)


def test_rigid_identity_apply_is_noop():
    rigid = Rigid.identity((4,))
    torch.manual_seed(25)
    pts = torch.randn(4, 3)
    assert torch.allclose(rigid.apply(pts), pts, atol=ATOL)


def test_rigid_to_tensor_4x4_structure_and_roundtrip():
    rot = Rotation(rot_mats=_random_rotations(3, 26))
    trans = torch.randn(3, 3)
    rigid = Rigid(rot, trans)
    t = rigid.to_tensor_4x4()
    assert t.shape == (3, 4, 4)
    assert torch.allclose(t[..., :3, :3], rot.get_rot_mats(), atol=ATOL)
    assert torch.allclose(t[..., :3, 3], trans, atol=ATOL)
    assert torch.allclose(t[..., 3, 3], torch.ones(3), atol=ATOL)
    torch.manual_seed(27)
    pts = torch.randn(3, 3)
    rebuilt = Rigid.from_tensor_4x4(t)
    assert torch.allclose(rebuilt.apply(pts), rigid.apply(pts), atol=ATOL)


def test_rigid_from_tensor_4x4_rejects_bad_shape():
    with pytest.raises(ValueError):
        Rigid.from_tensor_4x4(torch.randn(3, 3))


def test_rigid_to_from_tensor_7_roundtrip():
    rot = Rotation(quats=_axis_angle_quat(torch.tensor([0.3, 0.6, 0.2]), 0.9))
    trans = torch.randn(3)
    rigid = Rigid(rot, trans)
    t = rigid.to_tensor_7()
    assert t.shape == (7,)
    rebuilt = Rigid.from_tensor_7(t)
    torch.manual_seed(28)
    pts = torch.randn(3)
    assert torch.allclose(rebuilt.apply(pts), rigid.apply(pts), atol=ATOL)


def test_rigid_from_tensor_7_rejects_bad_shape():
    with pytest.raises(ValueError):
        Rigid.from_tensor_7(torch.randn(6))


def test_rigid_compose_q_update_vec_zero_update_is_noop():
    rigid = Rigid(
        Rotation(quats=_axis_angle_quat(torch.tensor([0.1, 0.2, 0.3]), 0.5)),
        torch.randn(3),
    )
    updated = rigid.compose_q_update_vec(torch.zeros(6))
    torch.manual_seed(29)
    pts = torch.randn(3)
    assert torch.allclose(updated.apply(pts), rigid.apply(pts), atol=ATOL)


def test_rigid_from_3_points_builds_orthonormal_frame():
    torch.manual_seed(30)
    p_neg_x = torch.randn(3)
    origin = torch.randn(3)
    p_xy = torch.randn(3)
    rigid = Rigid.from_3_points(p_neg_x, origin, p_xy)
    rot = rigid.get_rots().get_rot_mats()
    # Proper orthonormal rotation.
    assert torch.allclose(rot @ rot.transpose(-1, -2), torch.eye(3), atol=ATOL)
    assert torch.allclose(torch.linalg.det(rot), torch.tensor(1.0), atol=ATOL)
    # The origin maps to the frame origin, and p_neg_x lies on the frame's negative x-axis.
    assert torch.allclose(rigid.invert_apply(origin), torch.zeros(3), atol=ATOL)
    local_neg_x = rigid.invert_apply(p_neg_x)
    assert local_neg_x[0] < 0
    assert torch.allclose(local_neg_x[1:], torch.zeros(2), atol=ATOL)


def test_rigid_cat_and_unsqueeze_shapes():
    a = Rigid(Rotation(rot_mats=_random_rotations(3, 31)), torch.randn(3, 3))
    b = Rigid(Rotation(rot_mats=_random_rotations(3, 32)), torch.randn(3, 3))
    cat = Rigid.cat([a, b], dim=0)
    assert cat.shape == (6,)
    assert torch.allclose(cat[:3].get_trans(), a.get_trans(), atol=ATOL)
    assert a.unsqueeze(0).shape == (1, 3)
