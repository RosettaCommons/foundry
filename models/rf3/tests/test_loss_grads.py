"""Unit tests for the closed-form chiral/dihedral gradients in rf3.loss.loss.

``calc_ddihedralmse_dxyz`` returns the analytic gradient of the summed dihedral
loss ``sum_i (dihedral_i - true_dih_i)**2`` with respect to the four atoms
``a, b, c, d`` of each dihedral — a hand-derived replacement for autograd. Inputs
are ``(leading, K, 3)`` with one ``true_dih`` entry per dihedral ``K``; the leading
dim is broadcast over. ``calc_chiral_grads_flat_impl`` evaluates that gradient for
a set of chiral centres (each four atom indices into ``xyz``) and scatters the
per-centre gradients back onto the full atom tensor with ``index_add_`` — so atoms
shared between centres accumulate, and the optional ``no_grad_on_chiral_center``
flag drops the gradient on each centre's first atom.

All tests use float32, the production coordinate dtype: the closed form builds an
``eye(3)`` without an explicit dtype and raises on float64 inputs.
"""

import torch
from rf3.loss.loss import calc_chiral_grads_flat_impl, calc_ddihedralmse_dxyz


def _dihedral(a, b, c, d, eps=1e-6):
    """The forward that the closed-form gradient differentiates (same eps as the source)."""
    b0, b1, b2 = a - b, c - b, d - c
    b1n = b1 / (b1.norm(dim=-1, keepdim=True) + eps)
    v = b0 - (b0 * b1n).sum(-1, keepdim=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdim=True) * b1n
    x = (v * w).sum(-1)
    y = (torch.cross(b1n, v, dim=-1) * w).sum(-1)
    return torch.atan2(y + eps, x + eps)


# --- calc_ddihedralmse_dxyz -------------------------------------------------


def test_ddihedralmse_matches_autograd():
    torch.manual_seed(0)
    a, b, c, d = (torch.randn(1, 4, 3, requires_grad=True) for _ in range(4))
    true = torch.randn(4)
    loss = ((_dihedral(a, b, c, d) - true) ** 2).sum()
    ga, gb, gc, gd = torch.autograd.grad(loss, [a, b, c, d])
    expected = torch.stack([ga, gb, gc, gd], dim=-2)  # (1, K, 4 atoms, 3 coords)
    grads = calc_ddihedralmse_dxyz(a.detach(), b.detach(), c.detach(), d.detach(), true)
    torch.testing.assert_close(grads, expected, atol=1e-3, rtol=1e-3)


def test_ddihedralmse_zero_gradient_at_truth():
    torch.manual_seed(1)
    a, b, c, d = (torch.randn(1, 3, 3) for _ in range(4))
    true = _dihedral(a, b, c, d).reshape(3)  # the truth is the actual dihedral
    grads = calc_ddihedralmse_dxyz(a, b, c, d, true)
    # dmse/ddih = 2*(dih - true) is exactly 0, so every coordinate gradient is 0.
    assert torch.all(grads == 0.0)


def test_ddihedralmse_preserves_leading_shape():
    a, b, c, d = (torch.randn(2, 5, 3) for _ in range(4))
    grads = calc_ddihedralmse_dxyz(a, b, c, d, torch.randn(5))
    assert grads.shape == (2, 5, 4, 3)  # leading dims + (4 atoms, 3 coords)


# --- calc_chiral_grads_flat_impl --------------------------------------------


def test_chiral_grads_empty_centers_returns_zeros():
    xyz = torch.randn(1, 7, 3)
    grads = calc_chiral_grads_flat_impl(
        xyz, torch.zeros(0, 4, dtype=torch.long), torch.zeros(0), False
    )
    assert grads.shape == xyz.shape
    assert torch.all(grads == 0.0)


def test_chiral_grads_only_center_atoms_receive_gradient():
    torch.manual_seed(2)
    xyz = torch.randn(1, 7, 3)
    centers = torch.tensor([[1, 3, 4, 5]])
    grads = calc_chiral_grads_flat_impl(xyz, centers, torch.tensor([0.7]), False)
    # Atoms outside the centre stay exactly zero; the four centre atoms get gradient.
    assert torch.all(grads[:, [0, 2, 6]] == 0.0)
    assert grads[:, [1, 3, 4, 5]].abs().sum() > 0


def test_chiral_grads_accumulate_for_shared_atoms():
    torch.manual_seed(3)
    xyz = torch.randn(1, 8, 3)
    c1 = torch.tensor([[0, 1, 2, 3]])
    c2 = torch.tensor([[2, 4, 5, 6]])  # shares atom 2 with c1
    a1, a2 = torch.tensor([0.3]), torch.tensor([1.1])
    both = calc_chiral_grads_flat_impl(
        xyz, torch.cat([c1, c2]), torch.cat([a1, a2]), False
    )
    g1 = calc_chiral_grads_flat_impl(xyz, c1, a1, False)
    g2 = calc_chiral_grads_flat_impl(xyz, c2, a2, False)
    # index_add_ accumulates, so the combined gradient is the per-centre sum —
    # including the shared atom 2, which gets a contribution from both centres.
    torch.testing.assert_close(both, g1 + g2)


def test_chiral_grads_no_grad_on_center_zeroes_first_atom():
    torch.manual_seed(4)
    xyz = torch.randn(1, 7, 3)
    centers = torch.tensor([[1, 3, 4, 5]])  # atom 1 is the chiral centre
    angles = torch.tensor([0.7])
    with_grad = calc_chiral_grads_flat_impl(xyz, centers, angles, False)
    no_grad = calc_chiral_grads_flat_impl(xyz, centers, angles, True)
    # The flag zeroes the gradient on the centre atom (the first of the four)...
    assert with_grad[:, 1].abs().sum() > 0
    assert torch.all(no_grad[:, 1] == 0.0)
    # ...and leaves the other three atoms untouched.
    torch.testing.assert_close(no_grad[:, [3, 4, 5]], with_grad[:, [3, 4, 5]])
