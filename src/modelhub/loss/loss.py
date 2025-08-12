import logging

import torch

logger = logging.getLogger(__name__)


def calc_ddihedralmse_dxyz(a, b, c, d, true_dih, eps=1e-6):
    """
    Calculates the gradient of the dihedral angle with respect to the xyz coordinates using the closed form derivative.
    a, b, c, and d are atoms participating in the chiral center. true_dih is the true dihedral angle.

    Unlike the original implementation, this does NOT use autograd.
    """
    # I need to reshape this from n_symm, batch, n, 3 to n_symm * batch * n, 3)
    og_shape = a.shape
    # Expand the dihedral by the batch dimension to match n_atoms*batchs
    true_dih = true_dih.unsqueeze(0).repeat(a.shape[0], 1)
    a = a.view(-1, 3)
    b = b.view(-1, 3)
    c = c.view(-1, 3)
    d = d.view(-1, 3)
    true_dih = true_dih.view(-1)

    batch_size = a.shape[0]  # Support for batch size
    I = (
        torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(a.device)
    )  # Make batch-aware identity matrix

    # Compute b0, b1, b2
    b0 = a - b
    b1 = c - b
    b2 = d - c

    # Normalize b1
    b1_norm = torch.norm(b1, dim=-1, keepdim=True)
    b1n = b1 / (b1_norm + eps)

    # Compute orthogonal components v and w
    v = b0 - torch.sum(b0 * b1n, dim=-1, keepdim=True) * b1n
    w = b2 - torch.sum(b2 * b1n, dim=-1, keepdim=True) * b1n

    # Dihedral components x and y
    x = torch.sum(v * w, dim=-1)
    y = torch.sum(torch.cross(b1n, v, dim=-1) * w, dim=-1)

    # Dihedral angle
    dih = torch.atan2(y + eps, x + eps)

    # Compute MSE loss and manual gradients
    # mse_loss = torch.mean(torch.square(dih - true_dih))
    # mse_loss = torch.sum(torch.square(dih - true_dih))

    # Define matrices and gradients, adapted for batch
    db0_db = -I
    db1_db = -I
    db1_dc = I
    db2_dc = -I
    db0_da = I
    db2_dd = I
    # dmse_ddih = 2 * (dih - true_dih) / batch_size
    dmse_ddih = 2 * (dih - true_dih)
    ddih_dx = -y / (x**2 + y**2 + eps)
    ddih_dy = x / (x**2 + y**2 + eps)
    dy_dv = -torch.cross(b1n, w, dim=-1)
    dy_dw = torch.cross(b1n, v, dim=-1)
    dx_dv = w
    dx_dw = v

    dw_db1n = -torch.sum(b2 * b1n, dim=-1, keepdim=True).unsqueeze(-1) * I - torch.bmm(
        b2.unsqueeze(-1), b1n.unsqueeze(1)
    )

    db1n_db1 = (b1_norm + eps).unsqueeze(-1) * I / (b1_norm**2 + eps).unsqueeze(
        -1
    ) - torch.bmm(b1.unsqueeze(-1), b1.unsqueeze(1)) / (b1_norm**2 + eps).unsqueeze(-1)

    dv_db1n = -torch.sum(b0 * b1n, dim=-1, keepdim=True).unsqueeze(-1) * I - torch.bmm(
        b0.unsqueeze(-1), b1n.unsqueeze(1)
    )
    dv_db0 = I - torch.bmm(b1n.unsqueeze(-1), b1n.unsqueeze(1))
    dw_db2 = I - torch.bmm(b1n.unsqueeze(-1), b1n.unsqueeze(1))

    # Adjust sizes now for efficiency
    ddih_dx = ddih_dx.view(-1, 1, 1)
    ddih_dy = ddih_dy.view(-1, 1, 1)
    dmse_ddih = dmse_ddih.view(-1, 1, 1)
    dx_dv = dx_dv.unsqueeze(1)
    dx_dw = dx_dw.unsqueeze(1)
    dy_dv = dy_dv.unsqueeze(1)
    dy_dw = dy_dw.unsqueeze(1)

    # Gradient computations
    # wrt a
    dv_da = torch.matmul(dv_db0, db0_da)
    ddih_da = torch.bmm((ddih_dx * dx_dv), dv_da) + torch.bmm((ddih_dy * dy_dv), dv_da)
    dmse_da = torch.bmm(dmse_ddih, ddih_da)

    # wrt b
    db1n_db = torch.matmul(db1n_db1, db1_db)
    dv_db = torch.matmul(dv_db0, db0_db) + torch.matmul(
        dv_db1n.transpose(-1, -2), db1n_db
    )
    dw_db = torch.matmul(dw_db1n.transpose(-1, -2), db1n_db)
    dx_db = torch.bmm(dx_dv, dv_db) + torch.bmm(dx_dw, dw_db)
    dy_db = torch.bmm(dy_dv, dv_db) + torch.bmm(dy_dw, dw_db)
    ddih_db = torch.bmm(ddih_dx, dx_db) + torch.bmm(ddih_dy, dy_db)
    dmse_db = torch.bmm(dmse_ddih, ddih_db)

    # wrt c
    db1n_dc = torch.matmul(db1n_db1, db1_dc)
    dv_dc = torch.matmul(dv_db1n.transpose(-1, -2), db1n_dc)
    dw_dc = torch.matmul(dw_db2, db2_dc) + torch.matmul(
        dw_db1n.transpose(-1, -2), db1n_dc
    )
    dx_dc = torch.bmm(dx_dv, dv_dc) + torch.bmm(dx_dw, dw_dc)
    dy_dc = torch.bmm(dy_dv, dv_dc) + torch.bmm(dy_dw, dw_dc)
    ddih_dc = torch.bmm(ddih_dx, dx_dc) + torch.bmm(ddih_dy, dy_dc)
    dmse_dc = torch.bmm(dmse_ddih, ddih_dc)

    # wrt d
    dw_dd = torch.matmul(dw_db2, db2_dd)
    ddih_dd = torch.bmm((ddih_dx * dx_dw), dw_dd) + torch.bmm((ddih_dy * dy_dw), dw_dd)
    dmse_dd = torch.bmm(dmse_ddih, ddih_dd)

    # Reshape gradients back to original shape and prep for cat
    dmse_da = dmse_da.view(og_shape).unsqueeze(-2)
    dmse_db = dmse_db.view(og_shape).unsqueeze(-2)
    dmse_dc = dmse_dc.view(og_shape).unsqueeze(-2)
    dmse_dd = dmse_dd.view(og_shape).unsqueeze(-2)

    grads = torch.cat([dmse_da, dmse_db, dmse_dc, dmse_dd], dim=-2)
    return grads


def calc_chiral_grads_flat_impl(
    xyz, chiral_centers, chiral_center_dihedral_angles, no_grad_on_chiral_center
):
    """
    Calculates the gradient of the chiral centers with respect to the xyz coordinates using the closed form derivative.
    Args:
    xyz: torch.Tensor, shape (batch, n_atoms, 3)
    chiral_centers: torch.Tensor, shape (long) (n_centers, 4)
    chiral_center_dihedral_angles: torch.Tensor, shape (float) (n_centers, 1)

    Returns:
    grads: torch.Tensor, shape (batch, n_atoms, 3)
    """
    # (We want to track the gradient of the dihedral angle loss with respect to the xyz coordinates)
    xyz.requires_grad_(True)

    # Edge case: No chiral centers, return zero gradients
    if chiral_centers.shape[0] == 0:
        return torch.zeros(xyz.shape, device=xyz.device)

    # Get the coordinates of the four atoms that make up the chiral center
    chiral_dih = xyz[:, chiral_centers, :]

    # Calculate the gradient of the dihedral angle loss with respect to the xyz coordinates
    grads = torch.zeros_like(xyz).to(xyz.device)
    chiral_grads = calc_ddihedralmse_dxyz(
        chiral_dih[..., 0, :],
        chiral_dih[..., 1, :],
        chiral_dih[..., 2, :],
        chiral_dih[..., 3, :],
        chiral_center_dihedral_angles,
    )  # n_center, 4, 3

    if no_grad_on_chiral_center:
        chiral_grads[:, :, 0] = 0.0  # no gradient on chiral center

    # back to atom
    grads.index_add_(
        1,
        chiral_centers.flatten(),
        chiral_grads.flatten(start_dim=1, end_dim=2),
    )

    return grads
