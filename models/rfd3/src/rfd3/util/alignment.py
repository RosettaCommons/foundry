import logging

import torch

logger = logging.getLogger(__name__)


def weighted_rigid_align(
    X_L,  # [B, L, 3]
    X_gt_L,  # [B, L, 3]
    X_exists_L,  # [L]
    w_L,  # [B, L]
):
    """
    Weighted rigid body alignment of X_gt_L onto X_L with weights w_L
    Allows for "moving target" ground truth that is se3 invariant
    Following algorithm 28 in AF3 paper
    Returns:
      X_align_L: [B, L, 3]
    """
    assert X_L.shape == X_gt_L.shape
    assert X_L.shape[:-1] == w_L.shape

    # Assert `X_exists_L` is a boolean mask
    assert (
        X_exists_L.dtype == torch.bool
    ), "X_exists_L should be a boolean mask! Otherwise, the alignment will be incorrect (silent failure)!"

    X_resolved = X_L[:, X_exists_L]
    X_gt_resolved = X_gt_L[:, X_exists_L]
    w_resolved = w_L[:, X_exists_L]
    u_X = torch.sum(X_resolved * w_resolved.unsqueeze(-1), dim=-2) / torch.sum(
        w_resolved, dim=-1, keepdim=True
    )
    u_X_gt = torch.sum(X_gt_resolved * w_resolved.unsqueeze(-1), dim=-2) / torch.sum(
        w_resolved, dim=-1, keepdim=True
    )

    X_resolved = X_resolved - u_X.unsqueeze(-2)
    X_gt_resolved = X_gt_resolved - u_X_gt.unsqueeze(-2)

    # Computation of the covariance matrix
    C = torch.einsum("bji,bjk->bik", w_resolved[..., None] * X_gt_resolved, X_resolved)

    U, S, V = torch.linalg.svd(C)

    R = U @ V
    B, _, _ = X_L.shape
    F = torch.eye(3, 3, device=X_L.device)[None].tile(
        (
            B,
            1,
            1,
        )
    )

    F[..., -1, -1] = torch.sign(torch.linalg.det(R))
    R = U @ F @ V

    X_gt_L = X_gt_L - u_X_gt.unsqueeze(-2)
    X_align_L = X_gt_L @ R + u_X.unsqueeze(-2)

    return X_align_L.detach()


def get_rmsd(xyz1, xyz2, eps=1e-4):
    L = xyz1.shape[-2]
    rmsd = torch.sqrt(torch.sum((xyz2 - xyz1) * (xyz2 - xyz1), axis=(-1, -2)) / L + eps)
    return rmsd


def superimpose(xyz1, xyz2, mask, eps=1e-4):
    """
    Superimpose xyz1 onto xyz2 using mask
    """
    L = xyz1.shape[-2]
    assert mask.shape == (L,)
    assert xyz1.shape == xyz2.shape
    assert mask.dtype == torch.bool
