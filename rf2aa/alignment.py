import logging

import numpy as np
import torch
from rf2aa.debug import pretty_describe_dict

logger = logging.getLogger(__name__)

def weighted_rigid_align(
        X_L, # [B, L, 3]
        X_gt_L, # [B, L, 3]
        w_L, # [B, L]
    ):
    '''
    Weighted rigid body alignment of X_L onto X_gt_L
    Returns:
      X_align_L: [B, L, 3]
    '''
    assert X_L.shape == X_gt_L.shape
    assert X_L.shape[:-1] == w_L.shape

    u_X = torch.mean(X_L * w_L.unsqueeze(-1), dim=-2) / torch.mean(w_L, dim=-1, keepdim=True)
    u_X_gt = torch.mean(X_gt_L * w_L.unsqueeze(-1), dim=-2) / torch.mean(w_L, dim=-1, keepdim=True)

    X_L = X_L - u_X.unsqueeze(-2)
    X_gt_L = X_gt_L - u_X_gt.unsqueeze(-2)
    
    # Computation of the covariance matrix
    C = torch.transpose(X_gt_L, -1, -2) @ X_L

    U, S, V = torch.linalg.svd(C)

    R = U @ V
    B, _, _ = X_L.shape
    F = torch.eye(3,3, device=X_L.device)[None].tile((B,1,1,))

    F[...,-1, -1] = torch.sign(torch.linalg.det(R))
    R = U @ F @ V

    X_align_L = X_L @ R.transpose(-1, -2) + u_X_gt.unsqueeze(-2)

    return X_align_L.detach()

def get_rmsd(xyz1, xyz2, eps=1e-4):
    L = xyz1.shape[-2]
    rmsd = torch.sqrt(torch.sum((xyz2-xyz1)*(xyz2-xyz1), axis=(-1, -2)) / L + eps)
    return rmsd
