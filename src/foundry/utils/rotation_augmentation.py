import math

import torch

from foundry.utils.rigid import rot_vec_mul


def centre(X_L, X_exists_L):
    X_L = X_L.clone()
    X_L[X_exists_L] = X_L[X_exists_L] - torch.mean(
        X_L[X_exists_L], dim=-2, keepdim=True
    )
    X_L[~X_exists_L] = 0.0
    return X_L


def get_random_augmentation(X_L, s_trans):
    """
    Inputs:
        X_L [D, L, 3]: Batched atom coordinates
        s_trans (float): standard deviation of a global translation to be applied for each
            element in the batch
    """
    D, L, _ = X_L.shape
    R = uniform_random_rotation((D,)).to(X_L.device)
    noise = s_trans * torch.normal(mean=0, std=1, size=(D, 1, 3)).to(X_L.device)
    return rot_vec_mul(R[:, None], X_L) + noise


def centre_random_augmentation(X_L, X_exists_L, s_trans):
    X_L = centre(X_L, X_exists_L)
    return get_random_augmentation(X_L, s_trans)


def uniform_random_rotation(size):
    # Sample random angles for rotations around X, Y, and Z axes
    theta_x = torch.rand(size) * 2 * math.pi
    theta_y = torch.rand(size) * 2 * math.pi
    theta_z = torch.rand(size) * 2 * math.pi

    # Calculate the cosines and sines of the angles
    cos_x = torch.cos(theta_x)
    sin_x = torch.sin(theta_x)
    cos_y = torch.cos(theta_y)
    sin_y = torch.sin(theta_y)
    cos_z = torch.cos(theta_z)
    sin_z = torch.sin(theta_z)

    # Create the rotation matrices around X, Y, and Z axes
    rotation_x = torch.stack(
        [torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]]) for c, s in zip(cos_x, sin_x)]
    )

    rotation_y = torch.stack(
        [torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]]) for c, s in zip(cos_y, sin_y)]
    )

    rotation_z = torch.stack(
        [torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]]) for c, s in zip(cos_z, sin_z)]
    )

    # Combine the rotation matrices
    rotation_matrix = torch.matmul(rotation_z, torch.matmul(rotation_y, rotation_x))

    return rotation_matrix
