from itertools import combinations
from typing import Union

import numpy as np
import torch
from jaxtyping import Bool, Float
from numpy.typing import NDArray


def find_bin_midpoints(
    max_distance: float, num_bins: int, device: Union[str, torch.device] = "cpu"
) -> Float[torch.Tensor, "num_bins"]:
    """
    Find the bin midpoints for a given binning scheme. Used to find expectation of values when converting binned
    predictions to unbinned predictions. Assumes the minimum of the schema is 0.
    Args:
        max_distance: float, maximum distance
        num_bins: int, number of bins
        device: device to run on
    Returns:
        pae_midpoints: [num_bins], bin midpoints
    """
    bin_size = max_distance / num_bins
    bins = torch.linspace(
        bin_size, max_distance - bin_size, num_bins - 1, device=device
    )
    midpoints = (bins[1:] + bins[:-1]) / 2
    midpoints = torch.cat(
        [(bins[0] - bin_size / 2)[None], midpoints, bins[-1:] + bin_size / 2]
    )

    return midpoints


def unbin_logits(
    logits: Float[torch.Tensor, "B num_bins L X"], max_distance: float, num_bins: int
) -> Float[torch.Tensor, "B L L"]:
    """
    Unbin the logits to get the matrix
    Args:
        logits: [B, num_bins, L, X], binned logits  where X is 23 for plddt and L for pae and pde
        max_distance: float, maximum distance
        num_bins: int, number of bins
    Returns:
        unbinned: [B, L, L], unbinned matrix
    """
    midpoints = find_bin_midpoints(max_distance, num_bins, device=logits.device)
    probabilities = torch.nn.Softmax(dim=1)(logits).detach().float()
    unbinned = (probabilities * midpoints[None, :, None, None]).sum(dim=1)
    return unbinned


def create_chainwise_masks_1d(
    ch_label: NDArray[np.str_], device: Union[str, torch.device] = "cpu"
) -> dict[str, Bool[torch.Tensor, "L"]]:
    """
    Create 1D chainwise masks for a set of chain labels
    Args:
        ch_label: np.ndarray [L], chain labels
        device: torch.device, device to run on
    Returns:
        ch_masks: dict, chain maps chain letter to which elements to score for that chain
    """
    unique_chains = np.unique(ch_label)
    ch_masks = {}
    for chain in unique_chains:
        indices = torch.from_numpy((ch_label == chain)).to(
            dtype=torch.bool, device=device
        )
        ch_masks[chain] = indices
    return ch_masks


def create_chainwise_masks_2d(
    ch_label: NDArray[np.str_], device: Union[str, torch.device] = "cpu"
) -> dict[str, Bool[torch.Tensor, "L L"]]:
    """
    Create 2D chainwise masks for a set of chain labels
    Args:
        ch_label: np.ndarray [L], chain labels
        device: torch.device, device to run on
    Returns:
        ch_masks: dict, chain maps chain letter to which elements to score for that chain
    """
    unique_chains = np.unique(ch_label)
    ch_masks = {}
    for chain in unique_chains:
        indices = torch.from_numpy((ch_label == chain))
        mask = torch.outer(indices, indices).to(dtype=torch.bool, device=device)
        ch_masks[chain] = mask
    return ch_masks


def create_interface_masks_2d(
    ch_label: NDArray[np.str_], device: Union[str, torch.device] = "cpu"
) -> dict[tuple[str, str], Bool[torch.Tensor, "L L"]]:
    """
    Create interface masks for a set of chain labels
    Args:
        ch_label: np.ndarray [L], chain labels
        device: torch.device, device to run on
    Returns:
        pairs_to_score: dict mapping chain pairs to boolean masks
    """
    unique_chains = np.unique(ch_label)
    pairs_to_score = {}
    for chain_i, chain_j in combinations(unique_chains, 2):
        chain_i_indices = torch.from_numpy((ch_label == chain_i))
        chain_j_indices = torch.from_numpy((ch_label == chain_j))
        to_be_scored = torch.outer(chain_i_indices, chain_j_indices).to(
            dtype=torch.bool, device=device
        ) + torch.outer(chain_j_indices, chain_i_indices).to(
            dtype=torch.bool, device=device
        )
        pairs_to_score[(chain_i, chain_j)] = to_be_scored
    return pairs_to_score


def compute_mean_over_subsampled_pairs(
    matrix_to_mean: Float[torch.Tensor, "B L M"],
    pairs_to_score: Bool[torch.Tensor, "L M"],
    eps: float = 1e-6,
) -> Float[torch.Tensor, "B"]:
    """
    Compute the mean over a subsample of pairs in a 2d matrix. Returns a tensor with an element for each batch
    Args:
        matrix_to_mean: tensor of shape (batch, L, L)
        pairs_to_score: 2d tensor of shape (L, L) with 1s where pairs should be scored and 0s elsewhere
        eps: small epsilon value to avoid division by zero
    Returns:
        1d tensor of shape (batch,) with the mean over the subsampled pairs for each batch
    """
    B, L, M = matrix_to_mean.shape
    assert matrix_to_mean.shape == (
        B,
        L,
        M,
    ), "Matrix to mean should be of shape (batch, L, M)"
    assert pairs_to_score.shape == (L, M), "Pairs to score should be of shape (L, M)"
    batch = (matrix_to_mean * pairs_to_score).sum(dim=(-1, -2)) / (
        pairs_to_score.sum() + eps
    )
    assert batch.shape == (B,), "Batch should be of shape (batch,)"
    return batch


def compute_min_over_subsampled_pairs(
    matrix_to_min: Float[torch.Tensor, "B L M"],
    pairs_to_score: Bool[torch.Tensor, "L M"],
) -> Float[torch.Tensor, "B"]:
    """
    Compute the min over a subsample of pairs in a 2d matrix. Returns a tensor with an element for each batch
    Args:
        matrix_to_min: tensor of shape (batch, L, L)
        pairs_to_score: 2d tensor of shape (L, L) with 1s where pairs should be scored and 0s elsewhere
    Returns:
        1d tensor of shape (batch,) with the min over the subsampled pairs for each batch
    """
    B, L, M = matrix_to_min.shape
    assert matrix_to_min.shape == (
        B,
        L,
        M,
    ), "Matrix to min should be of shape (batch, L, M)"
    assert pairs_to_score.shape == (L, M), "Pairs to score should be of shape (L, M)"
    # Use torch.where to efficiently mask without cloning the entire matrix
    # This broadcasts pairs_to_score across the batch dimension
    masked_matrix = torch.where(
        pairs_to_score.bool(),  # condition (L, M) -> broadcasts to (B, L, M)
        matrix_to_min,  # if True: use original values (B, L, M)
        torch.tensor(
            float("inf"), device=matrix_to_min.device, dtype=matrix_to_min.dtype
        ),  # if False: use inf
    )

    # Flatten the last two dimensions and compute min across them
    batch = masked_matrix.view(B, -1).min(dim=-1)[0]

    assert batch.shape == (B,), "Batch should be of shape (batch,)"
    return batch


def spread_batch_into_dictionary(batch: Float[torch.Tensor, "B"]) -> dict[int, float]:
    """
    Given a batch of data, create a dictionary with keys as the batch index and value as the corresponding data
    Args:
        batch: 1D tensor of shape (B,)
    Returns:
        Dictionary mapping batch indices to float values
    """
    assert len(batch.shape) == 1, f"Batch should be a 1d tensor, {batch}"
    return {i: data.item() for i, data in enumerate(batch)}
