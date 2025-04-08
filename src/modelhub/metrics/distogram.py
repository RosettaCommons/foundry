from beartype.typing import Any

import torch.nn as nn

from modelhub.loss.af3_losses import distogram_loss
from modelhub.metrics.base import Metric
import torch
from jaxtyping import Float
from biotite.structure import AtomArrayStack
from datahub.utils.token import get_af3_token_representative_idxs
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np


class DistogramLoss(Metric):
    """Computes the distogram loss, taking into account the coordinate mask."""

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "pred_distogram": ("network_output", "distogram"),
            "X_rep_atoms_I": ("extra_info", "X_rep_atoms_I"),
            "crd_mask_rep_atoms_I": ("extra_info", "crd_mask_rep_atoms_I"),
        }

    def __init__(self):
        super().__init__()
        self.cce_loss = nn.CrossEntropyLoss(reduction="none")

    def compute(
        self,
        pred_distogram: Float[torch.Tensor, "I I n_bins"],
        X_rep_atoms_I: Float[torch.Tensor, "I 3"],
        crd_mask_rep_atoms_I: Float[torch.Tensor, "I"],
    ) -> dict[str, Any]:
        """Computes the distogram loss.

        Args:
            pred_distogram: The predicted distogram. Shape: [I, I, n_bins], where n_bins is the number of bins (64 + 1 = 65).
            X_rep_atoms_I: The ground-truth coordinates of the representative atoms for each token. Shape: [I, 3].
            crd_mask_rep_atoms_I: A boolean mask indicating which representative atoms are present. Shape: [I].
        """
        loss = distogram_loss(
            pred_distogram, X_rep_atoms_I, crd_mask_rep_atoms_I, self.cce_loss
        )
        return {"distogram_loss": loss.detach().item()}

def bin_distances(coords: Float[torch.Tensor, "... L 3"], min_distance: int = 2, max_distance: int = 22, n_bins: int = 64) -> Float[torch.Tensor, "... L L {n_bins}+1"]:
    # TODO: Refactor loss to use this function instead (more re-usable)
    """Converts coordinates into binned distances according to the given parameters.

    NOTE: Our returned number of bins will be n_bins + 1, as torch.bucketize adds an additional bin for values greater than the maximum.
    
    Args:
        coords (torch.Tensor): The input tensor of coordinates. May be batched.
        min_distance (float): The minimum distance for binning.
        max_distance (float): The maximum distance for binning.
        n_bins (int): The number of bins to use.

    Returns:
        torch.Tensor: The binned distances.
    """
    # Compute pairwise distances
    distance_map = torch.cdist(coords, coords)

    # (Replace NaN's with a large value to avoid issues with bucketize)
    distance_map = torch.nan_to_num(distance_map, nan=9999.0)
    
    # ... bin the distances
    n_bins = torch.linspace(min_distance, max_distance, n_bins).to(coords.device)
    binned_distances = torch.bucketize(distance_map, n_bins)
    
    return binned_distances


def masked_distogram_cross_entropy_loss(
    input: Float[torch.Tensor, "D I I n_bins"],
    target: Float[torch.Tensor, "D I I"],
    mask: Float[torch.Tensor, "I I"] = None,
) -> torch.Tensor:
    # TODO: Refactor loss to use this function instead (more re-usable)
    """Computes the masked cross-entropy between two distograms.

    Note that the cross-entropy loss is not symmetric; that is, H(x, y) != H(y, x).
    """
    # From the PyTorch documentation (where C = number of classes, N = batch size):
    # > Input: Shape: (C), (N, C) or (N, C, d1, d2, ..., dk)
    # > Target: Shape: (N) or (N, d1, d2, ..., dk) where each value should be between [0, C)
    input = rearrange(input, 'd i j n_bins -> d n_bins i j')
    loss = F.cross_entropy(input, target, reduction="none")

    # Apply mask and normalize
    masked_loss = loss * mask if mask is not None else loss
    normalized_loss = masked_loss.sum(dim=(-1, -2)) / mask.sum() + 1e-4 # [D]

    return normalized_loss


class DistogramComparisons(Metric):
    """Compares model distogram representations.

    Namely:
        - The representation from the TRUNK vs. GROUND TRUTH
        - The representation from the TRUNK vs. PREDICTED COORDINATES
    
    Optionally, we also subset to intra-ligand (atomized) distances.
    """
    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "X_L": ("network_output", "X_L"), # [D, L, 3]
            "trunk_pred_distogram": ("network_output", "distogram"), # [I, I, 65], where 65 is the number of bins (64 + 1)
            "X_rep_atoms_I": ("extra_info", "X_rep_atoms_I"), # [D, I, 3]
            "crd_mask_rep_atoms_I": ("extra_info", "crd_mask_rep_atoms_I"), # [D, I]
            "ground_truth_atom_array_stack": "ground_truth_atom_array_stack", 
        }

    def __init__(self, separate_atomized_tokens: bool = True):
        """
        Args:
            separate_atomized: Whether to log separate comparisons for atomized tokens
        """
        super().__init__()
        self.separate_atomized_tokens = separate_atomized_tokens

    def compute(
        self,
        X_L: Float[torch.Tensor, "D L 3"],
        trunk_pred_distogram: Float[torch.Tensor, "I I n_bins"],
        X_rep_atoms_I: Float[torch.Tensor, "D I 3"],
        crd_mask_rep_atoms_I: Float[torch.Tensor, "D I"],
        ground_truth_atom_array_stack: AtomArrayStack,
    ) -> dict[str, Any]:
        """Computes the distogram loss for the trunk vs. ground truth and trunk vs. predicted coordinates.

        Optionally, we also subset to intra-ligand (atomized) distances.

        Args:
            X_L: The predicted coordinates. Shape: [D, L, 3]
            trunk_pred_distogram: The prediction from the DistogramHead, which linearly projects the trunk features. Shape: [I, I, n_bins]
            X_rep_atoms_I: The ground-truth coordinates of the representative atoms for each token. Shape: [D, I, 3]
            crd_mask_rep_atoms_I: A boolean mask indicating which representative atoms are present. Shape: [D, I]
            ground_truth_atom_array_stack: The ground-truth atom array stack, one model per diffusion sample. Shape: [D, L]
        """
        MIN_ATOMIZED = 5

        # ... choose the first model, as we only care about 2D distance (frame-invariant)
        ground_truth_atom_array = ground_truth_atom_array_stack[0]

        _token_rep_idxs = torch.from_numpy(get_af3_token_representative_idxs(ground_truth_atom_array)).to(X_L.device)
        token_rep_atom_array = ground_truth_atom_array[get_af3_token_representative_idxs(ground_truth_atom_array)]

        # Create 2D coordinate mask for valid pairs of representative atoms
        crd_mask_rep_atom_II = crd_mask_rep_atoms_I.unsqueeze(-1) * crd_mask_rep_atoms_I.unsqueeze(-2)

        results = {}

        # ... trunk vs. ground truth
        binned_distogram_from_ground_truth = bin_distances(X_rep_atoms_I, n_bins=64)
        results["trunk_vs_ground_truth_cce"] = masked_distogram_cross_entropy_loss(
            trunk_pred_distogram.unsqueeze(0), binned_distogram_from_ground_truth.unsqueeze(0), crd_mask_rep_atom_II
        ).detach().item()

        # ... trunk vs. predicted coordinates
        # (Predicted coordinates are batched, so we build the distogram for each predicted structure)
        binned_distogram_from_pred_coords = bin_distances(X_L[:, _token_rep_idxs], n_bins=64)
        losses = masked_distogram_cross_entropy_loss(
            repeat(trunk_pred_distogram, "i j n_bins -> d i j n_bins", d=binned_distogram_from_pred_coords.shape[0]), 
            binned_distogram_from_pred_coords, 
            crd_mask_rep_atom_II
        )

        results.update({
            f"trunk_vs_pred_coords_cce_{i}": loss.detach().item()
            for i, loss in enumerate(losses)
        })

        if self.separate_atomized_tokens and np.sum(token_rep_atom_array.atomize) > MIN_ATOMIZED:
            # ... trunk vs. ground truth (atomized)

            # Create a mask that is both atomized and intra-residue
            same_pn_unit_mask_LL = np.equal.outer(token_rep_atom_array.pn_unit_iid, token_rep_atom_array.pn_unit_iid)
            same_res_id_mask_LL = np.equal.outer(token_rep_atom_array.res_id, token_rep_atom_array.res_id)
            atomized_mask_LL = np.outer(token_rep_atom_array.atomize, token_rep_atom_array.atomize)
            atomized_intra_mask = torch.from_numpy(same_pn_unit_mask_LL * same_res_id_mask_LL * atomized_mask_LL).to(X_L.device) * crd_mask_rep_atom_II

            # Compute the losses, applying the mask
            results["trunk_vs_ground_truth_cce_ligand_intra"] = masked_distogram_cross_entropy_loss(
                trunk_pred_distogram.unsqueeze(0), binned_distogram_from_ground_truth.unsqueeze(0), atomized_intra_mask
            ).detach().item()

            # ... trunk vs. predicted coordinates (atomized)
            losses = masked_distogram_cross_entropy_loss(
                repeat(trunk_pred_distogram, "i j n_bins -> d i j n_bins", d=binned_distogram_from_pred_coords.shape[0]), 
                binned_distogram_from_pred_coords, 
                atomized_intra_mask
            )
            results.update({
                f"trunk_vs_pred_coords_cce_ligand_intra_{i}": loss.detach().item()
                for i, loss in enumerate(losses)
            })

        return results


    

    
