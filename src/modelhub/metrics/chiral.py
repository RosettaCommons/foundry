from beartype.typing import Any
import torch
from jaxtyping import Float, Bool
from biotite.structure import AtomArrayStack, AtomArray

from modelhub.metrics.base import Metric
from modelhub.kinematics import get_dih
from datahub.transforms.af3_reference_molecule import get_af3_reference_molecule_features
from datahub.transforms.rdkit_utils import get_rdkit_chiral_centers
from datahub.transforms.chirals import add_af3_chiral_features
from cifutils.transforms.atom_array import ensure_atom_array_stack


def calc_chiral_loss_masked(pred: Float[torch.Tensor, "B L ... 3"], chirals: Float[torch.Tensor, "n_chiral 5"], mask: Bool[torch.Tensor, "I"], ground_truth_atom_array: AtomArray):
    """Calculate error in dihedral angles for chiral atoms

    Args:
        pred: predicted coords (B, L, :, 3)
        chirals: True coords (nchiral, 5); skip if 0 chiral sites. 5 dimension are indices for 4 atoms that make dihedral and the ideal angle they should form
        mask: Boolean mask of shape (I) indicating valid positions (e.g., non-NaN coordinates, desired residue type)

    Returns:
        chiral_loss_sum: sum of squared errors of chiral angles (B)
        n_chiral_centers: number of chiral centers in the structure 
        percent_correct_chirality: percentage of correctly predicted chiral centers (B)
    """
    if not chirals.numel() or not mask.sum():
        # ... no chiral centers; exit
        return {}

    # ... get the coordinates of all four atoms involved in each chiral center
    chiral_dih = pred[:, chirals[..., :-1].long()] # (n_chiral 5) -> (B, n_chiral, 4, 3)

    # ... for each chiral center, compute the dihedral angle
    pred_dih = get_dih(
        chiral_dih[..., 0, :],
        chiral_dih[..., 1, :],
        chiral_dih[..., 2, :],
        chiral_dih[..., 3, :],
    ) # [B, n_chiral]

    chiral_center_is_valid = mask[chirals[..., :-1].long()].all(dim=-1) # [L] -> [n_chiral] (a chiral center is valid iff ALL atoms are included)

    # ... total chiral loss (sum of squared errors)
    diff = pred_dih - chirals[..., -1] # [B, n_chiral]
    is_correct_chirality = torch.sign(pred_dih) == torch.sign(chirals[..., -1]) # [B, n_chiral]
    percent_correct_chirality = (is_correct_chirality[:, chiral_center_is_valid]).sum(dim=-1) / chiral_center_is_valid.sum(dim=-1) # [B]

    l = torch.square(diff[:, chiral_center_is_valid]).sum(dim=-1)# [B]

    return {
        "chiral_loss_sum": l, # [B]
        "n_chiral_centers": chiral_center_is_valid.sum(dim=-1), # [B]
        "percent_correct_chirality": percent_correct_chirality, # [B]
    }


class ChiralLoss(Metric):
    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "predicted_atom_array_stack": "predicted_atom_array_stack",
            "ground_truth_atom_array_stack": "ground_truth_atom_array_stack", 
            "chiral_feats": ("network_input", "f", "chiral_feats"),
        }
    
    def compute(
        self, 
        predicted_atom_array_stack: AtomArrayStack | AtomArray,
        ground_truth_atom_array_stack: AtomArrayStack | AtomArray,
        chiral_feats: Float[torch.Tensor, "n_chiral 5"] = None, 
    ):
        """Compute the chiral loss for the predicted and ground truth atom arrays.

        If chiral features are not directly provided, they will be re-computed from the AtomArrays.
        """
        predicted_atom_array_stack = ensure_atom_array_stack(predicted_atom_array_stack)
        ground_truth_atom_array_stack = ensure_atom_array_stack(ground_truth_atom_array_stack)

        chiral_loss = {}
        # (Choose the first model  - chirality does not depend on our data augmentation)
        ground_truth_atom_array = ground_truth_atom_array_stack[0]

        if chiral_feats is None:
            # Generate chiral features if not provided
            _, rdkit_mols = get_af3_reference_molecule_features(ground_truth_atom_array)
            chiral_centers = get_rdkit_chiral_centers(rdkit_mols)
            chiral_feats = add_af3_chiral_features(ground_truth_atom_array, chiral_centers, rdkit_mols)
        
        X_L = torch.from_numpy(predicted_atom_array_stack.coord).to(device=chiral_feats.device)

        categories = ["polymer", "non_polymer"]
        _polymer_mask = torch.from_numpy(ground_truth_atom_array.is_polymer).to(device=chiral_feats.device)
        # (Only consider non-NaN coordinates in the ground truth, since otherwise we can't compare dihedral angles)
        _valid_coord_mask = ~torch.isnan(torch.from_numpy(ground_truth_atom_array.coord)).any(dim=1).to(device=chiral_feats.device)
        masks = [_polymer_mask, ~_polymer_mask]

        for category, mask in zip(categories, masks):
            # ... compute the chiral loss, given the mask
            result = calc_chiral_loss_masked(
                X_L,
                chiral_feats,
                mask=mask & _valid_coord_mask,
                ground_truth_atom_array=ground_truth_atom_array,
            )
            
            if not result:
                # No chiral centers - skip
                continue
                
            # ... store the metric results, meaned over the diffusion batch
            if result["n_chiral_centers"] > 0:
                chiral_loss[f"{category}_n_chiral_centers"] = result["n_chiral_centers"].item()
                chiral_loss[f"{category}_chiral_loss_mean"] = (result["chiral_loss_sum"] / result["n_chiral_centers"]).mean().item()
                chiral_loss[f"{category}_percent_correct_chirality"] = result["percent_correct_chirality"].mean().item()

        return chiral_loss
