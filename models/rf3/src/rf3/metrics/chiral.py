import torch
from atomworks.io.transforms.atom_array import ensure_atom_array_stack
from atomworks.ml.transforms.af3_reference_molecule import (
    get_af3_reference_molecule_features,
)
from atomworks.ml.transforms.chirals import add_af3_chiral_features
from atomworks.ml.transforms.rdkit_utils import get_rdkit_chiral_centers
from beartype.typing import Any
from biotite.structure import AtomArray, AtomArrayStack
from jaxtyping import Bool, Float
from rf3.kinematics import get_dih

from foundry.metrics.metric import Metric


def calc_chiral_metrics_masked(
    pred: Float[torch.Tensor, "B L ... 3"],
    chirals: Float[torch.Tensor, "n_chiral 5"],
    mask: Bool[torch.Tensor, "I"],
):
    """Calculate metrics for chiral centers, including:
        - n_chiral_centers (B): number of chiral centers in the structure
        - chiral_loss_mean (B): mean of the squared errors of chiral angles
        - percent_correct_chirality (B): percentage of correctly predicted chiral centers

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
    chiral_dih = pred[
        :, chirals[..., :-1].long()
    ]  # (n_chiral 5) -> (B, n_chiral, 4, 3)

    # ... for each chiral center, compute the dihedral angle
    pred_dih = get_dih(
        chiral_dih[..., 0, :],
        chiral_dih[..., 1, :],
        chiral_dih[..., 2, :],
        chiral_dih[..., 3, :],
    )  # [B, n_chiral]

    # ... total chiral loss (sum of squared errors)
    diff = pred_dih - chirals[..., -1]  # [B, n_chiral]
    is_correct_chirality = torch.sign(pred_dih) == torch.sign(
        chirals[..., -1]
    )  # [B, n_chiral]

    # To avoid over-counting chirals, we should only keep one "row" for each chiral center (rather than enumerating all orderings)
    inf_tensor = torch.tensor(
        [-float("inf")], device=chirals.device, dtype=chirals.dtype
    )
    shifted = torch.cat([inf_tensor, chirals[:-1, 0]], dim=0)  # Shape [24]
    first_occurence_mask = chirals[:, 0] != shifted

    is_valid_chiral_center = mask[chirals[..., :-1].long()].all(
        dim=-1
    )  # [L] -> [n_chiral] (a chiral center is valid iff ALL atoms are included)
    # ... and only keep the first occurrence of each chiral center
    is_valid_chiral_center = is_valid_chiral_center & first_occurence_mask

    percent_correct_chirality = (is_correct_chirality[:, is_valid_chiral_center]).sum(
        dim=-1
    ) / is_valid_chiral_center.sum(dim=-1)  # [B]

    l = torch.square(diff[:, is_valid_chiral_center]).sum(dim=-1)  # [B]

    return {
        "chiral_loss_mean": l / mask.sum(),  # [B]
        "n_chiral_centers": is_valid_chiral_center.sum(dim=-1),  # [B]
        "percent_correct_chirality": percent_correct_chirality,  # [B]
    }


def compute_chiral_metrics(
    predicted_atom_array_stack: AtomArrayStack | AtomArray,
    ground_truth_atom_array_stack: AtomArrayStack | AtomArray,
    chiral_feats: Float[torch.Tensor, "n_chiral 5"] | None = None,
):
    """Compute chiral metrics from the predicted and ground truth atom arrays.

    If chiral features are not directly provided, they will be re-computed from the AtomArrays.

    Returns:
        dict: Dictionary containing chiral metrics, separated for polymers and non-polymers. The metrics are:
            - n_chiral_centers: number of chiral centers in the structure
            - chiral_loss_mean: mean of the squared errors of chiral angles
            - percent_correct_chirality: percentage of correctly predicted chiral centers
    """
    predicted_atom_array_stack = ensure_atom_array_stack(predicted_atom_array_stack)
    ground_truth_atom_array_stack = ensure_atom_array_stack(
        ground_truth_atom_array_stack
    )

    chiral_metrics = {}
    # (Choose the first model  - chirality does not depend on our data augmentation)
    ground_truth_atom_array = ground_truth_atom_array_stack[0]

    if chiral_feats is None:
        # Generate chiral features if not provided
        _, rdkit_mols = get_af3_reference_molecule_features(ground_truth_atom_array)
        chiral_centers = get_rdkit_chiral_centers(rdkit_mols)
        chiral_feats = add_af3_chiral_features(
            ground_truth_atom_array, chiral_centers, rdkit_mols
        )

    X_L = torch.from_numpy(predicted_atom_array_stack.coord).to(
        device=chiral_feats.device
    )

    categories = ["polymer", "non_polymer"]
    _polymer_mask = torch.from_numpy(ground_truth_atom_array.is_polymer).to(
        device=chiral_feats.device
    )
    # (Only consider non-NaN coordinates in the ground truth, since otherwise we can't compare dihedral angles)
    _valid_coord_mask = ~torch.isnan(
        torch.from_numpy(ground_truth_atom_array.coord)
    ).any(dim=1).to(device=chiral_feats.device)
    masks = [_polymer_mask, ~_polymer_mask]

    for category, mask in zip(categories, masks):
        # ... compute the chiral loss, given the mask
        result = calc_chiral_metrics_masked(
            X_L,
            chiral_feats,
            mask=mask & _valid_coord_mask,
        )

        if not result:
            # No chiral centers - skip
            continue

        # ... store the metric results, meaned over the diffusion batch
        if result["n_chiral_centers"] > 0:
            chiral_metrics[f"{category}_n_chiral_centers"] = result[
                "n_chiral_centers"
            ].item()
            chiral_metrics[f"{category}_chiral_loss_mean"] = (
                result["chiral_loss_mean"].mean().item()
            )
            chiral_metrics[f"{category}_percent_correct_chirality"] = (
                result["percent_correct_chirality"].mean().item()
            )

    return chiral_metrics


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
        chiral_metrics = compute_chiral_metrics(
            predicted_atom_array_stack,
            ground_truth_atom_array_stack,
            chiral_feats=chiral_feats,
        )

        return chiral_metrics
