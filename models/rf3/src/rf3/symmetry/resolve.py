"""Generalized symmetry resolution implementation, operating on the outputs of AtomWorks.io `parse` function."""

import logging
from typing import Any, Dict

import numpy as np
import torch
from atomworks.io.transforms.atom_array import ensure_atom_array_stack
from atomworks.ml.transforms.atom_array import AddGlobalTokenIdAnnotation
from atomworks.ml.transforms.atomize import AtomizeByCCDName
from atomworks.ml.transforms.base import Compose, convert_to_torch
from atomworks.ml.transforms.symmetry import FindAutomorphismsWithNetworkX
from biotite.structure import AtomArray, AtomArrayStack
from jaxtyping import Bool, Float, Int
from rf3.loss.af3_losses import (
    ResidueSymmetryResolution,
    SubunitSymmetryResolution,
)

logger = logging.getLogger(__name__)


def resolve_symmetries(
    predicted_atom_array: AtomArray | AtomArrayStack,
    ground_truth_atom_array: AtomArray | AtomArrayStack,
    resolve_residue_symmetries: bool = True,
    resolve_subunit_symmetries: bool = True,
) -> AtomArrayStack:
    """
    Generalized symmetry resolution for both residue- and subunit-level symmetries.

    Returns updated ground truth AtomArray with coordinates that minimize RMSD with the predicted structure.

    Args:
        predicted_atom_array: Predicted structure as AtomArray or AtomArrayStack
        ground_truth_atom_array: Ground truth structure as AtomArray or AtomArrayStack
        resolve_residue_symmetries: Whether to resolve residue-level symmetries
        resolve_subunit_symmetries: Whether to resolve subunit-level symmetries

    Returns:
        Updated ground truth AtomArray or AtomArrayStack with resolved coordinates
    """
    predicted_stack = ensure_atom_array_stack(predicted_atom_array)
    ground_truth_stack = ensure_atom_array_stack(ground_truth_atom_array)

    # Set ground truth coordinates to nan if they are nan in the predicted coordinates...
    ground_truth_stack.coord[np.isnan(predicted_stack.coord)] = np.nan

    # ... then nan-to-num the pred_aa coordinates (otherwise, the symmetry resolution may fail)
    # TODO: Update the symmetry resolution to handle NaNs in the predicted coordinates
    predicted_stack.coord = np.nan_to_num(predicted_stack.coord)

    # Extract predicted and ground truth coordinates
    X_pred: Float[torch.Tensor, "D L 3"] = torch.tensor(
        predicted_stack.coord, dtype=torch.float32
    )
    X_gt: Float[torch.Tensor, "D L 3"] = torch.tensor(
        ground_truth_stack.coord, dtype=torch.float32
    )

    # (Match dimensions)
    D_pred, L_pred = X_pred.shape[:2]
    D_gt, L_gt = X_gt.shape[:2]

    if D_pred != D_gt:
        if D_gt == 1:
            X_gt = X_gt.expand(D_pred, -1, -1)
        else:
            raise ValueError(
                f"Cannot broadcast ground truth of shape ({D_gt}) to prediction of shape ({D_pred})"
            )
    assert L_pred == L_gt, "Length mismatch: predicted {L_pred}, ground truth {L_gt}"

    # Generate symmetric features (e.g., automorphisms, entity information, etc.) inputs from ground truth
    symmetry_data = generate_symmetry_resolution_inputs_from_atom_array(
        ground_truth_stack
    )

    # Extract coordinate mask from ground truth stack
    crd_mask: Bool[torch.Tensor, "D L"]
    if "occupancy" in ground_truth_stack.get_annotation_categories():
        crd_mask = torch.tensor(ground_truth_stack.occupancy > 0.0, dtype=torch.bool)
    else:
        logger.warning(
            "No occupancy annotation found in ground truth, using coordinate validity mask (not NaN)"
        )
        crd_mask = ~torch.isnan(torch.tensor(ground_truth_stack.coord)).any(dim=-1)

    assert not torch.isnan(
        X_pred
    ).any(), "NaN coordinates found in predicted structure!"

    # Apply symmetry resolution (returns updated ground truth coordinates)
    X_gt_resolved: Float[torch.Tensor, "D L 3"] = apply_symmetry_resolution(
        X_pred=X_pred,
        X_gt=X_gt,
        crd_mask=crd_mask,
        automorphisms=symmetry_data["automorphisms"],
        molecule_entity=symmetry_data["molecule_entity"],
        molecule_iid=symmetry_data["molecule_iid"],
        crop_mask=symmetry_data["crop_mask"],
        coord_atom_lvl=symmetry_data["coord_atom_lvl"],
        mask_atom_lvl=symmetry_data["mask_atom_lvl"],
        resolve_residue=resolve_residue_symmetries,
        resolve_subunit=resolve_subunit_symmetries,
    )

    # Update the ground truth AtomArray with resolved coordinates
    result_stack = ground_truth_stack.copy()
    result_stack.coord = X_gt_resolved.cpu().numpy()

    return result_stack


def generate_symmetry_resolution_inputs_from_atom_array(
    atom_array: AtomArray | AtomArrayStack,
) -> Dict[str, Any]:
    """
    Generate all inputs needed for symmetry resolution from an AtomArray.

    Args:
        atom_array: Input AtomArray or AtomArrayStack

    Returns:
        Dictionary containing:
            - automorphisms: List[np.ndarray]
            - molecule_entity: torch.Tensor [N_atoms]
            - molecule_iid: torch.Tensor [N_atoms]
            - coord_atom_lvl: torch.Tensor [N_atoms, 3]
            - mask_atom_lvl: torch.Tensor [N_atoms]
            - atom_to_token_map: torch.Tensor [N_atoms]
            - crop_mask: torch.Tensor [N_atoms]
    """
    # (Take first model)
    atom_array_stack = ensure_atom_array_stack(atom_array)
    atom_array = atom_array_stack[0]

    # (Avoid modifying the original)
    atom_array = atom_array.copy()

    # Prepare transform pipeline to generate features
    transforms = [AtomizeByCCDName(atomize_by_default=True)]

    if "token_id" not in atom_array.get_annotation_categories():
        transforms.append(AddGlobalTokenIdAnnotation())

    transforms.append(FindAutomorphismsWithNetworkX())

    pipeline = Compose(transforms)
    data = pipeline({"atom_array": atom_array})
    atom_array = data["atom_array"]

    result: Dict[str, Any] = {}
    # Extract automorphisms
    result["automorphisms"] = data.get("automorphisms", [])
    # Extract molecule annotations (assert they exist)
    assert (
        "molecule_entity" in atom_array.get_annotation_categories()
    ), "molecule_entity annotation required"
    assert (
        "molecule_iid" in atom_array.get_annotation_categories()
    ), "molecule_iid annotation required"

    result["molecule_entity"] = atom_array.molecule_entity
    result["molecule_iid"] = atom_array.molecule_iid

    # Extract coordinates
    coords: np.ndarray = atom_array.coord
    result["coord_atom_lvl"] = coords

    # Extract mask from occupancy (like in lddt.py) - no batch dimension for SubunitSymmetryResolution
    mask: np.ndarray
    if "occupancy" in atom_array.get_annotation_categories():
        mask = atom_array.occupancy > 0.0
    else:
        # Fallback to coordinate validity
        mask = ~np.isnan(atom_array.coord).any(axis=-1)

    # Keep mask as [N_atoms] for SubunitSymmetryResolution compatibility
    result["mask_atom_lvl"] = mask

    # Extract atom to token map (like in lddt.py)
    if "token_id" in atom_array.get_annotation_categories():
        result["atom_to_token_map"] = atom_array.token_id.astype(np.int32)
    else:
        # This should not happen since AddGlobalTokenIdAnnotation was applied
        raise ValueError(
            "token_id annotation not found after AddGlobalTokenIdAnnotation"
        )

    # Create crop_mask (full range)
    result["crop_mask"] = np.arange(len(atom_array), dtype=np.int32)

    # Step 3: Convert all numpy arrays to torch tensors using convert_to_torch
    # First, create a temporary dict with the keys we want to convert
    torch_data = {
        "molecule_entity": result["molecule_entity"],
        "molecule_iid": result["molecule_iid"],
        "coord_atom_lvl": result["coord_atom_lvl"],
        "mask_atom_lvl": result["mask_atom_lvl"],
        "atom_to_token_map": result["atom_to_token_map"],
        "crop_mask": result["crop_mask"],
    }

    # Convert to torch tensors
    torch_data = convert_to_torch(torch_data, list(torch_data.keys()))

    # Update result with torch tensors
    result.update(torch_data)

    return result


def apply_symmetry_resolution(
    X_pred: Float[torch.Tensor, "D L 3"],
    X_gt: Float[torch.Tensor, "D L 3"],
    crd_mask: Bool[torch.Tensor, "D L"],
    automorphisms: list,
    molecule_entity: Int[torch.Tensor, "N_atoms"],
    molecule_iid: Int[torch.Tensor, "N_atoms"],
    crop_mask: Int[torch.Tensor, "N_atoms"],
    coord_atom_lvl: Float[torch.Tensor, "N_atoms 3"],
    mask_atom_lvl: Bool[torch.Tensor, "N_atoms"],
    resolve_residue: bool = True,
    resolve_subunit: bool = True,
) -> Float[torch.Tensor, "D L 3"]:
    """
    Apply the actual symmetry resolution using the existing classes and return updated coordinates.

    Args:
        X_pred: Predicted coordinates [D, L, 3]
        X_gt: Ground truth coordinates [D, L, 3]
        crd_mask: Coordinate mask [D, L]
        automorphisms: List of automorphism groups
        molecule_entity: Molecule entity IDs [N_atoms]
        molecule_iid: Molecule instance IDs [N_atoms]
        crop_mask: Crop mask indices [N_atoms]
        coord_atom_lvl: Atom-level coordinates [N_atoms, 3]
        mask_atom_lvl: Atom-level mask [N_atoms]
        resolve_residue: Whether to resolve residue symmetries
        resolve_subunit: Whether to resolve subunit symmetries

    Returns:
        Updated ground truth coordinates [D, L, 3]
    """
    # Prepare loss_input dictionary for existing classes
    loss_input: Dict[str, torch.Tensor] = {
        "X_gt_L": X_gt.clone(),
        "crd_mask_L": crd_mask.clone(),
    }

    # Apply subunit symmetry resolution
    if resolve_subunit:
        subunit_resolver = SubunitSymmetryResolution()

        # Create symmetry resolution input
        symmetry_resolution: Dict[str, torch.Tensor] = {
            "molecule_entity": molecule_entity,
            "molecule_iid": molecule_iid,
            "crop_mask": crop_mask,
            "coord_atom_lvl": coord_atom_lvl,
            "mask_atom_lvl": mask_atom_lvl,
        }

        # Create network output dict
        network_output: Dict[str, torch.Tensor] = {"X_L": X_pred}

        # Apply subunit resolution
        logger.info("Applying subunit symmetry resolution")
        loss_input = subunit_resolver(network_output, loss_input, symmetry_resolution)

    # Apply residue symmetry resolution
    if resolve_residue and automorphisms:
        logger.info("Applying residue symmetry resolution")
        residue_resolver = ResidueSymmetryResolution()

        # Create network output dict
        network_output: Dict[str, torch.Tensor] = {"X_L": X_pred}

        # Apply residue resolution
        loss_input = residue_resolver(network_output, loss_input, automorphisms)

    # Return the updated ground truth coordinates
    return loss_input["X_gt_L"]
