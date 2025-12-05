import itertools
from typing import List

import einops
import numpy as np
import pandas as pd
import torch
import tree
from beartype.typing import Any
from biotite.structure import AtomArray, AtomArrayStack
from omegaconf import DictConfig
from rf3.chemical import NHEAVY
from rf3.metrics.metric_utils import (
    compute_mean_over_subsampled_pairs,
    compute_min_over_subsampled_pairs,
    create_chainwise_masks_1d,
    create_chainwise_masks_2d,
    create_interface_masks_2d,
    spread_batch_into_dictionary,
    unbin_logits,
)


def get_mean_atomwise_plddt(
    plddt_logits: torch.Tensor,
    is_real_atom: torch.Tensor,
    max_value: float,
) -> torch.Tensor:
    """Aggregate plddts.

    Args:
        plddt_logits: Tensor of shape [B, n_token, max_atoms_in_a_token * n_bin] with logits
        is_real_atom: Boolean mask of shape [B, n_token, max_atoms_in_a_token] indicating which atoms are real (i.e., not padding)
        max_value: Maximum value for pLDDT (assigned to the last bin)

    Returns:
        plddt: Tensor of shape [B,] with the mean atom-wise pLDDT for each batch
    """
    assert (
        plddt_logits.ndim == 3
    ), "plddt_logits must be a 3D tensor (B, n_token, max_atoms_in_a_token * n_bins)"

    # TODO: Replace with the last dimension of is_real_atom; right now that number is too large (36) because it includes hydrogens
    max_atoms_in_a_token = NHEAVY

    # Since the pLDDT logits have the last dimension (max_atoms_in_a_token * n_bins), we can calculate n_bins directly
    assert (
        plddt_logits.shape[-1] % max_atoms_in_a_token == 0
    ), "The last dimension of plddt_logits must be divisible by max_atoms_in_a_token!"
    n_bins = plddt_logits.shape[-1] // max_atoms_in_a_token

    # ... reshape to match what unbin_logits expects
    reshaped_plddt_logits = einops.rearrange(
        plddt_logits,
        "... n_token (max_atoms_in_a_token n_bins) -> ... n_bins n_token max_atoms_in_a_token",
        max_atoms_in_a_token=max_atoms_in_a_token,
        n_bins=n_bins,
    ).float()  # [..., n_token, n_bins * max_atoms_in_a_token] -> [ ..., n_bins, n_token, max_atoms_in_a_token]

    plddt = unbin_logits(
        reshaped_plddt_logits,
        max_value,
        n_bins,
    )

    is_real_atom = is_real_atom.to(device=plddt.device)

    #  ... create mask indicating which atoms are "real" (i.e., not padding) and calculate the mean
    mask = is_real_atom[:, :max_atoms_in_a_token].unsqueeze(0)
    atomwise_plddt_mean = (plddt * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2))

    return atomwise_plddt_mean


def compile_af3_confidence_outputs(
    plddt_logits: torch.Tensor,
    pae_logits: torch.Tensor,
    pde_logits: torch.Tensor,
    chain_iid_token_lvl: torch.Tensor,
    is_real_atom: torch.Tensor,
    example_id: str,
    confidence_loss_cfg: DictConfig | dict,
) -> dict[str, Any]:
    # TODO: Refactor to accept an AtomArray
    # TODO: Taking the confidence_loss_cfg does not align with functional programming best-practices; we should instead take  the max_value and n_bins as arguments

    """Given the confidence logits, computes the confidence metrics for the model's predictions.

    Returns:
        dict[str, Any]: A dictionary containing the following:
            - confidence_df: A DataFrame containing the aggregate confidence metrics at the chain- and interface-level
            - plddt: The pLDDT logits
            - pae: The pAE logits
            - pde: The pDE logits
    """

    # Reorder the input tensors to be in (B, n_bins, ...) format for unbinning
    plddt = unbin_logits(
        plddt_logits.reshape(
            -1,
            plddt_logits.shape[1],
            NHEAVY,
            confidence_loss_cfg.plddt.n_bins,
        )
        .permute(0, 3, 1, 2)
        .float(),
        confidence_loss_cfg.plddt.max_value,
        confidence_loss_cfg.plddt.n_bins,
    )

    # Unbin the pae and pde logits
    pae = unbin_logits(
        pae_logits.permute(0, 3, 1, 2).float(),
        confidence_loss_cfg.pae.max_value,
        confidence_loss_cfg.pae.n_bins,
    )
    pde = unbin_logits(
        pde_logits.permute(0, 3, 1, 2).float(),
        confidence_loss_cfg.pde.max_value,
        confidence_loss_cfg.pde.n_bins,
    )

    # Calculate interface metrics
    interface_masks = create_interface_masks_2d(chain_iid_token_lvl, device=pae.device)
    pae_interface = {
        k: spread_batch_into_dictionary(compute_mean_over_subsampled_pairs(pae, v))
        for k, v in interface_masks.items()
    }
    pde_interface = {
        k: spread_batch_into_dictionary(compute_mean_over_subsampled_pairs(pde, v))
        for k, v in interface_masks.items()
    }

    pae_interface_min = {
        k: spread_batch_into_dictionary(compute_min_over_subsampled_pairs(pae, v))
        for k, v in interface_masks.items()
    }

    pde_interface_min = {
        k: spread_batch_into_dictionary(compute_min_over_subsampled_pairs(pde, v))
        for k, v in interface_masks.items()
    }
    # Calculate chainwise metrics
    chain_masks_2d = create_chainwise_masks_2d(chain_iid_token_lvl, device=pae.device)
    pae_chainwise = {
        k: spread_batch_into_dictionary(compute_mean_over_subsampled_pairs(pae, v))
        for k, v in chain_masks_2d.items()
    }
    pde_chainwise = {
        k: spread_batch_into_dictionary(compute_mean_over_subsampled_pairs(pde, v))
        for k, v in chain_masks_2d.items()
    }

    chain_masks_1d = create_chainwise_masks_1d(
        chain_iid_token_lvl, device=is_real_atom.device
    )
    plddt_chainwise = {
        k: spread_batch_into_dictionary(
            compute_mean_over_subsampled_pairs(
                plddt, is_real_atom[..., :NHEAVY] * v[:, None]
            )
        )
        for k, v in chain_masks_1d.items()
    }

    # Aggregate confidence data
    confidence_data = {
        "example_id": example_id,
        "mean_plddt": spread_batch_into_dictionary(
            compute_mean_over_subsampled_pairs(plddt, is_real_atom[..., :NHEAVY])
        ),
        "mean_pae": spread_batch_into_dictionary(pae.mean(dim=(-1, -2))),
        "mean_pde": spread_batch_into_dictionary(pde.mean(dim=(-1, -2))),
        "chain_wise_mean_plddt": plddt_chainwise,
        "chain_wise_mean_pae": pae_chainwise,
        "chain_wise_mean_pde": pde_chainwise,
        "interface_wise_mean_pae": pae_interface,
        "interface_wise_mean_pde": pde_interface,
        "interface_wise_min_pae": pae_interface_min,
        "interface_wise_min_pde": pde_interface_min,
    }

    # Generate DataFrame rows
    num_batches = plddt.shape[0]
    chains = np.unique(chain_iid_token_lvl)
    chain_pairs = list(itertools.combinations(chains, 2))

    # For every batch, chain, and interface (chain pair), generate a dataframe row
    chain_rows = [
        {
            "example_id": example_id,
            "chain_chainwise": chain,
            "chainwise_plddt": confidence_data["chain_wise_mean_plddt"][chain][
                batch_idx
            ],
            "chainwise_pde": confidence_data["chain_wise_mean_pde"][chain][batch_idx],
            "chainwise_pae": confidence_data["chain_wise_mean_pae"][chain][batch_idx],
            "overall_plddt": confidence_data["mean_plddt"][batch_idx],
            "overall_pde": confidence_data["mean_pde"][batch_idx],
            "overall_pae": confidence_data["mean_pae"][batch_idx],
            "batch_idx": batch_idx,
        }
        for batch_idx in range(num_batches)
        for chain in chains
    ]

    interface_rows = [
        {
            "example_id": example_id,
            "chain_i_interface": chain_i,
            "chain_j_interface": chain_j,
            "pae_interface": confidence_data["interface_wise_mean_pae"][
                (chain_i, chain_j)
            ][batch_idx],
            "pde_interface": confidence_data["interface_wise_mean_pde"][
                (chain_i, chain_j)
            ][batch_idx],
            "min_pae_interface": confidence_data["interface_wise_min_pae"][
                (chain_i, chain_j)
            ][batch_idx],
            "min_pde_interface": confidence_data["interface_wise_min_pde"][
                (chain_i, chain_j)
            ][batch_idx],
            "overall_plddt": confidence_data["mean_plddt"][batch_idx],
            "overall_pde": confidence_data["mean_pde"][batch_idx],
            "overall_pae": confidence_data["mean_pae"][batch_idx],
            "batch_idx": batch_idx,
        }
        for batch_idx in range(num_batches)
        for (chain_i, chain_j) in chain_pairs
    ]

    return {
        "confidence_df": pd.DataFrame(itertools.chain([*chain_rows, *interface_rows])),
        "plddt": plddt,
        "pae": pae,
        "pde": pde,
    }


def compile_af3_style_confidence_outputs(
    plddt_logits: torch.Tensor,
    pae_logits: torch.Tensor,
    pde_logits: torch.Tensor,
    chain_iid_token_lvl: torch.Tensor | np.ndarray,
    is_real_atom: torch.Tensor,
    atom_array: AtomArray,
    confidence_loss_cfg: DictConfig | dict,
    batch_idx: int = 0,
) -> dict[str, Any]:
    """Compile confidence outputs in AlphaFold3-compatible format.

    Returns a dict with:
        - summary_confidences: Dict for {name}_summary_confidences.json
        - confidences: Dict for {name}_confidences.json (per-atom data)
        - plddt, pae, pde: Raw tensors for further processing
    """
    # Unbin logits
    plddt = unbin_logits(
        plddt_logits.reshape(
            -1,
            plddt_logits.shape[1],
            NHEAVY,
            confidence_loss_cfg.plddt.n_bins,
        )
        .permute(0, 3, 1, 2)
        .float(),
        confidence_loss_cfg.plddt.max_value,
        confidence_loss_cfg.plddt.n_bins,
    )

    pae = unbin_logits(
        pae_logits.permute(0, 3, 1, 2).float(),
        confidence_loss_cfg.pae.max_value,
        confidence_loss_cfg.pae.n_bins,
    )
    pde = unbin_logits(
        pde_logits.permute(0, 3, 1, 2).float(),
        confidence_loss_cfg.pde.max_value,
        confidence_loss_cfg.pde.n_bins,
    )

    # Get chain information
    if isinstance(chain_iid_token_lvl, torch.Tensor):
        chain_iid_token_lvl = chain_iid_token_lvl.cpu().numpy()
    chains = list(np.unique(chain_iid_token_lvl))
    n_chains = len(chains)

    # Calculate chainwise metrics
    chain_masks_1d = create_chainwise_masks_1d(
        chain_iid_token_lvl, device=is_real_atom.device
    )
    chain_masks_2d = create_chainwise_masks_2d(chain_iid_token_lvl, device=pae.device)

    # Chain-level pLDDT
    chain_plddt = {}
    for chain, mask in chain_masks_1d.items():
        chain_plddt[chain] = compute_mean_over_subsampled_pairs(
            plddt, is_real_atom[..., :NHEAVY] * mask[:, None]
        )[batch_idx].item()

    # Chain-level PAE (intra-chain)
    chain_pae = {}
    for chain, mask in chain_masks_2d.items():
        chain_pae[chain] = compute_mean_over_subsampled_pairs(pae, mask)[
            batch_idx
        ].item()

    # Chain-pair PAE/PDE (inter-chain, for iptm-like metric)
    interface_masks = create_interface_masks_2d(chain_iid_token_lvl, device=pae.device)
    chain_pair_pae = {}
    chain_pair_pae_min = {}
    chain_pair_pde = {}
    chain_pair_pde_min = {}
    for (chain_i, chain_j), mask in interface_masks.items():
        chain_pair_pae[(chain_i, chain_j)] = compute_mean_over_subsampled_pairs(
            pae, mask
        )[batch_idx].item()
        chain_pair_pae_min[(chain_i, chain_j)] = compute_min_over_subsampled_pairs(
            pae, mask
        )[batch_idx].item()
        chain_pair_pde[(chain_i, chain_j)] = compute_mean_over_subsampled_pairs(
            pde, mask
        )[batch_idx].item()
        chain_pair_pde_min[(chain_i, chain_j)] = compute_min_over_subsampled_pairs(
            pde, mask
        )[batch_idx].item()

    # Overall metrics for this batch
    overall_plddt = compute_mean_over_subsampled_pairs(
        plddt, is_real_atom[..., :NHEAVY]
    )[batch_idx].item()
    overall_pae = pae[batch_idx].mean().item()
    overall_pde = pde[batch_idx].mean().item()

    # Build chain_pair matrices (NxN)
    chain_pair_pae_matrix = [[None] * n_chains for _ in range(n_chains)]
    chain_pair_pae_min_matrix = [[None] * n_chains for _ in range(n_chains)]
    chain_pair_pde_matrix = [[None] * n_chains for _ in range(n_chains)]
    chain_pair_pde_min_matrix = [[None] * n_chains for _ in range(n_chains)]
    for i, chain_i in enumerate(chains):
        for j, chain_j in enumerate(chains):
            if i != j and (chain_i, chain_j) in chain_pair_pae:
                chain_pair_pae_matrix[i][j] = round(
                    chain_pair_pae[(chain_i, chain_j)], 2
                )
                chain_pair_pae_min_matrix[i][j] = round(
                    chain_pair_pae_min[(chain_i, chain_j)], 2
                )
                chain_pair_pde_matrix[i][j] = round(
                    chain_pair_pde[(chain_i, chain_j)], 2
                )
                chain_pair_pde_min_matrix[i][j] = round(
                    chain_pair_pde_min[(chain_i, chain_j)], 2
                )

    # Extract per-atom pLDDT values
    atom_plddts = plddt[batch_idx][is_real_atom[..., :NHEAVY]].cpu().tolist()

    # Extract atom/token chain and residue info from atom_array
    atom_chain_ids = atom_array.chain_id.tolist()
    token_chain_ids = list(chain_iid_token_lvl)
    token_res_ids = list(
        range(len(chain_iid_token_lvl))
    )  # Simplified; could map to actual res_id

    # PAE matrix for this batch
    pae_matrix = pae[batch_idx].cpu().tolist()

    # Build summary_confidences (AlphaFold3-style + RF3 extensions)
    summary_confidences = {
        "chain_ptm": [round(chain_plddt.get(c, 0.0), 2) for c in chains],
        "chain_pair_pae_min": chain_pair_pae_min_matrix,
        "chain_pair_pde_min": chain_pair_pde_min_matrix,
        "chain_pair_pae": chain_pair_pae_matrix,
        "chain_pair_pde": chain_pair_pde_matrix,
        "overall_plddt": round(overall_plddt, 4),
        "overall_pde": round(overall_pde, 4),
        "overall_pae": round(overall_pae, 4),
        # Note: ptm, iptm, has_clash should be populated from metrics_output
    }

    # Build full confidences (per-atom data)
    confidences = {
        "atom_chain_ids": atom_chain_ids,
        "atom_plddts": [round(p, 2) for p in atom_plddts],
        "pae": [[round(v, 2) for v in row] for row in pae_matrix],
        "token_chain_ids": token_chain_ids,
        "token_res_ids": token_res_ids,
    }

    return {
        "summary_confidences": summary_confidences,
        "confidences": confidences,
        "plddt": plddt,
        "pae": pae,
        "pde": pde,
    }


def compute_batch_indices_with_lowest_predicted_error(
    plddt: torch.Tensor,
    is_real_atom: torch.Tensor,
    pae: torch.Tensor,
    confidence_loss_cfg: dict | DictConfig,
    chain_iid_token_lvl: torch.Tensor,
    is_ligand: torch.Tensor,
    interfaces_to_score: list[tuple],
    pn_units_to_score: list[tuple],
) -> dict[str, Any]:
    """Given the confidence logits, computes the index within the diffusion batch of the best predicted structure.

    Metrics include pAE, pLDDT, and pDE, among others.

    Returns:
        dict[str, Any]: A dictionary containing the following keys:
            - pae_idx: The index within the diffusion batch of the structure with the best overall pAE (Predicted Aligned Error)
            - pde_idx: The index within the diffusion batch of the structure with the best overall pDE (Predicted Distance Error)
            - plddt_idx: The index within the diffusion batch of the structure with the best overall pLDDT (Predicted Local Distance
            Difference Test)
            - best_chain_to_all_idx: The index within the diffusion batch of the structure with the best pAE subsampled over any
            pair (i,j) where i == chain or j == chain
            - best_chain_to_self_idx: The index within the diffusion batch of the structure with the best pAE subsampled over any
            pair (i,j) where i == chain and j == chain
            - best_interface_idx: For each interface between two scored PN Units, the index within the diffusion batch of the
            structure with the best mean pAE for all (i,j) where i == interface_chain or j == interface_chain and i != j
            - best_lig_ipae_idx: The index within the diffusion batch for the best pAE subsambled over any pair (i,j)
            where i == chain or j == chain and i != j and i or j is a ligand
    """
    # TODO: Have this function take an `AtomArray` as input so we quickly build masks with much less code
    # TODO: Explore how we can write this function more concisely
    return_dict = {}

    # AF3's ranking metrics work like this, but using ptm instead of ipae:
    scored_chains, interfaces, interface_chains = _select_scored_units(
        interfaces_to_score, pn_units_to_score
    )

    chain_to_all_masks = _create_chain_to_all_masks(chain_iid_token_lvl, scored_chains)
    chain_to_self_masks = _create_chain_to_self_masks(
        chain_iid_token_lvl, scored_chains
    )
    interface_masks, lig_chains = _create_interface_masks(
        chain_iid_token_lvl, interfaces, is_ligand
    )

    # map everything to gpu
    gpu = plddt.device
    chain_to_all_masks = tree.map_structure(
        lambda x: x.to(gpu) if hasattr(x, "cpu") else x, chain_to_all_masks
    )
    chain_to_self_masks = tree.map_structure(
        lambda x: x.to(gpu) if hasattr(x, "cpu") else x, chain_to_self_masks
    )
    interface_masks = tree.map_structure(
        lambda x: x.to(gpu) if hasattr(x, "cpu") else x, interface_masks
    )

    # Reshape logits to B, K, L, NHEAVY
    plddt = (
        plddt.reshape(
            -1,
            plddt.shape[1],
            NHEAVY,
            confidence_loss_cfg.plddt.n_bins,
        )
        .permute(0, 3, 1, 2)
        .float()
    )
    # Reshape the pae and pde logits to B, K, L, L
    pae_logits = pae.permute(0, 3, 1, 2).float()
    pde_logits = pae.permute(0, 3, 1, 2).float()

    pae_logits_unbinned = unbin_logits(
        pae_logits, confidence_loss_cfg.pae.max_value, confidence_loss_cfg.pae.n_bins
    )
    plddt_logits_unbinned = unbin_logits(
        plddt, confidence_loss_cfg.plddt.max_value, confidence_loss_cfg.plddt.n_bins
    )
    pde_logits_unbinned = unbin_logits(
        pde_logits, confidence_loss_cfg.pde.max_value, confidence_loss_cfg.pde.n_bins
    )

    complex_pae = pae_logits_unbinned.mean(dim=(1, 2))
    complex_pde = pde_logits_unbinned.mean(dim=(1, 2))
    complex_plddt = (plddt_logits_unbinned * is_real_atom[..., :NHEAVY]).sum(
        dim=(1, 2)
    ) / is_real_atom[..., :NHEAVY].sum()

    return_dict["pae_idx"] = torch.argmin(complex_pae)
    return_dict["pde_idx"] = torch.argmin(complex_pde)
    return_dict["plddt_idx"] = torch.argmax(complex_plddt)

    chain_to_self_paes = _get_masked_error_per_chain(
        scored_chains, chain_to_self_masks, pae_logits_unbinned
    )
    chain_to_all_paes = _get_masked_error_per_chain(
        scored_chains, chain_to_all_masks, pae_logits_unbinned
    )
    interface_chain_paes = _get_masked_error_per_chain(
        interface_chains, interface_masks, pae_logits_unbinned
    )
    # average over both interfaces
    average_interface_paes = _get_average_error_per_interface(
        interfaces, lig_chains, interface_chain_paes
    )

    return_dict["best_chain_to_all_idx"] = _get_lowest_error_indices(chain_to_all_paes)
    return_dict["best_chain_to_self_idx"] = _get_lowest_error_indices(
        chain_to_self_paes
    )
    return_dict["best_interface_idx"] = _get_lowest_error_indices(
        average_interface_paes
    )
    # for ligands, we don't average the error
    return_dict["best_lig_ipae_idx"] = _get_lowest_error_ligand_indices(
        interface_chain_paes, interfaces, lig_chains
    )
    return return_dict


def annotate_atom_array_b_factor_with_plddt(
    atom_array: AtomArray | AtomArrayStack,
    plddt: torch.Tensor,
    is_real_atom: torch.Tensor,
) -> List[AtomArray]:
    """Annotates the b_factor of an AtomArray with the pLDDT values in the occupancy field.

    Args:
        atom_array: The AtomArray or AtomArrayStack to annotate
        plddt: The pLDDT tensor of shape (B, I, NHEAVY)
        is_real_atom: A mask indicating which atoms are in the structure of shape (I, NHEAVY)

    Returns:
        list[AtomArray]: The annotated list of AtomArrays. We must return a list of AtomArrays
            because the AtomArray class does not support setting different values as annotations
            other than the coordinate feature.
    """
    atom_wise_plddt = plddt[:, is_real_atom[..., :NHEAVY]]
    assert atom_wise_plddt.shape[1] == atom_array.array_length()
    atom_array_list = []
    # bitotite's AtomArray does not support setting different values as annotations other than
    # the coordinate feature, so we convert atom_array to a list of AtomArrays
    if isinstance(atom_array, AtomArrayStack):
        for i, aa in enumerate(atom_array):
            aa.set_annotation("b_factor", atom_wise_plddt[i].cpu().numpy())
            atom_array_list.append(aa)
    else:
        assert atom_wise_plddt.shape[0] == 1
        atom_array.set_annotation("b_factor", atom_wise_plddt[0].cpu().numpy())
        atom_array_list.append(atom_array)

    for aa in atom_array_list:
        assert np.isnan(aa.b_factor).sum() == 0

    return atom_array_list


def _select_scored_units(
    interfaces_to_score: list[tuple], pn_units_to_score: list[tuple]
):
    scored_chains = []
    interfaces = []
    interface_chains = []
    for k in interfaces_to_score:
        interfaces.append(f"{k[0]}-{k[1]}")
        interface_chains.append(k[0])
        interface_chains.append(k[1])
    for k in pn_units_to_score:
        scored_chains.append(k[0])

    return scored_chains, interfaces, interface_chains


def _create_chain_to_all_masks(ch_label, chains_to_score):
    unique_chains = np.unique(ch_label)
    I = len(ch_label)
    chain_to_all_masks = {}
    for chain in unique_chains:
        if chain in chains_to_score:
            indices = torch.from_numpy((ch_label == chain))
            mask = indices.unsqueeze(0) | indices.unsqueeze(1)
            # set the diagonal to false
            mask = mask & ~torch.eye(I, device=mask.device, dtype=torch.bool)
            chain_to_all_masks[chain] = mask
    return chain_to_all_masks


def _create_chain_to_self_masks(ch_label, chains_to_score):
    unique_chains = np.unique(ch_label)
    I = len(ch_label)
    chain_to_self_masks = {}
    for chain in unique_chains:
        if chain in chains_to_score:
            indices = torch.from_numpy((ch_label == chain))
            mask = indices.unsqueeze(0) & indices.unsqueeze(1)
            # set the diagonal to false
            mask = mask & ~torch.eye(I, device=mask.device, dtype=torch.bool)
            chain_to_self_masks[chain] = mask
    return chain_to_self_masks


def _create_interface_masks(ch_label, interfaces, is_ligand):
    interface_masks = {}
    interface_chains = []
    ligand_chains = []
    for interface in interfaces:
        interface_chains.append(interface.split("-")[0])
        interface_chains.append(interface.split("-")[1])
    interface_chains = set(interface_chains)
    for chain in interface_chains:
        chain_indices = torch.from_numpy((ch_label == chain))

        to_self = chain_indices.unsqueeze(0) & chain_indices.unsqueeze(1)
        to_all = chain_indices.unsqueeze(0) | chain_indices.unsqueeze(1)
        interface_mask = to_all & ~to_self
        interface_masks[chain] = interface_mask

        if torch.all(is_ligand[chain_indices]):
            ligand_chains.append(chain)

    return interface_masks, ligand_chains


def _get_masked_error_per_chain(chains, masks, unbinned_logits):
    error = {}
    for chain in chains:
        mask = masks[chain]
        chain_error = compute_mean_over_subsampled_pairs(unbinned_logits, mask)
        error[chain] = chain_error

    return error


def _get_average_error_per_interface(interfaces, lig_chains, interface_errors):
    average_error = {}
    for interface in interfaces:
        chain_a = interface.split("-")[0]
        chain_b = interface.split("-")[1]
        average_error[interface] = (
            interface_errors[chain_a] + interface_errors[chain_b]
        ) / 2

    return average_error


def _get_lowest_error_indices(errors):
    lowest_error_indices = {}
    for k, v in errors.items():
        lowest_error_indices[k] = torch.argmin(v)

    return lowest_error_indices


def _get_lowest_error_ligand_indices(errors, interfaces, lig_chains):
    # ligands are a special case in AF3, where they only consider the ligand chain's error and not the average for the interface
    lowest_error_indices = {}
    for interface in interfaces:
        chain_a = interface.split("-")[0]
        chain_b = interface.split("-")[1]
        if chain_a in lig_chains or chain_b in lig_chains:
            if chain_a in lig_chains:
                lig_chain = chain_a
            elif chain_b in lig_chains:
                lig_chain = chain_b

            lowest_error_indices[interface] = torch.argmin(errors[lig_chain])
        else:
            # assign a random value to avoid key errors downstream; sorting ligand interfaces
            # from other types is handles in analysis
            lowest_error_indices[interface] = 0

    return lowest_error_indices
