from itertools import combinations

import numpy as np
import torch
from rf2aa.chemical import ChemicalData as ChemData


def unbin_rf3_metrics(plddt_logits, pae_logits, pde_logits, is_real_atom, plddt_config, pae_config, pde_config, eps=1e-6, pae_mask=None, pde_mask=None):
    """
    Calculate unbinned metrics for batch size > 1.

    Arguments:
        plddt_logits: [B, n_bins, L, 23], binned logits for plddt
        pae_logits: [B, n_bins, L, L], binned logits for pae
        pde_logits: [B, n_bins, L, L], binned logits for pde
        is_real_atom: [B, L, 36] or [L, 36], mask for real atoms in atom-36 representation
        plddt_config: dict, config for plddt bin settings
        pae_config: dict, config for pae_bin settings
        pde_config: dict, config for pde_bin settings
        eps: float, small value to avoid division by zero
        pae_mask: [L, L], [B, L, L], or None, mask for pae
        pde_mas: [L, L,], [B, L, L], or None, mask for pde
    Returns:
        plddt, pae, pde: [B], per structure metrics
    """
    #add a batch dimension if not present so we can use for both batched and unbatched masks
    if len(is_real_atom.shape) == 2:
        is_real_atom = is_real_atom.unsqueeze(0)
    plddt_unbinned = unbin_logits(plddt_logits, plddt_config.max_value, plddt_config.n_bins)
    plddt_unbinned = plddt_unbinned * is_real_atom[..., :ChemData().NHEAVY]
    plddt = plddt_unbinned.sum(dim=(1,2)) / is_real_atom.sum(dim=(1,2)) + eps

    pae_unbinned = unbin_logits(pae_logits, pae_config.max_value, pae_config.n_bins)
    if pae_mask is not None:
        if len(pae_mask.shape) == 2:
            pae_mask = pae_mask.unsqueeze(0)
        pae_unbinned = pae_unbinned * pae_mask
        pae = pae_unbinned.sum(dim=(1,2)) / (pae_mask.sum(dim=(1,2)) + eps)
    else:
        pae = pae_unbinned.mean(dim=(1,2))
    
    pde_unbinned = unbin_logits(pde_logits, pde_config.max_value, pde_config.n_bins)
    if pde_mask is not None:
        if len(pde_mask.shape) == 2:
            pde_mask = pde_mask.unsqueeze(0)
        pde_unbinned = pde_unbinned * pde_mask
        pde = pde_unbinned.sum(dim=(1,2)) / (pde_mask.sum(dim=(1,2)) + eps)
    else:
        pde = pde_unbinned.mean(dim=(1,2))

    return plddt, pae, pde

def find_bin_midpoints(max_distance, num_bins, device="cpu"):
    """
    Find the bin midpoints for a given binning scheme. Used to find expectation of values when converting binned 
    predictions to unbinned predictions. Assumes the minimum of the schema is 0. 
    Args:
        max_distance: float, maximum distance
        num_bins: int, number of bins
    Returns:
        pae_midpoints: [num_bins], bin midpoints
    """
    bin_size = max_distance / num_bins
    bins = torch.linspace(bin_size, max_distance - bin_size, num_bins-1, device=device)
    midpoints = (bins[1:] + bins[:-1]) / 2
    midpoints = torch.cat([(bins[0]-bin_size/2)[None], midpoints, bins[-1:]+bin_size/2])

    return midpoints

def unbin_logits(logits, max_distance, num_bins):
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


def write_confidence_metrics(outputs, path, device="cpu"):
    """
    Write the confidence metrics to the end of a score file.
    """
    plddt_logit_stack = outputs["confidence"]["plddt_logits"]
    pae_logits = outputs["confidence"]["pae_logits"]
    pde_logits = outputs["confidence"]["pde_logits"]
    ch_label = outputs["confidence"]["chain_iid_token_lvl"]

    #instantiate the metrics
    complex_plddt = []
    complex_pae = []
    complex_pde = []
    interface_pae = []
    chain_pae = {}

    # Construct the masks
    unique_chains = np.unique(ch_label)
    ch_masks = {}
    for chain in unique_chains:
        indices = torch.from_numpy((ch_label == chain))
        mask = torch.outer(indices, indices).to(dtype=torch.bool, device=device)
        ch_masks[chain] = mask
        chain_pae[chain] = []

    # Construct the interface mask
    if len(unique_chains) > 1:
        interface_mask = torch.from_numpy(ch_label[None,:] != ch_label[:,None]).to(dtype=torch.bool, device=device)
    else:
        interface_mask = torch.zeros(len(ch_label), len(ch_label), device=device, dtype=torch.bool)

    for i in range(plddt_logit_stack.shape[0]):
        plddt_logits = plddt_logit_stack[i].unsqueeze(0)
        plddt_logits = plddt_logits.reshape(plddt_logits.shape[0], -1, plddt_logits.shape[1], ChemData().NHEAVY)

        plddt, pae, pde = unbin_rf3_metrics(plddt_logits.float(), pae_logits[i].unsqueeze(0).permute(0,3,1,2).float(), pde_logits[i].unsqueeze(0).permute(0,3,1,2).float(), outputs["confidence"]["rf2aa_seq"].to(device), is_real_atom=outputs["confidence"]['is_real_atom'].to(device))
        complex_plddt.append(plddt)
        complex_pae.append(pae)
        complex_pde.append(pde)

        _, i_pae, _ = unbin_rf3_metrics(plddt_logits.float(), pae_logits[i].unsqueeze(0).permute(0,3,1,2).float(), pde_logits[i].unsqueeze(0).permute(0,3,1,2).float(), outputs["confidence"]["rf2aa_seq"].to(device), pae_mask=interface_mask, is_real_atom=outputs["confidence"]['is_real_atom'].to(device))
        interface_pae.append(i_pae)

        for chain, chain_mask in ch_masks.items():
            _, ch_pae, _ = unbin_rf3_metrics(plddt_logits.float(), pae_logits[i].unsqueeze(0).permute(0,3,1,2).float(), pde_logits[i].unsqueeze(0).permute(0,3,1,2).float(), outputs["confidence"]["rf2aa_seq"].to(device), pae_mask=chain_mask, is_real_atom=outputs["confidence"]['is_real_atom'].to(device))
            chain_pae[chain].append(ch_pae)

    header = "STRUCTURE\t"
    complex_pae_line = "COMPLEX PAE\t"
    complex_pde_line = "COMPLEX PDE\t"  
    complex_plddt_line = "COMPLEX PLDDT\t"
    interface_pae_line = "INTERFACE PAE\t"
    chain_pae_lines = {}
    for chain in unique_chains:
        chain_pae_lines[chain] = f"CHAIN {chain} PAE\t"
    for i in range(len(complex_plddt)):
        header += f"{i}\t"
        complex_pae_line += f"{complex_pae[i]:.4f}\t"
        complex_pde_line += f"{complex_pde[i]:.4f}\t"
        complex_plddt_line += f"{complex_plddt[i]:.4f}\t"
        interface_pae_line += f"{interface_pae[i]:.4f}\t"
        for chain in unique_chains:
            chain_pae_lines[chain] += f"{chain_pae[chain][i]:.4f}\t"

    with open(path, "w") as f:
        f.write("\n")
        f.write(header + "\n")
        f.write(complex_pae_line + "\n")
        f.write(complex_pde_line + "\n")
        f.write(complex_plddt_line + "\n")
        f.write(interface_pae_line + "\n")
        for chain in unique_chains:
            f.write(chain_pae_lines[chain] + "\n")

def create_chainwise_masks_1d(ch_label, device="cpu"):
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
        indices = torch.from_numpy((ch_label == chain)).to(dtype=torch.bool, device=device)
        ch_masks[chain] = indices
    return ch_masks

def create_chainwise_masks_2d(ch_label, device="cpu"):
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

def create_interface_masks_2d(ch_label, device="cpu"):
    """
    Create interface masks for a set of chain labels
    """
    unique_chains = np.unique(ch_label)
    pairs_to_score = {}
    for chain_i, chain_j in combinations(unique_chains, 2):
        chain_i_indices = torch.from_numpy((ch_label == chain_i))
        chain_j_indices = torch.from_numpy((ch_label == chain_j))
        to_be_scored = \
            torch.outer(chain_i_indices, chain_j_indices).to(dtype=torch.bool, device=device) \
                + torch.outer(chain_j_indices, chain_i_indices).to(dtype=torch.bool, device=device)
        pairs_to_score[(chain_i, chain_j)] = to_be_scored
    return pairs_to_score