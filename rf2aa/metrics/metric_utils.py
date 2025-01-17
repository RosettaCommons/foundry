
import numpy as np
import torch
from rf2aa.chemical import ChemicalData as ChemData

def unbin_rf3_metrics (plddt_logits, pae_logits, pde_logits, seq, eps = 1e-4, pae_mask=None, is_real_atom=None):
    #kept for legacy reasons, though this should be fed as tok_idx constructed version in af3/rf3 versions.
    if is_real_atom is None:
        is_real_atom = ChemData().heavyatom_mask.to(seq.device)[seq]

    lddt_bins = torch.linspace(0.02, 1.0, 50, device=plddt_logits.device)
    plddt_unbinned = plddt_logits * is_real_atom[None, None, :, :ChemData().NHEAVY]
    plddt_unbinned = torch.nn.Softmax(dim=1)(plddt_logits)
    plddt_unbinned = plddt_unbinned * lddt_bins[None, :, None, None]
    plddt_unbinned = plddt_unbinned[..., is_real_atom[..., :ChemData().NHEAVY]]
    plddt = plddt_unbinned.sum() / (is_real_atom.sum() + eps)
    
    pae_bins = torch.linspace(0.25, 31.75, 64, device=plddt_logits.device)
    pae_unbinned = torch.nn.Softmax(dim=1)(pae_logits).detach().float()
    pae_unbinned = (pae_unbinned * pae_bins[None, :, None, None]).sum(dim=1)
    if pae_mask is not None:
        pae_unbinned = pae_unbinned * pae_mask[None, :, :]
        pae = pae_unbinned.sum() / (pae_mask.sum() + eps)
    else:
        pae = pae_unbinned.mean()

    pde_unbinned = torch.nn.Softmax(dim=1)(pde_logits).detach().float()
    pde_unbinned = (pde_unbinned * pae_bins[None, :, None, None]).sum(dim=1)
    pde = pde_unbinned.mean()

    return plddt, pae, pde

def get_ipae_metrics_from_binned(pae_logits, same_chain, token_indices):
    """
    Calculate ipae and ipTM for a set of ligand indices

    
    Arguments:
        pae_logits: [1, 64, L, L], binned pae logits
        same_chain: [1, L, L], binned pae logits
        token_indices: [M], indices of tokens you want to calculate ipae and ipTM for
    """
    assert same_chain.shape[0] == 1
    assert pae_logits.shape[0] == 1
    pae_bins = torch.linspace(0.25, 31.75, 64, device=pae_logits.device)
    pae_unbinned = torch.nn.Softmax(dim=1)(pae_logits).detach().float()
    pae_matrix = (pae_unbinned * pae_bins[None, :, None, None]).sum(dim=1)

    pae_matrix = pae_matrix.squeeze(0)
    same_chain = same_chain.squeeze(0)

    L = pae_matrix.shape[-1]

    def f(e_ij, Nres):
        d0 = 1.24 * torch.pow(max(Nres, torch.tensor(19))-15, 1/3) - 1.8
        den = 1 + torch.square(e_ij / d0)
        return 1 / den
    
    ipTM = None
    ipae_list = []
    for i in token_indices:
        ipTM_i = 0
        for j in range(L):
            if same_chain[i,j]: continue
            ipTM_i += f(pae_matrix[i,j], (~same_chain[i, :]).sum())
            ipae_list.append(pae_matrix[i,j])
        ipTM_i /= (~same_chain[i, :]).sum().item()
        if ipTM is None:
            ipTM = ipTM_i
        elif ipTM_i > ipTM:
            ipTM = ipTM_i

    iPAE = sum(ipae_list) / len(ipae_list)
    return ipTM, iPAE

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

    #construct the masks
    unique_chains = np.unique(ch_label)
    ch_masks = {}
    for chain in unique_chains:
        mask = torch.zeros(len(ch_label), len(ch_label), dtype=torch.bool, device=device)
        for i in range(len(ch_label)):
            for j in range(i, len(ch_label)):
                if ch_label[i] == ch_label[j] and ch_label[i] == chain:
                    mask[i,j] = True
                    mask[j,i] = True
        ch_masks[chain] = mask
        chain_pae[chain] = []

    interface_mask = torch.zeros(len(ch_label), len(ch_label), device=device, dtype=torch.bool)
    if len(unique_chains) > 1:
        for i in range(len(ch_label)):
            for j in range(i, len(ch_label)):
                if ch_label[i] != ch_label[j]:
                    interface_mask[i,j] = True
                    interface_mask[j,i] = True
    

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