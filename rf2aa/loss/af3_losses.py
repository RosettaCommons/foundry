import hydra

import torch
import torch.nn as nn
from rf2aa.chemical import ChemicalData as ChemData
from rf2aa.alignment import weighted_rigid_align
from rf2aa.model.af3_with_rollout import ConfidenceLoss
from rf2aa.training.checkpoint import activation_checkpointing



class Loss(nn.Module):

    def __init__(self, **losses):
        super().__init__()
        self.to_compute = []
        for loss_name, loss in losses.items():
            loss_fn = hydra.utils.instantiate(loss)
            print(f"Adding loss {loss_name} to the loss function")
            self.to_compute.append(loss_fn)
        
    def forward(
        self,
        network_input,
        network_output,
        loss_input,
    ):
        loss_dict = {}
        loss = 0
        for loss_fn in self.to_compute:
            loss_, loss_dict_ = loss_fn(network_input, network_output, loss_input)
            loss += loss_
            loss_dict.update(loss_dict_)
        return loss, loss_dict

class DiffusionLoss(nn.Module):

    def __init__(
        self,
        weight,
        sigma_data,
        alpha_dna,
        alpha_rna,
        alpha_ligand,
        edm_lambda,
        se3_invariant_loss,
        clamp_diffusion_loss,
    ):
        super().__init__()
        self.weight = weight
        self.sigma_data = sigma_data
        self.alpha_dna = alpha_dna
        self.alpha_rna = alpha_rna
        self.alpha_ligand = alpha_ligand
        if edm_lambda:
            # original EDM scaling factor
            self.get_lambda = lambda sigma: (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        else:
            # AF3 uses a weird scaling factor for their loss
            self.get_lambda = lambda sigma: (sigma ** 2 + self.sigma_data ** 2) / (sigma + self.sigma_data) ** 2
        self.se3_invariant_loss = se3_invariant_loss
        self.clamp_diffusion_loss = clamp_diffusion_loss

    def forward(self, network_input, network_output, loss_input):
        X_L = network_output["X_L"] # D, L, 3
        D = X_L.shape[0]
        X_gt_L = loss_input["X_gt_L"]
        crd_mask_L = loss_input["crd_mask_L"]
        tok_idx = network_input["f"]["atom_to_token_map"]
        t = network_input["t"] # (D,)
        
        w_L = 1 + (
            network_input["f"]["is_dna"] * self.alpha_dna +
            network_input["f"]["is_rna"] * self.alpha_rna +
            network_input["f"]["is_ligand"] * self.alpha_ligand
        )[tok_idx].to(torch.float)
        w_L = w_L[None].expand(D,-1) * crd_mask_L
        
        if self.se3_invariant_loss:
            # check if this is correct
            X_gt_aligned_L = weighted_rigid_align(X_L, X_gt_L, crd_mask_L[0], w_L)
        else:
            X_gt_aligned_L = X_gt_L
        #l_mse = 1/3 * torch.div(
            #torch.sum(w_L.masked_select(crd_mask_L) * torch.sum((X_L.masked_select(crd_mask_L[...,None].expand(-1,-1,3)) - X_gt_aligned_L.masked_select(crd_mask_L[...,None].expand(-1,-1,3))) ** 2, dim=-1), dim=-1),
            #(torch.sum(crd_mask_L[0]) + 1e-4)
        #)
        X_gt_aligned_L = torch.nan_to_num(X_gt_aligned_L)
        l_mse = 1/3 * torch.div(
            torch.sum(w_L * torch.sum((X_L - X_gt_aligned_L) ** 2, dim=-1), dim=-1),
            torch.sum(crd_mask_L[0]) + 1e-4
        ) # w_L is already updated by the mask

        assert l_mse.shape == (D,)
        l_diffusion = self.get_lambda(t) * l_mse
        l_diffusion = torch.clamp(l_diffusion, max=2) if self.clamp_diffusion_loss else l_diffusion

        l_diffusion_total = torch.mean(l_diffusion)
        # smoothed lddt loss
        smoothed_lddt_loss_ = smoothed_lddt_loss(
            X_L, 
            X_gt_L, 
            crd_mask_L, 
            network_input["f"]["is_dna"], 
            network_input["f"]["is_rna"], 
            tok_idx,
            #tag=network_input["id"]
        )
        l_diffusion_total += smoothed_lddt_loss_.mean()
        loss_dict = {
            "diffusion_loss": l_diffusion.detach(),
            "smoothed_lddt_loss": smoothed_lddt_loss_.detach(),
            "t": t.detach(),
        }
    
        return self.weight*l_diffusion_total, loss_dict

def smoothed_lddt_loss_naive(X_L, X_gt_L_aligned, crd_mask_L, is_dna, is_rna, tok_idx):
    """
    computes lddt with a sigmoid within each bucket to smooth the loss
    X_L: (D, L, 3)
    X_gt_L_aligned: (D, L, 3)
    crd_mask_L: (D, L)
    is_dna: (L,)
    is_rna: (L,)
    tok_idx: (L,)

    returns: (D,)
    """
    predicted_distances = torch.cdist(X_L, X_L)
    ground_truth_distances = torch.cdist(X_gt_L_aligned, X_gt_L_aligned)
    ground_truth_distances[ground_truth_distances.isnan()] = 9999.0
    difference_distances = torch.abs(ground_truth_distances - predicted_distances)
    lddt_matrix = torch.zeros_like(difference_distances)
    lddt_matrix = 0.25 * torch.sigmoid(4.0 - difference_distances) + \
                  0.25 * torch.sigmoid(2.0 - difference_distances) + \
                  0.25 * torch.sigmoid(1.0 - difference_distances) + \
                  0.25 * torch.sigmoid(0.5 - difference_distances)
    # remove unresolved atoms, atoms within same residue
    in_same_residue_LL = tok_idx[:, None] == tok_idx[None, :]
    is_na_L = is_dna[tok_idx] | is_rna[tok_idx]
    is_close_distance = (ground_truth_distances < 30) * is_na_L + (ground_truth_distances < 15) * ~is_na_L
    mask = crd_mask_L[0] & ~in_same_residue_LL & is_close_distance[0]
    lddt = (lddt_matrix * mask[None]).sum(dim=(-1, -2)) / (mask.sum(dim=(-1, -2)) + 1e-6)
    return 1 - lddt

def smoothed_lddt_loss(X_L, X_gt_L, crd_mask_L, is_dna, is_rna, tok_idx, eps=1e-6):

    @activation_checkpointing
    def _dolddt(X_L, X_gt_L, crd_mask_L, is_dna, is_rna, tok_idx, eps, use_amp=True):
        B,L = X_L.shape[:2]
        first_index,second_index = torch.triu_indices(L,L,1, device=X_L.device)
    
        if use_amp:
            X_L = X_L.to(torch.bfloat16)
            X_gt_L = X_gt_L.to(torch.bfloat16)
            
        # compute the unique distances between all pairs of atoms
        X_gt_L = X_gt_L.nan_to_num()

        # only use native 1 (assumes dist map identical btwn all copies)
        ground_truth_distances = torch.linalg.norm(X_gt_L[0:1,first_index]-X_gt_L[0:1,second_index], dim=-1)

        with torch.amp.autocast('cuda',enabled=use_amp, dtype=torch.bfloat16):
            # only score pairs that are close enough in the ground truth
            is_na_L = is_dna[tok_idx][first_index] | is_rna[tok_idx][first_index]
            pair_mask = torch.logical_and(
                ground_truth_distances>0,
                ground_truth_distances<torch.where(is_na_L, 30.0, 15.0)
            )
            del is_na_L

            lddtO = torch.sum(pair_mask)
    
            # only score pairs that are resolved in the ground truth
            pair_mask *= (crd_mask_L[0:1,first_index] * crd_mask_L[0:1,second_index])
            # don't score pairs that are in the same token
            pair_mask *= (tok_idx[None,first_index] != tok_idx[None,second_index])
    
            _,valid_pairs = pair_mask.nonzero(as_tuple=True)
            pair_mask = pair_mask[:,valid_pairs].to(X_L.dtype)
            ground_truth_distances = ground_truth_distances[:,valid_pairs]    
            first_index,second_index = first_index[valid_pairs],second_index[valid_pairs]

            predicted_distances = torch.linalg.norm(X_L[:,first_index]-X_L[:,second_index], dim=-1)
        
            delta_distances = torch.abs(predicted_distances-ground_truth_distances+eps)
            del predicted_distances, ground_truth_distances

            lddt = 0.25*(
                torch.sum( torch.sigmoid( 0.5 - delta_distances )*pair_mask, dim=(1) )
                +torch.sum( torch.sigmoid( 1.0 - delta_distances )*pair_mask, dim=(1) )
                +torch.sum( torch.sigmoid( 2.0 - delta_distances )*pair_mask, dim=(1) )
                +torch.sum( torch.sigmoid( 4.0 - delta_distances )*pair_mask, dim=(1) )
            ) / (torch.sum( pair_mask, dim=(1) ) + eps)
        return 1-lddt

    return _dolddt(X_L, X_gt_L, crd_mask_L, is_dna, is_rna, tok_idx, eps)



def distogram_loss(pred_distogram, X_rep_atoms_I, crd_mask_rep_atoms_I, cce_loss, min_distance=2, max_distance=22, bins=64):
    """
    computes distogram loss 
    """ 
    distance_map = torch.cdist(X_rep_atoms_I, X_rep_atoms_I)
    distance_map[distance_map.isnan()] = 9999.0
    bins = torch.linspace(min_distance, max_distance, bins).to(X_rep_atoms_I.device)
    binned_distances = torch.bucketize(distance_map, bins)
    crd_mask_rep_atom_II = crd_mask_rep_atoms_I.unsqueeze(-1) * crd_mask_rep_atoms_I.unsqueeze(-2)
    distogram_cce = cce_loss(pred_distogram.permute(-1,-2,-3)[None], binned_distances[None])
    return distogram_cce[..., crd_mask_rep_atom_II].sum() / (crd_mask_rep_atom_II.sum() + 1e-4)

class DistogramLoss(nn.Module):

    def __init__(self, weight):
        super().__init__()
        self.weight = weight
        self.cce_loss = nn.CrossEntropyLoss(reduction='none')
        self.eps = 1e-4

    def forward(
        self, 
        network_input,
        network_output,
        loss_input
    ):
        pred_distogram = network_output["distogram"]
        X_rep_atoms_I = loss_input["X_rep_atoms_I"]
        crd_mask_rep_atoms_I = loss_input["crd_mask_rep_atoms_I"]
        loss = distogram_loss(pred_distogram, X_rep_atoms_I, crd_mask_rep_atoms_I, self.cce_loss)
        return self.weight * loss, {"distogram_loss": loss.detach()} 

class NullLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self, 
        network_input,
        network_output,
        loss_input
    ):
        loss = 0
        for key, val in network_output.items():
            val[val.isnan()] = 0
            loss += torch.sum(val) * 0
        
        return loss, {}
