import torch
import torch.nn as nn
from rf2aa.chemical import ChemicalData as ChemData
from rf2aa.alignment import weighted_rigid_align


def calc_smoothed_lddt_loss(X_gt_L, X_L, crd_mask_I, seq, tok_idx, is_dna, is_rna):
    """
    compute smoothed lddt loss from AF3 paper
    """
    # compute distances between ground truth atoms
    ground_truth_distances = torch.cdist(X_gt_L,X_gt_L)
    # compute distances between predicted atoms
    predicted_distances = torch.cdist(X_L, X_L)
    # compute LDDT score for each pair of distances
    difference_distances = torch.abs(ground_truth_distances - predicted_distances)
    lddt_matrix = torch.zeros_like(difference_distances)
    lddt_matrix = 0.25 * torch.sigmoid(4.0 - difference_distances) + \
                    0.25 * torch.sigmoid(2.0 - difference_distances) + \
                    0.25 * torch.sigmoid(1.0 - difference_distances) + \
                    0.25 * torch.sigmoid(0.5 - difference_distances) 
    # remove unresolved atoms, atoms within same residue
    is_real_atom = ChemData().heavyatom_mask.to(seq.device)[seq]
    is_resolved_atom_L = crd_mask_I[is_real_atom]
    is_unresolved_distance_LL = is_resolved_atom_L[...,None] & is_resolved_atom_L[None,...]
    in_same_residue_LL = tok_idx[:,None] == tok_idx[None,:]

    is_na_L = is_dna[tok_idx] | is_rna[tok_idx]
    is_close_distance = (ground_truth_distances < 30) * is_na_L + (ground_truth_distances < 15) * ~is_na_L
    mask = is_unresolved_distance_LL & ~in_same_residue_LL & is_close_distance[0]
    lddt = torch.div(lddt_matrix[:, mask].sum(dim=(-1)), (mask.sum(dim=(-1,-2)) + 1e-6))
    return 1 - lddt

class DiffusionLoss:
    def __init__(self,
                 weight,
                 sigma_data,
                 alpha_dna,
                 alpha_rna,
                 alpha_ligand,
                 edm_lambda, # Use EDM-style loss weighting
                 se3_invariant_loss,
                 clamp_diffusion_loss
                 ):
        self.sigma_data = sigma_data
        self.alpha_dna = alpha_dna
        self.alpha_rna = alpha_rna
        self.alpha_ligand = alpha_ligand
        self.se3_invariant_loss = se3_invariant_loss
        self.weight = weight
        self.clamp_diffusion_loss = clamp_diffusion_loss
        
        # AF3-style loss weighting
        self.get_lambda = lambda sigma: (sigma**2 + self.sigma_data**2) / (sigma + self.sigma_data)**2
        if edm_lambda:
            # Use EDM-style loss weighting
            self.get_lambda = lambda sigma:  (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data)**2
    
    def __call__(self,
                 f,
                 X_L, # [D, L, 3]
                 X_gt_L, # [D, L, 3]
                 t, # [D]
                 seq, # [I],
                 crd_mask_I, # [I]  # Mask for resolved atoms
                 ):
        D = X_L.shape[0]
        w_L = 1 + (
            f['is_dna']*self.alpha_dna +
            f['is_rna'] * self.alpha_rna + 
            f['is_ligand'] * self.alpha_ligand
        )[f['tok_idx']].to(torch.float)
        
        is_resolved_atom_L = convert_residue_mask_to_allatom_mask(crd_mask_I, seq)
        w_L = w_L * is_resolved_atom_L  
        # Align ground truth onto predictions.
        if self.se3_invariant_loss:
            X_gt_aligned_L = weighted_rigid_align(X_L, X_gt_L, is_resolved_atom_L, w_L.tile(D, 1))
        else:
            X_gt_aligned_L = X_gt_L
        l_mse = 1/3 * torch.div(torch.sum(w_L * torch.sum((X_L-X_gt_aligned_L)**2, dim=-1), dim=-1), torch.sum(is_resolved_atom_L)) # [D]

        assert l_mse.shape == (D,)
        l_diffusion = self.get_lambda(t) * l_mse
        if self.clamp_diffusion_loss:
            l_diffusion = torch.clamp(l_diffusion, max=2)
        
        smoothed_lddt_loss = calc_smoothed_lddt_loss(X_gt_L, X_L, crd_mask_I, seq, f['tok_idx'], f['is_dna'], f['is_rna'])
        loss_dict_batched = {
            'diffusion_loss': l_diffusion,
            'smoothed_lddt_loss': smoothed_lddt_loss,
        }

        # TODO: implement auxiliary losses

        loss_dict = {k:v.mean() for k,v in loss_dict_batched.items()}
        l_total = sum(loss_dict.values())
        loss_dict_batched['total_diffusion_loss'] = l_total
        loss_dict_batched = {k: v.detach() for k,v in loss_dict_batched.items()}
        return self.weight*l_total, loss_dict_batched

class DistogramLoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.cce_loss = nn.CrossEntropyLoss(reduction='none')
        self.weight = weight
        self.eps = 1e-4

    def __call__(
        self,
        distogram_pred, # [I, I, 37]
        X_gt_L, # [D, L, 3]     
        crd_mask_I, # [I]
        seq,
        f 
    ):
        # Convert to I, 36
        I = seq.shape[0]
        is_real_atom = ChemData().heavyatom_mask.to(X_gt_L.device)[seq]
        X_gt_I = torch.zeros((seq.shape[0], ChemData().NTOTAL, 3), device=X_gt_L.device)
        X_gt_I[is_real_atom] = X_gt_L[0]
        #    cbeta for all protein residues except glycine
        #    calpha for glycine
        #    c4 for purines 
        #   c2 for pyrimidines
        seq_is_protein = f["is_protein"].to(torch.bool)
        use_cbeta = seq_is_protein & (seq != ChemData().aa2num["GLY"])
        use_calpha = seq_is_protein & (seq == ChemData().aa2num["GLY"])
        use_c4 = (seq == ChemData().aa2num[" DA"]) | (seq == ChemData().aa2num[" DG"]) | (seq == ChemData().aa2num[" RA"]) | (seq == ChemData().aa2num[" RG"])
        use_c2 = (seq == ChemData().aa2num[" DC"]) | (seq == ChemData().aa2num[" DT"]) | (seq == ChemData().aa2num[" RC"]) | (seq == ChemData().aa2num[" RU"])
        idx_to_use = torch.ones_like(seq) 
        idx_to_use[use_cbeta] = 5 # cbeta
        idx_to_use[use_calpha] = 1 # calpha
        idx_to_use[use_c4] = 8 # c4
        idx_to_use[use_c2] = 2 # c2

        dist_node = X_gt_I[torch.arange(seq.shape[0]), idx_to_use]
        crd_mask_I = crd_mask_I[torch.arange(seq.shape[0]), idx_to_use]

        crd_mask_II = crd_mask_I.unsqueeze(-1) * crd_mask_I.unsqueeze(-2)
        dist = torch.cdist(dist_node, dist_node, compute_mode="donot_use_mm_for_euclid_dist")
        from rf2aa.data.dataloader_adaptor_af3 import discretize_distance_matrix
        distogram_target = discretize_distance_matrix(dist, num_bins=64, min_distance=2, max_distance=22)
        cce_loss =  self.cce_loss(distogram_pred.permute(-1,-2,-3)[None], distogram_target[None])
        cce_loss = torch.sum(cce_loss[..., crd_mask_II])/(torch.sum(crd_mask_II) + self.eps)
        loss_dict = {"distogram_loss": cce_loss.detach()} 
        return self.weight * cce_loss, loss_dict

class Loss(nn.Module):
    def __init__(self,
                 diffusion_loss,
                 distogram_loss,
                 ):
        super().__init__()
        self.diffusion_loss = DiffusionLoss(**diffusion_loss)
        self.distogram_loss = DistogramLoss(**distogram_loss)

    def forward(self,
                network_input,
                network_output,
                loss_input,
                ):
        loss_dict = {}
        diffusion_loss, diffusion_loss_dict = self.diffusion_loss(
                                            network_input["f"], 
                                             network_output["X_L"], 
                                             loss_input["X_gt_L"], 
                                             network_input["t"], 
                                             loss_input["seq"], 
                                             loss_input["crd_mask_I"]
                                             )
        
        distogram_loss, distogram_loss_dict = self.distogram_loss(
                                            network_output["distogram"], 
                                             loss_input["X_gt_L"], 
                                             loss_input["crd_mask_I"],
                                             loss_input["seq"],
                                             network_input["f"]
                                             )
        loss_dict.update(diffusion_loss_dict)
        loss_dict.update(distogram_loss_dict)
        return diffusion_loss + distogram_loss, loss_dict 
                

def convert_residue_mask_to_allatom_mask(crd_mask_I, seq):
    """
    Converts a residue mask to an atom mask. The atom mask is True if any atom in the residue is True.
    """
    is_real_atom = ChemData().heavyatom_mask.to(seq.device)[seq]
    is_resolved_atom_L = crd_mask_I[is_real_atom]
    return is_resolved_atom_L

