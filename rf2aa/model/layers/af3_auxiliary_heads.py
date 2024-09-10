import torch
import torch.nn as nn
import torch.nn.functional as F
from rf2aa.chemical import ChemicalData as ChemData
from rf2aa.data.dataloader_adaptor_af3 import discretize_distance_matrix
from rf2aa.model.AF3_structure import linearNoBias, PairformerBlock
#from rf2aa.util import calc_rmsd


class ConfidenceHead(nn.Module):
    """ Algorithm 31 """
    def __init__(self, 
                c_s,
                c_z,
                n_pairformer_layers,
                pairformer,
                n_bins_pae,
                n_bins_pde,
                n_bins_plddt,
                n_bins_exp_resolved,
                 ):
        super(ConfidenceHead, self).__init__()
        self.process_s_inputs_right = linearNoBias(449, c_z)
        self.process_s_inputs_left = linearNoBias(449, c_z)
        self.process_pred_distances = linearNoBias(11, c_z)

        self.pairformer = nn.ModuleList([PairformerBlock(c_s=c_s, c_z=c_z,**pairformer) for _ in range(n_pairformer_layers)])

        self.predict_pae = linearNoBias(c_z, n_bins_pae)
        self.predict_pde = linearNoBias(c_z, n_bins_pde)
        self.predict_plddt = linearNoBias(c_s, ChemData().NHEAVY* n_bins_plddt)
        self.predict_exp_resolved = linearNoBias(c_s, ChemData().NHEAVY * n_bins_exp_resolved)


    def forward(
                self, 
                S_inputs_I,
                S_trunk_I,
                Z_trunk_II,
                X_pred_L,
                f
                ):
        # stopgrad on S_trunk_I, Z_trunk_II, X_pred_L but not S_inputs_I (4.3.5)
        S_trunk_I = S_trunk_I.detach()
        Z_trunk_II = Z_trunk_II.detach()
        X_pred_L = X_pred_L.detach()

        # embed S_inputs_I twice
        S_inputs_I_right = self.process_s_inputs_right(S_inputs_I)
        S_inputs_I_left = self.process_s_inputs_left(S_inputs_I)
        # add outer product of two linear embeddings of S_inputs_I  to Z_II
        # TODO: check the unsqueezed dimension is the correct one
        Z_trunk_II = Z_trunk_II + S_inputs_I_right.unsqueeze(-2) * S_inputs_I_left.unsqueeze(-3)

        # embed distances of representative atom from every token
        # in the pair representation
        rep_atoms = self.find_rep_atoms(f)
        X_pred_rep_I = X_pred_L[rep_atoms]
        dist = torch.cdist(X_pred_rep_I, X_pred_rep_I)
        # TODO: need to use this function correctly (check what the bins are)
        # bins are 3.375 to 20.375 in 1.75 increments
        dist_one_hot = F.one_hot(discretize_distance_matrix(dist, min_distance=3.375, max_distance=20.875, num_bins=10), num_classes=11)
        Z_trunk_II = Z_trunk_II + dist_one_hot

        # process with pairformer stack
        S_trunk_residual_I = S_trunk_I.clone()
        Z_trunk_residual_II = Z_trunk_II.clone()
        for n in range(len(self.pairformer)):
            S_trunk_I, Z_trunk_II = self.pairformer[n](S_trunk_I, Z_trunk_II)
        S_trunk_I = S_trunk_residual_I + S_trunk_I
        Z_trunk_II = Z_trunk_residual_II + Z_trunk_II 

        # linearly project for each prediction task
        pde_logits = self.predict_pde(Z_trunk_II + Z_trunk_II.transpose(-2,-3)) #BUG: needs to be symmetrized correctly

        pae_logits = self.predict_pae(Z_trunk_II)

        plddt_logits = self.predict_plddt(S_trunk_I)
        exp_resolved_logits = self.predict_exp_resolved(S_trunk_I)

        return dict(
            pde_logits= pde_logits,
            pae_logits= pae_logits,
            plddt_logits= plddt_logits,
            exp_resolved_logits= exp_resolved_logits
        )
    
    def find_rep_atoms(self, f):
        seq = f["msa"][0].argmax(-1) # get the sequence of the first sequence in the MSA
        # compute a I length tensor with the indices in the L representation of the representative atoms for each token
        is_real_atom = ChemData().heavyatom_mask.to(seq.device)[seq]
        #    cbeta for all protein residues except glycine
        #    calpha for glycine
        #    c4 for purines 
        #    c2 for pyrimidines
        seq_is_protein = f["is_protein"].to(torch.bool)
        use_cbeta = seq_is_protein & (seq != ChemData().aa2num["GLY"]) & (seq != ChemData().aa2num["UNK"])
        use_calpha = seq_is_protein & ((seq == ChemData().aa2num["GLY"]) | (seq == ChemData().aa2num["UNK"]))
        use_c4 = (seq == ChemData().aa2num[" DA"]) | (seq == ChemData().aa2num[" DG"]) | (seq == ChemData().aa2num[" RA"]) | (seq == ChemData().aa2num[" RG"])
        use_c2 = (seq == ChemData().aa2num[" DC"]) | (seq == ChemData().aa2num[" DT"]) | (seq == ChemData().aa2num[" RC"]) | (seq == ChemData().aa2num[" RU"])
        idx_to_use = torch.ones_like(seq) 
        idx_to_use[use_cbeta] = 4 # cbeta
        idx_to_use[use_calpha] = 1 # calpha
        idx_to_use[use_c4] = 8 # c4
        idx_to_use[use_c2] = 2 # c2
        
        indices = torch.arange(is_real_atom.sum()).to(is_real_atom.device)
        absolute_indices = torch.zeros_like(is_real_atom.long()) 
        absolute_indices[is_real_atom] = indices
        rep_atoms = torch.gather(absolute_indices, 1, idx_to_use[:,None])
        return rep_atoms[:, 0]


class FindOptimalPermutation(nn.Module):

    def __init__(self):
        super(FindOptimalPermutation, self).__init__()
    
    def forward(
                self,
                X_gt_symm_I, 
                X_pred_L, 
                crd_mask_I,
                f
    ):
        # convert X_pred_L to X_pred_I
        D, L, _ = X_pred_L.shape
        I = f["seq"].shape[0]
        # create container for X_pred_I
        X_pred_I = torch.zeros(I, ChemData().NTOTAL, 3, device=X_pred_L.device)
        # use CHemData().heavyatom_mask to get indices of real atoms
        is_real_atom = ChemData().heavyatom_mask.to(f["seq"].device)[f["seq"]]
        X_pred_I[is_real_atom] = X_pred_L

        # get the indices of each chain in the input
        first_chain_indices = self.find_first_chain_indices(f)

        # get chain 0 in predicted and true

        rms, U, cP, cT = calc_rmsd(X_gt_symm_I[first_chain_indices], X_pred_I[first_chain_indices], crd_mask_I[first_chain_indices])
        
        # perform anchor alignment
        X_pred_anchor_I = torch.matmul(X_pred_I-cP, U)+cT
        # iterate through the rest of the chains
        # for each chain, find the optimal permutation
        # and reorder the indices
        # for chain in chains:
        #     for chain2 in chains:
        #         if entity id of chain2 and chain are the same
        #             measure the rmsd between the two chains
        #      find the minimum rmsd assignment possible

        pass

    def find_first_chain_indices(self, f):
        """
        given an input entity_id, return a list of lists with indices that represent 
        identical chains
        """
        # asym_id is unique for each chain
        asym_id = f["asym_id"]

        # take first chain and align it as a reference point 
        return torch.where(asym_id == asym_id[0])[0] 
    
    def find_chain_breaks(self, f):
        """
        given an input entity_id, return a list of lists with indices that represent 
        chain breaks
        """
        # asym_id is unique for each chain
        asym_id = f["asym_id"]
        # find where the asym_id changes
        #TODO: check if this is correct
        chain_ends = torch.where(asym_id[:-1] != asym_id[1:])[0] + 1
        #reorient as a list of indices
        chain_ends = chain_ends.nonzero().squeeze().tolist()
        # add the last index
        chain_ends.append(len(asym_id))
        return chain_ends

class PredictedConfidenceLoss(nn.Module):

    def __init__(
            self,
            weight_pae,
            weight_loss
            ):
        super(PredictedConfidenceLoss, self).__init__()
        self.resolve_permutation_symmetry = FindOptimalPermutation()
        self.weight_pae = weight_pae
        self.weight_loss = weight_loss

    def forward(
        self,   
        predicted_confidence,
        X_pred_L, 
        X_gt_symm_I,
        crd_mask_I,
        f
    ):
        X_gt_I = self.resolve_permutation_symmetry(
            X_gt_symm_I, 
            X_pred_L, 
            crd_mask_I, 
            f
        )        
    