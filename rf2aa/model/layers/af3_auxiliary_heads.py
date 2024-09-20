import torch
import torch.nn as nn
import torch.nn.functional as F
import rf2aa
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

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(
                self, 
                S_inputs_I,
                S_trunk_I,
                Z_trunk_II,
                X_pred_L,
                f,
                seq
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
        rep_atoms = find_rep_atoms(seq)
        print("rep_atoms", rep_atoms)
        print("X_pred_L", X_pred_L.shape)
        X_pred_rep_I = X_pred_L.index_select(1, rep_atoms)

        dist = torch.cdist(X_pred_rep_I, X_pred_rep_I)
        # TODO: need to use this function correctly (check what the bins are)
        # bins are 3.375 to 20.375 in 1.75 increments
        dist_one_hot = F.one_hot(discretize_distance_matrix(dist, min_distance=3.375, max_distance=20.875, num_bins=10), num_classes=11)
        Z_trunk_II = Z_trunk_II + self.process_pred_distances(dist_one_hot.float())

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
        #assert pde_logits.is_leaf and pae_logits.is_leaf and plddt_logits.is_leaf and exp_resolved_logits.is_leaf

        return dict(
            pde_logits= pde_logits,
            pae_logits= pae_logits,
            plddt_logits= plddt_logits,
            exp_resolved_logits= exp_resolved_logits
        )
    
    
def find_rep_atoms(seq):
    # compute a I length tensor with the indices in the L representation of the representative atoms for each token
    is_real_atom = ChemData().heavyatom_mask.to(seq.device)[seq]
    #    cbeta for all protein residues except glycine
    #    calpha for glycine
    #    c4 for purines 
    #    c2 for pyrimidines
    seq_is_protein = rf2aa.util.is_protein(seq).to(torch.bool)
    use_cbeta = seq_is_protein & (seq != ChemData().aa2num["GLY"]) & (seq != ChemData().aa2num["UNK"])
    use_calpha = seq_is_protein & ((seq == ChemData().aa2num["GLY"]) | (seq == ChemData().aa2num["UNK"]))
    
    #TODO: need to handle unknown DNA/RNA residues
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
