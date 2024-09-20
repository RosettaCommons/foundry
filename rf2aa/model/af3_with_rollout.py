import torch
import torch.nn as nn

import rf2aa
from rf2aa.chemical import ChemicalData as ChemData
from rf2aa.loss.loss import resolve_equiv_natives, resolve_equiv_natives_asmb
from rf2aa.model.layers.af3_auxiliary_heads import find_rep_atoms


class AF3_with_rollout(nn.Module):
    """ Implements rollout on each training step """
    def __init__(self, model, confidence, sampler):
        super(AF3_with_rollout, self).__init__()
        self.model = model
        self.confidence = confidence
        self.sampler = sampler
        self.find_optimal_permutation = FindOptimalPermutation()
        self.num_timesteps = 20

    def forward(
            self, 
            input, 
            n_cycle, 
            X_gt_I_symm, 
            crd_mask_I, 
            seq, 
            no_sync
        ):
        # first do forward pass
        with torch.enable_grad():
            trunk_output = self.model.trunk_forward(input, n_cycle, no_sync)
        # save embeddings
        # do rollout conditioned on embeddings
        # with nograd? 
        with torch.no_grad():
            noise_schedule = self.sampler.construct_noise_schedule(self.num_timesteps, 0, 1).to(input["f"]["msa"].device)
            diffusion_output = self.sampler.sample_diffusion(
                input["f"],
                trunk_output["S_inputs_I"].clone().detach(),
                trunk_output["S_I"].clone().detach(),
                trunk_output["Z_II"].clone().detach(), 
                noise_schedule,
                step_scale=1.5, # int he paper it says they changed this during the rollout
                D=1
            )

            assert diffusion_output["X_L"].shape[0] == 1
            
            # find ground truth permutation
            X_gt_L, X_exists_L = self.find_optimal_permutation(
                diffusion_output["X_L"].clone().detach(),
                X_gt_I_symm,
                crd_mask_I,
                seq,
                input["f"]
            )

        trunk_output = self.model.post_recycle(
            trunk_output["S_inputs_I"],
            trunk_output["S_init_I"],
            trunk_output["Z_init_II"],
            trunk_output["S_I"],
            trunk_output["Z_II"],
            input["f"],
            input["X_noisy_L"],
            input["t"]
        )
        # TODO: run diffusion training by noising the ground truth permutation closest to the rollout
        # run confidence model on embeddings and output structure
        with torch.enable_grad():
            confidence = self.confidence(
                trunk_output["S_inputs_I"],
                trunk_output["S_I"],
                trunk_output["Z_II"],
                diffusion_output["X_L"],
                input["f"],
                seq
            )

        ##assert trunk_output["X_L"].is_leaf == True
        #assert trunk_output["distogram"].is_leaf == True
        #assert confidence["plddt_logits"].is_leaf == True
        #assert confidence["pae_logits"].is_leaf == True
        #assert confidence["pde_logits"].is_leaf == True
        #assert confidence["exp_resolved_logits"].is_leaf == True
        ## return output
        return dict(
            X_gt_L=X_gt_L.detach(),
            X_exists_L=X_exists_L.detach(),
            X_L=trunk_output["X_L"],
            X_pred_rollout_L=diffusion_output["X_L"],
            plddt=confidence["plddt_logits"],
            pae=confidence["pae_logits"],
            pde=confidence["pde_logits"],
            exp_resolved=confidence["exp_resolved_logits"],
            distogram=trunk_output["distogram"],
        )

class ConfidenceLoss(nn.Module):

    def __init__(self, 
                plddt,
                pae,
                pde,
                exp_resolved,
                weight=1,
        ):
        super(ConfidenceLoss, self).__init__()
        self.weight = weight
        self.plddt = plddt
        self.pae = pae
        self.pde = pde
        self.exp_resolved = exp_resolved
        self.cce = nn.CrossEntropyLoss(reduction="none")
        self.eps = 1e-6
    
    def forward(
        self,
        network_input, 
        network_output,
        loss_input,
    ):
        I = loss_input["seq"].shape[0]
        X_gt_L = network_output["X_gt_L"]
        X_exists_L = network_output["X_exists_L"]
        #X_exists_L = loss_input["crd_mask_I"][ChemData().heavyatom_mask.to(loss_input["seq"].device)[loss_input["seq"]]][None]
        X_pred_L = network_output["X_pred_rollout_L"]

        true_lddt_binned, is_resolved_I = self.calc_lddt(X_pred_L, X_gt_L, X_exists_L, loss_input["seq"])
        plddt_loss = self.cce(
            network_output["plddt"].reshape(I, self.plddt.n_bins, ChemData().NHEAVY), 
            true_lddt_binned[:, :ChemData().NHEAVY].long()
            ) * is_resolved_I[:, :ChemData().NHEAVY]
        print("plddt_loss", plddt_loss)
        plddt_loss = plddt_loss.sum() / (is_resolved_I.sum() + self.eps)

        #true_pae_binned, is_valid_pair = self.calc_pae(X_pred_L, X_gt_L, X_exists_L)
        #pae_loss = self.cce(network_output["pae_logits"], true_pae_binned) * is_valid_pair
        #pae_loss = pae_loss.sum() / (is_valid_pair.sum() + self.eps)
        pae_loss = (network_output["pae"] * 0 ).sum()

        true_pde_binned, is_valid_pair = self.calc_pde(X_pred_L, X_gt_L, X_exists_L, loss_input["seq"])
        pde_predicted = network_output["pde"].permute(0,3,1,2)
        #
        pde_loss = self.cce(pde_predicted, true_pde_binned) * is_valid_pair
        #pde_loss = cross_entropy_loss(pde_predicted, true_pde_binned.squeeze(0)) #* is_valid_pair
        print("pde_loss", pde_loss)
        print("pde loss", (pde_loss > 0).sum())
        print("is_valid_pair", (is_valid_pair > 0).sum())
        print("pde_loss shape"  , pde_loss.shape)
        print("is_valid_pair shape", is_valid_pair.shape)

        pde_loss = pde_loss.sum() / (is_valid_pair.sum() + self.eps)

        #exp_resolved_logits = self.subsample_exp_resolved_logits(network_output["exp_resolved"], loss_input["seq"])
        exp_resolved_logits = network_output["exp_resolved"]
        exp_resolved_loss = self.cce(
            exp_resolved_logits.reshape(I, 2, ChemData().NHEAVY),
            is_resolved_I[:, :ChemData().NHEAVY].long() 
             ) * ChemData().heavyatom_mask.to(loss_input["seq"].device)[loss_input["seq"]][:, :ChemData().NHEAVY]
        exp_resolved_loss = exp_resolved_loss.sum() / (ChemData().heavyatom_mask.to(loss_input["seq"].device)[loss_input["seq"]][:, :ChemData().NHEAVY].sum() + self.eps)
        loss_dict = dict(
            plddt_loss=plddt_loss.detach(),
            pae_loss=pae_loss.detach(),
            pde_loss=pde_loss.detach(),
            exp_resolved_loss=exp_resolved_loss.detach(),
        )
        return self.weight * (plddt_loss + pae_loss + pde_loss + exp_resolved_loss), loss_dict

    def calc_lddt(self, X_pred_L, X_gt_L, X_exists_L, seq):
        I = seq.shape[0]
        ground_truth_distances = torch.cdist(X_gt_L, X_gt_L, compute_mode="donot_use_mm_for_euclid_dist")
        predicted_distances = torch.cdist(X_pred_L, X_pred_L, compute_mode="donot_use_mm_for_euclid_dist")
        
        X_exists_LL = X_exists_L.unsqueeze(-1) * X_exists_L.unsqueeze(-2)
        is_real_atom = ChemData().heavyatom_mask.to(seq.device)[seq]
        tok_idx = is_real_atom.nonzero()[:, 0]

        difference_distances = torch.abs(ground_truth_distances - predicted_distances)
        lddt_matrix = torch.zeros_like(difference_distances)
        lddt_matrix = 0.25 * (difference_distances < 4.0) + \
                    0.25 * (difference_distances < 2.0) + \
                    0.25 * (difference_distances < 1.0) + \
                    0.25 * (difference_distances < 0.5)
        in_same_residue_LL = tok_idx.unsqueeze(-1) == tok_idx.unsqueeze(-2)
        close_distances_LL = ground_truth_distances < 15.0

        # include distances where both atoms are resolved and not in the same residue, and are within an inclusion radius (15A)
        mask_LL = X_exists_LL * ~in_same_residue_LL * close_distances_LL
        lddt_per_atom_L = (lddt_matrix * mask_LL).sum(-1) / (mask_LL.sum(-1)+self.eps)
        # remove unresolved residues
        #lddt_per_atom_L = lddt_per_atom_L * X_exists_L
        # only aggregate over the resolved atoms in each residue
        lddt_per_atom_I = torch.zeros_like(is_real_atom, dtype=torch.float32)
        lddt_per_atom_I[is_real_atom] = lddt_per_atom_L
        X_exists_I = torch.zeros_like(is_real_atom, dtype=torch.bool)
        X_exists_I[is_real_atom] = X_exists_L
        lddt_per_atom_binned = self.bin_values(lddt_per_atom_I, max_value=self.plddt.max_value, n_bins=self.plddt.n_bins)
        return lddt_per_atom_binned, X_exists_I
#        lddt_per_residue = torch.zeros((I,), device=lddt_per_atom_L.device).index_add_(
            #0,
            #tok_idx, 
            #lddt_per_atom_L[0], 
        #)
        #resolved_atoms_per_residue = torch.zeros_like(seq, dtype=torch.float32).index_add_(0, tok_idx, X_exists_L[0].float())       
        #assert torch.all(resolved_atoms_per_residue <= is_real_atom.sum(-1))

        #mean_lddt_per_residue = lddt_per_residue / (resolved_atoms_per_residue + self.eps)

        #true_lddt_binned = self.bin_values(mean_lddt_per_residue, max_value=self.plddt.max_value, n_bins=self.plddt.n_bins)
        #has_any_resolved = resolved_atoms_per_residue > 0
        #return true_lddt_binned, has_any_resolved

    def calc_pae(self, X_pred_L, X_gt_L, X_exists_L, seq):
        # based on token identity, get frame_indices
        predicted_distances = torch.cdist(X_pred_L, X_pred_L, compute_mode="donot_use_mm_for_euclid_dist")
        def get_frame_indices(aa):
            
            if rf2aa.util.is_protein(aa):
                return torch.arange(0, 3), torch.tensor([1])
            elif rf2aa.util.is_dna(aa) or rf2aa.util.is_rna(aa):
                # C1', C3', C4'
                return torch.tensor([1, 9, 8]), torch.tensor([1])
            else:
                # TODO: implement for small molecules
                return torch.zeros(3), torch.tensor([0])    
        
        def get_frame_indices_from_seq(seq):
            return torch.cat([get_frame_indices(aa) for aa in seq])

        # get atom positions for each frame
        # compute aligned error

        return 0.0

    def calc_pde(self, X_pred_L, X_gt_L, X_exists_L, seq):
        rep_atoms = find_rep_atoms(seq)
        X_pred_I = X_pred_L.index_select(1, rep_atoms)
        X_gt_I = X_gt_L.index_select(1, rep_atoms)
        X_exists_I = X_exists_L.index_select(1, rep_atoms)
        predicted_distances = torch.cdist(X_pred_I, X_pred_I, compute_mode="donot_use_mm_for_euclid_dist")
        ground_truth_distances = torch.cdist(X_gt_I, X_gt_I, compute_mode="donot_use_mm_for_euclid_dist")
        difference_distances = torch.abs(ground_truth_distances - predicted_distances)
        true_pde_binned = self.bin_values(difference_distances, max_value=self.pde.max_value, n_bins=self.pde.n_bins)
        X_exists_II = X_exists_I.unsqueeze(-1) * X_exists_I.unsqueeze(-2)
        return true_pde_binned.detach(), X_exists_II.detach()
    
    def subsample_exp_resolved_logits(self, exp_resolved_logits, seq):
        I = seq.shape[0]
        is_real_atom = ChemData().heavyatom_mask.to(seq.device)[seq][:, :ChemData().NHEAVY]
        
        return exp_resolved_logits.reshape(I, ChemData().NHEAVY, 2)[is_real_atom]


    def bin_values(self, values, max_value, n_bins):
        # assumes that the bins go from 0 to max_value
        bin_size = max_value / n_bins
        bins = torch.linspace(0, max_value, n_bins-1, device=values.device)
        return torch.bucketize(values, bins, right=True)
    

class FindOptimalPermutation(nn.Module):

    def __init__(self):
        super(FindOptimalPermutation, self).__init__()

    def forward(
        self, 
        X_pred_L,
        X_gt_symm_I,
        crd_mask_I,
        seq, 
        f
    ):
        is_real_atom = ChemData().heavyatom_mask.to(seq.device)[seq]
        # create container for X_pred_I
        I = seq.shape[0]
        assert X_pred_L.shape[0] == 1
        X_pred_I = torch.zeros(1, I, ChemData().NTOTAL, 3, device=X_pred_L.device)
        X_pred_I[:, is_real_atom] = X_pred_L

        if any(rf2aa.util.is_atom(seq)):
            Ls_prot, Ls_sm = self.get_chain_ls(f, seq)

            print("Ls_prot", Ls_prot)
            print("Ls_sm", Ls_sm)
            ch_label = f["entity_id"][None]
            print("ch_label", ch_label)
            print("seq", seq)
            X_gt_I, crd_mask_I = resolve_equiv_natives_asmb(X_pred_I, X_gt_symm_I, crd_mask_I, ch_label, Ls_prot, Ls_sm)
        else:
            X_gt_I, crd_mask_I = resolve_equiv_natives(X_pred_I, X_gt_symm_I, crd_mask_I)
        
        # convert X_gt_I back to L dimension
        X_gt_L = X_gt_I[:, is_real_atom]
        X_exists_L = crd_mask_I[:, is_real_atom]
        return X_gt_L, X_exists_L

    def get_chain_ls(self, f, seq):
        asym_id = f["asym_id"]
        Ls = []
        for i in range(asym_id.max()+1):
            Ls.append((asym_id == i).sum())
        print("Ls", Ls)
        print("asym_id", asym_id)
        i_start = 0
        Ls_prot = []
        Ls_sm = []
        for L in Ls:
            # check if the L is a protein or a small molecule
            # if protein, append to Ls_prot
            # if small molecule, append to Ls_sm
            if all(rf2aa.util.is_atom(seq[i_start:i_start+L])):
                Ls_sm.append(L.item())
            else:
                Ls_prot.append(L.item())

            i_start += L 
        return Ls_prot, Ls_sm             

def cross_entropy_loss(logits, targets):
    """
    Compute the categorical cross-entropy loss.
    
    Parameters:
    logits (torch.Tensor): The raw model outputs, shape (N, C) where N is the batch size, C is the number of classes.
    targets (torch.Tensor): The ground truth labels, shape (N,).
    
    Returns:
    loss (torch.Tensor): The average cross-entropy loss over the batch.
    """
    import torch.nn.functional as F 
    # Step 1: Apply softmax to logits to get probabilities
    probs = F.softmax(logits, dim=1)  # Shape: (N, C)
    import pdb; pdb.set_trace()
    # Step 2: Compute log of probabilities
    log_probs = torch.log(probs)  # Shape: (N, C)
    
    # Step 3: Gather the log-probabilities of the correct class for each sample
    # `targets` is expected to contain the index of the correct class for each sample in the batch
    batch_size = logits.shape[0]
    correct_log_probs = log_probs[range(batch_size), targets]  # Shape: (N,)
    
    # Step 4: Compute the negative log likelihood loss
    loss = -torch.mean(correct_log_probs)  # Average over batch
    print("probs", probs)
    print("log_probs", log_probs)
    print("correct_log_probs", correct_log_probs)
    print("loss", loss)
    return loss