import torch
import torch.nn as nn

import rf2aa
from rf2aa.chemical import ChemicalData as ChemData
from rf2aa.loss.loss import resolve_equiv_natives, resolve_equiv_natives_asmb, mask_unresolved_frames, mask_unresolved_frames_batched
from rf2aa.model.layers.af3_auxiliary_heads import find_rep_atoms
from rf2aa.util import rigid_from_3_points, is_atom, get_frames
import torch.utils.checkpoint as checkpoint

from scipy.stats import spearmanr
from rf2aa.training.checkpoint import create_custom_forward


class AF3_with_rollout(nn.Module):
    """ Implements rollout on each training step """
    def __init__(self, model, confidence, sampler, config):
        super(AF3_with_rollout, self).__init__()
        self.model = model
        self.confidence = confidence
        self.sampler = sampler
        self.find_optimal_permutation = FindOptimalPermutation()
        self.num_timesteps = 20
        self.config = config

    def forward(
            self, 
            input, 
            n_cycle, 
            # X_gt_I_symm, 
            # crd_mask_I, 
            seq, 
            rep_atom_idxs,
            no_sync,
            frame_atom_idxs=None
        ):
        # first do forward pass
        with torch.no_grad():
            trunk_output = self.model.trunk_forward(input, n_cycle, no_sync)
        # save embeddings
        # do rollout conditioned on embeddings
        # with nograd? 
        # with torch.no_grad():
            noise_schedule = self.sampler.construct_noise_schedule(self.num_timesteps, 0, 1).to(input["f"]["msa"].device)
            diffusion_output = self.sampler.sample_diffusion(
                input["f"],
                trunk_output["S_inputs_I"].clone().detach(),
                trunk_output["S_I"].clone().detach(),
                trunk_output["Z_II"].clone().detach(), 
                noise_schedule,
                step_scale=1.5, # int he paper it says they changed this during the rollout
                D=self.config.dataset_params.diffusion_batch_size_rollout
            )

            # print('diffusion_output:', diffusion_output["X_L"].shape)
            B = diffusion_output["X_L"].shape[0]
            
            # find ground truth permutation
            # X_gt_L, X_exists_L = self.find_optimal_permutation(
            #     diffusion_output["X_L"].clone().detach(),
            #     X_gt_I_symm,
            #     crd_mask_I,
            #     seq,
            #     input["f"]
            # )

        #     trunk_output = self.model.post_recycle(
        #         trunk_output["S_inputs_I"],
        #         trunk_output["S_init_I"],
        #         trunk_output["Z_init_II"],
        #         trunk_output["S_I"],
        #         trunk_output["Z_II"],
        #         input["f"],
        #         input["X_noisy_L"],
        #         input["t"]
        # )
        
        # run confidence model on embeddings and output structure
        #Bug in deepspeed backwards requires us to do it with batch size 1
        confidence_stack = {}
        confidence_stack['plddt_logits'] = None
        confidence_stack['pae_logits'] = None
        confidence_stack['pde_logits'] = None
        confidence_stack['exp_resolved_logits'] = None
        with torch.enable_grad():
            for i in range(B):

                confidence = checkpoint.checkpoint(create_custom_forward(self.confidence, frame_atom_idxs=frame_atom_idxs),
                    trunk_output["S_inputs_I"],
                    trunk_output["S_I"],
                    trunk_output["Z_II"],
                    diffusion_output["X_L"][i].unsqueeze(0),
                    seq,
                    rep_atom_idxs,
                    use_reentrant=False
                )

                for k, v in confidence.items():
                    if confidence_stack[k] is not None:
                        confidence_stack[k] = torch.cat((confidence_stack[k], v), dim=0)
                    else:
                        confidence_stack[k] = v

        return dict(
            # X_gt_L=X_gt_L.detach(),
            # X_exists_L=X_exists_L.detach(),
            X_L=None,
            X_pred_rollout_L=diffusion_output["X_L"],
            plddt=confidence_stack["plddt_logits"],
            pae=confidence_stack["pae_logits"],
            pde=confidence_stack["pde_logits"],
            exp_resolved=confidence_stack["exp_resolved_logits"],
            distogram=None,
        )
    
    # def confidence_wrapper(self, S_inputs_I, S_I, Z_II, X_L, seq, rep_atom_idxs, frame_atom_idxs=None, use_amp=True):
    #     return self.confidence(S_inputs_I, S_I, Z_II, X_L, seq, rep_atom_idxs, frame_atom_idxs=frame_atom_idxs, use_amp=use_amp)

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
        I = loss_input["seq"].shape[-1]
        
        X_gt_L = loss_input["X_gt_L"]
        X_exists_L = loss_input["crd_mask_L"]
        X_pred_L = network_output["X_pred_rollout_L"]
        B = X_pred_L.shape[0]
        print('B:', B)

        true_lddt_binned, is_resolved_I = self.calc_lddt(X_pred_L, X_gt_L, X_exists_L, loss_input["seq"], loss_input["is_real_atom"])
        plddt_loss = self.cce(
            network_output["plddt"].reshape(-1, self.plddt.n_bins, I, ChemData().NHEAVY), 
            true_lddt_binned[..., :ChemData().NHEAVY].long()
            ) * is_resolved_I[..., :ChemData().NHEAVY] 
        plddt_loss = plddt_loss.sum() / (is_resolved_I.sum() + self.eps)

        pae_logits = network_output["pae"]
        true_pae_binned, pae_logits, valid_pae_pairs = self.calc_pae(loss_input, X_pred_L, X_gt_L, X_exists_L, pae_logits, loss_input['frame_atom_idxs'])
        pae_loss = self.cce(pae_logits, true_pae_binned) * valid_pae_pairs
        pae_loss = pae_loss.sum() / (valid_pae_pairs.sum() + self.eps)

        true_pde_binned, is_valid_pair = self.calc_pde(X_pred_L, X_gt_L, X_exists_L, loss_input["seq"], loss_input['rep_atom_idxs'])
        pde_predicted = network_output["pde"].permute(0,3,1,2)
        pde_loss = self.cce(pde_predicted, true_pde_binned) * is_valid_pair
        pde_loss = pde_loss.sum() / (is_valid_pair.sum() + self.eps)

        #exp_resolved_logits = self.subsample_exp_resolved_logits(network_output["exp_resolved"], loss_input["seq"])
        exp_resolved_logits = network_output["exp_resolved"]
        exp_resolved_loss = self.cce(
            exp_resolved_logits.reshape(B, 2, I, ChemData().NHEAVY),
            is_resolved_I[:, :, :ChemData().NHEAVY].long() 
             ) * loss_input["is_real_atom"][:, :ChemData().NHEAVY]
        exp_resolved_loss = exp_resolved_loss.sum() / (loss_input["is_real_atom"][:, :ChemData().NHEAVY].sum() + self.eps)
        exp_resolved_loss = exp_resolved_loss / B

        loss_dict = dict(
            plddt_loss=plddt_loss.detach(),
            pae_loss=pae_loss.detach(),
            pde_loss=pde_loss.detach(),
            exp_resolved_loss=exp_resolved_loss.detach(),
        )

        #Get correlations across and within batches
        #unbin values
        true_lddt_unbinned = ((true_lddt_binned.detach() + 1) * 0.02 ) * is_resolved_I
        true_lddt_batchmean = true_lddt_unbinned.sum(dim=(1,2)) / (is_resolved_I.sum(dim=(1,2)) + self.eps)
        true_lddt = true_lddt_unbinned.sum() / (is_resolved_I.sum() + self.eps)

        true_pae = ((true_pae_binned.detach() + 1) * 0.5 - 0.25) * valid_pae_pairs
        true_pae_batchmean = true_pae.sum(dim=(1,2)) / (valid_pae_pairs.sum(dim=(1,2)) + self.eps)
        true_pae = true_pae.sum() / (valid_pae_pairs.sum() + self.eps)

        true_pde = ((true_pde_binned.detach() + 1) * 0.5 - .25) * is_valid_pair
        true_pde_batchmean = true_pde.sum(dim=(1,2)) / (is_valid_pair.sum(dim=(1,2)) + self.eps)
        true_pde = true_pde.sum() / (is_valid_pair.sum() + self.eps)

        #now do similarly for predicted values
        lddt_bins = torch.linspace(0.02, 1.0, 50, device=true_lddt_binned.device)
        plddt_unbinned = network_output["plddt"].reshape(B, self.plddt.n_bins, I, ChemData().NHEAVY).detach().float()
        plddt_unbinned = torch.nn.Softmax(dim=1)(plddt_unbinned)
        plddt_unbinned = (plddt_unbinned * lddt_bins[None, :, None, None]).sum(dim=1) * is_resolved_I[..., :ChemData().NHEAVY]
        plddt_batchmean = plddt_unbinned.sum(dim=(1,2)) / (is_resolved_I.sum(dim=(1,2)) + self.eps)
        plddt = plddt_unbinned.sum() / (is_resolved_I.sum() + self.eps)
           
        pae_bins = torch.linspace(0.25, 31.75, 64, device=true_pae_binned.device)
        pae_unbinned = torch.nn.Softmax(dim=1)(pae_logits).detach().float()
        pae_unbinned = (pae_unbinned * pae_bins[None, :, None, None]).sum(dim=1) * valid_pae_pairs
        pae_batchmean = pae_unbinned.sum(dim=(1,2)) / (valid_pae_pairs.sum(dim=(1,2)) + self.eps)
        pae = (pae_unbinned * valid_pae_pairs).sum() / (valid_pae_pairs.sum() + self.eps)

        pde_unbinned = torch.nn.Softmax(dim=1)(pde_predicted).detach().float()
        pde_unbinned = (pde_unbinned * pae_bins[None, :, None, None]).sum(dim=1) * is_valid_pair
        pde = (pde_unbinned * is_valid_pair).sum() / (is_valid_pair.sum() + self.eps)
        pde_batchmean = pde_unbinned.detach().mean(dim=(1,2))

        print('in train loss calc, predicted error is :', plddt, pae, pde)
        print('in train loss calc, true error is:', true_lddt, true_pae, true_pde)

        # Calculate Spearman rank correlation
        plddt_rank_corr, lddt_spearman_p = spearmanr(true_lddt_batchmean.cpu().numpy(), plddt_batchmean.cpu().numpy())
        pae_rank_corr, pae_spearman_p = spearmanr(true_pae_batchmean.cpu().numpy(), pae_batchmean.cpu().numpy())
        pde_rank_corr, pde_spearman_p = spearmanr(true_pde_batchmean.cpu().numpy(), pde_batchmean.cpu().numpy())
        print('spearman_plddt_corr:', plddt_rank_corr)
        print('spearman_pae_corr:', pae_rank_corr)
        print('spearman_pde_corr:', pde_rank_corr)
        print('plddt_spread:', plddt_batchmean.max() - plddt_batchmean.min())
        print('pae_spread:', pae_batchmean.max() - pae_batchmean.min())
        print('pde_spread:', pde_batchmean.max() - pde_batchmean.min())
        print('true plddt spread:', true_lddt_batchmean.max() - true_lddt_batchmean.min())
        print('true pae spread:', true_pae_batchmean.max() - true_pae_batchmean.min())
        print('true pde spread:', true_pde_batchmean.max() - true_pde_batchmean.min())

        # #an easy way of incentivizing ranking accuracy is the following (Listnet):
        # rank_plddt_t = torch.nn.Softmax(dim=0)(true_lddt_batchmean)
        # rank_plddt_p = torch.nn.Softmax(dim=0)(plddt_batchmean)
        # rank_pae_t = torch.nn.Softmax(dim=0)(true_pae_batchmean)
        # rank_pae_p = torch.nn.Softmax(dim=0)(pae_batchmean)
        # rank_pde_t = torch.nn.Softmax(dim=0)(true_pde_batchmean)
        # rank_pde_p = torch.nn.Softmax(dim=0)(pde_batchmean)

        # plddt_rank_loss = -torch.mean(rank_plddt_t * torch.log(rank_plddt_p))
        # pae_rank_loss = -torch.mean(rank_pae_t * torch.log(rank_pae_p))
        # pde_rank_loss = -torch.mean(rank_pde_t * torch.log(rank_pde_p))
        # print('rank_loss:', plddt_rank_loss, pae_rank_loss, pde_rank_loss)

        return self.weight * (self.plddt.weight * plddt_loss + self.pae.weight * pae_loss + self.pde.weight * pde_loss + self.exp_resolved.weight * exp_resolved_loss), loss_dict

    def calc_lddt(self, X_pred_L, X_gt_L, X_exists_L, seq, is_real_atom):
        I = seq.shape[-1]
        B = X_pred_L.shape[0]
        L = X_pred_L.shape[1]

        #If seq is too long, split the batches to deal with a memory issue during validation
        if I > 384:
            ground_truth_distances = torch.cdist(X_gt_L[:B//2], X_gt_L[:B//2], compute_mode="donot_use_mm_for_euclid_dist")
            predicted_distances = torch.cdist(X_pred_L[:B//2], X_pred_L[:B//2], compute_mode="donot_use_mm_for_euclid_dist")
            
            ground_truth_distances2 = torch.cdist(X_gt_L[B//2:], X_gt_L[B//2:], compute_mode="donot_use_mm_for_euclid_dist")
            predicted_distances2 = torch.cdist(X_pred_L[B//2:], X_pred_L[B//2:], compute_mode="donot_use_mm_for_euclid_dist")

            ground_truth_distances = torch.cat((ground_truth_distances, ground_truth_distances2), dim=0)
            predicted_distances = torch.cat((predicted_distances, predicted_distances2), dim=0)
        else:
            ground_truth_distances = torch.cdist(X_gt_L, X_gt_L, compute_mode="donot_use_mm_for_euclid_dist")
            predicted_distances = torch.cdist(X_pred_L, X_pred_L, compute_mode="donot_use_mm_for_euclid_dist")


        X_exists_LL = X_exists_L.unsqueeze(-1) * X_exists_L.unsqueeze(-2)
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
        lddt_per_atom_I = lddt_per_atom_I.unsqueeze(0).repeat(B, 1, 1)

        lddt_per_atom_I[:,is_real_atom] = lddt_per_atom_L
        X_exists_I = torch.zeros_like(is_real_atom, dtype=torch.bool)
        X_exists_I = X_exists_I.unsqueeze(0).repeat(B, 1, 1)
        X_exists_I[:, is_real_atom] = X_exists_L
        lddt_per_atom_binned = self.bin_values(lddt_per_atom_I, max_value=self.plddt.max_value, n_bins=self.plddt.n_bins)

        return lddt_per_atom_binned, X_exists_I


    
    def calc_pae(self, loss_input, X_pred_L, X_gt_L, X_exists_L, pae_logits, frame_atom_idxs, eps=1e-4):

        seq = loss_input["seq"]
        atom_frames = loss_input["atom_frames"]
        B = X_pred_L.shape[0]

        #Construct the faux atom-36 representation so we can use existing machinery to get frames
        frame_atom_idxs = frame_atom_idxs.unsqueeze(0).expand(B, -1, -1)
        X_pred_I = torch.zeros(B, seq.shape[-1], 36, 3, device=X_pred_L.device)
        X_pred_I[...,0,:] = torch.gather(X_pred_L, 1, frame_atom_idxs[...,0].unsqueeze(-1).expand(-1, -1, 3))
        X_pred_I[...,1,:] = torch.gather(X_pred_L, 1, frame_atom_idxs[...,1].unsqueeze(-1).expand(-1, -1, 3))
        X_pred_I[...,2,:] = torch.gather(X_pred_L, 1, frame_atom_idxs[...,2].unsqueeze(-1).expand(-1, -1, 3))

        X_gt_I = torch.zeros(B, seq.shape[-1], 36, 3, device=X_gt_L.device)
        X_gt_I[...,0,:] = torch.gather(X_gt_L, 1, frame_atom_idxs[...,0].unsqueeze(-1).expand(-1, -1, 3))
        X_gt_I[...,1,:] = torch.gather(X_gt_L, 1, frame_atom_idxs[...,1].unsqueeze(-1).expand(-1, -1, 3))
        X_gt_I[...,2,:] = torch.gather(X_gt_L, 1, frame_atom_idxs[...,2].unsqueeze(-1).expand(-1, -1, 3))

        atom_mask = torch.zeros(B, seq.shape[-1], 36, device=X_exists_L.device, dtype=torch.bool)
        atom_mask[...,0] = torch.gather(X_exists_L, 1, frame_atom_idxs[...,0])
        atom_mask[...,1] = torch.gather(X_exists_L, 1, frame_atom_idxs[...,1])
        atom_mask[...,2] = torch.gather(X_exists_L, 1, frame_atom_idxs[...,2])

        frames, frame_mask = get_frames(0, 0, seq.unsqueeze(0).repeat(B,1), ChemData().frame_indices.to(seq.device), atom_frames)

        N, L, natoms, _ = X_pred_I.shape

        # flatten middle dims so can gather across residues
        X_prime = X_pred_I.reshape(N, L*natoms, -1, 3).repeat(1,1,ChemData().NFRAMES,1)
        Y_prime = X_gt_I.reshape(N, L*natoms, -1, 3).repeat(1,1,ChemData().NFRAMES,1)
        frames_reindex_batched, frame_mask_batched = mask_unresolved_frames_batched(frames, frame_mask, atom_mask)

        X_x = torch.gather(X_prime, 1, frames_reindex_batched[...,0:1].repeat(1,1,1,3))
        X_y = torch.gather(X_prime, 1, frames_reindex_batched[...,1:2].repeat(1,1,1,3))
        X_z = torch.gather(X_prime, 1, frames_reindex_batched[...,2:3].repeat(1,1,1,3))
        uX,tX = rigid_from_3_points(X_x, X_y, X_z)

        Y_x = torch.gather(Y_prime, 1, frames_reindex_batched[...,0:1].repeat(1,1,1,3))
        Y_y = torch.gather(Y_prime, 1, frames_reindex_batched[...,1:2].repeat(1,1,1,3))
        Y_z = torch.gather(Y_prime, 1, frames_reindex_batched[...,2:3].repeat(1,1,1,3))
        uY,tY = rigid_from_3_points(Y_x, Y_y, Y_z)

        uX = uX[:,:, 0]
        uY = uY[:,:, 0]

        # Compute xij_ca across the batch
        # uX: (B, L, 3), X_pred_I: (B, A, 3), X_y: (B, L, 3)
        xij_ca = torch.einsum(
            'bfji,bfaj->bfai',
            uX,  # select valid frames for backbone, shape (B, N_valid_frames, 3)
            X_pred_I[:, None, :, 1] - X_y[:, :, None, 0]
        )  # Result: (B, N_valid_frames, N_valid_ca, 3)

        # Compute xij_ca_t across the batch
        # uY: (B, L, 3), X_gt_I: (B, A, 3), Y_y: (B, L, 3)
        xij_ca_t = torch.einsum(
            'bfji,bfaj->bfai',
            uY,  # select valid frames for backbone, shape (B, N_valid_frames, 3)
            X_gt_I[:, None, :, 1] - Y_y[:, :, None, 0]
        )  # Result: (B, N_valid_frames, N_valid_ca, 3)

        valid_frames = frame_mask_batched[:,:,0] # valid backbone frames (B,I)
        valid_ca = atom_mask[:,:,1] # valid CA atoms (B,I)
        valid_pairs = valid_frames[:,:,None] & valid_ca[:,None,:] # valid pairs (B,I,I)

        eij_label = torch.sqrt(torch.square(xij_ca - xij_ca_t).sum(dim=-1)+eps).clone().detach()
        true_pae_label = self.bin_values(eij_label, max_value=self.pae.max_value, n_bins=self.pae.n_bins)
        pae_logits = pae_logits.permute(0, 3, 1, 2) # (1, nbins, N_frames, N_ca)

        return true_pae_label.detach(), pae_logits, valid_pairs

    def calc_pde(self, X_pred_L, X_gt_L, X_exists_L, seq, rep_atoms):
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
        I = seq.shape[-1]
        B = exp_resolved_logits.shape[0]
        is_real_atom = ChemData().heavyatom_mask.to(seq.device)[seq][:, :ChemData().NHEAVY]
        
        return exp_resolved_logits.reshape(B, I, ChemData().NHEAVY, 2)[:, is_real_atom]


    def bin_values(self, values, max_value, n_bins):
        # assumes that the bins go from 0 to max_value
        bin_size = max_value / n_bins
        bins = torch.linspace(bin_size, max_value - bin_size, n_bins-1, device=values.device)
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

def convert_allatom_coords_to_residue_coords(X_pred_L, seq):
    """
    Reverse of convert_residue_coords_to_allatom_coords.

    X_pred_L: (B, N_real_atoms, 3) - Coordinates for only real atoms (heavy atoms)
    seq: (B, I) - Sequence tensor indicating residue types

    Returns:
        X_pred: (B, I, natoms, 3) - Full tensor with coordinates for all atoms,
                                    with zeroed entries for non-heavy atoms.
    """
    # Get the mask for real atoms based on the sequenceChemData()
    is_real_atom = ChemData().heavyatom_mask.to(seq.device)[seq]  # (B, I, natoms)
    # print('is_real_atom:', is_real_atom.shape, is_real_atom)

    # Initialize the full tensor with zeros
    B = X_pred_L.shape[0]
    I = seq.shape[-1]
    natoms = is_real_atom.shape[-1]
    X_pred = torch.zeros((B, I, natoms, 3), device=seq.device)

    # Fill in only the "real atom" positions from X_pred_L
    X_pred[:,is_real_atom] = X_pred_L

    return X_pred

def convert_allatom_mask_to_residue_mask(X_pred_L, seq):
    """
    Reverse of convert_residue_coords_to_allatom_coords for X_exists_L.

    X_pred_L: (B, N_real_atoms) - Coordinates for only real atoms (heavy atoms)
    seq: (B, I) - Sequence tensor indicating residue types

    Returns:
        X_pred: (B, I, natoms, 3) - Full tensor with coordinates for all atoms,
                                    with zeroed entries for non-heavy atoms.
    """
    # Get the mask for real atoms based on the sequenceChemData()
    is_real_atom = ChemData().heavyatom_mask.to(seq.device)[seq]  # (B, I, natoms)
    # print('is_real_atom:', is_real_atom.shape, is_real_atom)

    # Initialize the full tensor with zeros
    B = X_pred_L.shape[0]
    I = seq.shape[-1]
    natoms = is_real_atom.shape[-1]
    X_pred = torch.zeros((B, I, natoms), device=seq.device, dtype=torch.bool)

    # Fill in only the "real atom" positions from X_pred_L
    X_pred[:, is_real_atom] = X_pred_L

    return X_pred