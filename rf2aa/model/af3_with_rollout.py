import torch
import torch.nn as nn

import rf2aa
from rf2aa.chemical import ChemicalData as ChemData
from rf2aa.loss.loss import mask_unresolved_frames_batched
from rf2aa.util import rigid_from_3_points, get_frames
import torch.utils.checkpoint as checkpoint

from scipy.stats import spearmanr
from rf2aa.training.checkpoint import create_custom_forward


class AF3_with_rollout(nn.Module):
    """ Implements rollout on each training step """
    def __init__(self, model, confidence, sampler, batch_size_rollout):
        super(AF3_with_rollout, self).__init__()
        self.model = model
        self.confidence = confidence
        self.sampler = sampler
        self.num_timesteps = 20
        self.batch_size_rollout = batch_size_rollout

    def forward(
            self, 
            input, 
            n_cycle, 
            seq, 
            rep_atom_idxs,
            no_sync,
            frame_atom_idxs=None
        ):
        # first do forward pass
        with torch.no_grad():
            trunk_output = self.model.trunk_forward(input, n_cycle, no_sync)

            noise_schedule = self.sampler.construct_noise_schedule(self.num_timesteps, 0, 1).to(input["f"]["msa"].device)
            diffusion_output = self.sampler.sample_diffusion(
                input["f"],
                trunk_output["S_inputs_I"].clone().detach(),
                trunk_output["S_I"].clone().detach(),
                trunk_output["Z_II"].clone().detach(), 
                noise_schedule,
                step_scale=1.5, # int he paper it says they changed this during the rollout
                D=self.batch_size_rollout
            )
        
        # run confidence model on embeddings and output structure
        #Bug in deepspeed backwards requires us to do it with batch size 1
        confidence_stack = {}
        confidence_stack['plddt_logits'] = None
        confidence_stack['pae_logits'] = None
        confidence_stack['pde_logits'] = None
        confidence_stack['exp_resolved_logits'] = None
        with torch.enable_grad():
            for i in range(self.batch_size_rollout):

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
            X_L=None,
            X_pred_rollout_L=diffusion_output["X_L"],
            plddt=confidence_stack["plddt_logits"],
            pae=confidence_stack["pae_logits"],
            pde=confidence_stack["pde_logits"],
            exp_resolved=confidence_stack["exp_resolved_logits"],
            distogram=None,
        )


class ConfidenceLoss(nn.Module):

    def __init__(self, 
                plddt,
                pae,
                pde,
                exp_resolved,
                weight=1,
                rank_loss=None,
                log_statistics=False
        ):
        super(ConfidenceLoss, self).__init__()
        self.weight = weight
        self.plddt = plddt
        self.pae = pae
        self.pde = pde
        self.exp_resolved = exp_resolved
        self.cce = nn.CrossEntropyLoss(reduction="none")
        self.eps = 1e-6
        self.rank_loss = rank_loss
        self.log_statistics = log_statistics
    
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

        true_lddt_binned, is_resolved_I = self.calc_lddt(X_pred_L, X_gt_L, X_exists_L, loss_input["seq"], loss_input["is_real_atom"], loss_input['terminal_oxygen_idxs'])
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
        pde_logits = network_output["pde"].permute(0,3,1,2)
        pde_loss = self.cce(pde_logits, true_pde_binned) * is_valid_pair
        pde_loss = pde_loss.sum() / (is_valid_pair.sum() + self.eps)

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

        confidence_loss = (self.plddt.weight * plddt_loss + self.pae.weight * pae_loss + self.pde.weight * pde_loss + self.exp_resolved.weight * exp_resolved_loss)

        if self.log_statistics or self.rank_loss.use_listnet_loss:
            #Get correlations across and within batches
            #unbin values
            #NOTE: for plddt we take the bin value as the upper threshold of the bin, for pae and pde we take the midpoint (consistent with rf2aa)
            lddt_bin_size = self.plddt.max_value / self.plddt.n_bins
            true_lddt_unbinned = ((true_lddt_binned.detach() + 1) * lddt_bin_size ) * is_resolved_I
            true_lddt_batchmean = true_lddt_unbinned.sum(dim=(1,2)) / (is_resolved_I.sum(dim=(1,2)) + self.eps)
            true_lddt = true_lddt_unbinned.sum() / (is_resolved_I.sum() + self.eps)

            pae_bin_size = self.pae.max_value / self.pae.n_bins
            true_pae = ((true_pae_binned.detach() + 1) * pae_bin_size - (pae_bin_size / 2)) * valid_pae_pairs
            true_pae_batchmean = true_pae.sum(dim=(1,2)) / (valid_pae_pairs.sum(dim=(1,2)) + self.eps)
            true_pae = true_pae.sum() / (valid_pae_pairs.sum() + self.eps)

            pde_bin_size = self.pde.max_value / self.pde.n_bins
            true_pde = ((true_pde_binned.detach() + 1) * pde_bin_size - (pde_bin_size / 2)) * is_valid_pair
            true_pde_batchmean = true_pde.sum(dim=(1,2)) / (is_valid_pair.sum(dim=(1,2)) + self.eps)
            true_pde = true_pde.sum() / (is_valid_pair.sum() + self.eps)

            #now do similarly for predicted values
            lddt_bins = torch.linspace(lddt_bin_size, self.plddt.max_value, self.plddt.n_bins, device=true_lddt_binned.device)
            plddt_unbinned = network_output["plddt"].reshape(B, self.plddt.n_bins, I, ChemData().NHEAVY).detach().float()
            plddt_unbinned = torch.nn.Softmax(dim=1)(plddt_unbinned)
            plddt_unbinned = (plddt_unbinned * lddt_bins[None, :, None, None]).sum(dim=1) * is_resolved_I[..., :ChemData().NHEAVY]
            plddt_batchmean = plddt_unbinned.sum(dim=(1,2)) / (is_resolved_I.sum(dim=(1,2)) + self.eps)
            plddt = plddt_unbinned.sum() / (is_resolved_I.sum() + self.eps)
                
            pae_bins = torch.linspace((pae_bin_size / 2), (self.pae.max_value - (pae_bin_size / 2)), self.pae.n_bins, device=true_pae_binned.device)
            pae_unbinned = torch.nn.Softmax(dim=1)(pae_logits).detach().float()
            pae_unbinned = (pae_unbinned * pae_bins[None, :, None, None]).sum(dim=1) * valid_pae_pairs
            pae_batchmean = pae_unbinned.sum(dim=(1,2)) / (valid_pae_pairs.sum(dim=(1,2)) + self.eps)
            pae = (pae_unbinned * valid_pae_pairs).sum() / (valid_pae_pairs.sum() + self.eps)

            pde_bins = torch.linspace((pde_bin_size / 2), (self.pde.max_value - (pde_bin_size / 2)), self.pde.n_bins, device=true_pde_binned.device)
            pde_unbinned = torch.nn.Softmax(dim=1)(pde_logits).detach().float()
            pde_unbinned = (pde_unbinned * pde_bins[None, :, None, None]).sum(dim=1) * is_valid_pair
            pde = (pde_unbinned * is_valid_pair).sum() / (is_valid_pair.sum() + self.eps)
            pde_batchmean = pde_unbinned.detach().mean(dim=(1,2))

            if self.log_statistics:
                self.log_correlation_statistics(plddt, pae, pde, true_lddt, true_pae, true_pde, true_lddt_batchmean, true_pae_batchmean, true_pde_batchmean, plddt_batchmean, pae_batchmean, pde_batchmean, loss_dict)

            if self.rank_loss.use_listnet_loss:
                # #an easy way of incentivizing ranking accuracy is the following (Listnet):
                rank_plddt_t = torch.nn.Softmax(dim=0)(true_lddt_batchmean)
                rank_plddt_p = torch.nn.Softmax(dim=0)(plddt_batchmean)
                rank_pae_t = torch.nn.Softmax(dim=0)(true_pae_batchmean)
                rank_pae_p = torch.nn.Softmax(dim=0)(pae_batchmean)
                rank_pde_t = torch.nn.Softmax(dim=0)(true_pde_batchmean)
                rank_pde_p = torch.nn.Softmax(dim=0)(pde_batchmean)

                plddt_rank_loss = -torch.mean(rank_plddt_t * torch.log(rank_plddt_p))
                pae_rank_loss = -torch.mean(rank_pae_t * torch.log(rank_pae_p))
                pde_rank_loss = -torch.mean(rank_pde_t * torch.log(rank_pde_p))

                rank_loss_dict = dict(
                    plddt_rank_loss=plddt_rank_loss,
                    pae_rank_loss=pae_rank_loss,
                    pde_rank_loss=pde_rank_loss
                )
                loss_dict.update(rank_loss_dict)
                confidence_loss += (plddt_rank_loss + pae_rank_loss + pde_rank_loss) * self.rank_loss.weight

        return self.weight * confidence_loss, loss_dict

    def calc_lddt(self, X_pred_L, X_gt_L, X_exists_L, seq, is_real_atom, terminal_oxygen_idxs):
        tok_idx = is_real_atom.nonzero()[:, 0]

        #Don't calculate plddt for terminal oxygens, so in those cases we excise those idxs from the L representation and update is_real_atom
        if terminal_oxygen_idxs is not None:
            #NOTE: We don't clone is_real_atom, as modifying the parent dictionary has the added benefit of fixing the terimanl oxygens for the exp_resolved calculation as well.
            #NOTE: tradeoff here between doing this as a loop or a mask. There's usually no terminal oxygens, and when they do exist only normally
            #one or two, maybe better to do the loop
            for idx in terminal_oxygen_idxs:
                X_pred_L = torch.cat((X_pred_L[:, :idx], X_pred_L[:, idx+1:]), dim=1)
                X_gt_L = torch.cat((X_gt_L[:, :idx], X_gt_L[:, idx+1:]), dim=1)
                X_exists_L = torch.cat((X_exists_L[:, :idx], X_exists_L[:, idx+1:]), dim=1)
                affected_tok_idx_I = tok_idx[idx]
                
                #We need to update the is_real_atom to have one less real atom for the affected residue, so we set the last True value to false
                is_real_atom[affected_tok_idx_I][is_real_atom[affected_tok_idx_I].nonzero()[-1]] = False

                #Update the tok_idx to reflect the new indices
                tok_idx = is_real_atom.nonzero()[:, 0]

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

        #Construct the backbone atoms in the faux atom-36 representation so we can use existing machinery to get frames
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

    def bin_values(self, values, max_value, n_bins):
        # assumes that the bins go from 0 to max_value
        bin_size = max_value / n_bins
        bins = torch.linspace(bin_size, max_value - bin_size, n_bins-1, device=values.device)
        return torch.bucketize(values, bins, right=True)
    
    def log_correlation_statistics(self, plddt, pae, pde, true_lddt, true_pae, true_pde, true_lddt_batchmean, true_pae_batchmean, true_pde_batchmean, plddt_batchmean, pae_batchmean, pde_batchmean, loss_dict):

        # Calculate Spearman rank correlation
        plddt_rank_corr, lddt_spearman_p = spearmanr(true_lddt_batchmean.cpu().numpy(), plddt_batchmean.cpu().numpy())
        pae_rank_corr, pae_spearman_p = spearmanr(true_pae_batchmean.cpu().numpy(), pae_batchmean.cpu().numpy())
        pde_rank_corr, pde_spearman_p = spearmanr(true_pde_batchmean.cpu().numpy(), pde_batchmean.cpu().numpy())

        loss_dict.update({
            'pred_err_plddt': plddt,
            'pred_err_pae': pae,
            'pred_err_pde': pde,
            'true_err_plddt': true_lddt,
            'true_err_pae': true_pae,
            'true_err_pde': true_pde,
            'plddt_rank_corr': torch.tensor(plddt_rank_corr),
            'pae_rank_corr': torch.tensor(pae_rank_corr),
            'pde_rank_corr': torch.tensor(pde_rank_corr),
            'plddt_spread': plddt_batchmean.max() - plddt_batchmean.min(),
            'pae_spread': pae_batchmean.max() - pae_batchmean.min(),
            'pde_spread': pde_batchmean.max() - pde_batchmean.min(),
            'true_plddt_spread': true_lddt_batchmean.max() - true_lddt_batchmean.min(),
            'true_pae_spread': true_pae_batchmean.max() - true_pae_batchmean.min(),
            'true_pde_spread': true_pde_batchmean.max() - true_pde_batchmean.min(),
        })
        