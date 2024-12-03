import torch
import torch.nn as nn

import rf2aa
from rf2aa.chemical import ChemicalData as ChemData
from rf2aa.loss.loss import resolve_equiv_natives, resolve_equiv_natives_asmb, mask_unresolved_frames, mask_unresolved_frames_batched
from rf2aa.model.layers.af3_auxiliary_heads import find_rep_atoms
from rf2aa.util import rigid_from_3_points, is_atom, get_frames
import torch.utils.checkpoint as checkpoint


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
            no_sync
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

            print('diffusion_output:', diffusion_output["X_L"].shape)
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
                # confidence = self.confidence(
                #     trunk_output["S_inputs_I"],
                #     trunk_output["S_I"],
                #     trunk_output["Z_II"],
                #     diffusion_output["X_L"][0].unsqueeze(0),
                #     # input["f"],
                #     seq,
                #     rep_atom_idxs
                # )

                confidence = checkpoint.checkpoint(
                    self.confidence_wrapper,
                    trunk_output["S_inputs_I"],
                    trunk_output["S_I"],
                    trunk_output["Z_II"],
                    diffusion_output["X_L"][0].unsqueeze(0),
                    seq,
                    rep_atom_idxs,
                    use_reentrant=False
                    )

                confidence_stack[i] = confidence
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
    
    def confidence_wrapper(self, S_inputs_I, S_I, Z_II, X_L, seq, rep_atom_idxs, use_amp=True):
        return self.confidence(S_inputs_I, S_I, Z_II, X_L, seq, rep_atom_idxs, use_amp=use_amp)

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

        true_lddt_binned, is_resolved_I = self.calc_lddt(X_pred_L, X_gt_L, X_exists_L, loss_input["seq"], loss_input["is_real_atom"])
        plddt_loss = self.cce(
            network_output["plddt"].reshape(-1, self.plddt.n_bins, I, ChemData().NHEAVY), 
            true_lddt_binned[..., :ChemData().NHEAVY].long()
            ) * is_resolved_I[..., :ChemData().NHEAVY] 
        plddt_loss = plddt_loss.sum() / (is_resolved_I.sum() + self.eps)

        pae_logits = network_output["pae"]
        alt_true, alt_pae_logits, pae_comp_valid = self.calc_pae_alt(loss_input, X_pred_L, X_gt_L, X_exists_L, pae_logits, loss_input['frame_atom_idxs'])
        if pae_comp_valid:
            print('alt_pae_logits:', alt_pae_logits.shape, alt_pae_logits.dtype)
            print('alt_true:', alt_true.shape, alt_true.dtype)
            alt_pae_loss = torch.mean(self.cce(alt_pae_logits, alt_true))
        else:
            print('alt_pae_logits:', alt_pae_logits.shape, alt_pae_logits.dtype)
            print('alt_true:', alt_true.shape, alt_true.dtype)
            alt_pae_loss = torch.mean(self.cce(alt_pae_logits, alt_true))
            print('alt_pae_loss:', alt_pae_loss)
            return torch.tensor(0.0, device=alt_true.device, requires_grad=True), {}
            # alt_pae_loss = torch.tensor(0.0, device=alt_true.device)
        pae_loss = alt_pae_loss
        true_pae_binned = alt_true
        masked_pae_logits = alt_pae_logits

        true_pde_binned, is_valid_pair = self.calc_pde(X_pred_L, X_gt_L, X_exists_L, loss_input["seq"], loss_input['rep_atom_idxs'])
        pde_predicted = network_output["pde"].permute(0,3,1,2)
        pde_loss = self.cce(pde_predicted, true_pde_binned) * is_valid_pair
        pde_loss = pde_loss.sum() / (is_valid_pair.sum() + self.eps)

        #exp_resolved_logits = self.subsample_exp_resolved_logits(network_output["exp_resolved"], loss_input["seq"])
        exp_resolved_logits = network_output["exp_resolved"]
        exp_resolved_loss = self.cce(
            exp_resolved_logits.reshape(B, 2, I, ChemData().NHEAVY),
            is_resolved_I[:, :, :ChemData().NHEAVY].long() 
             ) * ChemData().heavyatom_mask.to(loss_input["seq"].device)[loss_input["seq"]][:, :ChemData().NHEAVY]
        exp_resolved_loss = exp_resolved_loss.sum() / (ChemData().heavyatom_mask.to(loss_input["seq"].device)[loss_input["seq"]][:, :ChemData().NHEAVY].sum() + self.eps)
        exp_resolved_loss = exp_resolved_loss / B

        loss_dict = dict(
            plddt_loss=plddt_loss.detach(),
            pae_loss=pae_loss.detach(),
            pde_loss=pde_loss.detach(),
            exp_resolved_loss=exp_resolved_loss.detach(),
        )

        # plddt_loss *= 0
        # pae_loss *= 1
        # pde_loss *= 0
        # exp_resolved_loss *= 0

        #Get the real plddt and reshape for unbinning
        
        #get the bin values
        true_lddt_unbinned = (true_lddt_binned.detach() + 1) * 0.02
        true_lddt_unbinned = true_lddt_unbinned[..., is_resolved_I]
        true_lddt = true_lddt_unbinned.sum() / (is_resolved_I.sum() + self.eps)

        interface_mask = torch.zeros(loss_input["seq"].shape[-1], loss_input["seq"].shape[-1], device=loss_input["seq"].device, dtype=torch.bool)
        if torch.any(is_atom(loss_input["seq"])):
            atom_mask = is_atom(loss_input["seq"])  # shape: [L]
            
            # Create interface mask for both directions at once using outer products
            protein_mask = ~atom_mask # shape: [L]
            ligand_mask = atom_mask   # shape: [L]
            
            # This will create a [L, L] mask where True indicates protein-ligand interfaces
            interface_mask = (protein_mask.unsqueeze(1) & ligand_mask.unsqueeze(0)) | \
                            (ligand_mask.unsqueeze(1) & protein_mask.unsqueeze(0))

        true_pae = ((true_pae_binned.detach() + 1) * .5 - .25).mean()

        true_pde = (((true_pde_binned.detach() + 1) * is_valid_pair) * 0.5 - .25).sum() / (is_valid_pair.sum() + self.eps)

        #now do similarly for predicted values
        lddt_bins = torch.linspace(0.02, 1.0, 50, device=true_lddt_binned.device)
        plddt_unbinned = network_output["plddt"].reshape(B, self.plddt.n_bins, I, ChemData().NHEAVY).detach().float()
        plddt_unbinned = torch.nn.Softmax(dim=1)(plddt_unbinned)
        plddt_unbinned = plddt_unbinned * lddt_bins[None, :, None, None]
        plddt_unbinned = plddt_unbinned.sum(dim=1)
        plddt_unbinned = plddt_unbinned[..., is_resolved_I[..., :ChemData().NHEAVY]]
        plddt = plddt_unbinned.sum() / (is_resolved_I.sum() + self.eps)
        
        pae_bins = torch.linspace(0.25, 31.75, 64, device=true_pae_binned.device)
        if pae_comp_valid:
            pae_unbinned = torch.nn.Softmax(dim=1)(masked_pae_logits).detach().float()
            pae_unbinned = (pae_unbinned * pae_bins[None, :, None, None]).sum(dim=1)
            pae = pae_unbinned.mean()

            if interface_mask.shape[0] == pae_unbinned.shape[-1] and interface_mask.shape[0] == pae_unbinned.shape[-2]:
                pae_interface = (pae_unbinned * interface_mask[None, None, ...]).sum() / (interface_mask.sum() + self.eps)
                true_pae_interface = (((true_pae_binned.detach() + 1) * .5 - .25) * interface_mask[None, ...]).sum() / (interface_mask.sum() + self.eps)
                print(f'i_pae pred: {pae_interface}, i_pae true: {true_pae_interface}')
        else:
            pae = 0.0

        pde_unbinned = torch.nn.Softmax(dim=1)(pde_predicted).detach().float()
        pde_unbinned = (pde_unbinned * pae_bins[None, :, None, None]).sum(dim=1)
        pde = (pde_unbinned * is_valid_pair).sum() / (is_valid_pair.sum() + self.eps)

        print('in train loss calc, predicted error is :', plddt, pae, pde)
        print('in train loss calc, true error is:', true_lddt, true_pae, true_pde)

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


    def calc_pae_alt(self, loss_input, X_pred_L, X_gt_L, X_exists_L, pae_logits, frame_atom_idxs, eps=1e-4):


        seq = loss_input["seq"]
        atom_frames = loss_input["atom_frames"]
        # allatom_mask = ChemData().heavyatom_mask.to(seq.device)
        # is_valid_atom = allatom_mask[seq].to(seq.device)
        # i_res, i_atom = is_valid_atom.bool().nonzero(as_tuple=True)
        B = X_pred_L.shape[0]

        #get the atom-36 representations
        # if X_pred_I.dim() < 3:
        #     X_pred_I = X_pred_I.unsqueeze(0)

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


        # frames, frame_mask = get_rf3_frames(0, 0, seq.unsqueeze(0).repeat(B,1), ChemData().frame_indices.to(seq.device), atom_frames)
        frames, frame_mask = get_frames(0, 0, seq.unsqueeze(0).repeat(B,1), ChemData().frame_indices.to(seq.device), atom_frames)

        

        N, L, natoms, _ = X_pred_I.shape

        # flatten middle dims so can gather across residues
        X_prime = X_pred_I.reshape(N, L*natoms, -1, 3).repeat(1,1,ChemData().NFRAMES,1)
        Y_prime = X_gt_I.reshape(N, L*natoms, -1, 3).repeat(1,1,ChemData().NFRAMES,1)
        frames_reindex_batched, frame_mask_batched = mask_unresolved_frames_batched(frames, frame_mask, atom_mask)
        # frames_reindex, frame_mask = mask_unresolved_frames(frames[...,1], frame_mask, atom_mask)

        if torch.sum(frame_mask_batched) == 0:
            return torch.zeros(B, X_pred_I.shape[1], X_pred_I.shape[1], device=X_pred_I.device, dtype=torch.int64).detach(), torch.zeros(B, 64, X_pred_I.shape[1], X_pred_I.shape[1], device=X_pred_I.device, dtype=torch.bfloat16), False
            
        # X_x = torch.gather(X_prime, 1, frames_reindex[...,0:1].repeat(N,1,1,3))
        # X_y = torch.gather(X_prime, 1, frames_reindex[...,1:2].repeat(N,1,1,3))
        # X_z = torch.gather(X_prime, 1, frames_reindex[...,2:3].repeat(N,1,1,3))
        # uX,tX = rigid_from_3_points(X_x, X_y, X_z)

        # Y_x = torch.gather(Y_prime, 1, frames_reindex[...,0:1].repeat(1,1,1,3))
        # Y_y = torch.gather(Y_prime, 1, frames_reindex[...,1:2].repeat(1,1,1,3))
        # Y_z = torch.gather(Y_prime, 1, frames_reindex[...,2:3].repeat(1,1,1,3))
        # uY,tY = rigid_from_3_points(Y_x, Y_y, Y_z)

        X_x = torch.gather(X_prime, 1, frames_reindex_batched[...,0:1].repeat(1,1,1,3))
        X_y = torch.gather(X_prime, 1, frames_reindex_batched[...,1:2].repeat(1,1,1,3))
        X_z = torch.gather(X_prime, 1, frames_reindex_batched[...,2:3].repeat(1,1,1,3))
        uX,tX = rigid_from_3_points(X_x, X_y, X_z)

        Y_x = torch.gather(Y_prime, 1, frames_reindex_batched[...,0:1].repeat(1,1,1,3))
        Y_y = torch.gather(Y_prime, 1, frames_reindex_batched[...,1:2].repeat(1,1,1,3))
        Y_z = torch.gather(Y_prime, 1, frames_reindex_batched[...,2:3].repeat(1,1,1,3))
        uY,tY = rigid_from_3_points(Y_x, Y_y, Y_z)

        frame_mask_bb = frame_mask_batched[0,:,0] # valid backbone frames (L,)
        atom_mask_ca = atom_mask[0,:,1] # valid CA atoms (L,)

        xij_ca = torch.einsum(
        'fji,faj->fai',
        uX[-1,frame_mask_bb,0],
        X_pred_I[-1,None,atom_mask_ca,1] - X_y[-1,frame_mask_bb,None,0]
        ) # (N_valid_frames, N_valid_ca, 3)

        xij_ca_t = torch.einsum(
            'fji,faj->fai',
            uY[-1,frame_mask_bb,0],
            X_gt_I[-1,None,atom_mask_ca,1] - Y_y[-1,frame_mask_bb,None,0]
        ) # (N_valid_frames, N_valid_ca, 3)

        
        frame_mask_bb = frame_mask_batched[:,:,0] # valid backbone frames (L,)
        atom_mask_ca = atom_mask[:,:,1] # valid CA atoms (L,)
        try:
            uX = uX[frame_mask_bb].reshape(B, -1, 9, 3, 3)
            uX = uX[:,:, 0]
            uY = uY[frame_mask_bb].reshape(B, -1, 9, 3, 3)
            # print('uY', uY.shape)
            uY = uY[:,:, 0]
            # print('uY', uY.shape)

            X_pred_I_ca = X_pred_I[atom_mask_ca].reshape(B, -1, natoms, 3)
            X_gt_I_ca = X_gt_I[atom_mask_ca].reshape(B, -1, natoms, 3)
            X_y = X_y[frame_mask_bb].reshape(B, -1, ChemData().NFRAMES, 3)
            Y_y = Y_y[frame_mask_bb].reshape(B, -1, ChemData().NFRAMES, 3)
            print('X_pred_I_ca:', X_pred_I_ca.shape)
            print('X_gt_I_ca:', X_gt_I_ca.shape)
            print('X_y:', X_y.shape)
            print('Y_y:', Y_y.shape)
            print('uX:', uX.shape)
            print('uY:', uY.shape)
        except Exception as e:
            print('failed to perform uX reshape')
            print(e)
            print('seq', seq)
            print('atom_frames', atom_frames.shape)
            print('frames', frames.shape)
            print('frame_mask', frame_mask.shape)
            print('X_pred_L', X_pred_L.shape)
            print('X_gt_L', X_gt_L.shape)
            print('X_exists_L', X_exists_L.shape)
            print('pae_logits', pae_logits.shape)
            print('uX', uX.shape)
            print('X_pred_I', X_pred_I.shape)
            print('X_gt_I', X_gt_I.shape)
            print('frame_mask_bb', frame_mask_bb.shape)
            print('atom_mask_ca', atom_mask_ca.shape)
            print('returning zero tensors')
            return torch.zeros(B, X_pred_I.shape[1], X_pred_I.shape[1], device=X_pred_I.device, dtype=torch.int64).detach(), torch.zeros(B, 64, X_pred_I.shape[1], X_pred_I.shape[1], device=X_pred_I.device, dtype=torch.bfloat16), False
        # uX = uX[:,:, 0]
        # uY = uY[frame_mask_bb].reshape(B, -1, 9, 3, 3)
        # # print('uY', uY.shape)
        # uY = uY[:,:, 0]
        # # print('uY', uY.shape)

        # X_pred_I_ca = X_pred_I[atom_mask_ca].reshape(B, -1, natoms, 3)
        # X_gt_I_ca = X_gt_I[atom_mask_ca].reshape(B, -1, natoms, 3)
        # X_y = X_y[frame_mask_bb].reshape(B, -1, ChemData().NFRAMES, 3)
        # Y_y = Y_y[frame_mask_bb].reshape(B, -1, ChemData().NFRAMES, 3)

        # Compute xij_ca across the batch
        # uX: (B, L, 3), X_pred_I: (B, A, 3), X_y: (B, L, 3)
        xij_ca_batched = torch.einsum(
            'bfji,bfaj->bfai',
            uX,  # select valid frames for backbone, shape (B, N_valid_frames, 3)
            X_pred_I_ca[:, None, :, 1] - X_y[:, :, None, 0]
        )  # Result: (B, N_valid_frames, N_valid_ca, 3)

        # Compute xij_ca_t across the batch
        # uY: (B, L, 3), X_gt_I: (B, A, 3), Y_y: (B, L, 3)
        xij_ca_t_batched = torch.einsum(
            'bfji,bfaj->bfai',
            uY,  # select valid frames for backbone, shape (B, N_valid_frames, 3)
            X_gt_I_ca[:, None, :, 1] - Y_y[:, :, None, 0]
        )  # Result: (B, N_valid_frames, N_valid_ca, 3)

        xij_ca = xij_ca_batched
        xij_ca_t = xij_ca_t_batched

        eij_label = torch.sqrt(torch.square(xij_ca - xij_ca_t).sum(dim=-1)+eps).clone().detach()
        true_pae_label = self.bin_values(eij_label, max_value=self.pae.max_value, n_bins=self.pae.n_bins)
        logit_pae_masked = pae_logits[:,frame_mask_batched[0,:,0]][:,:,atom_mask[0,:,1]] # (1, N_valid_frames, N_valid_ca, nbins)
        logit_pae_masked = logit_pae_masked.permute(0, 3, 1, 2) # (1, nbins, N_valid_frames, N_valid_ca)

        return true_pae_label.detach(), logit_pae_masked, True

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