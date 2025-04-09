import torch
import torch.nn as nn

from modelhub.alignment import weighted_rigid_align
from modelhub.loss.af3_losses import DiffusionLoss
from modelhub.training.checkpoint import activation_checkpointing


class VerboseDiffusionLoss(DiffusionLoss):

    def __init__(
        self, 
        alpha_virtual_atom=1.0,
        alpha_motif=1.0,
        w_fixed_motif = 1.0,
        ldtt_weight=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.alpha_virtual_atom = alpha_virtual_atom
        self.alpha_motif = alpha_motif

        self.w_fixed_motif = w_fixed_motif
        self.lddt_weight = ldtt_weight

    def forward(self, network_input, network_output, loss_input):
        X_L = network_output["X_L"] # D, L, 3
        D = X_L.shape[0]
        X_gt_L = loss_input["X_gt_L"] # X_gt_L
        crd_mask_L = loss_input["crd_mask_L"] # (D, L)
        tok_idx = network_input["f"]["atom_to_token_map"]
        t = network_input["t"] # (D,)

        is_virtual_atom = network_input['f']['is_virtual'][tok_idx]  # L,
        is_motif_atom_with_fixed_pos = network_input["f"]["is_motif_atom_with_fixed_pos"] # L
        is_motif_token = network_input["f"]["is_motif_token"] # N

        # ... Calculate LDDT loss at the beginning
        smoothed_lddt_loss_, lddt_loss_dict = smoothed_lddt_loss(
            X_L,
            X_gt_L,
            crd_mask_L,
            network_input["f"]["is_dna"], 
            network_input["f"]["is_rna"], 
            tok_idx,
            is_virtual=network_input["f"].get("is_virtual", None),
            alpha_virtual=self.alpha_virtual_atom,
            return_extras=True
        ) # D,

        # ... MSE loss weighting
        w_L = 1 
        w_L = w_L + (
            is_virtual_atom * self.alpha_virtual_atom 
        ).float()
        w_L = w_L + (
            is_motif_token * self.alpha_motif
        ).float()[tok_idx]

        # ... Alignment to fixed motif:
        w_L_fixed = w_L.clone()
        w_L_fixed[~is_motif_atom_with_fixed_pos] = 0
        w_L_fixed[is_virtual_atom] = 0

        w_L = w_L[None].expand(D, -1) * crd_mask_L
        w_L_fixed = w_L_fixed[None].expand(D, -1) * crd_mask_L
        
        # ... Alignment for global loss requires removal of virtual atoms
        w_L_align = w_L.clone()
        w_L_align[:, is_virtual_atom] = 0
        
        # ... Alignments
        X_gt_aligned_L = weighted_rigid_align(X_L, X_gt_L, crd_mask_L[0], w_L_align)
        X_gt_aligned_L = torch.nan_to_num(X_gt_aligned_L)
        X_gt_fixed_aligned_L = weighted_rigid_align(
            X_L, X_gt_L, crd_mask_L[0], w_L_fixed,
        ) # D, M
        X_gt_fixed_aligned_L = torch.nan_to_num(X_gt_fixed_aligned_L)
        
        # ... Global aligned loss
        l_mse_L = w_L * torch.sum((X_L - X_gt_aligned_L) ** 2, dim=-1) 
        l_mse_L = torch.div(l_mse_L, 3 * torch.sum(crd_mask_L[0]) + 1e-4) # D, L
        l_mse = self.get_lambda(t) * l_mse_L.sum(-1)
        def mse_to_loss(mask, l=l_mse_L, t_=t):
            l = l[:, mask]
            """Convert mse loss to diffusion loss using the lambda schedule."""
            if l.numel() == 0:
                return None
            return (l.sum(-1) * self.get_lambda(t_))

        # ... Add fixed motif loss
        if self.w_fixed_motif > 0 and torch.any(is_motif_atom_with_fixed_pos):
            # ... Loss for fixed parts of the aligned motif (should be zero)
            l_mse_L_fixed_motif = w_L[..., is_motif_atom_with_fixed_pos] * torch.sum(
                (X_L[..., is_motif_atom_with_fixed_pos, :] - 
                 X_gt_fixed_aligned_L[..., is_motif_atom_with_fixed_pos, :]) ** 2, dim=-1

            )
            l_mse_L_fixed_motif = torch.div(l_mse_L_fixed_motif, 3 * torch.sum(crd_mask_L[0]) + 1e-4)  # D, M fixed

            # ... Loss for diffused parts of fixed motif (higher)
            # NOTE: is currently the whole diffused region
            l_mse_L_diffused_motif = w_L[..., ~is_motif_atom_with_fixed_pos] * torch.sum(
                (X_L[..., ~is_motif_atom_with_fixed_pos, :] - 
                 X_gt_fixed_aligned_L[..., ~is_motif_atom_with_fixed_pos, :]) ** 2, dim=-1
            )
            l_mse_L_diffused_motif = torch.div(l_mse_L_diffused_motif, 3 * torch.sum(crd_mask_L[0]) + 1e-4)  # D, M unfixed
            
            # ... MSE to diffusion loss
            l_mse_fixed_motif = self.get_lambda(t) * l_mse_L_fixed_motif.sum(-1)
            l_mse_diffused_motif = self.get_lambda(t) * l_mse_L_diffused_motif.sum(-1)
            l_mse_diffused_motif = torch.clamp(l_mse_diffused_motif, max=2) if self.clamp_diffusion_loss else l_mse_diffused_motif

            # ... combined into single motif-aligned loss
            l_mse_motif_aligned = (self.w_fixed_motif * l_mse_fixed_motif + l_mse_diffused_motif)

            l_mse_total = (l_mse + l_mse_motif_aligned) / 2
        else:
            l_mse_motif_aligned = None
            l_mse_diffused_motif = None
            l_mse_fixed_motif = None
            l_mse_total = l_mse

        # ... Reorder the indices to be in ascending t and split by high/low
        t, indices = torch.sort(t)
        smoothed_lddt_loss_ = smoothed_lddt_loss_[indices]
        l_mse_low, l_mse_high = torch.split(l_mse[indices], [len(l_mse)//2, len(l_mse) - len(l_mse)//2])

        
        loss_dict = {
            "mse_loss_mean": l_mse,
            "mse_loss_low_t": l_mse_low,
            "mse_loss_high_t": l_mse_high,
            "mse_loss_global": l_mse,
            "smoothed_lddt_loss_mean": smoothed_lddt_loss_,
            
            # Motif-aligned losses
            "mse_loss_fixed_motif_align": l_mse_motif_aligned,
            "mse_loss_fixed_motif_align_recapitulation": l_mse_fixed_motif,
            "mse_loss_fixed_motif_align_diffused": l_mse_diffused_motif,
            

            # Diffusion losses for components
            "diffusion_loss_is_backbone": mse_to_loss(network_input["f"]["is_backbone"]),
            "diffusion_loss_is_sidechain": mse_to_loss(network_input["f"]["is_sidechain"]),
            "diffusion_loss_is_ligand": mse_to_loss(network_input["f"]["is_ligand"][tok_idx]),
            "diffusion_loss_is_dna": mse_to_loss(network_input["f"]["is_dna"][tok_idx]),
            "diffusion_loss_is_rna": mse_to_loss(network_input["f"]["is_rna"][tok_idx]) ,
            "diffusion_loss_is_virtual": mse_to_loss(is_virtual_atom),
            "diffusion_loss_is_non_virtual": mse_to_loss(~is_virtual_atom),
            "diffusion_loss_is_central": mse_to_loss(network_input["f"]["is_central"]),

            # Other return values
            # "t": t.detach(),
            # "mse_loss": l_mse.detach(),
            # "smoothed_lddt_loss": smoothed_lddt_loss_.detach(),
        }
        loss_dict.update(lddt_loss_dict)
        loss_dict = {k: torch.mean(v).detach() for k, v in loss_dict.items() if v is not None}

        # ... Return
        l_mse_total = torch.clamp(l_mse_total, max=2) if self.clamp_diffusion_loss else l_mse_total
        l_diffusion_total = torch.mean(l_mse_total)  # D, -> scalar
        if self.lddt_weight > 0:
            l_diffusion_total += self.lddt_weight * smoothed_lddt_loss_.mean()
        return self.weight*l_diffusion_total, loss_dict


def smoothed_lddt_loss(X_L, X_gt_L, crd_mask_L, is_dna, is_rna, tok_idx, is_virtual=None, alpha_virtual=1.0, return_extras=False, eps=1e-6):

    @activation_checkpointing
    def _dolddt(X_L, X_gt_L, crd_mask_L, is_dna, is_rna, tok_idx, eps, use_amp=True):
        B,L = X_L.shape[:2]
        first_index,second_index = torch.triu_indices(L,L,1, device=X_L.device)

        # compute the unique distances between all pairs of atoms
        X_gt_L = X_gt_L.nan_to_num()

        # only use native 1 (assumes dist map identical btwn all copies)
        ground_truth_distances = torch.linalg.norm(X_gt_L[0:1,first_index]-X_gt_L[0:1,second_index], dim=-1)

        # only score pairs that are close enough in the ground truth
        is_na_L = is_dna[tok_idx][first_index] | is_rna[tok_idx][first_index]
        pair_mask = torch.logical_and(
            ground_truth_distances>0,
            ground_truth_distances<torch.where(is_na_L, 30.0, 15.0)
        )
        del is_na_L

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

        if is_virtual is not None:
            pair_mask[:, (is_virtual[first_index] * is_virtual[second_index])] *= alpha_virtual

        # I assume gradients flow better if we sum first rather than keeping everything in D, L...
        lddt = 0.25*(
            torch.sum( torch.sigmoid( 0.5 - delta_distances )*pair_mask, dim=(1) )
            +torch.sum( torch.sigmoid( 1.0 - delta_distances )*pair_mask, dim=(1) )
            +torch.sum( torch.sigmoid( 2.0 - delta_distances )*pair_mask, dim=(1) )
            +torch.sum( torch.sigmoid( 4.0 - delta_distances )*pair_mask, dim=(1) )
        ) / (torch.sum( pair_mask, dim=(1) ) + eps)

        if not return_extras:
            return 1-lddt

        # ...Hence we recalculate the losses here and pick out the parts of interest
        with torch.no_grad():
            lddt_ = 0.25 * (
                torch.sigmoid( 0.5 - delta_distances )
                +torch.sigmoid( 1.0 - delta_distances )
                +torch.sigmoid( 2.0 - delta_distances )
                +torch.sigmoid( 4.0 - delta_distances )
            ) * pair_mask / (torch.sum( pair_mask, dim=(1) ) + eps)

            def filter_lddt(mask, scale=1.0):
                mask = mask.to(pair_mask.dtype)
                if mask.ndim > 1:
                    mask = mask[0]
                mask = (mask[first_index] * mask[second_index])[None].expand(pair_mask.shape[0],-1)
                mask = (mask * pair_mask).to(bool)
                return (1 - torch.sum(lddt_[:, mask[0]] * scale, dim=(1))).mean().detach().cpu()

            extra_lddts = {}
            extra_lddts['mean_lddt'] = filter_lddt(torch.full_like(crd_mask_L, 1.0, device=X_L.device))
            extra_lddts['mean_lddt_dna'] = filter_lddt(is_dna[tok_idx])
            extra_lddts['mean_lddt_rna'] = filter_lddt(is_rna[tok_idx])
            extra_lddts['mean_lddt_protein'] = filter_lddt(~is_dna[tok_idx] & ~is_rna[tok_idx])
            if is_virtual is not None:
                extra_lddts['mean_lddt_virtual'] = filter_lddt(is_virtual, scale=1/alpha_virtual)
                extra_lddts['mean_lddt_non_virtual'] = filter_lddt(~is_virtual)

        return 1-lddt, extra_lddts

    return _dolddt(X_L, X_gt_L, crd_mask_L, is_dna, is_rna, tok_idx, eps)


class SequenceLoss(nn.Module):

    def __init__(self, weight, min_t=0, max_t=torch.inf):
        super().__init__()
        self.weight = weight
        self.min_t = min_t
        self.max_t = max_t
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(
        self, 
        network_input,
        network_output,
        loss_input
    ):
        t = network_input["t"]
        valid_t_mask = (t >= self.min_t) & (t < self.max_t)

        pred_seq = network_output["Seq_I"][valid_t_mask]
        seq = loss_input["Seq_gt_I"]
        if seq.ndim == 1:
            seq = seq[None].repeat(pred_seq.shape[0], 1)
        else:
            seq = seq[valid_t_mask]

        best_guess_seq = pred_seq.argmax(dim=-1)
        seq_recovery = (best_guess_seq == seq).float().mean(dim=-1)
        loss = self.ce_loss(pred_seq.permute(0,2,1), seq)
        loss = loss.mean(dim=-1)
        return self.weight * loss.mean(), {"token_lvl_sequence_loss": loss.detach(), "seq_recovery": seq_recovery.detach()}
