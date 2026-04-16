import torch
import torch.nn as nn

from foundry.training.checkpoint import activation_checkpointing


class SequenceLoss(nn.Module):
    def __init__(self, weight, min_t=0, max_t=torch.inf):
        super().__init__()
        self.weight = weight
        self.min_t = min_t
        self.max_t = max_t
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")

    def forward(self, network_input, network_output, loss_input):
        t = network_input["t"]  # (B,)
        valid_t = (self.min_t <= t) & (t < self.max_t)  # bool mask over batch
        n_valid_t = valid_t.sum()

        # Grab network outputs
        sequence_logits_I = network_output["sequence_logits_I"]  # (B, L, 32)
        sequence_indices_I = network_output["sequence_indices_I"]  # (B, L)

        if n_valid_t == 0:
            zero = sequence_logits_I.sum() * 0.0
            return zero, {
                "valid_t_fraction": torch.tensor([0.0]),
                "n_valid_t": torch.tensor([0.0]),
            }

        pred_seq = sequence_logits_I[valid_t]  # (V, L, 32)
        gt_seq = loss_input["seq_token_lvl"]  # [L,]
        gt_seq = gt_seq.unsqueeze(0).expand(n_valid_t, -1)  # (V, L)
        w_seq = loss_input["sequence_valid_mask"]  # [L,]

        # Cross‑entropy token loss
        token_loss = self.loss_fn(pred_seq.permute(0, 2, 1), gt_seq)  # (V, L)
        token_loss = token_loss * w_seq[None]  # (V, L)
        token_loss = token_loss.mean(dim=-1)  # (V,)

        _, order = torch.sort(t[valid_t])  # low‑t first
        sequence_indices_I = sequence_indices_I[valid_t]
        recovery = (sequence_indices_I == gt_seq).float()  # (V, L)
        recovery = recovery[order]  # reorder by t
        recovery = recovery[..., (w_seq > 0).bool()]  # [V, L_valid]
        lowest_t_rec = recovery[0].mean()  # scalar

        outs = {
            "token_lvl_sequence_loss": token_loss.mean().detach(),
            "seq_recovery": recovery.mean().detach(),
            "lowest_t_seq_recovery": lowest_t_rec.detach(),
            "valid_t_fraction": valid_t.float().mean().detach(),
            "n_valid_t": n_valid_t.float(),
        }
        token_loss = torch.clamp(token_loss.mean(), max=4)
        return self.weight * token_loss, outs


class DiffusionLoss(nn.Module):
    def __init__(
        self,
        *,
        weight,
        sigma_data,
        lddt_weight,
        alpha_virtual_atom=1.0,
        alpha_unindexed_diffused=1.0,
        alpha_polar_residues=1.0,
        alpha_ligand=2.0,
        unindexed_t_alpha=1.0,
        unindexed_norm_p=1.0,
        lp_weight=0.0,
        **_,  # dump args from old configs
    ):
        super().__init__()
        self.weight = weight
        self.lddt_weight = lddt_weight
        self.sigma_data = sigma_data

        self.alpha_unindexed_diffused = alpha_unindexed_diffused
        self.alpha_virtual_atom = alpha_virtual_atom
        self.unindexed_norm_p = unindexed_norm_p
        self.unindexed_t_alpha = unindexed_t_alpha
        self.lp_weight = lp_weight
        self.alpha_ligand = alpha_ligand
        self.alpha_polar_residues = alpha_polar_residues

        self.get_lambda = (
            lambda sigma: (sigma**2 + self.sigma_data**2)
            / (sigma * self.sigma_data) ** 2
        )

    def forward(self, network_input, network_output, loss_input):
        X_L = network_output["X_L"]  # D, L, 3
        D = X_L.shape[0]
        crd_mask_L = loss_input["crd_mask_L"]  # (D, L)
        crd_mask_L = crd_mask_L.unsqueeze(0).expand(D, -1)
        tok_idx = network_input["f"]["atom_to_token_map"]
        t = network_input["t"]  # (D,)
        is_original_unindexed_token = loss_input["is_original_unindexed_token"][tok_idx]
        is_polar_atom = network_input["f"]["is_polar"][tok_idx]
        is_ligand = network_input["f"]["is_ligand"][tok_idx]
        is_virtual_atom = network_input["f"]["is_virtual"]  # L
        is_sidechain_atom = network_input["f"]["is_sidechain"]  # L
        is_sidechain_atom = is_sidechain_atom & ~is_virtual_atom

        w_L = torch.ones_like(tok_idx, dtype=X_L.dtype)
        w_L[is_original_unindexed_token] = (
            w_L[is_original_unindexed_token] * self.alpha_unindexed_diffused
        )
        w_L[is_virtual_atom] *= self.alpha_virtual_atom
        w_L[is_ligand] *= self.alpha_ligand

        # Upweight polar residues
        w_L[is_polar_atom] *= self.alpha_polar_residues
        w_L = w_L[None].expand(D, -1) * crd_mask_L

        X_gt_L = torch.nan_to_num(loss_input["X_gt_L_in_input_frame"])
        l_mse_L = w_L * torch.sum((X_L - X_gt_L) ** 2, dim=-1)
        l_mse_L = torch.div(l_mse_L, 3 * torch.sum(crd_mask_L[0]) + 1e-4)  # D, L

        if torch.any(is_original_unindexed_token):
            t_exp = t[:, None].expand(-1, X_L.shape[1])  # [D, L]
            t_exp = (
                t_exp * (~is_original_unindexed_token)
                + self.unindexed_t_alpha * t_exp * is_original_unindexed_token
            )

            l_global = (self.get_lambda(t_exp) * l_mse_L).sum(-1)

            # Get renormalization factor to equalize expectation of the loss
            r = self.get_lambda(t * self.unindexed_t_alpha) / self.get_lambda(t)
            t_factor = crd_mask_L.sum(-1) / (
                r * crd_mask_L[:, is_original_unindexed_token].sum(-1)
                + crd_mask_L[:, ~is_original_unindexed_token].sum(-1)
            )
            assert t_factor.shape == (D,), t_factor.shape
            l_global = l_global * t_factor
        else:
            l_global = self.get_lambda(t) * l_mse_L.sum(-1)

        assert l_global.shape == (D,), l_global.shape

        if torch.any(is_original_unindexed_token):
            lp_norm_L = w_L * torch.linalg.norm(
                X_L - X_gt_L, ord=self.unindexed_norm_p, dim=-1
            )  # [D, L]
            lp_norm_unindexed_diffused = lp_norm_L * is_original_unindexed_token[None]
            lp_norm_unindexed_diffused = torch.div(
                lp_norm_unindexed_diffused,
                self.alpha_unindexed_diffused
                * 3
                * torch.sum(is_original_unindexed_token)
                + 1e-4,
            )  # D, L
            lp_norm_unindexed_diffused = lp_norm_unindexed_diffused.sum(
                -1
            ) * self.get_lambda(self.unindexed_t_alpha * t)

            l_total = l_global + self.lp_weight * lp_norm_unindexed_diffused
        else:
            lp_norm_unindexed_diffused = None
            lp_norm_L = None
            l_total = l_global

        # ... Aggregate
        l_mse_total = torch.clamp(l_total, max=2)
        assert l_mse_total.shape == (
            D,
        ), f"Expected l_total to be of shape (D,), got {l_total.shape}"
        l_mse_total = torch.mean(l_mse_total)  # D, -> scalar

        # ... Return
        if self.lddt_weight > 0:
            # ... Calculate LDDT loss at the beginning
            smoothed_lddt_loss_, lddt_loss_dict = smoothed_lddt_loss(
                X_L,
                X_gt_L,
                crd_mask_L,
                network_input["f"]["is_dna"],
                network_input["f"]["is_rna"],
                tok_idx,
                return_extras=True,
            )  # D,
            l_total = l_mse_total + self.lddt_weight * smoothed_lddt_loss_.mean()
        else:
            lddt_loss_dict = {}
            l_total = l_mse_total
        # ... Return additional losses
        t, indices = torch.sort(t)
        l_mse_low, l_mse_high = torch.split(l_global[indices], [D // 2, D - D // 2])
        loss_dict = {
            "mse_loss_mean": l_mse_total,
            "mse_loss_low_t": l_mse_low,
            "mse_loss_high_t": l_mse_high,
            "lp_norm": lp_norm_L,
            "lp_norm_unindexed_diffused": lp_norm_unindexed_diffused,
        } | lddt_loss_dict
        loss_dict = {
            k: torch.mean(v).detach() for k, v in loss_dict.items() if v is not None
        }

        return self.weight * l_total, loss_dict


def smoothed_lddt_loss(
    X_L,
    X_gt_L,
    crd_mask_L,
    is_dna,
    is_rna,
    tok_idx,
    is_virtual=None,
    alpha_virtual=1.0,
    return_extras=False,
    eps=1e-6,
):
    @activation_checkpointing
    def _dolddt(X_L, X_gt_L, crd_mask_L, is_dna, is_rna, tok_idx, eps, use_amp=True):
        B, L = X_L.shape[:2]
        first_index, second_index = torch.triu_indices(L, L, 1, device=X_L.device)

        # compute the unique distances between all pairs of atoms
        X_gt_L = X_gt_L.nan_to_num()

        # only use native 1 (assumes dist map identical btwn all copies)
        ground_truth_distances = torch.linalg.norm(
            X_gt_L[0:1, first_index] - X_gt_L[0:1, second_index], dim=-1
        )

        # only score pairs that are close enough in the ground truth
        is_na_L = is_dna[tok_idx][first_index] | is_rna[tok_idx][first_index]
        pair_mask = torch.logical_and(
            ground_truth_distances > 0,
            ground_truth_distances < torch.where(is_na_L, 30.0, 15.0),
        )
        del is_na_L

        # only score pairs that are resolved in the ground truth
        pair_mask *= crd_mask_L[0:1, first_index] * crd_mask_L[0:1, second_index]

        # don't score pairs that are in the same token
        pair_mask *= tok_idx[None, first_index] != tok_idx[None, second_index]

        _, valid_pairs = pair_mask.nonzero(as_tuple=True)
        pair_mask = pair_mask[:, valid_pairs].to(X_L.dtype)
        ground_truth_distances = ground_truth_distances[:, valid_pairs]
        first_index, second_index = first_index[valid_pairs], second_index[valid_pairs]

        predicted_distances = torch.linalg.norm(
            X_L[:, first_index] - X_L[:, second_index], dim=-1
        )

        delta_distances = torch.abs(predicted_distances - ground_truth_distances + eps)
        del predicted_distances, ground_truth_distances

        if is_virtual is not None:
            pair_mask[:, (is_virtual[first_index] * is_virtual[second_index])] *= (
                alpha_virtual
            )

        # I assume gradients flow better if we sum first rather than keeping everything in D, L...
        lddt = (
            0.25
            * (
                torch.sum(torch.sigmoid(0.5 - delta_distances) * pair_mask, dim=(1))
                + torch.sum(torch.sigmoid(1.0 - delta_distances) * pair_mask, dim=(1))
                + torch.sum(torch.sigmoid(2.0 - delta_distances) * pair_mask, dim=(1))
                + torch.sum(torch.sigmoid(4.0 - delta_distances) * pair_mask, dim=(1))
            )
            / (torch.sum(pair_mask, dim=(1)) + eps)
        )

        if not return_extras:
            return 1 - lddt

        # ...Hence we recalculate the losses here and pick out the parts of interest
        with torch.no_grad():
            lddt_ = (
                0.25
                * (
                    torch.sigmoid(0.5 - delta_distances)
                    + torch.sigmoid(1.0 - delta_distances)
                    + torch.sigmoid(2.0 - delta_distances)
                    + torch.sigmoid(4.0 - delta_distances)
                )
                * pair_mask
                / (torch.sum(pair_mask, dim=(1)) + eps)
            )

            def filter_lddt(mask, scale=1.0):
                mask = mask.to(pair_mask.dtype)
                if mask.ndim > 1:
                    mask = mask[0]
                mask = (mask[first_index] * mask[second_index])[None].expand(
                    pair_mask.shape[0], -1
                )
                mask = (mask * pair_mask).to(bool)
                return (
                    (1 - torch.sum(lddt_[:, mask[0]] * scale, dim=(1)))
                    .mean()
                    .detach()
                    .cpu()
                )

            extra_lddts = {}
            extra_lddts["mean_lddt"] = filter_lddt(
                torch.full_like(crd_mask_L, 1.0, device=X_L.device)
            )
            extra_lddts["mean_lddt_dna"] = filter_lddt(is_dna[tok_idx])
            extra_lddts["mean_lddt_rna"] = filter_lddt(is_rna[tok_idx])
            extra_lddts["mean_lddt_protein"] = filter_lddt(
                ~is_dna[tok_idx] & ~is_rna[tok_idx]
            )
            # NOTE: This also seems to have issues at epoch level, as with n_valid_t
            # before. Will leave as-is for now but may want to spoof as 0 in the future.
            if is_virtual is not None:
                extra_lddts["mean_lddt_virtual"] = filter_lddt(
                    is_virtual, scale=1 / alpha_virtual
                )
                extra_lddts["mean_lddt_non_virtual"] = filter_lddt(~is_virtual)

        return 1 - lddt, extra_lddts

    return _dolddt(X_L, X_gt_L, crd_mask_L, is_dna, is_rna, tok_idx, eps)
