import torch
import torch.nn as nn
from rf3.chemical import NFRAMES, NHEAVY, frame_indices

# TODO: REFACTOR; COPIED FROM RF2AA. WE NEED TO ADD DOCSTRINGS, EXAMPLES, HOPEFULLY TESTS, AND CLEAN UP
from rf3.metrics.metric_utils import (
    compute_mean_over_subsampled_pairs,
    unbin_logits,
)
from rf3.utils.frames import (
    get_frames,
    mask_unresolved_frames_batched,
    rigid_from_3_points,
)
from scipy.stats import spearmanr


class ConfidenceLoss(nn.Module):
    def __init__(
        self,
        plddt,
        pae,
        pde,
        exp_resolved,
        weight=1,
        rank_loss=None,
        log_statistics=False,
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
        X_gt_L = loss_input["X_gt_L"]
        X_exists_L = loss_input["crd_mask_L"]
        X_pred_L = network_output["X_pred_rollout_L"]
        B = X_pred_L.shape[0]
        I = loss_input["is_real_atom"].shape[0]

        true_lddt_binned, is_resolved_I = self.calc_lddt(
            X_pred_L, X_gt_L, X_exists_L, loss_input["seq"], loss_input["is_real_atom"]
        )

        plddt_logits = (
            network_output["plddt"]
            .reshape(-1, I, NHEAVY, self.plddt.n_bins)
            .permute(0, 3, 1, 2)
        )
        plddt_loss = (
            self.cce(
                plddt_logits,
                true_lddt_binned[..., :NHEAVY].long(),
            )
            * is_resolved_I[..., :NHEAVY]
        )
        plddt_loss = plddt_loss.sum() / (is_resolved_I.sum() + self.eps)

        pae_logits = network_output["pae"]
        true_pae_binned, pae_logits, valid_pae_pairs = self.calc_pae(
            loss_input,
            X_pred_L,
            X_gt_L,
            X_exists_L,
            pae_logits,
            loss_input["frame_atom_idxs"],
        )
        pae_loss = self.cce(pae_logits, true_pae_binned) * valid_pae_pairs
        pae_loss = pae_loss.sum() / (valid_pae_pairs.sum() + self.eps)

        true_pde_binned, is_valid_pair = self.calc_pde(
            X_pred_L, X_gt_L, X_exists_L, loss_input["rep_atom_idxs"]
        )
        pde_logits = network_output["pde"].permute(0, 3, 1, 2)
        pde_loss = self.cce(pde_logits, true_pde_binned) * is_valid_pair
        pde_loss = pde_loss.sum() / (is_valid_pair.sum() + self.eps)

        exp_resolved_logits = network_output["exp_resolved"]
        exp_resolved_loss = (
            self.cce(
                exp_resolved_logits.reshape(
                    B, I, NHEAVY, self.exp_resolved.n_bins
                ).permute(0, 3, 1, 2),
                is_resolved_I[:, :, :NHEAVY].long(),
            )
            * loss_input["is_real_atom"][:, :NHEAVY]
        )
        exp_resolved_loss = exp_resolved_loss.sum() / (
            loss_input["is_real_atom"][:, :NHEAVY].sum() + self.eps
        )
        exp_resolved_loss = exp_resolved_loss / B

        loss_dict = dict(
            plddt_loss=plddt_loss.detach(),
            pae_loss=pae_loss.detach(),
            pde_loss=pde_loss.detach(),
            exp_resolved_loss=exp_resolved_loss.detach(),
        )

        confidence_loss = (
            self.plddt.weight * plddt_loss
            + self.pae.weight * pae_loss
            + self.pde.weight * pde_loss
            + self.exp_resolved.weight * exp_resolved_loss
        )

        if self.log_statistics or self.rank_loss.use_listnet_loss:
            # Get correlations across and within batches
            # Get the true values per metric
            true_lddt, true_lddt_per_structure = self.get_true_metrics(
                true_lddt_binned, self.plddt, is_resolved_I
            )
            true_pae, true_pae_per_structure = self.get_true_metrics(
                true_pae_binned, self.pae, valid_pae_pairs
            )
            true_pde, true_pde_per_structure = self.get_true_metrics(
                true_pde_binned, self.pde, is_valid_pair
            )

            # reorder the input tensors to be in (B, n_bins, ...) format for unbinning
            # pae and pde were already reordered above
            plddt_logit_stack = network_output["plddt"]
            plddt_per_structure = unbin_logits(
                plddt_logit_stack.reshape(
                    -1,
                    I,
                    NHEAVY,
                    self.plddt.n_bins,
                )
                .permute(0, 3, 1, 2)
                .float(),
                self.plddt.max_value,
                self.plddt.n_bins,
            )
            pae_per_structure = unbin_logits(
                pae_logits, self.pae.max_value, self.pae.n_bins
            )
            pde_per_structure = unbin_logits(
                pde_logits, self.pde.max_value, self.pde.n_bins
            )

            plddt_per_structure = torch.cat(
                [
                    compute_mean_over_subsampled_pairs(
                        plddt_per_structure[i][None],
                        is_resolved_I[i, ..., :NHEAVY],
                    )
                    for i in range(plddt_logit_stack.shape[0])
                ],
                dim=0,
            )
            pae_per_structure = torch.cat(
                [
                    compute_mean_over_subsampled_pairs(
                        pae_per_structure[i][None], is_valid_pair[i]
                    )
                    for i in range(pae_per_structure.shape[0])
                ],
                dim=0,
            )
            pde_per_structure = torch.cat(
                [
                    compute_mean_over_subsampled_pairs(
                        pde_per_structure[i][None], is_valid_pair[i]
                    )
                    for i in range(pde_per_structure.shape[0])
                ],
                dim=0,
            )

            plddt = plddt_per_structure.mean()
            pae = pae_per_structure.mean()
            pde = pde_per_structure.mean()

            if self.log_statistics:
                self.log_correlation_statistics(
                    plddt,
                    pae,
                    pde,
                    true_lddt,
                    true_pae,
                    true_pde,
                    true_lddt_per_structure,
                    true_pae_per_structure,
                    true_pde_per_structure,
                    plddt_per_structure,
                    pae_per_structure,
                    pde_per_structure,
                    loss_dict,
                )

            if self.rank_loss.use_listnet_loss:
                # an easy way of incentivizing ranking accuracy is the following (Listnet):
                plddt_rank_loss = self.listnet_loss(
                    true_lddt_per_structure, plddt_per_structure
                )
                pae_rank_loss = self.listnet_loss(
                    true_pae_per_structure, pae_per_structure
                )
                pde_rank_loss = self.listnet_loss(
                    true_pde_per_structure, pde_per_structure
                )

                rank_loss_dict = dict(
                    plddt_rank_loss=plddt_rank_loss.detach(),
                    pae_rank_loss=pae_rank_loss.detach(),
                    pde_rank_loss=pde_rank_loss.detach(),
                )
                loss_dict.update(rank_loss_dict)
                confidence_loss += (
                    plddt_rank_loss + pae_rank_loss + pde_rank_loss
                ) * self.rank_loss.weight

        return self.weight * confidence_loss, loss_dict

    def calc_lddt(self, X_pred_L, X_gt_L, X_exists_L, seq, is_real_atom):
        tok_idx = is_real_atom.nonzero()[:, 0]

        I = is_real_atom.shape[0]
        B = X_pred_L.shape[0]

        # If structure is too big, split the batches to deal with a memory issue
        if I > 384:
            ground_truth_distances = torch.cdist(
                X_gt_L[: B // 2],
                X_gt_L[: B // 2],
                compute_mode="donot_use_mm_for_euclid_dist",
            )
            predicted_distances = torch.cdist(
                X_pred_L[: B // 2],
                X_pred_L[: B // 2],
                compute_mode="donot_use_mm_for_euclid_dist",
            )

            ground_truth_distances2 = torch.cdist(
                X_gt_L[B // 2 :],
                X_gt_L[B // 2 :],
                compute_mode="donot_use_mm_for_euclid_dist",
            )
            predicted_distances2 = torch.cdist(
                X_pred_L[B // 2 :],
                X_pred_L[B // 2 :],
                compute_mode="donot_use_mm_for_euclid_dist",
            )

            ground_truth_distances = torch.cat(
                (ground_truth_distances, ground_truth_distances2), dim=0
            )
            predicted_distances = torch.cat(
                (predicted_distances, predicted_distances2), dim=0
            )
        else:
            ground_truth_distances = torch.cdist(
                X_gt_L, X_gt_L, compute_mode="donot_use_mm_for_euclid_dist"
            )
            predicted_distances = torch.cdist(
                X_pred_L, X_pred_L, compute_mode="donot_use_mm_for_euclid_dist"
            )

        X_exists_LL = X_exists_L.unsqueeze(-1) * X_exists_L.unsqueeze(-2)

        difference_distances = torch.abs(ground_truth_distances - predicted_distances)
        lddt_matrix = torch.zeros_like(difference_distances)
        lddt_matrix = (
            0.25 * (difference_distances < 4.0)
            + 0.25 * (difference_distances < 2.0)
            + 0.25 * (difference_distances < 1.0)
            + 0.25 * (difference_distances < 0.5)
        )
        in_same_residue_LL = tok_idx.unsqueeze(-1) == tok_idx.unsqueeze(-2)
        close_distances_LL = ground_truth_distances < 15.0

        # include distances where both atoms are resolved and not in the same residue, and are within an inclusion radius (15A)
        mask_LL = X_exists_LL * ~in_same_residue_LL * close_distances_LL
        lddt_per_atom_L = (lddt_matrix * mask_LL).sum(-1) / (mask_LL.sum(-1) + self.eps)

        # only aggregate over the resolved atoms in each residue
        lddt_per_atom_I = torch.zeros_like(is_real_atom, dtype=torch.float32)
        lddt_per_atom_I = lddt_per_atom_I.unsqueeze(0).repeat(B, 1, 1)

        lddt_per_atom_I[:, is_real_atom] = lddt_per_atom_L
        X_exists_I = torch.zeros_like(is_real_atom, dtype=torch.bool)
        X_exists_I = X_exists_I.unsqueeze(0).repeat(B, 1, 1)
        X_exists_I[:, is_real_atom] = X_exists_L
        lddt_per_atom_binned = self.bin_values(
            lddt_per_atom_I, max_value=self.plddt.max_value, n_bins=self.plddt.n_bins
        )

        return lddt_per_atom_binned, X_exists_I

    def calc_pae(
        self,
        loss_input,
        X_pred_L,
        X_gt_L,
        X_exists_L,
        pae_logits,
        frame_atom_idxs,
        eps=1e-4,
    ):
        seq = loss_input["seq"]
        atom_frames = loss_input["atom_frames"]
        B = X_pred_L.shape[0]

        # Construct the backbone atoms in the faux atom-36 representation so we can use existing machinery to get frames
        frame_atom_idxs = frame_atom_idxs.unsqueeze(0).expand(B, -1, -1)
        X_pred_I = torch.zeros(B, seq.shape[-1], 36, 3, device=X_pred_L.device)
        X_pred_I[..., 0, :] = torch.gather(
            X_pred_L, 1, frame_atom_idxs[..., 0].unsqueeze(-1).expand(-1, -1, 3)
        )
        X_pred_I[..., 1, :] = torch.gather(
            X_pred_L, 1, frame_atom_idxs[..., 1].unsqueeze(-1).expand(-1, -1, 3)
        )
        X_pred_I[..., 2, :] = torch.gather(
            X_pred_L, 1, frame_atom_idxs[..., 2].unsqueeze(-1).expand(-1, -1, 3)
        )

        X_gt_I = torch.zeros(B, seq.shape[-1], 36, 3, device=X_gt_L.device)
        X_gt_I[..., 0, :] = torch.gather(
            X_gt_L, 1, frame_atom_idxs[..., 0].unsqueeze(-1).expand(-1, -1, 3)
        )
        X_gt_I[..., 1, :] = torch.gather(
            X_gt_L, 1, frame_atom_idxs[..., 1].unsqueeze(-1).expand(-1, -1, 3)
        )
        X_gt_I[..., 2, :] = torch.gather(
            X_gt_L, 1, frame_atom_idxs[..., 2].unsqueeze(-1).expand(-1, -1, 3)
        )

        atom_mask = torch.zeros(
            B, seq.shape[-1], 36, device=X_exists_L.device, dtype=torch.bool
        )
        atom_mask[..., 0] = torch.gather(X_exists_L, 1, frame_atom_idxs[..., 0])
        atom_mask[..., 1] = torch.gather(X_exists_L, 1, frame_atom_idxs[..., 1])
        atom_mask[..., 2] = torch.gather(X_exists_L, 1, frame_atom_idxs[..., 2])

        frames, frame_mask = get_frames(
            0,
            0,
            seq.unsqueeze(0).repeat(B, 1),
            frame_indices.to(seq.device),
            atom_frames,
        )

        N, L, natoms, _ = X_pred_I.shape

        # flatten middle dims so can gather across residues
        X_prime = X_pred_I.reshape(N, L * natoms, -1, 3).repeat(1, 1, NFRAMES, 1)
        Y_prime = X_gt_I.reshape(N, L * natoms, -1, 3).repeat(1, 1, NFRAMES, 1)
        frames_reindex_batched, frame_mask_batched = mask_unresolved_frames_batched(
            frames, frame_mask, atom_mask
        )

        X_x = torch.gather(
            X_prime, 1, frames_reindex_batched[..., 0:1].repeat(1, 1, 1, 3)
        )
        X_y = torch.gather(
            X_prime, 1, frames_reindex_batched[..., 1:2].repeat(1, 1, 1, 3)
        )
        X_z = torch.gather(
            X_prime, 1, frames_reindex_batched[..., 2:3].repeat(1, 1, 1, 3)
        )
        uX, tX = rigid_from_3_points(X_x, X_y, X_z)

        Y_x = torch.gather(
            Y_prime, 1, frames_reindex_batched[..., 0:1].repeat(1, 1, 1, 3)
        )
        Y_y = torch.gather(
            Y_prime, 1, frames_reindex_batched[..., 1:2].repeat(1, 1, 1, 3)
        )
        Y_z = torch.gather(
            Y_prime, 1, frames_reindex_batched[..., 2:3].repeat(1, 1, 1, 3)
        )
        uY, tY = rigid_from_3_points(Y_x, Y_y, Y_z)

        uX = uX[:, :, 0]
        uY = uY[:, :, 0]

        # Compute xij_ca across the batch
        # uX: (B, L, 3), X_pred_I: (B, A, 3), X_y: (B, L, 3)
        xij_ca = torch.einsum(
            "bfji,bfaj->bfai",
            uX,  # select valid frames for backbone, shape (B, N_valid_frames, 3)
            X_pred_I[:, None, :, 1] - X_y[:, :, None, 0],
        )  # Result: (B, N_valid_frames, N_valid_ca, 3)

        # Compute xij_ca_t across the batch
        # uY: (B, L, 3), X_gt_I: (B, A, 3), Y_y: (B, L, 3)
        xij_ca_t = torch.einsum(
            "bfji,bfaj->bfai",
            uY,  # select valid frames for backbone, shape (B, N_valid_frames, 3)
            X_gt_I[:, None, :, 1] - Y_y[:, :, None, 0],
        )  # Result: (B, N_valid_frames, N_valid_ca, 3)

        valid_frames = frame_mask_batched[:, :, 0]  # valid backbone frames (B,I)
        valid_ca = atom_mask[:, :, 1]  # valid CA atoms (B,I)
        valid_pairs = (
            valid_frames[:, :, None] & valid_ca[:, None, :]
        )  # valid pairs (B,I,I)

        eij_label = (
            torch.sqrt(torch.square(xij_ca - xij_ca_t).sum(dim=-1) + eps)
            .clone()
            .detach()
        )
        true_pae_label = self.bin_values(
            eij_label, max_value=self.pae.max_value, n_bins=self.pae.n_bins
        )
        pae_logits = pae_logits.permute(0, 3, 1, 2)  # (1, nbins, N_frames, N_ca)

        return true_pae_label.detach(), pae_logits, valid_pairs

    def calc_pde(self, X_pred_L, X_gt_L, X_exists_L, rep_atoms):
        X_pred_I = X_pred_L.index_select(1, rep_atoms)
        X_gt_I = X_gt_L.index_select(1, rep_atoms)
        X_exists_I = X_exists_L.index_select(1, rep_atoms)
        predicted_distances = torch.cdist(
            X_pred_I, X_pred_I, compute_mode="donot_use_mm_for_euclid_dist"
        )
        ground_truth_distances = torch.cdist(
            X_gt_I, X_gt_I, compute_mode="donot_use_mm_for_euclid_dist"
        )
        difference_distances = torch.abs(ground_truth_distances - predicted_distances)
        true_pde_binned = self.bin_values(
            difference_distances, max_value=self.pde.max_value, n_bins=self.pde.n_bins
        )
        X_exists_II = X_exists_I.unsqueeze(-1) * X_exists_I.unsqueeze(-2)
        return true_pde_binned.detach(), X_exists_II.detach()

    def bin_values(self, values, max_value, n_bins):
        # assumes that the bins go from 0 to max_value
        bin_size = max_value / n_bins
        bins = torch.linspace(
            bin_size, max_value - bin_size, n_bins - 1, device=values.device
        )
        return torch.bucketize(values, bins, right=True)

    def log_correlation_statistics(
        self,
        plddt,
        pae,
        pde,
        true_lddt,
        true_pae,
        true_pde,
        true_lddt_per_structure,
        true_pae_per_structure,
        true_pde_per_structure,
        plddt_per_structure,
        pae_per_structure,
        pde_per_structure,
        loss_dict,
    ):
        # Calculate Spearman rank correlation
        plddt_rank_corr, lddt_spearman_p = spearmanr(
            true_lddt_per_structure.cpu().numpy(), plddt_per_structure.cpu().numpy()
        )
        pae_rank_corr, pae_spearman_p = spearmanr(
            true_pae_per_structure.cpu().numpy(), pae_per_structure.cpu().numpy()
        )
        pde_rank_corr, pde_spearman_p = spearmanr(
            true_pde_per_structure.cpu().numpy(), pde_per_structure.cpu().numpy()
        )

        loss_dict.update(
            {
                "pred_err_plddt": plddt,
                "pred_err_pae": pae,
                "pred_err_pde": pde,
                "true_err_plddt": true_lddt,
                "true_err_pae": true_pae,
                "true_err_pde": true_pde,
                "plddt_rank_corr": torch.tensor(plddt_rank_corr),
                "pae_rank_corr": torch.tensor(pae_rank_corr),
                "pde_rank_corr": torch.tensor(pde_rank_corr),
                "plddt_spread": plddt_per_structure.max() - plddt_per_structure.min(),
                "pae_spread": pae_per_structure.max() - pae_per_structure.min(),
                "pde_spread": pde_per_structure.max() - pde_per_structure.min(),
                "true_plddt_spread": true_lddt_per_structure.max()
                - true_lddt_per_structure.min(),
                "true_pae_spread": true_pae_per_structure.max()
                - true_pae_per_structure.min(),
                "true_pde_spread": true_pde_per_structure.max()
                - true_pde_per_structure.min(),
            }
        )

    def get_true_metrics(self, true_metric_binned, metric_config, mask):
        # Calculate the true metric values from the binned values along with the per structure metrics
        bin_size = metric_config.max_value / metric_config.n_bins
        true_metric_unbinned = (
            (true_metric_binned.detach() + 1) * bin_size - (bin_size / 2)
        ) * mask
        true_metric_per_structure = true_metric_unbinned.sum(dim=(1, 2)) / (
            mask.sum(dim=(1, 2)) + self.eps
        )
        true_metric = true_metric_unbinned.sum() / (mask.sum() + self.eps)

        return true_metric, true_metric_per_structure

    def listnet_loss(self, true_metric_per_structure, pred_metric_per_structure):
        # Calculate the ListNet loss
        rank_true = torch.nn.Softmax(dim=0)(true_metric_per_structure)
        rank_pred = torch.nn.Softmax(dim=0)(pred_metric_per_structure)
        return -torch.mean(rank_true * torch.log(rank_pred))
