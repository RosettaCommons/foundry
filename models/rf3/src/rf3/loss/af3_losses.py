import numpy as np
import torch
import torch.nn as nn

from foundry.training.checkpoint import activation_checkpointing
from foundry.utils.alignment import weighted_rigid_align


# resolve residue-level symmetries in native vs pred
class ResidueSymmetryResolution(nn.Module):
    def _get_best(self, x_pred, x_native, x_native_mask, a_i):
        mask = torch.zeros_like(x_native_mask[0])
        mask[a_i[0]] = True
        d_pred = torch.cdist(x_pred[:, mask], x_pred[:, ~mask])
        x_nat_j = x_native.clone()
        for j in range(a_i.shape[0]):
            x_nat_j[:, a_i[0]] = x_native[:, a_i[j]]
            d_nat = torch.cdist(x_nat_j[:, mask], x_nat_j[:, ~mask])
            drms_j = torch.square(d_pred - d_nat).nan_to_num()
            drms_j[drms_j > 15] = 15
            drms_j = torch.mean(drms_j, dim=(-1, -2))
            if j == 0:
                bestj = torch.zeros(
                    x_pred.shape[0], dtype=torch.long, device=x_pred.device
                )
                bestrms = drms_j
            else:
                bestj[drms_j < bestrms] = j
                bestrms[drms_j < bestrms] = drms_j[drms_j < bestrms]
        # x_nat_j[:,a_i[0]] = x_native[:,a_i[j]]
        for j in range(x_pred.shape[0]):
            x_native[j, a_i[0]] = x_native[j, a_i[bestj[j]]]
            x_native_mask[j, a_i[0]] = x_native_mask[j, a_i[bestj[j]]]

        return x_native, x_native_mask

    def forward(self, network_output, loss_input, automorph_input):
        x_pred = network_output["X_L"]
        x_native = loss_input["X_gt_L"]
        x_native_mask = loss_input["crd_mask_L"]
        for a_i in automorph_input:
            if a_i.shape[0] == 1:
                continue
            a_i = torch.tensor(a_i, device=x_pred.device)
            x_native, x_native_mask = self._get_best(
                x_pred, x_native, x_native_mask, a_i
            )

        loss_input["X_gt_L"] = x_native
        loss_input["crd_mask_L"] = x_native_mask

        return loss_input


# Resolve subunit-level symmetries in native vs pred
class SubunitSymmetryResolution(nn.Module):
    def __init__(self, **losses):
        super().__init__()

    def _rms_align(self, X_fixed, X_moving):
        # input:
        #   X_fixed = predicted = Nbatch x L x 3
        #   X_moving = native = Nambig x L x 3
        # output:
        #   X_pre = Nambig x Nbatch x 3
        #   U = Nambig x Nbatch x 3 x 3
        #   X_post = Nambig x Nbatch x 3
        assert X_fixed.shape[-2:] == X_moving.shape[-2:]
        Nbatch = X_fixed.shape[0]
        Nambig = X_moving.shape[0]
        X_fixed = X_fixed[None, :]
        X_moving = X_moving[:, None]

        u_X_fixed = torch.mean(X_fixed, dim=-2)
        u_X_moving = torch.mean(X_moving, dim=-2)

        X_fixed = X_fixed - u_X_fixed.unsqueeze(-2)
        X_moving = X_moving - u_X_moving.unsqueeze(-2)

        C = torch.einsum("...ji,...jk->...ik", X_moving, X_fixed)
        U, S, V = torch.linalg.svd(C)
        R = U @ V
        F = torch.eye(3, 3, device=X_fixed.device)[None, None].repeat(
            Nambig, Nbatch, 1, 1
        )
        F[..., -1, -1] = torch.sign(torch.linalg.det(R))
        R = U @ F @ V
        return u_X_moving, R, u_X_fixed

    def _greedy_resolve_mapping(
        self,
        dist,
        iid_to_index,
        entity_to_index,
        iids_by_entity,
        entity_by_iids,
        nmodel_by_iid,
    ):
        # returns:
        #    best_xform      tensor [i]->transform number
        #    best_assignment dict{pred_iid:[native_iids]} (batch)
        nTransforms = dist.shape[0]
        nIid = dist.shape[1]
        nBatch = dist.shape[-1]
        toAssign = [k for k, v in nmodel_by_iid.items() if v > 0]

        # sort equiv groups by # resolved residues
        # first make that list
        nmodel_by_equiv = {
            int(i): 0 for i in entity_to_index.keys()
        }  # torch.zeros(nEquiv,dtype=torch.long,device=dist.device)
        for i, iid in enumerate(toAssign):
            nmodel_by_equiv[entity_by_iids[iid]] += nmodel_by_iid[iid]
        equiv_order = sorted(
            nmodel_by_equiv, key=nmodel_by_equiv.get
        )  # torch.argsort(nmodel_by_equiv,descending=True)

        best_cost = torch.zeros(nBatch, device=dist.device)
        best_xform = torch.zeros(nBatch, dtype=torch.long, device=dist.device)
        best_assignment = {
            int(i): torch.zeros(nBatch, dtype=torch.long, device=dist.device)
            for i in toAssign
        }
        for t in range(nTransforms):
            # then sort with most res first
            cost = torch.zeros(nBatch, device=dist.device)
            assignment = {
                int(i): torch.full(
                    (nBatch,), int(i), dtype=torch.long, device=dist.device
                )
                for i in toAssign
            }

            for i_equiv in equiv_order:
                mask_equiv = torch.zeros(
                    (nIid, nIid), dtype=torch.bool, device=dist.device
                )
                iids_in_i_equiv = iids_by_entity[i_equiv]
                nIids_in_i_equiv = iids_in_i_equiv.shape[0]
                iid_idxs_in_i_equiv = np.vectorize(iid_to_index.__getitem__)(
                    iids_in_i_equiv
                )

                nResolvedEntities_i = len(
                    [
                        nmodel_by_iid[int(i)]
                        for i in iids_in_i_equiv
                        if nmodel_by_iid[i] > 0
                    ]
                )

                mask_equiv[
                    iid_idxs_in_i_equiv[:, None], iid_idxs_in_i_equiv[None, :]
                ] = True
                wted_dist = dist[t, mask_equiv].nan_to_num(1e9)

                # greedily assign min RMS within each equiv group
                # print ('work on eq group',iid_idxs_in_i_equiv)
                # print ('toAssign',toAssign)
                for i in range(nResolvedEntities_i):
                    wted_dist = wted_dist.view(
                        nIids_in_i_equiv * nIids_in_i_equiv, nBatch
                    )
                    pn = torch.argmin(wted_dist, dim=0)

                    # special case: if there is NO seq overlap between predicted and native peptides,
                    # fall back to identity assignment
                    if (wted_dist[pn] == 1e9).all():
                        break

                    # weight the total cost by #residues
                    cost += (
                        wted_dist[pn, torch.arange(nBatch, device=wted_dist.device)]
                        * nmodel_by_iid[iids_in_i_equiv[i]]
                    )
                    i_nat, i_pred = pn // nIids_in_i_equiv, pn % nIids_in_i_equiv
                    for j, (ii_nat, ii_pred) in enumerate(zip(i_nat, i_pred)):
                        assignment[int(iids_by_entity[int(i_equiv)][ii_pred])][j] = (
                            iids_by_entity[int(i_equiv)][ii_nat]
                        )

                    wted_dist = wted_dist.view(
                        nIids_in_i_equiv, nIids_in_i_equiv, nBatch
                    )
                    for i in range(i_nat.shape[0]):
                        wted_dist[i_nat[i], :, i] = 1e6
                        wted_dist[:, i_pred[i], i] = 1e6
            if t == 0:
                best_cost = cost
                best_assignment = assignment
            else:
                mask = cost < best_cost
                best_cost[mask] = cost[mask]
                for i, bi in best_assignment.items():
                    best_assignment[i][mask] = assignment[i][mask]
                best_xform[mask] = t

        return (best_xform, best_assignment)

    def _resolve_subunits(
        self, mol_entities, mol_iid, crop_mask, x_native, mask_native, x_pred
    ):
        # print('x_native',x_native.shape, x_native)
        Nbatch = x_pred.shape[0]

        # index -> entity
        all_entities = torch.unique(mol_entities)
        # entity -> index
        entity_to_index = {int(ii): i for i, ii in enumerate(all_entities)}

        # index -> iid
        all_iids = torch.unique(mol_iid).cpu().numpy()
        Niids = len(all_iids)
        # iid -> index
        iid_to_index = {int(ii): i for i, ii in enumerate(all_iids)}

        # entity -> iid list
        iids_by_entity = {
            int(i): torch.unique(mol_iid[mol_entities == i]).long().cpu().numpy()
            for i in all_entities
        }
        # iid -> entity list
        entity_by_iids = {
            int(i): torch.unique(mol_entities[mol_iid == i]).long().cpu().item()
            for i in all_iids
        }

        # 1) get the iid with most resolved residues
        mask = torch.zeros(
            mol_entities.shape[0], dtype=torch.bool, device=mol_iid.device
        )
        mask[crop_mask] = 1
        mask_by_iid = {int(i): mask[mol_iid == i] for i in all_iids}
        mask_native_by_iid = {int(i): mask_native[mol_iid == i] for i in all_iids}
        nmodeled_by_iid = {
            int(i): torch.sum(mask_by_iid[i]) for i in mask_native_by_iid.keys()
        }

        iid_src_idx = max(
            nmodeled_by_iid, key=nmodeled_by_iid.get
        )  # int(nmodeled_by_iid.argmax())
        entity_src_idx = entity_by_iids[iid_src_idx]
        native_by_iid = {int(i): x_native[mol_iid == i] for i in all_iids}
        pred_by_iid = {int(ii): x_pred[:, mol_iid[crop_mask] == ii] for ii in all_iids}

        # align it to all equivalent targets
        equiv_native_iids = iids_by_entity[entity_src_idx]

        # output:
        #   xpres = Ntrans x Nbatch x 3
        #   U = Ntrans x Nbatch x 3 x 3
        #   xposts = Ntrans x Nbatch x 3
        xpres, Us, xposts = [], [], []

        for n in equiv_native_iids:
            nat_n = native_by_iid[int(n)][mask_by_iid[int(iid_src_idx)]]
            pred_n = pred_by_iid[int(iid_src_idx)]
            mask_unres = ~nat_n[..., 0].isnan()
            nat_n = nat_n[mask_unres]
            pred_n = pred_n[:, mask_unres]

            if mask_unres.sum() > 3:
                xpre, U, xpost = self._rms_align(pred_n, nat_n[None])
                xpres.append(xpre)
                Us.append(U)
                xposts.append(xpost)

        xpres, Us, xposts = (
            torch.cat(xpres, dim=0),
            torch.cat(Us, dim=0),
            torch.cat(xposts, dim=0),
        )

        # build up the matrix of COMs
        # nat_com[i,j] = com of native iid i using crop mask from pred iid j (if compatible)
        nat_com = torch.full((Niids, Niids, 3), np.nan, device=Us.device)
        for i in all_iids:
            equiv_native_iids = iids_by_entity[entity_by_iids[i]]
            for j in equiv_native_iids:
                mask_ij = mask_by_iid[int(j)] * ~native_by_iid[int(i)][:, 0].isnan()
                if torch.any(mask_ij):
                    nat_com[iid_to_index[i], iid_to_index[j]] = torch.mean(
                        native_by_iid[int(i)][mask_ij], dim=0
                    )

        # pred_com[i,j] = com using native mask from iid i on pred iid j
        pred_com = torch.full((Niids, Niids, Nbatch, 3), np.nan, device=Us.device)
        for i in all_iids:
            equiv_native_iids = iids_by_entity[entity_by_iids[i]]
            for j in equiv_native_iids:
                mask_ij = ~native_by_iid[int(i)][:, 0].isnan()[mask_by_iid[int(j)]]
                if torch.any(mask_ij):
                    pred_com[iid_to_index[i], iid_to_index[j]] = torch.mean(
                        pred_by_iid[int(j)][:, mask_ij], dim=1
                    )
                # else:
                #    print ('no map',i,j)

        # apply all transforms to native
        nat_com = (
            torch.einsum(
                "ijkx,ijlxy->ijkly",
                nat_com[None, :, :, :] - xpres[:, None, :, :],
                Us[:, None],
            )
            + xposts[:, None, None]
        )

        # collect all distances
        #   dist[i,j,k,l] - distance assigning ...
        #      transform i of
        #      iid j of native to
        #      iid k of pred for
        #      all l models
        dist = torch.linalg.norm(pred_com[None, :, :] - nat_com, dim=-1)

        # solve mapping
        transforms, assignment = self._greedy_resolve_mapping(
            dist,
            iid_to_index,
            entity_to_index,
            iids_by_entity,
            entity_by_iids,
            nmodeled_by_iid,
        )

        # generate output stack
        x_native_aln = torch.zeros_like(x_pred)
        x_native_mask = torch.zeros(
            x_pred.shape[:2], dtype=torch.bool, device=x_pred.device
        )
        for i, si in assignment.items():
            for t in range(x_native_aln.shape[0]):
                mask_src = mol_iid == i
                x_native_aln[t, mask_src[mask]] = native_by_iid[int(si[t])][
                    mask_by_iid[int(i)]
                ]
                x_native_mask[t, mask_src[mask]] = mask_native_by_iid[int(si[t])][
                    mask_by_iid[int(i)]
                ]

        return (x_native_aln, x_native_mask)

    def forward(self, network_output, loss_input, symm_input):
        x_pred = network_output["X_L"]
        mol_entities = symm_input["molecule_entity"].to(x_pred.device)
        mol_iid = symm_input["molecule_iid"].to(x_pred.device)
        crop_mask = symm_input["crop_mask"].to(x_pred.device)
        x_native = symm_input["coord_atom_lvl"].to(x_pred.device)
        mask_native = symm_input["mask_atom_lvl"].to(x_pred.device)

        x_native_aln, x_native_mask = self._resolve_subunits(
            mol_entities, mol_iid, crop_mask, x_native, mask_native, x_pred
        )

        loss_input["X_gt_L"] = x_native_aln
        loss_input["crd_mask_L"] = x_native_mask

        return loss_input


class ProteinLigandBondLoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, network_input, network_output, loss_input):
        # find p/l bonds at token level
        is_ligand = network_input["f"]["is_ligand"]
        is_inter_polymer_ligand = torch.outer(is_ligand, ~is_ligand)
        token_bonds = network_input["f"]["token_bonds"]
        pl_bonds = token_bonds * is_inter_polymer_ligand
        first_tok, second_tok = pl_bonds.nonzero(as_tuple=True)

        # early exit
        if first_tok.numel() == 0:
            return torch.tensor(0.0), {"protein_ligand_bond_loss": torch.tensor(0.0)}

        # map tokens to atom level
        atom2token = network_input["f"]["atom_to_token_map"]
        pl_atoms = torch.zeros(
            (1, atom2token.shape[0], atom2token.shape[0]),
            dtype=torch.bool,
            device=atom2token.device,
        )
        for i, j in zip(first_tok, second_tok):
            pl_atoms += (atom2token == i)[None, :, None] * (atom2token == j)[
                None, None, :
            ]

        crd_mask_LL = (
            loss_input["crd_mask_L"][:, None] * loss_input["crd_mask_L"][:, :, None]
        )
        resolved_bonds = pl_atoms * crd_mask_LL

        # the mask may be different for each structure in the batch, so resolve bonds at the per-batch level
        b, atom1, atom2 = resolved_bonds.nonzero(as_tuple=True)

        # get loss
        X_L = network_output["X_L"]
        X_gt_L = loss_input["X_gt_L"]
        predicted_distances = torch.linalg.norm(X_L[b, atom1] - X_L[b, atom2], dim=-1)
        ground_truth_distances = torch.linalg.norm(
            X_gt_L[b, atom1] - X_gt_L[b, atom2], dim=-1
        )
        mask_bonded = ground_truth_distances < 2.4
        loss = torch.mean(
            torch.square(
                predicted_distances[mask_bonded] - ground_truth_distances[mask_bonded]
            )
        )

        return self.weight * loss, {"protein_ligand_bond_loss": loss.detach()}


class DiffusionLoss(nn.Module):
    def __init__(
        self,
        weight,
        sigma_data,
        alpha_dna,
        alpha_rna,
        alpha_ligand,
        edm_lambda,
        se3_invariant_loss,
        clamp_diffusion_loss,
    ):
        super().__init__()
        self.weight = weight
        self.sigma_data = sigma_data
        self.alpha_dna = alpha_dna
        self.alpha_rna = alpha_rna
        self.alpha_ligand = alpha_ligand
        if edm_lambda:
            # original EDM scaling factor
            self.get_lambda = (
                lambda sigma: (sigma**2 + self.sigma_data**2)
                / (sigma * self.sigma_data) ** 2
            )
        else:
            # AF3 uses a weird scaling factor for their loss
            self.get_lambda = (
                lambda sigma: (sigma**2 + self.sigma_data**2)
                / (sigma + self.sigma_data) ** 2
            )
        self.se3_invariant_loss = se3_invariant_loss
        self.clamp_diffusion_loss = clamp_diffusion_loss

    def forward(self, network_input, network_output, loss_input):
        X_L = network_output["X_L"]  # D, L, 3
        D = X_L.shape[0]
        X_gt_L = loss_input["X_gt_L"]
        crd_mask_L = loss_input["crd_mask_L"]
        tok_idx = network_input["f"]["atom_to_token_map"]
        t = network_input["t"]  # (D,)

        w_L = 1 + (
            network_input["f"]["is_dna"] * self.alpha_dna
            + network_input["f"]["is_rna"] * self.alpha_rna
            + network_input["f"]["is_ligand"] * self.alpha_ligand
        )[tok_idx].to(torch.float)
        w_L = w_L[None].expand(D, -1) * crd_mask_L

        if self.se3_invariant_loss:
            # check if this is correct
            X_gt_aligned_L = weighted_rigid_align(X_L, X_gt_L, crd_mask_L[0], w_L)
        else:
            X_gt_aligned_L = X_gt_L
        X_gt_aligned_L = torch.nan_to_num(X_gt_aligned_L)
        l_mse = (
            1
            / 3
            * torch.div(
                torch.sum(w_L * torch.sum((X_L - X_gt_aligned_L) ** 2, dim=-1), dim=-1),
                torch.sum(crd_mask_L[0]) + 1e-4,
            )
        )  # w_L is already updated by the mask

        assert l_mse.shape == (D,)
        l_diffusion = self.get_lambda(t) * l_mse
        l_diffusion = (
            torch.clamp(l_diffusion, max=2)
            if self.clamp_diffusion_loss
            else l_diffusion
        )

        l_diffusion_total = torch.mean(l_diffusion)
        # smoothed lddt loss
        smoothed_lddt_loss_ = smoothed_lddt_loss(
            X_L,
            X_gt_L,
            crd_mask_L,
            network_input["f"]["is_dna"],
            network_input["f"]["is_rna"],
            tok_idx,
            # tag=network_input["id"]
        )
        l_diffusion_total += smoothed_lddt_loss_.mean()
        loss_dict = {
            "diffusion_loss": l_diffusion.detach(),
            "smoothed_lddt_loss": smoothed_lddt_loss_.detach(),
            "t": t.detach(),
        }

        return self.weight * l_diffusion_total, loss_dict


def _smoothed_lddt_loss_naive(X_L, X_gt_L_aligned, crd_mask_L, is_dna, is_rna, tok_idx):
    """
    computes lddt with a sigmoid within each bucket to smooth the loss
    X_L: (D, L, 3)
    X_gt_L_aligned: (D, L, 3)
    crd_mask_L: (D, L)
    is_dna: (L,)
    is_rna: (L,)
    tok_idx: (L,)

    returns: (D,)
    """
    predicted_distances = torch.cdist(X_L, X_L)
    ground_truth_distances = torch.cdist(X_gt_L_aligned, X_gt_L_aligned)
    ground_truth_distances[ground_truth_distances.isnan()] = 9999.0
    difference_distances = torch.abs(ground_truth_distances - predicted_distances)
    lddt_matrix = torch.zeros_like(difference_distances)
    lddt_matrix = (
        0.25 * torch.sigmoid(4.0 - difference_distances)
        + 0.25 * torch.sigmoid(2.0 - difference_distances)
        + 0.25 * torch.sigmoid(1.0 - difference_distances)
        + 0.25 * torch.sigmoid(0.5 - difference_distances)
    )
    # remove unresolved atoms, atoms within same residue
    in_same_residue_LL = tok_idx[:, None] == tok_idx[None, :]
    is_na_L = is_dna[tok_idx] | is_rna[tok_idx]
    is_close_distance = (ground_truth_distances < 30) * is_na_L + (
        ground_truth_distances < 15
    ) * ~is_na_L
    mask = crd_mask_L[0] & ~in_same_residue_LL & is_close_distance[0]
    lddt = (lddt_matrix * mask[None]).sum(dim=(-1, -2)) / (
        mask.sum(dim=(-1, -2)) + 1e-6
    )
    return 1 - lddt


def smoothed_lddt_loss(X_L, X_gt_L, crd_mask_L, is_dna, is_rna, tok_idx, eps=1e-6):
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

        return 1 - lddt

    return _dolddt(X_L, X_gt_L, crd_mask_L, is_dna, is_rna, tok_idx, eps)


def distogram_loss(
    pred_distogram,
    X_rep_atoms_I,
    crd_mask_rep_atoms_I,
    cce_loss,
    min_distance=2,
    max_distance=22,
    bins=64,
):
    """
    computes distogram loss
    """
    distance_map = torch.cdist(X_rep_atoms_I, X_rep_atoms_I)
    distance_map[distance_map.isnan()] = 9999.0
    bins = torch.linspace(min_distance, max_distance, bins).to(X_rep_atoms_I.device)
    # Note that torch.bucketize adds a catch-all bin for values outside the range,
    # so we end up with n_bins + 1 bins (65 in the case of AF-3)
    binned_distances = torch.bucketize(distance_map, bins)
    crd_mask_rep_atom_II = crd_mask_rep_atoms_I.unsqueeze(
        -1
    ) * crd_mask_rep_atoms_I.unsqueeze(-2)
    distogram_cce = cce_loss(
        pred_distogram.permute(-1, -2, -3)[None], binned_distances[None]
    )
    return distogram_cce[..., crd_mask_rep_atom_II].sum() / (
        crd_mask_rep_atom_II.sum() + 1e-4
    )


class DistogramLoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
        self.cce_loss = nn.CrossEntropyLoss(reduction="none")
        self.eps = 1e-4

    def forward(self, network_input, network_output, loss_input):
        pred_distogram = network_output["distogram"]
        X_rep_atoms_I = loss_input["coord_token_lvl"]
        crd_mask_rep_atoms_I = loss_input["mask_token_lvl"]
        loss = distogram_loss(
            pred_distogram, X_rep_atoms_I, crd_mask_rep_atoms_I, self.cce_loss
        )
        return self.weight * loss, {"distogram_loss": loss.detach()}


class NullLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, network_input, network_output, loss_input):
        loss = 0
        for key, val in network_output.items():
            val[val.isnan()] = 0
            loss += torch.sum(val) * 0

        return loss, {}
