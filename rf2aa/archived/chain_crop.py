import itertools
import random

import numpy as np
import torch
from scipy.sparse.csgraph import shortest_path

from rf2aa.util import center_and_realign_missing


# slice long chains without retaining continuity of the chain
def get_discontiguous_crop(l, mask, device, crop_size, unclamp=False):
    mask = ~(mask[:, :3].sum(dim=-1) < 3.0)
    exists = mask.nonzero()[:, 0]
    if len(exists) <= crop_size:
        return exists

    n_backbone = len(exists)
    lower_bound = 0
    upper_bound = n_backbone - crop_size + 1
    start = np.random.randint(lower_bound, upper_bound)
    return exists[start : start + crop_size]


# slice long chains
def get_crop(l, mask, device, crop_size, unclamp=False):
    sel = torch.arange(l, device=device)
    if l <= crop_size:
        return sel

    size = crop_size

    mask = ~(mask[:, :3].sum(dim=-1) < 3.0)
    exists = mask.nonzero().flatten()
    if unclamp:  # bias it toward N-term.. (follow what AF did.. but don't know why)
        x = np.random.randint(len(exists)) + 1
        res_idx = exists[torch.randperm(x)[0]].item()
    else:
        res_idx = exists[torch.randperm(len(exists))[0]].item()

    lower_bound = max(0, res_idx - size + 1)
    upper_bound = min(l - size + 1, res_idx + 1)

    start = np.random.randint(lower_bound, upper_bound)
    return sel[start : start + size]


# devide crop between multiple (2+) chains
#   >20 res / chain
def rand_crops(ls, maxlen, minlen=20):
    base = [min(minlen, l) for l in ls]
    nremain = [max(0, l - minlen) for l in ls]

    # this must be inefficient...
    pool = []
    for i in range(len(ls)):
        pool.extend([i] * nremain[i])
    pool = random.sample(pool, maxlen - sum(base))
    chosen = [base[i] + sum(p == i for p in pool) for i in range(len(ls))]
    return torch.tensor(chosen)


def get_complex_crop(len_s, mask, device, params):
    tot_len = sum(len_s)
    sel = torch.arange(tot_len, device=device)

    crops = rand_crops(len_s, params["CROP"])

    offset = 0
    sel_s = list()
    for k in range(len(len_s)):
        mask_chain = ~(mask[offset : offset + len_s[k], :3].sum(dim=-1) < 3.0)
        exists = mask_chain.nonzero()[0]
        res_idx = exists[torch.randperm(len(exists))[0]].item()
        lower_bound = max(0, res_idx - crops[k] + 1)
        upper_bound = min(len_s[k] - crops[k], res_idx) + 1
        start = np.random.randint(lower_bound, upper_bound) + offset
        sel_s.append(sel[start : start + crops[k]])
        offset += len_s[k]
    return torch.cat(sel_s)


def get_spatial_crop(xyz, mask, sel, len_s, params, label, cutoff=10.0, eps=1e-4):
    device = xyz.device

    # get interface residues
    #   interface defined as chain 1 versus all other chains
    cond = torch.cdist(xyz[: len_s[0], 1], xyz[len_s[0] :, 1]) < cutoff
    cond = torch.logical_and(
        cond, mask[: len_s[0], None, 1] * mask[None, len_s[0] :, 1]
    )
    i, j = torch.where(cond)
    ifaces = torch.cat([i, j + len_s[0]])
    if len(ifaces) < 1:
        print("ERROR: no iface residue????", label)
        return get_complex_crop(len_s, mask, device, params)
    cnt_idx = ifaces[np.random.randint(len(ifaces))]

    dist = (
        torch.cdist(xyz[:, 1], xyz[cnt_idx, 1][None]).reshape(-1)
        + torch.arange(len(xyz), device=xyz.device) * eps
    )
    cond = mask[:, 1] * mask[cnt_idx, 1]
    dist[~cond] = 999999.9
    _, idx = torch.topk(dist, params["CROP"], largest=False)

    sel, _ = torch.sort(sel[idx])
    return sel


# this is a bit of a mess...
def get_na_crop(
    seq,
    xyz,
    mask,
    sel,
    len_s,
    params,
    negative=False,
    incl_protein=True,
    cutoff=12.0,
    bp_cutoff=4.0,
    eps=1e-4,
):
    device = xyz.device

    # get base pairing NA bases
    repatom = torch.zeros(sum(len_s), dtype=torch.long, device=xyz.device)
    repatom[seq == 22] = 15  # DA - N1
    repatom[seq == 23] = 14  # DC - N3
    repatom[seq == 24] = 15  # DG - N1
    repatom[seq == 25] = 14  # DT - N3
    repatom[seq == 27] = 12  # A - N1
    repatom[seq == 28] = 15  # C - N3
    repatom[seq == 29] = 12  # G - N1
    repatom[seq == 30] = 15  # U - N3

    if not incl_protein:  # either 1 or 2 NA chains
        if len(len_s) == 2:
            # 2 RNA chains
            xyz_na1_rep = torch.gather(
                xyz[: len_s[0]], 1, repatom[: len_s[0], None, None].repeat(1, 1, 3)
            ).squeeze(1)
            xyz_na2_rep = torch.gather(
                xyz[len_s[0] :], 1, repatom[len_s[0] :, None, None].repeat(1, 1, 3)
            ).squeeze(1)
            cond = torch.cdist(xyz_na1_rep, xyz_na2_rep) < bp_cutoff

            mask_na1_rep = torch.gather(
                mask[: len_s[0]], 1, repatom[: len_s[0], None]
            ).squeeze(1)
            mask_na2_rep = torch.gather(
                mask[len_s[0] :], 1, repatom[len_s[0] :, None]
            ).squeeze(1)
            cond = torch.logical_and(
                cond, mask_na1_rep[:, None] * mask_na2_rep[None, :]
            )

            if torch.sum(cond) == 0:
                i = np.random.randint(len_s[0])
                j = np.random.randint(len_s[1])
                while not mask[i, 1] or not mask[j, 1]:
                    i = np.random.randint(len_s[0])
                    j = np.random.randint(len_s[1])
                cond[i, j] = True
        else:
            # 1 RNA chains
            xyz_na_rep = torch.gather(
                xyz, 1, repatom[:, None, None].repeat(1, 1, 3)
            ).squeeze(1)
            cond = torch.cdist(xyz_na_rep, xyz_na_rep) < bp_cutoff
            mask_na_rep = torch.gather(mask, 1, repatom[:, None]).squeeze(1)
            cond = torch.logical_and(cond, mask_na_rep[:, None] * mask_na_rep[None, :])

            if torch.sum(cond) == 0:
                i = np.random.randint(len_s[0] - 1)
                while not mask[i, 1] or not mask[i + 1, 1]:
                    i = np.random.randint(len_s[0] - 1)
                cond[i, i + 1] = True

    else:  # either 1prot+1NA, 1prot+2NA or 2prot+2NA
        # find NA:NA basepairs
        if len(len_s) >= 3:
            if len(len_s) == 3:
                na1s, na2s = len_s[0], len_s[0] + len_s[1]
            else:
                na1s, na2s = len_s[0] + len_s[1], len_s[0] + len_s[1] + len_s[2]

            xyz_na1_rep = torch.gather(
                xyz[na1s:na2s], 1, repatom[na1s:na2s, None, None].repeat(1, 1, 3)
            ).squeeze(1)
            xyz_na2_rep = torch.gather(
                xyz[na2s:], 1, repatom[na2s:, None, None].repeat(1, 1, 3)
            ).squeeze(1)
            cond_bp = torch.cdist(xyz_na1_rep, xyz_na2_rep) < bp_cutoff

            mask_na1_rep = torch.gather(
                mask[na1s:na2s], 1, repatom[na1s:na2s, None]
            ).squeeze(1)
            mask_na2_rep = torch.gather(mask[na2s:], 1, repatom[na2s:, None]).squeeze(1)
            cond_bp = torch.logical_and(
                cond_bp, mask_na1_rep[:, None] * mask_na2_rep[None, :]
            )

        # find NA:prot contacts
        if not negative:
            # get interface residues
            #   interface defined as chain 1 versus all other chains
            if len(len_s) == 4:
                first_na = len_s[0] + len_s[1]
            else:
                first_na = len_s[0]

            xyz_na_rep = torch.gather(
                xyz[first_na:], 1, repatom[first_na:, None, None].repeat(1, 1, 3)
            ).squeeze(1)
            cond = torch.cdist(xyz[:first_na, 1], xyz_na_rep) < cutoff
            mask_na_rep = torch.gather(
                mask[first_na:], 1, repatom[first_na:, None]
            ).squeeze(1)
            cond = torch.logical_and(
                cond, mask[:first_na, None, 1] * mask_na_rep[None, :]
            )

        # random NA:prot contact for negatives
        if negative or torch.sum(cond) == 0:
            if len(len_s) == 4:
                nprot, nna = len_s[0] + len_s[1], sum(len_s[2:])
            else:
                nprot, nna = len_s[0], sum(len_s[1:])

            # pick a random pair of residues
            cond = torch.zeros((nprot, nna), dtype=torch.bool)
            i, j = np.random.randint(nprot), np.random.randint(nna)
            while not mask[i, 1]:
                i = np.random.randint(nprot)
            while not mask[nprot + j, 1]:
                j = np.random.randint(nna)
            cond[i, j] = True

    # a) build a graph of costs:
    #     cost (i,j in same chain) = abs(i-j)
    #     cost (i,j in different chains) = { 0 if i,j are an interface
    #                                    = { 999 if i,j are NOT an interface
    if len(len_s) >= 3:
        if len(len_s) == 4:
            nprot, nna1, nna2 = len_s[0] + len_s[1], len_s[2], len_s[3]
            diag_1 = np.full((nprot, nprot), 999)
            diag_1[: len_s[0], : len_s[0]] = np.abs(
                np.arange(len_s[0])[:, None] - np.arange(len_s[0])[None, :]
            )
            diag_1[len_s[0] :, len_s[0] :] = np.abs(
                np.arange(len_s[1])[:, None] - np.arange(len_s[1])[None, :]
            )
        else:
            nprot, nna1, nna2 = len_s[0], len_s[1], len_s[2]
            diag_1 = np.abs(np.arange(nprot)[:, None] - np.arange(nprot)[None, :])

        diag_2 = np.abs(np.arange(len_s[-2])[:, None] - np.arange(len_s[-2])[None, :])
        diag_3 = np.abs(np.arange(len_s[-1])[:, None] - np.arange(len_s[-1])[None, :])
        int_1_2 = np.full((nprot, nna1), 999)
        int_1_3 = np.full((nprot, nna2), 999)
        int_2_3 = np.full((nna1, nna2), 999)
        int_1_2[cond[:, :nna1]] = 1
        int_1_3[cond[:, nna1:]] = 1
        int_2_3[cond_bp] = 0

        inter = np.block(
            [
                [diag_1, int_1_2, int_1_3],
                [int_1_2.T, diag_2, int_2_3],
                [int_1_3.T, int_2_3.T, diag_3],
            ]
        )
    elif len(len_s) == 2:
        int_1_2 = np.full((len_s[0], len_s[1]), 999)
        int_1_2[cond] = 1
        inter = np.block(
            [
                [
                    np.abs(np.arange(len_s[0])[:, None] - np.arange(len_s[0])[None, :]),
                    int_1_2,
                ],
                [
                    int_1_2.T,
                    np.abs(np.arange(len_s[1])[:, None] - np.arange(len_s[1])[None, :]),
                ],
            ]
        )
    else:
        inter = np.abs(np.arange(len_s[0])[:, None] - np.arange(len_s[0])[None, :])
        inter[cond] = 1

    # b) pick a random interface residue
    intface, _ = torch.where(cond)
    startres = intface[np.random.randint(len(intface))]

    # c) traverse graph starting from chosen residue
    d_res = shortest_path(inter, directed=False, indices=startres)
    _, idx = torch.topk(
        torch.from_numpy(d_res).to(device=device), params["CROP"], largest=False
    )

    sel, _ = torch.sort(sel[idx])

    return sel


def crop_sm_compl(
    prot_xyz,
    lig_xyz,
    Ls,
    crop_size,
    mask_prot,
    seq_prot,
    select_farthest_residues: bool = False,
    min_resolved_residues: int = 32,
):
    """
    choose residues with calphas close to a random ligand atom

    select_farthest_residues : bool
        If True, select the farthest residues from the ligand rather than the
        closest.
    min_resolved_residues : int
        Minimum number of residues to keep in the cropped structure before
        considering unresolved residue positions.
    """
    # ligand_com = torch.nanmean(lig_xyz, dim=[0,1]).expand(1,3)
    i_face_xyz = lig_xyz[np.random.randint(len(lig_xyz))]

    # set missing residue coordinates to their nearest neighbor in the same chain,
    # to avoid artifactual distances from missing coordinates being at the origin
    realigned_prot_xyz = center_and_realign_missing(
        prot_xyz[0], mask_prot[0], seq_prot, should_center=False
    )

    is_resolved_mask = mask_prot[0].any(axis=-1)
    dist = torch.cdist(
        realigned_prot_xyz[:, 1].unsqueeze(0), i_face_xyz.unsqueeze(0)
    ).flatten()

    if select_farthest_residues:
        # select the farthest valid residue from the ligand as the center of spherical crop
        dist_nans_as_smallest = torch.nan_to_num(dist, nan=-torch.inf)
        max_defined_residue_index = torch.argmax(dist_nans_as_smallest)
        position_of_furthest_defined_residue = realigned_prot_xyz[
            max_defined_residue_index, 1
        ]
        dist = torch.cdist(
            realigned_prot_xyz[:, 1].unsqueeze(0),
            position_of_furthest_defined_residue.unsqueeze(0),
        ).flatten()

    # Note: we want to never select NaN values so we set them to be the "farthest"
    # away residues from the ligand.
    nan_fill_value = torch.inf
    dist = torch.nan_to_num(dist, nan=nan_fill_value)

    # These next two blocks merit some explanation. I want to select the
    # values of dist that are smallest, BUT under the constraint that I select
    # at least min_resolved_residues where is_resolved_mask is True.
    # So, I first select min_resolved_residues of the smallest indices
    # of dist where is_resolved_mask is True. I then set those indices to have
    # infinitely large values so we don't select them again. I then select
    # the remaining closest residues from the resultant distance array. This
    # should solve the problem of picking only unresolved residues in a reasonable way.
    min_resolved_residues = min(torch.sum(is_resolved_mask), min_resolved_residues)
    indices_orig = torch.arange(len(dist))
    indices_orig_of_resolved = indices_orig[is_resolved_mask]
    dist_of_resolved = dist[is_resolved_mask]
    _, top_k_of_resolved = torch.topk(dist_of_resolved, min_resolved_residues)
    sel_min_resolved = indices_orig_of_resolved[top_k_of_resolved]

    remaining_residues_to_select = crop_size - len(lig_xyz) - min_resolved_residues
    assert remaining_residues_to_select > 0, (
        f"For some reason I encountered a scenario in which we are cropping a protein but I was unable to select enough residues from that protein. This probably means you passed a protein that was too short into this function. {crop_size}, {len(lig_xyz)}, {min_resolved_residues}"
    )
    dist[sel_min_resolved] = nan_fill_value
    remaining_residues_to_select = min(remaining_residues_to_select, len(dist))
    _, sel_remaining = torch.topk(dist, remaining_residues_to_select)
    sel_unsorted = torch.concat((sel_min_resolved, sel_remaining))
    sel, _ = torch.sort(sel_unsorted)

    # select the whole ligand
    lig_sel = torch.arange(lig_xyz.shape[0]) + Ls[0]

    return torch.cat((sel, lig_sel))


# fd: change use_partial_ligands to False to prevent oversized crops
def crop_sm_compl_assembly(
    all_xyz, all_mask, Ls_prot, Ls_sm, n_crop, use_partial_ligands=False
):
    """Choose residues with the `n_crop` closest C-alphas to a random atom on
    query ligand. Operates on multi-chain assemblies. Nearby ligands are
    included if all of their (unmasked) atoms are in the crop. Otherwise none
    of the atoms of that ligand are included in crop. Nearby protein chains are
    excluded if they are too short and have too few contacts to ligands or
    other protein chains.

    Parameters
    ----------
    all_xyz : torch.Tensor (L_total, N_atoms, 3)
        Coordinates of full assembly with all protein chains, followed by all ligand chains.
        1st ligand chain is assumed to be query ligand.
    all_mask : torch.Tensor (L_total, N_atoms)
        Boolean mask for whether each atom in `all_xyz` is valid
    res_mask : torch.Tensor (L_total,) bool
        Boolean mask for which residues/ligand atoms exist.
    Ls_prot : list (N_prot_chains,)
        Lengths of protein chains
    Ls_sm : list (N_lig_chains,)
        Lengths of ligand chains
    n_crop : int
        Number of nearest residues or ligand atoms to include in crop
    use_partial_ligands : bool
        Whether to keep ligands in crop if they have some atoms masked. (Default: False)

    Returns
    -------
    sel : torch.Tensor (N_residues, )
        Indices of positions inside crop. Will always include entire query ligand and whole
        ligands that are inside crop. Ligands partially inside crop will be removed, so
        length of `sel` may be less than `n_crop`
    """
    # choose random non-masked atom in query ligand
    L_prot = sum(Ls_prot)
    qlig_idx = torch.where(all_mask[L_prot : L_prot + Ls_sm[0], 1])[0] + L_prot
    ca_xyz = all_xyz[:, 1]
    query_atom = ca_xyz[np.random.choice(qlig_idx)]

    # closest `n_crop` residues to query atom
    dist = torch.cdist(
        ca_xyz.unsqueeze(0),
        query_atom.unsqueeze(0),
        compute_mode="donot_use_mm_for_euclid_dist",
    ).flatten()
    dist = torch.nan_to_num(dist, nan=999999)
    res_mask = torch.cat([all_mask[:L_prot, :3].all(dim=-1), all_mask[L_prot:, 1]])

    idx = torch.argsort(dist)
    idx = idx[
        torch.isin(idx, torch.where(res_mask)[0])
    ]  # exclude invalid residues from crop
    idx = idx[:n_crop]

    # always include every query ligand atom, regardless of if they're in topk
    query_lig_idx = np.arange(Ls_sm[0]) + L_prot
    sel = np.unique(np.concatenate([idx.numpy(), query_lig_idx]))
    # partially masked or partially cropped ligands
    offset = L_prot
    for L_sm in Ls_sm:
        curr_lig_idx = np.arange(L_sm) + offset
        curr_lig_idx_valid = np.where(res_mask[curr_lig_idx])[0] + offset

        # ligand has masked atoms and we don't want this
        if (not use_partial_ligands) and (len(curr_lig_idx) != len(curr_lig_idx_valid)):
            sel = np.setdiff1d(sel, curr_lig_idx)
            # continue
        else:
            if np.isin(curr_lig_idx_valid, sel).all():
                # all non-masked atoms are in crop; add back masked atoms to avoid messing up frames
                sel = np.unique(np.concatenate([sel, curr_lig_idx]))
            else:
                # some non-masked ligand atoms missing, remove entire ligand from crop
                sel = np.setdiff1d(sel, curr_lig_idx)

        offset += L_sm

    # remove protein chains that are short and don't contact other proteins or ligands
    # distance between protein C-alphas
    prot_sel = sel[sel < L_prot]
    lig_sel = sel[sel >= L_prot]

    # dist_prot_ca = torch.cdist(ca_xyz[prot_sel], ca_xyz[prot_sel]) # (L_prot, L_prot)
    # distance between closest heavy atom on each residue and ligand atoms
    dist_all = torch.cdist(all_xyz[sel], ca_xyz[sel])
    dist_all[~all_mask[sel]] = 99999
    dist_all, _ = dist_all.min(dim=1)  # (L_sel_prot, L_sel_sm)

    offset = 0
    for L in Ls_prot:
        # protein-ligand contacts (heavy atom within 4A)
        prot_chain_sel = np.logical_and(prot_sel >= offset, prot_sel < offset + L)

        # assumes proteins come before ligands
        is_prot_chain_sel = torch.zeros(dist_all.shape[0], dtype=bool)
        is_prot_chain_sel[: prot_chain_sel.shape[0]] = torch.tensor(prot_chain_sel)

        dist_ = dist_all[is_prot_chain_sel][:, ~is_prot_chain_sel]
        num_contacts = (dist_ < 4).sum()
        # number of residues in crop
        curr_chain_idx = np.where(res_mask)[0]
        curr_chain_idx = curr_chain_idx[
            (curr_chain_idx >= offset) & (curr_chain_idx < offset + L)
        ]
        num_residues = np.isin(curr_chain_idx, sel).sum()

        # if (num_residues < 8) or (num_prot_contacts < 10) or (num_lig_contacts < 10):
        if (num_residues < 8) or (num_contacts < 10):
            sel = np.setdiff1d(sel, curr_chain_idx)
            # print(f'removed chain from crop: (num_residues={num_residues} '\
            #      f'num_lig_contacts={num_lig_contacts})')

        offset += L

    # this is probably excessive but check again to make sure all the small molecules have contacts to proteins
    # and remove those that do not
    # need to hold on to the old prot_sel for the failure case
    prot_sel_new = sel[sel < L_prot]
    lig_sel = sel[sel >= L_prot]

    # distance between closest heavy atom on each residue and ligand atoms
    dist_prot_lig = torch.cdist(all_xyz[prot_sel_new], ca_xyz[lig_sel])
    dist_prot_lig[~all_mask[prot_sel_new]] = 99999
    dist_prot_lig, _ = dist_prot_lig.min(dim=1)  # (L_sel_prot, L_sel_sm)

    offset = L_prot
    for L_sm in Ls_sm:
        lig_chain_sel = np.logical_and(lig_sel >= offset, lig_sel < offset + L_sm)
        if np.all(lig_chain_sel == False):
            offset += L_sm
            continue
        dist_ = dist_prot_lig[:][:, lig_chain_sel]
        # dist_ = dist_[res_mask_prot][:,res_mask_lig[lig_chain_sel]]
        num_lig_contacts = (dist_ < 4).sum()

        if num_lig_contacts < 4:
            curr_chain_idx = np.arange(L_sm) + offset
            sel = np.setdiff1d(sel, curr_chain_idx)
        offset += L_sm

    if len(sel) == 0:  # we accidentally removed all the chains..
        if (
            len(prot_sel) == 0
        ):  # if all the neighbors are not protein (seen in chlorophylls in photosystems for example)
            sel = (
                get_crop(Ls_prot[0], all_mask[: Ls_prot[0]], all_xyz.device, n_crop)
                .cpu()
                .numpy()
            )
        else:
            sel = prot_sel
    return torch.from_numpy(sel).long()


def crop_sm_compl_asmb_contig(
    all_xyz, all_mask, Ls_prot, Ls_sm, bond_feats, n_crop, use_partial_ligands=False
):
    """
    instead of conducting a radial crop around a random atom, construct a crop with contiguous protein segments
    the way this works is that a graph data structure is constructed where contiguous residues are connected,
    close interchain contacts are connected and residues within a ligand are fully connected. each edge is weighted
    and then the crop is chosen by selecting a random residue and traversing the graph to find the n_crop closest nodes
    """

    def find_edges_based_on_distance(
        all_xyz,
        all_mask,
        chain_i_start_index,
        chain_i_end_index,
        chain_j_start_index,
        chain_j_end_index,
        dist_cutoff,
    ):
        xyz_chain_i = all_xyz[chain_i_start_index:chain_i_end_index]
        xyz_chain_j = all_xyz[chain_j_start_index:chain_j_end_index]

        dist = torch.cdist(xyz_chain_i[:, 1], xyz_chain_j[:, 1])  # calpha distogram
        chain_i_ca_mask = all_mask[chain_i_start_index:chain_i_end_index, 1]
        chain_j_ca_mask = all_mask[chain_j_start_index:chain_j_end_index, 1]

        mask_2d = chain_i_ca_mask[:, None] * chain_j_ca_mask[None, :]
        dist[~mask_2d] = 99999
        new_edges = (dist < dist_cutoff).nonzero()
        return new_edges

    L = all_xyz.shape[0]
    num_prot_chains = len(Ls_prot)
    num_sm_chains = len(Ls_sm)
    # construct weighted graph
    graph = np.full((L, L), n_crop, dtype=np.float32)

    # set neighboring residues to have edge weight = 1
    for chain_index, L_prot in enumerate(Ls_prot):
        chain_start_index = sum(Ls_prot[:chain_index])
        residues = torch.arange(L_prot - 1) + chain_start_index
        graph[residues, residues + 1] = 1
        graph[residues + 1, residues] = 1

    # set all intra ligand chain values to 0 so that if one atom is sampled the whole ligand is sampled (we will still confirm this later)
    total_protein_L = sum(Ls_prot)
    for chain_index in range(len(Ls_sm)):
        chain_start_index = sum(Ls_sm[:chain_index]) + total_protein_L
        chain_end_index = sum(Ls_sm[: chain_index + 1]) + total_protein_L
        graph[chain_start_index:chain_end_index, chain_start_index:chain_end_index] = (
            0.1
        )

    # set interchain edges between protein chains
    for chain_i, chain_j in itertools.combinations(range(num_prot_chains), 2):
        chain_i_start_index = sum(Ls_prot[:chain_i])
        chain_i_end_index = sum(Ls_prot[: chain_i + 1])

        chain_j_start_index = sum(Ls_prot[:chain_j])
        chain_j_end_index = sum(Ls_prot[: chain_j + 1])

        new_edges = find_edges_based_on_distance(
            all_xyz,
            all_mask,
            chain_i_start_index,
            chain_i_end_index,
            chain_j_start_index,
            chain_j_end_index,
            dist_cutoff=8,
        )
        for edge in new_edges:
            start = edge[0] + chain_i_start_index
            end = edge[1] + chain_j_start_index
            graph[start, end] = 8
            graph[end, start] = 8

    # set interchain edges between proteins and small molecules (non_covalent)
    for protein_chain, sm_chain in itertools.product(
        range(num_prot_chains), range(num_sm_chains)
    ):
        protein_chain_start_index = sum(Ls_prot[:protein_chain])
        protein_chain_end_index = sum(Ls_prot[: protein_chain + 1])

        sm_chain_start_index = sum(Ls_sm[:sm_chain]) + total_protein_L
        sm_chain_end_index = sum(Ls_sm[: sm_chain + 1]) + total_protein_L
        if torch.any(
            bond_feats[protein_chain_start_index:protein_chain_end_index][
                :, sm_chain_start_index:sm_chain_end_index
            ]
            == 8
        ):  # skip chains that are covalently connected
            continue
        new_edges = find_edges_based_on_distance(
            all_xyz,
            all_mask,
            protein_chain_start_index,
            protein_chain_end_index,
            sm_chain_start_index,
            sm_chain_end_index,
            dist_cutoff=5,
        )
        for edge in new_edges:
            start = edge[0] + protein_chain_start_index
            end = edge[1] + sm_chain_start_index
            graph[start, end] = 2
            graph[end, start] = 2

    # edges to covalent modifications should be similar to residue edges not ligand edges
    covalent_bonds = (bond_feats == 6).nonzero()
    for bond in covalent_bonds:
        graph[bond[0], bond[1]] = 1
        graph[bond[1], bond[0]] = 1

    # find an interface residue to start at by finding random residue near a ligand atom
    starting_edges = find_edges_based_on_distance(
        all_xyz,
        all_mask,
        0,
        Ls_prot[0],
        total_protein_L,
        total_protein_L + Ls_sm[0],
        dist_cutoff=10,
    )
    if starting_edges.numel() == 0:
        startres = random.randint(0, Ls_prot[0])
    else:
        startres = random.choice(starting_edges[:, 0].unique()).item()

    d_res = shortest_path(graph, directed=False, indices=startres)
    n_crop = min(d_res.shape[0], n_crop)

    _, idx = torch.topk(
        torch.from_numpy(d_res).to(device=all_xyz.device), n_crop, largest=False
    )
    sel, _ = torch.sort(idx)

    # make sure that all ligands were fully pulled into the crop
    # print(f"total number of chain: {len(Ls_sm)}")
    for sm_chain_index, L_sm in enumerate(Ls_sm):
        sm_chain_start_index = sum(Ls_sm[:sm_chain_index]) + total_protein_L
        chain_indices = torch.arange(L_sm) + sm_chain_start_index
        chain_in_crop = torch.isin(
            chain_indices, sel
        )  # tensor with length chain_indices indicating which elements from chain_indices are in sel
        is_subset = torch.all(chain_in_crop)
        has_overlap = torch.any(chain_in_crop)
        if has_overlap == True:
            if is_subset == False:
                # if sm_chain_index == 0:
                #    print("WARNING: PART OF QUERY LIGAND WAS CROPPED; ADDING REST BACK IN")
                #    sel = torch.cat((sel, chain_indices), dim=0)
                #    sel = sel.unique()
                #    continue
                if use_partial_ligands == False:
                    crop_in_chain = torch.isin(
                        sel, chain_indices
                    )  # tensor with length of sel indicating which indices in sel are also in chain_indices
                    sel = sel[~crop_in_chain]
                else:
                    sel = torch.cat((sel, chain_indices), dim=0)
                    sel = sel.unique()
    return sel


def crop_chirals(chirals, atom_sel):
    """
    this function returns only chiral centers that appear in molecules that are chosen after cropping
    chirals (nchirals, 5) first four indices in the second dimension are indices and the fifth is the angle that chiral center forms
    atom_sel: 1D tensor of small molecule atoms chosen to include in the crop
    """
    if chirals.numel() == 0:  # no chirals in this selection
        return chirals

    # clone so that we don't modify the original tensor
    chirals = chirals.clone()
    chiral_indices = chirals[:, :4].long()
    keep_mask = torch.isin(chiral_indices, atom_sel).all(dim=1)

    num_indices_less_than_chiral_index = (
        atom_sel[:, None, None] < chiral_indices[None]
    ).sum(dim=0)
    chirals[:, :4] = num_indices_less_than_chiral_index.float()
    chirals = chirals[keep_mask]
    return chirals
