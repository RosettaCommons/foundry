import torch
import pickle
import gzip
from typing import Dict, Optional, Any

import numpy as np
from rf2aa.data.chain_crop import (
    get_crop,
    crop_sm_compl_asmb_contig,
    crop_sm_compl_assembly,
    crop_chirals,
)
from rf2aa.util import (
    get_protein_bond_feats,
    center_and_realign_missing,
    idx_from_Ls,
    same_chain_2d_from_Ls,
    reindex_protein_feats_after_atomize,
    reassign_symmetry_after_cropping,
)
from rf2aa.data.data_loader import (
    blank_template,
    merge_a3m_hetero,
    generate_xyz_prev,
    get_bond_distances,
    get_term_feats,
    MSAFeaturize,
)
from rf2aa.data.loaders.protein_partners import load_protein_partners
from rf2aa.data.loaders.small_molecule_partners import (
    load_small_molecule_partners,
    prune_lig_partners,
)


def get_cif_metadata(item: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    pdb_chain = item["CHAINID"]
    pdb_id = pdb_chain.split("_")[0]
    out = pickle.load(gzip.open(params["MOL_DIR"] + f"/{pdb_id[1:3]}/{pdb_id}.pkl.gz"))
    if len(out) == 4:
        chains, asmb, covale, modres = out
    elif len(out) == 5:
        chains, asmb, covale, _, modres = out
    else:
        raise ValueError(f"cif parser returns {len(out)} values")

    i_a = str(item["ASSEMBLY"])
    asmb_xfs = asmb[i_a]

    cif_outs = {
        "chains": chains,
        "asmb": asmb,
        "covale": covale,
        "modres": modres,
        "asmb_xfs": asmb_xfs,
        "pdb_id": pdb_id,
        "pdb_chain": pdb_chain,
    }
    return cif_outs


def get_partner_lists(
    item,
    params,
    num_protein_chains: Optional[int] = None,
    num_ligand_chains: Optional[int] = None,
):
    # list of proteins and ligands to featurize
    prot_partners = [p for p in item["PARTNERS"] if p[-1] == "polypeptide(L)"]
    prot_partners = prot_partners[: params["MAXPROTCHAINS"]]
    if num_protein_chains is not None:
        prot_partners = prot_partners[
            : min(num_protein_chains, params["MAXPROTCHAINS"])
        ]

    lig_partners = lig_partners = [p for p in item["PARTNERS"] if p[-1] == "nonpoly"]
    lig_partners = prune_lig_partners(lig_partners, params)
    lig_partners = [(item["LIGAND"], item["LIGXF"], -1, -1, "nonpoly")] + lig_partners

    lig_partners = lig_partners[: params["MAXLIGCHAINS"]]
    if num_ligand_chains is not None:
        lig_partners = lig_partners[: min(num_ligand_chains, params["MAXLIGCHAINS"])]
    return prot_partners, lig_partners


def pad_merge_protein_small_molecule_tensors(
    protein_x, small_molecule_x, fill_value: Any = np.nan
):
    n_symm_prot = protein_x.shape[0]
    n_symm_sm = small_molecule_x.shape[0]
    l_total = protein_x.shape[1] + small_molecule_x.shape[1]
    x_total = torch.full(
        (max(n_symm_prot, n_symm_sm), l_total, *protein_x.shape[2:]), fill_value
    )
    x_total[:n_symm_prot, : protein_x.shape[1]] = protein_x
    if small_molecule_x.shape[0] > 0:
        x_total[:n_symm_sm, protein_x.shape[1] :, 1] = small_molecule_x

    return x_total


def merge_outs(protein_outs, small_molecule_outs, params, random_noise: float = 5.0):
    # Combine protein and ligand true coordinates
    xyz = pad_merge_protein_small_molecule_tensors(
        protein_outs["xyz_prot"], small_molecule_outs["xyz_sm"], np.nan
    )
    mask = pad_merge_protein_small_molecule_tensors(
        protein_outs["mask_prot"], small_molecule_outs["mask_sm"], False
    )

    # combine protein & ligand templates
    N_tmpl = protein_outs["xyz_t_prot"].shape[0]
    xyz_t_sm, f1d_t_sm, mask_t_sm, _ = blank_template(
        N_tmpl, sum(small_molecule_outs["Ls_sm"]), random_noise
    )
    xyz_t = torch.cat([protein_outs["xyz_t_prot"], xyz_t_sm], dim=1)
    f1d_t = torch.cat([protein_outs["f1d_t_prot"], f1d_t_sm], dim=1)
    mask_t = torch.cat([protein_outs["mask_t_prot"], mask_t_sm], dim=1)

    # bond features
    bond_feats_prot = [get_protein_bond_feats(L) for L in protein_outs["Ls_prot"]]
    bond_feats_list = bond_feats_prot + small_molecule_outs["bond_feats_sm"]
    bond_feats = torch.block_diag(
        *bond_feats_list
    ).long()

    # other features
    idx = idx_from_Ls(protein_outs["Ls_prot"] + small_molecule_outs["Ls_sm"])
    same_chain = same_chain_2d_from_Ls(
        protein_outs["Ls_prot"] + small_molecule_outs["Ls_sm"]
    )
    ch_label = torch.cat(
        [
            protein_outs["ch_label_prot"],
            small_molecule_outs["ch_label_sm"]
            + protein_outs["ch_label_prot"].max()
            + 1,
        ]
    )

    # load msa
    a3m_sm = {
        "msa": small_molecule_outs["msa_sm"],
        "ins": torch.zeros_like(small_molecule_outs["msa_sm"]),
    }
    a3m = merge_a3m_hetero(
        protein_outs["a3m_prot"],
        a3m_sm,
        [sum(protein_outs["Ls_prot"]), sum(small_molecule_outs["Ls_sm"])],
    )
    msa = a3m["msa"].long()
    ins = a3m["ins"].long()
    assert msa.shape[1] == xyz.shape[1], "msa shape and xyz shape don't match"

    merged_outs = {
        "xyz": xyz,
        "mask": mask,
        "xyz_t": xyz_t,
        "f1d_t": f1d_t,
        "mask_t": mask_t,
        "bond_feats": bond_feats,
        "idx": idx,
        "same_chain": same_chain,
        "ch_label": ch_label,
        "msa": msa,
        "ins": ins,
    }
    return merged_outs


def loader_sm_compl_assembly(
    item,
    params,
    chid2hash={},
    chid2taxid={},
    chid2smpartners=None,
    task="sm_compl_asmb",
    num_protein_chains=None,
    num_ligand_chains=None,
    pick_top=True,
    random_noise=5.0,
    fixbb=False,
    remove_residue=True,
):
    """Load protein/ligand assembly from pre-parsed CIF files. Outputs can
    represent multiple chains, which are ordered from most to least contacts
    with query ligand.  Protein chains all come before ligand chains, and
    protein chains with identical sequences are grouped contiguously.

    `all_partners` is a list of 5-tuples representing ligands and protein
    chains near the query ligand that should be featurized as part of the
    assembly. The 5-tuple has the form

        (partner, xforms, num_contacts, min_dist, partner_type)

    If `partner_type` is "polypeptide", then `partner` is the chain letter and
    `xforms` is an integer index of a coordinate transform in `asmb_xfs`. If
    `partner_type` is "nonpoly", then `partner` is a list of tuples
    `(chain_letter, res_num, res_name)` representing a ligand and `xforms` is a
    list of tuples `(chain_letter, xform_index)` representing transforms.
    `num_contacts` is the number of heavy atoms within 5A of the query ligand.
    `min_dist` is the minimum distance in angstroms between a heavy atom and
    the ligand.
    """
    cif_outs = get_cif_metadata(item, params)
    pdb_id = cif_outs["pdb_id"]

    prot_partners, lig_partners = get_partner_lists(
        item, params, num_protein_chains, num_ligand_chains
    )

    protein_outs = load_protein_partners(
        prot_partners,
        params,
        pdb_id,
        cif_outs,
        chid2hash,
        chid2taxid,
        pick_top=pick_top,
        random_noise=random_noise,
    )

    small_molecule_outs = load_small_molecule_partners(
        lig_partners,
        prot_partners,
        cif_outs,
        params,
        mod_residues_to_atomize=protein_outs["mod_residues_to_atomize"],
    )
    Ls_prot = protein_outs["Ls_prot"]
    Ls_sm = small_molecule_outs["Ls_sm"]

    merged_outs = merge_outs(
        protein_outs, small_molecule_outs, params, random_noise=random_noise
    )

    xyz = merged_outs["xyz"]
    mask = merged_outs["mask"]
    bond_feats = merged_outs["bond_feats"]
    idx = merged_outs["idx"]
    xyz_t = merged_outs["xyz_t"]
    f1d_t = merged_outs["f1d_t"]
    mask_t = merged_outs["mask_t"]
    same_chain = merged_outs["same_chain"]
    ch_label = merged_outs["ch_label"]
    msa = merged_outs["msa"]
    ins = merged_outs["ins"]

    if small_molecule_outs["residues_to_atomize"]:
        (
            msa,
            ins,
            xyz,
            mask,
            bond_feats,
            idx,
            xyz_t,
            f1d_t,
            mask_t,
            same_chain,
            ch_label,
            Ls_prot,
            Ls_sm,
        ) = reindex_protein_feats_after_atomize(
            small_molecule_outs["residues_to_atomize"],
            prot_partners,
            msa,
            ins,
            xyz,
            mask,
            bond_feats,
            idx,
            xyz_t,
            f1d_t,
            mask_t,
            same_chain,
            ch_label,
            Ls_prot,
            Ls_sm,
            small_molecule_outs["akeys_sm"],
            remove_residue=remove_residue,
        )

    ntempl = xyz_t.shape[0]
    xyz_t = torch.stack(
        [
            center_and_realign_missing(xyz_t[i], mask_t[i], same_chain=same_chain)
            for i in range(ntempl)
        ]
    )

    xyz_prev, mask_prev = generate_xyz_prev(xyz_t, mask_t, params)

    xyz_prev = torch.nan_to_num(xyz_prev)
    xyz = torch.nan_to_num(xyz)
    xyz_t = torch.nan_to_num(xyz_t)

    # keep track of protein positions for reindexing chirals after crop
    L_total = sum(protein_outs["Ls_prot"]) + sum(small_molecule_outs["Ls_sm"])
    is_prot = torch.zeros(L_total)
    is_prot[: sum(protein_outs["Ls_prot"])] = 1

    # N/C-terminus features for MSA features (need to generate before cropping)
    term_info = get_term_feats(protein_outs["Ls_prot"] + small_molecule_outs["Ls_sm"])
    term_info[sum(protein_outs["Ls_prot"]) :, :] = (
        0  # ligand chains don't get termini features
    )

    # crop around query ligand (1st sm chain)
    # always need to run cropping function to remove erroneous ligand partners
    if sum(small_molecule_outs["Ls_sm"]) == 0:
        sel = get_crop(len(idx), mask[0], msa.device, params["CROP"])
    else:
        if params["RADIAL_CROP"]:
            sel = crop_sm_compl_assembly(
                xyz[0], mask[0], Ls_prot, Ls_sm, params["CROP"]
            )
        else:
            sel = crop_sm_compl_asmb_contig(
                xyz[0],
                mask[0],
                Ls_prot,
                Ls_sm,
                bond_feats,
                params["CROP"],
                use_partial_ligands=False,
            )
    mask = reassign_symmetry_after_cropping(
        sel, protein_outs["Ls_prot"], ch_label, mask, item
    )

    msa = msa[:, sel]
    ins = ins[:, sel]
    xyz = xyz[:, sel]
    mask = mask[:, sel]
    xyz_t = xyz_t[:, sel]
    f1d_t = f1d_t[:, sel]
    mask_t = mask_t[:, sel]
    xyz_prev = xyz_prev[sel]
    mask_prev = mask_prev[sel]
    idx = idx[sel]
    same_chain = same_chain[sel][:, sel]
    bond_feats = bond_feats[sel][:, sel]
    ch_label = ch_label[sel]
    is_prot = is_prot[sel]
    term_info = term_info[sel]

    # crop small molecule features, assumes all sm chains are after all protein chains
    atom_sel = sel[sel >= sum(protein_outs["Ls_prot"])] - sum(
        protein_outs["Ls_prot"]
    )  # 0 index all the selected atoms
    frames = small_molecule_outs["frames"]
    chirals = small_molecule_outs["chirals"]
    
    frames = frames[atom_sel]
    chirals = crop_chirals(chirals, atom_sel)

    # reindex chiral atom positions - assumes all sm chains are after all protein chains
    if chirals.shape[0] > 0:
        L1 = is_prot.sum()
        chirals[:, :-1] = chirals[:, :-1] + L1

    dist_matrix = get_bond_distances(bond_feats)

    # create MSA features from cropped msa and insertions
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(
        msa.long(),
        ins.long(),
        params,
        p_mask=params["p_msa_mask"],
        term_info=term_info,
        fixbb=fixbb,
        seed_msa_clus=protein_outs["seed_msa_clus"],
    )

    return (
        seq.long(),
        msa_seed_orig.long(),
        msa_seed.float(),
        msa_extra.float(),
        mask_msa,
        xyz.float(),
        mask,
        idx.long(),
        xyz_t.float(),
        f1d_t.float(),
        mask_t,
        xyz_prev.float(),
        mask_prev,
        same_chain,
        False,
        False,
        frames,
        bond_feats,
        dist_matrix,
        chirals,
        ch_label,
        "C1",
        task,
        item,
    )


def loader_sm_compl_assembly_single(*args, **kwargs):
    kwargs["num_protein_chains"] = 1
    return loader_sm_compl_assembly(*args, **kwargs)
