import gzip
import pickle
import warnings
from typing import Any, Dict, Optional

import numpy as np
import torch

from rf2aa.data.chain_crop import (
    crop_chirals,
    get_crop,
)
from rf2aa.data.data_loader import (
    MSAFeaturize,
    blank_template,
    generate_xyz_prev,
    get_bond_distances,
    get_term_feats,
    merge_a3m_hetero,
)
from rf2aa.data.loaders.crop import sm_compl_crop_factory, universal_crop_factory
from rf2aa.data.loaders.polymer_partners import load_polymer_partners
from rf2aa.data.loaders.small_molecule_partners import (
    load_small_molecule_partners,
    prune_lig_partners,
)
from rf2aa.util import (
    center_and_realign_missing,
    get_protein_bond_feats,
    idx_from_Ls,
    reassign_symmetry_after_cropping,
    reindex_protein_feats_after_atomize,
    same_chain_2d_from_Ls,
)


def get_cif_metadata(
    pdb_id: str, assembly: str, params: Dict[str, Any]
) -> Dict[str, Any]:
    out = pickle.load(gzip.open(params["MOL_DIR"] + f"/{pdb_id[1:3]}/{pdb_id}.pkl.gz"))
    if len(out) == 4:
        chains, asmb, covale, modres = out
    elif len(out) == 5:
        chains, asmb, covale, _, modres = out
    else:
        raise ValueError(f"cif parser returns {len(out)} values")

    asmb_xfs = asmb[assembly]

    cif_outs = {
        "chains": chains,
        "asmb": asmb,
        "covale": covale,
        "modres": modres,
        "asmb_xfs": asmb_xfs,
        "pdb_id": pdb_id,
        "assembly": assembly,
    }
    return cif_outs


def get_partner_lists(
    item,
    params,
    num_protein_chains: Optional[int] = None,
    num_ligand_chains: Optional[int] = None,
):
    # list of proteins and ligands to featurize
    polymer_types = ["polypeptide(L)", "polydeoxyribonucleotide", "polyribonucleotide"]
    polymer_partners = [p for p in item["PARTNERS"] if p[-1] in polymer_types]
    polymer_partners = polymer_partners[: params["MAXPROTCHAINS"]]
    if num_protein_chains is not None:
        polymer_partners = polymer_partners[
            : min(num_protein_chains, params["MAXPROTCHAINS"])
        ]

    lig_partners = lig_partners = [p for p in item["PARTNERS"] if p[-1] == "nonpoly"]
    lig_partners = prune_lig_partners(lig_partners, params)

    if "LIGAND" in item and "LIGXF" in item:
        lig_partners = [
            (item["LIGAND"], item["LIGXF"], -1, -1, "nonpoly")
        ] + lig_partners

    lig_partners = lig_partners[: params["MAXLIGCHAINS"]]
    if num_ligand_chains is not None:
        lig_partners = lig_partners[: min(num_ligand_chains, params["MAXLIGCHAINS"])]
    return polymer_partners, lig_partners


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


def merge_outs(combined_outs, random_noise: float = 5.0):
    # Combine protein and ligand true coordinates
    xyz = pad_merge_protein_small_molecule_tensors(
        combined_outs["xyz_poly"], combined_outs["xyz_sm"], np.nan
    )
    mask = pad_merge_protein_small_molecule_tensors(
        combined_outs["mask_poly"], combined_outs["mask_sm"], False
    )

    # combine protein & ligand templates
    N_tmpl = combined_outs["xyz_t_poly"].shape[0]
    xyz_t_sm, f1d_t_sm, mask_t_sm, _ = blank_template(
        N_tmpl, sum(combined_outs["Ls_sm"]), random_noise
    )
    xyz_t = torch.cat([combined_outs["xyz_t_poly"], xyz_t_sm], dim=1)
    f1d_t = torch.cat([combined_outs["f1d_t_poly"], f1d_t_sm], dim=1)
    mask_t = torch.cat([combined_outs["mask_t_poly"], mask_t_sm], dim=1)

    # bond features
    bond_feats_poly = [get_protein_bond_feats(L) for L in combined_outs["Ls_poly"]]
    bond_feats_list = bond_feats_poly + combined_outs["bond_feats_sm"]
    bond_feats = torch.block_diag(*bond_feats_list).long()

    # other features
    idx = idx_from_Ls(combined_outs["Ls_poly"] + combined_outs["Ls_sm"])
    same_chain = same_chain_2d_from_Ls(
        combined_outs["Ls_poly"] + combined_outs["Ls_sm"]
    )
    ch_label = torch.cat(
        [
            combined_outs["ch_label_poly"],
            combined_outs["ch_label_sm"] + combined_outs["ch_label_poly"].max() + 1,
        ]
    )

    # load msa
    a3m_sm = {
        "msa": combined_outs["msa_sm"],
        "ins": torch.zeros_like(combined_outs["msa_sm"]),
    }
    a3m = merge_a3m_hetero(
        combined_outs["a3m_poly"],
        a3m_sm,
        [sum(combined_outs["Ls_poly"]), sum(combined_outs["Ls_sm"])],
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
        "Ls_poly": combined_outs["Ls_poly"],
        "Ls_sm": combined_outs["Ls_sm"],
        "residues_to_atomize": combined_outs["residues_to_atomize"],
        "ch_letters_poly": combined_outs["ch_letters"],
        "akeys_sm": combined_outs["akeys_sm"],
        "frames": combined_outs["frames"],
        "chirals": combined_outs["chirals"],
        "seed_msa_clus": combined_outs["seed_msa_clus"],
        "ref_pos_atom36": combined_outs["ref_pos_atom36"],
        "ref_mask": combined_outs["ref_mask"],
        "ref_atom_name_chars": combined_outs["ref_atom_name_chars"],
        "residue_idx": combined_outs["residue_idx"],
        "asym_idx": combined_outs["asym_idx"],
        "symm_idx": combined_outs["symm_idx"],
        "ref_charge": combined_outs["ref_charge"],
    }
    return merged_outs


def reindex_atomize_wrapper(merged_outs, polymer_partners, remove_residue=True):
    if merged_outs["residues_to_atomize"]:
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
            Ls_poly,
            Ls_sm,
        ) = reindex_protein_feats_after_atomize(
            merged_outs["residues_to_atomize"],
            polymer_partners,
            merged_outs["msa"],
            merged_outs["ins"],
            merged_outs["xyz"],
            merged_outs["mask"],
            merged_outs["bond_feats"],
            merged_outs["idx"],
            merged_outs["xyz_t"],
            merged_outs["f1d_t"],
            merged_outs["mask_t"],
            merged_outs["same_chain"],
            merged_outs["ch_label"],
            merged_outs["Ls_poly"],
            merged_outs["Ls_sm"],
            merged_outs["akeys_sm"],
            remove_residue=remove_residue,
        )
        merged_outs["msa"] = msa
        merged_outs["ins"] = ins
        merged_outs["xyz"] = xyz
        merged_outs["mask"] = mask
        merged_outs["bond_feats"] = bond_feats
        merged_outs["idx"] = idx
        merged_outs["xyz_t"] = xyz_t
        merged_outs["f1d_t"] = f1d_t
        merged_outs["mask_t"] = mask_t
        merged_outs["same_chain"] = same_chain
        merged_outs["ch_label"] = ch_label
        merged_outs["Ls_poly"] = Ls_poly
        merged_outs["Ls_sm"] = Ls_sm
    return merged_outs


def get_prev_and_term_info(merged_outs, params):
    mask_t = merged_outs["mask_t"]
    xyz_t = merged_outs["xyz_t"]
    xyz = merged_outs["xyz"]

    ntempl = xyz_t.shape[0]
    xyz_t = torch.stack(
        [
            center_and_realign_missing(
                xyz_t[i], mask_t[i], same_chain=merged_outs["same_chain"]
            )
            for i in range(ntempl)
        ]
    )

    xyz_prev, mask_prev = generate_xyz_prev(xyz_t, mask_t, params)

    xyz_prev = torch.nan_to_num(xyz_prev)
    xyz = torch.nan_to_num(xyz)
    xyz_t = torch.nan_to_num(xyz_t)

    # keep track of protein positions for reindexing chirals after crop
    L_total = sum(merged_outs["Ls_poly"]) + sum(merged_outs["Ls_sm"])
    is_poly = torch.zeros(L_total)
    is_poly[: sum(merged_outs["Ls_poly"])] = 1

    # N/C-terminus features for MSA features (need to generate before cropping)
    term_info = get_term_feats(merged_outs["Ls_poly"] + merged_outs["Ls_sm"])
    term_info[sum(merged_outs["Ls_poly"]) :, :] = (
        0  # ligand chains don't get termini features
    )

    merged_outs["xyz_t"] = xyz_t
    merged_outs["xyz_prev"] = xyz_prev
    merged_outs["mask_prev"] = mask_prev
    merged_outs["is_poly"] = is_poly
    merged_outs["term_info"] = term_info
    return merged_outs


def get_crop_sel(merged_outs, item, params):
    # crop around query ligand (1st sm chain)
    # always need to run cropping function to remove erroneous ligand partners
    idx = merged_outs["idx"]
    mask = merged_outs["mask"]
    msa = merged_outs["msa"]
    Ls_poly = merged_outs["Ls_poly"]
    Ls_sm = merged_outs["Ls_sm"]

    crop_params = params["crop_params"]
    crop_probabilities = crop_params["crop_probabilities"]

    crop_factory = universal_crop_factory
    if (
        len(Ls_sm) > 0
        and len(Ls_poly) > 0
        and crop_params["use_sm_crop_for_sm_compl_examples"]
    ):
        crop_factory = sm_compl_crop_factory

    crop_type_selection = np.random.uniform()
    crop_probability_offset = 0.0
    crop_fn = None
    for crop_type, crop_prob in crop_probabilities.items():
        if crop_type_selection < crop_prob + crop_probability_offset:
            crop_fn = crop_factory[crop_type]
            break
        crop_probability_offset += crop_prob

    if crop_fn is None:
        warnings.warn(
            f"Given crop probabilities {crop_probabilities} do not sum to 1.0. Defaulting to a linear crop."
        )
        sel = get_crop(len(idx), mask[0], msa.device, params["CROP"])
    else:
        sel = crop_fn(
            merged_outs,
            item,
            crop_size=params["CROP"],
            interface_selection_cutoff=crop_params["interface_selection_cutoff"],
        )
    sel = torch.sort(sel).values
    return sel


def apply_crop_sel(merged_outs, sel, item):
    frames = merged_outs["frames"]
    chirals = merged_outs["chirals"]
    mask = reassign_symmetry_after_cropping(
        sel, merged_outs["Ls_poly"], merged_outs["ch_label"], merged_outs["mask"], item
    )

    merged_outs["msa"] = merged_outs["msa"][:, sel]
    merged_outs["ins"] = merged_outs["ins"][:, sel]
    merged_outs["xyz"] = merged_outs["xyz"][:, sel]
    merged_outs["mask"] = mask[:, sel]
    merged_outs["xyz_t"] = merged_outs["xyz_t"][:, sel]
    merged_outs["f1d_t"] = merged_outs["f1d_t"][:, sel]
    merged_outs["mask_t"] = merged_outs["mask_t"][:, sel]
    merged_outs["xyz_prev"] = merged_outs["xyz_prev"][sel]
    merged_outs["mask_prev"] = merged_outs["mask_prev"][sel]
    merged_outs["idx"] = merged_outs["idx"][sel]
    merged_outs["same_chain"] = merged_outs["same_chain"][sel][:, sel]
    merged_outs["bond_feats"] = merged_outs["bond_feats"][sel][:, sel]
    merged_outs["ch_label"] = merged_outs["ch_label"][sel]
    merged_outs["is_poly"] = merged_outs["is_poly"][sel]
    merged_outs["term_info"] = merged_outs["term_info"][sel]
    merged_outs["ref_pos_atom36"] = merged_outs["ref_pos_atom36"][sel]
    merged_outs["ref_mask"] = merged_outs["ref_mask"][sel]
    merged_outs["ref_atom_name_chars"] = merged_outs["ref_atom_name_chars"][sel]
    merged_outs["residue_idx"] = merged_outs["residue_idx"][sel]
    merged_outs["asym_idx"] = merged_outs["asym_idx"][sel]
    merged_outs["symm_idx"] = merged_outs["symm_idx"][sel]
    merged_outs["ref_charge"] = merged_outs["ref_charge"][sel]
    # crop small molecule features, assumes all sm chains are after all protein chains
    atom_sel = sel[sel >= sum(merged_outs["Ls_poly"])] - sum(
        merged_outs["Ls_poly"]
    )  # 0 index all the selected atoms

    frames = frames[atom_sel]
    chirals = crop_chirals(chirals, atom_sel)

    # reindex chiral atom positions - assumes all sm chains are after all protein chains
    if chirals.shape[0] > 0:
        L1 = merged_outs["is_poly"].sum()
        chirals[:, :-1] = chirals[:, :-1] + L1

    merged_outs["frames"] = frames
    merged_outs["chirals"] = chirals

    dist_matrix = get_bond_distances(merged_outs["bond_feats"])
    merged_outs["dist_matrix"] = dist_matrix
    return merged_outs


def apply_featurize_msa(merged_outs, params, fixbb=False):
    # create MSA features from cropped msa and insertions
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(
        merged_outs["msa"].long(),
        merged_outs["ins"].long(),
        params,
        p_mask=params["p_msa_mask"],
        term_info=merged_outs["term_info"],
        fixbb=fixbb,
        seed_msa_clus=merged_outs["seed_msa_clus"],
    )
    merged_outs["seq"] = seq
    merged_outs["msa_seed_orig"] = msa_seed_orig
    merged_outs["msa_seed"] = msa_seed
    merged_outs["msa_extra"] = msa_extra
    merged_outs["mask_msa"] = mask_msa
    return merged_outs


def add_metadata(merged_outs):
    # This flag is whether or not to apply unclamped FAPE at loss calculation time.
    # It is a flag in dataloading because at some point some datasets experimented with
    # unclamping the FAPE loss some fraction of the time.
    merged_outs["unclamp"] = False

    # Whether or not the complex is a true complex or a decoy complex, e.g.
    # a complex that doesn't form. This is used in some auxiliary tasks.
    merged_outs["negative"] = False

    # The symmetry group of the complex. This is used downstream to copy
    # coordinates to all symmetrically equivalent positions.
    merged_outs["symmetry_group"] = "C1"
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
    cif_outs=None,
    **kwargs,
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
    chain_id = item["CHAINID"]
    pdb_id = chain_id.split("_")[0]
    assembly = str(item["ASSEMBLY"])
    if cif_outs is None:
        cif_outs = get_cif_metadata(pdb_id, assembly, params)

    # changing num_protein_chains -> num_polymer_chains
    # will be a separate PR because that requires some config changes
    polymer_partners, lig_partners = get_partner_lists(
        item, params, num_protein_chains, num_ligand_chains
    )

    polymer_outs = load_polymer_partners(
        polymer_partners,
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
        polymer_partners,
        cif_outs,
        params,
        mod_residues_to_atomize=polymer_outs["mod_residues_to_atomize"],
    )
    polymer_keys = set(polymer_outs.keys())
    small_molecule_keys = set(small_molecule_outs.keys())

    # make sure no common keys before combining
    common_keys = polymer_keys.intersection(small_molecule_keys)
    assert len(common_keys) == 0, (
        f"Keys {common_keys} are present in both polymer and small molecule outputs"
    )
    combined_outs = {**polymer_outs, **small_molecule_outs}
    from rf2aa.data.loaders.data_transforms import pipeline

    combined_outs = pipeline(combined_outs)
    merged_outs = merge_outs(combined_outs, random_noise=random_noise)

    merged_outs = reindex_atomize_wrapper(
        merged_outs, polymer_partners, remove_residue=remove_residue
    )
    merged_outs = get_prev_and_term_info(merged_outs, params)

    sel = get_crop_sel(merged_outs, item, params)
    merged_outs = apply_crop_sel(merged_outs, sel, item)
    merged_outs = apply_featurize_msa(merged_outs, params, fixbb=fixbb)

    merged_outs = add_metadata(merged_outs)
    remove_keys = []
    for k, v in merged_outs.items():
        if not isinstance(v, torch.Tensor) and not isinstance(v, bool):
            remove_keys.append(k)
    merged_outs = {k: v for k, v in merged_outs.items() if k not in remove_keys}

    merged_outs["item"] = item
    if "symmetry_group" not in merged_outs:
        merged_outs["symmetry_group"] = "C1"

    return merged_outs


#    return (
# merged_outs["seq"].long(),
# merged_outs["msa_seed_orig"].long(),
# merged_outs["msa_seed"].float(),
# merged_outs["msa_extra"].float(),
# merged_outs["mask_msa"],
# merged_outs["xyz"].float(),
# merged_outs["mask"],
# merged_outs["idx"].long(),
# merged_outs["xyz_t"].float(),
# merged_outs["f1d_t"].float(),
# merged_outs["mask_t"],
# merged_outs["xyz_prev"].float(),
# merged_outs["mask_prev"],
# merged_outs["same_chain"],
# merged_outs["unclamp"],
# merged_outs["negative"],
# merged_outs["frames"],
# merged_outs["bond_feats"],
# merged_outs["dist_matrix"],
# merged_outs["chirals"],
# merged_outs["ch_label"],
# merged_outs["symmetry_group"],
# task,
# item,
# )


def loader_sm_compl_assembly_single(*args, **kwargs):
    kwargs["num_protein_chains"] = 1
    return loader_sm_compl_assembly(*args, **kwargs)
