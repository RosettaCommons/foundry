import torch
import ast
import numpy as np
import networkx as nx

from collections import OrderedDict
from typing import Optional, Dict, Any

from rf2aa.data.parsers import parse_mol
from rf2aa.data.chain_crop import crop_chirals
from rf2aa.chemical import ChemicalData as ChemData
from rf2aa.kinematics import get_chirals
from rf2aa.data.identical_ligands import get_extra_identical_copies_from_chains
from rf2aa.util import (
    get_nxgraph,
    get_atom_frames,
    get_bond_feats,
    cif_ligand_to_xyz,
    cif_ligand_to_obmol,
    get_automorphs,
    get_ligand_atoms_bonds,
)


def prune_lig_partners(lig_partners, params):
    lig_partners = [
        p
        for p in lig_partners
        if (
            p[0][0][2] not in ChemData().METAL_RES_NAMES
            or np.random.rand() < params["P_METAL"]
        )
    ]  # remove metals (dep. on param)
    lig_partners = [
        p for p in lig_partners if p[0][0][2] not in params["ligands_to_remove"]
    ]  # fd remove exclusion list
    return lig_partners


def find_residues_to_atomize_covale(lig_partners, prot_partners, covale):
    """
    Updates partner lists to have atomized residues when residues are making
    covalent bonds with small molecules.  Also returns list of atomized
    residues so the other features, MSA, templates etc can be removed.

    Parameters
    ----------
    lig_partners : list of 5-tuples
        Ligands in this assembly. Format is as described in `loader_sm_compl_assembly`.
    prot_partners : list of 5-tuples
        Protein chains in this assembly. Format is as described in `loader_sm_compl_assembly`.
    covale : list
        List of cifutils.Bond objects representing inter-chain bonds in this PDB entry.

    Returns
    -------
    lig_partners : list of 5-tuples
        New list of ligands in this assembly, with additional "ligands" corresponding to
        residues to atomize.
    residues_to_atomize : list of tuples ((ch_letter, res_num, res_name), (ch_letter, xform_index))
    """
    if len(covale) == 0:
        return lig_partners, set()

    residues_to_atomize = set()
    for bond in covale:
        # ignore bonds to hydrogens -- these are artifacts of PDB curation
        if bond.a[-1][0] == "H" or bond.b[-1][0] == "H":
            continue

        res_key = None
        i_prot = None
        i_lig = None

        # find protein partner that is bonded to ligand
        for i, (ch_letter, i_xf, n_contacts, min_dist, ptype) in enumerate(
            prot_partners
        ):
            if bond.a[0] == ch_letter:
                i_prot = i
                res_key = bond.a
                break
            elif bond.b[0] == ch_letter:
                i_prot = i
                res_key = bond.b
                break

        # find ligand partner that is bonded to protein
        for i, (ligand, ch_xfs, n_contacts, min_dist, ptype) in enumerate(lig_partners):
            if any(
                [bond.a[:3] == lig_res or bond.b[:3] == lig_res for lig_res in ligand]
            ):
                i_lig = i
                break

        if i_prot is not None and i_lig is not None:
            lig_partner = lig_partners[i_lig]
            prot_partner = prot_partners[i_prot]

            # append to ligand partner the protein residue that it's bonded to
            lig_partner[0].append(res_key[:3])
            lig_partner[1].append(prot_partner[:2])

            # record this residue to remove from residue representations
            residues_to_atomize.add((res_key[:3], prot_partner[:2]))

    return lig_partners, residues_to_atomize


def featurize_single_ligand(ligand, chains, covale, lig_xf_s, asmb_xfs, params):
    """Loads a single ligand in a specific assembly from a parsed CIF file into
    tensors. If more than one coordinate transform exists for the ligand, the
    additional copies of the molecule are concatenated into the symmetry
    permutation dimension if atom occupancy is fractional (between 0 and 1) and
    appended to a list (for later concatenation along the residue dimension) if
    atom occupancy is equal to 1.0.

    Parameters
    ----------
    ligand : list of tuples (chain_letter, res_num, res_name)
    chains : dict
        Dictionary mapping chain letters to cifutils.Chain objects representing
        the chains in a PDB entry.
    covale : list
        List of cifutils.Bond objects representing inter-chain bonds in this
        PDB entry.
    lig_xf_s : list of tuples (chain_letter, transform_index)
    asmb_xfs : list of tuples (chain_letter, np.array(4,4))
        Coordinate transforms for this assembly
    params : dict
        Data loader parameters.

    Returns
    -------
    All outputs will be a list of length `N_lig` (number of copies of this
    ligand in the assembly):

    xyz_lig : list of torch.Tensor (N_permutation, L, 3), float
        Atom coordinates of this ligand. This list will have length > 1 if
        multiple coordinate transforms exist and atom occupancy is 1. If
        multiple transforms exist and atom occupancy is between 0 and 1, this
        will be a list with one coordinate tensor containing multiple sets of
        coordinates in the symmetry dimension.
    mask_lig : list of torch.Tensor (N_permutation, L), bool
        Mask that is true if a certain atom exists.
    msa_lig : list of torch.Tensor (L_total,)
    bond_feats_lig : list of torch.Tensor (L, L)
    akeys_lig : list of tuples (chain_id, residue_num, residue_name, atom_name)
    Ls_lig : list of int
    frames_lig : list of torch.Tensor (L, 3, 2)
    chirals_lig : list of torch.Tensor (N_chirals, 5)
    resnames : list
        Residue names of each ligand in the returned lists
    """
    lig_atoms, lig_bonds = get_ligand_atoms_bonds(ligand, chains, covale)

    xyz_lig, mask_lig, msa_lig, bond_feats_lig, akeys_lig, Ls_lig = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    frames_lig, chirals_lig, resname_lig = [], [], []

    # fd keep track of the ligands that are added along length dimension (as opposed to conformation)
    unique_lig = []

    for lig_xf in lig_xf_s:  # all possible locations for this ligand
        ch2xf = dict(lig_xf)
        xyz_, occ_, msa_, chid_, akeys_ = cif_ligand_to_xyz(lig_atoms, asmb_xfs, ch2xf)
        if (occ_ == 0).all():
            continue  # no valid atom positions
        mask_ = occ_ > 0  # partially occupied atoms are considered valid

        mol_, bond_feats_ = cif_ligand_to_obmol(xyz_, akeys_, lig_atoms, lig_bonds)
        xyz_, mask_ = get_automorphs(mol_, xyz_, mask_)

        # clamp number of atom permutations to save GPU memory
        if xyz_.shape[0] > params["MAXNSYMM"]:
            print(
                f"WARNING: Too many atom permutations ({xyz_.shape[0]}) in {ligand}. "
                f'Keeping only {params["MAXNSYMM"]}.'
            )
            xyz_ = xyz_[: params["MAXNSYMM"]]
            mask_ = mask_[: params["MAXNSYMM"]]

        G = get_nxgraph(mol_)
        frames_ = get_atom_frames(msa_, G, omit_permutation=params["OMIT_PERMUTATE"])
        chirals_ = get_chirals(mol_, xyz_[0])
        # if ligand has too many masked atoms, remove all masked atoms
        # this avoids wasting compute on ligands missing entire chemical fragments
        # while keeping (most) masked atoms that are isolated and integral to a given fragment
        if (~mask_[0]).sum() > params["MAXMASKEDLIGATOMS"]:
            xyz_ = xyz_[:, mask_[0]]
            occ_ = occ_[mask_[0]]
            msa_ = msa_[mask_[0]]
            chid_ = chid_[mask_[0]]
            akeys_ = [k for m, k in zip(mask_[0], akeys_) if m]
            bond_feats_ = bond_feats_[mask_[0]][:, mask_[0]]
            G = nx.Graph(bond_feats_.cpu().numpy())
            frames_ = get_atom_frames(
                msa_, G, omit_permutation=params["OMIT_PERMUTATE"]
            )
            chirals_ = crop_chirals(chirals_, torch.where(mask_[0])[0])
            mask_ = mask_[:, mask_[0]]

        if chirals_.numel() > 0:
            chirals_[:, :-1] = chirals_[:, :-1] + sum(Ls_lig)

        if ((occ_ < 1) & (occ_ > 0)).any():
            # partial occupancy, add to permutation dimension
            # if not ((occ_<1) & (occ_>0)).all():
            #    print('WARNING: Partial occupancy for a subset of atoms in ligand', ligand)
            #    print('         Adding to permutation dimension as alternate coordinates.')
            if len(xyz_lig) == 0:
                xyz_lig = [xyz_]
                mask_lig = [mask_]
                msa_lig = [msa_]
                bond_feats_lig = [bond_feats_]
                akeys_lig = [akeys_]
                Ls_lig = [xyz_.shape[1]]
                frames_lig = [frames_]
                chirals_lig = [chirals_]
                resname_lig = ["_".join([res[2] for res in ligand])]
                unique_lig = [True]
            else:
                xyz_lig[0] = torch.cat([xyz_lig[0], xyz_], dim=0)
                mask_lig[0] = torch.cat([mask_lig[0], mask_], dim=0)
                unique_lig.append(False)
        else:
            # full occupancy, add as new chain
            xyz_lig.append(xyz_)
            mask_lig.append(mask_)
            msa_lig.append(msa_)
            bond_feats_lig.append(bond_feats_)
            akeys_lig.append(akeys_)
            Ls_lig.append(xyz_.shape[1])
            frames_lig.append(frames_)
            chirals_lig.append(chirals_)
            resname_lig.append("_".join([res[2] for res in ligand]))
            unique_lig.append(True)

    return (
        xyz_lig,
        mask_lig,
        msa_lig,
        bond_feats_lig,
        akeys_lig,
        Ls_lig,
        frames_lig,
        chirals_lig,
        resname_lig,
        unique_lig,
    )


def featurize_asmb_ligands(partners, params, chains, asmb_xfs, covale):
    """Loads multiple ligands chains from parsed CIF assembly into tensors.
    Outputs will contain ligands in the order that they appear in
    `partners` (decreasing number of contacts to query ligand). Leading
    dimension of output coordinates contains atom position permutations for
    each ligand.  Atom permutations between different ligands that are
    identical are not generated here, so loss must be calculated in a way that
    accounts for inter-ligand swap permutations.

    Parameters
    ----------
    partners : list of 5-tuples (partner, transform_index, num_contacts, min_dist, partner_type)
        Ligands to featurize. All elements should have `partner_type =
        'nonpoly'` and `partner` is a list of tuples (chain_letter, res_num,
        res_name) corresponding to the residues that make up this ligand.
        `transform_index` will be a list of tuples (chain_letter, idx)
        indicating the index of the coordinate transform for each chain in the
        ligand.
    params : dict
        Parameters for the data loader
    chains : dict
        Dictionary mapping chain letters to cifutils.Chain objects representing
        the chains in a PDB entry.
    asmb_xfs : list of 2-tuples (chain_id, torch.Tensor(4,4))
        Coordinate transforms for the current assembly
    covale : dict
        List of cifutils.Bond objects representing inter-chain bonds in this
        PDB entry.

    Returns
    -------
    xyz_sm : tensor (N_atom_permutation, L_total, N_atoms, 3)
        Atom coordinates of all the ligands.
    mask_sm : tensor (N_atom_permutation, L_total, N_atoms)
        Boolean mask for whether an atom exists in `xyz_sm`.
    msa_sm : tensor (L_total,)
        Integer-coded "sequence" (atom types) of the ligands
    bond_feats_sm : list of tensors (L_chain, L_chain)
        List of bond feature matrices for each ligand chain
    frames : (L_total, 3, 2)
        Frame atom relative indices for each ligand atom
    chirals : (N_chiral_atoms, 5)
        Chiral features (4 residue indices and 1 ideal dihedral angle) for each
        chiral center
    sm_Ls : list
        Length of each ligand
    ch_label_sm : tensor (L_total,)
        Integer-coded chain identity for each ligand. Ligands are assigned the
        same code if their representation as an ordered list of tuples
        (residue_name, atom_name) is the same.
    akeys_sm : list
        list of tuples (chid, resnum, resname, atomtype). Used downstream to map atom identities to index in xyz tensors
    lig_names : list
        Name of each ligand (residue name(s) joined by '_')
    """
    # group ligands with identical chain & residue numbers
    # these may represent alternate locations of the same molecule
    # and need to be loaded into permutation dimension
    ligands = []
    lig2xf = OrderedDict()
    for p in partners:
        if p[-1] != "nonpoly":
            continue
        ligands.append(p[0])
        k = str(p[0])  # make multires ligand into string for using as dict key
        if k not in lig2xf:
            lig2xf[k] = []
        lig2xf[k].append(p[1])

    # load all ligands
    (
        xyz_sm,
        mask_sm,
    ) = (
        [],
        [],
    )
    msa_sm, bond_feats_sm, akeys_sm, Ls_sm, frames, chirals, resnames = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    # fd keep track of the ligands that are added along length dimension (as opposed to conformation)
    uniques = []

    for ligkey, lig_xf_s in lig2xf.items():

        ligand = ast.literal_eval(ligkey)

        (
            xyz_lig,
            mask_lig,
            msa_lig,
            bond_feats_lig,
            akeys_lig,
            Ls_lig,
            frames_lig,
            chirals_lig,
            resname_lig,
            unique_lig,
        ) = featurize_single_ligand(ligand, chains, covale, lig_xf_s, asmb_xfs, params)

        # residue numbering offset for chirals
        for i in range(len(chirals_lig)):
            if chirals_lig[i].shape[0] > 0:
                chirals_lig[i][:, :-1] = chirals_lig[i][:, :-1] + sum(Ls_sm)

        xyz_sm.extend(xyz_lig)
        mask_sm.extend(mask_lig)
        msa_sm.extend(msa_lig)
        bond_feats_sm.extend(bond_feats_lig)
        akeys_sm.extend(akeys_lig)
        Ls_sm.extend(Ls_lig)
        frames.extend(frames_lig)
        chirals.extend(chirals_lig)
        resnames.extend(resname_lig)
        uniques.extend(unique_lig)

    # concatenate features
    msa_sm = torch.cat(msa_sm, dim=0)
    frames = torch.cat(frames, dim=0)
    chirals = torch.cat(chirals, dim=0)

    # concatenate coordinates with enough room for the largest symmetry permutation dimension
    N_symm = max([xyz_.shape[0] for xyz_ in xyz_sm])
    xyz_out = torch.full((N_symm, sum(Ls_sm), 3), np.nan)
    mask_out = torch.full((N_symm, sum(Ls_sm)), False)
    i_res = 0

    for xyz_, mask_ in zip(xyz_sm, mask_sm):
        N_symm_, L_ = xyz_.shape[:2]
        xyz_out[:N_symm_, i_res : i_res + L_] = xyz_
        mask_out[:N_symm_, i_res : i_res + L_] = mask_
        i_res += L_
    xyz_sm, mask_sm = xyz_out, mask_out

    # detect which ligands are the same
    # ligands are considered identical if they have identical string representations
    # of an ordered list of (lig name, atom name) tuples
    lig_dict = dict()
    for i in range(len(akeys_sm)):
        ak = str(sorted([x[2:] for x in akeys_sm[i]]))  # [(lig_name, atom_name), ...]
        if ak not in lig_dict:
            lig_dict[ak] = []
        lig_dict[ak].append(i)

    keymap = dict(zip(lig_dict.keys(), range(len(lig_dict))))
    idx2label = dict()
    for k, v in lig_dict.items():
        for idx in v:
            idx2label[idx] = keymap[k]

    ch_label_sm = [torch.full((L_,), idx2label[i]) for i, L_ in enumerate(Ls_sm)]
    ch_label_sm = torch.cat(ch_label_sm, dim=0)

    return (
        xyz_sm,
        mask_sm,
        msa_sm[None],
        bond_feats_sm,
        frames,
        chirals,
        Ls_sm,
        ch_label_sm,
        akeys_sm,
        resnames,
        uniques,
    )


def featurize_ligand_from_string(
    ligand_string: str, format: str = "smiles", omit_permutation: bool = False
):
    """featurize_ligand_from_smiles Featurizes a smiles string
    representing a ligand, in a way that can be input into RF2 All Atom.

    Args:
        smiles_string (str): A Smiles String representing a small molecule.

    Returns:
        _type_: Same outputs as _load_sm_from_item, as if the ligand was loaded
        from the RF database.
    """
    generate_conformer = False
    if format == "inchi" or format == "smiles" or format == "smi":
        # We only generate conformers if we are reading from a format
        # that doesn't speicify the coordinates
        generate_conformer = True

    mol, msa_sm, _, xyz_sm, mask_sm = parse_mol(
        filename=ligand_string,
        filetype=format,
        string=True,
        generate_conformer=generate_conformer,
    )
    small_molecule_length = mol.NumAtoms()

    chirals = get_chirals(mol, xyz_sm[0])
    bond_feats_sm = get_bond_feats(mol)

    mol_graph = get_nxgraph(mol)
    frames = get_atom_frames(
        msa_sm,
        mol_graph,
        omit_permutation=omit_permutation,
    )
    ch_label_sm = torch.zeros((small_molecule_length), dtype=int)
    lig_names = [ligand_string]

    akeys_sm = []
    return (
        xyz_sm,
        mask_sm,
        msa_sm.unsqueeze(0),
        [bond_feats_sm],
        frames,
        chirals,
        [small_molecule_length],
        ch_label_sm,
        akeys_sm,
        lig_names,
    )


def remove_unsupported_metals(
    lig_partners,
    xyz_prot,
    mask_prot,
    xyz_sm,
    mask_sm,
    msa_sm,
    bond_feats_sm,
    frames,
    chirals,
    Ls_sm,
    ch_label_sm,
    akeys_sm,
    resnames,
    residues_to_atomize,
    prot_partners,
    asmb_xfs,
    chains,
    covale,
    params,
    mod_residues_to_atomize,
    num_ligand_chains,
    uniques,
    min_metal_contacts,
    min_metal_contact_dist,
):
    i_start = 0
    lig_partners_new = []
    rebuild = False

    nligands = len(lig_partners)
    assert (
        len(uniques) >= nligands
    )  # fd this is > since atomized residues may implicitly get added

    i_unique = -1
    for i_lig in range(nligands):
        if not uniques[i_lig]:
            res = resnames[i_unique]
            if res not in ChemData().METAL_RES_NAMES:
                lig_partners_new.append(
                    lig_partners[i_lig]
                )  # alternate conf of a non-metal, keep
            continue

        i_unique += 1
        res = resnames[i_unique]
        i_stop = i_start + Ls_sm[i_unique]

        if res in ChemData().METAL_RES_NAMES:
            assert Ls_sm[i_unique] == 1

            # 1) get ligand contacts
            ds = torch.linalg.norm(xyz_sm[0] - xyz_sm[0, i_start], dim=-1)
            nself = sum(ds[mask_sm[0]] < min_metal_contact_dist) - 1  # -1 for self

            # 2) get protein contacts
            ds = torch.linalg.norm(xyz_prot[0, :, 1] - xyz_sm[0, i_start], dim=-1)
            # trim to residue contacts (8 is maximal CA/SC atom distance)
            resmask = (ds < (min_metal_contact_dist + 8.0)) * mask_prot[0, :, 1]
            ds = torch.linalg.norm(xyz_prot[0, resmask, :] - xyz_sm[0, i_start], dim=-1)
            nprot = sum(ds[mask_prot[0, resmask]] < min_metal_contact_dist)
            if nprot + nself >= min_metal_contacts:
                lig_partners_new.append(lig_partners[i_lig])
            else:
                rebuild = True
        else:
            # nonmetal, keep
            lig_partners_new.append(lig_partners[i_lig])

        i_start = i_stop

    if rebuild:
        if len(lig_partners_new) == 0:  # no ligands left after trimming
            return (
                torch.tensor([]),
                torch.tensor([], dtype=torch.bool),
                torch.tensor([]),
                [],
                torch.tensor([], dtype=torch.long),
                torch.tensor([]),
                [],
                torch.tensor([], dtype=torch.long),
                [],
                [],
                [],
                [],
            )
        else:
            return load_small_molecule_partners(
                lig_partners_new,
                prot_partners,
                asmb_xfs,
                chains,
                covale,
                params,
                mod_residues_to_atomize,
                num_ligand_chains=num_ligand_chains,
                check_for_nonpartner_duplicates=False,
            )
    else:
        return (
            xyz_sm,
            mask_sm,
            msa_sm,
            bond_feats_sm,
            frames,
            chirals,
            Ls_sm,
            ch_label_sm,
            akeys_sm,
            resnames,
            residues_to_atomize,
            uniques,
        )

def get_empty_small_molecule_partners() -> Dict[str, Any]:
    lig_outs = {
        "xyz_sm": torch.zeros((1, 0, 3), dtype=torch.float32),
        "mask_sm": torch.zeros((1, 0), dtype=torch.bool),
        "msa_sm": torch.zeros((1, 0), dtype=torch.long),
        "bond_feats_sm": [],
        "frames": torch.zeros((0, 3, 2), dtype=torch.long),
        "chirals": torch.zeros((0, 5), dtype=torch.float32),
        "Ls_sm": [],
        "ch_label_sm": torch.zeros((0,), dtype=torch.long),
        "akeys_sm": [],
        "lig_names": [],
        "residues_to_atomize": [],
        "uniques": [],
    }
    return lig_outs


def load_small_molecule_partners(
    lig_partners,
    prot_partners,
    cif_outs,
    params,
    mod_residues_to_atomize,
    check_for_nonpartner_duplicates: Optional[bool] = False,
) -> Dict[str, Any]:
    # update ligand partners to atomize residues that are covalently linked to proteins
    lig_partners, residues_to_atomize = find_residues_to_atomize_covale(
        lig_partners, prot_partners, cif_outs["covale"]
    )

    # subsample non-standard residues to atomize
    mod_residues_to_atomize = [
        res
        for res in mod_residues_to_atomize
        if np.random.rand() < params["P_ATOMIZE_MODRES"]
    ]

    # update ligand partners and residues_to_atomize with modified residues to be atomized
    lig_partners.extend(
        [
            (
                [res_tuple],
                [ch_xf],
                -1,
                "nonpoly",
            )  # multi-res ligand format
            for (res_tuple, ch_xf) in mod_residues_to_atomize
        ]
    )
    residues_to_atomize.update(set(mod_residues_to_atomize))

    if len(lig_partners) == 0:
        return get_empty_small_molecule_partners()

    # load ligands
    (
        xyz_sm,
        mask_sm,
        msa_sm,
        bond_feats_sm,
        frames,
        chirals,
        Ls_sm,
        ch_label_sm,
        akeys_sm,
        lig_names,
        uniques,
    ) = featurize_asmb_ligands(
        lig_partners,
        params,
        cif_outs["chains"],
        cif_outs["asmb_xfs"],
        cif_outs["covale"],
    )

    if check_for_nonpartner_duplicates:
        try:
            xyz_sm, mask_sm = get_extra_identical_copies_from_chains(
                cif_outs["chains"],
                cif_outs["covale"],
                cif_outs["asmb_xfs"],
                xyz_sm,
                mask_sm,
                Ls_sm,
                akeys_sm,
                lig_partners,
                exclude_covalent_to_protein=False,
            )
        except Exception as e:
            print(
                "Failed to get extra identical copies of the ligands from the cif assembly."
            )
            print(f"The error was: {e}.")
            pass

    lig_outs = {
        "xyz_sm": xyz_sm,
        "mask_sm": mask_sm,
        "msa_sm": msa_sm,
        "bond_feats_sm": bond_feats_sm,
        "frames": frames,
        "chirals": chirals,
        "Ls_sm": Ls_sm,
        "ch_label_sm": ch_label_sm,
        "akeys_sm": akeys_sm,
        "lig_names": lig_names,
        "residues_to_atomize": residues_to_atomize,
        "uniques": uniques,
    }
    return lig_outs
