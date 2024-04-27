import pandas as pd
import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Optional, Set
from rf2aa.cifutils import Chain, Bond
from rf2aa.chemical import ChemicalData
from rf2aa.util import cif_ligand_to_xyz, cif_prot_to_xyz, get_ligand_atoms_bonds
from rf2aa.data.data_loader import get_msa
from criterion_utils import get_ligand_name_from_query_ligand


def get_residue_set_for_chain(chain: Chain) -> Set[Tuple[str, str, str]]:
    return set([(k[0], k[1], k[2]) for k in chain.atoms.keys()])


def preparse_all_chains(
    chains: Dict[str, Chain],
    ignore_ligands: List[str] = ["HOH", "DOD", "EDO", "PEG", "GOL"],
) -> Dict[str, Any]:
    chain_xyz_dict = {}
    good_chain_types = [
        "polypeptide(L)",
        "polydeoxyribonucleotide",
        "polyribonucleotide",
        "nonpoly",
    ]
    for chain_id, chain in chains.items():
        if chain.type not in good_chain_types:
            continue
        chain_name = next(iter(chain.atoms.keys()))[2]
        if chain_name in ignore_ligands:
            continue

        chain_xyz_dict[chain_id] = chain_to_xyz(chain)
    return chain_xyz_dict


def residue_set_covers_all_atoms(
    residue_set: Set[Tuple[str, str, str]], chain: Chain
) -> bool:
    chain_residue_set = get_residue_set_for_chain(chain)
    return chain_residue_set == residue_set


def preprocess_asmb(
    asmb: Dict[str, List[Tuple[str, np.ndarray]]]
) -> Dict[str, List[Tuple[str, torch.Tensor]]]:
    new_asmb = {}
    for k, v in asmb.items():
        new_asmb[k] = [(x[0], torch.from_numpy(x[1]).float()) for x in v]
    return new_asmb


def get_bonded_partners(
    lig_tuple: Tuple[str, ...], bonds: List[Bond]
) -> List[Tuple[str, ...]]:
    """Get a list of tuples representing a set of bonded ligand residues.

    Parameters
    ----------
    lig_tuple : tuple (chain id, res num, lig name)
        3-tuple representation of a ligand residue
    bonds : list
        List of cifutils.Bond objects, representing all the bonds that
        `lig_tuple` may participate in.

    Returns
    -------
    partners : list
        List of 3-tuples representing ligand residues that contain bonds to
        `lig_tuple`. Does not include `lig_tuple` itself.
    """
    partners = set()
    new_bonds = []
    for bond in bonds:
        if bond.a[:3] == lig_tuple:
            partners.add(bond.b[:3])
        elif bond.b[:3] == lig_tuple:
            partners.add(bond.a[:3])
        else:
            new_bonds.append(bond)

    partners = set([p for p in partners if p != lig_tuple])

    new_partners = []
    for p in partners:
        new_partners.append(get_bonded_partners(p, new_bonds))

    for new_p in new_partners:
        partners.update(new_p)

    return partners


def get_ligands(
    chains: Dict[str, Chain],
    covale: List[Bond],
    ignore_ligands: List[str] = ["HOH", "DOD", "EDO", "PEG", "GOL"],
) -> Tuple[
    List[List[Tuple[str, ...]]], List[List[Tuple[Tuple[str, ...], Tuple[str, ...]]]]
]:
    """Gets a list of lists of ligand residue 3-tuples, representing all the
    ligands contained in a given PDB assembly.
    Parameters
    ----------
    chains : dict
        Dictionary mapping chain letters to cifutils.Chain objects representing
        the chains in a PDB entry.
    covale : list
        List of cifutils.Bond objects representing inter-chain bonds in this
        PDB entry.

    Returns
    -------
    ligands : list
        List of lists of 3-tuples (chain id, res num, lig name), representing
        all the covalently bonded sets of ligand residues that make up each full
        small molecule ligand in this PDB entry.
    lig_covale : list
        Covalent bonds from protein to any residue in each ligand in `ligands`
    """
    # collect all ligand residues and potential inter-ligand bonds
    lig_res_s = list(
        set(
            [
                x[:3]
                for ch in chains.values()
                if ch.type == "nonpoly"
                for x in ch.atoms
                if x[2] not in ignore_ligands
            ]
        )
    )
    bonds = []
    for i_ch, ch in chains.items():
        if ch.type == "nonpoly":
            bonds.extend(ch.bonds)
    inter_ligand_bonds = []
    prot_lig_bonds = []
    for bond in covale:
        if chains[bond.a[0]].type == "nonpoly" and chains[bond.b[0]].type == "nonpoly":
            bonds.append(bond)
            inter_ligand_bonds.append(bond)
        if sorted([chains[bond.a[0]].type, chains[bond.b[0]].type]) == [
            "nonpoly",
            "polypeptide(L)",
        ]:
            prot_lig_bonds.append(bond)

    # make list of bonded ligands (lists of ligand residues)
    ligands = []
    lig_covale = []
    while len(lig_res_s) > 0:
        res = lig_res_s[0]
        lig = get_bonded_partners(res, bonds)
        lig.add(res)
        lig = sorted(list(lig))
        lig_res_s = [res for res in lig_res_s if res not in lig]
        ligands.append(lig)
        lig_covale.append(
            [
                (bond.a, bond.b)
                for bond in prot_lig_bonds
                if any([bond.a[:3] == res or bond.b[:3] == res for res in lig])
            ]
        )

    return ligands, lig_covale


def filter_ligands(
    ligands: List[List[Tuple[str, ...]]],
    lig_covale: List[List[Tuple]],
    max_unique_copies_per_cif: int = 20,
) -> Tuple:
    ligand_ids = [get_ligand_name_from_query_ligand(lig) for lig in ligands]
    unique_ligand_ids = set(ligand_ids)
    rng = np.random.default_rng(42)

    indices = []
    for id in unique_ligand_ids:
        indices_with_id = [i for i, lig_id in enumerate(ligand_ids) if lig_id == id]
        count = len(indices_with_id)
        if count > max_unique_copies_per_cif:
            print(
                f"WARNING: Ligand {id} has {count} unique copies in this PDB entry. Only the first {max_unique_copies_per_cif} will be used."
            )
            indices_with_id = rng.choice(
                indices_with_id, max_unique_copies_per_cif, replace=False
            )
        indices.extend(indices_with_id)

    ligands = [ligands[i] for i in indices]
    lig_covale = [lig_covale[i] for i in indices]
    return ligands, lig_covale


def chain_to_xyz(
    ch: Chain,
    residue_set: List[Tuple[str, ...]] = None,
    chain_xyz_dict: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str], List[str], set]:
    """Featurizes a cifutils.Chain into a torch.Tensor of atom coordinates
    suitable for RoseTTAFold.

    Parameters
    ----------
    ch : cifutils.Chain
        Representation of a protein/DNA/RNA chain from a PDB entry
    residue_set : list
        List of 3-tuples (chain id, res id, res name) representing specific
        residues to featurize. If provided, only atoms belonging to these
        residues will be featurized. Used to featurize a particular
        ligand while ignoring other ligands or protein residues on the same
        chain.

    Returns
    ------
    xyz : torch.Tensor (N_residues, N_atoms, 3)
        Atom coordinates, with standard atom ordering as in RF. Small molecules
        will have each atom coordinate assigned to each "residue" in its
        C-alpha atom slot (index 1 of dimension 1).
    mask : torch.Tensor (N_residues, N_atoms)
        Boolean tensor indicating if an atom exists at a given location in
        `xyz`
    seq : torch.Tensor (N_residues,)
        Tensor of integers (long) encoding the amino acid, base, or element at
        each residue position in `xyz`.
    chid : list
        List of chain letters for each residue
    resi : list
        List of residue numbers (as strings) for each residue. For ligands,
        this might be the same number across many residue slots
    unrec_elements : set
        Set of atomic numbers of elements that aren't in current RF alphabet
        and have been featurized to `ATM` (unknown element)
    """
    chain_id = ch.id
    if chain_xyz_dict is not None and chain_id in chain_xyz_dict:
        if residue_set is None or residue_set_covers_all_atoms(residue_set, ch):
            return chain_xyz_dict[chain_id]

    if ch.type in ["polypeptide(L)", "polydeoxyribonucleotide", "polyribonucleotide"]:
        idx = [int(k[1]) for k in ch.atoms]
        i_min, i_max = np.min(idx), np.max(idx)
        L = i_max - i_min + 1
    elif ch.type == "nonpoly":
        atoms_no_H = {
            k: v for k, v in ch.atoms.items() if v.element != 1
        }  # exclude hydrogens
        L = len(atoms_no_H)

    xyz = torch.zeros(L, ChemicalData().NTOTAL, 3)
    mask = torch.zeros(L, ChemicalData().NTOTAL).bool()
    seq = torch.full((L,), np.nan)
    chid = ["-"] * L
    resi = ["-"] * L

    unrec_elements = set()

    # chain-type-specific unknown tokens from RF alphabet
    unk = {
        "polypeptide(L)": 20,
        "polydeoxyribonucleotide": 26,
        "polyribonucleotide": 31,
    }

    aa2long_ = [
        [x.strip() if x is not None else None for x in y]
        for y in ChemicalData().aa2long
    ]
    aa2num_ = {k.strip(): v for k, v in ChemicalData().aa2num.items()}

    if ch.type in ["polypeptide(L)", "polydeoxyribonucleotide", "polyribonucleotide"]:
        for k, v in ch.atoms.items():
            if k[2] == "HOH":
                continue  # skip waters
            if residue_set is not None and k[:3] not in residue_set:
                continue
            i_res = int(k[1]) - i_min
            aa_name = "R" + k[2] if ch.type == "polyribonucleotide" else k[2]
            if aa_name in aa2num_ and (aa2num_[aa_name] <= 31):  # standard AA/DNA/RNA
                aa = aa2num_[aa_name]
                if k[3] in aa2long_[aa]:  # atom name exists in RF nomenclature
                    i_atom = aa2long_[aa].index(k[3])  # atom index
                    xyz[i_res, i_atom, :] = torch.tensor(v.xyz)
                    mask[i_res, i_atom] = v.occ
                seq[i_res] = aa
            else:
                seq[i_res] = unk[ch.type]  # unknown
            chid[i_res] = k[0]
            resi[i_res] = k[1]

    elif ch.type == "nonpoly":
        for i, (k, v) in enumerate(atoms_no_H.items()):
            if k[2] == "HOH":
                continue  # skip waters
            if residue_set is not None and k[:3] not in residue_set:
                continue
            xyz[i, 1, :] = torch.tensor(v.xyz)
            mask[i, 1] = v.occ  # fractional occupancies cast to True
            if v.element not in ChemicalData().atomnum2atomtype:
                seq[i] = aa2num_["ATM"]
                unrec_elements.add(v.element)
            else:
                seq[i] = aa2num_[ChemicalData().atomnum2atomtype[v.element]]
            chid[i] = k[0]
            resi[i] = k[1]

    return xyz, mask, seq, chid, resi, unrec_elements


def get_ligand_xyz(
    chains: Dict[str, Chain],
    asmb_xforms: List[Tuple[str, np.ndarray]],
    ligand: List[Tuple[str, ...]],
    seed_ixf: int = None,
    chain_xyz_dict: Optional[Dict[str, Any]] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    List[str],
    List[str],
    List[Tuple[str, int]],
]:
    """Featurizes atom coordinates of a (potentially multi-residue) ligand.
    Used only for geometric comparisons such as neighbor detection. Does not
    process chemical features such as bond graphs that are needed for actual
    input to RF.

    For multi-residue ligands, starts by constructing a tensor with the
    coodinates of the 1st residue, and applies the 1st coordinate transform for
    that residue's chain that appears in the provided list of transforms.
    For each subsequent residue, all transforms that exist for
    that residue's chain are tried, but only the transformed coordinates with the
    single closest atom to the featurized ligand so far (i.e. makes a covalent
    bond) are kept. Optionally, the 1st residue can be featurized using a
    transform with index `seed_ixf` that's not the first in the transform list,
    to get an alternative location for this ligand.

    Parameters
    ----------
    chains : dict[str, cifutils.Chain]
        Chains in this PDB entry
    asmb_xforms : list
        List of tuples (chain letter, transform matrix) representing coordinate
        transforms for all the chains in this assembly.
    ligand : list
        List of 3-tuples (chain id, res num, lig name), representing a specific
        ligand (and all its constituent residues)
    seed_ixf : int
        Index in `asmb_xforms` of the coordinate transform to apply to the 1st
        ligand residue.
    """
    assert (
        seed_ixf is None or asmb_xforms[seed_ixf][0] == ligand[0][0]
    ), "ERROR: Seed transform index is not consistent with provided ligand"

    asmb_xform_chids = [x[0] for x in asmb_xforms]

    ligand_chids = []
    for x in ligand:
        if x[0] not in ligand_chids:
            ligand_chids.append(x[0])

    lig_xyz = torch.tensor([]).float()
    lig_mask = torch.tensor([]).bool()
    lig_seq = torch.tensor([]).long()
    lig_chid, lig_resi, lig_i_ch_xf = [], [], []

    for i_ch_lig in ligand_chids:
        ch = chains[i_ch_lig]

        xyz, mask, seq, chid, resi, unrec_elements = chain_to_xyz(
            ch,
            residue_set=ligand,
            chain_xyz_dict=chain_xyz_dict,
        )
        if not mask[:, 1].any():
            continue  # all CA's/ligand atoms missing
        # if len(unrec_elements)>0:
        #     print('pdbid', pdbid, 'ligands',ligands, 'chain',ch.id,
        #           'unrecognized elements', unrec_elements)

        if len(lig_xyz) == 0:
            if seed_ixf is not None and i_ch_lig == ligand_chids[0]:
                # use provided seed transform for 1st ligand residue
                i_xf_chosen = seed_ixf
            else:
                # use 1st transform that exists for this ligand residue's chain
                i_xf_chosen = asmb_xform_chids.index(ch.id)
        else:  # use xform w/ the single closest contact (i.e. bond) to built-up ligand so far
            min_dist = []
            for i_xf, i_ch in enumerate(asmb_xform_chids):
                if i_ch != ch.id:
                    continue
                xf = asmb_xforms[i_xf][1]
                u, r = xf[:3, :3], xf[:3, 3]
                xyz_xf = torch.einsum("ij,raj->rai", u, xyz) + r[None, None]
                dist = torch.cdist(
                    lig_xyz[lig_mask[:, 1], 1],
                    xyz_xf[mask[:, 1], 1],
                    compute_mode="donot_use_mm_for_euclid_dist",
                )
                min_dist.append((i_xf, dist.min()))
            i_xf_chosen = min(min_dist, key=lambda x: x[1])[0]

        xf = asmb_xforms[i_xf_chosen][1]
        u, r = xf[:3, :3], xf[:3, 3]
        xyz_xf = torch.einsum("ij,raj->rai", u, xyz) + r[None, None]

        lig_xyz = torch.cat([lig_xyz, xyz_xf], dim=0)
        lig_mask = torch.cat([lig_mask, mask], dim=0)
        lig_seq = torch.cat([lig_seq, seq], dim=0)
        lig_chid.extend(chid)
        lig_resi.extend(resi)
        lig_i_ch_xf.append((ch.id, i_xf_chosen))

    return lig_xyz, lig_mask, lig_seq, lig_chid, lig_resi, lig_i_ch_xf


def get_contacting_chains(
    asmb_chains: List[Chain],
    asmb_xforms: List[Tuple[str, np.ndarray]],
    lig_xyz_xf: torch.Tensor,
    lig_i_ch_xf: List[Tuple[str, int]],
    chain_xyz_dict: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> List[Tuple[str, int, int, float, str]]:
    """Gets protein or nucleic acid chains containing any heavy atom within 30A
    of query ligand. Returned chains are ordered from most to least heavy atoms
    within 5A.

    Parameters
    ----------
    asmb_chains : list
        List of cifutils.Chain objects representing the chains belonging to a
        particular assembly.
    asmb_xforms : list
        List of tuples (chain letter, transform matrix) representing coordinate
        transforms for all the chains in this assembly.
    lig_xyz_xf : torch.Tensor (N_atoms, 3)
        Atom coordinates of the query ligand (after applying a specific
        coordinate transform)
    lig_i_ch_xf : list
        List of tuples (chain letter, transform index) specifying the specific
        transform (in `asmb_xforms`) used to featurize that chain when
        query ligand was constructed. Used to exclude query ligand from the
        returned list of contacting chains.

    Returns
    -------
    contacts : list
        List of tuples (chain letter, transform index, number of contacts,
        chain type) representing chains that are near query ligand, in
        order from most contacts to least contacts (heavy atoms < 5A).
    """
    include_nucleic_acids = kwargs.get("include_nucleic_acids", False)
    close_dist = kwargs.get("close_dist_prot", 30.0)
    contact_dist = kwargs.get("contact_dist_prot", 6.5)
    min_close = kwargs.get("min_close_prot", 1)
    max_partners = kwargs.get("max_prot_partners", 10)

    chain_types = ["polypeptide(L)"]
    if include_nucleic_acids:
        chain_types.append("polydeoxyribonucleotide")
        chain_types.append("polyribonucleotide")

    contacts = []
    for ch in asmb_chains:
        if ch.type not in chain_types:
            continue
        xyz, mask, seq, chid, resi, unrec_elements = chain_to_xyz(
            ch, chain_xyz_dict=chain_xyz_dict
        )

        for i_xf, (xf_ch, xf) in enumerate(asmb_xforms):
            if xf_ch != ch.id:
                continue
            xf = xf
            u, r = xf[:3, :3], xf[:3, 3]
            xyz_xf = torch.einsum("ij,raj->rai", u, xyz) + r[None, None]
            ca_xyz = xyz_xf[:, 1][mask[:, 1]]

            dist = torch.cdist(
                lig_xyz_xf, ca_xyz, compute_mode="donot_use_mm_for_euclid_dist"
            )

            num_contacts = (dist < contact_dist).sum()
            num_close = (dist < close_dist).sum()

            if num_close >= min_close and (ch.id, i_xf) not in lig_i_ch_xf:
                contacts.append(
                    (ch.id, i_xf, int(num_contacts), float(dist.min()), ch.type)
                )
                if len(contacts) >= max_partners:
                    break

    # sort by more to fewer contacts, then lower to higher min distance
    return sorted(contacts, key=lambda x: (x[2], -x[3]), reverse=True)


def get_contacting_ligands(
    ligands: List[Tuple[str, ...]],
    chains: Dict[str, Chain],
    asmb_xforms: List[Tuple[str, np.ndarray]],
    qlig: Tuple[str, ...],
    qlig_xyz: torch.Tensor,
    qlig_chxf: List[Tuple[str, int]],
    chain_xyz_dict: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> List[Tuple[List[Tuple[str, ...]], List[Tuple[str, int]], int, float, str]]:
    """Gets partner ligands in contact with query ligand.

    Contacts are defined as any heavy atom within 5A or all heavy atoms of
    partner ligand within 30A of query ligand.

    Parameters
    ----------
    ligands : list
        List of lists of 3-tuples (chain id, res num, lig name), representing
        all the covalently bonded sets of ligand residues that make up each full
        small molecule ligand to be assessed for contacts to query ligand.
    chains : dict[str, cifutils.Chain]
        Chains in this PDB entry
    asmb_xforms : list
        List of tuples (chain letter, transform matrix) representing coordinate
        transforms for all the chains in this assembly.
    qlig : tuple (chain letter, res num, res name)
        Tuple with identifying information for the query ligand
    qlig_xyz : torch.Tensor (N_atoms, 3)
        Atom coordinates of the query ligand (after applying a specific
        coordinate transform)
    qlig_chxf: list
        List of tuples (chain letter, transform index) specifying the specific
        transform (in `asmb_xforms`) used to featurize that chain when
        query ligand was constructed. Used to exclude query ligand from the
        returned list of contacting chains.

    Returns
    -------
    contacts : list
        List of tuples (ligand_list, chain_transforms, number of contacts,
        chain type) representing ligands that make contact with query ligand,
        sorted in order from most to least contacts. `ligand_list` is a list
        of tuples, representing a specific (possibly multi-residue) ligand.
        `chain_transforms` is a list of 2-tuples (chain letter, transform
        index) representing the transforms associating that ligand with a
        unique 3D location.  It is possible for number of contacts to be 0
        because ligands are also considered partners if all their atoms are
        within 30A.
    """
    contact_dist = kwargs.get("contact_dist_lig", 5.0)
    close_dist = kwargs.get("close_dist_lig", 30.0)
    max_partners = kwargs.get("max_lig_partners", 10)
    qlig_xyz_mean = qlig_xyz.mean(dim=0, keepdim=True)

    contacts = []
    for lig in ligands:

        # if there is more than 1 transform for this ligand's first residue,
        # try to construct it using each transform
        asmb_xform_chids = [x[0] for x in asmb_xforms]
        seed_ixf_s = [
            i for i, chlet in enumerate(asmb_xform_chids) if chlet == lig[0][0]
        ]

        # edge case: `covale` implies multiresidue ligand but the residues aren't in same assembly
        if not set([res[0] for res in lig]).issubset(asmb_xform_chids):
            continue

        for seed_ixf in seed_ixf_s:
            lig_xyz, lig_mask, lig_seq, lig_chid, lig_resi, lig_chxf = get_ligand_xyz(
                chains,
                asmb_xforms,
                lig,
                seed_ixf,
                chain_xyz_dict=chain_xyz_dict,
            )

            # don't include query ligand in its original location among partners
            if lig == qlig and lig_chxf == qlig_chxf:
                continue

            if lig_xyz.numel() == 0:
                continue

            lig_xyz_valid = lig_xyz[lig_mask[:, 1], 1]

            if lig_xyz_valid.numel() == 0:
                continue

            lig_xyz_valid_mean = lig_xyz_valid.mean(dim=0, keepdim=True)
            lig_lig_mean_dist = torch.sqrt(
                ((lig_xyz_valid_mean - qlig_xyz_mean) ** 2).sum()
            ).item()

            if lig_lig_mean_dist > close_dist:
                continue

            dist = torch.cdist(
                qlig_xyz, lig_xyz_valid, compute_mode="donot_use_mm_for_euclid_dist"
            )  # (N_atoms_query, N_atoms_partner)

            num_contacts = (dist < contact_dist).sum()
            is_close = (dist < close_dist).all()

            # filter out partner ligand residues that weren't loaded (all atoms have 0 occupancy)
            lig = [res for res in lig if res[0] in [x[0] for x in lig_chxf]]

            if is_close:
                contacts.append(
                    (lig, lig_chxf, int(num_contacts), float(dist.min()), "nonpoly")
                )
                if len(contacts) >= max_partners:
                    break

    # sort by more to fewer contacts, then lower to higher min distance
    return sorted(contacts, key=lambda x: (x[2], -x[3]), reverse=True)


def deduplicate_xforms(
    xforms: List[Tuple[str, np.ndarray]]
) -> List[Tuple[str, np.ndarray]]:
    """Removes duplicated coordinate transform matrices from the list returned
    by cifutils.Parser. Not necessary in recent versions of the parser, but
    used to debug a previous version that sometimes returned duplicated
    transforms."""
    new_xforms = []
    for i_ch, xf in xforms:
        exists = False
        for i_ch2, xf2 in new_xforms:
            if i_ch == i_ch2 and np.allclose(xf, xf2):
                exists = True
                break
        if not exists:
            new_xforms.append((i_ch, xf))
    return new_xforms


def has_non_biological_bonds(
    covalents: List[Tuple[Tuple[str, ...], Tuple[str, ...]]]
) -> bool:
    """Detects non-biological bonds"""
    has_oxygen_oxygen_bond = any(
        [a1[3][0] == "O" and a2[3][0] == "O" for (a1, a2) in covalents]
    )
    has_fluorine_fluorine_bond = any(
        [a1[3][0] == "F" and a2[3][0] == "F" for (a1, a2) in covalents]
    )
    is_oxy_hydroxy = any(
        [
            a1[2] == "O"
            or a2[2] == "O"
            or a1[2] == "OH"
            or a2[2] == "OH"
            or a1[2] == "HOH"
            or a2[2] == "HOH"
            for (a1, a2) in covalents
        ]
    )
    return has_oxygen_oxygen_bond or has_fluorine_fluorine_bond or is_oxy_hydroxy


def is_clashing(
    qlig_xyz_valid: torch.Tensor,
    xyz_chains: List[torch.Tensor],
    mask_chains: List[torch.Tensor],
    chain_letters: List[str],
    asmb_xforms: List[Tuple[str, np.ndarray]],
) -> bool:
    """Detects if a ligand is within 1A of any protein in its assembly."""
    for chain_letter, xyz, mask in zip(chain_letters, xyz_chains, mask_chains):
        for transform_chain_letter, transform_matrix in asmb_xforms:
            if transform_chain_letter != chain_letter:
                continue
            transform_matrix = transform_matrix
            u, r = transform_matrix[:3, :3], transform_matrix[:3, 3]
            xyz_xf = torch.einsum("ij,raj->rai", u, xyz) + r[None, None]

            atom_xyz = xyz_xf[:, : ChemicalData().NHEAVY][
                mask[:, : ChemicalData().NHEAVY].numpy(), :
            ]
            dist = torch.cdist(
                qlig_xyz_valid, atom_xyz, compute_mode="donot_use_mm_for_euclid_dist"
            )
            if (dist < 1).any():
                return True
    return False


def get_chains_in_assembly(
    chains: Dict[str, Chain],
    asmb: Dict[str, List[Tuple[str, np.ndarray]]],
    assembly: str,
) -> List[Chain]:
    assembly_transforms = asmb[assembly]
    assembly_chain_ids = [x[0] for x in assembly_transforms]
    assembly_chains = [chains[i_ch] for i_ch in set(assembly_chain_ids)]
    return assembly_chains


def get_coordinates_from_chain_list(
    chains: List[Chain],
    chain_xyz_dict: Optional[Dict[str, Any]] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str]]:
    xyz_chains = []
    mask_chains = []
    chain_letters = []

    for chain in chains:
        if chain.type != "polypeptide(L)":
            continue
        xyz, mask, _, _, _, _ = chain_to_xyz(chain, chain_xyz_dict=chain_xyz_dict)
        xyz_chains.append(xyz)
        mask_chains.append(mask)
        chain_letters.append(chain.id)
    return xyz_chains, mask_chains, chain_letters


def strip_ligand(
    chains: Dict[str, Chain],
    asmb_xforms: Dict[str, List[Tuple[str, np.ndarray]]],
    ligand: List[Tuple[str, ...]],
    chain_xyz_dict: Optional[Dict[str, Any]] = None,
) -> Tuple:
    qlig_xyz, qlig_mask, qlig_seq, qlig_chid, qlig_resi, qlig_chxf = get_ligand_xyz(
        chains,
        asmb_xforms,
        ligand,
        chain_xyz_dict=chain_xyz_dict,
    )
    query_ligand = [res for res in ligand if res[0] in [x[0] for x in qlig_chxf]]

    qlig_xyz_valid = qlig_xyz[qlig_mask[:, 1], 1]
    return (
        query_ligand,
        qlig_xyz,
        qlig_xyz_valid,
        qlig_mask,
        qlig_seq,
        qlig_chid,
        qlig_resi,
        qlig_chxf,
    )


def query_ligand_is_bad(
    qlig_xyz: torch.Tensor,
    qlig_xyz_valid: torch.Tensor,
    xyz_chains: List[torch.Tensor],
    mask_chains: List[torch.Tensor],
    chain_letters: List[str],
    asmb_xforms: List[Tuple[str, np.ndarray]],
) -> bool:
    if qlig_xyz.numel() == 0:
        return True

    if qlig_xyz_valid.numel() == 0:
        return True

    if is_clashing(qlig_xyz_valid, xyz_chains, mask_chains, chain_letters, asmb_xforms):
        return True

    return False


def get_partners(
    query_ligand: List[Tuple[str, ...]],
    qlig_xyz_valid: torch.Tensor,
    qlig_chxf: List[Tuple[str, int]],
    chains: Dict[str, Chain],
    assembly_chains: List[Chain],
    asmb_xforms: List[Tuple[str, np.ndarray]],
    covale: List[Tuple[Tuple[str, ...], Tuple[str, ...]]],
    ligands: List[List[Tuple[str, ...]]],
    xyz_chains: List[torch.Tensor],
    mask_chains: List[torch.Tensor],
    chain_letters: List[str],
    chain_xyz_dict: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Optional[Dict[str, Any]]:
    prot_na_contacts = get_contacting_chains(
        assembly_chains,
        asmb_xforms,
        qlig_xyz_valid,
        qlig_chxf,
        chain_xyz_dict=chain_xyz_dict,
        **kwargs,
    )

    lig_contacts = get_contacting_ligands(
        ligands,
        chains,
        asmb_xforms,
        query_ligand,
        qlig_xyz_valid,
        qlig_chxf,
        chain_xyz_dict=chain_xyz_dict,
        **kwargs,
    )

    partners = sorted(
        prot_na_contacts + lig_contacts, key=lambda x: (x[2], -x[3]), reverse=True
    )
    partners = filter_partners(
        partners, chains, xyz_chains, mask_chains, chain_letters, asmb_xforms, covale
    )
    return partners


def get_partner_length(chains: Dict[str, Chain], partner: Tuple) -> int:
    if partner[-1] == "nonpoly":
        ligand_list = partner[0]
        cumulative_sum = 0
        for lig in ligand_list:
            atoms = chains[lig[0]].atoms
            cumulative_sum += sum([atom.element != 1 for atom in atoms.values()])
        return cumulative_sum
    else:
        return len(chains[partner[0]].sequence)


def get_primary_protein_chain(partners: List[Tuple]) -> Optional[str]:
    prot_contacts = [x for x in partners if x[-1] == "polypeptide(L)" and x[2] > 0]
    if len(prot_contacts) == 0:
        return None
    return prot_contacts[0][0]


def filter_partners(
    partners: List[Tuple],
    chains: Dict[str, Chain],
    xyz_chains: List[torch.Tensor],
    mask_chains: List[torch.Tensor],
    chain_letters: List[str],
    asmb_xforms: List[Tuple[str, np.ndarray]],
    covale: List[Tuple[Tuple[str, ...], Tuple[str, ...]]],
    chain_xyz_dict: Optional[Dict[str, Any]] = None,
) -> List[Tuple]:
    new_partners = []
    for p in partners:
        if p[-1] == "nonpoly":
            # remove covalent partners with non-biological bonds
            bonds = []
            for bond in covale:
                if any([bond.a[:3] == res or bond.b[:3] == res for res in p[0]]):
                    bonds.append((bond.a, bond.b))
            if len(bonds) > 0:
                if has_non_biological_bonds(bonds):
                    continue
            # remove partners with clash to protein
            plig = p[0]
            lig_xyz, lig_mask, lig_seq, lig_chid, lig_resi, lig_chxf = get_ligand_xyz(
                chains,
                asmb_xforms,
                plig,
                seed_ixf=dict(p[1])[plig[0][0]],
                chain_xyz_dict=chain_xyz_dict,
            )

            lig_xyz_valid = lig_xyz[lig_mask]
            clash = is_clashing(
                lig_xyz_valid, xyz_chains, mask_chains, chain_letters, asmb_xforms
            )
            if clash:
                continue
        new_partners.append(p)
    return new_partners


def get_ligand_atom_counts(
    ligand: List[Tuple],
    chains: Dict[str, Any],
    asmb_xfs: List[Tuple[str, np.ndarray]],
    covale: List[Tuple[Tuple[str, ...], Tuple[str, ...]]],
    lig_xf: Dict[str, int],
) -> Tuple[int, int]:
    lig_atoms, lig_bonds = get_ligand_atoms_bonds(ligand, chains, covale)
    lig_ch2xf = dict(lig_xf)

    xyz_sm, mask_sm, msa_sm, chid_sm, lig_akeys = cif_ligand_to_xyz(
        lig_atoms, asmb_xfs, lig_ch2xf
    )

    lig_atoms = xyz_sm.shape[0]
    lig_atoms_resolved = mask_sm.sum().item()
    return lig_atoms, lig_atoms_resolved


def cif_coords_match_msa_len(
    pdb_hash: str,
    protein_chain: Chain,
    pdb_dir: str = "/projects/ml/TrRosetta/PDB-2021AUG02",
    chain_xyz_dict: Optional[Dict[str, Any]] = None,
) -> bool:
    xyz_prot, _, _, _, _, _ = chain_to_xyz(protein_chain, chain_xyz_dict=chain_xyz_dict)

    a3mA = get_msa(
        pdb_dir + "/a3m/" + pdb_hash[:3] + "/" + pdb_hash + ".a3m.gz", pdb_hash
    )
    protein_length_from_cif = xyz_prot.shape[0]
    msa_length_matches_cif = a3mA["msa"].shape[1] == protein_length_from_cif
    return msa_length_matches_cif


def get_master_df() -> pd.DataFrame:
    master_df = pd.read_csv("/projects/ml/TrRosetta/PDB-2021AUG02/list_v02.csv")
    master_df["HASH"] = master_df["HASH"].apply(lambda x: f"{x:06d}")
    return master_df


def get_chid2hash() -> Dict[str, int]:
    master_df = get_master_df()
    chid2hash = dict(zip(master_df["CHAINID"].tolist(), master_df["HASH"].tolist()))
    return chid2hash
