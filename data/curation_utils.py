import sys, os, time, pickle, gzip
import numpy as np
import pandas as pd
import torch
from collections import Counter

script_dir = os.path.dirname(os.path.abspath(__file__)) 
sys.path.insert(0,script_dir+'/../')
sys.path.insert(0,script_dir+'/../rf2aa/')
import rf2aa.chemical as chemical
from rf2aa.chemical import aa2num, aa2long, NTOTAL, NHEAVY

aa2long_ = [[x.strip() if x is not None else None for x in y] for y in aa2long]
aa2num_ = {k.strip():v for k,v in aa2num.items()}

def get_bonded_partners(lig_tuple, bonds):
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
        if bond.a[:3]==lig_tuple:
            partners.add(bond.b[:3])
        elif bond.b[:3]==lig_tuple:
            partners.add(bond.a[:3])
        else:
            new_bonds.append(bond)

    partners = set([p for p in partners if p!=lig_tuple])

    new_partners = []
    for p in partners:
        new_partners.append(get_bonded_partners(p, new_bonds))

    for new_p in new_partners:
        partners.update(new_p)

    return partners


def get_ligands(chains, covale):
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
    lig_res_s = list(set([x[:3] for ch in chains.values() if ch.type=='nonpoly'
                                for x in ch.atoms if x[2]!='HOH']))
    bonds = []
    for i_ch,ch in chains.items():
        if ch.type=='nonpoly':
            bonds.extend(ch.bonds)
    inter_ligand_bonds = []
    prot_lig_bonds = []
    for bond in covale:
        if chains[bond.a[0]].type=='nonpoly' and chains[bond.b[0]].type=='nonpoly':
            bonds.append(bond)
            inter_ligand_bonds.append(bond)
        if sorted([chains[bond.a[0]].type, chains[bond.b[0]].type])==['nonpoly','polypeptide(L)']:
            prot_lig_bonds.append(bond)

    # make list of bonded ligands (lists of ligand residues)
    ligands = []
    lig_covale = []
    while len(lig_res_s)>0:
        res = lig_res_s[0]
        lig = get_bonded_partners(res, bonds)
        lig.add(res)
        lig = sorted(list(lig))
        lig_res_s = [res for res in lig_res_s if res not in lig]
        ligands.append(lig)
        lig_covale.append([(bond.a, bond.b) for bond in prot_lig_bonds
                           if any([bond.a[:3]==res or bond.b[:3]==res for res in lig])])

    return ligands, lig_covale

def chain_to_xyz(ch, residue_set=None):
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
    if ch.type in ['polypeptide(L)', 'polydeoxyribonucleotide','polyribonucleotide']:
        idx = [int(k[1]) for k in ch.atoms]
        i_min, i_max = np.min(idx), np.max(idx)
        L = i_max - i_min + 1
    elif ch.type == 'nonpoly':
        atoms_no_H = {k:v for k,v in ch.atoms.items() if v.element != 1} # exclude hydrogens
        L = len(atoms_no_H)

    xyz = torch.zeros(L, NTOTAL, 3)
    mask = torch.zeros(L, NTOTAL).bool()
    seq = torch.full((L,), np.nan)
    chid = ['-']*L
    resi = ['-']*L

    unrec_elements = set()

    # chain-type-specific unknown tokens from RF alphabet
    unk = {'polypeptide(L)':20, 'polydeoxyribonucleotide':26, 'polyribonucleotide':31}

    if ch.type in ['polypeptide(L)', 'polydeoxyribonucleotide','polyribonucleotide']:
        for k,v in ch.atoms.items():
            if k[2]=='HOH': continue # skip waters
            if residue_set is not None and k[:3] not in residue_set:
                continue
            i_res = int(k[1])-i_min
            aa_name = 'R'+k[2] if ch.type == 'polyribonucleotide' else k[2]
            if aa_name in aa2num_ and (aa2num_[aa_name]<=31): # standard AA/DNA/RNA
                aa = aa2num_[aa_name]
                if k[3] in aa2long_[aa]: # atom name exists in RF nomenclature
                    i_atom = aa2long_[aa].index(k[3]) # atom index
                    xyz[i_res, i_atom, :] = torch.tensor(v.xyz)
                    mask[i_res, i_atom] = v.occ
                seq[i_res] = aa
            else:
                seq[i_res] = unk[ch.type] # unknown
            chid[i_res] = k[0]
            resi[i_res] = k[1]

    elif ch.type=='nonpoly':
        for i,(k,v) in enumerate(atoms_no_H.items()):
            if k[2]=='HOH': continue # skip waters
            if residue_set is not None and k[:3] not in residue_set:
                continue
            xyz[i, 1, :] = torch.tensor(v.xyz)
            mask[i, 1] = v.occ # fractional occupancies cast to True
            if v.element not in chemical.atomnum2atomtype:
                seq[i] = aa2num_['ATM']
                unrec_elements.add(v.element)
            else:
                seq[i] = aa2num_[chemical.atomnum2atomtype[v.element]]
            chid[i] = k[0]
            resi[i] = k[1]

    return xyz, mask, seq, chid, resi, unrec_elements


def get_ligand_xyz(chains, asmb_xforms, ligand, seed_ixf=None):
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
    assert(seed_ixf is None or asmb_xforms[seed_ixf][0] == ligand[0][0]), \
        'ERROR: Seed transform index is not consistent with provided ligand'

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

        xyz, mask, seq, chid, resi, unrec_elements = chain_to_xyz(ch, residue_set=ligand)
        if not mask[:,1].any():
            continue # all CA's/ligand atoms missing
        # if len(unrec_elements)>0:
        #     print('pdbid', pdbid, 'ligands',ligands, 'chain',ch.id,
        #           'unrecognized elements', unrec_elements)

        if len(lig_xyz) == 0:
            if seed_ixf is not None and i_ch_lig==ligand_chids[0]: 
                # use provided seed transform for 1st ligand residue
                i_xf_chosen = seed_ixf
            else:
                # use 1st transform that exists for this ligand residue's chain
                i_xf_chosen = asmb_xform_chids.index(ch.id)
        else: # use xform w/ the single closest contact (i.e. bond) to built-up ligand so far
            min_dist = []
            for i_xf, i_ch in enumerate(asmb_xform_chids):
                if i_ch != ch.id: continue
                xf = torch.tensor(asmb_xforms[i_xf][1]).float()
                u,r = xf[:3,:3], xf[:3,3]
                xyz_xf = torch.einsum('ij,raj->rai', u, xyz) + r[None,None]
                dist = torch.cdist(lig_xyz[lig_mask[:,1],1], xyz_xf[mask[:,1],1],
                                   compute_mode='donot_use_mm_for_euclid_dist')
                min_dist.append((i_xf,dist.min()))
            i_xf_chosen = min(min_dist, key=lambda x: x[1])[0]

        xf = torch.tensor(asmb_xforms[i_xf_chosen][1]).float()
        u,r = xf[:3,:3], xf[:3,3]
        xyz_xf = torch.einsum('ij,raj->rai', u, xyz) + r[None,None]

        lig_xyz = torch.cat([lig_xyz, xyz_xf], dim=0)
        lig_mask = torch.cat([lig_mask, mask], dim=0)
        lig_seq = torch.cat([lig_seq, seq], dim=0)
        lig_chid.extend(chid)
        lig_resi.extend(resi)
        lig_i_ch_xf.append((ch.id, i_xf_chosen))

    return lig_xyz, lig_mask, lig_seq, lig_chid, lig_resi, lig_i_ch_xf

def get_contacting_chains(asmb_chains, asmb_xforms, lig_xyz_xf, lig_i_ch_xf):
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
    contacts = []
    for ch in asmb_chains:
        if ch.type not in ['polypeptide(L)', 'polydeoxyribonucleotide',
                           'polyribonucleotide']:
            continue
        xyz, mask, seq, chid, resi, unrec_elements = chain_to_xyz(ch)

        for i_xf, (xf_ch, xf) in enumerate(asmb_xforms):
            if xf_ch != ch.id: continue
            xf = torch.tensor(xf).float()
            u,r = xf[:3,:3], xf[:3,3]
            xyz_xf = torch.einsum('ij,raj->rai', u, xyz) + r[None,None]

            atom_xyz = xyz_xf[:,:NHEAVY][mask[:,:NHEAVY].numpy(),:]
            dist = torch.cdist(lig_xyz_xf, atom_xyz, compute_mode='donot_use_mm_for_euclid_dist')

            ca_mask = torch.zeros_like(mask[:,:NHEAVY]).bool()
            ca_mask[:,1] = True
            ca_mask = ca_mask[mask[:,:NHEAVY]]

            num_close = (dist[:,ca_mask]<30).any(dim=0).sum() # num C-alphas within 30A
            num_contacts = (dist<5).sum()

            if (num_close > 5) and ((ch.id, i_xf) not in lig_i_ch_xf):
                contacts.append((ch.id, i_xf, int(num_contacts), float(dist.min()), ch.type))
                
    # sort by more to fewer contacts, then lower to higher min distance
    return sorted(contacts, key=lambda x: (x[2], -x[3]), reverse=True)

def get_contacting_ligands(ligands, chains, asmb_xforms, qlig, qlig_xyz, qlig_chxf):
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
    contacts = []
    for lig in ligands:

        # if there is more than 1 transform for this ligand's first residue,
        # try to construct it using each transform
        asmb_xform_chids = [x[0] for x in asmb_xforms]
        seed_ixf_s = [i for i,chlet in enumerate(asmb_xform_chids) if chlet==lig[0][0]]

        # edge case: `covale` implies multiresidue ligand but the residues aren't in same assembly
        if not set([res[0] for res in lig]).issubset(asmb_xform_chids):
            continue

        for seed_ixf in seed_ixf_s:
            lig_xyz, lig_mask, lig_seq, lig_chid, lig_resi, lig_chxf = \
                get_ligand_xyz(chains, asmb_xforms, lig, seed_ixf)

            # don't include query ligand in its original location among partners
            if lig == qlig and lig_chxf == qlig_chxf:
                continue

            if lig_xyz.numel()==0:
                continue

            lig_xyz_valid = lig_xyz[lig_mask[:,1],1]

            if lig_xyz_valid.numel()==0:
                continue

            dist = torch.cdist(qlig_xyz, lig_xyz_valid,
                               compute_mode='donot_use_mm_for_euclid_dist') # (N_atoms_query, N_atoms_partner)

            num_contacts = (dist<5).sum()
            mindist_to_partner, _ = dist.min(dim=0) # (N_atoms_partner,)

            # filter out partner ligand residues that weren't loaded (all atoms have 0 occupancy)
            lig = [res for res in lig if res[0] in [x[0] for x in lig_chxf]]

            if (num_contacts > 0) or (mindist_to_partner<30).all():
                contacts.append((lig, lig_chxf, int(num_contacts), float(dist.min()), 'nonpoly'))

    # sort by more to fewer contacts, then lower to higher min distance
    return sorted(contacts, key=lambda x: (x[2], -x[3]), reverse=True)


def deduplicate_xforms(xforms):
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


