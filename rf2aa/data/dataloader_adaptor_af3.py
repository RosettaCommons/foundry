import os
import logging
import math

import numpy as np
import torch
import torch.nn.functional as F
from icecream import ic

from rf2aa.kinematics import xyz_to_t2d
from rf2aa.symmetry import symm_subunit_matrix, find_symm_subs
from rf2aa.util import  is_atom, \
    Ls_from_same_chain_2d, xyz_t_to_frame_xyz, get_prot_sm_mask
from rf2aa.chemical import ChemicalData as ChemData
from rf2aa.flow_matching import data_transforms
from rf2aa.debug import pretty_describe_dict
from rf2aa.util import rigid_from_3_points
from rf2aa.flow_matching.rigid_utils import rot_vec_mul
from rf2aa.set_seed import seed_all
from rf2aa.util import writepdb
from rf2aa import pymol_tools
from rf2aa.pymol import cmd


logger = logging.getLogger(__name__)


def within_group_unique_ids(group_ids, element_ids):
    # Initialize a dictionary to store unique element mappings for each group
    unique_mappings = {}
    unique_id_counter = 0
    
    # Initialize a list to store the resulting within-group unique ids
    within_group_unique = []
    
    # Iterate over the group ids and element ids simultaneously
    for group_id, element_id in zip(group_ids, element_ids):
        # Check if the current group_id is already in the unique_mappings dictionary
        if group_id in unique_mappings:
            # If the element_id is already mapped to a unique id within this group, use it
            if element_id in unique_mappings[group_id]:
                within_group_unique.append(unique_mappings[group_id][element_id])
            # If the element_id is not yet mapped to a unique id within this group, assign a new unique id
            else:
                unique_mappings[group_id][element_id] = unique_id_counter
                within_group_unique.append(unique_id_counter)
                unique_id_counter += 1
        # If the current group_id is not yet in the unique_mappings dictionary, add it
        else:
            unique_mappings[group_id] = {element_id: unique_id_counter}
            within_group_unique.append(unique_id_counter)
            unique_id_counter += 1
    
    # Convert the resulting list to a PyTorch tensor
    within_group_unique_tensor = torch.tensor(within_group_unique)
    
    return within_group_unique_tensor

def integer_tokenize(iterable):
    # Create a dictionary mapping unique elements to integers
    unique_elements = list(set(iterable))
    mapping = {element: i for i, element in enumerate(unique_elements)}
    
    # Convert iterable to integer tensor
    int_tensor = torch.tensor([mapping[element] for element in iterable])
    
    return int_tensor

af3_num2aa = [
        # Amino acids:
        'ALA','ARG','ASN','ASP','CYS',
        'GLN','GLU','GLY','HIS','ILE',
        'LEU','LYS','MET','PHE','PRO',
        'SER','THR','TRP','TYR','VAL',
        'UNK', # 20 + 1
        # DNA
        'DA','DC','DG','DT', 'DUNK',
        # RNA
        'RA','RC','RG',' RU', 'RUNK',
        # GAP
        'GAP',
]

af3_aa2num = {x:i for i,x in enumerate(af3_num2aa)}

aa_coarse_from_fine = {
    'ALA':'ALA',
    'ARG':'ARG',
    'ASN':'ASN',
    'ASP':'ASP',
    'CYS':'CYS',
    'GLN':'GLN',
    'GLU':'GLU',
    'GLY':'GLY',
    'HIS':'HIS',
    'ILE':'ILE',
    'LEU':'LEU',
    'LYS':'LYS',
    'MET':'MET',
    'PHE':'PHE',
    'PRO':'PRO',
    'SER':'SER',
    'THR':'THR',
    'TRP':'TRP',
    'TYR':'TYR',
    'VAL':'VAL',
    'UNK':'UNK',
    # 'MAS':'MAS',
    ' DA':'DA',
    ' DC':'DC',
    ' DG':'DG',
    ' DT':'DT',
    ' DX':'DUNK',
    ' RA':'RA',
    ' RC':'RC',
    ' RG':'RG',
    ' RU':'RU',
    ' RX':'RUNK',
    # 'HIS_D':'UNK',
    'Al':'UNK',
    'As':'UNK',
    'Au':'UNK',
    'B':'UNK',
    'Be':'UNK',
    'Br':'UNK',
    'C':'UNK',
    'Ca':'UNK',
    'Cl':'UNK',
    'Co':'UNK',
    'Cr':'UNK',
    'Cu':'UNK',
    'F':'UNK',
    'Fe':'UNK',
    'Hg':'UNK',
    'I':'UNK',
    'Ir':'UNK',
    'K':'UNK',
    'Li':'UNK',
    'Mg':'UNK',
    'Mn':'UNK',
    'Mo':'UNK',
    'N':'UNK',
    'Ni':'UNK',
    'O':'UNK',
    'Os':'UNK',
    'P':'UNK',
    'Pb':'UNK',
    'Pd':'UNK',
    'Pr':'UNK',
    'Pt':'UNK',
    'Re':'UNK',
    'Rh':'UNK',
    'Ru':'UNK',
    'S':'UNK',
    'Sb':'UNK',
    'Se':'UNK',
    'Si':'UNK',
    'Sn':'UNK',
    'Tb':'UNK',
    'Te':'UNK',
    'U':'UNK',
    'W':'UNK',
    'V':'UNK',
    'Y':'UNK',
    'Zn':'UNK',
    'ATM':'UNK'
}
from enum import Enum

class TokenType(Enum):
    PROTEIN = 1
    DNA = 2
    RNA = 3
    LIGAND = 4

aa_restype_from_fine = {
    'ALA': TokenType.PROTEIN,
    'ARG': TokenType.PROTEIN,
    'ASN': TokenType.PROTEIN,
    'ASP': TokenType.PROTEIN,
    'CYS': TokenType.PROTEIN,
    'GLN': TokenType.PROTEIN,
    'GLU': TokenType.PROTEIN,
    'GLY': TokenType.PROTEIN,
    'HIS': TokenType.PROTEIN,
    'ILE': TokenType.PROTEIN,
    'LEU': TokenType.PROTEIN,
    'LYS': TokenType.PROTEIN,
    'MET': TokenType.PROTEIN,
    'PHE': TokenType.PROTEIN,
    'PRO': TokenType.PROTEIN,
    'SER': TokenType.PROTEIN,
    'THR': TokenType.PROTEIN,
    'TRP': TokenType.PROTEIN,
    'TYR': TokenType.PROTEIN,
    'VAL': TokenType.PROTEIN,
    'UNK': TokenType.PROTEIN,
    # 'MAS':'MAS',
    ' DA': TokenType.DNA,
    ' DC': TokenType.DNA,
    ' DG': TokenType.DNA,
    ' DT': TokenType.DNA,
    ' DX': TokenType.DNA,
    ' RA': TokenType.RNA,
    ' RC': TokenType.RNA,
    ' RG': TokenType.RNA,
    ' RU': TokenType.RNA,
    ' RX': TokenType.RNA,
    # 'HIS_D':'UNK',
    'Al':  TokenType.LIGAND,
    'As':  TokenType.LIGAND,
    'Au':  TokenType.LIGAND,
    'B':   TokenType.LIGAND,
    'Be':  TokenType.LIGAND,
    'Br':  TokenType.LIGAND,
    'C':   TokenType.LIGAND,
    'Ca':  TokenType.LIGAND,
    'Cl':  TokenType.LIGAND,
    'Co':  TokenType.LIGAND,
    'Cr':  TokenType.LIGAND,
    'Cu':  TokenType.LIGAND,
    'F':   TokenType.LIGAND,
    'Fe':  TokenType.LIGAND,
    'Hg':  TokenType.LIGAND,
    'I':   TokenType.LIGAND,
    'Ir':  TokenType.LIGAND,
    'K':   TokenType.LIGAND,
    'Li':  TokenType.LIGAND,
    'Mg':  TokenType.LIGAND,
    'Mn':  TokenType.LIGAND,
    'Mo':  TokenType.LIGAND,
    'N':   TokenType.LIGAND,
    'Ni':  TokenType.LIGAND,
    'O':   TokenType.LIGAND,
    'Os':  TokenType.LIGAND,
    'P':   TokenType.LIGAND,
    'Pb':  TokenType.LIGAND,
    'Pd':  TokenType.LIGAND,
    'Pr':  TokenType.LIGAND,
    'Pt':  TokenType.LIGAND,
    'Re':  TokenType.LIGAND,
    'Rh':  TokenType.LIGAND,
    'Ru':  TokenType.LIGAND,
    'S':   TokenType.LIGAND,
    'Sb':  TokenType.LIGAND,
    'Se':  TokenType.LIGAND,
    'Si':  TokenType.LIGAND,
    'Sn':  TokenType.LIGAND,
    'Tb':  TokenType.LIGAND,
    'Te':  TokenType.LIGAND,
    'U':   TokenType.LIGAND,
    'W':   TokenType.LIGAND,
    'V':   TokenType.LIGAND,
    'Y':   TokenType.LIGAND,
    'Zn':  TokenType.LIGAND,
    'ATM': TokenType.LIGAND,
}

element_codes = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',  # Atomic numbers 1-10
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',  # Atomic numbers 11-20
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',  # Atomic numbers 21-30
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',  # Atomic numbers 31-40
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',  # Atomic numbers 41-50
    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',  # Atomic numbers 51-60
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',  # Atomic numbers 61-70
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',  # Atomic numbers 71-80
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',  # Atomic numbers 81-90
    'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',  # Atomic numbers 91-100
    'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',  # Atomic numbers 101-110
    'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og',  # Atomic numbers 111-118
    'Uue', 'Ubn', 'Ubu', 'Ubb', 'Ubt', 'Ubq', 'Ubp', 'Ubh',  # Atomic numbers 119-126
    'the_element', 'of_surprise' # Non-existant elements 127-128
]


element_code_to_num =  {x:i for i,x in enumerate(element_codes)}


def discretize_distance_matrix(distance_matrix, num_bins=38, min_distance=3.25, max_distance=50.75):
    # Calculate the bin width
    bin_width = (max_distance - min_distance) / num_bins
    bins = torch.arange(num_bins, device=distance_matrix.device) * bin_width + min_distance

    # Discretize distances into bins (bucketize automatically places out-of-range values in the last bin)
    binned_distances = torch.bucketize(distance_matrix, bins)
    
    return binned_distances

def torch_vectorize(pyfunc):
    def f(*args, **kwargs):
        out_np = np.vectorize(pyfunc)(*args, **kwargs)
        return torch.tensor(out_np)
    return f

def prepare_input_af3(inputs, D, s_trans, sigma_data, random_augmentation, only_ca, device="cpu",):
    logger.debug('prepare_input_af3 input:\n' + pretty_describe_dict(inputs))
    #(
        #seq, msa, msa_masked, msa_full, mask_msa, true_crds, mask_crds, idx_pdb, 
        #xyz_t, t1d, mask_t, xyz_prev, mask_prev, same_chain, unclamp, negative, 
        #atom_frames, bond_feats, dist_matrix, chirals, ch_label, symmgp, task, item
    #) = inputs
    ## transfer inputs to device
    #B, _, N, I = msa.shape
    ##logger.debug('\n\n\n\n\n')
    #logger.debug('prepare_input_af3 input:\n' + pretty_describe_dict(dict(
        #seq=seq,
        #msa=msa,
        #msa_masked=msa_masked,
        #msa_full=msa_full,
        #mask_msa=mask_msa,
        #true_crds=true_crds,
        #mask_crds=mask_crds,
        #idx_pdb=idx_pdb, 
        #xyz_t=xyz_t,
        #t1d=t1d,
        #mask_t=mask_t,
        #xyz_prev=xyz_prev,
        #mask_prev=mask_prev, 
        #same_chain=same_chain,
        #unclamp=unclamp,
        #negative=negative,
        #atom_frames=atom_frames,
        #bond_feats=bond_feats,
        #dist_matrix=dist_matrix,
        #chirals=chirals,
        #ch_label=ch_label,
        #symmgp=symmgp,
        #task=task,
        #item=item,
    #)))
    NUM_TEMPLATE_DISTOGRAM_BINS = 38
    # Strip batch dimension
    
    msa = inputs["msa_extra"][0,0,..., :ChemData().NAATOKENS].argmax(dim=-1)
    idx_pdb = inputs["idx"][0]
    ch_label = inputs["ch_label"][0]
    true_crds = inputs["xyz"][0,0]
    seq = inputs["seq"][0,0]
    xyz_t = inputs["xyz_t"][0]
    mask_t = inputs["mask_t"][0]
    mask_crds = inputs["mask"][0,0]
    bond_feats = inputs["bond_feats"][0]
    t1d = inputs["f1d_t"]

    N_token = seq.shape[0]

    logger.debug(f'{ch_label[:5]}')
    aa = [ChemData().num2aa[num] for num in seq]
    aa1 = [ChemData().to1letter.get(k, k) for k in aa]
    # aa_af3 = [aa_coarse_from_fine[s] for s in aa]


    # Converts a ChemData() sequence token to an af3 token
    @torch_vectorize
    def af3num_from_num(num):
        chemdata_code = ChemData().num2aa[num]
        coarse = aa_coarse_from_fine[chemdata_code]
        return af3_aa2num[coarse]
    
    @np.vectorize
    def get_token_type(num):
        code3 = ChemData().num2aa[num]
        return aa_restype_from_fine[code3]

    f = {}

    ### Residue level ###
    f['residue_index'] = idx_pdb
    f['token_index'] = torch.arange(N_token)
    f['asym_id'] = ch_label
    # Hacked:
    f['entity_id'] = torch.zeros(N_token)
    f['sym_id'] = within_group_unique_ids(f['entity_id'], f['asym_id'])
    # f['restype'] = F.one_hot(torch.tensor([af3_aa2num[aa_af3]]), len(af3_num2aa))
    f['restype'] = F.one_hot(af3num_from_num(seq), len(af3_num2aa))
    # token_type = torch.tensor([aa_restype_from_fine[aa]])
    # token_type = np.vectorize(aa_restype_from_fine.__getitem__)(aa)
    token_type = get_token_type(seq)
    f['is_protein'] = torch.tensor(token_type == TokenType.PROTEIN)
    f['is_rna'] = torch.tensor(token_type == TokenType.RNA)
    f['is_dna'] = torch.tensor(token_type == TokenType.DNA)
    f['is_ligand'] = torch.tensor(token_type == TokenType.LIGAND)


    ### Atom level ###
    allatom_mask = ChemData().heavyatom_mask.to(device, non_blocking=True)

    # remove symmetry dimension
    # if len(true_crds.shape) == 4:
    #     true_crds = true_crds[0:1]
    #     mask_crds = mask_crds[0:1]
    # true_crds = true_crds[0]
    
    # want to unroll the coordinate tensors to get the full coordinates in (atoms, 3)
    is_real_atom = allatom_mask[seq].bool()

    if only_ca:
        is_real_atom[:] = False
        is_real_atom[:, 1] = True

    tok_idx = is_real_atom.nonzero()[:,0]
    within_tok_idx = is_real_atom.nonzero()[:,1]
    N_atom = len(tok_idx)
    f['tok_idx'] = tok_idx
    # atom_mask = mask_crds[is_real_atom]
    # t = interpolant.sample_t(D)
    f['ref_pos'] = inputs["ref_pos_atom36"][0][is_real_atom]
    f['ref_mask'] = inputs["ref_mask"][0][is_real_atom]

    element = [ChemData().aa2elt[seq[tok]][within_tok] for tok, within_tok in zip(tok_idx, within_tok_idx)]
    f['ref_element'] = F.one_hot(torch.tensor([element_code_to_num[e] if e in element_code_to_num else len(element_codes) - 1 for e in element]), len(element_codes))

    # Hacked:
    f['ref_charge'] = torch.zeros((N_atom))
    f['ref_atom_name_chars'] = F.one_hot(inputs["ref_atom_name_chars"][0][is_real_atom].long(), num_classes=64)
    f['ref_space_uid'] = integer_tokenize(list(zip(f['asym_id'], f['residue_index'])))[tok_idx]

    ### MSA ###
    f['msa'] = F.one_hot(af3num_from_num(msa), len(af3_num2aa))

    # Hacked
    N_msa = msa.shape[0]
    f['has_deletion'] = torch.zeros((N_msa, N_token))
    f['deletion_value'] = torch.zeros((N_msa, N_token))
    f['profile'] = torch.zeros((N_token, 32))
    f['deletion_mean'] = torch.zeros((N_token))

    ### Templates ###
    N_templ = xyz_t.shape[0]
    # Hacked:
    template_seq = t1d[0].argmax(dim=-1) # [T, I]
    assert (template_seq < ChemData().NPROTAAS - 1).all() # only 20 AA + 1 UNK (No mask)

    f['template_restype'] = F.one_hot(af3num_from_num(template_seq), len(af3_num2aa))

    template_is_protein = torch.tensor(get_token_type(template_seq) == TokenType.PROTEIN)
    template_is_gly = template_seq == ChemData().aa2num['GLY']
    template_atom_name = np.where(template_is_gly, ' CA ', ' CB ')
    template_protein_beta_idx = torch_vectorize(lambda token, atom_name: ChemData().aa2long[token].index(atom_name))(
        template_seq[template_is_protein],
        template_atom_name[template_is_protein]
    )
    template_beta_idx = torch.full((N_templ, N_token), 0)
    template_beta_idx[template_is_protein] = template_protein_beta_idx
    template_beta_exists = torch.gather(mask_t, dim=2, index=template_beta_idx[..., None]).squeeze(-1)
    f['template_pseudo_beta_mask'] = template_beta_exists * template_is_protein # .reshape?
    f['template_backbone_frame_mask'] = mask_t[:, :, torch.tensor([0,1,2])].all(dim=-1)
    # Reshape index_tensor to match the dimensions of xyz_t except for the last dimension
    index_tensor_expanded = template_beta_idx[...,None,None].expand(-1, -1, -1, 3)
    template_pseudo_beta = torch.gather(xyz_t, dim=2, index=index_tensor_expanded).squeeze(-2) #.squeeze(-1)
    template_pseudo_beta_distogram = torch.cdist(template_pseudo_beta, template_pseudo_beta)
    template_pseudo_beta_distogram *= f['template_pseudo_beta_mask'].unsqueeze(-1) * f['template_pseudo_beta_mask'].unsqueeze(-2)
    f['template_distogram'] = F.one_hot(discretize_distance_matrix(
        template_pseudo_beta_distogram,
        num_bins=NUM_TEMPLATE_DISTOGRAM_BINS,
    ), num_classes=NUM_TEMPLATE_DISTOGRAM_BINS + 1) # +1 for out-of-range values

    CA_IDX = 1
    template_ca = xyz_t[..., CA_IDX, :]
    template_ca_disp = template_ca.unsqueeze(-2) - template_ca.unsqueeze(-3)
    
    # add epsilon to avoid division by zero
    eps = 1e-4
    template_ca_disp_unit = template_ca_disp / (torch.linalg.norm(template_ca_disp, dim=-1, keepdim=True) + eps)
    template_R, _ = rigid_from_3_points(
        xyz_t[:,:,0],
        xyz_t[:,:,1],
        xyz_t[:,:,2],
    )

    has_ca = mask_t[..., CA_IDX]
    both_have_ca = has_ca[..., None, :] * has_ca[..., None]
    template_unit_vector = rot_vec_mul(template_R[:,:, None], template_ca_disp_unit)
    template_unit_vector[both_have_ca] = 0
    assert template_unit_vector.isnan().sum() == 0
    f['template_unit_vector'] = template_unit_vector
    
    has_ligand_2d = (f['is_ligand'].unsqueeze(-2) + f['is_ligand'].unsqueeze(-1)).bool()
    # is_ligand_ligand = f['is_ligand'].unsqueeze(-2) * f['is_ligand'].unsqueeze(-1)
    
    # Hacked (as covalent bonds are not represented in bond_feats and 2.4A filter not applied)
    f['token_bonds'] = has_ligand_2d * (bond_feats > 0)

    X_gt_L = true_crds[is_real_atom]
    atom_mask = mask_crds[is_real_atom]
    t = sigma_data * torch.exp(-1.2 + 1.5 * torch.normal(mean=0, std=1, size=(D,)))
    X_gt_L = centre(X_gt_L, atom_mask)
    X_gt_L = X_gt_L.tile(D,1,1)

    if random_augmentation:
        X_gt_L = get_random_augmentation(X_gt_L, s_trans=s_trans)

    _, L, _ = X_gt_L.shape
    t_tiled = t[:, None, None].tile(1, L, 3)
    noise = torch.normal(mean=0, std=t_tiled)
    X_noisy_L = X_gt_L + noise

    return (
        # network input
        dict(
            X_noisy_L=X_noisy_L,
            t=t,
            f=f,
        ),
        # loss input (trues)
        dict(
            X_gt_L=X_gt_L,
            crd_mask_I = is_real_atom,
            seq=seq,
            bond_feats=bond_feats,
        )
    )

def centre(X_L, X_exists_L):
    X_L = X_L.clone()
    X_L[X_exists_L] = X_L[X_exists_L] - torch.mean(X_L[X_exists_L], dim=-2, keepdim=True)
    X_L[~X_exists_L] = 0.0
    return X_L

def get_random_augmentation(X_L, s_trans):
    '''
    Inputs: 
        X_L [D, L, 3]: Batched atom coordinates
        s_trans (float): standard deviation of a global translation to be applied for each
            element in the batch
    '''
    D, L, _ = X_L.shape
    R = uniform_random_rotation((D,)).to(X_L.device)
    noise = s_trans * torch.normal(mean=0, std=1, size=(D,1,3)).to(X_L.device)
    return rot_vec_mul(R[:,None], X_L) + noise

def centre_random_augmentation(X_L, X_exists_L, s_trans):
    X_L = centre(X_L, X_exists_L)
    return get_random_augmentation(X_L, s_trans)

def uniform_random_rotation(size):
    # Sample random angles for rotations around X, Y, and Z axes
    theta_x = torch.rand(size) * 2 * math.pi
    theta_y = torch.rand(size) * 2 * math.pi
    theta_z = torch.rand(size) * 2 * math.pi
    
    # Calculate the cosines and sines of the angles
    cos_x = torch.cos(theta_x)
    sin_x = torch.sin(theta_x)
    cos_y = torch.cos(theta_y)
    sin_y = torch.sin(theta_y)
    cos_z = torch.cos(theta_z)
    sin_z = torch.sin(theta_z)
    
    # Create the rotation matrices around X, Y, and Z axes
    rotation_x = torch.stack([torch.tensor([[1, 0, 0],
                                             [0, c, -s],
                                             [0, s, c]]) for c, s in zip(cos_x, sin_x)])
    
    rotation_y = torch.stack([torch.tensor([[c, 0, s],
                                             [0, 1, 0],
                                             [-s, 0, c]]) for c, s in zip(cos_y, sin_y)])
    
    rotation_z = torch.stack([torch.tensor([[c, -s, 0],
                                             [s, c, 0],
                                             [0, 0, 1]]) for c, s in zip(cos_z, sin_z)])
    
    # Combine the rotation matrices
    rotation_matrix = torch.matmul(rotation_z, torch.matmul(rotation_y, rotation_x))
    
    return rotation_matrix


def get_default_noise_schedule(t_init=0):
    T= 200
    t_norm = torch.arange(t_init, 1 + 1/T, 1/T)
    s_max = 160
    s_min = 4e-4
    p = 7
    sigma_data = 16
    return sigma_data * (s_max**(1/p) + t_norm * (s_min ** (1/p) - s_max ** (1/p))) ** p


def get_starting_noise_level(t_init=0):
    noise_schedule = get_default_noise_schedule(t_init)
    return noise_schedule[0]