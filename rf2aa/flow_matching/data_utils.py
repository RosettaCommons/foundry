import json
import dataclasses
import numpy as np
import collections
import logging
import string
import pickle
import os
import torch
from typing import List, Dict, Any
from rf2aa.flow_matching import rigid_utils as ru
#from torch_geometric.utils import scatter
from torch import scatter
from glob import glob

Rigid = ru.Rigid
#Protein = protein.Protein

logger = logging.getLogger(__name__)

# Global map from chain characters to integers.
ALPHANUMERIC = string.ascii_letters + string.digits + ' '
CHAIN_TO_INT = {
    chain_char: i for i, chain_char in enumerate(ALPHANUMERIC)
}
INT_TO_CHAIN = {
    i: chain_char for i, chain_char in enumerate(ALPHANUMERIC)
}

NM_TO_ANG_SCALE = 10.0
ANG_TO_NM_SCALE = 1 / NM_TO_ANG_SCALE

CHAIN_FEATS = [
    'atom_positions', 'aatype', 'atom_mask', 'residue_index', 'b_factors'
]

#NUM_TOKENS = residue_constants.restype_num
#MASK_TOKEN_INDEX = residue_constants.restypes_with_x.index('X')
#CA_IDX = residue_constants.atom_order['CA']

to_numpy = lambda x: x.detach().cpu().numpy()
"""
aatype_to_seq = lambda aatype: ''.join([
        residue_constants.restypes_with_x[x] for x in aatype])
seq_to_aatype = lambda seq: [
        residue_constants.restypes_with_x.index(x) for x in seq]
"""

class CPU_Unpickler(pickle.Unpickler):
    """Pytorch pickle loading workaround.

    https://github.com/pytorch/pytorch/issues/16797
    """
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


def remove_com_from_tensor_7(tensor_7):
    """remove_com_from_tensor_7 removes the center of mass along the L dimension from a tensor of shape (..., L, 7).
    """
    rigid = Rigid.from_tensor_7(tensor_7)
    rigid_trans = rigid.get_trans()
    COM = rigid_trans.mean(dim=[-2], keepdim=True)
    rigid_trans = rigid_trans - COM
    rigid._trans = rigid_trans
    return rigid.to_tensor_7()


def create_rigid(rots, trans):
    rots = ru.Rotation(rot_mats=rots)
    return Rigid(rots=rots, trans=trans)


def batch_align_structures(pos_1, pos_2, mask=None):
    if pos_1.shape != pos_2.shape:
        raise ValueError('pos_1 and pos_2 must have the same shape.')
    if pos_1.ndim != 3:
        raise ValueError(f'Expected inputs to have shape [B, N, 3]')

    num_batch = pos_1.shape[0]
    device = pos_1.device
    batch_indices = (
        torch.ones(*pos_1.shape[:2], device=device, dtype=torch.int64) 
        * torch.arange(num_batch, device=device)[:, None]
    )
    flat_pos_1 = pos_1.reshape(-1, 3)
    flat_pos_2 = pos_2.reshape(-1, 3)
    flat_batch_indices = batch_indices.reshape(-1)
    if mask is None:
        aligned_pos_1, aligned_pos_2, align_rots = align_structures(
            flat_pos_1, flat_batch_indices, flat_pos_2)
        aligned_pos_1 = aligned_pos_1.reshape(num_batch, -1, 3)
        aligned_pos_2 = aligned_pos_2.reshape(num_batch, -1, 3)
        return aligned_pos_1, aligned_pos_2, align_rots

    flat_mask = mask.reshape(-1).bool()
    _, _, align_rots = align_structures(
        flat_pos_1[flat_mask],
        flat_batch_indices[flat_mask],
        flat_pos_2[flat_mask]
    )
    aligned_pos_1 = torch.bmm(
        pos_1,
        align_rots
    )
    return aligned_pos_1, pos_2, align_rots


def adjust_oxygen_pos(
    atom_37: torch.Tensor, pos_is_known = None
) -> torch.Tensor:
    """
    Imputes the position of the oxygen atom on the backbone by using adjacent frame information.
    Specifically, we say that the oxygen atom is in the plane created by the Calpha and C from the
    current frame and the nitrogen of the next frame. The oxygen is then placed c_o_bond_length Angstrom
    away from the C in the current frame in the direction away from the Ca-C-N triangle.

    For cases where the next frame is not available, for example we are at the C-terminus or the
    next frame is not available in the data then we place the oxygen in the same plane as the
    N-Ca-C of the current frame and pointing in the same direction as the average of the
    Ca->C and Ca->N vectors.

    Args:
        atom_37 (torch.Tensor): (N, 37, 3) tensor of positions of the backbone atoms in atom_37 ordering
                                which is ['N', 'CA', 'C', 'CB', 'O', ...]
        pos_is_known (torch.Tensor): (N,) mask for known residues.
    """

    N = atom_37.shape[0]
    assert atom_37.shape == (N, 37, 3)

    # Get vectors to Carbonly from Carbon alpha and N of next residue. (N-1, 3)
    # Note that the (N,) ordering is from N-terminal to C-terminal.

    # Calpha to carbonyl both in the current frame.
    calpha_to_carbonyl: torch.Tensor = (atom_37[:-1, 2, :] - atom_37[:-1, 1, :]) / (
        torch.norm(atom_37[:-1, 2, :] - atom_37[:-1, 1, :], keepdim=True, dim=1) + 1e-4
    )
    # For masked positions, they are all 0 and so we add 1e-4 to avoid division by 0.
    # The positions are in Angstroms and so are on the order ~1 so 1e-4 is an insignificant change.

    # Nitrogen of the next frame to carbonyl of the current frame.
    nitrogen_to_carbonyl: torch.Tensor = (atom_37[:-1, 2, :] - atom_37[1:, 0, :]) / (
        torch.norm(atom_37[:-1, 2, :] - atom_37[1:, 0, :], keepdim=True, dim=1) + 1e-4
    )

    carbonyl_to_oxygen: torch.Tensor = calpha_to_carbonyl + nitrogen_to_carbonyl  # (N-1, 3)
    carbonyl_to_oxygen = carbonyl_to_oxygen / (
        torch.norm(carbonyl_to_oxygen, dim=1, keepdim=True) + 1e-4
    )

    atom_37[:-1, 4, :] = atom_37[:-1, 2, :] + carbonyl_to_oxygen * 1.23

    # Now we deal with frames for which there is no next frame available.

    # Calpha to carbonyl both in the current frame. (N, 3)
    calpha_to_carbonyl_term: torch.Tensor = (atom_37[:, 2, :] - atom_37[:, 1, :]) / (
        torch.norm(atom_37[:, 2, :] - atom_37[:, 1, :], keepdim=True, dim=1) + 1e-4
    )
    # Calpha to nitrogen both in the current frame. (N, 3)
    calpha_to_nitrogen_term: torch.Tensor = (atom_37[:, 0, :] - atom_37[:, 1, :]) / (
        torch.norm(atom_37[:, 0, :] - atom_37[:, 1, :], keepdim=True, dim=1) + 1e-4
    )
    carbonyl_to_oxygen_term: torch.Tensor = (
        calpha_to_carbonyl_term + calpha_to_nitrogen_term
    )  # (N, 3)
    carbonyl_to_oxygen_term = carbonyl_to_oxygen_term / (
        torch.norm(carbonyl_to_oxygen_term, dim=1, keepdim=True) + 1e-4
    )

    # Create a mask that is 1 when the next residue is not available either
    # due to this frame being the C-terminus or the next residue is not
    # known due to pos_is_known being false.

    if pos_is_known is None:
        pos_is_known = torch.ones((atom_37.shape[0],), dtype=torch.int64, device=atom_37.device)

    next_res_gone: torch.Tensor = ~pos_is_known.bool()  # (N,)
    next_res_gone = torch.cat(
        [next_res_gone, torch.ones((1,), device=pos_is_known.device).bool()], dim=0
    )  # (N+1, )
    next_res_gone = next_res_gone[1:]  # (N,)

    atom_37[next_res_gone, 4, :] = (
        atom_37[next_res_gone, 2, :]
        + carbonyl_to_oxygen_term[next_res_gone, :] * 1.23
    )

    return atom_37


def write_pkl(
        save_path: str, pkl_data: Any, create_dir: bool = False, use_torch=False):
    """Serialize data into a pickle file."""
    if create_dir:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if use_torch:
        torch.save(pkl_data, save_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(save_path, 'wb') as handle:
            pickle.dump(pkl_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pkl(read_path: str, verbose=True, use_torch=False, map_location=None):
    """Read data from a pickle file."""
    try:
        if use_torch:
            return torch.load(read_path, map_location=map_location)
        else:
            with open(read_path, 'rb') as handle:
                return pickle.load(handle)
    except Exception as e:
        try:
            with open(read_path, 'rb') as handle:
                return CPU_Unpickler(handle).load()
        except Exception as e2:
            if verbose:
                print(f'Failed to read {read_path}. First error: {e}\n Second error: {e2}')
            raise(e)

def chain_str_to_int(chain_str: str):
    chain_int = 0
    if len(chain_str) == 1:
        return CHAIN_TO_INT[chain_str]
    for i, chain_char in enumerate(chain_str):
        chain_int += CHAIN_TO_INT[chain_char] + (i * len(ALPHANUMERIC))
    return chain_int

#def parse_chain_feats(chain_feats, scale_factor=1., center=True):
    #chain_feats['bb_mask'] = chain_feats['atom_mask'][:, CA_IDX]
    #bb_pos = chain_feats['atom_positions'][:, CA_IDX]
    #if center:
        #bb_center = np.sum(bb_pos, axis=0) / (np.sum(chain_feats['bb_mask']) + 1e-4)
        #centered_pos = chain_feats['atom_positions'] - bb_center[None, None, :]
        #scaled_pos = centered_pos / scale_factor
    #else:
        #scaled_pos = chain_feats['atom_positions'] / scale_factor
    #chain_feats['atom_positions'] = scaled_pos * chain_feats['atom_mask'][..., None]
    #chain_feats['bb_positions'] = chain_feats['atom_positions'][:, CA_IDX]
    #return chain_feats

def concat_np_features(
        np_dicts: List[Dict[str, np.ndarray]], add_batch_dim: bool):
    """Performs a nested concatenation of feature dicts.

    Args:
        np_dicts: list of dicts with the same structure.
            Each dict must have the same keys and numpy arrays as the values.
        add_batch_dim: whether to add a batch dimension to each feature.

    Returns:
        A single dict with all the features concatenated.
    """
    combined_dict = collections.defaultdict(list)
    for chain_dict in np_dicts:
        for feat_name, feat_val in chain_dict.items():
            if add_batch_dim:
                feat_val = feat_val[None]
            combined_dict[feat_name].append(feat_val)
    # Concatenate each feature
    for feat_name, feat_vals in combined_dict.items():
        combined_dict[feat_name] = np.concatenate(feat_vals, axis=0)
    return combined_dict


def center_zero(pos: torch.Tensor, batch_indexes: torch.LongTensor) -> torch.Tensor:
    """
    Move the molecule center to zero for sparse position tensors.

    Args:
        pos: [N, 3] batch positions of atoms in the molecule in sparse batch format.
        batch_indexes: [N] batch index for each atom in sparse batch format.

    Returns:
        pos: [N, 3] zero-centered batch positions of atoms in the molecule in sparse batch format.
    """
    assert len(pos.shape) == 2 and pos.shape[-1] == 3, "pos must have shape [N, 3]"

    means = scatter(pos, batch_indexes, dim=0, reduce="mean")
    return pos - means[batch_indexes]


@torch.no_grad()
def align_structures(
    batch_positions: torch.Tensor,
    batch_indices: torch.Tensor,
    reference_positions: torch.Tensor,
    broadcast_reference: bool = False,
):
    """
    Align structures in a ChemGraph batch to a reference, e.g. for RMSD computation. This uses the
    sparse formulation of pytorch geometric. If the ChemGraph is composed of a single system, then
    the reference can be given as a single structure and broadcasted. Returns the structure
    coordinates shifted to the geometric center and the batch structures rotated to match the
    reference structures. Uses the Kabsch algorithm (see e.g. [kabsch_align1]_). No permutation of
    atoms is carried out.

    Args:
        batch_positions (Tensor): Batch of structures (e.g. from ChemGraph) which should be aligned
          to a reference.
        batch_indices (Tensor): Index tensor mapping each node / atom in batch to the respective
          system (e.g. batch attribute of ChemGraph batch).
        reference_positions (Tensor): Reference structure. Can either be a batch of structures or a
          single structure. In the second case, broadcasting is possible if the input batch is
          composed exclusively of this structure.
        broadcast_reference (bool, optional): If reference batch contains only a single structure,
          broadcast this structure to match the ChemGraph batch. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tensors containing the centered positions of batch
          structures rotated into the reference and the centered reference batch.

    References
    ----------
    .. [kabsch_align1] Lawrence, Bernal, Witzgall:
       A purely algebraic justification of the Kabsch-Umeyama algorithm.
       Journal of research of the National Institute of Standards and Technology, 124, 1. 2019.
    """
    # Minimize || Q @ R.T - P ||, which is the same as || Q - P @ R ||
    # batch_positions     -> P [BN x 3]
    # reference_positions -> Q [B / BN x 3]

    if batch_positions.shape[0] != reference_positions.shape[0]:
        if broadcast_reference:
            # Get number of systems in batch and broadcast reference structure.
            # This assumes, all systems in the current batch correspond to the reference system.
            # Typically always the case during evaluation.
            num_molecules = int(torch.max(batch_indices) + 1)
            reference_positions = reference_positions.repeat(num_molecules, 1)
        else:
            raise ValueError("Mismatch in batch dimensions.")

    # Center structures at origin (takes care of translation alignment)
    batch_positions = center_zero(batch_positions, batch_indices)
    reference_positions = center_zero(reference_positions, batch_indices)

    # Compute covariance matrix for optimal rotation (Q.T @ P) -> [B x 3 x 3].
    cov = scatter(
        batch_positions[:, None, :] * reference_positions[:, :, None], batch_indices, dim=0, reduce="sum"
    )

    # Perform singular value decomposition. (all [B x 3 x 3])
    u, _, v_t = torch.linalg.svd(cov)
    # Convenience transposes.
    u_t = u.transpose(1, 2)
    v = v_t.transpose(1, 2)

    # Compute rotation matrix correction for ensuring right-handed coordinate system
    # For comparison with other sources: det(AB) = det(A)*det(B) and det(A) = det(A.T)
    sign_correction = torch.sign(torch.linalg.det(torch.bmm(v, u_t)))
    # Correct transpose of U: diag(1, 1, sign_correction) @ U.T
    u_t[:, 2, :] = u_t[:, 2, :] * sign_correction[:, None]

    # Compute optimal rotation matrix (R = V @ diag(1, 1, sign_correction) @ U.T).
    rotation_matrices = torch.bmm(v, u_t)

    # Rotate batch positions P to optimal alignment with Q (P @ R)
    batch_positions_rotated = torch.bmm(
        batch_positions[:, None, :],
        rotation_matrices[batch_indices],
    ).squeeze(1)

    return batch_positions_rotated, reference_positions, rotation_matrices


#def parse_pdb_feats(
        #pdb_name: str,
        #pdb_path: str,
        #scale_factor=1.,
        ## TODO: Make the default behaviour read all chains.
        #chain_id='A',
    #):
    #"""
    #Args:
        #pdb_name: name of PDB to parse.
        #pdb_path: path to PDB file to read.
        #scale_factor: factor to scale atom positions.
        #mean_center: whether to mean center atom positions.
    #Returns:
        #Dict with CHAIN_FEATS features extracted from PDB with specified
        #preprocessing.
    #"""
    #parser = PDB.PDBParser(QUIET=True)
    #structure = parser.get_structure(pdb_name, pdb_path)
    #struct_chains = {
        #chain.id: chain
        #for chain in structure.get_chains()}

    #def _process_chain_id(x):
        #chain_prot = process_chain(struct_chains[x], x)
        #chain_dict = dataclasses.asdict(chain_prot)

        ## Process features
        #feat_dict = {x: chain_dict[x] for x in CHAIN_FEATS}
        #return parse_chain_feats(
            #feat_dict, scale_factor=scale_factor)

    #if isinstance(chain_id, str):
        #return _process_chain_id(chain_id)
    #elif isinstance(chain_id, list):
        #return {
            #x: _process_chain_id(x) for x in chain_id
        #}
    #elif chain_id is None:
        #return {
            #x: _process_chain_id(x) for x in struct_chains
        #}
    #else:
        #raise ValueError(f'Unrecognized chain list {chain_id}')


#def process_chain(chain: Chain, chain_id: str) -> Protein:
    #"""Convert a PDB chain object into a AlphaFold Protein instance.

    #Forked from alphafold.common.protein.from_pdb_string

    #WARNING: All non-standard residue types will be converted into UNK. All
        #non-standard atoms will be ignored.

    #Took out lines 94-97 which don't allow insertions in the PDB.
    #Sabdab uses insertions for the chothia numbering so we need to allow them.

    #Took out lines 110-112 since that would mess up CDR numbering.

    #Args:
        #chain: Instance of Biopython's chain class.

    #Returns:
        #Protein object with protein features.
    #"""
    #atom_positions = []
    #aatype = []
    #atom_mask = []
    #residue_index = []
    #b_factors = []
    #chain_ids = []
    #for res in chain:
        #res_shortname = residue_constants.restype_3to1.get(res.resname, 'X')
        #restype_idx = residue_constants.restype_order.get(
            #res_shortname, residue_constants.restype_num)
        #pos = np.zeros((residue_constants.atom_type_num, 3))
        #mask = np.zeros((residue_constants.atom_type_num,))
        #res_b_factors = np.zeros((residue_constants.atom_type_num,))
        #for atom in res:
            #if atom.name not in residue_constants.atom_types:
                #continue
            #pos[residue_constants.atom_order[atom.name]] = atom.coord
            #mask[residue_constants.atom_order[atom.name]] = 1.
            #res_b_factors[residue_constants.atom_order[atom.name]
                          #] = atom.bfactor
        #aatype.append(restype_idx)
        #atom_positions.append(pos)
        #atom_mask.append(mask)
        #residue_index.append(res.id[1])
        #b_factors.append(res_b_factors)
        #chain_ids.append(chain_id)

    #return Protein(
        #atom_positions=np.array(atom_positions),
        #atom_mask=np.array(atom_mask),
        #aatype=np.array(aatype),
        #residue_index=np.array(residue_index),
        #chain_index=np.array(chain_ids),
        #b_factors=np.array(b_factors))


#def extract_sequence_from_pdb(file_path):
    #parser = PDB.PDBParser(QUIET=True)
    #structure = parser.get_structure('decoy', file_path)

    ## Extract all chains
    #struct_chains = {
        #chain.id.upper(): chain
        #for chain in structure.get_chains()}
    #if len(struct_chains) > 1:
        #raise ValueError('Only compatible with monomers')

    ## Convert chain id into int
    #chain_id = list(struct_chains.keys())[0]
    #chain = struct_chains[chain_id]
    #chain_prot = parsers.process_chain(chain, chain_id)
    #chain_dict = dataclasses.asdict(chain_prot)
    #chain_dict = parse_chain_feats(chain_dict)
    #return chain_dict['aatype']


#def center_on_motif(scaffold_mask, trans_1):
    #motif_mask = 1 - scaffold_mask
    #motif_1 = trans_1 * motif_mask[..., None]
    #motif_com = torch.sum(motif_1, dim=1) / (torch.sum(motif_mask) + 1)
    #trans_1 -= motif_com[:, None, :]
    #return trans_1

#def get_synthetic_data_folder(dataset_cfg):
    #synthetic_data_folders = glob(os.path.join(os.path.dirname(dataset_cfg.synthetic_data_folder), "*"))
    #target_pid = os.getpid() if rank_zero_only.rank == 0 else os.getppid()
    #for folder in synthetic_data_folders:
        #current_date = os.path.dirname(dataset_cfg.synthetic_data_folder).split('_')[0]
        #if current_date in folder and str(target_pid) in folder:
            #return folder
    #print("Didnt find any matching synthetic data folder")
    #return None

def initialize(sequence_config):
    global NUM_TOKENS
    global MASK_TOKEN_INDEX
    logger.info(f'initializing {sequence_config.num_tokens=} {sequence_config.mask_token_index=}')
    NUM_TOKENS = sequence_config.num_tokens
    MASK_TOKEN_INDEX = sequence_config.mask_token_index

def pretty_describe_dict(d):
    mapped = describe_dict(d)
    mapped = tree.map_structure(str, mapped)
    return json.dumps(mapped, indent=4)

def describe_dict(d):
    return tree.map_structure(describe, d)

def describe(t: torch.Tensor):
    out = [f'type:{type(t)}']
    if hasattr(t, 'shape'):
        out.append(f'shape:{str(t.shape)}')
    if hasattr(t, 'dtype'):
        out.append(f'dtype:{str(t.dtype)}')
    if hasattr(t, 'device'):
        out.append(f'device:{str(t.device)}')
    return ' '.join(out)

def safe_shape(t: torch.Tensor):
    if hasattr(t, 'shape'):
        return t.shape
    return None
