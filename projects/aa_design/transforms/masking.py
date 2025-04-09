import ast
import logging
import random
import warnings

import hydra
import networkx as nx
import numpy as np
from biotite.structure import AtomArray
from datahub.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_is_instance,
)
from datahub.transforms.atom_array import add_global_token_id_annotation
from datahub.transforms.base import Transform
from datahub.utils.token import (
    get_token_count,
    get_token_starts,
    spread_token_wise,
)

from modelhub.common import exists
from projects.aa_design.transforms.masks import InvalidMaskException, Mask

nx.from_numpy_matrix = nx.from_numpy_array
logger = logging.getLogger(__name__)
NHEAVYPROT = 14


class CreateMasks(Transform):
    """
    Masks generator applied on the atom array.

    See `masks.py` for RandomMask

    Args:
      train_masks: List[RandomMask]
      token_encoding: encoding of tokens, see datahub encoding_definition
      seed (int): random seed, for controling the masking results

    Return:
      atom_array with four more annotations on the atom level: is_motif, can_be_gp.
    """
    requires_previous_transforms = [
        'FlagAndReassignCovalentModifications', 
        'SubsampleToTypes',
    ] # We use is_protein in the PPI mask

    def check_input(self, data: dict):
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["pn_unit_id", "pn_unit_iid"])

    def __init__(self, *,
        train_masks: dict[Mask],
        sequence_encoding,
        seed=None
    ):
        if exists(train_masks):
            train_masks = hydra.utils.instantiate(train_masks, _recursive_=True)
        self.train_masks = train_masks
        self.sequence_encoding = sequence_encoding
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def inference_forward(self, atom_array):

        for annotation in Mask.required_annotations:
            tmp = atom_array.get_annotation(annotation).copy()
            atom_array.get_annotation(annotation).dtype = bool
            if isinstance(tmp[0], (str, np.str_)):
                tmp = np.array([ast.literal_eval(x) for x in tmp], dtype=bool)
            atom_array.set_annotation(annotation, tmp)
        # else:
            # warnings.warn("is_motif_atom not provided in atom array")
            # print("Warning: is_motif_atom not found in atom array. Setting all atoms to not be motif.")
            # atom_array = Mask.set_default_motif_annotations(atom_array)

        return atom_array

    def sample_masks(self, data):
        '''
        Sample a mask using a valid mask from the training masks.
        '''

        valid_masks = [
            mask for mask in self.train_masks.values() if mask.is_valid_for_example(data)
        ]

        if len(valid_masks) == 0:
            raise InvalidMaskException()
        
        p_mask = np.array([mask.frequency for mask in valid_masks])
        p_mask /= p_mask.sum()
        i_mask = np.random.choice(np.arange(len(p_mask)), p=p_mask)
        mask = valid_masks[i_mask]
        
        while valid_masks:
            try:
                i_mask = np.random.choice(np.arange(len(valid_masks)), p=p_mask)
                mask = valid_masks[i_mask]
                atom_array = mask.sample(data)
                Mask.check_has_required_annotations(atom_array)
                return atom_array, type(mask).__name__
            
            except Exception as e:
                print(e)
                raise e

                # Fallback to other valid masks...
                valid_masks.pop(i_mask)
                p_mask = np.delete(p_mask, i_mask)
                if len(valid_masks) == 0:  # ... Until exhausted
                    raise InvalidMaskException("All valid masks failed during sampling.")
                p_mask = p_mask / p_mask.sum()
    
    def forward(self, data: dict) -> dict:
        '''
        Creates conditional atom array annotations via:
            - input atom array during inference (See input_parsing.py)
            - sampling masks during training (See masks.py)

        NB: Convention is that if it has token in the name it'll be the same for every atom in the token
        and if it's got atom in the name it might not be.
        '''
        # For now is_motif and is_masked_token are boolean nots of each other
        # You could imagine not all of the motif need be fully unmasked (e.g. tip atoms)

        atom_array_input = data['atom_array']
        is_inference = data['is_inference']

        if is_inference:
            atom_array_input = self.inference_forward(atom_array_input)
        else:
            atom_array_input, sampled_mask_name = self.sample_masks(data)
            data['sampled_mask_name'] = sampled_mask_name

        atom_array = post_process_conditional_atom_array(atom_array_input, is_inference)

        # For unindexed scaffolding, we must provide an unindexing pair mask to ensure original positions aren't leaked:
        if 'feats' not in data:
            data['feats'] = {}
        data['feats']['unindexing_pair_mask'] = create_unindexing_pair_mask(atom_array)
        data["atom_array"] = atom_array
        return data

def post_process_conditional_atom_array(atom_array, is_inference):
    '''
    Default-wrapup of atom array after annotating with mask attributes
    '''
    # Check required annotations
    Mask.check_has_required_annotations(atom_array)

    # Expand unindexed motifs if necessary
    atom_array = expand_unindexed_motifs(
        atom_array, zero_out_orig_coordinates=is_inference
    )
    
    # Reset global token IDs after possible padding
    atom_array = add_global_token_id_annotation(atom_array)

    return atom_array    

def expand_unindexed_motifs(atom_array: AtomArray, zero_out_orig_coordinates: bool) -> AtomArray:
    '''
    Takes atom array and motif indices and padds the atom array to include unindexed motif atoms.

    is_motif_atom_without_index - indicates which atoms are indexed and which aren't (guideposts)

    Sets:
    '''
    atom_array.orig_res_id = atom_array.res_id.copy()  # back up original residue id for training metrics
    is_motif_atom_without_index = atom_array.get_annotation('is_motif_atom_without_index').copy()
    if not np.any(is_motif_atom_without_index):
        return atom_array
    
    # Copy to new atom array
    atom_array_to_concat = atom_array[is_motif_atom_without_index].copy()

    # Common bug: Must do by token, not just the fixed atoms, as we wan't to remove all original coordinate information
    is_motif_token_without_index = spread_token_wise(
        atom_array,
        atom_array.is_motif_atom_without_index[get_token_starts(atom_array)]
    )
    # ... Remove original coordinate information
    if zero_out_orig_coordinates:
        # See input_parsing.py for consistency in zeroing out noise init.
        atom_array.coord[is_motif_token_without_index] = 0.0
    else:
        # HACK: separate resid (causes issues with later transforms, either way not revealed to the model.)
        atom_array_to_concat.res_id = atom_array_to_concat.res_id + np.max(atom_array.res_id)

    # Reset is_motif_atom and is_motif_atom_without_index to contain no motif annotations where unindexed
    # TODO: Cleanup with mask.set_default_annotations
    zeros = np.zeros(is_motif_token_without_index.sum(), dtype=bool)
    atom_array.is_motif_atom[is_motif_token_without_index] = zeros
    atom_array.is_motif_atom_with_fixed_pos[is_motif_token_without_index] = zeros
    atom_array.is_motif_atom_without_index[is_motif_token_without_index] = zeros
    atom_array.is_motif_token[is_motif_token_without_index] = zeros
    atom_array.token_has_sequence[is_motif_token_without_index] = zeros

    # ... Set the residue as unknown such that the CA can be used as repr atom
    atom_array_to_concat.res_name = np.full(
        atom_array_to_concat.array_length(), "UNK", dtype=atom_array_to_concat.res_name.dtype
    )
    # TODO: for tipatoms, handle representative atoms for tipatom unindexed motifs (atomize?), currently this will also
    # assume the CA is the second atom in the token (see mask token in PadWithVirtual transform)

    # Concatenate unindexed parts to the end
    # TODO: Random concatenation to avoid striped attn edge effects?
    atom_array_full = atom_array + atom_array_to_concat

    # Ensure tokens are recognised as seperate
    assert get_token_count(atom_array_full) == \
        get_token_count(atom_array) + get_token_count(atom_array_to_concat), \
        f'Failed to create uniquely recognised tokens after concatenation. ' \
        f'Concatenated tokens: {get_token_count(atom_array_full)}\n'

    return atom_array_full


def create_unindexing_pair_mask(atom_array):
    '''
    Create L,L boolean matrix indicating the tokens which should absolutely
    not know the relative positions of one another.
    
    Used as input to the models' relative position encoding.

    atom_array: padded atom array
    '''
    token_starts = get_token_starts(atom_array)
    if not np.any(atom_array.is_motif_atom_without_index):
        L = len(token_starts)
        return np.zeros((L, L), dtype=bool)

    token_level_array = atom_array[token_starts]
    is_motif_token_without_index = token_level_array.is_motif_atom_without_index
    
    # ... First component of mask is that no unindexed atoms should talk to indexed ones.
    mask = is_motif_token_without_index[:, None] == ~is_motif_token_without_index[None, :]

    # ... Then, within unindexed tokens, seperate the islands based on where the token id breaks
    unindexed_all_LL = is_motif_token_without_index[ :, None] & is_motif_token_without_index[None, :]
    unindexed_token_level_array = token_level_array[is_motif_token_without_index]
    breaks = np.diff(unindexed_token_level_array.token_id) != 1  # (M-1,)
    group_ids = np.concatenate(([0], np.cumsum(breaks)))  # prepend 0 for M-1
    mask_unindexed_MM = group_ids[:, None] != group_ids[None, :]

    mask[unindexed_all_LL] = mask_unindexed_MM.flatten()

    return mask
