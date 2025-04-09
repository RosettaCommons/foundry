'''
Class-based motif masking system
'''

import logging
from abc import ABC, abstractmethod

import networkx as nx
import numpy as np
from datahub.utils.token import (
    get_token_count,
    get_token_starts,
    spread_token_wise,
)

nx.from_numpy_matrix = nx.from_numpy_array
logger = logging.getLogger(__name__)


class Mask(ABC):
    '''
    Base class for random masks to randomly create problems for the model to solve during training for a given example
    '''
    required_annotations = [
        'is_motif_atom',
        'is_motif_atom_without_index',
        'is_motif_atom_with_fixed_pos',
        'is_motif_token',
        'token_has_sequence',
    ]

    def __init__(self, frequency):
        self.frequency = frequency

    @staticmethod
    def set_default_annotations(atom_array, fill=False):
        '''
        Adds default annotations to the atom array
        '''
        for annotation in Mask.required_annotations:
            if annotation == 'is_motif_atom_without_index':
                atom_array.set_annotation(annotation, np.full(atom_array.array_length(), False, dtype=bool))
            atom_array.set_annotation(annotation, np.full(atom_array.array_length(), fill, dtype=bool))
        return atom_array

    @staticmethod
    def check_has_required_annotations(atom_array):
        '''
        Checks if the atom array has the default annotations
        '''
        received = atom_array.get_annotation_categories()
        for required_annotation in Mask.required_annotations:
            if required_annotation not in received:
                raise InvalidMaskException(f"Missing annotation category in mask output: {required_annotation}")
        return True

    @abstractmethod
    def is_valid_for_example(self, data):
        '''
        Returns true whether this mask can be applied to the data instance

        E.g. only use this transform if data metadata contains key or if data contains type
        '''
        return False

    @abstractmethod
    def sample(self, data):
        '''
        Sets annotations for the atom array:
        - is_motif_with_fixed
        - is_motif_without_index: which atoms are flagged for guideposting
        '''
        atom_array = data['atom_array']
        return atom_array


class UnconditionalMask(Mask):
    '''
    Basic mask for unconditional training. 
    '''
    def is_valid_for_example(self, data):
        ''''For unconditional generation, the example must contain protein'''
        is_protein = data['atom_array'].is_protein
        if not np.any(is_protein):
            return False
        return True

    def sample(self, data):
        '''
        Set all atoms to be masked
        '''
        atom_array = data['atom_array']
        atom_array = self.set_default_annotations(atom_array)
        return atom_array


class IndexedMask(UnconditionalMask):
    '''
    Indexed problem mask
    '''
    def __init__(self, 
        frequency,
        island_len_min,
        island_len_max,
        n_islands_min,
        n_islands_max,
        p_mask_motif_sequence,
        p_mask_motif_sidechains,
        n_motif_tokens_max=None,
        motif_conditioning_strategy='uniform',
    ):
        self.frequency = frequency
        self.island_len_min = island_len_min
        self.island_len_max = island_len_max
        self.n_islands_min = n_islands_min
        self.n_islands_max = n_islands_max
        self.n_motif_tokens_max = n_motif_tokens_max
        self.motif_conditioning_strategy = motif_conditioning_strategy

        # pretty often include the oxygen in the backbone in backbone-conditioned designs
        self.p_mask_motif_sequence = p_mask_motif_sequence
        self.p_mask_motif_sidechains = p_mask_motif_sidechains
        self.p_include_oxygen_in_backbone_mask = 0.95

    def sample_motif_tokens(self, atom_array):
        '''
        Samples what tokens should be considered motif.
        '''
        token_level_array = atom_array[get_token_starts(atom_array)]

        # initialize motif tokens as all non-protein tokens
        is_motif_token = ~token_level_array.is_protein
        n_protein_tokens = np.sum(token_level_array.is_protein)

        # Potential BUG: I think this will have issues with chain junctions
        n_islands = np.random.randint(self.n_islands_min, self.n_islands_max + 1)
        islands_mask = generate_island_mask(
            n_protein_tokens,
            island_len_min=self.island_len_min,
            island_len_max=self.island_len_max,
            n_islands=n_islands,
            max_length=self.n_motif_tokens_max,
        )
        is_motif_token[token_level_array.is_protein] = islands_mask
        return spread_token_wise(atom_array, is_motif_token)

    def sample_sequence_mask(self, atom_array):
        '''
        Samples what kind of conditioning to apply to motif tokens.
        
        NB: token_has_sequence is meant to symbolize that not all tokens are motif. 
        
        Argument attrs:
            - is_motif_token
        '''
        if self.motif_conditioning_strategy == 'uniform':
            if random_condition(self.p_mask_motif_sequence):
                token_has_sequence = np.zeros(atom_array.array_length(), dtype=bool)
            else:
                token_has_sequence = atom_array.is_motif_token.copy()
        elif self.motif_conditioning_strategy == 'random':
            # TODO: Sample different conditions based on islands
            raise NotImplementedError("Random conditioning not implemented yet")

        # By default reveal sequence for non-protein
        # TODO: Check this works with the reference conformer generation (left out for now)
        # token_has_sequence = token_has_sequence | ~atom_array.is_protein
        return token_has_sequence

    def sample_atom_mask(self, atom_array):
        '''
        Samples which atoms in motif tokens should be masked

        Argument attrs:
            - is_motif_token
            - token_has_sequence
        '''
        is_motif_atom = atom_array.is_motif_token.copy()
        
        if random_condition(self.p_mask_motif_sidechains):
            backbone_atoms = ['N', 'C', 'CA']
            if random_condition(self.p_include_oxygen_in_backbone_mask):
                backbone_atoms.append('O')
            is_motif_atom = is_motif_atom & np.isin(atom_array.atom_name, backbone_atoms)
        
        # By default set non-protein as motif too
        is_motif_atom = is_motif_atom | ~atom_array.is_protein
        return is_motif_atom

    def sample_fixed_mask(self, atom_array):
        '''
        Samples which atoms in motif tokens should be masked

        Argument attrs:
            - is_motif_token
            - token_has_sequence
            - is_motif_atom
        '''
        is_motif_atom_with_fixed_pos = atom_array.is_motif_atom.copy()
        return is_motif_atom_with_fixed_pos

    def sample_unindexed_mask(self, atom_array):
        '''
        Samples which atoms in motif tokens should be flagged for unindexing.
        
        Argument attrs:
            - is_motif_token
            - token_has_sequence
            - is_motif_atom
            - is_motif_atom_with_fixed_pos
        '''
        is_motif_atom_without_index = np.zeros(atom_array.array_length(), dtype=bool)
        return is_motif_atom_without_index

    def sample(self, data):
        atom_array = data['atom_array']

        # Set `is_motif_token`
        is_motif_token = self.sample_motif_tokens(atom_array)
        atom_array.set_annotation("is_motif_token", is_motif_token)

        # Set `token_has_sequence`
        token_has_sequence = self.sample_sequence_mask(atom_array)
        atom_array.set_annotation("token_has_sequence", token_has_sequence)

        # Set `is_motif_atom`
        is_motif_atom = self.sample_atom_mask(atom_array)
        atom_array.set_annotation("is_motif_atom", is_motif_atom)

        # Set `is_motif_atom_with_fixed_pos`
        is_motif_atom_with_fixed_pos = self.sample_fixed_mask(atom_array)
        atom_array.set_annotation("is_motif_atom_with_fixed_pos", is_motif_atom_with_fixed_pos)

        # Set `is_motif_atom_without_index`
        is_motif_atom_without_index = self.sample_unindexed_mask(atom_array)
        atom_array.set_annotation("is_motif_atom_without_index", is_motif_atom_without_index)

        return atom_array

class UnindexedMask(IndexedMask):
    '''
    Unindexed problem mask
    '''
    def sample_unindexed_mask(self, atom_array):
        '''
        Samples which atoms in motif tokens should be flagged for unindexing.
        
        Argument attrs:
            - is_motif_token
            - token_has_sequence
            - is_motif_atom
            - is_motif_atom_with_fixed_pos
        '''
        is_motif_atom_without_index = atom_array.is_motif_atom.copy()

        # Exclude non-protein from being unindexed
        is_motif_atom_without_index[~atom_array.is_protein] = False

        return is_motif_atom_without_index

##############################################################################################
# Additional classes
##############################################################################################

# Haven't tested if this works but this could be how you can do PPI masking
# class PPIMask(IndexedMask):
#     def is_valid_for_example(self, data):
#         result = 'query_pn_unit_iids' in data
#         self.query_pn_unit_iid = data['query_pn_unit_iids']
#         return result and super().is_valid_for_example(data)

#     def sample_motif_tokens(self, atom_array):
#         is_motif_token = atom_array.chain_id == self.query_pn_unit_iid
#         return is_motif_token

class TipatomIndexedMask(UnconditionalMask):
    ...

class SMBinderMask(UnindexedMask):

    def sample_motif_tokens(self, atom_array):
        '''TODO: Select tokens based on nearby residues to small molecule'''
        ...


def generate_island_mask(array_length, island_len_min=5, island_len_max=30, n_islands=1, max_length=None):
    """
    Generate a boolean mask of length `array_length` with random contiguous islands (True segments)
    while optionally constraining the total number of True values.
    
    Args:
        array_length (int): Total length of the boolean array.
        island_len_min (int): Minimum island length (inclusive).
        island_len_max (int): Maximum island length (inclusive).
        n_islands (int): Number of islands to attempt to generate.
        max_length (int, optional): Maximum allowed total number of True values in the output.
                                    If None, no constraint is applied.
        seed (int, optional): Random seed for reproducibility.
    
    Returns:
        np.ndarray: Boolean array of length `array_length` with island positions set to True.
    """
    mask = np.zeros(array_length, dtype=bool)
    for _ in range(n_islands):
        current_total = mask.sum()
        if max_length is not None:
            if current_total >= max_length:
                break
            remaining = max_length - current_total
        else:
            remaining = None  # not used
        
        # Randomly select a candidate island length.
        candidate_length = np.random.randint(island_len_min, island_len_max + 1)
        candidate_length = min(candidate_length, array_length)  # Fit into array
        
        # Choose a random starting index ensuring the island fits.
        high_start = array_length - candidate_length
        if high_start <= 0:
            start = 0
        else:
            start = np.random.randint(0, high_start + 1)
        
        # Evaluate the segment that would be activated.
        segment = mask[start:start + candidate_length]
        new_trues = np.sum(~segment)
        
        # If we have a maximum True budget and adding all new positions would exceed it, adjust the island.
        if max_length is not None and new_trues > remaining:
            # We try to trim the island so that it adds at most `remaining` new True values.
            count_new = 0
            adjusted_length = 0
            for i in range(candidate_length):
                if not mask[start + i]:
                    count_new += 1
                adjusted_length += 1
                # Once we've added as many new trues as allowed, break.
                if count_new >= remaining:
                    break
            # Only add the island if its adjusted length meets the minimum requirement.
            if adjusted_length < island_len_min:
                continue  # Skip this island and try the next one.
            mask[start:start + adjusted_length] = True
        else:
            # No max constraint or this candidate island fits within the remaining budget.
            mask[start:start + candidate_length] = True
    
    assert mask.sum() <= array_length, "Generated mask exceeds array length."
    assert mask.sum() != 0, "Generated mask is empty."
    return mask

def random_condition(p_cond):
    """
    Made this function because I always get confused by which order the
    inequality should be
    """
    assert 0 <= p_cond <= 1, "p_cond must be between 0 and 1"
    if p_cond == 0:
        return False
    else:
        return np.random.rand() < p_cond

class InvalidMaskException(Exception):
    def __init__(self, message="Could not create a valid mask for data example."):
        super().__init__(message)



        # # For any necessary post-processing of the atom array after creating spoofed input file.
        # if 'is_motif_atom' in atom_array.get_annotation_categories():
        #     is_motif_atom = atom_array.is_motif_atom.copy()
        #     # Annotations can sometimes be loaded as strings if they're set as boolean
        #     if isinstance(is_motif_atom[0], (str, np.str_)):
        #         atom_array.is_motif_atom.dtype=bool  # this line is necessary to make sure nit changes dtype
        #         atom_array.is_motif_atom = np.array([ast.literal_eval(x) for x in is_motif_atom], dtype=bool)
        #     else:
        #         atom_array.is_motif_atom = np.array(is_motif_atom, dtype=bool)

        #     if 'token_has_sequence' in atom_array.get_annotation_categories():
        #         has_sequence = atom_array.token_has_sequence.copy()
        #         if isinstance(has_sequence[0], (str, np.str_)):
        #             atom_array.token_has_sequence.dtype=bool
        #             atom_array.token_has_sequence = np.array([ast.literal_eval(x) for x in has_sequence], dtype=bool)

        #     if 'is_motif_atom_without_index' in atom_array.get_annotation_categories():
        #         has_sequence = atom_array.is_motif_atom_without_index.copy()
        #         if isinstance(has_sequence[0], (str, np.str_)):
        #             atom_array.is_motif_atom_without_index.dtype=bool
        #             atom_array.is_motif_atom_without_index = np.array([ast.literal_eval(x) for x in has_sequence], dtype=bool)

        # # ... Token is considered motif if any
        # is_motif_token = np.array([any(mask) for mask in is_motif_atom_by_token])  # (n_residues,) 
        
        # # ... Sequence is revelead if (all and indexed) or if provided already;
        # if 'token_has_sequence' in atom_array.get_annotation_categories():
        #     is_motif_token_with_seq = atom_array.token_has_sequence.copy()[token_starts]
        # else:
        #     is_motif_token_with_seq = np.array([all(mask) for mask in is_motif_atom_by_token]) \
        #         &  ~atom_array.is_motif_atom_without_index[token_starts]
            