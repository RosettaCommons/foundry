from typing import Any
import numpy as np
from functools import partial
from biotite.structure import AtomArray
from atomworks.ml.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_is_instance,
)
from atomworks.ml.transforms.base import Transform
from rfd3.transforms.conditioning_utils import sample_island_tokens
from rfd3.transforms.na_geom_utils import (
    annotate_na_ss, 
    annotate_na_ss_from_data_specification, 
    DEFAULT_NA_SS_FEATURE_INFO,
)

from atomworks.ml.utils.token import spread_token_wise, get_token_starts

def na_ss_feats_from_annotation(atom_array: AtomArray,
                                token_starts= None, 
                                n_tokens = None,
                                return_as_onehot = True,
                                ) -> np.ndarray:
    """
    Takes in atom array and constucts a base pair feature matrix from annotations,
    according to to custom feature constuction + masking system.
    This featurization utilizes info from BasePairEnum to assign int values 
    to paired, unpaired, and masked positions in the matrix.

    Args:
        * atom_array: AtomArray with bp_partners annotation at atom level
        * token_starts (optional): indices of token starts in the atom array
        * n_tokens (optional): number of tokens (length of token_starts)
        * return_as_onehot (optional): if False, return integer-encoded 
                    matrix instead of one-hot encoded matrix

    returns:
        * na_ss_matrix:
            If ``return_as_onehot`` is True (default):
                    np.ndarray of shape (n_tokens, n_tokens, n_classes) 
                        with one-hot encoded values according to BasePairEnum

            If ``return_as_onehot`` is False :
                    np.ndarray of shape (n_tokens, n_tokens) 
                        with int values according to BasePairEnum


    """
    # Get this info from atom_array, or avoid if given
    if (token_starts is None) or (n_tokens is None):
        token_starts = get_token_starts(atom_array)
        n_tokens = len(token_starts)
    

    # Collect token inds for paired or loop positions:
    pair_inds = []
    loop_inds = []
    token_bp_partners = atom_array.get_annotation("bp_partners")[token_starts] # get bp_partners at token level
    assert len(token_bp_partners) == n_tokens, "Length of token_bp_partners should match n_tokens"
    for i, j_list in enumerate(token_bp_partners):
        if j_list is not None:
            if len(j_list) > 0:
                for j in j_list:
                    pair_inds.append((i, j))
            else:
                loop_inds.append(i)

    # The standard system for constructing meaningful base pair features:
    # 0). Initialize with values of UNSPECIFIED (0): int matrix of shape (n_tokens, n_tokens)
    na_ss_matrix = np.full((n_tokens, n_tokens), DEFAULT_NA_SS_FEATURE_INFO["NA_SS_MASK"], dtype=np.int64)

    # 1). Fill in with values of PAIR (1) at positions that have bp_partners annotated as a non-empty list
    for pair_i, pair_j in pair_inds:
        na_ss_matrix[pair_i, pair_j] = DEFAULT_NA_SS_FEATURE_INFO["NA_SS_PAIR"]
        na_ss_matrix[pair_j, pair_i] = DEFAULT_NA_SS_FEATURE_INFO["NA_SS_PAIR"] # ensure symmetry

    # 2). Fill in with values of LOOP (2) at positions that have bp_partners annotated as an empty list (explicitly unpaired)
    # (we make full stripes across that position's row/col to indicate that NONE of those other positions are paired )
    for loop_i in loop_inds:
        na_ss_matrix[loop_i, :] = DEFAULT_NA_SS_FEATURE_INFO["NA_SS_LOOP"]
        na_ss_matrix[:, loop_i] = DEFAULT_NA_SS_FEATURE_INFO["NA_SS_LOOP"] # ensure symmetry
        
    # Optional: convert NA-SS matrix to one-hot encoding according for model input:
    if return_as_onehot:
        na_ss_matrix = np.eye(len(DEFAULT_NA_SS_FEATURE_INFO), dtype=np.int64)[na_ss_matrix]
    
    return na_ss_matrix


class CalculateNucleicAcidGeomFeats(Transform):
    """
        Transform for constructing nucleic-acid conditioning features.

        This transform currently produces only nucleic-acid secondary-structure (NA-SS)
        features as a 2D token-token matrix with 3 bins:
            * 0: mask / unspecified
            * 1: paired
            * 2: loop / explicitly unpaired

        Training:
            - Computes geometry/H-bond-based base pairs and writes them onto the AtomArray
                via the ``bp_partners`` annotation (annotation-first), then reconstructs the
                matrix (and optionally masks parts of it) before one-hot encoding.

        Inference:
            - Interprets user-provided secondary-structure specifications, writes the same
                ``bp_partners`` annotation, then follows the same matrix + one-hot path.

        Note: helical-parameter features are not implemented/used in this refactored path.
    """

    def __init__(
        self,
        is_inference,
        # Conditional sampling parameters all stored in this dict:
        meta_conditioning_probabilities: dict[str, float] = None,
        
        # Mask control paramerers:
        nucleic_ss_min_shown: float = 0.2,
        nucleic_ss_max_shown: float = 1.0,
        n_islands_min: int = 1,
        n_islands_max: int = 6,

        # USE_RF2AA_NAMES: bool = False,
        NA_only: bool = False,
        planar_only : bool = True,

    ):
        # Critical, must always have to know how to handle
        self.is_inference = is_inference 
        
        self.meta_conditioning_probabilities = meta_conditioning_probabilities or {}
        
        # Control whether we show some nucleic SS or default to full 2D mask
        self.p_is_nucleic_ss_example = self.meta_conditioning_probabilities.get("p_is_nucleic_ss_example", 0.0)

        # Control whether we define full SS or just part of it (only applies if is NA SS example)
        self.p_show_partial_feats    = self.meta_conditioning_probabilities.get("p_nucleic_ss_show_partial_feats", 0.0)

        # Some frac of time default to only showing canonical base pairs
        self.p_canonical_bp_filter   = self.meta_conditioning_probabilities.get("p_canonical_bp_filter", 0.5)
        
        # mask patterning control to make things resemble design scenarios
        self.nucleic_ss_min_shown    = nucleic_ss_min_shown
        self.nucleic_ss_max_shown    = nucleic_ss_max_shown
        self.n_islands_min           = n_islands_min
        self.n_islands_max           = n_islands_max

        # Filters for what can be considered a planar contact interaction
        self.NA_only = NA_only # only annotate base-like interactions for nucleic acid residues
        self.planar_only = planar_only # only consider planar atoms in sidechains for geometry calculations,



    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["res_name"])
        # maybe do later: check_atom_array_has_hydrogen(data)

    def _sample_training_flags(self) -> tuple[bool, bool]:
        """Sample booleans controlling whether/how features are shown in training."""
        is_nucleic_ss_example = bool(np.random.rand() < self.p_is_nucleic_ss_example)
        give_partial_feats = bool(
                    np.random.rand() < self.p_show_partial_feats
                )
        return is_nucleic_ss_example, give_partial_feats

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        # Calculate n_tokens (assuming one token per residue for simplicity)
        token_starts = get_token_starts(atom_array)
        n_tokens = len(token_starts)
        # token_level_array = atom_array[token_starts]

        # Handle the training case with ground truth and masking
        if not self.is_inference:

            # First, annotate as usual
            is_nucleic_ss_example, give_partial_feats = self._sample_training_flags()

            if is_nucleic_ss_example:
            
                atom_array = annotate_na_ss(atom_array,
                            NA_only=self.NA_only,
                            planar_only=self.planar_only,
                            p_canonical_bp_filter=self.p_canonical_bp_filter,
                            )

                # Generate symmetric partner annotations at the token level for masking purposes.
                # choice for object-consistency: if already masked/undefined: be a list mapping to self-index.
                partner_sym_map = {
                    i: atom_array.bp_partners[ts_i] if atom_array.bp_partners[ts_i] is not None else [i]
                    for i, ts_i in enumerate(token_starts)
                    }

                # # Sample mask on token level:
                token_mask_to_show = self._sample_where_to_show_ss(
                    n_tokens,
                    is_nucleic_ss_example=is_nucleic_ss_example,
                    give_partial_feats=give_partial_feats,
                    partner_sym_map=partner_sym_map,
                )  # Mask vec for tokens where ss shown

                # Spread mask to atom level
                is_ss_shown = spread_token_wise(atom_array, token_mask_to_show)
                
                # Extract the base pair annotations
                bp_partners_atom = atom_array.get_annotation("bp_partners")

                # Remove unshown positions from bp_partners annotation
                bp_partners_atom[~is_ss_shown] = None
                
                # Reset the annotation with newly hidden positions
                atom_array.set_annotation("bp_partners", bp_partners_atom)
            else:
                atom_array.set_annotation("bp_partners", np.array([None]*len(atom_array)))

        # Inference case: create from commandline args
        else:
            """
            Different cases handled:
            - 1). Single dot-bracket string
            - 2). multiple dot bracket strings with chain/ind ranges specified
            - 3). Lists of paired indices
            """
            atom_array = annotate_na_ss_from_data_specification(
                data,
                overwrite=True,
            )

        # Check feats existence and update:
        if "feats" not in data:
            data["feats"] = {}

        data.setdefault("log_dict", {})
        log_dict = data["log_dict"]
        data["log_dict"] = log_dict
        data["atom_array"] = atom_array
        
        return data


    def _sample_where_to_show_ss(self, n_tokens: int,
                                 is_nucleic_ss_example: bool = True,
                                 give_partial_feats: bool = True,
                                 partner_sym_map: dict[int, list[int]] = None,
                                 ) -> np.ndarray:
        """Sample token-level islands indicating which SS rows/cols to reveal.
            This custom function allows for enforcing symmetry in the shown features according 
            to the partner_sym_map, which encodes which tokens are partners in the SS 
            matrix and thus should be masked/unmasked together to maintain consistency.
        
        """
        # If NOT is_nucleic_ss_example, set is_shown to all False
        if not is_nucleic_ss_example:
            token_mask_to_show = np.zeros((n_tokens,), dtype=bool)
        
        # If NOT give_partial_feats, set is_shown to all True
        if not give_partial_feats:
            token_mask_to_show = np.ones((n_tokens,), dtype=bool)
        else:
            # Get numerical parameters for that govern the mask pattern
            frac_shown = (
                self.nucleic_ss_min_shown
                + (self.nucleic_ss_max_shown - self.nucleic_ss_min_shown) * np.random.rand()
            )
            frac_shown = float(np.clip(frac_shown, 0.0, 1.0))
            max_length = int(np.ceil(frac_shown * n_tokens))
            if max_length <= 0:
                token_mask_to_show = np.zeros((n_tokens,), dtype=bool)
            island_len_min = max(1, int(frac_shown * n_tokens // max(int(self.n_islands_max), 1)))
            island_len_max = max(1, int(frac_shown * n_tokens // max(int(self.n_islands_min), 1)))
            island_len_min = min(island_len_min, n_tokens)
            island_len_max = min(island_len_max, n_tokens)
            island_len_max = max(island_len_max, island_len_min)

            # Sample the actual mask using the utility function:
            token_mask_to_show = sample_island_tokens(
                n_tokens,
                island_len_min=island_len_min,
                island_len_max=island_len_max,
                n_islands_min=self.n_islands_min,
                n_islands_max=self.n_islands_max,
                max_length=max_length,
            )

            # Handle symmetry by iterating through the partner_sym_map items and setting 
            # `partner_mask_to_show` at partner positions to match `token_mask_to_show`
            # initialize as all shown so effect comes from hiding + logical AND condition
            partner_mask_to_show = np.ones_like(token_mask_to_show)
            for token_i, partner_ind_list in partner_sym_map.items():
                for partner_ind in partner_ind_list:
                    partner_mask_to_show[partner_ind] = token_mask_to_show[token_i]

            # Combine the original mask with the partner mask to ensure symmetry
            token_mask_to_show = token_mask_to_show & partner_mask_to_show

        return token_mask_to_show
        
