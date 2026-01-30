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
    bp_partner_to_ss_matrix,
)

from atomworks.ml.utils.token import spread_token_wise, get_token_starts

def get_bp_feats_from_atom_array(
    atom_array: AtomArray,
) -> np.ndarray:
    """Build NA-SS features from atom_array annotations, assuming 'bp_partners' is present.

    This function reconstructs the SS matrix from the 'bp_partners' annotation on the atom_array,
    then one-hot encodes it into a 3-class matrix (mask, pair, loop).
    """
    # Fixed feature info (inferred from usage in other functions)
    feature_info = {
        'NA_SS_MASK': 0,  # Unspecified
        'NA_SS_PAIR': 1,  # Paired
        'NA_SS_LOOP': 2,  # Loop / unpaired
        'num_classes_nucleic_ss': 3,
    }

    # Check for required annotation
    if "bp_partners" not in atom_array.get_annotation_categories():
        raise ValueError("atom_array must have 'bp_partners' annotation for NA-SS feature building.")

    # Reconstruct SS matrix from annotations
    na_ss_matrix = np.asarray(
        bp_partner_to_ss_matrix(
            atom_array,
            feature_info=feature_info,
            NA_only=False,  # Include all residues (logic from other utils)
            planar_only=True,  # Use planar interactions (common default)
            include_loops=True,  # Include loop states
        ),
        dtype=np.int64,
    )

    # One-hot encode the matrix
    na_ss_matrix_int = np.asarray(na_ss_matrix, dtype=np.int64)
    eye = np.eye(int(feature_info['num_classes_nucleic_ss']), dtype=np.int64)
    return eye[na_ss_matrix_int]


def _build_na_ss_features_from_annotations(
    atom_array: AtomArray,
    *,
    feature_info: dict,
    num_classes: int,
    NA_only: bool,
    planar_only: bool,
    is_nucleic_ss_example: bool,
    give_partial_feats: bool,
    get_feature_mask_fn,
) -> np.ndarray:
    """Reconstruct SS matrix from annotations, optionally mask, then one-hot."""
    na_ss_matrix = np.asarray(
        bp_partner_to_ss_matrix(
            atom_array,
            feature_info=feature_info,
            NA_only=NA_only,
            planar_only=planar_only,
            include_loops=True,
        ),
        dtype=np.int64,
    )

    n_tokens = int(na_ss_matrix.shape[0])

    if give_partial_feats:
        is_shown = (
            np.asarray(get_feature_mask_fn(n_tokens), dtype=bool)
            if is_nucleic_ss_example
            else np.zeros((n_tokens,), dtype=bool)
        )
        na_ss_matrix[~is_shown, :] = feature_info["NA_SS_MASK"]
        na_ss_matrix[:, ~is_shown] = feature_info["NA_SS_MASK"]

    na_ss_matrix_int = np.asarray(na_ss_matrix, dtype=np.int64)
    eye = np.eye(int(num_classes), dtype=np.int64)
    return eye[na_ss_matrix_int]


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
                via the ``bp_partner`` annotation (annotation-first), then reconstructs the
                matrix (and optionally masks parts of it) before one-hot encoding.

        Inference:
            - Interprets user-provided secondary-structure specifications, writes the same
                ``bp_partner`` annotation, then follows the same matrix + one-hot path.

        Note: helical-parameter features are not implemented/used in this refactored path.
    """

    def __init__(
        self,
        is_inference,
        add_nucleic_ss_feats: bool = True,

        p_is_nucleic_ss_example: float = 0.3,
        p_show_partial_feats: float = 0.5,
        nucleic_ss_min_shown: float = 0.0,
        nucleic_ss_max_shown: float = 1.0,
        n_islands_min: int = 1,
        n_islands_max: int = 6,
        p_canonical_bp_filter: float = 0.0,

        # USE_RF2AA_NAMES: bool = False,
        NA_only: bool = False,
        planar_only : bool = True,

    ):
        # Critical, must always have to know how to handle
        self.is_inference = is_inference 

        # For sampling whether we add nucleic-ss features (extra t2d)
        self.add_nucleic_ss_feats    = add_nucleic_ss_feats
        self.p_canonical_bp_filter   = p_canonical_bp_filter # enforce that bp labels are only canonical
        self.p_is_nucleic_ss_example = p_is_nucleic_ss_example
        self.nucleic_ss_min_shown    = nucleic_ss_min_shown
        self.nucleic_ss_max_shown    = nucleic_ss_max_shown
        self.n_islands_min           = n_islands_min
        self.n_islands_max           = n_islands_max

        self.p_show_partial_feats = p_show_partial_feats

        # Filters for what can be considered a planar contact interaction
        self.NA_only = NA_only # only annotate base-like interactions for nucleic acid residues
        self.planar_only = planar_only # only consider planar atoms in sidechains for geometry calculations,
        self.p_canonical_bp_filter = p_canonical_bp_filter # probability of enforcing canonical base pair filter

        # Inds of annotation types in the nucleic-ss features (stack of 3 matrices):
        self.feature_info = {
            'NA_SS_MASK' : 0, # Unspecified, or sm, or protein:
            'NA_SS_PAIR' : 1,
            'NA_SS_LOOP' : 2,
            'num_classes_nucleic_ss' : 3,
        }


    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["res_name"])
        # maybe do later: check_atom_array_has_hydrogen(data)

    def _sample_training_flags(self) -> tuple[bool, bool]:
        """Sample booleans controlling whether/how features are shown in training."""
        is_nucleic_ss_example = bool(
                    self.add_nucleic_ss_feats
                    and (np.random.rand() < self.p_is_nucleic_ss_example)
                )
        give_partial_feats = bool(
                    np.random.rand() < self.p_show_partial_feats
                )
        return is_nucleic_ss_example, give_partial_feats

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        # Calculate n_tokens (assuming one token per residue for simplicity)
        token_starts = get_token_starts(atom_array)
        token_level_array = atom_array[token_starts]
        token_ids = [int(t) for t in token_level_array.token_id]
        n_tokens = len(token_starts)
        print(" DO I NEED TO CHANGE TO TOKEN_ID???")
        # Handle the training case with ground truth and masking:
        if not self.is_inference:

            # First, annotate as usual
            # atom_array = annotate_na_ss(atom_array, **kwargs)
            atom_array = annotate_na_ss(atom_array,
                            NA_only=self.NA_only,
                            planar_only=self.planar_only,
                            p_canonical_bp_filter=self.p_canonical_bp_filter,
                            )
            
            # Sample mask on token level:
            is_nucleic_ss_example, give_partial_feats = self._sample_training_flags()
            is_ss_shown = self._sample_where_to_show_ss(n_tokens,
                                                     is_nucleic_ss_example=is_nucleic_ss_example,
                                                     give_partial_feats=give_partial_feats) # Mask vec for tokens where ss shown
            # Spread mask to atom level
            is_ss_shown = spread_token_wise(atom_array, is_ss_shown)

            
            # Extract the base pair annotations
            bp_partners_atom = atom_array.get_annotation("bp_partners")

            # Remove unshown positions from bp_partners annotation
            bp_partners_atom[~is_ss_shown] = None
            
            # Reset the annotation with newly hidden positions
            atom_array.set_annotation("bp_partners", bp_partners_atom)

        # Inference case: create from commandline args
        else:
            """
            Different cases handled:
            - 1). Single dot-bracket string
            - 2). multiple dot bracket strings with chain/ind ranges specified
            - 3). Lists of paired indices

            """
            is_nucleic_ss_example=True
            give_partial_feats=False
            atom_array = annotate_na_ss_from_data_specification(
                data,
                overwrite=True,
            )

        # Check feats existence and update:
        if "feats" not in data:
            data["feats"] = {}

        # data["feats"].update(nucleic_features)
        data.setdefault("log_dict", {})
        log_dict = data["log_dict"]
        data["log_dict"] = log_dict
        data["atom_array"] = atom_array

        return data


    def _sample_where_to_show_ss(self, n_tokens: int,
                                 is_nucleic_ss_example: bool = True,
                                 give_partial_feats: bool = True,
                                 ) -> np.ndarray:
        """Sample token-level islands indicating which SS rows/cols to reveal."""
        # If NOT is_nucleic_ss_example, set is_shown to all False
        if not is_nucleic_ss_example:
            return np.zeros((n_tokens,), dtype=bool)
        
        # If NOT give_partial_feats, set is_shown to all True
        if not give_partial_feats:
            return np.ones((n_tokens,), dtype=bool)
        else:
            frac_shown = (
                self.nucleic_ss_min_shown
                + (self.nucleic_ss_max_shown - self.nucleic_ss_min_shown) * np.random.rand()
            )
            frac_shown = float(np.clip(frac_shown, 0.0, 1.0))
            max_length = int(np.ceil(frac_shown * n_tokens))
            if max_length <= 0:
                return np.zeros((n_tokens,), dtype=bool)

            island_len_min = max(1, int(frac_shown * n_tokens // max(int(self.n_islands_max), 1)))
            island_len_max = max(1, int(frac_shown * n_tokens // max(int(self.n_islands_min), 1)))
            island_len_min = min(island_len_min, n_tokens)
            island_len_max = min(island_len_max, n_tokens)
            island_len_max = max(island_len_max, island_len_min)
        
            return sample_island_tokens(
                n_tokens,
                island_len_min=island_len_min,
                island_len_max=island_len_max,
                n_islands_min=self.n_islands_min,
                n_islands_max=self.n_islands_max,
                max_length=max_length,
            )
        