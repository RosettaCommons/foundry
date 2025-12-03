"""
Contains (a) global conditioning syntax and (b) transforms for pipeline

Conditioning pipeline:
    inference --- create_atom_array_from_design_specification ---|
                                                                 |---> CreateConditionedArray
    training  ---           SampleConditioningFlags           ---|
"""

import ast
import copy
import logging

import biotite.structure as struc
import hydra
import networkx as nx
import numpy as np
from atomworks.ml.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_is_instance,
)
from atomworks.ml.transforms.atom_array import (
    add_global_token_id_annotation,
    add_protein_termini_annotation,
)
from atomworks.ml.transforms.base import Transform
from atomworks.ml.utils.token import (
    apply_and_spread_token_wise,
    get_token_count,
    get_token_starts,
)
from biotite.structure import AtomArray
from rfd3.constants import (
    OPTIONAL_CONDITIONING_VALUES,
    REQUIRED_CONDITIONING_ANNOTATIONS,
)
from rfd3.transforms.conditioning_utils import random_condition
from rfd3.transforms.util_transforms import (
    add_representative_atom,
)

from foundry.common import exists

nx.from_numpy_matrix = nx.from_numpy_array
logger = logging.getLogger(__name__)
NHEAVYPROT = 14


#################################################################################
# Base conditioning definititions
#################################################################################


def get_motif_features(atom_array):
    is_fixed = atom_array.is_motif_atom_with_fixed_coord.astype(bool)
    is_sequence_fixed = atom_array.is_motif_atom_with_fixed_seq.astype(bool)
    is_unindexed = atom_array.is_motif_atom_unindexed.astype(bool)

    # Motif atom if has any conditioning
    is_motif_atom = is_fixed | is_sequence_fixed | is_unindexed
    is_motif_token = apply_and_spread_token_wise(
        atom_array, is_motif_atom, function=lambda x: np.any(x)
    )  # Has any atoms with conditioning

    return {"is_motif_atom": is_motif_atom, "is_motif_token": is_motif_token}


def set_default_conditioning_annotations(
    atom_array,
    motif=False,
    unindexed=False,
    mask=None,
    dtype=bool,
    additional: set | list = None,
):
    """
    Adds default annotations to the atom array

    Args:
        motif: True if default for a fully fixed motif, False if default for a fully diffused motif
        unindexed: True if the tokens in the atom array should be motif
        mask: boolean mask for array of which atoms to apply the assignments to.
    NB: In both cases, the defaults for unindexed are False
    """

    # All annotations set to true for motif
    fill = True if motif else False
    if mask is not None:
        # TODO: support defaulting to nulls
        check_has_required_conditioning_annotations(atom_array)
        trues = np.full(mask.sum(), True, dtype=dtype)
        falses = np.full(mask.sum(), False, dtype=dtype)

        atom_array.is_motif_atom_unindexed[mask] = trues if unindexed else falses
        atom_array.is_motif_atom_unindexed_motif_breakpoint[mask] = falses

        # Others:
        for annotation in REQUIRED_CONDITIONING_ANNOTATIONS:
            if annotation in [
                "is_motif_atom_unindexed",
                "is_motif_atom_unindexed_motif_breakpoint",
            ]:
                continue

            vals = copy.deepcopy(atom_array.get_annotation(annotation))
            vals[mask] = trues if fill else falses
            atom_array.set_annotation(annotation, vals)
    else:
        for annotation in REQUIRED_CONDITIONING_ANNOTATIONS:
            if annotation in [
                "is_motif_atom_unindexed",
            ]:
                atom_array.set_annotation(
                    annotation,
                    np.full(atom_array.array_length(), unindexed, dtype=dtype),
                )
            elif annotation in [
                "is_motif_atom_unindexed_motif_breakpoint",
            ]:
                atom_array.set_annotation(
                    annotation, np.full(atom_array.array_length(), False, dtype=dtype)
                )
            else:
                atom_array.set_annotation(
                    annotation, np.full(atom_array.array_length(), fill, dtype=dtype)
                )

    if additional is not None:
        for annot, val in OPTIONAL_CONDITIONING_VALUES.items():
            if (
                annot in additional
                and annot not in atom_array.get_annotation_categories()
            ):
                atom_array.set_annotation(
                    annot, np.full(atom_array.array_length(), val)
                )

    return atom_array


def check_has_required_conditioning_annotations(
    atom_array, required=REQUIRED_CONDITIONING_ANNOTATIONS
):
    """
    Checks if the atom array has the correct conditioning annotations
    """
    received = atom_array.get_annotation_categories()
    for required_annotation in required:
        if required_annotation not in received:
            raise InvalidSampledConditionException(
                f"Missing annotation category in atom_array: {required_annotation}"
            )
    return True


def convert_existing_annotations_to_bool(
    atom_array, annotations=REQUIRED_CONDITIONING_ANNOTATIONS
):
    # When loading from cif, annotations are loaded as strings when they should be boolean
    for annotation in annotations:
        if annotation not in atom_array.get_annotation_categories():
            continue
        tmp = atom_array.get_annotation(annotation).copy()
        atom_array.get_annotation(annotation).dtype = bool
        if isinstance(tmp[0], (str, np.str_, np.dtypes.StrDType)):
            tmp = np.array([ast.literal_eval(x) for x in tmp], dtype=bool)
        else:
            tmp = np.asarray(tmp, dtype=bool)
        atom_array.set_annotation(annotation, tmp)
    return atom_array


def convert_existing_annotations_to_int(
    atom_array, annotations=REQUIRED_CONDITIONING_ANNOTATIONS
):
    # When loading from cif, annotations are loaded as strings when they should be boolean
    for annotation in annotations:
        if annotation not in atom_array.get_annotation_categories():
            continue
        tmp = atom_array.get_annotation(annotation).copy()
        if isinstance(tmp[0], (str, np.str_, np.bool_, bool, np.dtypes.BoolDType)):
            tmp = np.array([int(x) for x in tmp], dtype=int)
        atom_array.set_annotation(annotation, tmp)
    return atom_array


class StrtoBoolforIsXFeatures(Transform):
    def check_input(self, *args, **kwargs):
        pass

    def __init__(self):
        pass

    def forward(self, data):
        atom_array = data["atom_array"]
        convert_existing_annotations_to_bool(atom_array)
        data["atom_array"] = atom_array
        return data


class InvalidSampledConditionException(Exception):
    def __init__(self, message="Error during sampling of condition."):
        self.message = message
        super().__init__(self.message)


#################################################################################
# Transform for pipeline (training & inference)
#################################################################################


class SampleConditioningType(Transform):
    """
    Applies conditional assignments

    Args:
      train_conditions: List[RandomMask]
      seed (int): random seed, for controling the masking results

    Return:
      atom_array with three more annotations:
       - is_motif_token: tokens to be motif
       - is_motif_atom: atoms to be motif
       - is_motif_atom_with_fixed_seq: for which atom we know the true restype
    """

    requires_previous_transforms = [
        "AssignTypes",
    ]

    def __init__(
        self,
        *,
        train_conditions: dict,
        meta_conditioning_probabilities: dict,
        sequence_encoding,
    ):
        if exists(train_conditions):
            train_conditions = hydra.utils.instantiate(
                train_conditions, _recursive_=True
            )
        self.meta_conditioning_probabilities = meta_conditioning_probabilities
        self.train_conditions = train_conditions
        self.sequence_encoding = sequence_encoding

    def check_input(self, data: dict):
        assert not data["is_inference"], "This transform is only used during training!"
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["pn_unit_id", "pn_unit_iid"])
        existing = [
            cat in REQUIRED_CONDITIONING_ANNOTATIONS
            for cat in data["atom_array"].get_annotation_categories()
        ]
        assert not any(
            existing
        ), "Conditioning annotations already set! found {}".format(existing)
        assert "conditions" in data, "Conditioning dict not initialized"

    def forward(self, data):
        valid_conditions = [
            cond
            for cond in self.train_conditions.values()
            if cond.frequency > 0 and cond.is_valid_for_example(data)
        ]

        if len(valid_conditions) == 0:
            raise InvalidSampledConditionException("No valid condition was found.")

        p_cond = np.array([cond.frequency for cond in valid_conditions])
        if p_cond.sum() == 0:
            raise InvalidSampledConditionException(
                "No valid condition was found with non-zero frequency."
            )
        p_cond = p_cond.astype(np.float64)
        p_cond /= p_cond.sum()
        i_cond = np.random.choice(np.arange(len(p_cond)), p=p_cond)
        cond = valid_conditions[i_cond]

        data["sampled_condition"] = cond
        data["sampled_condition_name"] = cond.name
        data["sampled_condition_cls"] = cond.__class__

        # Sample canonical conditioning flags for downstream processing
        for k, p in self.meta_conditioning_probabilities.items():
            data["conditions"][k] = random_condition(p)

        return data


class SampleConditioningFlags(Transform):
    requires_previous_transforms = [
        "FlagAndReassignCovalentModifications",
        "AssignTypes",
        "SampleConditioningType",
    ]  # We use is_protein in the PPI training condition

    def check_input(self, data):
        assert not data[
            "is_inference"
        ], "This transform is only used during training! Validation using sampled conditions is not implemented yet"
        assert "sampled_condition" in data

    def forward(self, data: dict) -> dict:
        cond = data["sampled_condition"]

        # Sample canonical conditioning flags for atom array
        atom_array = cond.sample(data)
        data["atom_array"] = atom_array

        return data


class UnindexFlaggedTokens(Transform):
    """
    Serves as the merge point between training / infernece conditioning pipelines
    """

    def __init__(self, central_atom):
        """
        Args:
            central_atom: The atom to use as the central atom for unindexed motifs.
        """
        super().__init__()
        self.central_atom = central_atom

    def check_input(self, data: dict):
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)

    def expand_unindexed_motifs(
        self, atom_array: AtomArray, pop_orig_tokens: bool
    ) -> AtomArray:
        """
        Takes atom array and motif indices and padds the atom array to include unindexed motif atoms.

        is_motif_atom_unindexed - Whether an atom is flagged to be a guidepost
        During training, the original coordinates are left behind for the model to learn to diffuse,
        during inference, the original tokens are removed by default.
        """
        # back up original residue id for training metrics
        atom_array.set_annotation("orig_res_id", atom_array.res_id.copy())
        is_motif_atom_unindexed = atom_array.is_motif_atom_unindexed.copy()
        if not np.any(is_motif_atom_unindexed):
            return atom_array

        # ... A token is to be unindexed if any atoms in the token are unindexed
        max_resid = np.max(atom_array.res_id)
        starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
        token_to_unindex = struc.spread_residue_wise(
            atom_array,
            struc.apply_residue_wise(
                atom_array,
                is_motif_atom_unindexed,
                function=lambda x: np.any(x),
            ),
        )
        assert token_to_unindex.sum() > 0, "No tokens to unindex!"
        idxs = np.arange(atom_array.array_length())
        unindexed_tokens = []
        for i, (start, end) in enumerate(zip(starts[:-1], starts[1:])):
            if not token_to_unindex[start]:
                continue
            subset_mask = np.isin(idxs, idxs[start:end])
            token = copy.deepcopy(atom_array[subset_mask])
            token = token[token.is_motif_atom_unindexed]
            token.res_id = token.res_id + max_resid
            token.is_C_terminus[:] = False
            token.is_N_terminus[:] = False
            assert token.is_protein.all(), f"Cannot unindex non-protein token: {token}"
            token = add_representative_atom(token, central_atom=self.central_atom)
            unindexed_tokens.append(token)

        # ... Remove original tokens e.g. during inference
        if pop_orig_tokens:
            atom_array = atom_array[~token_to_unindex]
            # Reassign Termini features
            atom_array = add_protein_termini_annotation(atom_array)
        else:
            # Reset is_motif_atom and is_motif_atom_unindexed to contain no motif annotations where unindexed
            # I.e model should view the original tokens the same as every other diffused token
            atom_array.is_motif_atom[token_to_unindex] = False
            atom_array.is_motif_atom_with_fixed_coord[token_to_unindex] = False
            atom_array.is_motif_token[token_to_unindex] = False
            atom_array.is_motif_atom_with_fixed_seq[token_to_unindex] = False
            atom_array.is_motif_atom_unindexed[token_to_unindex] = False
            atom_array.is_motif_atom_unindexed_motif_breakpoint[token_to_unindex] = (
                False
            )

        # Concatenate unindexed parts to the end
        atom_array_full = struc.concatenate([atom_array] + unindexed_tokens)
        atom_array_to_concat = struc.concatenate(unindexed_tokens)
        # Ensure tokens are recognised as seperate
        n_unindexed_tokens = get_token_count(atom_array_to_concat)
        assert n_unindexed_tokens == len(
            unindexed_tokens
        ), f"Expected {len(unindexed_tokens)} but got {n_unindexed_tokens}"
        assert (
            get_token_count(atom_array_full)
            == get_token_count(atom_array) + n_unindexed_tokens
        ), (
            f"Failed to create uniquely recognised tokens after concatenation.\n"
            f"Concatenated tokens: {get_token_count(atom_array_full)}, unindexed: {n_unindexed_tokens}"
        )

        return atom_array_full

    def create_unindexed_masks(
        self,
        atom_array,
        is_inference=False,
    ):
        """
        Create L,L boolean matrix indicating the tokens which should absolutely
        not know the relative positions of one another.

            False when positional leakage is allowed
            True when positional leakage is disallowed

        Used as input to the models' relative position encoding.

        breaks:
            boolean atom-wise array indicating which token breaks the group ids up.
            if all are false, all indices are leaked. If the first break of the unindexed tokens is
            True, the cross-motif couplings are leaked but not the global index

        atom_array: padded atom array
        """
        token_starts = get_token_starts(atom_array)
        token_level_array = atom_array[token_starts]
        is_motif_token_unindexed = token_level_array.is_motif_atom_unindexed

        # ... Grab breaks from the token level array
        unindexed_token_level_array = token_level_array[is_motif_token_unindexed]
        breaks = unindexed_token_level_array.is_motif_atom_unindexed_motif_breakpoint

        leak_all = not np.any(breaks)
        if leak_all:
            if is_inference and np.any(is_motif_token_unindexed):
                logger.info("Indexing all unindexed components")
            L = len(token_starts)
            return np.zeros((L, L), dtype=bool), is_motif_token_unindexed

        # ... First component of mask is that no unindexed atoms should talk to indexed ones.
        mask = (
            is_motif_token_unindexed[:, None] == ~is_motif_token_unindexed[None, :]
        )  # [intra indexed + intra unindexed]

        # ... Then, within unindexed tokens, seperate the islands based on where the token id breaks
        unindexed_all_LL = (
            is_motif_token_unindexed[:, None] & is_motif_token_unindexed[None, :]
        )  # [intra unindexed]

        ########################################################################################
        # Determine intra-unindexed resid leakage
        ########################################################################################
        # ... Mask out intra-unindexed off-diagonals as necessary
        group_ids = np.cumsum(breaks)
        mask_unindexed_MM = group_ids[:, None] != group_ids[None, :]
        mask[unindexed_all_LL] = mask_unindexed_MM.flatten()

        return mask, is_motif_token_unindexed

    def forward(self, data: dict):
        atom_array = data["atom_array"]
        if "feats" not in data:
            data["feats"] = {}

        # ... Ensure conditioning flags are set correctly
        # NOTE: Join point for inference and training conditioning pipelines
        check_has_required_conditioning_annotations(atom_array)

        is_unindexed_token = apply_and_spread_token_wise(
            atom_array,
            atom_array.is_motif_atom_unindexed.copy(),
            function=lambda x: np.any(x),
        )

        # Expand unindexed motifs if necessary
        atom_array_expanded = self.expand_unindexed_motifs(
            atom_array,
            pop_orig_tokens=data["is_inference"],
        )

        # Provide the atom-wise mask for the regions which should be diffused into the guideposts
        # the original token was unindexed if any of the atoms where unindexed
        n_expanded_atoms = (
            atom_array_expanded.array_length() - atom_array.array_length()
        )
        mask = np.concatenate([is_unindexed_token, np.zeros(n_expanded_atoms)])
        if "ground_truth" not in data:
            data["ground_truth"] = {}
        data["ground_truth"]["is_original_unindexed_token"] = mask.astype(bool)

        # Reset global token IDs after possible padding
        atom_array_expanded = add_global_token_id_annotation(atom_array_expanded)

        # For unindexed scaffolding, we must provide an unindexing pair mask to ensure original positions aren't leaked to:
        # (I) RPE of the token initializer and (II) the atom attention base sequence mask
        mask_II, mask_I = self.create_unindexed_masks(
            atom_array_expanded, is_inference=data["is_inference"]
        )
        data["feats"]["unindexing_pair_mask"] = mask_II
        data["feats"]["is_motif_token_unindexed"] = mask_I
        data["atom_array"] = atom_array_expanded
        return data
