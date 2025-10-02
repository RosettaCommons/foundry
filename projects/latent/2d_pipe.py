"""Transform pipeline for Atom14 Design with 2D Conditioning"""

import logging

# Turn off warnings for now
# warnings.filterwarnings("ignore", category=RuntimeWarning)
# warnings.filterwarnings("ignore", category=DeprecationWarning)
import warnings
from pathlib import Path
from typing import Final

import biotite.structure as struc
import numpy as np
import torch
import torch.nn.functional as F
from beartype.typing import Any
from cifutils.constants import (
    AF3_EXCLUDED_LIGANDS,
    GAP,
    STANDARD_AA,
    STANDARD_DNA,
    STANDARD_RNA,
)
from cifutils.utils.selection import get_residue_starts
from datahub.encoding_definitions import AF3SequenceEncoding
from datahub.enums import GroundTruthConformerPolicy
from datahub.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
)
from datahub.transforms.af3_reference_molecule import (
    ELEMENT_NAME_TO_ATOMIC_NUMBER,
    _encode_atom_names_like_af3,
    get_af3_reference_molecule_features,
)
from datahub.transforms.atom_array import (
    AddGlobalAtomIdAnnotation,
    AddGlobalTokenIdAnnotation,
    AddProteinTerminiAnnotation,
    AddWithinChainInstanceResIdx,
    AddWithinPolyResIdxAnnotation,
    ComputeAtomToTokenMap,
    CopyAnnotation,
    get_within_entity_idx,
)
from datahub.transforms.atomize import (
    AtomizeByCCDName,
    FlagNonPolymersForAtomization,
)
from datahub.transforms.base import (
    AddData,
    Compose,
    ConditionalRoute,
    ConvertToTorch,
    RandomRoute,
    RemoveKeys,
    SubsetToKeys,
    Transform,
)
from datahub.transforms.bonds import AddAF3TokenBondFeatures
from datahub.transforms.cached_residue_data import LoadCachedResidueLevelData
from datahub.transforms.covalent_modifications import (
    FlagAndReassignCovalentModifications,
)
from datahub.transforms.crop import (
    CropContiguousLikeAF3,
    CropSpatialLikeAF3,
)
from datahub.transforms.diffusion.batch_structures import (
    BatchStructuresForDiffusionNoising,
)
from datahub.transforms.diffusion.edm import SampleEDMNoise
from datahub.transforms.featurize_unresolved_residues import (
    MaskPolymerResiduesWithUnresolvedFrameAtoms,
    PlaceUnresolvedTokenAtomsOnRepresentativeAtom,
    PlaceUnresolvedTokenOnClosestResolvedTokenInSequence,
)
from datahub.transforms.filters import (
    FilterToSpecifiedPNUnits,
    HandleUndesiredResTokens,
    RemoveHydrogens,
    RemoveNucleicAcidTerminalOxygen,
    RemovePolymersWithTooFewResolvedResidues,
    RemoveTerminalOxygen,
    RemoveUnresolvedLigandAtomsIfTooMany,
    RemoveUnresolvedPNUnits,
)
from datahub.utils.token import (
    apply_and_spread_token_wise,
    get_token_starts,
)

from modelhub.common import exists

# from projects.aa_design.constants import (
#     CENTRAL_ATOM,
#     MASKED_ATOM_NAME,
#     MASKED_RES_NAME,
#     VIRTUAL_ATOM_ELEMENT,
#     VIRTUAL_ATOM_NAME_PREFIX,
# )
from projects.aa_design.transforms.condition import (
    C_CRD,
    C_CTR,
    C_DIS,
    C_HOT,
    C_IDX,
    C_NTR,
    C_SEQ,
)
from projects.aa_design.transforms.condition_2d.annotator import (
    ensure_annotations,
)
from projects.aa_design.transforms.condition_2d.design_task import (
    SampleDesignTask,
    TipAtomDistanceTask,
    UnconditionalTask,
)
from projects.aa_design.transforms.condition_2d.random_atomize_residues import (
    AtomizeByMaskFunction,
    RandomAtomizeResidues,
)
from projects.aa_design.transforms.condition_2d.virtual_atoms import (
    MaskAnnotationsForTokensWithoutSequenceConditioning,
)
from projects.aa_design.transforms.conditioning_base import UnindexFlaggedTokens
from projects.aa_design.transforms.design_transforms import (
    AddGroundTruthSequence,
)
from projects.aa_design.transforms.util_transforms import (
    AggregateFeaturesLikeAF3WithoutMSA,
    RemoveTokensWithoutCorrespondingCentralAtom,
    get_af3_token_representative_masks,
)
from projects.aa_design.transforms.virtual_atoms import PadTokensWithVirtualAtoms

warnings.filterwarnings(
    "ignore", message="Category 'chem_comp_bond' not found", category=UserWarning
)
warnings.filterwarnings(
    "ignore", message="The coordinates are missing", category=UserWarning
)

# Turn DeprecationWarnings into exceptions
# warnings.filterwarnings("error", category=DeprecationWarning)

warnings.filterwarnings("ignore", message="datetime", category=DeprecationWarning)
logging.getLogger("datahub").setLevel(logging.ERROR)
logging.getLogger("cifutils").setLevel(logging.ERROR)
logging.getLogger("cifutils.tools.rdkit").setLevel(logging.ERROR)


######################################################################################
# Common transforms
######################################################################################
af3_sequence_encoding = AF3SequenceEncoding()

CENTRAL_ATOM: Final[str] = "CB"
"""Central atom name for virtual atoms."""

VIRTUAL_ATOM_ELEMENT: Final[str] = "X"
"""Virtual atom element."""

VIRTUAL_ATOM_NAME_PREFIX: Final[str] = "V"
"""Virtual atom name prefix."""

MASKED_ATOM_NAME: Final[str] = "VX"
"""The symbol to use for masked atoms."""

MASKED_RES_NAME = GAP
"""The symbal to use for masked residues
  "<G>" - Residue name used for all masked atoms (both virtual and real atoms with masked identities)
"""


def get_diffusion_transforms(
    *,
    sigma_data: float,
    diffusion_batch_size: int,
):
    return [
        ConvertToTorch(keys=["feats"]),
        # Prepare coordinates for noising (without modifying the ground truth)
        # ... add placeholder coordinates for noising
        CopyAnnotation(annotation_to_copy="coord", new_annotation="coord_to_be_noised"),
        # ... handling of unresolved residues (NOTE: best done after inputs are processed)
        PlaceUnresolvedTokenAtomsOnRepresentativeAtom(
            annotation_to_update="coord_to_be_noised"
        ),
        PlaceUnresolvedTokenOnClosestResolvedTokenInSequence(
            annotation_to_update="coord_to_be_noised",
            annotation_to_copy="coord_to_be_noised",
        ),
        # Feature aggregation
        AggregateFeaturesLikeAF3WithoutMSA(),
        # ... batching and noise sampling for diffusion
        BatchStructuresForDiffusionNoising(batch_size=diffusion_batch_size),
        SampleEDMNoise(
            sigma_data=sigma_data, diffusion_batch_size=diffusion_batch_size
        ),
    ]


######################################################################################
# Custom Transforms
######################################################################################


class EncodeAtomLevelFeaturesWithSequenceMasking(Transform):
    """
    Encodes atom-level reference features using featurization annotations with sequence masking.

    Uses featurization annotations (`element_to_featurize`, `res_name_to_featurize`, etc.) instead of
    ground truth annotations for generating reference features. This allows sequence masking to be
    handled by upstream transforms that update the featurization annotations appropriately.

    The following features are added to `data['feats']`:
        - ref_pos: Reference atom positions. (np.ndarray, shape: (n_atoms, 3), dtype: float32)
        - ref_mask: Reference atom mask. (np.ndarray, shape: (n_atoms,), dtype: bool)
        - ref_element: Reference atom element indices. (np.ndarray, shape: (n_atoms,), dtype: int64)
        - ref_charge: Reference atom charges. (np.ndarray, shape: (n_atoms,), dtype: int8)
        - ref_space_uid: Unique residue segment index. (np.ndarray, shape: (n_atoms,), dtype: int64)
        - ref_pos_is_ground_truth: Boolean indicator for whether the reference_conformer is ground truth. (np.ndarray, shape: (n_atoms,), dtype: bool)
        - motif_pos: Ground truth motif positions (if coordinate conditioned, otherwise 0s). (np.ndarray, shape: (n_atoms, 3), dtype: float32)
        - is_seq_conditioned_atom_level: Boolean indicator for sequence conditioning. (torch.Tensor, shape: (n_atoms,), dtype: bool)
        - is_dist_conditioned_atom_level: Boolean indicator for distance conditioning. (torch.Tensor, shape: (n_atoms,), dtype: bool)
        - mask_hotspot_1_atom: Boolean indicator for hotspot conditioning. (torch.Tensor, shape: (n_atoms,), dtype: bool)
        - feature_hotspot_1_atom: Boolean indicator for whether the atom is a hotspot. (torch.Tensor, shape: (n_atoms,), dtype: bool)
        - feature_distance_2_atom: Distance feature for 2D conditioning. (np.ndarray, shape: (n_atoms, n_atoms), dtype: float32)
        - mask_distance_2_atom: Mask for 2D conditioning. (np.ndarray, shape: (n_atoms, n_atoms), dtype: bool)

    Args:
        **kwargs: Additional keyword arguments passed to `get_af3_reference_molecule_features` (e.g., conformer generation timeout).
    """

    def __init__(
        self, ground_truth_conformer_policy=GroundTruthConformerPolicy.IGNORE, **kwargs
    ):
        DEFAULT_KWARGS = dict(
            conformer_generation_timeout=(3.0, 0.15),
        )
        self.conformer_generation_kwargs = DEFAULT_KWARGS | kwargs
        self.ground_truth_conformer_policy = ground_truth_conformer_policy

    def check_input(self, data: dict):
        check_contains_keys(data, ["atom_array"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        L = atom_array.array_length()  # n_atoms

        # ... Set up default reference features (all zeros)
        ref_pos = np.zeros_like(atom_array.coord, dtype=np.float32)
        ref_mask = np.zeros((L,), dtype=bool)
        ref_element = np.zeros((L,), dtype=np.int64)
        ref_charge = np.zeros((L,), dtype=np.int8)
        ref_atom_name_chars = np.zeros((L, 4), dtype=np.int8)
        ref_pos_is_ground_truth = np.zeros((L,), dtype=bool)

        # Get residue boundaries and assign unique IDs to each residue segment
        residue_starts = get_residue_starts(atom_array, add_exclusive_stop=True)
        ref_space_uid = struc.segments.spread_segment_wise(
            residue_starts, np.arange(len(residue_starts) - 1, dtype=np.int64)
        )

        # ... Create reference features for sequence-conditioned atoms
        is_seq_conditioned = C_SEQ.mask(atom_array, default="generate")
        if np.any(is_seq_conditioned):
            # We need an atom array with ground-truth atom names for reference conformer generation and the ground truth conformer policy
            atom_array_with_gt_atom_name = atom_array.copy()
            atom_array_with_gt_atom_name.atom_name = (
                atom_array.gt_atom_name
            )  # Set in PadTokensWithVirtualAtoms
            atom_array_with_gt_atom_name.set_annotation(
                "ground_truth_conformer_policy",
                np.full(
                    atom_array_with_gt_atom_name.array_length(),
                    self.ground_truth_conformer_policy.value,
                ),
            )

            # If we are showing the model the sequence, we must generate conformers
            reference_features, _ = get_af3_reference_molecule_features(
                atom_array_with_gt_atom_name[is_seq_conditioned],
                cached_residue_level_data=data["cached_residue_level_data"]
                if "cached_residue_level_data" in data
                else None,
                **self.conformer_generation_kwargs,
            )  # (n_atoms_with_seq_conditioning, n_features)

            # ... overwrite reference features for atoms with sequence conditioning
            ref_pos[is_seq_conditioned] = reference_features["ref_pos"]
            ref_mask[is_seq_conditioned] = reference_features["ref_mask"]
            ref_charge[is_seq_conditioned] = reference_features["ref_charge"]
            ref_atom_name_chars[is_seq_conditioned] = reference_features[
                "ref_atom_name_chars"
            ]
            ref_pos_is_ground_truth[is_seq_conditioned] = reference_features[
                "ref_pos_is_ground_truth"
            ]

        # ... show element for all 'real' backbone atoms & any non-standard AA's
        ref_element = np.array(
            [
                ELEMENT_NAME_TO_ATOMIC_NUMBER.get(a, 0)
                for a in atom_array.element_to_featurize
            ]
        )
        ref_atom_name_chars = _encode_atom_names_like_af3(
            atom_array.atom_name_to_featurize
        )
        # ensure_annotations(atom_array, "is_protein_backbone", "is_standard_aa")
        # is_element_shown = atom_array.mask("(~is_virtual & is_protein_backbone) | ~is_standard_aa")
        # ref_element[is_element_shown] = atom_array.atomic_number[is_element_shown]

        # 2D Features
        mask_dist_2d = C_DIS.mask(atom_array, default="generate").as_dense_array()
        feature_dist_2d = C_DIS.annotation(
            atom_array, default="generate"
        ).as_dense_array()
        feature_dist_2d[~mask_dist_2d] = 0.0

        reference_features = {
            # ... standard AF3 `ref` features
            "ref_pos": ref_pos,  # (n_atoms, 3)
            "ref_mask": ref_mask,  # (n_atoms)
            "ref_element": ref_element,  # (n_atoms)
            "ref_charge": ref_charge,  # (n_atoms)
            "ref_space_uid": ref_space_uid,  # (n_atoms)
            "ref_atom_name_chars": ref_atom_name_chars,  # (n_atoms, 4, 64)
            "ref_pos_is_ground_truth": ref_pos_is_ground_truth,  # (n_atoms)
            "motif_pos": np.nan_to_num(
                C_CRD.annotation(atom_array, default="generate")
            ),  # (n_atoms, 3)
            # ... extra condition features
            C_SEQ.get_mask_name(1, "atom"): C_SEQ.mask(
                atom_array, default="generate"
            ),  # (n_atoms)
            # TODO(Discuss w. Nate): Do we still need this?
            C_DIS.get_mask_name(1, "atom"): C_DIS.mask(atom_array, default="generate")
            .as_dense_array(default=False)
            .any(axis=0),  # (n_atoms)
            # TODO(Discuss w. Max): Do we need both?
            C_HOT.get_mask_name(1, "atom"): C_HOT.mask(
                atom_array, default="generate"
            ),  # (n_atoms)
            C_HOT.get_feature_name(1, "atom"): C_HOT.annotation(
                atom_array, default="generate"
            ),  # (n_atoms)
            # 2D Features
            C_DIS.get_feature_name(2, "atom"): feature_dist_2d,  # (n_atoms, n_atoms)
            C_DIS.get_mask_name(2, "atom"): mask_dist_2d,  # (n_atoms, n_atoms)
        }
        # Verify all features have n_atoms as first dimension
        assert all(
            v.shape[0] == L for v in reference_features.values()
        ), "All features must have n_atoms as first dimension"

        # Sanity Check: All features are null for unconditioned atoms
        idx_unconditioned = ~is_seq_conditioned
        assert np.allclose(
            ref_pos[idx_unconditioned], np.zeros_like(ref_pos[idx_unconditioned])
        ), "ref_pos not null for unconditioned atoms"
        assert np.all(
            ~ref_mask[idx_unconditioned]
        ), "ref_mask not null for unconditioned atoms"
        # assert np.all(
        #     ref_element[idx_unconditioned] == 0
        # ), "ref_element not null for unconditioned atoms"
        assert np.all(
            ref_charge[idx_unconditioned] == 0
        ), "ref_charge not null for unconditioned atoms"

        if "feats" not in data:
            data["feats"] = {}

        data["feats"].update(reference_features)

        return data


class HackInRequiredAnnotations(Transform):
    """
    Hack to add in some annotations that UnindexFlaggedTokens should have added but didn't.
    This functionality is taken from AddIsX, which already implements this but later in the pipeline.
    Required annotations are:
        "is_motif_atom",
        "is_motif_token",
        "is_motif_atom_unindexed",
        "is_motif_atom_unindexed_motif_breakpoint",
        "is_motif_atom_with_fixed_seq",
        "is_motif_atom_with_fixed_coord",
        "is_flexible_motif_atom",

    NOTE: In the future we want to rework UnindexFlaggedTokens to not need this hack.
    """

    def check_input(self, data):
        check_contains_keys(data, ["atom_array"])

    def forward(self, data: dict) -> dict:
        # TODO: We should be storing in the ground truth key, not feats, unless it is literally a feature for the model
        # (vs. info that we use for losses / metrics)

        atom_array = data["atom_array"]

        ########## HACK: SPOOF LEGACY MOTIF FEATURES ##########
        # --- alias table --
        a = atom_array
        # ------------------

        from datahub.utils.token import (
            apply_token_wise,
            get_token_starts,
            spread_token_wise,
        )

        token_starts = get_token_starts(a)
        token_segments = np.concatenate([token_starts, [a.array_length()]])

        # `is_motif_atom`
        # ... get relevant atom-level annotations
        has_coord = C_CRD.mask(a, default="generate")
        has_dist = C_DIS.mask(a, default="generate").as_dense_array(False).any(axis=0)
        has_idx = C_IDX.mask(a, default="generate")
        has_seq = C_SEQ.mask(a, default="generate")
        is_motif_atom = has_coord | has_dist
        a.set_annotation("is_motif_atom", is_motif_atom)
        a.set_annotation("is_motif_atom_with_fixed_coord", has_coord)
        # NOTE: You would think this should be is_motif_atom & has_seq, but based on PadTokensWithVirtualAtoms
        # it really is just has_seq.
        a.set_annotation("is_motif_atom_with_fixed_seq", has_seq)
        a.set_annotation(
            "is_flexible_motif_atom",
            has_seq & ~is_motif_atom & np.isin(a.atom_name, ["N", "CA", "C", "O"]),
        )  # TODO: Confirm this is desired behavior
        a.set_annotation("is_motif_atom_unindexed", is_motif_atom & ~has_idx)
        a.set_annotation(
            "is_motif_atom_unindexed_motif_breakpoint", np.zeros_like(is_motif_atom)
        )  # FIXME: Hack for now but will not work when we try unindexing motifs!

        # `is_motif_token`
        # ... get relevant token-level annotations
        _to_token_lvl = lambda x: apply_token_wise(  # noqa: E731
            a, x, np.any, token_starts=token_segments
        )
        token_has_coord = _to_token_lvl(has_coord)
        token_has_dist = _to_token_lvl(has_dist)
        # ... combine them to create legacy features
        is_motif_token = token_has_coord | token_has_dist
        a.set_annotation(
            "is_motif_token", spread_token_wise(a, is_motif_token, token_segments)
        )

        data["atom_array"] = a
        return data


class AddIsX(Transform):
    def __init__(
        self,
        X=[
            "is_backbone",  # ... part of the protein backbone (N, CA, C, O)
            "is_sidechain",  # ... part of the protein sidechain (all atoms except N, CA, C, O, OXT)
            "is_virtual",  # ... virtual atoms that do not exist in the ground truth
            "is_central",  # ... token representative atom (CA)
            "is_ca",  # ... true CA atoms
            "is_masked",  # ... atoms that are masked out (i.e. appear virtual to the model but exist in the ground truth)
        ],
        central_atom=CENTRAL_ATOM,
        virtual_atom_element=VIRTUAL_ATOM_ELEMENT,
    ):
        self.X = X
        self.central_atom = central_atom
        self.virtual_atom_element = virtual_atom_element

    def check_input(self, data):
        check_contains_keys(data, ["atom_array", "feats"])

    def forward(self, data: dict) -> dict:
        # TODO: We should be storing in the ground truth key, not feats, unless it is literally a feature for the model
        # (vs. info that we use for losses / metrics)

        atom_array = data["atom_array"]
        # ... Add backbone and sidechain annotations
        ensure_annotations(
            atom_array,
            # "is_protein",
            "is_protein_backbone",
            "is_protein_sidechain",
        )

        _token_rep_mask = get_af3_token_representative_masks(
            atom_array, central_atom=self.central_atom
        )

        # Initialize ground_truth dict if it doesn't exist
        if "feats" not in data:
            data["feats"] = {}

        # ... Basic features
        if "is_backbone" in self.X:
            is_backbone = atom_array.get_annotation("is_protein_backbone")
            data["feats"]["is_backbone"] = torch.from_numpy(is_backbone).to(
                dtype=torch.bool
            )

        if "is_sidechain" in self.X:
            is_sidechain = atom_array.get_annotation("is_protein_sidechain")
            data["feats"]["is_sidechain"] = torch.from_numpy(is_sidechain).to(
                dtype=torch.bool
            )

        # Virtual atom feats
        if "is_virtual" in self.X:
            data["feats"]["is_virtual"] = (
                atom_array.element == self.virtual_atom_element
            )

        if "is_masked" in self.X:
            data["feats"]["is_masked"] = (
                atom_array.element_to_featurize == self.virtual_atom_element
            )

        # ... Central
        if "is_central" in self.X:
            data["feats"]["is_central"] = _token_rep_mask

        # NOTE: Check end of function for is_ca. Need to do it then because for now we are relying on some of the spoofed legacy features

        # Set occupancy feature
        if data.get(
            "is_inference", False
        ):  # HACK: Pretend all occupancy is 1.0 during inference
            data["feats"]["has_zero_occupancy"] = np.zeros_like(
                atom_array.occupancy, dtype=bool
            )
        else:
            data["feats"]["has_zero_occupancy"] = atom_array.occupancy == 0.0

        ########## HACK: SPOOF LEGACY MOTIF FEATURES ##########
        # --- alias table --
        f = data["feats"]
        a = atom_array
        from_np = torch.from_numpy
        # ------------------
        # TODO: Investigate this

        from datahub.utils.token import (
            apply_token_wise,
            get_token_starts,
            spread_token_wise,
        )

        n_atoms = atom_array.array_length()
        token_starts = get_token_starts(a)
        token_segments = np.concatenate([token_starts, [a.array_length()]])
        n_tokens = len(token_starts)

        # `is_motif_atom`
        # ... get relevant atom-level annotations
        has_coord = C_CRD.mask(a, default="generate")
        has_dist = C_DIS.mask(a, default="generate").as_dense_array(False).any(axis=0)
        has_idx = C_IDX.mask(a, default="generate")
        has_seq = C_SEQ.mask(a, default="generate")
        is_motif_atom = has_coord | has_dist
        a.set_annotation("is_motif_atom", is_motif_atom)
        a.set_annotation("is_motif_atom_with_fixed_coord", has_coord)
        # NOTE: You would think this should be is_motif_atom & has_seq, but based on PadTokensWithVirtualAtoms
        # it really is just has_seq.
        a.set_annotation("is_motif_atom_with_fixed_seq", has_seq)
        a.set_annotation(
            "is_flexible_motif_atom",
            has_seq & ~is_motif_atom & np.isin(a.atom_name, ["N", "CA", "C", "O"]),
        )  # TODO: Confirm this is desired behavior
        a.set_annotation("is_motif_atom_unindexed", is_motif_atom & ~has_idx)
        a.set_annotation(
            "is_motif_atom_unindexed_motif_breakpoint", np.zeros_like(is_motif_atom)
        )  # FIXME: Hack for now but will not work when we try unindexing motifs!
        # ... combine them to create legacy features (L, )
        f["is_motif_atom"] = from_np(a.is_motif_atom)
        f["is_motif_atom_with_fixed_coord"] = from_np(a.is_motif_atom_with_fixed_coord)
        f["ref_is_motif_atom_with_fixed_coord"] = from_np(
            a.is_motif_atom_with_fixed_coord
        )
        f["is_flexible_motif_atom"] = from_np(a.is_flexible_motif_atom)
        f["is_motif_atom_with_fixed_seq"] = from_np(a.is_motif_atom_with_fixed_seq)
        f["is_motif_atom_unindexed"] = from_np(a.is_motif_atom_unindexed)  # (L,)
        f["ref_is_motif_atom_unindexed"] = from_np(a.is_motif_atom_unindexed)  # (L,)
        # TODO: What is the difference between `is_motif_atom` and `ref_is_motif_atom`?
        # [1, 0] = non-motif, [0, 1] = motif
        ref_is_motif_atom = np.zeros((n_atoms, 2), dtype=bool)
        ref_is_motif_atom[~is_motif_atom, 0] = True
        ref_is_motif_atom[is_motif_atom, 1] = True
        f["ref_is_motif_atom"] = from_np(ref_is_motif_atom)  # (L, 2)
        # [1, 0, 0] = non-motif; [0, 1, 0] = indexed motif; [0, 0, 1] = unindexed motif
        ref_motif_atom_type = np.zeros((n_atoms, 3), dtype=bool)
        ref_motif_atom_type[~is_motif_atom, 0] = True
        ref_motif_atom_type[is_motif_atom & has_idx, 1] = True
        ref_motif_atom_type[is_motif_atom & ~has_idx, 2] = True
        f["ref_motif_atom_type"] = from_np(ref_motif_atom_type)  # (L, 3)

        # `is_motif_token`
        # ... get relevant token-level annotations
        _to_token_lvl = lambda x: apply_token_wise(  # noqa: E731
            a, x, np.any, token_starts=token_segments
        )
        token_has_coord = _to_token_lvl(has_coord)
        token_has_dist = _to_token_lvl(has_dist)
        token_has_idx = _to_token_lvl(has_idx)
        # ... combine them to create legacy features
        is_motif_token = token_has_coord | token_has_dist
        a.set_annotation(
            "is_motif_token", spread_token_wise(a, is_motif_token, token_segments)
        )
        f["is_motif_token"] = from_np(is_motif_token)  # (I,)
        # TODO: What is the difference between `is_motif_token` and `ref_is_motif_token`?  NOTE: Max - ref_is_motif_token is one-hot encoded so it has dimension 2. However I don't think it's actually used anywhere by the model.
        # Token-level motif indicator: [1, 0] = non-motif, [0, 1] = motif
        ref_is_motif_token = np.zeros((n_tokens, 2), dtype=bool)
        ref_is_motif_token[~is_motif_token, 0] = True
        ref_is_motif_token[is_motif_token, 1] = True
        f["ref_is_motif_token"] = from_np(ref_is_motif_token)  # (I, 2)
        # [1, 0, 0] = non-motif; [0, 1, 0] = indexed motif; [0, 0, 1] = unindexed motif
        ref_motif_token_type = np.zeros((n_tokens, 3), dtype=bool)
        ref_motif_token_type[~is_motif_token, 0] = True
        ref_motif_token_type[is_motif_token & token_has_idx, 1] = True
        ref_motif_token_type[is_motif_token & ~token_has_idx, 2] = True
        f["ref_motif_token_type"] = from_np(ref_motif_token_type)  # (I, 3)
        f["is_motif_token_with_fully_fixed_coord"] = apply_token_wise(
            a, has_coord, np.all, token_starts=token_segments
        )

        # TODO: What is the difference between this and is_motif_atom?
        f["ref_is_motif_atom_mask"] = is_motif_atom

        # ... CA
        if "is_ca" in self.X:
            # NOTE from Max: This seems to be the fix to the glycine bug -- use CA as your central atom instead of CB for certain tasks.
            # This feature is called `is_ca` but it really should probably be called `is_central_atom_if_central_atom_was_ca`.
            # Basically we sometimes want to use the central atom, but can't use the real central atom (usually CB) because it will leak the glycine's identity.
            # So instead we use `is_ca` in those spots which means it not only needs to mark CA atoms but also all central atoms (for ligands and such)

            # Split into components to handle separately
            atom_array_indexed = atom_array[~atom_array.is_motif_atom_unindexed]
            _token_rep_mask_indexed = get_af3_token_representative_masks(
                atom_array_indexed, central_atom="CA"
            )
            if atom_array.is_motif_atom_unindexed.any():
                atom_array_unindexed = atom_array[atom_array.is_motif_atom_unindexed]

                # Ensure is_ca represents one and the first atom only for unindexed tokens
                def first_nonzero(n):
                    assert n > 0
                    x = np.zeros(n, dtype=bool)
                    x[0] = 1
                    return x

                starts = get_token_starts(atom_array_unindexed, add_exclusive_stop=True)
                _token_rep_mask_unindexed = np.concatenate(
                    [
                        first_nonzero(end - start)
                        for start, end in zip(starts[:-1], starts[1:])
                    ]
                )
                _token_rep_mask = np.concatenate(
                    [
                        _token_rep_mask_indexed,
                        _token_rep_mask_unindexed,
                    ],
                    axis=0,
                )
            else:
                _token_rep_mask = _token_rep_mask_indexed
            data["feats"]["is_ca"] = _token_rep_mask

        return data


class EncodeTokenLevelFeaturesWithSequenceMasking(Transform):
    """
    Encodes token-level model features using specified featurization annotations.

    Uses `res_name_to_featurize` instead of `res_name` for encoding residue types.
    This allows for sequence masking to be handled by upstream transforms that update the featurization annotations appropriately.

    Computes and stores the following token-level features in the `data['feats']` dictionary:
        - `residue_index`: Index of the residue within its chain (int, shape: (N_tokens,))
        - `token_index`: Index of the token in the sequence (int, shape: (N_tokens,))
        - `asym_id`: Unique integer for each distinct chain instance (int, shape: (N_tokens,))
        - `entity_id`: Unique integer for each distinct sequence entity (int, shape: (N_tokens,))
        - `sym_id`: Unique integer within chains of the same sequence (int, shape: (N_tokens,))
        - `restype`: One-hot encoding of the residue type (float, shape: (N_tokens, n_tokens)), using featurization annotations
        - `is_protein`, `is_rna`, `is_dna`, `is_ligand`: Boolean masks for molecule type (bool, shape: (N_tokens,))

    Metadata for chain and entity names is stored in `data['feat_metadata']`.

    Args:
        sequence_encoding (AF3SequenceEncoding):
            An encoding object that provides methods for mapping residue names to AF3 token indices and one-hot encodings.
    """

    def __init__(
        self,
        sequence_encoding: AF3SequenceEncoding,
    ):
        self.sequence_encoding = sequence_encoding

    def check_input(self, data: dict[str, Any]) -> None:
        check_atom_array_annotation(
            data,
            [
                "atomize",
                "pn_unit_iid",
                "chain_entity",
                "res_name",
                "res_name_to_featurize",
                "within_chain_res_idx",
            ],
        )

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array = data["atom_array"]

        # ... get token-level array
        token_starts = get_token_starts(atom_array)
        n_tokens = len(token_starts)
        token_level_array = atom_array[token_starts]

        # ... identifier tokens
        # ... (residue)
        residue_index = token_level_array.within_chain_res_idx
        # ... (token)
        token_index = np.arange(len(token_starts))
        # ... (chain instance)
        asym_name, asym_id = np.unique(
            token_level_array.pn_unit_iid, return_inverse=True
        )
        # ... (chain entity)
        entity_name, entity_id = np.unique(
            token_level_array.pn_unit_entity, return_inverse=True
        )
        # ... (within chain entity)
        sym_name, sym_id = get_within_entity_idx(token_level_array, level="pn_unit")

        # ... molecule type (protein, RNA, DNA, ligand) - use ground truth res_name
        _aa_like_res_names = self.sequence_encoding.all_res_names[
            self.sequence_encoding.is_aa_like
        ]
        is_protein = np.isin(token_level_array.res_name, _aa_like_res_names)

        _rna_like_res_names = self.sequence_encoding.all_res_names[
            self.sequence_encoding.is_rna_like
        ]
        is_rna = np.isin(token_level_array.res_name, _rna_like_res_names)

        _dna_like_res_names = self.sequence_encoding.all_res_names[
            self.sequence_encoding.is_dna_like
        ]
        is_dna = np.isin(token_level_array.res_name, _dna_like_res_names)

        is_ligand = ~(is_protein | is_rna | is_dna)

        # ... sequence tokens - use featurization annotations
        res_names_to_featurize = token_level_array.res_name_to_featurize

        restype = self.sequence_encoding.encode(res_names_to_featurize)
        restype = F.one_hot(
            torch.tensor(restype), num_classes=self.sequence_encoding.n_tokens
        )

        # Indicator variables for conditioning (atom-level, but indicates token-level conditioning)
        is_dist_conditioned_atom_level = np.any(
            C_DIS.mask(atom_array, 2, "atom", default="generate").as_dense_array(),
            axis=1,
        )
        token_segments = np.concatenate([token_starts, [atom_array.array_length()]])
        is_dist_conditioned_token_level = torch.from_numpy(
            apply_and_spread_token_wise(
                atom_array,
                is_dist_conditioned_atom_level,
                np.any,
                token_starts=token_segments,
            )
        )[token_starts]  # (L,)
        is_seq_conditioned_token_level = torch.from_numpy(
            C_SEQ.mask(token_level_array, default="generate")
        )  # (L,)

        # ... add terminus type # TODO: Turn into proper `Condition`
        terminus_type = torch.zeros((n_tokens, 2), dtype=torch.long)
        is_c_terminus = C_CTR.mask(token_level_array, default="raise")
        is_n_terminus = C_NTR.mask(token_level_array, default="raise")
        terminus_type[is_c_terminus, 0] = 1
        terminus_type[is_n_terminus, 1] = 1

        # ... add to data dict
        if "feats" not in data:
            data["feats"] = {}
        if "feat_metadata" not in data:
            data["feat_metadata"] = {}

        # Build dictionary of features
        new_feats = {
            "residue_index": residue_index,  # (N_tokens) (int)
            "token_index": token_index,  # (N_tokens) (int)
            "asym_id": asym_id,  # (N_tokens) (int)
            "entity_id": entity_id,  # (N_tokens) (int)
            "sym_id": sym_id,  # (N_tokens) (int)
            "restype": restype,  # (N_tokens, 32) (float, one-hot) (using featurization annotations)
            "is_protein": is_protein,  # (N_tokens) (bool)
            "is_rna": is_rna,  # (N_tokens) (bool)
            "is_dna": is_dna,  # (N_tokens) (bool)
            "is_ligand": is_ligand,  # (N_tokens) (bool)
            "terminus_type": terminus_type,  # (N_tokens, 2) (int)
            C_DIS.get_mask_name(
                1, "token"
            ): is_dist_conditioned_token_level,  # (N_tokens,) (bool)
            C_SEQ.get_mask_name(
                1, "token"
            ): is_seq_conditioned_token_level,  # (N_tokens,) (bool)
        }

        # Assert all features have matching first dimension
        n_tokens = len(residue_index)
        for key, value in new_feats.items():
            assert (
                value.shape[0] == n_tokens
            ), f"{key} has first dim {value.shape[0]} but expected {n_tokens}!"

        # Merge into data dict
        data["feats"] |= new_feats

        # Maps from numerical indices to string names (returned from np.unique with return_inverse=True)
        # (May be helpful for debugging)
        data["feat_metadata"] |= {
            "asym_name": asym_name,  # (N_asyms)
            "entity_name": entity_name,  # (N_entities)
            "sym_name": sym_name,  # (N_entities)
        }

        return data


class RemoveCenterOfMass(Transform):
    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        center_of_mass = atom_array.coord[atom_array.mask("~has_nan_coord()")].mean(
            axis=0
        )
        atom_array.coord -= center_of_mass
        data["atom_array"] = atom_array
        return data


class JitterCenterOfMass(Transform):
    def __init__(self, jitter_sigma: float = 8.0):
        self.jitter_sigma = jitter_sigma

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        atom_array.coord += np.random.normal(0, self.jitter_sigma, (3,))
        data["atom_array"] = atom_array
        return data


######################################################################################
# Pipelines
######################################################################################
class _ListBuilder:
    """A small convenience class to build lists element by element with the '+=' operator"""

    def __init__(self):
        self.list = []

    def __add__(self, other):
        if isinstance(other, list):
            self.list.extend(other)
        elif isinstance(other, _ListBuilder):
            self.list.extend(other.list)
        else:
            self.list.append(other)
        return self

    def tolist(self):
        return self.list


_DEFAULT_DESIGN_TASKS = {
    "unconditional": {
        "transform": UnconditionalTask(),
        "frequency": 0.01,
    },
    "tip_atom_distance": {
        "transform": TipAtomDistanceTask(
            min_residues=2,
            max_residues=20,
            p_tip_atom=0.8,
            knockout_p=0.8,
            dropout_min_fraction=0.1,
            dropout_max_fraction=0.9,
        ),
        "frequency": 1.0,
    },
}
"""A dummy list of examplary design tasks to provide a discoverable interface for the design task sampling system.

For actual training runs this should be set via the hydra config.
"""


def _get_design_task_name(data: dict) -> str:
    """Get the design task name from the data dict.
    NOTE: This is implemented as a global function to allow pickle-ing for multiprocessing.
    """
    return data["task"]["name"]


def _build_rfd3_train_pipeline(
    crop_size: int | None = None,
    crop_contiguous_probability: float = 0.5,
    crop_spatial_probability: float = 0.5,
    crop_center_cutoff_distance: float = 15.0,
    max_atoms_in_crop: int | None = None,
    is_inference: bool = False,
    p_atomize_residues: float = 0.0,
    design_tasks: dict[str, dict[str, Any]] = _DEFAULT_DESIGN_TASKS,
    undesired_res_names: list[str] = AF3_EXCLUDED_LIGANDS,
    central_atom: str = CENTRAL_ATOM,
    **kwargs,  # to dump remaining args for now
) -> list[Transform]:
    # TODO: Step 0 (before picking example): Sample problem type
    # T = transform shorthand for readability
    T = _ListBuilder()

    # (We may want to run the train pipe with is_inference=True; e.g., for simple validations of our training objective)
    T += AddData({"is_inference": is_inference})
    ###########################################################
    ### Step 1: Filter unwanted information                 ###
    ###########################################################
    # ... cleanup
    T += RemoveKeys(["atom_array_stack"], require_keys_exist=False)
    T += RemoveHydrogens()
    T += FilterToSpecifiedPNUnits(
        extra_info_key_with_pn_unit_iids_to_keep="all_pn_unit_iids_after_processing"
    )
    T += RemoveTerminalOxygen()
    T += RemoveNucleicAcidTerminalOxygen()
    T += RemoveUnresolvedPNUnits()
    T += HandleUndesiredResTokens(
        undesired_res_tokens=undesired_res_names
    )  # e.g., non-standard residues
    T += RemovePolymersWithTooFewResolvedResidues(min_residues=1)
    T += MaskPolymerResiduesWithUnresolvedFrameAtoms()
    T += RemoveUnresolvedLigandAtomsIfTooMany(unresolved_ligand_atom_limit=5)
    T += RemoveTokensWithoutCorrespondingCentralAtom(central_atom=central_atom)

    # ... add basic annotations
    T += AddGlobalAtomIdAnnotation()
    T += AddWithinChainInstanceResIdx()
    T += AddWithinPolyResIdxAnnotation()
    T += AddProteinTerminiAnnotation()  # Also handled by conditions now?
    T += FlagAndReassignCovalentModifications()
    T += FlagNonPolymersForAtomization()
    T += RandomAtomizeResidues(p_atomize=p_atomize_residues)
    T += AtomizeByCCDName(
        atomize_by_default=True,
        res_names_to_ignore=STANDARD_AA + STANDARD_RNA + STANDARD_DNA,
        move_atomized_part_to_end=False,
        validate_atomize=False,
    )

    # ... crop
    if crop_size and (crop_contiguous_probability > 0 or crop_spatial_probability > 0):
        assert crop_size > 0, "Crop size must be greater than 0"
        assert (
            crop_center_cutoff_distance > 0
        ), "Crop center cutoff distance must be greater than 0"

        T += RandomRoute.from_list(
            [
                (
                    crop_contiguous_probability,  # DISCUSS: Does this even make sense for design??
                    CropContiguousLikeAF3(
                        crop_size=crop_size,
                        keep_uncropped_atom_array=False,
                        max_atoms_in_crop=max_atoms_in_crop,
                        annotate_crop_boundary=True,
                    ),
                ),
                (
                    crop_spatial_probability,
                    CropSpatialLikeAF3(
                        crop_size=crop_size,
                        crop_center_cutoff_distance=crop_center_cutoff_distance,
                        keep_uncropped_atom_array=False,
                        max_atoms_in_crop=max_atoms_in_crop,
                        raise_if_missing_query=False,
                        annotate_crop_boundary=True,
                    ),
                ),
            ]
        )

    # ... remove center of mass & add jitter information to not leak the crop center
    # TODO: Switch for CenterRandomAugmentation
    T += RemoveCenterOfMass()
    T += JitterCenterOfMass(jitter_sigma=5.0)

    ###########################################################
    ### Step 2: Build design task                           ###
    ###########################################################
    # Design task sampling: Sample and apply design tasks
    #
    # This section implements a flexible design task sampling system where different protein design
    # objectives can be conditionally applied based on the input data. The system works as follows:
    # 1. SampleDesignTask evaluates all available design tasks and samples one based on their
    #    frequencies among eligible tasks (those that can be applied to the current data)
    # 2. Each design task implements can_apply() to determine applicability and
    #    annotate_for_task_selection() to add required annotations
    # 3. The sampled task is then applied via ConditionalRoute to generate the appropriate
    #    design problem configuration
    # 4. Possible `DesignTaskModifier` functions can then be applied afterwards to modify the task
    #    These exist to reduce the overhead of writing full-fledged new design tasks by simply allowing
    #    to change the condition annotation values (e.g. `rasa`, `sequence`, `distance`) for a given,
    #    existing task.
    T += SampleDesignTask(design_tasks=design_tasks)

    # ... create the design task
    T += ConditionalRoute(
        condition_func=_get_design_task_name,
        transform_map={name: task["transform"] for name, task in design_tasks.items()},
    )

    # ... (optionally) modify the design task post-hoc
    # TODO: Template this

    return T.tolist()


def _build_rfd3_featurize_pipeline(
    n_atoms_per_token: int = 14,
    central_atom: str = CENTRAL_ATOM,
    sigma_data: float = 16.0,
    diffusion_batch_size: int = 32,
    return_atom_array: bool = False,
    residue_cache_dir: str | None = None,
    association_scheme: str | None = None,
) -> list[Transform]:
    T = _ListBuilder()

    if exists(residue_cache_dir):
        T += LoadCachedResidueLevelData(
            dir=Path(residue_cache_dir),
            sharding_depth=1,
        )

    # MISSING TRANSFORMS FROM OTHER PIPELINE (Besides all the ones that would be replaced by design tasks):
    # AddIsDAminoAcidFeat() and RandomlyMirrorInputs()
    # MotifCenterRandomAugmentation()
    # AugmentNoise()

    ######################################################################################
    # Virtual Atoms and Masked Atoms
    ######################################################################################
    # ... Annotate finalized token ids
    # TODO: Add terminus_type
    T += AddGlobalTokenIdAnnotation()

    # ... Copy annotations to create featurization annotations
    T += CopyAnnotation(
        annotation_to_copy="res_name", new_annotation="res_name_to_featurize"
    )
    T += CopyAnnotation(
        annotation_to_copy="atom_name", new_annotation="atom_name_to_featurize"
    )
    T += CopyAnnotation(
        annotation_to_copy="element", new_annotation="element_to_featurize"
    )
    T += HackInRequiredAnnotations()  # UnindexFlaggedTokens needs these annotations
    # ... unindex flagged tokens. In this pipeline, we don't have any unindexed tokens.
    T += UnindexFlaggedTokens(
        central_atom=central_atom
    )  # TODO: Fix the hacks in this function to actually work in this framework!!!
    # ... add virtual atoms to protein residues (WITHOUT sequence conditioning)
    # (Since if we know the sequence, we know exactly how many atoms there are — no need for virtual atoms)
    T += PadTokensWithVirtualAtoms(  # Use this over AddVirtualAtoms for now to handle the naming permutations
        n_atoms_per_token=n_atoms_per_token,
        atom_to_pad_from=central_atom,
        association_scheme=association_scheme,
    )
    # ... Create masked atoms: mask chemical identities for all atoms in tokens without sequence conditioning
    # This includes both virtual atoms (is_virtual=True) and real atoms (is_virtual=False)
    T += MaskAnnotationsForTokensWithoutSequenceConditioning(
        masked_atom_element=VIRTUAL_ATOM_ELEMENT,
        masked_atom_name=MASKED_ATOM_NAME,
        masked_res_name=MASKED_RES_NAME,  # For featurization masking
        res_name_annotation_to_featurize="res_name_to_featurize",
        atom_name_annotation_to_featurize="atom_name_to_featurize",
        element_annotation_to_featurize="element_to_featurize",
    )

    ######################################################################################
    # Featurize all conditions
    ######################################################################################
    # ... Compute atom-to-token mapping after token structure is finalized (we have added virtual atoms already)
    T += ComputeAtomToTokenMap()
    # ... AF3 token level-encoding, with sequence masking when we don't have sequence
    T += EncodeTokenLevelFeaturesWithSequenceMasking(
        sequence_encoding=af3_sequence_encoding,
    )
    # ... Atom-level reference features
    T += EncodeAtomLevelFeaturesWithSequenceMasking()
    # ... Bonds
    T += AddAF3TokenBondFeatures()
    # ... Add useful features for losses / metrics
    # (We add to ground truth, not feats, to distinguish from features that we show the model vs. features that we use for losses and metrics)
    T += AddIsX(
        X=[
            # Basic
            "is_backbone",
            "is_sidechain",
            # Virtual atom
            "is_masked",
            "is_virtual",
            "is_central",
            "is_ca",
        ],
        central_atom=central_atom,
    )
    T += AddGroundTruthSequence(sequence_encoding=af3_sequence_encoding)
    # EDM-style wrap-up  (no additional features added at this point)
    T += get_diffusion_transforms(
        sigma_data=sigma_data, diffusion_batch_size=diffusion_batch_size
    )

    # Subset to necessary keys only
    keys_to_keep = [
        "example_id",
        "feats",
        "t",
        "noise",
        "ground_truth",
        "coord_atom_lvl_to_be_noised",
        "symmetry_resolution",
        "extra_info",
        "task",
    ]

    if return_atom_array:
        keys_to_keep.append("atom_array")

    T += SubsetToKeys(keys_to_keep)

    return T.tolist()


class AnnotateConditionsForInference(Transform):
    """Annotate conditions for inference."""

    def check_input(self, data: dict) -> None:
        assert data.get(
            "is_inference", False
        ), "This transform should only be run in inference mode."

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        # TODO: Revisit these once the `sampling` of lengths/contigs is implemented.
        # Explicitly set conditions whose defaults may change if the atom-array is in-context conditioned
        # (e.g. when running `unindexed`)
        # ... terminus type conditioning
        is_n_terminus = C_NTR.mask(atom_array, default="generate")
        is_c_terminus = C_CTR.mask(atom_array, default="generate")

        C_NTR.set_mask(atom_array, is_n_terminus)
        C_CTR.set_mask(atom_array, is_c_terminus)

        return data


def _build_rfd3_validation_pipeline(
    atomize_distance_conditioned_tokens: bool = False,
    **kwargs,  # to dump remaining args for now
) -> list[Transform]:
    # T = transform shorthand for readability
    # ... initialize
    T = _ListBuilder()
    T += AddData({"is_inference": True})
    T += RemoveTerminalOxygen()

    ###########################################################
    ### Step 1: Add potentially missing information         ###
    ###########################################################
    # TODO: This needs to sample a set of residues inbetween & at the end of our motif
    #       (essentially the same as contigs in RFD)
    # T += SampleSequenceLength()
    T += AddGlobalAtomIdAnnotation(allow_overwrite=True)
    T += AddWithinChainInstanceResIdx()
    T += AtomizeByCCDName(
        atomize_by_default=True,
        res_names_to_ignore=STANDARD_AA + STANDARD_RNA + STANDARD_DNA,
    )
    if atomize_distance_conditioned_tokens:
        T += AtomizeByMaskFunction(
            mask_function=lambda x: C_DIS.mask(x, default="generate")
            .as_dense_array(default=False)
            .any(axis=0),
        )
    T += AnnotateConditionsForInference()
    return T.tolist()


def build_rfd3_train_pipeline(
    # ... training specific
    crop_size: int | None = None,
    crop_contiguous_probability: float = 0.5,
    crop_spatial_probability: float = 0.5,
    crop_center_cutoff_distance: float = 15,
    p_atomize_residues: float = 0.0,
    # ... design task specific
    design_tasks: dict[str, dict] = _DEFAULT_DESIGN_TASKS,
    # ... featurization specific
    sigma_data: float = 0.5,
    diffusion_batch_size: int = 16,
    n_atoms_per_token: int = 14,
    central_atom: str = "CB",
    return_atom_array: bool = False,
    is_inference: bool = False,
    residue_cache_dir: str | None = None,
    association_scheme: str = "atom14",
    **kwargs,  # to dump remaining args for now
) -> list[Transform]:
    """Build the RFD3 training pipeline.

    Args:
        sigma_data: Scale of noise to add during training
        diffusion_batch_size: Number of diffusion samples to generate per batch
        central_atom: Name of the central atom to use for virtual atoms
        virtual_atom_element: Element symbol to use for virtual atoms
        return_atom_array: Whether to return the atom array in the output
        is_inference: Whether to run the pipeline in inference mode (in case we want to validate our training objective)
        residue_cache_dir: Directory to load residue-level cached data from

    Returns:
        List of transforms for the training pipeline
    """
    transforms = []
    transforms += _build_rfd3_train_pipeline(
        crop_size=crop_size,
        crop_contiguous_probability=crop_contiguous_probability,
        crop_spatial_probability=crop_spatial_probability,
        crop_center_cutoff_distance=crop_center_cutoff_distance,
        design_tasks=design_tasks,
        is_inference=is_inference,
        p_atomize_residues=p_atomize_residues,
        central_atom=central_atom,
    )
    transforms += _build_rfd3_featurize_pipeline(
        n_atoms_per_token=n_atoms_per_token,
        sigma_data=sigma_data,
        diffusion_batch_size=diffusion_batch_size,
        central_atom=central_atom,
        return_atom_array=return_atom_array,
        residue_cache_dir=residue_cache_dir,
        association_scheme=association_scheme,
    )
    return Compose(transforms)


def build_rfd3_validation_pipeline(
    # ... featurization specific
    sigma_data: float = 0.5,
    diffusion_batch_size: int = 16,
    n_atoms_per_token: int = 14,
    central_atom: str = "CB",
    return_atom_array: bool = False,
    atomize_distance_conditioned_tokens: bool = False,
    association_scheme: str = "atom14",
) -> list[Transform]:
    """Build the RFD3 validation pipeline.

    Args:
        sigma_data: Scale of noise to add during validation
        diffusion_batch_size: Number of diffusion samples to generate per batch
        central_atom: Name of the central atom to use for virtual atoms
        virtual_atom_element: Element symbol to use for virtual atoms
        return_atom_array: Whether to return the atom array in the output

    Returns:
        List of transforms for the validation pipeline
    """
    transforms = []
    transforms += _build_rfd3_validation_pipeline(
        atomize_distance_conditioned_tokens=atomize_distance_conditioned_tokens,
    )
    transforms += _build_rfd3_featurize_pipeline(
        sigma_data=sigma_data,
        diffusion_batch_size=diffusion_batch_size,
        central_atom=central_atom,
        n_atoms_per_token=n_atoms_per_token,
        return_atom_array=return_atom_array,
        association_scheme=association_scheme,
    )
    return Compose(transforms)