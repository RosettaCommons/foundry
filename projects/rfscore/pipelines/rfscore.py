from functools import partial
from os import PathLike
from pathlib import Path

import numpy as np
import torch
from cifutils.constants import (
    AF3_EXCLUDED_LIGANDS,
    STANDARD_AA,
    STANDARD_DNA,
    STANDARD_RNA,
)
from cifutils.enums import ChainType
from datahub.common import exists
from datahub.encoding_definitions import RF2AA_ATOM36_ENCODING, AF3SequenceEncoding
from datahub.transforms.af3_reference_molecule import (
    GetAF3ReferenceMoleculeFeatures,
    GroundTruthConformerPolicy,
    RandomApplyGroundTruthConformerByChainType,
)
from datahub.transforms.atom_array import (
    AddGlobalAtomIdAnnotation,
    AddGlobalTokenIdAnnotation,
    AddWithinChainInstanceResIdx,
    AddWithinPolyResIdxAnnotation,
    ComputeAtomToTokenMap,
    CopyAnnotation,
)
from datahub.transforms.atom_frames import (
    AddAtomFrames,
    AddIsRealAtom,
    AddPolymerFrameIndices,
)
from datahub.transforms.atomize import AtomizeByCCDName, FlagNonPolymersForAtomization
from datahub.transforms.base import (
    AddData,
    Compose,
    ConditionalRoute,
    ConvertToTorch,
    Identity,
    RandomRoute,
    SubsetToKeys,
)
from datahub.transforms.bonds import AddAF3TokenBondFeatures
from datahub.transforms.center_random_augmentation import CenterRandomAugmentation
from datahub.transforms.chirals import AddAF3ChiralFeatures
from datahub.transforms.covalent_modifications import (
    FlagAndReassignCovalentModifications,
)
from datahub.transforms.crop import CropContiguousLikeAF3, CropSpatialLikeAF3
from datahub.transforms.diffusion.batch_structures import (
    BatchStructuresForDiffusionNoising,
)
from datahub.transforms.diffusion.edm import SampleEDMNoise
from datahub.transforms.dna.pad_dna import PadDNA
from datahub.transforms.encoding import EncodeAF3TokenLevelFeatures, EncodeAtomArray
from datahub.transforms.feature_aggregation.af3 import AggregateFeaturesLikeAF3
from datahub.transforms.feature_aggregation.confidence import PackageConfidenceFeats
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
    RemoveUnresolvedPNUnits,
)
from datahub.transforms.msa.msa import (
    EncodeMSA,
    FeaturizeMSALikeAF3,
    FillFullMSAFromEncoded,
    LoadPolymerMSAs,
    PairAndMergePolymerMSAs,
)
from datahub.transforms.rdkit_utils import GetRDKitChiralCenters
from datahub.transforms.symmetry import FindAutomorphismsWithNetworkX
from omegaconf import DictConfig

from projects.rfscore.pipelines.composed import build_ground_truth_distogram_transform


def build_rfscore_transform_pipeline(
    *,
    # Training or inference (required)
    is_inference: bool,  # If True, we skip cropping, etc.
    # MSA dirs
    protein_msa_dirs: list[dict],
    rna_msa_dirs: list[dict],
    # Recycles
    n_recycles: int = 5,
    # Crop params
    crop_size: int = 384,
    crop_center_cutoff_distance: float = 15.0,
    crop_contiguous_probability: float = 0.5,
    crop_spatial_probability: float = 0.5,
    max_atoms_in_crop: int | None = None,
    # Undesired res names
    undesired_res_names: list[str] = AF3_EXCLUDED_LIGANDS,
    # Conformer generation params
    conformer_generation_timeout: float = 5.0,  # seconds
    use_element_for_atom_names_of_atomized_tokens: bool = False,
    # Template params
    max_msa_sequences: int = 10_000,  # Paper: 16,000, but we only have 10K stored on disk
    n_msa: int = 10_000,  # Paper: ?? I think ~12K?
    dense_msa: bool = True,  # True for AF3
    # Cache paths
    msa_cache_dir: PathLike | str | None = None,
    sigma_data: float = 16.0,
    diffusion_batch_size: int = 48,
    # Whether to include features for confidence head
    run_confidence_head: bool = False,
    # Whether to include atom array in the output
    return_atom_array: bool = True,
    # DNA
    pad_dna_p_skip: float = 0.0,
    # Conditioning
    p_unconditional: float = 0.2,
    # (Templates)
    train_template_noise_scales: dict | DictConfig = {
        "atomized": 1e-4,
        "not_atomized": 1.0,
    },
    allowed_chain_types_for_conditioning: list[int | str | ChainType]
    | None = None,  # None = no conditioning
    inference_template_noise_scales: dict | DictConfig = {
        "atomized": 1e-4,
        "not_atomized": 1e-4,
    },
    p_condition_per_token: float = 0.5,
    p_provide_inter_molecule_distances: float = 0.0,
    # (Reference Conformer)
    p_give_non_polymer_ref_conf: float = 0.2,
    p_give_polymer_ref_conf: float = 0.05,
    # (Skip MSA)
    p_skip_msa: float = 0.3,
):
    """Build the AF3 pipeline with specified parameters.

    This function constructs a pipeline of transforms for processing protein structures
    in a manner similar to AlphaFold 3. The pipeline includes steps for removing hydrogens,
    adding annotations, atomizing residues, cropping, adding templates, encoding features,
    and generating reference molecule features.

    Args:
        crop_size (int, optional): The size of the crop. Defaults to 384.
        crop_center_cutoff_distance (float, optional): The cutoff distance for spatial cropping.
            Defaults to 15.0.
        crop_contiguous_probability (float, optional): The probability of using contiguous cropping.
            Defaults to 0.5.
        crop_spatial_probability (float, optional): The probability of using spatial cropping.
            Defaults to 0.5.
        conformer_generation_timeout (float, optional): The timeout for conformer generation in seconds.
            Defaults to 10.0.

    Returns:
        Transform: A composed pipeline of transforms.

    Raises:
        AssertionError: If the crop probabilities do not sum to 1.0, if the crop size is not positive,
        or if the crop center cutoff distance is not positive.

    Note:
        The cropping method is chosen randomly based on the provided probabilities.
        The pipeline includes steps for processing the structure, adding annotations,
        and generating features required for AF3-like predictions.

    References:
        - AlphaFold 3 Supplementary Information.
          https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf
    """

    if (
        crop_contiguous_probability > 0 or crop_spatial_probability > 0
    ) and not is_inference:
        assert np.isclose(
            crop_contiguous_probability + crop_spatial_probability, 1.0, atol=1e-6
        ), "Crop probabilities must sum to 1.0"
        assert crop_size > 0, "Crop size must be greater than 0"
        assert (
            crop_center_cutoff_distance > 0
        ), "Crop center cutoff distance must be greater than 0"

    af3_sequence_encoding = AF3SequenceEncoding()
    rf2aa_sequence_encoding = RF2AA_ATOM36_ENCODING

    transforms = [
        AddData(
            {"is_inference": is_inference, "run_confidence_head": run_confidence_head}
        ),
        # Set the "unconditional" key stochastically to True or False
        RandomRoute(
            transforms=[
                AddData({"is_unconditional": True}),
                AddData({"is_unconditional": False}),
            ],
            probs=[p_unconditional, 1 - p_unconditional],
        ),
        RemoveHydrogens(),
        FilterToSpecifiedPNUnits(
            extra_info_key_with_pn_unit_iids_to_keep="all_pn_unit_iids_after_processing"
        ),  # Filter to non-clashing PN units
        RemoveTerminalOxygen(),
        RemoveUnresolvedPNUnits(),
        RemovePolymersWithTooFewResolvedResidues(min_residues=4),
        MaskPolymerResiduesWithUnresolvedFrameAtoms(),
        # NOTE: For inference, we must keep UNL to support ligands that are not in the CCD
        HandleUndesiredResTokens(
            undesired_res_tokens=undesired_res_names
        ),  # e.g., non-standard residues
        ConditionalRoute(
            condition_func=lambda data: data.get("is_inference", False),
            transform_map={
                True: Identity(),
                False: PadDNA(p_skip=pad_dna_p_skip),
            },
        ),
        FlagAndReassignCovalentModifications(),
        FlagNonPolymersForAtomization(),
        AddGlobalAtomIdAnnotation(),
        AtomizeByCCDName(
            atomize_by_default=True,
            res_names_to_ignore=STANDARD_AA + STANDARD_RNA + STANDARD_DNA,
            move_atomized_part_to_end=False,
            validate_atomize=False,
        ),
        RemoveNucleicAcidTerminalOxygen(),
        AddWithinChainInstanceResIdx(),
        AddWithinPolyResIdxAnnotation(),
    ]

    # +----------------------------------------------+
    # +------------------ CROPPING ------------------+
    # +----------------------------------------------+

    # ... crop around our query pn_unit(s) early, since we don't need the full structure moving forward
    cropping_transform = RandomRoute(
        transforms=[
            CropContiguousLikeAF3(
                crop_size=crop_size,
                keep_uncropped_atom_array=True,
                max_atoms_in_crop=max_atoms_in_crop,
            ),
            CropSpatialLikeAF3(
                crop_size=crop_size,
                crop_center_cutoff_distance=crop_center_cutoff_distance,
                keep_uncropped_atom_array=True,
                max_atoms_in_crop=max_atoms_in_crop,
            ),
        ],
        probs=[crop_contiguous_probability, crop_spatial_probability],
    )

    transforms.append(
        ConditionalRoute(
            condition_func=lambda data: data.get("is_inference", False),
            transform_map={
                True: Identity(),
                False: cropping_transform,
                # Default to Identity during inference (`is_inference == True`)
            },
        )
    )

    # +-----------------------------------------------------------+
    # +------------------ GROUND TRUTH TEMPLATE ------------------+
    # +-----------------------------------------------------------+

    # Ground truth template noising
    transforms.append(
        build_ground_truth_distogram_transform(
            template_noise_scales=train_template_noise_scales,
            allowed_chain_types_for_conditioning=allowed_chain_types_for_conditioning,
            p_condition_per_token=p_condition_per_token,
            p_provide_inter_molecule_distances=p_provide_inter_molecule_distances,
            is_inference=is_inference,
        )
    )

    # +----------------------------------------------------------------------+
    # +------------------ GROUND TRUTH REFERENCE CONFORMER ------------------+
    # +----------------------------------------------------------------------+

    transforms.append(
        RandomApplyGroundTruthConformerByChainType(
            chain_type_probabilities={
                tuple(ChainType.get_polymers()): p_give_polymer_ref_conf,
                tuple(ChainType.get_non_polymers()): p_give_non_polymer_ref_conf,
            },
            policy=GroundTruthConformerPolicy.ADD,
        )
    )

    transforms += [
        AddGlobalTokenIdAnnotation(),  # required for reference molecule features and TokenToAtomMap
        EncodeAF3TokenLevelFeatures(sequence_encoding=af3_sequence_encoding),
        GetAF3ReferenceMoleculeFeatures(
            conformer_generation_timeout=conformer_generation_timeout,
            should_generate_automorphisms_with_rdkit=False,  # We use NetworkX for automorphisms instead of RDKit
            use_element_for_atom_names_of_atomized_tokens=use_element_for_atom_names_of_atomized_tokens,
        ),
        FindAutomorphismsWithNetworkX(),  # Adds the  "automorphisms" key to the data dictionary
        ComputeAtomToTokenMap(),
        GetRDKitChiralCenters(),
        AddAF3ChiralFeatures(),
    ]

    transforms += [
        # ... load and pair MSAs
        # (With probability p_skip_msa, if `is_unconditional` is False, skip loading MSAs)
        ConditionalRoute(
            condition_func=lambda data: not data["is_unconditional"] and not data["is_inference"] and np.random.rand() < p_skip_msa,
            transform_map={
                True: LoadPolymerMSAs(
                    protein_msa_dirs=None,
                    rna_msa_dirs=None,
                    use_paths_in_chain_info=False,
                ),
                False: LoadPolymerMSAs(
                    protein_msa_dirs=protein_msa_dirs,
                    rna_msa_dirs=rna_msa_dirs,
                    max_msa_sequences=max_msa_sequences,  # maximum number of sequences to load (we later subsample further)
                    msa_cache_dir=Path(msa_cache_dir)
                    if exists(msa_cache_dir)
                    else None,
                    use_paths_in_chain_info=True,  # if there are paths specified in the `chain_info` for a given chain, use them
                ),
            },
        ),
        PairAndMergePolymerMSAs(dense=dense_msa),
        # ... encode MSA to AF-3 format
        EncodeMSA(
            encoding=af3_sequence_encoding,
            token_to_use_for_gap=af3_sequence_encoding.token_to_idx["<G>"],
        ),
        # ... fill MSA, indexing into only the portions of the polymers that are present in the cropped structure
        FillFullMSAFromEncoded(pad_token=af3_sequence_encoding.token_to_idx["<G>"]),
        AddAF3TokenBondFeatures(),
        # ... featurize MSA
        ConvertToTorch(
            keys=[
                "encoded",
                "feats",
                "full_msa_details",
            ]
        ),
        FeaturizeMSALikeAF3(
            encoding=af3_sequence_encoding,
            n_recycles=n_recycles,
            n_msa=n_msa,
        ),
        # Prepare coordinates for noising (without modifying the ground truth)
        # ... add placeholder coordinates for noising
        CopyAnnotation(annotation_to_copy="coord", new_annotation="coord_to_be_noised"),
        # ... handling of unresolved residues (note that these Transforms create the "atom_array_to_noise" dictionary, if not already present)
        PlaceUnresolvedTokenAtomsOnRepresentativeAtom(
            annotation_to_update="coord_to_be_noised"
        ),
        PlaceUnresolvedTokenOnClosestResolvedTokenInSequence(
            annotation_to_update="coord_to_be_noised",
            annotation_to_copy="coord_to_be_noised",
        ),
        # Feature aggregation
        AggregateFeaturesLikeAF3(),
        # ... batching and noise sampling for diffusion
        BatchStructuresForDiffusionNoising(batch_size=diffusion_batch_size),
        CenterRandomAugmentation(batch_size=diffusion_batch_size),
        SampleEDMNoise(
            sigma_data=sigma_data, diffusion_batch_size=diffusion_batch_size
        ),
    ]

    confidence_transforms = Compose(
        [
            # Additions required for confidence calculation
            EncodeAtomArray(rf2aa_sequence_encoding),
            AddAtomFrames(),
            AddIsRealAtom(rf2aa_sequence_encoding),
            AddPolymerFrameIndices(),
            # wrap it all together
            PackageConfidenceFeats(),
        ]
    )

    transforms.append(
        ConditionalRoute(
            condition_func=lambda data: data.get("run_confidence_head", False),
            transform_map={
                True: confidence_transforms,
                False: Identity(),
            },
        )
    )

    keys_to_keep = [
        "example_id",
        "feats",
        "t",
        "noise",
        "ground_truth",
        "coord_atom_lvl_to_be_noised",
        "automorphisms",
        "symmetry_resolution",
        "extra_info",
    ]
    if run_confidence_head:
        keys_to_keep.append("confidence_feats")
    if return_atom_array and is_inference:
        keys_to_keep.append("atom_array")

    transforms += [
        # Subset to only keys necessary
        SubsetToKeys(keys_to_keep)
    ]

    # ... compose final pipeline
    pipeline = Compose(transforms)

    return pipeline
