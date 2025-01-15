from os import PathLike
from pathlib import Path

import numpy as np
import torch
from cifutils.constants import AF3_EXCLUDED_LIGANDS, GAP, STANDARD_AA, STANDARD_DNA, STANDARD_RNA
from cifutils.enums import ChainType

from datahub.common import exists
from datahub.encoding_definitions import AF3SequenceEncoding
from datahub.transforms.af3_reference_molecule import GetAF3ReferenceMoleculeFeatures
from datahub.transforms.atom_array import (
    AddGlobalAtomIdAnnotation,
    AddGlobalTokenIdAnnotation,
    AddWithinChainInstanceResIdx,
    AddWithinPolyResIdxAnnotation,
    ComputeAtomToTokenMap,
    CopyAnnotation,
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
from datahub.transforms.covalent_modifications import FlagAndReassignCovalentModifications
from datahub.transforms.crop import CropContiguousLikeAF3, CropSpatialLikeAF3
from datahub.transforms.diffusion.batch_structures import BatchStructuresForDiffusionNoising
from datahub.transforms.diffusion.edm import SampleEDMNoise
from datahub.transforms.encoding import EncodeAF3TokenLevelFeatures
from datahub.transforms.feature_aggregation.af3 import AggregateFeaturesLikeAF3
from datahub.transforms.featurize_unresolved_residues import (
    PlaceUnresolvedTokenAtomsOnRepresentativeAtom,
    PlaceUnresolvedTokenOnClosestResolvedTokenInSequence,
)
from datahub.transforms.filters import (
    FilterToSpecifiedPNUnits,
    HandleUndesiredResTokens,
    RemoveHydrogens,
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
from datahub.transforms.symmetry import FindAutomorphismsWithNetworkX
from datahub.transforms.template import (
    AddRFTemplates,
    FeaturizeTemplatesLikeAF3,
    OneHotTemplateRestype,
    RandomSubsampleTemplates,
)


def build_af3_transform_pipeline(
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
    conformer_generation_timeout: float = 2.0,  # seconds
    # Template params
    max_n_template: int = 20,  # Maximum number of templates to return from our template search (distinct from n_template)
    n_template: int = 4,
    template_max_seq_similarity: float = 60.0,
    template_min_seq_similarity: float = 10.0,
    template_min_length: int = 10,
    template_allowed_chain_types: list[ChainType] = [ChainType.POLYPEPTIDE_L, ChainType.RNA],
    template_distogram_bins: torch.Tensor = torch.linspace(3.25, 50.75, 38),
    template_default_token: str = GAP,
    # MSA parameters
    max_msa_sequences: int = 10_000,  # Paper: 16,000, but we only have 10K stored on disk
    n_msa: int = 10_000,  # Paper: ?? I think ~12K?
    dense_msa: bool = True,  # True for AF3
    # Cache paths
    msa_cache_dir: PathLike | str | None = None,
    sigma_data: float = 16.0,
    diffusion_batch_size: int = 48,
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

    if (crop_contiguous_probability > 0 or crop_spatial_probability > 0) and not is_inference:
        assert np.isclose(
            crop_contiguous_probability + crop_spatial_probability, 1.0, atol=1e-6
        ), "Crop probabilities must sum to 1.0"
        assert crop_size > 0, "Crop size must be greater than 0"
        assert crop_center_cutoff_distance > 0, "Crop center cutoff distance must be greater than 0"

    af3_sequence_encoding = AF3SequenceEncoding()

    transforms = [
        AddData({"is_inference": is_inference}),
        RemoveHydrogens(),
        FilterToSpecifiedPNUnits(
            extra_info_key_with_pn_unit_iids_to_keep="all_pn_unit_iids_after_processing"
        ),  # Filter to non-clashing PN units
        RemoveTerminalOxygen(),
        RemoveUnresolvedPNUnits(),  # Remove PN units that are unresolved early (and also after cropping)
        RemovePolymersWithTooFewResolvedResidues(min_residues=4),  # Remove polymers with too few resolved residues
        HandleUndesiredResTokens(undesired_res_names),  # e.g., non-standard residues
        FlagAndReassignCovalentModifications(),
        FlagNonPolymersForAtomization(),
        AddGlobalAtomIdAnnotation(),
        AtomizeByCCDName(
            atomize_by_default=True,
            res_names_to_ignore=STANDARD_AA + STANDARD_RNA + STANDARD_DNA,
            move_atomized_part_to_end=False,
            validate_atomize=False,
        ),
        AddWithinChainInstanceResIdx(),
        AddWithinPolyResIdxAnnotation(),
    ]

    # Crop

    # ...crop around our query pn_unit(s) early, since we don't need the full structure moving forward
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

    training_template_loading_transforms = Compose(
        [
            AddRFTemplates(
                max_n_template=max_n_template,  # return at most max_n_template (e.g., 20 in AF-3) from our template search (we will then subsample)
                pick_top=False,
                max_seq_similarity=template_max_seq_similarity,
                min_seq_similarity=template_min_seq_similarity,
                min_template_length=template_min_length,
            ),
            # Subsample templates to n_template (from 20)
            RandomSubsampleTemplates(n_template=n_template),
        ]
    )
    inference_template_loading_transforms = AddRFTemplates(
        max_n_template=n_template,  # return at most n_template (e.g., 4 in AF-3) from our template search (no subsampling)
        pick_top=True,
        max_seq_similarity=template_max_seq_similarity,
        min_seq_similarity=template_min_seq_similarity,
        min_template_length=template_min_length,
    )

    transforms += [
        AddGlobalTokenIdAnnotation(),  # required for reference molecule features and TokenToAtomMap
        EncodeAF3TokenLevelFeatures(sequence_encoding=af3_sequence_encoding),
        GetAF3ReferenceMoleculeFeatures(
            conformer_generation_timeout=conformer_generation_timeout,
            should_generate_automorphisms_with_rdkit=False,  # We use NetworkX for automorphisms instead of RDKit
        ),
        FindAutomorphismsWithNetworkX(),  # Adds the  "automorphisms" key to the data dictionary
        ComputeAtomToTokenMap(),
        ConditionalRoute(
            condition_func=lambda data: data["is_inference"],
            transform_map={
                False: training_template_loading_transforms,
                True: inference_template_loading_transforms,
            },
        ),
        FeaturizeTemplatesLikeAF3(
            sequence_encoding=af3_sequence_encoding,
            gap_token=template_default_token,
            allowed_chain_type=template_allowed_chain_types,
            distogram_bins=template_distogram_bins,
        ),
    ]

    transforms += [
        # ...load and pair MSAs
        LoadPolymerMSAs(
            protein_msa_dirs=protein_msa_dirs,
            rna_msa_dirs=rna_msa_dirs,
            max_msa_sequences=max_msa_sequences,  # maximum number of sequences to load (we later subsample further)
            msa_cache_dir=Path(msa_cache_dir) if exists(msa_cache_dir) else None,
        ),
        PairAndMergePolymerMSAs(dense=dense_msa),
        # ...encode MSA to AF-3 format
        EncodeMSA(encoding=af3_sequence_encoding, token_to_use_for_gap=af3_sequence_encoding.token_to_idx["<G>"]),
        # ...fill MSA, indexing into only the portions of the polymers that are present in the cropped structure
        FillFullMSAFromEncoded(pad_token=af3_sequence_encoding.token_to_idx["<G>"]),
        AddAF3TokenBondFeatures(),
        # ...featurize MSA
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
        # ...add placeholder coordinates for noising
        CopyAnnotation(annotation_to_copy="coord", new_annotation="coord_to_be_noised"),
        # ...handling of unresolved residues (note that these Transforms create the "atom_array_to_noise" dictionary, if not already present)
        PlaceUnresolvedTokenAtomsOnRepresentativeAtom(annotation_to_update="coord_to_be_noised"),
        PlaceUnresolvedTokenOnClosestResolvedTokenInSequence(annotation_to_update="coord_to_be_noised"),
        # Feature aggregation
        AggregateFeaturesLikeAF3(),
        OneHotTemplateRestype(encoding=af3_sequence_encoding),
        # ...batching and noise sampling for diffusion
        BatchStructuresForDiffusionNoising(batch_size=diffusion_batch_size),
        CenterRandomAugmentation(batch_size=diffusion_batch_size),
        SampleEDMNoise(sigma_data=sigma_data, diffusion_batch_size=diffusion_batch_size),
        # ... remove all non-feature keys (to make compatible wit generic batch_collate, which only allows tensors, numpy arrays, str, etc.)
        SubsetToKeys(
            [
                "example_id",
                "feats",
                "t",
                "noise",
                "ground_truth",
                "coord_atom_lvl_to_be_noised",
                "automorphisms",
                "symmetry_resolution",
            ]
        ),
    ]

    # ... compose final pipeline
    pipeline = Compose(transforms)
    return pipeline
