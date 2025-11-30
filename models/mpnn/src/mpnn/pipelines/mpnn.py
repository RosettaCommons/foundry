from atomworks.constants import AF3_EXCLUDED_LIGANDS, STANDARD_AA, UNKNOWN_AA
from atomworks.enums import ChainTypeInfo
from atomworks.ml.transforms.atom_array import AddWithinChainInstanceResIdx
from atomworks.ml.transforms.atomize import (
    AtomizeByCCDName,
    FlagNonPolymersForAtomization,
)
from atomworks.ml.transforms.base import (
    AddData,
    Compose,
    ConditionalRoute,
    ConvertToTorch,
    Identity,
    SubsetToKeys,
)
from atomworks.ml.transforms.covalent_modifications import (
    FlagAndReassignCovalentModifications,
)
from atomworks.ml.transforms.featurize_unresolved_residues import (
    MaskResiduesWithSpecificUnresolvedAtoms,
)
from atomworks.ml.transforms.filters import (
    FilterToSpecifiedPNUnits,
    HandleUndesiredResTokens,
    RemoveHydrogens,
    RemoveUnresolvedTokens,
)
from mpnn.transforms.feature_aggregation.mpnn import (
    EncodeMPNNNonAtomizedTokens,
    FeaturizeAtomizedTokens,
    FeaturizeNonAtomizedTokens,
)
from mpnn.transforms.feature_aggregation.user_settings import (
    FeaturizeUserSettings,
)


def TrainingRoute(transform):
    return ConditionalRoute(
        condition_func=lambda data: data["is_inference"],
        transform_map={True: Identity(), False: transform},
    )


def InferenceRoute(transform):
    return ConditionalRoute(
        condition_func=lambda data: data["is_inference"],
        transform_map={False: Identity(), True: transform},
    )


def ModelTypeRoute(transform, model_type: str):
    return ConditionalRoute(
        condition_func=lambda data: data["model_type"] == model_type,
        transform_map={True: transform, False: Identity()},
    )


def build_mpnn_transform_pipeline(
    *,
    model_type: str = None,
    occupancy_threshold_sidechain: float = 0.5,
    occupancy_threshold_backbone: float = 0.8,
    is_inference: bool = False,
    minimal_return: bool = False,
    train_structure_noise_default: float = 0.1,
    undesired_res_names: list[str] = AF3_EXCLUDED_LIGANDS,
    device=None,
) -> Compose:
    """Build the MPNN transform pipeline.
    Args:
        model_type (str): Model type identifier to include in data. Must be
            provided. Defaults to None.
        occupancy_threshold_sidechain (float): Minimum occupancy to consider
            sidechain atoms as present in masks. Defaults to 0.5.
        occupancy_threshold_backbone (float): Minimum occupancy to consider
            backbone atoms as resolved. Residues with backbone atoms below this
            threshold will be masked entirely. Defaults to 0.8.
        train_structure_noise_default (float): Default standard deviation of
            Gaussian noise to add to atomic coordinates during training for data
            augmentation. Defaults to 0.1.
        is_inference (bool): Whether this is inference mode. Defaults to
            False (training mode).
        minimal_return (bool): Whether to return minimal intermediate data.
            Defaults to False.
        undesired_res_names (list[str]): List of residue names to treat as
            undesired and handle accordingly. Defaults to AF3_EXCLUDED_LIGANDS.
        device (str | torch.device, optional): Device to move tensors to.
            Defaults to None, which leads to default ConvertToTorch behavior.
    """
    if model_type not in ("protein_mpnn", "ligand_mpnn"):
        raise ValueError(f"Unsupported model_type: {model_type}")

    transforms = [
        AddData({"model_type": model_type}),
        AddData({"is_inference": is_inference}),
        # + --------- Filters --------- +
        RemoveHydrogens(),
        # ... during training, filter to non-clashing chains (which are
        # pre-computed and stored in the "extra_info" key)
        TrainingRoute(
            FilterToSpecifiedPNUnits(
                extra_info_key_with_pn_unit_iids_to_keep="all_pn_unit_iids_after_processing"
            ),
        ),
        # ... during training, remove undesired residues (e.g., non-biological
        # crystallization artifacts), mapping to the closest canonical residue
        # name where possible
        TrainingRoute(
            HandleUndesiredResTokens(undesired_res_tokens=undesired_res_names),
        ),
        # + --------- Atomization --------- +
        # ... add within-chain instance res idx
        AddWithinChainInstanceResIdx(),
        FlagAndReassignCovalentModifications(),
        # Atomization: keep standard AA + unknown AA as residues,
        # atomize everything else
        FlagNonPolymersForAtomization(),
        AtomizeByCCDName(
            atomize_by_default=True,
            res_names_to_ignore=STANDARD_AA + (UNKNOWN_AA,),
            move_atomized_part_to_end=False,
            validate_atomize=False,
        ),
        # + --------- Occupancy filtering --------- +
        MaskResiduesWithSpecificUnresolvedAtoms(
            chain_type_to_atom_names={
                ChainTypeInfo.PROTEINS: [
                    "N",
                    "CA",
                    "C",
                    "O",
                ],  # MPNN needs backbone + oxygen
            },
            occupancy_threshold=occupancy_threshold_backbone,
        ),
        RemoveUnresolvedTokens(),
        # +-------- Encoding and featurization --------- +
        AddData({"input_features": dict()}),
        # Encode and featurize non-atomized tokens
        EncodeMPNNNonAtomizedTokens(occupancy_threshold=occupancy_threshold_sidechain),
        FeaturizeNonAtomizedTokens(),
        # LigandMPNN specific featurization: featurize atomized tokens
        ModelTypeRoute(
            transform=FeaturizeAtomizedTokens(),
            model_type=model_type,
        ),
        # Featurize user settings
        FeaturizeUserSettings(
            is_inference=is_inference,
            minimal_return=minimal_return,
            train_structure_noise_default=train_structure_noise_default,
        ),
        # Convert to torch and subset keys
        ConvertToTorch(
            keys=["input_features"],
            **({"device": device} if device is not None else {}),
        ),
        SubsetToKeys(keys=["input_features", "atom_array"]),
    ]

    return Compose(transforms)
