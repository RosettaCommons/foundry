from functools import partial

import torch
from atomworks.enums import ChainType
from atomworks.ml.transforms._checks import check_atom_array_annotation
from atomworks.ml.transforms.crop import compute_local_hash
from omegaconf import DictConfig
from rf3.data.ground_truth_template import (
    FeaturizeNoisedGroundTruthAsTemplateDistogram,
    TokenGroupNoiseScaleSampler,
    af3_noise_scale_distribution_wrapped,
    af3_noise_scale_to_noise_level,
)


def annotate_pre_crop_hash(data: dict) -> dict:
    hash_pre = compute_local_hash(data["atom_array"])
    data["atom_array"].set_annotation("hash_pre", hash_pre)
    return data


def annotate_post_crop_hash(data: dict) -> dict:
    hash_post = compute_local_hash(data["atom_array"])
    data["atom_array"].set_annotation("hash_post", hash_post)
    return data


def set_to_occupancy_0_where_crop_hashes_differ(data: dict) -> dict:
    check_atom_array_annotation(
        data["atom_array"], ["hash_pre", "hash_post", "occupancy"]
    )

    # Create a mask of where hash_pre != hash_post
    atom_array = data["atom_array"]
    mask = atom_array.get_annotation("hash_pre") != atom_array.get_annotation(
        "hash_post"
    )

    # Where the hashes differ, set occupancy to 0
    atom_array.occupancy[mask] = 0

    return data


def build_ground_truth_distogram_transform(
    *,
    template_noise_scales: dict[str, float | None] | DictConfig,
    allowed_chain_types_for_conditioning: list[ChainType] | None = None,
    p_condition_per_token: float = 0.0,
    p_provide_inter_molecule_distances: float = 0.0,
    is_inference: bool = False,
) -> FeaturizeNoisedGroundTruthAsTemplateDistogram:
    """
    Build a FeaturizeNoisedGroundTruthAsTemplateDistogram transform for either training or inference.

    For inference, we must be deterministic, so we:
        - Use constant noise scales (1.0)
        - Always apply token-level conditioning

    Args:
        template_noise_scales (dict[str, float | None] | DictConfig):
            Noise scales for 'atomized' and 'not_atomized' tokens. If is_inference=True, these are used as constants.
            If is_inference=False, these are used as upper bounds for the noise scale distribution.
        allowed_chain_types_for_conditioning (list[ChainType] | None):
            List of allowed chain types for conditioning. None disables conditioning.
        p_condition_per_token (float):
            Probability of conditioning each eligible token.
        p_provide_inter_molecule_distances (float):
            Probability of providing inter-molecule (inter-chain) distances.
        is_inference (bool):
            If True, use constant noise scales for conditioning. If False, sample from provided distributions.

    Returns:
        FeaturizeNoisedGroundTruthAsTemplateDistogram: The configured transform.
    """
    mask_and_sampling_fns = []
    if is_inference:
        # Use constant noise scales for inference, rather than sampling (no stochasticity)
        if template_noise_scales["atomized"] is not None:
            mask_and_sampling_fns.append(
                (
                    lambda arr: arr.atomize,
                    lambda size: torch.ones(size) * template_noise_scales["atomized"],
                )
            )
        if template_noise_scales["not_atomized"] is not None:
            mask_and_sampling_fns.append(
                (
                    lambda arr: ~arr.atomize,
                    lambda size: torch.ones(size)
                    * template_noise_scales["not_atomized"],
                )
            )
    else:
        # Use noise scale distributions for training
        if template_noise_scales["atomized"] is not None:
            mask_and_sampling_fns.append(
                (
                    lambda arr: arr.atomize,
                    partial(
                        af3_noise_scale_distribution_wrapped,
                        upper_noise_level=af3_noise_scale_to_noise_level(
                            template_noise_scales["atomized"]
                        ).item(),
                    ),
                )
            )
        if template_noise_scales["not_atomized"] is not None:
            mask_and_sampling_fns.append(
                (
                    lambda arr: ~arr.atomize,
                    partial(
                        af3_noise_scale_distribution_wrapped,
                        upper_noise_level=af3_noise_scale_to_noise_level(
                            template_noise_scales["not_atomized"]
                        ).item(),
                    ),
                )
            )

    return FeaturizeNoisedGroundTruthAsTemplateDistogram(
        noise_scale_distribution=TokenGroupNoiseScaleSampler(
            mask_and_sampling_fns=tuple(mask_and_sampling_fns),
        ),
        allowed_chain_types=allowed_chain_types_for_conditioning,
        p_condition_per_token=p_condition_per_token,
        p_provide_inter_molecule_distances=p_provide_inter_molecule_distances,
    )
