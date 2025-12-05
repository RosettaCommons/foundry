import logging
from dataclasses import dataclass

import numpy as np
import torch
from atomworks.enums import ChainType
from atomworks.ml.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_is_instance,
)
from atomworks.ml.transforms.atomize import AtomizeByCCDName
from atomworks.ml.transforms.base import Transform
from atomworks.ml.utils.token import (
    get_af3_token_center_masks,
    get_token_starts,
)
from beartype.typing import Any, Callable, Final, Sequence
from biotite.structure import AtomArray
from jaxtyping import Bool, Float, Shaped
from torch import Tensor

from foundry.utils.torch import assert_no_nans

logger = logging.getLogger(__name__)

MaskingFunction = Callable[[AtomArray], Bool[Shaped, "n"]]
"""A function that takes in an AtomArray and returns a boolean mask."""

NoiseScaleSampler = Callable[[Sequence[int]], Float[Tensor, "..."] | float]
"""
A noise scale sampler that, when given a shape-tuple, returns a sample of
noise scales of the appropriate shape.

Examples:
    - partial(torch.normal, mean=0.0, std=1.0)
    - af3_noise_scale_distribution
    - af3_noise_scale_distribution_wrapped
"""


@dataclass
class TokenGroupNoiseScaleSampler:
    mask_and_sampling_fns: tuple[tuple[MaskingFunction, NoiseScaleSampler], ...]

    def __call__(self, atom_array: AtomArray) -> Tensor:
        # ... determine token centers
        token_center_mask = get_af3_token_center_masks(atom_array)  # [n_token] (bool)
        token_array = atom_array[token_center_mask]  # [n_token] (AtomArray)

        # ... sample a noise scale for each token group
        noise_scales = torch.full(
            size=(len(token_array),),
            fill_value=float("nan"),
            dtype=torch.float32,
        )
        for mask_fn, sampling_fn in self.mask_and_sampling_fns:
            mask = mask_fn(token_array)
            n_tokens_to_sample = mask.sum()
            if n_tokens_to_sample > 0:
                # ... all tokens in that group receive the same noise scale
                noise_scales[mask] = sampling_fn((1,))

        return noise_scales


DEFAULT_DISTOGRAM_BINS: Final[Float[Tensor, "63"]] = torch.concat(
    (
        torch.arange(1.0, 4.0, 0.1, device="cpu"),
        torch.arange(4.0, 20.5, 0.5, device="cpu"),
    )
)
"""
Default bins for discretizing distances in the distogram (in Angstrom).
    - 0.1A resolution from 1.0 -  4.0 A
    - 0.5A resolution from 4.0 - 20.0 A
Total number of bins: 64  (i.e. 63 bin boundaries above)
"""


def wrap_probability_distribution(
    samples: Float[Tensor, "..."],
    lower: float = float("-inf"),
    upper: float = float("inf"),
) -> Float[Tensor, "..."]:
    """
    Wrap a probability distribution around lower and upper bounds to create
    samples from the corresponding wrapped probability distribution.

    Args:
        - samples: Input tensor of samples to wrap
        - lower: Lower bound for wrapping (inclusive, unless infinite)
        - upper: Upper bound for wrapping (inclusive, unless infinite)

    Returns:
        - samples: Samples wrapped around the lower and upper bounds within
            the interval ]lower, upper[.
    Reference:
        - https://en.wikipedia.org/wiki/Wrapped_distribution
    """
    if lower > float("-inf") and upper < float("inf"):
        return ((samples - lower) % (upper - lower)) + lower
    elif lower > float("-inf"):
        return lower + (samples - lower).abs()
    elif upper < float("inf"):
        return upper - (samples - upper).abs()
    return samples


def wrapped_normal(
    mean: float,
    std: float,
    size: Sequence[int],
    *,
    lower: float = float("-inf"),
    upper: float = float("inf"),
    **normal_kwargs,
) -> Float[Tensor, "..."]:
    """Sample from a wrapped normal distribution."""
    samples = torch.normal(mean=mean, std=std, size=size, **normal_kwargs)
    return wrap_probability_distribution(samples, lower, upper)


def af3_noise_scale_to_noise_level(
    noise_scale: Tensor | float, eps: int = 1e-8
) -> Tensor:
    """Converts AlphaFold3 noise scale (t^) in Angstroms to noise level (t).

    This function converts from a noise scale in Angstroms (t^) to the
    corresponding standard normal noise level (t) using the formula:
        t = (log(t^/16.0) + 1.2) / 1.5

    Args:
        - noise_scale (Tensor): The noise scale (t^) in Angstroms,
            representing the standard deviation of positional noise.

    Returns:
        - noise_level (Tensor): The corresponding noise level (t) as a
            standard normal random variable.

    Notes:
        - We use the term 'noise-level' to refer to the standard normal random
        variable `t` in the AF3 paper and 'noise-scale' to refer to the variable
        `t^` which denotes the noise scale in Angstrom. This is the inverse
        operation of af3_noise_level_to_noise_scale().
        - To avoid taking the log of zero, we add a small constant to the
        denominator (16.0) in the formula.
    """
    noise_scale_tensor = torch.as_tensor(noise_scale)
    return (torch.log(torch.clamp(noise_scale_tensor, min=eps) / 16.0) + 1.2) / 1.5


def af3_noise_level_to_noise_scale(noise_level: Tensor | float) -> Tensor:
    """Convert AlphaFold3 noise level (t) to noise scale (t^) in Angstroms.

    This function converts from a standard normal noise level (t) to the
    corresponding noise scale in Angstroms (t^) using the formula:
        t^ = 16.0 * exp(1.5t - 1.2)     (log-N(log(0.04), 1.5^2))

    Args:
        - noise_level (Tensor): The noise level (t) as a standard normal random
            variable, sampled from N(0,1).

    Returns:
        - noise_scale (Tensor): The corresponding noise scale (t^) in Angstroms,
            representing the standard deviation of positional noise to apply.

    Note:
        This is the inverse operation of af3_noise_scale_to_noise_level(). The
        transformation is designed to convert between a normal distribution and
        a log-normal distribution with specific parameters chosen by AlphaFold3.
    """
    return 16.0 * torch.exp((torch.as_tensor(noise_level) * 1.5) - 1.2)


def af3_noise_scale_distribution(size: Sequence[int], **kwargs) -> Tensor:
    """
    The log-normal noise-scale distribution used in AF3 (in Angstrom).

       t^ = 16.0 * exp(1.5t - 1.2),
       where:
        - t  = noise-level ~ N(0,1)
        - t^ = noise-scale ~ log-N(log(0.04), 1.5^2)
    """
    noise_level = torch.normal(mean=0.0, std=1.0, size=size, **kwargs)
    return af3_noise_level_to_noise_scale(noise_level)


def af3_noise_scale_distribution_wrapped(
    size: Sequence[int],
    *,
    lower_noise_level: float = float("-inf"),
    upper_noise_level: float = float("inf"),
    **kwargs,
) -> Tensor:
    """
    The noise-scale distribution used in AF3 (in Angstrom), wrapped around the lower
    and upper bounds.

    WARNING: The lower/upper here correspond to the noise-level (t) (not noise-scale (t^)),
        wrapping happens in the noise-level space before converting to the corresponding
        log-normal noise-scale distribution (t^) in Angstroms.
    """
    noise_level = wrapped_normal(
        mean=0.0,
        std=1.0,
        size=size,
        lower=lower_noise_level,
        upper=upper_noise_level,
        **kwargs,
    )
    return af3_noise_level_to_noise_scale(noise_level)


def featurize_noised_ground_truth_as_template_distogram(
    atom_array: AtomArray,
    *,
    noise_scale: Float[Tensor, "n_token"] | float,
    distogram_bins: Float[Tensor, "n_bin_edges"],
    allowed_chain_types: list[ChainType],
    is_unconditional: bool = True,
    p_condition_per_token: float = 0.0,
    p_provide_inter_molecule_distances: float = 0.0,
    existing_annotation_to_check: str = "is_input_file_templated",
) -> dict[str, Tensor]:
    """Featurize noised ground truth as a template distogram for conditioning.

    Used to leak ground-truth information into the model.

    Args:
        atom_array (AtomArray): The input atom array. Must have 'chain_type', 'occupancy', and 'molecule_id' annotations.
        noise_scale (Tensor | float): Standard deviation of the noise to add to the ground truth.
            Different tokens may have different noise scales (e.g. one noise scale for
            side-chains, one for ligand atoms and one for backbone atoms).
            Units are in Angstrom. If given as tensor, must be of shape [n_token] (float).
        allowed_chain_types (list): List of allowed chain types. Only token pairs where BOTH
            tokens have a chain type in this list will have a distogram condition.
        distogram_bins (Tensor): Bins for discretizing distances in the distogram (in Angstrom).
            Shape: [n_bin].
        is_unconditional (bool): Whether we are sampling unconditionally.
            See Classifier-Free Diffusion Guidance (Ho et al., 2022) for details.
            Default: True (no conditioning).
        p_condition_per_token (float, optional):
            Probability of conditioning each eligible token. Default: 0.0 (no conditioning)
        p_provide_inter_molecule_distances (float, optional):
            Probability of providing inter-molecule (inter-chain) distances. Default: 0.0 (mask all inter-molecule pairs).
        existing_annotation_to_check (str):
            If this annotation exists in the AtomArray, we ALWAYS template where it is True.
            Useful for inference.

    Returns:
        dict[str, Tensor]:
            Dictionary with the following keys:
                - 'distogram_condition_noise_scale': Float[Tensor, "n_token"]. Noise scale for each token (0 for unconditioned tokens).
                - 'has_distogram_condition': Bool[Tensor, "n_token n_token"]. Mask indicating which token pairs are conditioned.
                - 'distogram_condition': Float[Tensor, "n_token n_token n_bins"]. One-hot encoded distogram for each token pair.

    NOTE:
        - We use the center atom for each token (CA for proteins, C1' for nucleic acids) in the token-level conditioning.
        - If a token is not conditioned, its noise scale is set to 0 and its pairwise distances are masked.
    """
    MASK_VALUE = float("nan")

    # Get full atom array token starts (useful for going from atom-level -> token-level annotations)
    _a_token_starts = get_token_starts(atom_array)  # [n_token] (int)
    _n_token = len(_a_token_starts)

    # Create one blank template (ground truth), initialized to mask tokens (we will only use the distogram, and ignore the other features)
    template_distogram = torch.full((_n_token, _n_token), fill_value=MASK_VALUE)

    # Sample Gaussian noise according to the noise scale for each token
    # NOTE: If a scalar noise scale is provided, it will be broadcasted to all tokens
    # NOTE: We sample noise independently for each token; no two tokens will have the exact same noise
    noise = torch.normal(mean=0.0, std=1.0, size=(_n_token, 3)) * noise_scale.unsqueeze(
        -1
    )

    # Get the center coordinates of the tokens (CA for proteins, C1' for nucleic acids), and add noise
    center_token_mask = get_af3_token_center_masks(atom_array)  # [n_atom] (bool)
    noisy_center_coords = (
        torch.from_numpy(atom_array.coord[center_token_mask]) + noise
    )  # [n_token, 3] (float)

    # Create a mask of supported chain types...
    tokens_with_supported_chain_types_mask = np.isin(
        atom_array.chain_type[center_token_mask], allowed_chain_types
    )  # [n_token] (bool)

    # ... and mask of tokens with resolved center atoms
    resolved_tokens_mask = (
        atom_array.occupancy[center_token_mask] > 0
    )  # [n_token] (bool)

    # The tokens to fill are those with supported chain types, resolved center atoms, and non-NaN noise
    token_to_fill_mask = (
        tokens_with_supported_chain_types_mask
        & resolved_tokens_mask
        & torch.isfinite(noise).all(dim=-1).numpy()
    )  # [n_token] (bool)

    # Check if existing annotation exists and force templating where it's True
    if (
        existing_annotation_to_check
        and existing_annotation_to_check in atom_array.get_annotation_categories()
    ):
        existing_annotation_values = atom_array.get_annotation(
            existing_annotation_to_check
        )[center_token_mask]
        forced_template_mask = np.asarray(existing_annotation_values, dtype=bool)
    else:
        forced_template_mask = np.full(_n_token, False, dtype=bool)

    # If unconditional, discard all conditioning...
    if is_unconditional:
        token_to_fill_mask = np.full_like(token_to_fill_mask, False)
    else:
        # Probability of masking each token
        _should_apply_condition = np.random.rand(_n_token) < p_condition_per_token
        token_to_fill_mask = (
            token_to_fill_mask & _should_apply_condition
        ) | forced_template_mask

    token_idxs_to_fill = np.where(token_to_fill_mask)[0]  # [n_token_to_fill] (int)

    # ... fill the template_distogram
    ix1, ix2 = np.ix_(token_idxs_to_fill, token_idxs_to_fill)
    template_distogram[ix1.astype(int), ix2.astype(int)] = torch.cdist(
        noisy_center_coords[token_to_fill_mask],
        noisy_center_coords[token_to_fill_mask],
        compute_mode="donot_use_mm_for_euclid_dist",  # Important for numerical stability
    )

    # (Create n_token x n_token mask, where True indicates a condition); e.g., True for all non-mask tokens
    token_to_fill_mask_II = token_to_fill_mask[:, None] & token_to_fill_mask[None, :]

    # ... mask inter-molecule distances, if required
    if np.random.rand() > p_provide_inter_molecule_distances:
        # Create a mask of tokens that belong to different molecules
        is_inter_molecule = (
            atom_array.molecule_id[center_token_mask][:, None]
            != atom_array.molecule_id[center_token_mask]
        )

        # ... mask inter-molecule distances
        token_to_fill_mask_II[is_inter_molecule] = False
        template_distogram[is_inter_molecule] = MASK_VALUE

    # Discretize distances into bins (NaNs go to last bin)
    template_distogram_binned: Tensor = torch.bucketize(
        template_distogram, boundaries=distogram_bins
    )  # (n_token, n_token)
    n_bins: int = len(distogram_bins) + 1
    template_distogram_onehot: Float[Tensor, "n_token n_token n_bins"] = (
        torch.nn.functional.one_hot(
            template_distogram_binned, num_classes=n_bins
        ).to(torch.float32)
    )

    # Expand noise_scale to (n_token,) if needed
    expanded_noise_scale: Float[Tensor, "n_token"] = (
        noise_scale.expand(_n_token)
        if isinstance(noise_scale, Tensor)
        else torch.full_like(noise, fill_value=noise_scale)
    )
    # Set noise scale to 0 for unconditioned tokens
    expanded_noise_scale[~token_to_fill_mask] = 0.0

    out: dict[str, Tensor] = {
        "distogram_condition_noise_scale": expanded_noise_scale,  # (n_token,)
        "has_distogram_condition": torch.as_tensor(
            token_to_fill_mask_II, dtype=torch.bool
        ),  # (n_token, n_token)
        "distogram_condition": template_distogram_onehot,  # (n_token, n_token, n_bins)
    }

    assert_no_nans(out, msg="Conditioning features contain NaNs!")

    return out


class FeaturizeNoisedGroundTruthAsTemplateDistogram(Transform):
    """Add noised ground truth as a template distogram.

    Creates template features by adding Gaussian noise to the ground truth
    coordinates and converting the resulting distances into a discretized
    distogram.

    Args:
        noise_scale_distribution (Callable): Function that returns the standard
            deviation of noise to add to the ground truth coordinates. Should take
            a sequence of dimensions and return a tensor or float. Default is
            af3_noise_scale_distribution.
        distogram_bins (Tensor): Bin boundaries for discretizing distances in
            the distogram. Shape [n_bins-1].
        allowed_chain_types (list): List of allowed chain types. Default is all chain types.
        p_condition_per_token (float): Per-token probability of conditioning, for those tokens that satisfy all other conditions.
            Default is 0.0 (no conditioning).
        p_provide_inter_molecule_distances (float): Probability of providing inter-molecule (inter-chain) distances.
            Default is 0.0 (no inter-molecule distances provided).
        existing_annotation_to_check (str): Name of an annotation in the AtomArray that,
            if present and True for a token, will force that token to be templated regardless of other conditions.
            Default is "is_input_file_templated".

    Adds the following features to the `feats` dict:
        - "distogram_condition_noise_scale": Noise scale for each
            token [n_token] (float)
        - "has_distogram_condition": Mask indicating which token pairs have a distogram
            condition [n_token, n_token] (bool)
        - "distogram_condition": One-hot encoded distogram
            [n_token, n_token, n_bins] (float)
    """

    requires_previous_transforms = [AtomizeByCCDName]

    def __init__(
        self,
        noise_scale_distribution: NoiseScaleSampler
        | TokenGroupNoiseScaleSampler = af3_noise_scale_distribution,
        distogram_bins: torch.Tensor = DEFAULT_DISTOGRAM_BINS,
        allowed_chain_types: list[ChainType] = ChainType.get_all_types(),
        p_condition_per_token: float = 0.0,
        p_provide_inter_molecule_distances: float = 0.0,
        existing_annotation_to_check: str = "is_input_file_templated",
    ):
        self.distogram_bins = distogram_bins
        self.noise_scale_distribution = noise_scale_distribution
        self.p_provide_inter_molecule_distances = p_provide_inter_molecule_distances
        self.allowed_chain_types = allowed_chain_types
        self.p_condition_per_token = p_condition_per_token
        self.existing_annotation_to_check = existing_annotation_to_check

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["chain_type", "occupancy"])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array = data["atom_array"]

        if isinstance(self.noise_scale_distribution, TokenGroupNoiseScaleSampler):
            # ... different noise scale for each token group
            noise_scale = self.noise_scale_distribution(atom_array)  # [n_token] (float)
        else:
            # ... same noise scale for all tokens
            noise_scale = self.noise_scale_distribution(size=(1,))  # [1] (float)

        template_features = featurize_noised_ground_truth_as_template_distogram(
            atom_array=atom_array,
            noise_scale=noise_scale,
            allowed_chain_types=self.allowed_chain_types,
            distogram_bins=self.distogram_bins,
            p_provide_inter_molecule_distances=self.p_provide_inter_molecule_distances,
            is_unconditional=data.get("is_unconditional", False),
            p_condition_per_token=self.p_condition_per_token,
            existing_annotation_to_check=self.existing_annotation_to_check,
        )

        # Add the template features to the `feats` dict
        if "feats" not in data:
            data["feats"] = {}
        data["feats"].update(template_features)

        return data
