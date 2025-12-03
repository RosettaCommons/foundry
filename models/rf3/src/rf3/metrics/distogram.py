from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from atomworks.ml.utils.token import get_af3_token_representative_idxs
from beartype.typing import Any, Literal
from biotite.structure import AtomArrayStack
from einops import rearrange, repeat
from jaxtyping import Bool, Float
from rf3.loss.af3_losses import distogram_loss

from foundry.metrics.metric import Metric
from foundry.utils.torch import assert_no_nans


@dataclass
class ComparisonConfig:
    """Configuration for token pair comparisons in distogram metrics."""

    token_a: Literal["all", "atomized", "non_atomized"] = "all"
    token_b: Literal["all", "atomized", "non_atomized"] = "all"
    relationship: Literal["all", "inter", "intra"] = "all"

    def __eq__(self, other):
        """Equality that accounts for token_a/token_b symmetry."""
        if not isinstance(other, type(self)):
            return False

        return self.relationship == other.relationship and {
            self.token_a,
            self.token_b,
        } == {other.token_a, other.token_b}

    def __hash__(self):
        """Hash function compatible with the equality definition."""
        return hash((frozenset([self.token_a, self.token_b]), self.relationship))

    def __str__(self):
        """String representation of the comparison config."""
        name = f"{self.token_a}_by_{self.token_b}"
        if self.relationship != "all":
            name += f"_{self.relationship}"
        return name

    def create_distogram_mask(
        self, token_rep_atom_array: AtomArrayStack
    ) -> Bool[np.ndarray, "I I"]:
        """Create a token-by-token mask indiciating which 2D pairs satisfy the ComparisonConfig's conditions."""
        type_masks = {
            "all": np.ones(len(token_rep_atom_array), dtype=bool),
            "atomized": token_rep_atom_array.atomize,
            "non_atomized": ~token_rep_atom_array.atomize,
        }
        # Create token pair mask
        if self.token_a == self.token_b:
            # (Both same)
            token_pair_mask = np.outer(
                type_masks[self.token_a], type_masks[self.token_b]
            )
        else:
            # (Different - must be symmetric)
            token_pair_mask = np.outer(
                type_masks[self.token_a], type_masks[self.token_b]
            ) | np.outer(type_masks[self.token_b], type_masks[self.token_a])

        # Apply relationship constraint
        if self.relationship != "all":
            intra_mask = np.equal.outer(
                token_rep_atom_array.pn_unit_iid, token_rep_atom_array.pn_unit_iid
            )
            if self.relationship == "intra":
                # Same chain ("intra")
                token_pair_mask = token_pair_mask & intra_mask
            else:
                # Different chains ("inter")
                token_pair_mask = token_pair_mask & (~intra_mask)

        return token_pair_mask


class DistogramLoss(Metric):
    """Computes the distogram loss, taking into account the coordinate mask."""

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "pred_distogram": ("network_output", "distogram"),
            "X_rep_atoms_I": ("extra_info", "coord_token_lvl"),
            "crd_mask_rep_atoms_I": ("extra_info", "mask_token_lvl"),
        }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cce_loss = nn.CrossEntropyLoss(reduction="none")

    def compute(
        self,
        pred_distogram: Float[torch.Tensor, "I I n_bins"],
        X_rep_atoms_I: Float[torch.Tensor, "I 3"],
        crd_mask_rep_atoms_I: Float[torch.Tensor, "I"],
    ) -> dict[str, Any]:
        """Computes the distogram loss.

        Args:
            pred_distogram: The predicted distogram. Shape: [I, I, n_bins], where n_bins is the number of bins (64 + 1 = 65).
            X_rep_atoms_I: The ground-truth coordinates of the representative atoms for each token. Shape: [I, 3].
            crd_mask_rep_atoms_I: A boolean mask indicating which representative atoms are present. Shape: [I].
        """
        loss = distogram_loss(
            pred_distogram, X_rep_atoms_I, crd_mask_rep_atoms_I, self.cce_loss
        )
        return {"distogram_loss": loss.detach().item()}


def bin_distances(
    coords: Float[torch.Tensor, "... L 3"],
    min_distance: int = 2,
    max_distance: int = 22,
    n_bins: int = 64,
) -> Float[torch.Tensor, "... L L {n_bins}+1"]:
    # TODO: Refactor loss to use this function instead (more re-usable)
    """Converts coordinates into binned distances according to the given parameters.

    NOTE: Our returned number of bins will be n_bins + 1, as torch.bucketize adds an additional bin for values greater than the maximum.

    Args:
        coords (torch.Tensor): The input tensor of coordinates. May be batched.
        min_distance (float): The minimum distance for binning.
        max_distance (float): The maximum distance for binning.
        n_bins (int): The number of bins to use.

    Returns:
        torch.Tensor: The binned distances.
    """
    # Compute pairwise distances
    distance_map = torch.cdist(coords, coords)

    # (Replace NaN's with a large value to avoid issues with bucketize)
    distance_map = torch.nan_to_num(distance_map, nan=9999.0)

    # ... bin the distances
    n_bins = torch.linspace(min_distance, max_distance, n_bins).to(coords.device)
    binned_distances = torch.bucketize(distance_map, n_bins)

    return binned_distances


def masked_distogram_cross_entropy_loss(
    input: Float[torch.Tensor, "D I I n_bins"],
    target: Float[torch.Tensor, "D I I"],
    mask: Float[torch.Tensor, "I I"] = None,
) -> Float[torch.Tensor, "D"]:
    # TODO: Refactor loss to use this function instead (more re-usable)
    """Computes the masked cross-entropy between two distograms.

    Note that the cross-entropy loss is not symmetric; that is, H(x, y) != H(y, x).
    """
    # From the PyTorch documentation (where C = number of classes, N = batch size):
    # > Input: Shape: (C), (N, C) or (N, C, d1, d2, ..., dk)
    # > Target: Shape: (N) or (N, d1, d2, ..., dk) where each value should be between [0, C)
    input = rearrange(input, "d i j n_bins -> d n_bins i j")
    loss = F.cross_entropy(input, target, reduction="none")

    # Apply mask and normalize
    masked_loss = loss * mask if mask is not None else loss
    normalized_loss = masked_loss.sum(dim=(-1, -2)) / mask.sum() + 1e-4  # [D]

    return normalized_loss


class DistogramComparisons(Metric):
    """Compares model distogram representations.

    Namely:
        - The representation from the TRUNK vs. GROUND TRUTH
        - The representation from the TRUNK vs. PREDICTED COORDINATES

    We subset to specific token pairs based on the provided ComparisonConfig.
    """

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "X_L": ("network_output", "X_L"),  # [D, L, 3]
            "trunk_pred_distogram": (
                "network_output",
                "distogram",
            ),  # [I, I, 65], where 65 is the number of bins (64 + 1)
            "ground_truth_atom_array_stack": "ground_truth_atom_array_stack",
            "X_rep_atoms_I": ("extra_info", "coord_token_lvl"),  # [D, I, 3]
            "crd_mask_rep_atoms_I": ("extra_info", "mask_token_lvl"),  # [D, I]
        }

    def __init__(
        self, comparison_configs: list[ComparisonConfig] | None = None, **kwargs
    ):
        """
        Args:
            comparison_configs: List of ComparisonConfig objects defining which comparisons to compute.
        """
        super().__init__(**kwargs)

        if comparison_configs is None:
            # Default comparisons
            comparison_configs = [
                ComparisonConfig("atomized", "atomized", "intra"),
                ComparisonConfig("atomized", "non_atomized", "inter"),
                ComparisonConfig("non_atomized", "non_atomized", "intra"),
                ComparisonConfig("all", "all", "all"),
            ]

        # Deduplicate (handle symmetries in token_a/token_b)
        self.comparison_configs = list(set(comparison_configs))

    def compute(
        self,
        X_L: Float[torch.Tensor, "D L 3"],
        trunk_pred_distogram: Float[torch.Tensor, "I I n_bins"],
        ground_truth_atom_array_stack: AtomArrayStack,
        X_rep_atoms_I: Float[torch.Tensor, "D I 3"] | None = None,
        crd_mask_rep_atoms_I: Float[torch.Tensor, "D I"] | None = None,
    ) -> dict[str, Any]:
        """Computes the distogram loss for the trunk vs. ground truth and trunk vs. predicted coordinates.

        Optionally, we also subset to intra-ligand (atomized) distances.

        Args:
            X_L: The predicted coordinates. Shape: [D, L, 3]
            trunk_pred_distogram: The prediction from the DistogramHead, which linearly projects the trunk features. Shape: [I, I, n_bins]
            ground_truth_atom_array_stack: The ground-truth atom array stack, one model per diffusion sample. Shape: [D, L]
            X_rep_atoms_I: The ground-truth coordinates of the representative atoms for each token. Shape: [D, I, 3]. If None, will be inferred from the ground_truth_atom_array_stack.
            crd_mask_rep_atoms_I: A boolean mask indicating which representative atoms are present. Shape: [D, I]. If None, will be inferred from the ground_truth_atom_array_stack.
        """
        MIN_PAIRS = 15
        results = {}

        # ... choose the first model, as we only care about 2D distance (frame-invariant)
        ground_truth_atom_array = ground_truth_atom_array_stack[0]

        _token_rep_idxs = get_af3_token_representative_idxs(ground_truth_atom_array)
        token_rep_idxs = torch.from_numpy(_token_rep_idxs).to(X_L.device)
        token_rep_atom_array = ground_truth_atom_array[_token_rep_idxs]

        # Create 2D coordinate mask for valid pairs of representative atoms
        if crd_mask_rep_atoms_I is None:
            # (If not provided, we will use the occupancy mask)
            crd_mask_rep_atoms_I = torch.from_numpy(
                token_rep_atom_array.occupancy > 0
            ).to(X_L.device)

        crd_mask_rep_atom_II = crd_mask_rep_atoms_I.unsqueeze(
            -1
        ) * crd_mask_rep_atoms_I.unsqueeze(-2)

        # Prepare distograms
        # (From the ground truth)
        if X_rep_atoms_I is None:
            # (If not provided, we will use the coordinates of the representative atoms)
            X_rep_atoms_I = torch.from_numpy(token_rep_atom_array.coord).to(X_L.device)
        binned_distogram_from_ground_truth = bin_distances(X_rep_atoms_I, n_bins=64)
        # (Predicted coordinates are batched, so we build the distogram for each predicted structure)
        binned_distogram_from_pred_coords = bin_distances(
            X_L[:, token_rep_idxs], n_bins=64
        )

        for config in self.comparison_configs:
            # ... create a token-by-token mask for this config, specifying which 2D pairs to compare
            token_pair_mask = config.create_distogram_mask(token_rep_atom_array)
            mask = (
                torch.from_numpy(token_pair_mask).to(X_L.device) & crd_mask_rep_atom_II
            )
            if mask.sum() < MIN_PAIRS:
                # (Skip if not enough pairs so we do not dilute our average)
                continue

            # ... generate a descriptive name for this config
            name = str(config)

            # Compute trunk vs. ground truth
            results[f"trunk_vs_ground_truth_cce_{name}"] = (
                masked_distogram_cross_entropy_loss(
                    trunk_pred_distogram.unsqueeze(0),
                    binned_distogram_from_ground_truth.unsqueeze(0),
                    mask,
                )
                .detach()
                .item()
            )

            # Compute trunk vs. predicted coordinates
            losses = masked_distogram_cross_entropy_loss(
                repeat(
                    trunk_pred_distogram,
                    "i j n_bins -> d i j n_bins",
                    d=binned_distogram_from_pred_coords.shape[0],
                ),
                binned_distogram_from_pred_coords,
                mask,
            )
            results.update(
                {
                    f"trunk_vs_pred_coords_cce_{name}_{i}": loss.detach().item()
                    for i, loss in enumerate(losses)
                }
            )

        return results


class DistogramEntropy(Metric):
    """Computes the entropy of the predicted distogram, subset to specific token pairs."""

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "trunk_pred_distogram": (
                "network_output",
                "distogram",
            ),  # [I, I, 65], where 65 is the number of bins (64 + 1)
            "ground_truth_atom_array_stack": "ground_truth_atom_array_stack",
            "crd_mask_rep_atoms_I": ("extra_info", "mask_token_lvl"),  # [D, I]
        }

    def __init__(
        self, comparison_configs: list[ComparisonConfig] | None = None, **kwargs
    ):
        """
        Args:
            comparison_configs: List of ComparisonConfig objects defining which comparisons to compute.
                If None, uses predefined configurations for atomized and non-atomized pairs.
        """
        super().__init__(**kwargs)

        if comparison_configs is None:
            # Default comparisons
            self.comparison_configs = [
                ComparisonConfig(
                    token_a="atomized", token_b="atomized", relationship="intra"
                ),  # Atomized-Atomized Intra
                ComparisonConfig(
                    token_a="non_atomized", token_b="non_atomized", relationship="intra"
                ),  # Non-Atomized-Non-Atomized Intra
                ComparisonConfig(
                    token_a="all", token_b="all", relationship="inter"
                ),  # All-All Inter
                ComparisonConfig(
                    token_a="all", token_b="all", relationship="all"
                ),  # All-All (everything)
            ]
        else:
            # Use provided comparison configurations
            self.comparison_configs = comparison_configs

    def compute(
        self,
        trunk_pred_distogram: Float[torch.Tensor, "I I n_bins"],
        ground_truth_atom_array_stack: AtomArrayStack,
        crd_mask_rep_atoms_I: Float[torch.Tensor, "D I"] | None = None,
    ) -> dict[str, Any]:
        """Computes the entropy of the predicted distogram distributions for different token pair subsets."""
        MIN_PAIRS = 15
        results = {}

        # Get the first model from the atom array stack
        ground_truth_atom_array = ground_truth_atom_array_stack[0]
        token_rep_atom_array = ground_truth_atom_array[
            get_af3_token_representative_idxs(ground_truth_atom_array)
        ]

        # Create 2D coordinate mask for valid pairs of representative atoms
        if crd_mask_rep_atoms_I is None:
            crd_mask_rep_atoms_I = torch.from_numpy(
                token_rep_atom_array.occupancy > 0
            ).to(trunk_pred_distogram.device)
        crd_mask_rep_atom_II = crd_mask_rep_atoms_I.unsqueeze(
            -1
        ) * crd_mask_rep_atoms_I.unsqueeze(-2)

        # Compute entropy for each comparison configuration
        for config in self.comparison_configs:
            # Create a token-by-token mask for this config, specifying which 2D pairs to analyze
            token_pair_mask = config.create_distogram_mask(
                token_rep_atom_array
            )  # [I, I]
            mask = (
                torch.from_numpy(token_pair_mask).to(trunk_pred_distogram.device)
                & crd_mask_rep_atom_II
            )  # [I, I]

            if mask.sum() < MIN_PAIRS:
                # Skip if not enough pairs to avoid diluting our average
                continue

            # Generate a descriptive name for this config
            name = str(config)

            # ... convert to probabilities via softmax
            trunk_pred_distogram_probs = torch.nn.functional.softmax(
                trunk_pred_distogram, dim=-1
            )

            # Compute entropy: -sum(p * log(p)) for each distribution
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            entropy = -torch.sum(
                trunk_pred_distogram_probs
                * torch.log(trunk_pred_distogram_probs + epsilon),
                dim=-1,
            )  # [I, I]

            # Apply mask and compute average entropy
            masked_entropy = entropy * mask
            assert_no_nans(masked_entropy)

            avg_entropy = masked_entropy.sum() / (mask.sum() + 1e-6)

            results[f"distogram_entropy_{name}"] = avg_entropy.detach().item()

        return results
