import numpy as np
from beartype.typing import Any
from biotite.structure import AtomArray
from jaxtyping import Bool, Float

from atomworks.enums import GroundTruthConformerPolicy
from atomworks.ml.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
)
from atomworks.ml.transforms.base import Transform


def add_ground_truth_reference_conformer(
    ref_pos: Float[np.ndarray, "L 3"], ref_pos_is_ground_truth: Bool[np.ndarray, "L"]
) -> Float[np.ndarray, "L 3"]:
    """Adds an additional feature, ref_pos_ground_truth, for exclusively ground truth conformers.

    For residues WITHOUT ground truth conformers, we initialize the feature to 0.
    For residues WITH ground truth conformers, we initialize the feature to the ground truth conformer (same as the ref_pos feature).
    """
    assert isinstance(
        ref_pos, np.ndarray
    ), "ref_pos must be a np array; ensure that this Transform is called before ConvertToTorch"

    ref_pos_ground_truth = np.zeros((ref_pos.shape[0], 3))

    # Where we used the ground truth conformer, add the ground truth conformer to the ref_pos_ground_truth feature
    ref_pos_ground_truth[ref_pos_is_ground_truth] = ref_pos[ref_pos_is_ground_truth]

    return ref_pos_ground_truth


class AddGroundTruthReferenceConformer(Transform):
    """Add an additional feature for exclusively ground truth conformers."""

    requires_previous_transforms = ["GetAF3ReferenceMoleculeFeatures"]

    def check_input(self, data: dict[str, Any]) -> None:
        check_atom_array_annotation(data, ["ground_truth_conformer_policy"])
        check_contains_keys(data, ["atom_array", "feats"])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        data["feats"]["ref_pos_ground_truth"] = add_ground_truth_reference_conformer(
            ref_pos=data["feats"]["ref_pos"],
            ref_pos_is_ground_truth=data["feats"]["ref_pos_is_ground_truth"],
        )
        return data


def noise_residues_with_ground_truth_conformer(
    atom_array: AtomArray, p: float = 0.5, sigma: float = 0.5
) -> AtomArray:
    """For residues that will use the ground truth conformer, add noise to the coordinates with probability p and standard deviation sigma."""
    ref_pos_will_be_ground_truth = (
        atom_array.ground_truth_conformer_policy == GroundTruthConformerPolicy.REPLACE
    )

    # Generate a random mask based on probability p
    noise_mask = np.random.rand(np.sum(ref_pos_will_be_ground_truth)) < p

    # Add noise to the coordinates where the mask is True
    noise = np.random.normal(
        scale=sigma, size=(np.sum(ref_pos_will_be_ground_truth), 3)
    )
    atom_array.coord[ref_pos_will_be_ground_truth][noise_mask] += noise[noise_mask]

    return atom_array


class NoiseResiduesWithGroundTruthConformer(Transform):
    """For residues that will use the ground truth conformer, add noise to the coordinates with probability p and standard deviation sigma."""

    incompatible_previous_transforms = ["GetAF3ReferenceMoleculeFeatures"]

    def __init__(self, p: float = 0.5, sigma: float = 0.5):
        """
        Args:
            p: Per-atom probability of adding noise to the coordinates
            sigma: Standard deviation of the noise to add
        """
        self.p = p
        self.sigma = sigma

    def check_input(self, data: dict[str, Any]) -> None:
        check_atom_array_annotation(data, ["ground_truth_conformer_policy"])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        data["atom_array"] = noise_residues_with_ground_truth_conformer(
            data["atom_array"], p=self.p, sigma=self.sigma
        )
        return data
