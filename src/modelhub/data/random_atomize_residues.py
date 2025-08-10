from typing import Callable

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray
from cifutils.constants import STANDARD_AA, STANDARD_DNA, STANDARD_RNA
from cifutils.utils.selection import get_annotation, get_residue_starts
from datahub.transforms._checks import check_atom_array_annotation
from datahub.transforms.base import Transform
from datahub.utils.token import apply_and_spread_token_wise


def random_atomize_residues(
    atom_array: AtomArray,
    p_atomize: float,
    res_names: list[str] | None = None,
    *,
    rng: np.random.Generator | None = None,
) -> AtomArray:
    """Randomly flag specified residues for atomization based on probability.

    Args:
        atom_array: Input AtomArray
        p_atomize: Probability of atomizing each residue (float between 0 and 1)
        res_names: List of residue names to consider for atomization (e.g., ["ALA", "GLY"])
        rng: Random number generator for reproducibility

    Returns:
        AtomArray with updated atomization annotation
    """
    if not 0 <= p_atomize <= 1:
        raise ValueError("p_atomize must be between 0 and 1")

    res_names = res_names or STANDARD_AA + STANDARD_RNA + STANDARD_DNA

    rng = rng or np.random

    # Get residue-level information
    residue_segments = get_residue_starts(atom_array, add_exclusive_stop=True)
    residue_starts = residue_segments[:-1]
    residue_res_names = atom_array.res_name[residue_starts]

    # Identify target residues and randomly select which to atomize
    is_target_residue = np.isin(residue_res_names, res_names)
    residue_atomize_mask = np.zeros(len(residue_res_names), dtype=bool)

    if np.any(is_target_residue):
        # Randomly select target residues to atomize
        target_indices = np.where(is_target_residue)[0]
        random_values = rng.random(len(target_indices))
        selected_targets = target_indices[random_values < p_atomize]
        residue_atomize_mask[selected_targets] = True

    # Spread residue-level mask to atom-level and update annotation
    atom_atomize_mask = struc.segments.spread_segment_wise(
        residue_segments, residue_atomize_mask
    )
    current_atomize = get_annotation(
        atom_array, "atomize", default=np.zeros(atom_array.array_length(), dtype=bool)
    )
    atom_array.set_annotation("atomize", current_atomize | atom_atomize_mask)

    return atom_array


class RandomAtomizeResidues(Transform):
    """Randomly flag specified residues for atomization based on probability."""

    def __init__(
        self,
        p_atomize: float,
        res_names: list[str] | None = None,
        *,
        rng: np.random.Generator | None = None,
    ):
        """Initialize the transform.

        Args:
            p_atomize: Probability of atomizing each residue (float between 0 and 1)
            res_names: List of residue names to consider for atomization (e.g., ["ALA", "GLY"])
            rng: Random number generator for reproducibility
        """
        self.p_atomize = p_atomize
        self.res_names = res_names
        self.rng = rng or np.random

    def check_input(self, data: dict) -> None:
        check_atom_array_annotation(data, ["res_name"])

    def forward(self, data: dict) -> dict:
        data["atom_array"] = random_atomize_residues(
            data["atom_array"],
            p_atomize=self.p_atomize,
            res_names=self.res_names,
            rng=self.rng,
        )
        return data


def atomize_by_mask(
    atom_array: AtomArray,
    mask: np.ndarray,
) -> AtomArray:
    """Atomize all tokens where any atom matches the given condition."""

    # Check if any atom in each token matches the condition
    token_mask = apply_and_spread_token_wise(atom_array, mask, np.any)

    # Update atomize annotation
    current_atomize = get_annotation(
        atom_array, "atomize", default=np.zeros(atom_array.array_length(), dtype=bool)
    )
    atom_array.set_annotation("atomize", current_atomize | token_mask)

    return atom_array


def atomize_by_annotation(
    atom_array: AtomArray,
    annotation: str,
    condition: callable,
) -> AtomArray:
    """Atomize all tokens where any atom matches the given condition.

    Args:
        atom_array: Input AtomArray
        annotation: Name of the annotation to check
        condition: Callable that takes annotation values and returns boolean mask

    Returns:
        AtomArray with updated atomization annotation
    """
    # Get the annotation values and apply condition
    # (We auto-detect n_body)
    annot_values = get_annotation(atom_array, annotation)
    atom_mask = condition(annot_values)
    return atomize_by_mask(atom_array, atom_mask)


class AtomizeByAnnotation(Transform):
    """Atomize all tokens where any atom matches the given condition."""

    def __init__(
        self,
        annotation: str,
        condition: callable = lambda x: x,
    ):
        """Initialize the transform.

        Args:
            annotation: Name of the annotation to check
            condition: Callable that takes annotation values and returns boolean mask
        """
        self.annotation = annotation
        self.condition = condition

    def forward(self, data: dict) -> dict:
        data["atom_array"] = atomize_by_annotation(
            data["atom_array"],
            annotation=self.annotation,
            condition=self.condition,
        )
        return data


class AtomizeByMaskFunction(Transform):
    """Atomize all tokens where any atom matches the given condition."""

    def __init__(self, mask_function: Callable[[AtomArray], np.ndarray]):
        self.mask_function = mask_function

    def forward(self, data: dict) -> dict:
        data["atom_array"] = atomize_by_mask(
            data["atom_array"],
            self.mask_function(data["atom_array"]),
        )
        return data
