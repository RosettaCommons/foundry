"""
Feature aggregation for polymer-ligand interface masks.

This module provides transforms to compute interface masks for polymer residues
that are at the interface with ligand molecules.
"""

from typing import Any

import numpy as np
from atomworks.ml.transforms._checks import check_atom_array_annotation
from atomworks.ml.transforms.base import Transform
from atomworks.ml.utils.token import get_token_starts


class FeaturizePolymerLigandInterfaceMask(Transform):
    """
    Compute a polymer mask indicating which residues are at the polymer-ligand
    interface.

    This transform processes an atom array to identify polymer residues that
    have any atoms within the specified distance threshold of ligand atoms.
    It expects that the atom array already has the
    'at_polymer_ligand_interface' annotation computed by the
    ComputePolymerLigandInterface transform.
    """

    def check_input(self, data: dict[str, Any]) -> None:
        """Check that required annotations are present."""
        check_atom_array_annotation(
            {"atom_array": data["atom_array"]},
            required=["element", "atomize", "at_polymer_ligand_interface"],
        )

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        """Compute polymer-ligand interface mask and add to input_features."""
        atom_array = data["atom_array"]

        # Get interface annotation that should already be computed
        interface_atoms = atom_array.at_polymer_ligand_interface

        # Get token starts to map atoms to residues
        token_starts = get_token_starts(atom_array)

        # Create residue-level interface mask for all tokens
        all_residue_interface_mask = np.zeros(len(token_starts), dtype=bool)

        # For each token (residue), check if any of its atoms are at the
        # interface.
        for i, start_idx in enumerate(token_starts):
            if i < len(token_starts) - 1:
                end_idx = token_starts[i + 1]
            else:
                end_idx = len(atom_array)

            # Check if any atom in this residue is at the interface
            residue_atoms = interface_atoms[start_idx:end_idx]
            all_residue_interface_mask[i] = np.any(residue_atoms)

        # Get token-level atomize annotation
        token_level_array = atom_array[token_starts]
        non_atomized_mask = ~token_level_array.atomize

        # Get interface mask for non-atomized residues only
        polymer_interface_mask = all_residue_interface_mask[non_atomized_mask]

        # Initialize input_features if it doesn't exist.
        if "input_features" not in data:
            data["input_features"] = {}

        # Add the interface mask to input_features
        data["input_features"]["polymer_ligand_interface_mask"] = (
            polymer_interface_mask.astype(np.bool_)
        )

        return data
