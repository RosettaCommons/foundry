"""
Utilities for computing polymer-ligand interface atoms.

This module provides a transform to identify and annotate polymer atoms that
are at the interface with ligand molecules, defined as atoms within a specified
distance threshold.
"""

from typing import Any

import numpy as np
from atomworks.ml.transforms._checks import check_atom_array_annotation
from atomworks.ml.transforms.base import Transform
from biotite.structure import AtomArray, CellList


class ComputePolymerLigandInterface(Transform):
    """
    Compute polymer and ligand atoms at the polymer-ligand interface and
    annotate the atom array with interface labels.

    An interface atom is defined as any polymer atom that is within the
    distance_threshold of any ligand atom, or vice versa.

    Args:
        distance_threshold (float): Maximum distance in Angstroms for
            considering atoms to be at the interface.
    """

    def __init__(self, distance_threshold: float):
        self.distance_threshold = distance_threshold

    def check_input(self, data: dict[str, Any]) -> None:
        """Check that required annotations are present."""
        check_atom_array_annotation(
            {"atom_array": data["atom_array"]}, required=["element", "atomize"]
        )

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        """Compute polymer-ligand interface and update atom array."""
        atom_array = data["atom_array"]

        # Create a copy to avoid modifying the original.
        result_array = atom_array.copy()

        # Identify polymer and ligand atoms
        polymer_mask, ligand_mask = self._identify_polymer_and_ligand_atoms(
            result_array
        )

        # If no valid atoms, return empty annotations.
        if not np.any(polymer_mask) or not np.any(ligand_mask):
            # If no polymer or ligand atoms found, return empty annotations.
            result_array.set_annotation(
                "at_polymer_ligand_interface",
                np.zeros(result_array.array_length(), dtype=bool),
            )
        else:
            # Extract coordinates for interface calculation
            polymer_atoms = result_array[polymer_mask]
            ligand_atoms = result_array[ligand_mask]

            # Compute interface atoms using efficient spatial search.
            (polymer_interface_indices, ligand_interface_indices) = (
                self._compute_interface_atoms(
                    polymer_atoms,
                    ligand_atoms,
                    polymer_mask,
                    ligand_mask,
                    self.distance_threshold,
                )
            )

            # Annotate the atom array with interface information
            result_array = self._annotate_interface_results(
                result_array, polymer_interface_indices, ligand_interface_indices
            )

        data["atom_array"] = result_array
        return data

    def _identify_polymer_and_ligand_atoms(
        self, atom_array: AtomArray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Identify polymer and ligand atoms in the atom array."""
        # Exclude atoms with invalid coordinates
        has_valid_coords = (~np.isnan(atom_array.coord)).any(axis=1)

        ligand_mask = atom_array.atomize & has_valid_coords
        polymer_mask = ~atom_array.atomize & has_valid_coords

        return polymer_mask, ligand_mask

    def _compute_interface_atoms(
        self,
        polymer_atoms: AtomArray,
        ligand_atoms: AtomArray,
        polymer_mask: np.ndarray,
        ligand_mask: np.ndarray,
        distance_threshold: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute interface atoms using spatial data structures.

        Returns:
            Tuple containing:
            - polymer_indices: Global indices of polymer atoms at interface
            - ligand_indices: Global indices of ligand atoms at interface
        """
        # Build CellList for ligand atoms
        ligand_cell_list = CellList(ligand_atoms, cell_size=distance_threshold)

        # Find polymer atoms within threshold of any ligand.
        polymer_at_interface_mask = ligand_cell_list.get_atoms(
            polymer_atoms.coord, distance_threshold, as_mask=True
        )
        polymer_interface_local_indices = np.where(
            np.any(polymer_at_interface_mask, axis=1)
        )[0]

        # Convert local indices to global indices.
        global_polymer_indices = np.where(polymer_mask)[0]
        polymer_interface_indices = global_polymer_indices[
            polymer_interface_local_indices
        ]

        # Build CellList for polymer atoms.
        polymer_cell_list = CellList(polymer_atoms, cell_size=distance_threshold)

        # Find ligand atoms within threshold of any polymer.
        ligand_at_interface_mask = polymer_cell_list.get_atoms(
            ligand_atoms.coord, distance_threshold, as_mask=True
        )
        ligand_interface_local_indices = np.where(
            np.any(ligand_at_interface_mask, axis=1)
        )[0]

        # Convert local indices to global indices.
        global_ligand_indices = np.where(ligand_mask)[0]
        ligand_interface_indices = global_ligand_indices[ligand_interface_local_indices]

        return (polymer_interface_indices, ligand_interface_indices)

    def _annotate_interface_results(
        self,
        atom_array: AtomArray,
        polymer_interface_indices: np.ndarray,
        ligand_interface_indices: np.ndarray,
    ) -> AtomArray:
        """Annotate the atom array with interface calculation results."""
        n_atoms = atom_array.array_length()

        # Initialize interface annotations.
        at_polymer_ligand_interface = np.zeros(n_atoms, dtype=bool)

        # Mark interface atoms.
        at_polymer_ligand_interface[polymer_interface_indices] = True
        at_polymer_ligand_interface[ligand_interface_indices] = True

        # Add annotation to atom array.
        atom_array.set_annotation(
            "at_polymer_ligand_interface", at_polymer_ligand_interface
        )
        return atom_array
