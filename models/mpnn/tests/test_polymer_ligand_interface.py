import numpy as np
import pytest
from atomworks.io.utils.atom_array_plus import as_atom_array
from biotite.structure import Atom, AtomArray
from mpnn.transforms.polymer_ligand_interface import (
    ComputePolymerLigandInterface,
)


class TestPolymerLigandInterface:
    """Test the polymer-ligand interface detection functionality."""

    def create_sample_atom_array(
        self,
        polymer_coords: np.ndarray,
        ligand_coords: np.ndarray,
        include_nan: bool = False,
    ) -> AtomArray:
        """Create a sample AtomArray with polymer and ligand atoms."""

        atoms = []

        # Add polymer atoms
        for i, coord in enumerate(polymer_coords):
            if include_nan and i == 0:  # Add NaN to first polymer atom
                coord = np.array([np.nan, np.nan, np.nan])

            atom = Atom(
                coord=coord,
                element="C",
                res_name="ALA",
                chain_id="A",
                res_id=i + 1,
                atom_name="CA",
                atomize=False,  # polymer atoms
                occupancy=1.0,
                b_factor=20.0,
            )
            atoms.append(atom)

        # Add ligand atoms
        for i, coord in enumerate(ligand_coords):
            if include_nan and i == 0:  # Add NaN to first ligand atom
                coord = np.array([np.nan, np.nan, np.nan])

            atom = Atom(
                coord=coord,
                element="O",
                res_name="LIG",
                chain_id="B",
                res_id=i + 100,
                atom_name="O1",
                atomize=True,  # ligand atoms
                occupancy=1.0,
                b_factor=20.0,
            )
            atoms.append(atom)

        return as_atom_array(atoms)

    def test_basic_interface_detection(self):
        """Test basic polymer-ligand interface detection."""
        # Create polymer atoms in a line
        polymer_coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],  # Far from ligand
            ]
        )

        # Create ligand atoms near some polymer atoms
        ligand_coords = np.array(
            [
                [0.5, 0.0, 0.0],  # Close to polymer atoms 0 and 1
                [1.5, 0.0, 0.0],  # Close to polymer atoms 1 and 2
            ]
        )

        atom_array = self.create_sample_atom_array(polymer_coords, ligand_coords)

        # Create the interface computer
        interface_computer = ComputePolymerLigandInterface(distance_threshold=3.0)

        # Compute interface
        result_data = interface_computer.forward({"atom_array": atom_array})
        result = result_data["atom_array"]

        # Check annotations exist
        assert hasattr(result, "at_polymer_ligand_interface")

        # Check shapes
        assert len(result.at_polymer_ligand_interface) == atom_array.array_length()

        # Check that the right atoms are identified as interface
        interface_flags = result.at_polymer_ligand_interface

        # Polymer atoms 0, 1, 2 should be at interface (close to ligands)
        # Polymer atom 3 should NOT be at interface (far from ligands)
        assert interface_flags[0] is True  # polymer atom close to ligand
        assert interface_flags[1] is True  # polymer atom close to ligand
        assert interface_flags[2] is True  # polymer atom close to ligand
        assert interface_flags[3] is False  # polymer atom far from ligand

        # All ligand atoms should be at interface (close to polymers)
        assert interface_flags[4] is True  # ligand atom close to polymers
        assert interface_flags[5] is True  # ligand atom close to polymers

    def test_distance_threshold_sensitivity(self):
        """Test that distance threshold correctly affects interface detection."""
        polymer_coords = np.array([[0.0, 0.0, 0.0]])
        ligand_coords = np.array([[2.0, 0.0, 0.0]])  # 2.0 Å away

        atom_array = self.create_sample_atom_array(polymer_coords, ligand_coords)

        # With 1.0 Å threshold - should find no interface
        interface_computer_small = ComputePolymerLigandInterface(distance_threshold=1.0)
        result_small_data = interface_computer_small.forward({"atom_array": atom_array})
        result_small = result_small_data["atom_array"]
        assert not np.any(result_small.at_polymer_ligand_interface)

        # With 3.0 Å threshold - should find interface
        interface_computer_large = ComputePolymerLigandInterface(distance_threshold=3.0)
        result_large_data = interface_computer_large.forward({"atom_array": atom_array})
        result_large = result_large_data["atom_array"]
        assert np.all(
            result_large.at_polymer_ligand_interface
        )  # both atoms should be interface

    def test_nan_coordinates_handling(self):
        """Test that atoms with NaN coordinates are not identified as interface."""
        polymer_coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
        ligand_coords = np.array(
            [
                [0.5, 0.0, 0.0],  # Close to polymers (will be set to NaN)
                [1.5, 0.0, 0.0],  # Valid ligand atom, close to polymer atom 1
            ]
        )

        # Create atom array with NaN coordinates for first polymer and first ligand
        atom_array = self.create_sample_atom_array(
            polymer_coords, ligand_coords, include_nan=True
        )

        interface_computer = ComputePolymerLigandInterface(distance_threshold=3.0)
        result_data = interface_computer.forward({"atom_array": atom_array})
        result = result_data["atom_array"]

        # Check that atoms with NaN coordinates are not identified as interface
        # (First polymer and first ligand have NaN coords)
        interface_flags = result.at_polymer_ligand_interface

        assert interface_flags[0] is False  # polymer atom with NaN coords
        assert (
            interface_flags[1] is True
        )  # polymer atom with valid coords, close to valid ligand
        assert interface_flags[2] is False  # ligand atom with NaN coords
        assert (
            interface_flags[3] is True
        )  # ligand atom with valid coords, close to valid polymer

    def test_no_polymer_atoms(self):
        """Test handling when no polymer atoms are present."""
        # Only ligand atoms
        ligand_coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )

        # Use create_sample_atom_array with empty polymer coords
        atom_array = self.create_sample_atom_array(
            polymer_coords=np.array([]), ligand_coords=ligand_coords
        )

        interface_computer = ComputePolymerLigandInterface(distance_threshold=5.0)
        result_data = interface_computer.forward({"atom_array": atom_array})
        result = result_data["atom_array"]

        # Should have empty interface annotations
        assert not np.any(result.at_polymer_ligand_interface)

    def test_no_ligand_atoms(self):
        """Test handling when no ligand atoms are present."""
        # Only polymer atoms
        polymer_coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )

        # Use create_sample_atom_array with empty ligand coords
        atom_array = self.create_sample_atom_array(
            polymer_coords=polymer_coords, ligand_coords=np.array([])
        )

        interface_computer = ComputePolymerLigandInterface(distance_threshold=5.0)
        result_data = interface_computer.forward({"atom_array": atom_array})
        result = result_data["atom_array"]

        # Should have empty interface annotations
        assert not np.any(result.at_polymer_ligand_interface)

    def test_missing_required_annotations(self):
        """Test that appropriate error is raised when required annotations are missing."""
        # Create atom array without atomize annotation
        polymer_coords = np.array([[0.0, 0.0, 0.0]])
        ligand_coords = np.array([[1.0, 0.0, 0.0]])

        atom_array = self.create_sample_atom_array(polymer_coords, ligand_coords)

        # Remove the atomize annotation
        atom_array.del_annotation("atomize")

        interface_computer = ComputePolymerLigandInterface(distance_threshold=3.0)

        # Should raise an error due to missing atomize annotation
        with pytest.raises(AttributeError):
            interface_computer.forward({"atom_array": atom_array})

    def test_identify_polymer_and_ligand_atoms(self):
        """Test the internal _identify_polymer_and_ligand_atoms method."""
        polymer_coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
        ligand_coords = np.array(
            [
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        )

        atom_array = self.create_sample_atom_array(polymer_coords, ligand_coords)

        interface_computer = ComputePolymerLigandInterface(distance_threshold=3.0)

        # Call the internal method
        polymer_mask, ligand_mask = (
            interface_computer._identify_polymer_and_ligand_atoms(atom_array)
        )

        # Check that we have the right number of each type
        assert np.sum(polymer_mask) == 2  # 2 polymer atoms
        assert np.sum(ligand_mask) == 2  # 2 ligand atoms

        # Check that masks are mutually exclusive
        assert not np.any(polymer_mask & ligand_mask)

        # Check that all atoms are accounted for
        assert np.all(polymer_mask | ligand_mask)

    def test_identify_atoms_with_nan_coords(self):
        """Test that atoms with NaN coordinates are excluded from both masks."""
        polymer_coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
        ligand_coords = np.array(
            [
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        )

        # Create atom array with NaN coordinates
        atom_array = self.create_sample_atom_array(
            polymer_coords, ligand_coords, include_nan=True
        )

        interface_computer = ComputePolymerLigandInterface(distance_threshold=3.0)

        # Call the internal method
        polymer_mask, ligand_mask = (
            interface_computer._identify_polymer_and_ligand_atoms(atom_array)
        )

        # Check that atoms with NaN coordinates are excluded
        # First polymer and first ligand have NaN coords
        assert not polymer_mask[0]  # polymer atom with NaN coords
        assert polymer_mask[1]  # polymer atom with valid coords
        assert not ligand_mask[2]  # ligand atom with NaN coords
        assert ligand_mask[3]  # ligand atom with valid coords
