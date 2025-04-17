import numpy as np
from beartype.typing import Any
from biotite.structure import AtomArrayStack
from datahub.transforms.sasa import calculate_atomwise_rasa

from modelhub.metrics.base import Metric


class UnresolvedRegionRASA(Metric):
    """
    This metric computes the RASA score for unresolved regions in a protein structure.
    The RASA score is defined as the ratio of the solvent-accessible surface area (SASA)
    of a residue in a protein structure to the SASA of the same residue in an extended conformation.
    """

    def __init__(self, probe_radius: float = 1.4, atom_radii: str | np.ndarray = "ProtOr", point_number: int = 100):
        super().__init__()
        self.probe_radius = probe_radius
        self.atom_radii = atom_radii
        self.point_number = point_number

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "predicted_atom_array_stack": ("predicted_atom_array_stack",),
            "ground_truth_atom_array_stack": ("ground_truth_atom_array_stack",),
        }

    def compute(
        self,
        predicted_atom_array_stack: AtomArrayStack,
        ground_truth_atom_array_stack: AtomArrayStack,
    ) -> dict[str, Any]:
        """
        Compute the RASA score for unresolved regions in a protein structure.

        Args:
            predicted_atom_array (AtomArray): The input atom array representing the  predicted protein structure.
            ground_truth_atom_array (AtomArray): The input atom array representing the ground truth protein structure.
            probe_radius (float, optional): Van-der-Waals radius of the probe in Angstrom. Defaults to 1.4 (for water).
            atom_radii (str | np.ndarray, optional): Atom radii set to use for calculation. Defaults to "ProtOr".
            point_number (int, optional): Number of points in the Shrake-Rupley algorithm to sample for calculating SASA. Defaults to 100.

        Returns:
            dict: A dictionary containing the RASA score and other relevant information.
        """

        # find unresolved regions
        # (polymer atoms with occupancy 0.0)
        atoms_to_score = ground_truth_atom_array_stack.is_polymer & (
            ground_truth_atom_array_stack.occupancy == 0.0
        )
        rasas = []
        # Calculate RASA
        for atom_array in predicted_atom_array_stack:
            rasa = calculate_atomwise_rasa(
                atom_array=atom_array,
                probe_radius=self.probe_radius,
                atom_radii=self.atom_radii,
                point_number=self.point_number,
            )
            rasas.append(rasa[atoms_to_score].mean())
        # Calculate the mean RASA score

        rasa = np.nanmean(rasas)
        output_dictionary = {
            f"unresolved_polymer_rasa_batch{i}": rasa for i, rasa in enumerate(rasas)
        }
        output_dictionary["mean_unresolved_polymer_rasa"] = rasa
        return output_dictionary
