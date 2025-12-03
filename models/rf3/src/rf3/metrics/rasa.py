import numpy as np
from atomworks.ml.transforms.sasa import calculate_atomwise_rasa
from beartype.typing import Any
from biotite.structure import AtomArrayStack

from foundry.metrics.metric import Metric


class UnresolvedRegionRASA(Metric):
    """
    This metric computes the RASA score for unresolved regions in a protein structure.
    The RASA score is defined as the ratio of the solvent-accessible surface area (SASA)
    of a residue in a protein structure to the SASA of the same residue in an extended conformation.
    """

    def __init__(
        self,
        probe_radius: float = 1.4,
        atom_radii: str | np.ndarray = "ProtOr",
        point_number: int = 100,
        include_resolved: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.probe_radius = probe_radius
        self.atom_radii = atom_radii
        self.point_number = point_number
        self.include_resolved = include_resolved

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
        """Compute the RASA score for unresolved regions in a protein structure.

        Args:
            predicted_atom_array (AtomArray): The input atom array representing the  predicted protein structure.
            ground_truth_atom_array (AtomArray): The input atom array representing the ground truth protein structure.
            probe_radius (float, optional): Van-der-Waals radius of the probe in Angstrom. Defaults to 1.4 (for water).
            atom_radii (str | np.ndarray, optional): Atom radii set to use for calculation. Defaults to "ProtOr".
            point_number (int, optional): Number of points in the Shrake-Rupley algorithm to sample for calculating SASA. Defaults to 100.
            include_resolved (bool, optional): Whether to include resolved regions in the RASA score. Defaults to False.

        Returns:
            dict: A dictionary containing the RASA score and other relevant information.
        """

        # find unresolved regions
        # (polymer atoms with occupancy 0.0)
        atoms_to_score_unresolved = ground_truth_atom_array_stack.is_polymer & (
            ground_truth_atom_array_stack.occupancy == 0.0
        )

        # find resolved regions (polymer atoms with occupancy > 0.0)
        atoms_to_score_resolved = ground_truth_atom_array_stack.is_polymer & (
            ground_truth_atom_array_stack.occupancy > 0.0
        )

        unresolved_rasas = []
        resolved_rasas = []

        # Calculate RASA
        for atom_array in predicted_atom_array_stack:
            try:
                rasa = calculate_atomwise_rasa(
                    atom_array=atom_array,
                    probe_radius=self.probe_radius,
                    atom_radii=self.atom_radii,
                    point_number=self.point_number,
                )
                unresolved_rasas.append(rasa[atoms_to_score_unresolved].mean())
                if self.include_resolved:
                    resolved_rasas.append(rasa[atoms_to_score_resolved].mean())
            except KeyError:
                unresolved_rasas.append(np.nan)
                if self.include_resolved:
                    resolved_rasas.append(np.nan)

        # Calculate the mean RASA scores
        # Pattern-match other metrics by appending "_i" to the metric name to represent multiple batches
        # (e.g., "unresolved_polymer_rasa_0", "unresolved_polymer_rasa_1", etc.)
        unresolved_rasa = np.nanmean(unresolved_rasas)
        output_dictionary = {
            f"unresolved_polymer_rasa_{i}": rasa
            for i, rasa in enumerate(unresolved_rasas)
        }
        output_dictionary["mean_unresolved_polymer_rasa"] = unresolved_rasa

        # ...  and add resolved region RASA scores if flag is enabled
        if self.include_resolved:
            resolved_rasa = np.nanmean(resolved_rasas)
            output_dictionary.update(
                {
                    f"resolved_polymer_rasa_{i}": rasa
                    for i, rasa in enumerate(resolved_rasas)
                }
            )
            output_dictionary["mean_resolved_polymer_rasa"] = resolved_rasa

        return output_dictionary
