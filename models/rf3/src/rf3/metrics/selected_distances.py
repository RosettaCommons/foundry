import numpy as np
from atomworks.ml.utils import nested_dict
from atomworks.ml.utils.selection import (
    get_mask_from_atom_selection,
    parse_selection_string,
)
from beartype.typing import Any
from biotite.structure import AtomArrayStack

from foundry.metrics.metric import Metric


class SelectedAtomByAtomDistances(Metric):
    """Computes all-by-all 2D distances given a list of selection strings"""

    def compute_from_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        """Override parent class to handle optional selection_strings parameter"""
        compute_inputs = {
            "atom_array_stack": nested_dict.getitem(
                kwargs, key="predicted_atom_array_stack"
            )
        }

        # Add selection_strings only if it exists
        try:
            compute_inputs["selection_strings"] = nested_dict.getitem(
                kwargs, key=("extra_info", "selection_strings")
            )
        except (KeyError, IndexError, TypeError):
            pass

        return self.compute(**compute_inputs)

    def compute(
        self,
        atom_array_stack: AtomArrayStack,
        selection_strings: list[str] | None = None,
    ) -> dict[str, Any]:
        # Short-circuit if no selection strings are provided
        if not selection_strings:
            return {}

        # ... select the specified atoms
        mask = np.zeros(atom_array_stack.array_length(), dtype=bool)
        atom_selections = [parse_selection_string(s) for s in selection_strings]
        for atom_selection in atom_selections:
            mask |= get_mask_from_atom_selection(atom_array_stack, atom_selection)
        selected_atom_array_stack = atom_array_stack[:, mask]

        # Create views with added dimensions for broadcasting
        # coord is (D, L, 3), we want pairwise distances for each D
        coord_i = selected_atom_array_stack.coord[:, :, np.newaxis, :]  # (D, L, 1, 3)
        coord_j = selected_atom_array_stack.coord[:, np.newaxis, :, :]  # (D, 1, L, 3)

        # Calculate pairwise differences and distances
        differences = coord_i - coord_j  # broadcasts to (D, L, L, 3)
        distances = np.linalg.norm(differences, axis=-1)  # (D, L, L)

        # Compute the mean and standard deviation across the D dimension
        mean_distances = np.mean(distances, axis=0)  # Shape: (L, L)
        std_distances = np.std(distances, axis=0)  # Shape: (L, L)

        # Name the features with the chain_id, res_name, res_id, atom_name
        def _format_atom_id(chain_id, res_name, res_id, atom_name):
            return f"{chain_id}/{res_name}/{res_id}/{atom_name}"

        vectorized_format = np.vectorize(_format_atom_id)
        id = vectorized_format(
            selected_atom_array_stack.chain_id,
            selected_atom_array_stack.res_name,
            selected_atom_array_stack.res_id,
            selected_atom_array_stack.atom_name,
        )

        # Create a 2x2 numpy arrays of names, where we concatenate the id ...
        id_i = np.char.add(id, "-")
        id_II = np.char.add(id_i[:, np.newaxis], id[np.newaxis, :])

        # ... and store the results in a dictionary, naming the columns with the concatenated id
        results = {}
        for i in range(len(id)):
            for j in range(
                i + 1, len(id)
            ):  # Only consider j > i to avoid symmetric duplicates
                col_id = id_II[i, j]
                mean = mean_distances[i, j]
                std = std_distances[i, j]
                results[f"{col_id}_mean"] = mean
                results[f"{col_id}_std"] = std

        return results
