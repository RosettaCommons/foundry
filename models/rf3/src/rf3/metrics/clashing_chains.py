import itertools
from typing import Any

import torch
from biotite.structure import AtomArrayStack

from foundry.metrics.metric import Metric


class CountClashingChains(Metric):
    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "X_L": ("network_output", "X_L"),
            "predicted_atom_array_stack": ("predicted_atom_array_stack",),
        }

    def compute(
        self,
        X_L: torch.Tensor,
        predicted_atom_array_stack: AtomArrayStack,
    ) -> dict[str, float]:
        """Compute the predicted interface TM-score (IPTM) from the predicted aligned error (PAE).
        Args:
            X_L: Predicted aligned error tensor.
            predicted_atom_array_stack: AtomArrayStack containing the predicted structure.
        Returns:
            clash_count: Computed clashing chains count.
        """
        D = X_L.shape[0]
        MIN_CLASH_DISTANCE = 1.1  # Minimum distance to consider a clash
        # Count the number of clashing chains
        pn_units = set(predicted_atom_array_stack.pn_unit_id)

        has_clash = torch.zeros((D), dtype=torch.bool)
        for chain_i, chain_j in itertools.combinations(pn_units, 2):
            # check to make sure they are both polymer chains

            chain_i_atoms = predicted_atom_array_stack[
                :, predicted_atom_array_stack.pn_unit_id == chain_i
            ]
            chain_j_atoms = predicted_atom_array_stack[
                :, predicted_atom_array_stack.pn_unit_id == chain_j
            ]
            if not chain_i_atoms[0, 0].is_polymer or not chain_j_atoms[0, 0].is_polymer:
                continue

            distances = torch.cdist(
                torch.from_numpy(chain_i_atoms.coord),
                torch.from_numpy(chain_j_atoms.coord),
            )
            num_clashes = (distances < MIN_CLASH_DISTANCE).sum(dim=-1).sum(dim=-1)
            has_clash_pair = (num_clashes > 100) | (
                num_clashes
                / (
                    max(chain_i_atoms.coord.shape[0], chain_j_atoms.coord.shape[0])
                    + 1e-6
                )
                > 0.5
            )
            has_clash = torch.logical_or(has_clash, has_clash_pair)
        assert has_clash.shape == (D,)
        # unpack the batch dimension into separate keys in the output dictionary

        clash_count_per_batch = {
            f"has_clash_{i}": int(has_clash[i]) for i in range(len(has_clash))
        }
        return clash_count_per_batch
