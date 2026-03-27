import logging

import numpy as np

from foundry.metrics.metric import Metric
from foundry.utils.ddp import RankedLogger

logging.basicConfig(level=logging.INFO)
global_logger = RankedLogger(__name__, rank_zero_only=False)


def calculate_ligand_contacts(
    atom_array_stack,
    cutoff_distance=4.0,
):
    """
    Count number of atom contacts within cutoff of any ligand atom.

    Parameters
    ----------
    atom_array_stack : AtomArrayStack
        Shape: (n_models, n_atoms)
    cutoff_distance : float
        Distance cutoff in Å

    Returns
    -------
    total_contacts : int
    mean_contacts_per_model : float
    """

    cutoff_sq = cutoff_distance**2
    contacts_per_model = []

    n_models = len(atom_array_stack)

    for i in range(n_models):
        atoms = atom_array_stack[i]

        coords = atoms.coord
        hetero_mask = atoms.hetero.astype(bool)

        # Skip if no ligand
        if not np.any(hetero_mask):
            contacts_per_model.append(0)
            continue

        ligand_coords = coords[hetero_mask]
        non_ligand_coords = coords[~hetero_mask]

        if len(non_ligand_coords) == 0:
            contacts_per_model.append(0)
            continue

        # Pairwise squared distances
        diff = non_ligand_coords[:, None, :] - ligand_coords[None, :, :]
        dist_sq = np.sum(diff**2, axis=-1)

        # Any ligand within cutoff
        contact_mask = np.any(dist_sq < cutoff_sq, axis=1)

        n_contacts = np.sum(contact_mask)
        contacts_per_model.append(n_contacts)

    contacts_per_model = np.array(contacts_per_model)

    return (
        int(np.sum(contacts_per_model)),
        float(np.mean(contacts_per_model)),
        float(np.mean(contacts_per_model)) / hetero_mask.sum(),
    )


class LigandContactMetrics(Metric):
    def __init__(
        self,
        *,
        cutoff_distance: float = 4.0,
        restrict_to_nucleic: bool = True,
    ):
        super().__init__()
        self.cutoff_distance = cutoff_distance
        self.restrict_to_nucleic = restrict_to_nucleic

    @property
    def kwargs_to_compute_args(self):
        return {
            "predicted_atom_array_stack": ("predicted_atom_array_stack",),
        }

    def compute(self, *, predicted_atom_array_stack):
        if self.restrict_to_nucleic:
            if (
                predicted_atom_array_stack[0].is_rna.sum()
                + predicted_atom_array_stack[0].is_dna.sum()
                == 0
            ):
                return {}
        try:
            total_contacts, mean_contacts, mean_contacts_per_atom = (
                calculate_ligand_contacts(
                    atom_array_stack=predicted_atom_array_stack,
                    cutoff_distance=self.cutoff_distance,
                )
            )
        except Exception as e:
            global_logger.error(
                f"Error calculating ligand contact metrics: {e} | Skipping"
            )
            return {}

        return {
            "mean_ligand_contacts_per_model": float(mean_contacts),
            "mean_ligand_contacts_per_atom": float(mean_contacts_per_atom),
        }
