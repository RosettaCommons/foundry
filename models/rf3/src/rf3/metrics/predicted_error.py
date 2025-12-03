from typing import Any

import torch
from rf3.metrics.metric_utils import find_bin_midpoints

from foundry.metrics.metric import Metric


def compute_ptm(
    pae: torch.Tensor,
    to_calculate: torch.Tensor | None,
    max_distance: float = 32,
    bin_count: int = 64,
):
    """Compute the predicted TM-score (PTM) from the predicted aligned error (PAE).

    Args:
        pae: Predicted aligned error tensor.
        to_calculate: Tensor indicating which residues to calculate PTM for.

    Returns:
        ptm: Computed predicted TM-score.
    """
    D, I = pae.shape[0], pae.shape[1]
    if to_calculate is None:
        to_calculate = torch.ones((I, I), dtype=torch.bool, device=pae.device)

    bin_centers = find_bin_midpoints(
        max_distance, bin_count, device=pae.device
    )  # TODO: get this from config
    pae = torch.nn.Softmax(dim=-1)(pae).detach().float()
    normalization_factor = 1.24 * (max(I, 19) - 15.0) ** (1 / 3) - 1.8
    denominator = 1 / (1 + (bin_centers / (normalization_factor)) ** 2)
    pae = pae * denominator[None, None, None, :]  # Broadcast to match pae shape

    pae = pae.sum(dim=-1)  # Sum over the last dimension
    pae = (pae * to_calculate[None]).sum(dim=-1) / (to_calculate.sum(dim=-1) + 1e-6)
    ptm = pae.max(dim=-1).values
    assert ptm.shape == (D,)
    return ptm


class ComputePTM(Metric):
    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "pae": ("network_output", "pae"),
            "asym_id": ("network_input", "f", "asym_id"),
        }

    def compute(
        self,
        pae: torch.Tensor,
        asym_id: torch.Tensor,
    ) -> dict[str, float]:
        """Compute the predicted TM-score (PTM) from the predicted aligned error (PAE).
        Args:
            pae: Predicted aligned error tensor.
            asym_id: AtomArrayStack containing the predicted structure.
        Returns:
            ptm: Computed predicted TM-score.
        """
        ptm = compute_ptm(pae, None)
        # split the batch dimension into separate keys in the output dictionary
        ptm = ptm.cpu().numpy()
        ptm = {f"ptm_{i}": ptm[i] for i in range(len(ptm))}
        return ptm


class ComputeIPTM(Metric):
    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "pae": ("network_output", "pae"),
            "asym_id": ("network_input", "f", "asym_id"),
            "is_ligand": ("network_input", "f", "is_ligand"),
        }

    def compute(
        self,
        pae: torch.Tensor,
        asym_id: torch.Tensor,
        is_ligand: torch.Tensor,
    ) -> dict[str, float]:
        """Compute the predicted interface TM-score (iPTM) from the predicted aligned error (PAE).
        Args:
            pae: Predicted aligned error tensor.
            predicted_atom_array_stack: AtomArrayStack containing the predicted structure.
        Returns:
            iptm: Computed interface TM-score.
        """
        unique, counts = torch.unique(asym_id, return_counts=True)
        to_calculate = asym_id[None, :] != asym_id[:, None]
        iptm = compute_ptm(pae, to_calculate)

        # make a protein - ligand mask
        protein_mask = is_ligand == 0
        ligand_mask = is_ligand == 1
        # calculate iptm for protein-protein, protein-ligand, and ligand-ligand interfaces
        protein_protein_mask = (
            protein_mask[None, :] & protein_mask[:, None] * to_calculate
        )
        protein_ligand_mask = (
            (protein_mask[None, :] & ligand_mask[:, None])
            | (ligand_mask[None, :] & protein_mask[:, None])
        ) * to_calculate
        ligand_ligand_mask = ligand_mask[None, :] & ligand_mask[:, None] * to_calculate
        # calculate iptm for each interface type
        iptm_protein_protein = compute_ptm(pae, protein_protein_mask)
        iptm_protein_ligand = compute_ptm(pae, protein_ligand_mask)
        iptm_ligand_ligand = compute_ptm(pae, ligand_ligand_mask)

        # split the batch dimension into separate keys in the output dictionary
        iptm = iptm.cpu().numpy()
        iptm = {f"iptm_{i}": iptm[i] for i in range(len(iptm))}
        iptm_protein_protein = iptm_protein_protein.cpu().numpy()
        iptm_protein_protein = {
            f"iptm_protein_protein_{i}": iptm_protein_protein[i]
            for i in range(len(iptm_protein_protein))
        }
        iptm_protein_ligand = iptm_protein_ligand.cpu().numpy()
        iptm_protein_ligand = {
            f"iptm_protein_ligand_{i}": iptm_protein_ligand[i]
            for i in range(len(iptm_protein_ligand))
        }
        iptm_ligand_ligand = iptm_ligand_ligand.cpu().numpy()
        iptm_ligand_ligand = {
            f"iptm_ligand_ligand_{i}": iptm_ligand_ligand[i]
            for i in range(len(iptm_ligand_ligand))
        }
        iptm.update(iptm_protein_protein)
        iptm.update(iptm_protein_ligand)
        iptm.update(iptm_ligand_ligand)
        return iptm
