import torch
from beartype.typing import Any

from modelhub.metrics.base import Metric


def calc_lddt(
    X_L,
    X_gt_L,
    crd_mask_L,
    tok_idx,
    pairs_to_score=None,
    distance_cutoff=15.0,
    eps=1e-6,
):
    """Calculates LDDT scores.

    Args:
        X_L: Predicted coordinates (D, L, 3).
        X_gt_L: Ground truth coordinates (D, L, 3).
        crd_mask_L: Coordinate mask (D, L).
        tok_idx: Token index of each atom (L,).
        pairs_to_score: Pairs to score (L, L) or None.
        distance_cutoff: Distance cutoff for scoring.
        eps: Small epsilon to prevent division by zero.

    Returns:
        LDDT scores as a tensor.
    """
    # TODO: Refactor for clarity
    D, L = X_L.shape[:2]
    if pairs_to_score is None:
        pairs_to_score = torch.ones((L, L), dtype=torch.bool).triu(0).to(X_L.device)
    else:
        assert pairs_to_score.shape == (L, L)
        pairs_to_score = pairs_to_score.triu(0).to(X_L.device)

    first_index, second_index = torch.nonzero(pairs_to_score, as_tuple=True)

    lddt = []
    for d in range(D):
        ground_truth_distances = torch.linalg.norm(
            X_gt_L[d, first_index] - X_gt_L[d, second_index], dim=-1
        )

        pair_mask = torch.logical_and(
            ground_truth_distances > 0, ground_truth_distances < distance_cutoff
        )

        # only score pairs that are resolved in the ground truth
        pair_mask *= crd_mask_L[d, first_index] * crd_mask_L[d, second_index]
        # don't score pairs that are in the same token
        pair_mask *= tok_idx[first_index] != tok_idx[second_index]

        valid_pairs = pair_mask.nonzero(as_tuple=True)
        pair_mask = pair_mask[valid_pairs].to(X_L.dtype)
        ground_truth_distances = ground_truth_distances[valid_pairs]
        first_index, second_index = first_index[valid_pairs], second_index[valid_pairs]

        predicted_distances = torch.linalg.norm(
            X_L[d, first_index] - X_L[d, second_index], dim=-1
        )

        delta_distances = torch.abs(predicted_distances - ground_truth_distances + eps)
        del predicted_distances, ground_truth_distances

        lddt.append(
            0.25
            * (
                torch.sum((delta_distances < 4.0) * pair_mask)
                + torch.sum((delta_distances < 2.0) * pair_mask)
                + torch.sum((delta_distances < 1.0) * pair_mask)
                + torch.sum((delta_distances < 0.5) * pair_mask)
            )
            / (torch.sum(pair_mask) + eps)
        )

    return torch.tensor(lddt)


class AllAtomLDDT(Metric):
    """Computes all-atom LDDT scores."""

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "X_L": ("network_output", "X_L"),
            "X_gt_L": ("extra_info", "X_gt_L"),
            "crd_mask_L": ("extra_info", "crd_mask_L"),
            "tok_idx": ("network_input", "f", "atom_to_token_map"),
        }

    def compute(
        self,
        X_L: torch.Tensor,
        X_gt_L: torch.Tensor,
        crd_mask_L: torch.Tensor,
        tok_idx: torch.Tensor,
    ) -> dict:
        """Calculates all-atom LDDT.

        Args:
            X_L: Predicted coordinates (D, L, 3).
            X_gt_L: Ground truth coordinates (D, L, 3).
            crd_mask_L: Coordinate mask (D, L), indicating which atoms are resolved.
            tok_idx: Atom-level map to token index (L,).

        Returns:
            A dictionary with all-atom LDDT scores.
        """
        tok_idx = tok_idx.cpu().numpy()

        all_atom_lddt = calc_lddt(
            X_L=X_L,
            X_gt_L=X_gt_L,
            crd_mask_L=crd_mask_L,
            tok_idx=torch.tensor(tok_idx).to(X_L.device),
            pairs_to_score=None,  # By default, score all pairs, except those within the same token
            distance_cutoff=15.0,
        )

        return {
            "best_of_1_lddt": all_atom_lddt[0].item(),
            f"best_of_{len(all_atom_lddt)}_lddt": all_atom_lddt.max().item(),
        }


# TODO: Rewrite with new Metrics API
class ByTypeInterfaceLDDT(Metric):
    """Computes interface LDDT, grouped by interface type"""

    def compute(
        self, network_input: dict, network_output: dict, extra_info: dict, **kwargs
    ) -> dict:
        """Calculates interface LDDT.

        Args:
            network_input: Network input data.
            network_output: Network output data.
            extra_info: Additional data for metric computation.
        """
        # Short-circuit
        if "interfaces_to_score" not in extra_info:
            return []

        interface_results = []

        # Map from token to pn_unit_iid
        pn_unit_iid_token_lvl = extra_info["chain_iid_token_lvl"]  # [n_tokens]

        # Map from atom to token
        tok_idx = network_input["f"]["atom_to_token_map"].cpu().numpy()  # [n_atoms]

        # Loop over the interfaces to score (e.g., pn_unit_i, pn_unit_j, interface_type)
        interfaces_to_score = eval(extra_info["interfaces_to_score"]) if isinstance(extra_info["interfaces_to_score"], str) else extra_info["interfaces_to_score"]
        for pn_unit_i, pn_unit_j, interface_type in interfaces_to_score:
            # Get tokens in pn_unit_i and pn_unit_j
            pn_unit_i_tokens = pn_unit_iid_token_lvl == pn_unit_i
            pn_unit_j_tokens = pn_unit_iid_token_lvl == pn_unit_j

            # Convert the token level to the atom level
            pn_unit_i_atoms = pn_unit_i_tokens[tok_idx]
            pn_unit_j_atoms = pn_unit_j_tokens[tok_idx]

            # Compute the outer product of chain_i and chain_j, which represents the interface
            chain_ij_atoms = torch.einsum(
                "L, K -> LK",
                torch.tensor(pn_unit_i_atoms),
                torch.tensor(pn_unit_j_atoms),
            ).to(network_output["X_L"].device)

            # Symmetrize the interface so we can later multiply with an upper triangular without losing information
            chain_ij_atoms = chain_ij_atoms | chain_ij_atoms.T

            # compute lddt using the pairs_to_score from the intersection
            lddt = calc_lddt(
                network_output["X_L"],
                extra_info["X_gt_L"],
                extra_info["crd_mask_L"],
                torch.tensor(tok_idx).to(network_output["X_L"].device),
                pairs_to_score=chain_ij_atoms,
                distance_cutoff=30.0,
            )

            # add the results to the interface_results list
            n = len(lddt)
            result = {
                "pn_units": [pn_unit_i, pn_unit_j],
                "type": interface_type,
                "best_of_1_lddt": lddt[0].item(),
                f"best_of_{n}_lddt": lddt.max().item(),
            }

            # if confidence features are present, add them
            if "confidence" in network_output:
                pae_idx = network_output["confidence"]["pae_idx"]
                pde_idx = network_output["confidence"]["pde_idx"]
                plddt_idx = network_output["confidence"]["plddt_idx"]
                # TODO: This lookup would be best implemented as a sorted Tuple of PN Unit IIDs or a symmetric 2D lookup table rather than with non-symmeterized strings
                af3_style_ipae_idx = network_output["confidence"]["best_interface_idx"][
                    f"{pn_unit_i}-{pn_unit_j}"
                ]
                result.update(
                    {
                        "oracle_by_pae": lddt[pae_idx].item(),
                        "oracle_by_pde": lddt[pde_idx].item(),
                        "oracle_by_plddt": lddt[plddt_idx].item(),
                        "oracle_by_af3_style_ipae": lddt[af3_style_ipae_idx].item(),
                        "oracle_by_af3_style_lig_ipae": lddt[
                            network_output["confidence"]["best_lig_ipae_idx"][
                                f"{pn_unit_i}-{pn_unit_j}"
                            ]
                        ].item(),
                    }
                )

            interface_results.append(result)

        return interface_results


# TODO: Rewrite with new Metrics API
class ChainLDDTByType(Metric):
    """Computes chain-wise LDDT, grouped by chain type"""

    def compute(
        self, network_input: dict, network_output: dict, extra_info: dict, **kwargs
    ) -> dict:
        """Calculates chain (PN unit) LDDT.

        Args:
            network_input: Network input data.
            network_output: Network output data.
            extra_info: Additional data for metric computation.

        Returns:
            A dictionary with chain LDDT scores.
        """
        if "pn_units_to_score" not in extra_info:
            return []

        chain_results = []

        chain_iid_token_lvl = extra_info["chain_iid_token_lvl"]
        tok_idx = network_input["f"]["atom_to_token_map"].cpu().numpy()

        # For all chains (pn_units) to score...
        pn_units_to_score = eval(extra_info["pn_units_to_score"]) if isinstance(extra_info["pn_units_to_score"], str) else extra_info["pn_units_to_score"]
        for chain, chain_type in pn_units_to_score:
            # ... get tokens in chain_i and chain_j
            chain_tokens = chain_iid_token_lvl == chain

            # ... convert the token level to the atom level
            chain_atoms = chain_tokens[tok_idx]

            # ... compute the outer product of the chain with itself (the definition of intra-lddt)
            chain_ij_atoms = torch.einsum(
                "L, K -> LK", torch.tensor(chain_atoms), torch.tensor(chain_atoms)
            ).to(network_output["X_L"].device)

            # ... compute lddt using the pairs_to_score from the interface
            lddt = calc_lddt(
                network_output["X_L"],
                extra_info["X_gt_L"],
                extra_info["crd_mask_L"],
                torch.tensor(tok_idx).to(network_output["X_L"].device),
                pairs_to_score=chain_ij_atoms,
            )

            # ... and finally add the results to the chain_results list
            n = len(lddt)
            result = {
                "pn_units": [chain],
                "type": chain_type,
                "best_of_1_lddt": lddt[0].item(),
                f"best_of_{n}_lddt": lddt.max().item(),
            }

            if "confidence" in network_output:
                result.update(
                    {
                        "oracle_by_pae": lddt[
                            network_output["confidence"]["pae_idx"]
                        ].item(),
                        "oracle_by_pde": lddt[
                            network_output["confidence"]["pde_idx"]
                        ].item(),
                        "oracle_by_plddt": lddt[
                            network_output["confidence"]["plddt_idx"]
                        ].item(),
                        "oracle_by_af3_style_chain": lddt[
                            network_output["confidence"]["best_chain_to_all_idx"][chain]
                        ].item(),
                        "oracle_by_af3_style_single_chain": lddt[
                            network_output["confidence"]["best_chain_to_self_idx"][
                                chain
                            ]
                        ].item(),
                    }
                )
            chain_results.append(result)

        return chain_results

# TODO: Refactor to use new metrics API
class ByTypeLDDT(Metric):
    """Calculates LDDT scores by type for both chains and interfaces"""

    def __init__(self):
        self.interface_lddt = ByTypeInterfaceLDDT()
        self.chain_lddt = ChainLDDTByType()

    def compute(
        self, network_input: dict, network_output: dict, extra_info: dict, **kwargs
    ) -> dict:
        # Compute interface LDDT scores
        interface_results = self.interface_lddt.compute(
            network_input, network_output, extra_info
        )

        # Compute chain LDDT scores
        chain_results = self.chain_lddt.compute(
            network_input, network_output, extra_info
        )

        # Merge the results
        combined_results = interface_results + chain_results

        return combined_results



# TODO: Rewrite with new Metrics API
class LDDTByDiffusionStep(Metric):
    def compute(self, network_input, network_output, loss_input):
        lddt_by_step = {"lddt_by_step": []}
        tok_idx = network_input["f"]["atom_to_token_map"].cpu().numpy()
        for i, X_L in enumerate(network_output["X_denoised_L_traj"]):
            lddt = calc_lddt(
                X_L,
                loss_input["X_gt_L"],
                loss_input["crd_mask_L"],
                torch.tensor(tok_idx).to(network_output["X_L"].device),
            )
            lddt_by_step["lddt_by_step"].append(lddt)
        return lddt_by_step
