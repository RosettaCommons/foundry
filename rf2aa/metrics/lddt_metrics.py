import torch
import torch.nn as nn
import numpy as np

from rf2aa.metrics.metrics_base import Metric


def calc_lddt(X_L, X_gt_L, crd_mask_L, tok_idx, pairs_to_score=None):
    """
    X_L: predicted coordinates (D, L, 3)
    X_gt_L: ground truth coordinates (D, L, 3)
    crd_mask_L: mask of coordinates (D, L,)
    tok_idx: token index of each atom (L,) 
    pairs_to_score: pairs to score (L, L) | None
    """
    D, L = X_L.shape[:2]
    
    if pairs_to_score is None:
        pairs_to_score = torch.ones((L, L), dtype=torch.bool)
    else:
        assert pairs_to_score.shape == (L, L)
    
    # Compute distance matrix
    predicted_distances = torch.cdist(X_L, X_L)
    ground_truth_distances = torch.cdist(X_gt_L, X_gt_L)
    ground_truth_distances[ground_truth_distances.isnan()] = 9999.0
    difference_distances = torch.abs(ground_truth_distances - predicted_distances)

    lddt_matrix = torch.zeros_like(difference_distances)
    lddt_matrix = 0.25 * (difference_distances < 4.0) + \
                  0.25 * (difference_distances < 2.0) + \
                  0.25 * (difference_distances < 1.0) + \
                  0.25 * (difference_distances < 0.5)

    is_close_distance_LL = (ground_truth_distances < 15.0)
    in_same_residue_LL = tok_idx[None, :] == tok_idx[:, None]
    to_score_LL = pairs_to_score[None] & is_close_distance_LL & ~in_same_residue_LL

    lddt = (lddt_matrix * to_score_LL[None]).sum(dim=(-1,-2)) / (to_score_LL.sum(dim=(-1,-2)) + 1e-6)
    return lddt



class InterfaceLDDT(Metric):

    def __call__(self, 
                network_input, 
                network_output, 
                loss_input
        ):
        interface_lddt = {
            "interface_lddt": []
        }
        chain_iid_token_lvl = loss_input["chain_iid_token_lvl"]
        tok_idx = network_input["f"]["atom_to_token_map"].cpu().numpy()
        for chain_i, chain_j, interface_type in loss_input["interfaces_to_score"]:
            
            # get tokens in chain_i and chain_j
            chain_i_tokens = chain_iid_token_lvl == chain_i
            chain_j_tokens = chain_iid_token_lvl == chain_j
            # convert the token level to the atom level
            chain_i_atoms = chain_i_tokens[tok_idx]
            chain_j_atoms = chain_j_tokens[tok_idx]
            # compute the intersection of chain_i and chain_j

            chain_ij_atoms = torch.einsum(
                                "L, K -> LK", 
                                torch.tensor(chain_i_atoms), 
                                torch.tensor(chain_j_atoms)
                                ).to(network_output["X_L"].device)

            #compute lddt using the pairs_to_score from the intersection
            lddt = calc_lddt(
                network_output["X_L"],
                loss_input["X_gt_L"],
                loss_input["crd_mask_L"],
                torch.tensor(tok_idx).to(network_output["X_L"].device),
                pairs_to_score=chain_ij_atoms
            )
            interface_lddt["interface_lddt"].append(lddt)
        return interface_lddt

