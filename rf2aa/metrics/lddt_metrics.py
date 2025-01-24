import torch
import torch.nn as nn
import numpy as np
import tree

from rf2aa.metrics.metrics_base import Metric
from rf2aa.alignment import weighted_rigid_align


def calc_lddt(X_L, X_gt_L, crd_mask_L, tok_idx, pairs_to_score=None, distance_cutoff=15.0, use_amp=True, eps=1e-6):
    """
    X_L: predicted coordinates (D, L, 3)
    X_gt_L: ground truth coordinates (D, L, 3)
    crd_mask_L: mask of coordinates (D, L,)
    tok_idx: token index of each atom (L,) 
    pairs_to_score: pairs to score (L, L) | None
    """
    D, L = X_L.shape[:2]
    if pairs_to_score is None:
        pairs_to_score = torch.ones((L, L), dtype=torch.bool).triu(0).to(X_L.device)
    else:
        assert pairs_to_score.shape == (L, L)
        pairs_to_score = pairs_to_score.triu(0).to(X_L.device)

    first_index,second_index = torch.nonzero(pairs_to_score,as_tuple=True)

    lddt = []
    for d in range(D):
        ground_truth_distances = torch.linalg.norm(X_gt_L[d,first_index]-X_gt_L[d,second_index], dim=-1)
  
        pair_mask = torch.logical_and(
            ground_truth_distances>0,
            ground_truth_distances<distance_cutoff
        )

        # only score pairs that are resolved in the ground truth
        pair_mask *= (crd_mask_L[d,first_index] * crd_mask_L[d,second_index])
        # don't score pairs that are in the same token
        pair_mask *= (tok_idx[first_index] != tok_idx[second_index])

        valid_pairs = pair_mask.nonzero(as_tuple=True)
        pair_mask = pair_mask[valid_pairs].to(X_L.dtype)
        ground_truth_distances = ground_truth_distances[valid_pairs]    
        first_index,second_index = first_index[valid_pairs],second_index[valid_pairs]

        predicted_distances = torch.linalg.norm(X_L[d,first_index]-X_L[d,second_index], dim=-1)
    
        delta_distances = torch.abs(predicted_distances-ground_truth_distances+eps)
        del predicted_distances, ground_truth_distances

        lddt.append( 0.25*(
                torch.sum( (delta_distances < 4.0)*pair_mask )
                +torch.sum( (delta_distances < 2.0)*pair_mask )
                +torch.sum( (delta_distances < 1.0)*pair_mask )
                +torch.sum( (delta_distances < 0.5)*pair_mask )
            ) / (torch.sum( pair_mask ) + eps)
        )

    return torch.tensor(lddt)



class InterfaceLDDT(Metric):

    def __call__(self, 
                network_input, 
                network_output, 
                loss_input
        ):
        interface_lddt = {
            "interface_lddt_first": [],
            "interface_lddt_best": []
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

            # symmetrize
            chain_ij_atoms = chain_ij_atoms | chain_ij_atoms.T

            #compute lddt using the pairs_to_score from the intersection
            lddt = calc_lddt(
                network_output["X_L"],
                loss_input["X_gt_L"],
                loss_input["crd_mask_L"],
                torch.tensor(tok_idx).to(network_output["X_L"].device),
                pairs_to_score=chain_ij_atoms,
                distance_cutoff=30.0
            )

            interface_lddt["interface_lddt_first"].append(lddt[0].item())
            interface_lddt["interface_lddt_best"].append(lddt.max().item())
        return interface_lddt
    

class ConfidenceInterfaceLDDT(Metric):

    def __call__(self, 
                network_input, 
                network_output, 
                loss_input
        ):
        interface_lddt = {
            "interface_lddt_first": [],
            "interface_lddt_best": [],
            "interface_lddt_pae": [],
            "interface_lddt_pde": [],
            "interface_lddt_plddt": [],
            "interface_lddt_af3_style_ipae": [],
            "interface_lddt_af3_style_lig_ipae": [],
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
                pairs_to_score=chain_ij_atoms,
                distance_cutoff=30.0
            )
            pae_idx = loss_input["pae_idx"]
            pde_idx = loss_input["pde_idx"]
            plddt_idx = loss_input["plddt_idx"]
            af3_style_ipae_idx = loss_input["best_interface_idx"][f'{chain_i}-{chain_j}']
            interface_lddt["interface_lddt_first"].append(lddt[0].item())
            interface_lddt["interface_lddt_best"].append(lddt.max().item())
            interface_lddt["interface_lddt_pae"].append(lddt[pae_idx].item())
            interface_lddt["interface_lddt_pde"].append(lddt[pde_idx].item())
            interface_lddt["interface_lddt_plddt"].append(lddt[plddt_idx].item())
            interface_lddt["interface_lddt_af3_style_ipae"].append(lddt[af3_style_ipae_idx].item())
            interface_lddt["interface_lddt_af3_style_lig_ipae"].append(lddt[loss_input["best_lig_ipae_idx"][f'{chain_i}-{chain_j}']].item())
        return interface_lddt

class ConfidenceChainLDDT(Metric):

    def __call__(self, 
                network_input, 
                network_output, 
                loss_input
        ):
        chain_lddt = {
            "chain_lddt_first": [],
            "chain_lddt_best": [],
            "chain_lddt_pae": [],
            "chain_lddt_pde": [],
            "chain_lddt_plddt": [],
            "chain_lddt_af3_style_chain": [],
            "chain_lddt_af3_style_single_chain": []
        }
        chain_iid_token_lvl = loss_input["chain_iid_token_lvl"]
        tok_idx = network_input["f"]["atom_to_token_map"].cpu().numpy()
        for chain_i, chain_type in loss_input["pn_units_to_score"]:
            #print(chain_type)
            # get tokens in chain_i and chain_j
            chain_i_tokens = chain_iid_token_lvl == chain_i
            chain_j_tokens = chain_iid_token_lvl == chain_i
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

            
            chain_lddt["chain_lddt_first"].append(lddt[0].item())
            chain_lddt["chain_lddt_best"].append(lddt.max().item())
            chain_lddt["chain_lddt_pae"].append(lddt[loss_input["pae_idx"]].item())
            chain_lddt["chain_lddt_pde"].append(lddt[loss_input["pde_idx"]].item())
            chain_lddt["chain_lddt_plddt"].append(lddt[loss_input["plddt_idx"]].item())
            chain_lddt["chain_lddt_af3_style_chain"].append(lddt[loss_input["best_chain_to_all_idx"][chain_i]].item())
            chain_lddt["chain_lddt_af3_style_single_chain"].append(lddt[loss_input["best_chain_to_self_idx"][chain_i]].item())
        return chain_lddt

class LigRMSD(Metric):
    #TODO: move these to a separate file, here for backwards compatibility with configs
    def __call__(self, 
                 network_input, 
                 network_output, 
                 loss_input
        ):
        raise NotImplementedError()



class InterfacePocketLigandRMSD(Metric):
    #TODO: move these to a separate file, here for backwards compatibility with configs

    """
    Compute the Ligand RMSD for each interface in the interfaces_to_score list.
    
    The ligand RMSD is computed only for interface protein-ligand chains.
    Given a chain pair (chain_i, chain_j) and the interface type, the RMSD is computed as follows:
    - if the interface_type is protein_ligand: continue
    - Rigid align the GT coordinates of onto the predicted coordinates using only the CA atoms within 10A of the ligand in chain_i or chain_j
    - Compute the RMSD between the aligned GT coordinates and the predicted coordinates of the ligand atoms

    Note: if the interface is not between a protein-ligand pair, the RMSD is set to -1
    """
    def __call__(self, 
                 network_input, 
                 network_output, 
                 loss_input
        ):
        raise NotImplementedError()


class ChainLDDT(Metric):

    def __call__(self, 
                network_input, 
                network_output, 
                loss_input
        ):
        chain_lddt = {
            "chain_lddt_first": [],
            "chain_lddt_best": []
        }
        chain_iid_token_lvl = loss_input["chain_iid_token_lvl"]
        tok_idx = network_input["f"]["atom_to_token_map"].cpu().numpy()
        for chain_i, chain_type in loss_input["pn_units_to_score"]:
            # get tokens in chain_i and chain_j
            chain_i_tokens = chain_iid_token_lvl == chain_i
            chain_j_tokens = chain_iid_token_lvl == chain_i
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

            chain_lddt["chain_lddt_first"].append(lddt[0].item())
            chain_lddt["chain_lddt_best"].append(lddt.max().item())
        return chain_lddt
    
class LDDTByDiffusionStep(Metric):

    def __call__(self,
                    network_input,
                    network_output,
                    loss_input
    ):
        lddt_by_step = {
            "lddt_by_step": []
        }
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

class SmoothedLDDT(nn.Module):

    def __call__(
        self,
        network_input,
        network_output,
        loss_input
    ):
        raise NotImplementedError()