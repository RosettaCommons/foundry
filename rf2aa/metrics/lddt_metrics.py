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

    if use_amp:
        X_L = X_L.to(torch.bfloat16)
        X_gt_L = X_gt_L.to(torch.bfloat16)

    lddt = []
    for d in range(D):
        ground_truth_distances = torch.linalg.norm(X_gt_L[d,first_index]-X_gt_L[d,second_index], dim=-1)
  
        with torch.amp.autocast('cuda',enabled=use_amp, dtype=torch.bfloat16):
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
            #print(interface_type,chain_i, chain_j) 
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
            "interface_lddt_ipae": [],
            "interface_lddt_pae": [],
            "interface_lddt_pde": [],
            "interface_lddt_plddt": [],
        }
        chain_iid_token_lvl = loss_input["chain_iid_token_lvl"]
        tok_idx = network_input["f"]["atom_to_token_map"].cpu().numpy()
        for chain_i, chain_j, interface_type in loss_input["interfaces_to_score"]:
            #print(interface_type,chain_i, chain_j) 
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
            ipae_idx = loss_input["ipae_idx"]
            pae_idx = loss_input["pae_idx"]
            pde_idx = loss_input["pde_idx"]
            plddt_idx = loss_input["plddt_idx"]
            interface_lddt["interface_lddt_first"].append(lddt[0].item())
            interface_lddt["interface_lddt_best"].append(lddt.max().item())
            interface_lddt["interface_lddt_ipae"].append(lddt[ipae_idx].item())
            interface_lddt["interface_lddt_pae"].append(lddt[pae_idx].item())
            interface_lddt["interface_lddt_pde"].append(lddt[pde_idx].item())
            interface_lddt["interface_lddt_plddt"].append(lddt[plddt_idx].item())
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
            "chain_lddt_ipae": [],
            "chain_lddt_pae": [],
            "chain_lddt_pde": [],
            "chain_lddt_plddt": [],
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
            chain_lddt["chain_lddt_ipae"].append(lddt[loss_input["ipae_idx"]].item())
            chain_lddt["chain_lddt_pae"].append(lddt[loss_input["pae_idx"]].item())
            chain_lddt["chain_lddt_pde"].append(lddt[loss_input["pde_idx"]].item())
            chain_lddt["chain_lddt_plddt"].append(lddt[loss_input["plddt_idx"]].item())
        print('confidence_chain_lddt', chain_lddt)
        return chain_lddt

class LigRMSD(Metric):

    def __call__(self, 
                 network_input, 
                 network_output, 
                 loss_input
        ):
        lig_rmsd = {
            "first_lig_rmsd": [],
            "best_lig_rmsd": [],
            "ipae_lig_rmsd": [],
            "pae_lig_rmsd": [],
            "pde_lig_rmsd": [],
            "plddt_lig_rmsd": []
        }

        if not torch.any(network_input['f']['is_ligand']):
            return lig_rmsd

        #identify the ligand atoms
        tok_idx = network_input["f"]["atom_to_token_map"]
        ligand_mask = network_input['f']['is_ligand'][tok_idx]

        #decide which atoms we should use for alignment
        alignment_mask = loss_input["alignment_mask"]

        #perform an align weighted on non-ligand atoms
        X_L = network_output["X_L"]
        X_gt_L = loss_input["X_gt_L"]
        X_exists_L = loss_input["crd_mask_L"]

        #find all atoms within 10A of the ligand in the ground truth
        # Step 1: Extract ligand coordinates
        ligand_coords = X_gt_L[:, ligand_mask, :]  # Shape: [b, num_ligand_atoms, 3]

        # Step 2: Compute pairwise distances
        # Expand and broadcast to calculate distances
        all_coords_expanded = X_gt_L[:, :, None, :]  # Shape: [b, l, 1, 3]
        ligand_coords_expanded = ligand_coords[:, None, :, :]  # Shape: [b, 1, num_ligand_atoms, 3]

        distances = torch.norm(all_coords_expanded - ligand_coords_expanded, dim=-1)  # Shape: [b, l, num_ligand_atoms]

        # Step 3: Find minimum distance for each atom and threshold at 10 Å
        close_atoms = (distances <= 10.0).any(dim=-1)  # Shape: [b, l]

        #align only on non-ligand atoms that are close to the ligand (and Ca for protein)
        w_L = ~ligand_mask
        w_L = w_L * alignment_mask
        w_L = w_L * close_atoms
        #convert to float
        w_L = w_L.to(torch.float32)

        w_L = w_L.to(X_L.device)
        w_L = w_L.expand(X_L.shape[0], -1)
        # X_L_aligned = weighted_rigid_align(X_L, X_gt_L, X_exists_L[0], w_L)
        # print('X_L_aligned', X_L_aligned.shape, X_L_aligned)
        rmsd = []
        for i in range(X_L.shape[0]):

            X_L_aligned = weighted_rigid_align(X_L[i].unsqueeze(0), X_gt_L[i].unsqueeze(0), X_exists_L[i], w_L[i].unsqueeze(0))

            #now for all ligand atoms, compute the RMSD
            ligand_mask = network_input['f']['is_ligand'][tok_idx]
            ligand_mask = ligand_mask * X_exists_L[i]
            ligand_mask = ligand_mask.unsqueeze(0).unsqueeze(-1)
            ligand_mask = ligand_mask.expand(-1, -1, 3)
            ligand_mask = ligand_mask.to(X_L.device)
            diff = (X_L_aligned - X_L[i].unsqueeze(0))**2 * ligand_mask
            #convert nans to 0
            diff[torch.isnan(diff)] = 0


            ligand_rmsd = torch.sqrt(
                torch.sum(
                    diff,
                    dim=(-1, -2)
                ) / (torch.sum(ligand_mask, dim=(-1, -2)) + 1e-8)
            )
            rmsd.append(ligand_rmsd)

        ipae_idx = loss_input["ipae_idx"]
        pae_idx = loss_input["pae_idx"]
        pde_idx = loss_input["pde_idx"]
        plddt_idx = loss_input["plddt_idx"]

        lig_rmsd["first_lig_rmsd"].append(rmsd[0])
        lig_rmsd["best_lig_rmsd"].append(min(rmsd))
        lig_rmsd["ipae_lig_rmsd"].append(rmsd[ipae_idx])
        lig_rmsd["pae_lig_rmsd"].append(rmsd[pae_idx])
        lig_rmsd["pde_lig_rmsd"].append(rmsd[pde_idx])
        lig_rmsd["plddt_lig_rmsd"].append(rmsd[plddt_idx])
        print('lig_rmsd', lig_rmsd)
        return lig_rmsd
    
# class ConfidenceLossMetric(Metric):
#     def __call__(self,network_input,network_output,loss_input):
#         loss = {
#             'pae_loss':[],
#             'pde_loss':[],
#             'plddt_loss':[],
#             'exp_resolved_loss':[]
#         }

#         loss

#         return loss



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
        pass