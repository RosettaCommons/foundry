import torch
import torch.nn as nn
import tree
from typing import Any

from rf2aa.metrics.metrics_base import Metric
from rf2aa.metrics.metric_utils import unbin_logits, create_interface_masks_2d, create_chainwise_masks_2d, create_chainwise_masks_1d, unbin_rf3_metrics
from rf2aa.chemical import ChemicalData as ChemData

import numpy as np
import pandas as pd
from itertools import combinations


def compute_mean_over_subsampled_pairs(matrix_to_mean, pairs_to_score):
    """
    Compute the mean over a subsample of pairs in a 2d matrix. Returns a tensor with an element for each batch
    Args:
        matrix_to_mean: tensor of shape (batch, L, L)
        pairs_to_score: 2d tensor of shape (L, L) with 1s where pairs should be scored and 0s elsewhere
    Returns:
        1d tensor of shape (batch,) with the mean over the subsampled pairs for each batch
    """
    B, L = matrix_to_mean.shape[:2]
    assert matrix_to_mean.shape == (B, L, L), "Matrix to mean should be of shape (batch, L, L)"
    assert pairs_to_score.shape == (L, L), "Pairs to score should be of shape (L, L)"   
    batch = (matrix_to_mean * pairs_to_score).sum(dim=(-1,-2)) / pairs_to_score.sum()
    assert batch.shape == (B,), "Batch should be of shape (batch,)"
    return batch


def spread_batch_into_dictionary(batch):
    """
    Given a batch of data, create a dictionary with keys as the batch index and value as the corresponding data
    """
    assert len(batch.shape) == 1, f"Batch should be a 1d tensor, {batch}" 
    return {i: data.item() for i, data in enumerate(batch)}


class WriteAF3Confidence(Metric):
    """
    Given some config setups of pae, plddt, and pde, computes aggregate metrics for the model's confidence predictions
    TO be used at inference time for users to know how confident their predictions are.
    """
    def __init__(self, pae, plddt, pde, **kwargs):
        super().__init__()
        self.pae = pae
        self.plddt = plddt
        self.pde = pde

    def __call__(self,
                network_input,
                network_output,
                loss_input 
                ) -> Any:
        plddt_logit_stack = network_output["confidence"]["plddt_logits"]
        pae_logits = network_output["confidence"]["pae_logits"]
        pde_logits = network_output["confidence"]["pde_logits"]
        ch_label = network_output["confidence"]["chain_iid_token_lvl"]
        is_real_atom = network_output["confidence"]["is_real_atom"]
        if len(is_real_atom.shape) == 2:
            is_real_atom = is_real_atom.unsqueeze(0)

        # reorder the input tensors to be in (B, n_bins, ...) format for unbinning 
        plddt = unbin_logits(plddt_logit_stack.reshape(plddt_logit_stack.shape[0], -1, plddt_logit_stack.shape[1], ChemData().NHEAVY).float(), self.plddt.max_value, self.plddt.n_bins)
        pae = unbin_logits(pae_logits.permute(0,3,1,2).float(), self.pae.max_value, self.pae.n_bins)
        pde = unbin_logits(pde_logits.permute(0,3,1,2).float(), self.pde.max_value, self.pde.n_bins)

        pae_interface = {}
        pde_interface = {}
        for interface, pairs_to_score in create_interface_masks_2d(ch_label, device=pae.device).items():
            pae_interface[interface] = spread_batch_into_dictionary(compute_mean_over_subsampled_pairs(pae, pairs_to_score))
            pde_interface[interface] = spread_batch_into_dictionary(compute_mean_over_subsampled_pairs(pde, pairs_to_score))

        pae_chainwise = {}
        pde_chainwise = {}
        for chain, pairs_to_score in create_chainwise_masks_2d(ch_label, device=pae.device).items():
            pae_chainwise[chain] = spread_batch_into_dictionary(compute_mean_over_subsampled_pairs(pae, pairs_to_score))
            pde_chainwise[chain] = spread_batch_into_dictionary(compute_mean_over_subsampled_pairs(pde, pairs_to_score))
        
        plddt_chainwise = {}
        for chain, mask in create_chainwise_masks_1d(ch_label).items():
            plddt_real_atoms = plddt * is_real_atom[..., :ChemData().NHEAVY]
            plddt_real_atoms = plddt_real_atoms[:, mask, :].sum(dim=(1,2)) / is_real_atom[:, mask, :ChemData().NHEAVY].sum(dim=(1,2))
            plddt_chainwise[chain] = spread_batch_into_dictionary(plddt_real_atoms)

        confidence_data = {
            "example_id": loss_input["example_id"],
            "mean_plddt": spread_batch_into_dictionary(plddt.mean(dim=(-1, -2))),
            "mean_pae": spread_batch_into_dictionary(pae.mean(dim=(-1,-2))),
            "mean_pde": spread_batch_into_dictionary(pde.mean(dim=(-1,-2))),
            "chain_wise_mean_plddt": plddt_chainwise,
            "chain_wise_mean_pae": pae_chainwise,
            "chain_wise_mean_pde": pde_chainwise, 
            "interface_wise_mean_pae": pae_interface,
            "interface_wise_mean_pde": pde_interface,
        }
    
        num_batches = plddt.shape[0]
        chains = np.unique(ch_label)
        num_chains = len(chains)
        chain_pairs = list(combinations(chains, 2))

        #TODO: refactor to remove for loops
        rows = []
        for batch_idx in range(num_batches):
            for chain_id in range(num_chains):
                chain = chains[chain_id]
                row = {
                    'example_id': confidence_data['example_id'],
                    'chain_chainwise': chain,
                    'chainwise_plddt': confidence_data['chain_wise_mean_plddt'][chain][batch_idx],
                    'chainwise_pde': confidence_data['chain_wise_mean_pde'][chain][batch_idx],
                    'chainwise_pae': confidence_data['chain_wise_mean_pae'][chain][batch_idx],
                    'overall_plddt': confidence_data['mean_plddt'][batch_idx],
                    'overall_pde': confidence_data['mean_pde'][batch_idx],
                    'overall_pae': confidence_data['mean_pae'][batch_idx],
                    'batch_idx': batch_idx
                }
                rows.append(row)
            for interface in chain_pairs:
                chain_i, chain_j = interface
                row = {
                    'example_id': confidence_data['example_id'],
                    'chain_i_interface': chain_i,
                    'chain_j_interface': chain_j,
                    'pae_interface': confidence_data['interface_wise_mean_pae'][interface][batch_idx],
                    'pde_interface': confidence_data['interface_wise_mean_pde'][interface][batch_idx],
                    'overall_plddt': confidence_data['mean_plddt'][batch_idx],
                    'overall_pde': confidence_data['mean_pde'][batch_idx],
                    'overall_pae': confidence_data['mean_pae'][batch_idx],
                    'batch_idx': batch_idx
                }
                rows.append(row)
            
        return pd.DataFrame(rows)


class GetConfidenceIndices(Metric):

    def __call__(self,
                    network_input,
                    network_output,
                    loss_input
    ):

        # AF3's ranking metrics work like this, but using ptm instead of ipae:

        confidence_loss = loss_input['confidence_loss']
        del loss_input['confidence_loss']

        #get and sort the chains we need to score
        ch_label = loss_input["chain_iid_token_lvl"]
        interface_masks = {}
        chain_to_all_masks = {}
        chain_to_self_masks = {}
        interfaces = []
        scored_chains = []
        chains = []
        interface_chains = []
        for k in loss_input['interfaces_to_score']:
            interfaces.append(f'{k[0]}-{k[1]}')
            chains.append(k[0])
            chains.append(k[1])
            interface_chains.append(k[0])
            interface_chains.append(k[1])
        for k in loss_input['pn_units_to_score']:
            chains.append(k[0])
            scored_chains.append(k[0])
        chains = set(chains)

        # Masks prep
        I = loss_input['crd_mask_rep_atoms_I'].shape[-1]
        chain_to_all_masks = {}
        chain_to_self_masks = {}
        interface_masks = {}
        #AF3 handles lig chains specially
        lig_chains = []
        
        #AF3 does a number of unique masks, not simply chain-to-chain interface masks and chainwise masks
        for chain in chains:
            mask = ch_label == chain
            mask = torch.from_numpy(mask)
            chain_to_all_mask = mask.unsqueeze(0) | mask.unsqueeze(1)
            chain_to_self_mask = mask.unsqueeze(0) & mask.unsqueeze(1)
            interface_mask = chain_to_all_mask & ~chain_to_self_mask
            
            #the diagonal of the chain_to_all_mask is always false
            chain_to_all_mask = chain_to_all_mask & ~torch.eye(I, device=chain_to_all_mask.device, dtype=torch.bool)
            chain_to_all_masks[chain] = chain_to_all_mask

            #the diagonal of the chain_to_self_mask is always false
            chain_to_self_mask = chain_to_self_mask & ~torch.eye(I, device=chain_to_self_mask.device, dtype=torch.bool)
            chain_to_self_masks[chain] = chain_to_self_mask

            interface_masks[chain] = interface_mask
            
            if torch.all(network_input['f']['is_ligand'][mask]):
                lig_chains.append(chain)
            
            # Same-chain mask
            same_chain = (ch_label[:, None] == ch_label[None, :])  # Adds dimensions and compares
            same_chain = np.expand_dims(same_chain, axis=0)  # Add the final dimension

            #map everything to gpu
            gpu = network_output['confidence']['plddt_logits'].device
            chain_to_all_masks = tree.map_structure(lambda x: x.to(gpu) if hasattr(x, 'cpu') else x, chain_to_all_masks)
            chain_to_self_masks = tree.map_structure(lambda x: x.to(gpu) if hasattr(x, 'cpu') else x, chain_to_self_masks)
            interface_masks = tree.map_structure(lambda x: x.to(gpu) if hasattr(x, 'cpu') else x, interface_masks)
            same_chain = torch.from_numpy(same_chain).to(gpu)

        interface_err = {}
        for interface in interfaces:
            interface_err[interface] = []
        lig_err = {}
        for lig_chain in lig_chains:
            lig_err[lig_chain] = []
        chain_to_all_err = {}
        chain_to_self_err = {}
        for chain in scored_chains:
            chain_to_all_err[chain] = []
            chain_to_self_err[chain] = []

        confidence = network_output['confidence']

        plddt_logits = confidence['plddt_logits']

        #Reshape logits to B, K, L, NHEAVY
        plddt_logits = plddt_logits.reshape(plddt_logits.shape[0], -1, plddt_logits.shape[1], ChemData().NHEAVY).float()

        #Reshape the pae and pde logits to B, K, L, L
        pae_logits = confidence['pae_logits'].permute(0,3,1,2).float()
        pde_logits = confidence['pde_logits'].permute(0,3,1,2).float()

        pae_logits_unbinned = unbin_logits(pae_logits, confidence_loss.pae.max_value, 
                                            confidence_loss.pae.n_bins)

        #Complex metrics
        plddt, pae, pde = unbin_rf3_metrics(plddt_logits, pae_logits, pde_logits, loss_input['is_real_atom'], 
                                            confidence_loss.plddt, confidence_loss.pae, 
                                            confidence_loss.pde)
        
        loss_input['pae_idx'] = torch.argmin(pae)
        loss_input['pde_idx'] = torch.argmin(pde)
        loss_input['plddt_idx'] = torch.argmax(plddt)

        #AF3-style confidence ranking metrics
        for interface in interfaces:
            chain_a = interface.split('-')[0]
            chain_b = interface.split('-')[1]

            #af3 only considers the ligand chain metric when evaluating interfaces containing a ligand
            if (chain_a in lig_chains or chain_b in lig_chains) and not (chain_a in lig_chains and chain_b in lig_chains):
                if chain_a in lig_chains:
                    lig_chain = chain_a
                elif chain_b in lig_chains:
                    lig_chain = chain_b

                #if a ligand participates in more than 1 interface, we still only want to get calculate one score per batch
                if len(lig_err[lig_chain]) < 1:
                    lig_i_pae = compute_mean_over_subsampled_pairs(pae_logits_unbinned, interface_masks[lig_chain])

                    lig_err[lig_chain] = lig_i_pae
                
            #AF3 interface metrics take the average of each interface chain's interaction with all other chains
            chain_a_pae = compute_mean_over_subsampled_pairs(pae_logits_unbinned, interface_masks[chain_a])
            chain_b_pae = compute_mean_over_subsampled_pairs(pae_logits_unbinned, interface_masks[chain_b])
            
            interface_err[interface] = (chain_a_pae + chain_b_pae) / 2

        for chain in scored_chains:
            chain_to_all_pae = compute_mean_over_subsampled_pairs(pae_logits_unbinned, chain_to_all_masks[chain])
            chain_to_self_pae = compute_mean_over_subsampled_pairs(pae_logits_unbinned, chain_to_self_masks[chain])
            
            chain_to_all_err[chain] = chain_to_all_pae
            chain_to_self_err[chain] = chain_to_self_pae

        #get the interface indices
        best_interface_idx = {}
        best_lig_ipae_idx = {}
        for k, v in interface_err.items():
            best_interface_idx[k] = torch.argmin(v)

            #handle special af3-style lig case, where they only use metrics for the ligand chain at evaluation
            #if there's no ligand, still assign a random value for easier parsing of metrics
            best_lig_ipae_idx[k] = -1
            chain_1 = k.split('-')[0]
            chain_2 = k.split('-')[1]
            if chain_1 in lig_chains or chain_2 in lig_chains:
                if chain_1 in lig_chains:
                    lig_chain = chain_1
                elif chain_2 in lig_chains:
                    lig_chain = chain_2

                best_lig_ipae_idx[k] = torch.argmin(lig_err[lig_chain])

        loss_input['best_interface_idx'] = best_interface_idx
        loss_input['best_lig_ipae_idx'] = best_lig_ipae_idx

        #get the single chain indices
        best_chain_to_all_idx = {}
        best_chain_to_self_idx = {}
        for chain in scored_chains:
            best_chain_to_all_idx[chain] = torch.argmin(chain_to_all_err[chain])
            best_chain_to_self_idx[chain] = torch.argmin(chain_to_self_err[chain])

        loss_input['best_chain_to_all_idx'] = best_chain_to_all_idx
        loss_input['best_chain_to_self_idx'] = best_chain_to_self_idx

        return loss_input