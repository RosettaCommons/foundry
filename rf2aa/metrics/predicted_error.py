import torch
import torch.nn as nn
import tree
from typing import Any

from rf2aa.metrics.metrics_base import Metric
from rf2aa.metrics.metric_utils import \
    unbin_logits, \
    create_interface_masks_2d, \
    create_chainwise_masks_2d, \
    create_chainwise_masks_1d, \
    spread_batch_into_dictionary, \
    compute_mean_over_subsampled_pairs
from rf2aa.chemical import ChemicalData as ChemData

import numpy as np
import pandas as pd
from itertools import combinations


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
        for chain, residue_atom_indices_to_score in create_chainwise_masks_1d(ch_label).items():
            chain_is_real_atom = is_real_atom[..., :ChemData().NHEAVY] * residue_atom_indices_to_score
            plddt_chainwise[chain] = spread_batch_into_dictionary(compute_mean_over_subsampled_pairs(plddt, chain_is_real_atom))

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

        ch_label = loss_input['chain_iid_token_lvl']
        scored_chains, interfaces, interface_chains = select_scored_units(loss_input)

        chain_to_all_masks = create_chain_to_all_masks(ch_label, scored_chains)
        chain_to_self_masks = create_chain_to_self_masks(ch_label, scored_chains)
        interface_masks, lig_chains = create_interface_masks(ch_label, interfaces, loss_input['is_ligand'])

        #map everything to gpu
        gpu = network_output['confidence']['plddt_logits'].device
        chain_to_all_masks = tree.map_structure(lambda x: x.to(gpu) if hasattr(x, 'cpu') else x, chain_to_all_masks)
        chain_to_self_masks = tree.map_structure(lambda x: x.to(gpu) if hasattr(x, 'cpu') else x, chain_to_self_masks)
        interface_masks = tree.map_structure(lambda x: x.to(gpu) if hasattr(x, 'cpu') else x, interface_masks)

        confidence = network_output['confidence']

        plddt_logits = confidence['plddt_logits']

        #Reshape logits to B, K, L, NHEAVY
        is_real_atom = network_output['confidence']['is_real_atom']
        plddt_logits = plddt_logits.reshape(plddt_logits.shape[0], -1, plddt_logits.shape[1], ChemData().NHEAVY).float()
        #Reshape the pae and pde logits to B, K, L, L
        pae_logits = confidence['pae_logits'].permute(0,3,1,2).float()
        pde_logits = confidence['pde_logits'].permute(0,3,1,2).float()

        pae_logits_unbinned = unbin_logits(pae_logits, confidence_loss.pae.max_value, 
                                            confidence_loss.pae.n_bins)
        plddt_logits_unbinned = unbin_logits(plddt_logits, confidence_loss.plddt.max_value,
                                            confidence_loss.plddt.n_bins)
        pde_logits_unbinned = unbin_logits(pde_logits, confidence_loss.pde.max_value,
                                            confidence_loss.pde.n_bins)
        
        complex_pae = pae_logits_unbinned.mean(dim=(1,2))
        complex_pde = pde_logits_unbinned.mean(dim=(1,2))
        complex_plddt = (plddt_logits_unbinned * is_real_atom[..., :ChemData().NHEAVY]).sum(dim=(1,2)) / is_real_atom[..., :ChemData().NHEAVY].sum()

        loss_input['pae_idx'] = torch.argmin(complex_pae)
        loss_input['pde_idx'] = torch.argmin(complex_pde)
        loss_input['plddt_idx'] = torch.argmax(complex_plddt)

        chain_to_self_paes = get_masked_error_per_chain(scored_chains, chain_to_self_masks, pae_logits_unbinned)
        chain_to_all_paes = get_masked_error_per_chain(scored_chains, chain_to_all_masks, pae_logits_unbinned)
        interface_chain_paes = get_masked_error_per_chain(interface_chains, interface_masks, pae_logits_unbinned)
        #average over both interfaces
        average_interface_paes = get_average_error_per_interface(interfaces, lig_chains, interface_chain_paes)

        loss_input['best_chain_to_all_idx'] = get_lowest_error_indices(chain_to_all_paes)
        loss_input['best_chain_to_self_idx'] = get_lowest_error_indices(chain_to_self_paes)
        loss_input['best_interface_idx'] = get_lowest_error_indices(average_interface_paes)
        #for ligands, we don't average the error
        loss_input['best_lig_ipae_idx'] = get_lowest_error_ligand_indices(interface_chain_paes, interfaces, lig_chains)

        return loss_input

def select_scored_units(loss_input):
    scored_chains = []
    interfaces = []
    interface_chains = []
    for k in loss_input['interfaces_to_score']:
        interfaces.append(f'{k[0]}-{k[1]}')
        interface_chains.append(k[0])
        interface_chains.append(k[1])
    for k in loss_input['pn_units_to_score']:
        scored_chains.append(k[0])

    return scored_chains, interfaces, interface_chains

def create_chain_to_all_masks(ch_label, chains_to_score):
    unique_chains = np.unique(ch_label)
    I = len(ch_label)
    chain_to_all_masks = {}
    for chain in unique_chains:
        if chain in chains_to_score:
            indices = torch.from_numpy((ch_label == chain))
            mask = indices.unsqueeze(0) | indices.unsqueeze(1)
            #set the diagonal to false
            mask = mask & ~torch.eye(I, device=mask.device, dtype=torch.bool)
            chain_to_all_masks[chain] = mask
    return chain_to_all_masks

def create_chain_to_self_masks(ch_label, chains_to_score):
    unique_chains = np.unique(ch_label)
    I = len(ch_label)
    chain_to_self_masks = {}
    for chain in unique_chains:
        if chain in chains_to_score:
            indices = torch.from_numpy((ch_label == chain))
            mask = indices.unsqueeze(0) & indices.unsqueeze(1)
            #set the diagonal to false
            mask = mask & ~torch.eye(I, device=mask.device, dtype=torch.bool)
            chain_to_self_masks[chain] = mask
    return chain_to_self_masks

def create_interface_masks(ch_label, interfaces, is_ligand):
    interface_masks = {}
    interface_chains = []
    ligand_chains = []
    for interface in interfaces:
        interface_chains.append(interface.split('-')[0])
        interface_chains.append(interface.split('-')[1])
    interface_chains = set(interface_chains)
    for chain in interface_chains:
        chain_indices = torch.from_numpy((ch_label == chain))

        to_self = chain_indices.unsqueeze(0) & chain_indices.unsqueeze(1)
        to_all = chain_indices.unsqueeze(0) | chain_indices.unsqueeze(1)
        interface_mask = to_all & ~to_self
        interface_masks[chain] = interface_mask

        if torch.all(is_ligand[chain_indices]):
            ligand_chains.append(chain)

    return interface_masks, ligand_chains

def get_masked_error_per_chain(chains, masks, unbinned_logits):
    error = {}
    for chain in chains:
        mask = masks[chain]
        chain_error = compute_mean_over_subsampled_pairs(unbinned_logits, mask)
        error[chain] = chain_error

    return error

def get_average_error_per_interface(interfaces, lig_chains, interface_errors):
    average_error = {}
    for interface in interfaces:
        chain_a = interface.split('-')[0]
        chain_b = interface.split('-')[1]
        average_error[interface] = (interface_errors[chain_a] + interface_errors[chain_b]) / 2

    return average_error

def get_lowest_error_indices(errors):
    lowest_error_indices = {}
    for k, v in errors.items():
        lowest_error_indices[k] = torch.argmin(v)

    return lowest_error_indices

def get_lowest_error_ligand_indices(errors, interfaces, lig_chains):
    #ligands are a special case in AF3, where they only consider the ligand chain's error and not the average for the interface
    lowest_error_indices = {}
    for interface in interfaces:
        chain_a = interface.split('-')[0]
        chain_b = interface.split('-')[1]
        if chain_a in lig_chains or chain_b in lig_chains:
            if chain_a in lig_chains:
                lig_chain = chain_a
            elif chain_b in lig_chains:
                lig_chain = chain_b

            lowest_error_indices[interface] = torch.argmin(errors[lig_chain])
        else:
            #assign a random value to avoid key errors downstream; sorting ligand interfaces
            #from other types is handles in analysis
            lowest_error_indices[interface] = 0

    return lowest_error_indices