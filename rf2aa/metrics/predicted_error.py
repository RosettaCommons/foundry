import torch
import torch.nn as nn
from typing import Any

from rf2aa.metrics.metrics_base import Metric
from rf2aa.metrics.metric_utils import unbin_logits, create_interface_masks_2d, create_chainwise_masks_2d, create_chainwise_masks_1d
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
        for interface, pairs_to_score in create_interface_masks_2d(ch_label).items():
            pae_interface[interface] = spread_batch_into_dictionary(compute_mean_over_subsampled_pairs(pae, pairs_to_score))
            pde_interface[interface] = spread_batch_into_dictionary(compute_mean_over_subsampled_pairs(pde, pairs_to_score))

        pae_chainwise = {}
        pde_chainwise = {}
        for chain, pairs_to_score in create_chainwise_masks_2d(ch_label).items():
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

