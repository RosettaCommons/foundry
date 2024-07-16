import os
import csv
import shutil
import logging
from collections import defaultdict
import json

import numpy as np
from scipy.stats import norm
from icecream import ic
import torch
import torch.nn.functional as F
import tree
import pandas as pd
import numpy as np
from lightning.pytorch.callbacks import Callback
from lightning import Trainer, LightningModule
from lightning_fabric.loggers.csv_logs import _ExperimentWriter

from rf2aa.tensor_util import apply_to_tensors
from rf2aa.chemical import ChemicalData as ChemData
from rf2aa.debug import pretty_describe_dict
from rf2aa.model.AF3_structure import Loss
from rf2aa import pymol
from rf2aa.pymol import cmd
from rf2aa import pymol_tools

logger = logging.getLogger(__name__)

def flatten_dictionary(dictionary, parent_key='', separator='.'):
    flattened_dict = {}
    for key, value in dictionary.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key
        if isinstance(value, dict):
            flattened_dict.update(flatten_dictionary(value, new_key, separator))
        else:
            flattened_dict[new_key] = value
    return flattened_dict

class LogMetrics(Callback):
    
    def __init__(self, config):
        super().__init__()
        self.config = config

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx: int) -> None:

        logger.debug('on_train_batch_end outputs:\n' + pretty_describe_dict(outputs))
        
        outputs = tree.map_structure(lambda x: x.detach().cpu(), outputs)
        o = {}
        stratifications = defaultdict(list)
        for metric in [diffusion_losses, lddt_metrics]:
            metric_d, stratification_keys = metric(self.config, outputs)
            stratifications[stratification_keys].extend(metric_d.keys())
            o.update(metric_d)

        o['t'] = outputs['t']
        o['t_quantile_4'] = get_t_quantiles(outputs['t'], self.config.loss.sigma_data, 4)
        df = pd.DataFrame.from_dict(o)
        df = df.reindex(sorted(df.columns), axis=1)

        D, = outputs['t'].shape
        df['batch_idx'] = batch_idx
        df['data_idx'] = np.arange(D)
        df['global_step'] = trainer.global_step
        trainer.logger.log_df(df, stratifications=stratifications)
        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
      
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):   
        
        outputs = tree.map_structure(lambda x: x.detach().cpu(), outputs)
        o = {}
        for metric in [lddt_metrics]:
            metric_d, stratification_keys = metric(self.config, outputs)
            o.update(metric_d)
        df = pd.DataFrame.from_dict(o)
        df = df.reindex(sorted(df.columns), axis=1)
        ic(o)
        df['batch_idx'] = batch_idx
        df['global_step'] = trainer.global_step

        trainer.logger.log_df(df, stratifications={})
        return super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx) 

def lddt_metrics(config, outputs):
    # compute distances between ground truth atoms
    ground_truth_distances = torch.cdist(outputs['X_gt_L'], outputs['X_gt_L'])
    # compute distances between predicted atoms
    predicted_distances = torch.cdist(outputs['X_L'], outputs['X_L'])
    # compute LDDT score for each pair of distances
    difference_distances = torch.abs(ground_truth_distances - predicted_distances)
    lddt_matrix = torch.zeros_like(difference_distances)
    lddt_matrix = 0.25 * (difference_distances < 4.0) + \
                    0.25 * (difference_distances < 2.0) + \
                    0.25 * (difference_distances < 1.0) + \
                    0.25 * (difference_distances < 0.5) 
    # remove unresolved atoms, atoms within same residue
    is_real_atom = ChemData().heavyatom_mask[outputs['seq']]
    is_resolved_atom_L = outputs["crd_mask_I"][is_real_atom]
    is_unresolved_distance_LL = is_resolved_atom_L[...,None] & is_resolved_atom_L[None,...]
    in_same_residue_LL = outputs["f"]["tok_idx"][:,None] == outputs["f"]["tok_idx"][None,:]

    lddt_values = {}
    for mask, mask_type in get_lddt_masks(outputs):
        mask = mask & is_unresolved_distance_LL & ~in_same_residue_LL
        lddt = torch.div(lddt_matrix[:, mask].sum(dim=(-1)), mask.sum(dim=(-1,-2)))
        lddt_values[f"lddt_{mask_type}"] = lddt
    return lddt_values, ('t_quantile_4',)

def get_lddt_masks(outputs):
    D, L = outputs['X_L'].shape[:2]
    
    tok_idx = outputs["f"]["tok_idx"]
    is_protein_L = outputs["f"]["is_protein"][tok_idx]
    is_dna_L = outputs["f"]["is_dna"][tok_idx]
    is_rna_L = outputs["f"]["is_rna"][tok_idx]
    is_ligand_L = outputs["f"]["is_ligand"][tok_idx]
    asym_id_L = outputs["f"]["asym_id"][tok_idx]
    same_chain_LL = asym_id_L[:,None] == asym_id_L[None,:]
    for mask_type in ['all', 'protein_intra', 'protein_inter', 'ligand_intra', 'ligand_inter']:
        if mask_type == 'all':
            mask = torch.ones((L, L), dtype=torch.bool)
        elif mask_type == 'protein_intra':
            mask = is_protein_L[:,None] & is_protein_L[None,:] 
            mask *= same_chain_LL
        elif mask_type == 'protein_inter':
            mask = is_protein_L[:,None] & is_protein_L[None,:]
            mask *= ~same_chain_LL
        elif mask_type == 'ligand_intra':
            mask = is_ligand_L[:,None] & is_ligand_L[None,:]
            mask *= same_chain_LL
        elif mask_type == 'ligand_inter':
            mask = is_ligand_L[:,None] & is_ligand_L[None,:]
            mask *= ~same_chain_LL
        elif mask_type == 'protein_ligand_inter':
            mask = is_protein_L[:,None] & is_ligand_L[None,:]
        yield (mask, mask_type)

def diffusion_losses(config, outputs):

    loss = Loss(**config.loss)

    loss_dict_by_type = {}
    t = outputs['t']
    X_noisy_L = outputs['X_noisy_L']
    sigma_data = 16

    null_pred = (sigma_data**2 / (sigma_data**2 + t**2))[...,None,None] * X_noisy_L

    sigma_gt = torch.var(outputs['X_gt_L'], dim=(1,2))**0.5
    for input_type, X_L in (
        ('pred', outputs['X_L']),
        # ('input', outputs['X_noisy_L']),
        ('true', outputs['X_gt_L']),
        ('null_pred', null_pred),
    ):
        l_total, _, loss_dict_batched = loss(
            outputs['f'],
            X_L,
            outputs['X_gt_L'],
            outputs['t'],
        )
        # loss_dict_by_type[input_type] = loss_dict_batched
        loss_dict_batched_prefixed = {f'{k}.{input_type}':v for k,v in loss_dict_batched.items()}
        loss_dict_by_type.update(loss_dict_batched_prefixed)

        # Correcting for EDM : AF3 lambda conversion
        edm_corr = (t+loss.sigma_data)**2 / (t*loss.sigma_data)**2
        loss_dict_batched_edm = {k:v * edm_corr for k,v in loss_dict_batched.items()}
        loss_dict_batched_prefixed_edm = {f'{k}_edm.{input_type}':v for k,v in loss_dict_batched_edm.items()}
        loss_dict_by_type.update(loss_dict_batched_prefixed_edm)

        # Correcting for Var(gt) != sigma_data
        expected_loss_gt = 1 / (loss.sigma_data**2 + t**2) * (loss.sigma_data**2 + t**2 * sigma_gt**2 / loss.sigma_data**2)
        loss_dict_batched_edm_gt_corr = {k: edm_corr * v / expected_loss_gt for k,v in loss_dict_batched.items()}
        loss_dict_batched_prefixed_edm = {f'{k}_edm_gt_corr.{input_type}':v for k,v in loss_dict_batched_edm_gt_corr.items()}
        loss_dict_by_type.update(loss_dict_batched_prefixed_edm)
    
    o = flatten_dictionary(loss_dict_by_type)
    o['pred_over_null_pred'] = o['diffusion_loss.pred'] / o['diffusion_loss.null_pred']
    o['pred_over_null_pred_norm'] = o['diffusion_loss_edm_gt_corr.pred'] / o['diffusion_loss_edm_gt_corr.null_pred']
    return o, ('t_quantile_4',)

def get_normal_quantiles(n):
    # Generate n evenly spaced probabilities between 0 and 1
    probabilities = np.linspace(0, 1, n)
    # Use the percent point function (inverse CDF) of the standard normal distribution
    return norm.ppf(probabilities)

def get_t_quantiles(t, sigma_data, n):
    bins = sigma_data * np.exp(-1.2 + 1.5 * get_normal_quantiles(n+1))
    t_binned_list = []
    for t in t:
        t_bin = np.digitize(t, bins) - 1 
        bin_start = bins[t_bin]
        bin_end = bins[t_bin+1]
        t_binned = f't=[{bin_start:.2f},{bin_end:.2f})'
        t_binned_list.append(t_binned)
    return t_binned_list

class NetworkOutputGradSanityCheck(Callback):
    def __init__(self, call_n_times=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.call_n_times = call_n_times
        self.call_count = 0

    def on_after_backward(self, trainer, pl_module):

        if self.call_count < self.call_n_times:
            self.call_count += 1
            r_projection_weight = pl_module.model.model.diffusion_module.atom_attention_decoder.to_r_update[1].weight
            ic(
                torch.linalg.norm(r_projection_weight) if r_projection_weight is not None else None,
                torch.linalg.norm(r_projection_weight.grad) if r_projection_weight.grad is not None else None,
            )

class MonitorActivations(Callback):

    def make_hook(self, label):
        def hook(module, args, kwargs, output):
            activation_metrics = {
                f'{label}:inter_batch_cosine_similarity': F.cosine_similarity(
                    torch.flatten(output[0]),
                    torch.flatten(output[1]),
                    dim=0,
                ),
                f'{label}:intra_batch_cosine_similarity_to_elem_0': F.cosine_similarity(
                    output[0][0:1],
                    output[0],
                ).mean(),
            }
            self.log_dict(activation_metrics)
        return hook


    def setup(self, trainer, pl_module, stage):
        self.pl_module = pl_module
        self.trainer = trainer

        pl_module.model.model.diffusion_module.atom_attention_decoder.register_forward_hook(
            self.make_hook(
                'diffusion_module.atom_attention_decoder',
            ),
            with_kwargs=True
        )

class FindUnusedParameters(Callback):
    def __init__(self, only_once=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.only_once = only_once
        self.called = False

    def on_after_backward(self, trainer, pl_module):
        if self.called and self.only_once:
            return
        self.called=True
        # Calculate unused parameters after each batch
        unused_params = [name for name, param in pl_module.named_parameters() if param.grad is None]

        # Log unused parameters
        logging.info(f'global_step={pl_module.global_step}: parameters with no gradient: {json.dumps(unused_params, indent=4)}')
        if unused_params:
            raise Exception('storp')

class WriteToPymol(Callback):
    def __init__(self, only_once=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.only_once = only_once
        self.called = False
        pymol.init('http://chesaw.dhcp.ipd:9123')

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx: int) -> None:

        if self.called and self.only_once:
            return
        self.called=True

        pymol_tools.clear()
        predicted = outputs

        logger.info('predicted:\n' + pretty_describe_dict(predicted))
        ic(
            predicted['loss']
        )

        D = predicted['X_L'].shape[0]

        max_to_show = 16
        grid_slot = 1
        cmd.set('grid_mode', 1)
        for i in range(min(D, max_to_show)):

            X_gt_L = predicted['X_gt_L'][i]
            X_L = predicted['X_L'][i]
            X_noisy_L = predicted['X_noisy_L'][i]
            t = predicted['t'][i]

            label=pymol_tools.show_pymol(
                pymol_tools.to_atom37(X_noisy_L, predicted['crd_mask_I']),
                predicted['seq'],
                predicted['bond_feats'],
                label=f'input_{i}_t_{t.item():.2f}'
            )
            cmd.set('grid_slot', grid_slot, label)
            cmd.color('yellow', label)

            label=pymol_tools.show_pymol(
                pymol_tools.to_atom37(X_L, predicted['crd_mask_I']),
                predicted['seq'],
                predicted['bond_feats'],
                label=f'pred_{i}_t_{t.item():.2f}'
            )
            cmd.set('grid_slot', grid_slot, label)
            cmd.color('green', label)

            label = pymol_tools.show_pymol(
                pymol_tools.to_atom37(X_gt_L, predicted['crd_mask_I']),
                predicted['seq'],
                predicted['bond_feats'],
                label=f'gt_{i}'
            )
            cmd.set('grid_slot', grid_slot, label)
            cmd.color('blue', label)
            grid_slot += 1
        
        cmd.show_as('licorice', 'all')
        cmd.alter('name CA', 'vdw=2.0')
        # cmd.set('sphere_transparency', 0.0)
        cmd.show('spheres', 'name CA')


        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)