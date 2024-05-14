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
from rf2aa.debug import pretty_describe_dict
from rf2aa.model.AF3_structure import Loss
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
        for metric in [diffusion_losses]:
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
