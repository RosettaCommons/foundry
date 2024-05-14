import re
import random
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from icecream import ic
import numpy as np
from functools import partial
import hydra
import os
import time
import omegaconf
from contextlib import nullcontext
import datetime
from datetime import timedelta
import certifi
import warnings
import wandb
import logging
import tree
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from rf2aa.data.compose_dataset import compose_dataset, compose_single_item_dataset
from rf2aa.data.dataloader_adaptor import prepare_input, get_loss_calc_items, prepare_input_fm_allatom
from rf2aa.data.dataloader_adaptor_af3 import prepare_input_af3
from rf2aa.flow_matching.interpolant import Interpolant
from rf2aa.flow_matching.sampler import Sampler, AllAtomSampler
from rf2aa.debug import debug_unused_params, debug_used_params, debug_grads, pretty_describe_dict
from rf2aa.training.EMA import EMA, count_parameters
from rf2aa.loss.loss import translation_vector_field
from rf2aa.loss.loss_factory import get_loss_and_misc
from rf2aa.training.optimizer import add_weight_decay
from rf2aa.training.recycling import recycle_step_legacy, recycle_step_packed, recycle_step_gen, recycle_sampling, run_model_forward, recycle_step_generic
from rf2aa.model.network import RosettaFold
from rf2aa.model.RoseTTAFoldModel import LegacyRoseTTAFoldModule
from rf2aa.training.scheduler import get_stepwise_decay_schedule_with_warmup
import rf2aa.util as util
from rf2aa.util_module import XYZConverter
from rf2aa.chemical import ChemicalData as ChemData
from rf2aa.chemical import initialize_chemdata
from rf2aa.set_seed import seed_all
from rf2aa.model import AF3_structure
from rf2aa.callbacks import LogMetrics, FindUnusedParameters, NetworkOutputGradSanityCheck, MonitorActivations
from rf2aa.loggers import LitLogger
  
ic.configureOutput(includeContext=True)

logger = logging.getLogger(__name__)

#TODO: control environment variables from config
# limit thread counts
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:512"
# Update environment variable with correct path (needed for W&B upload)
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
## To reproduce errors

torch.set_num_threads(4)

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def get_param_sizes(model):
    o = {}
    for k, p in model.named_parameters():
        o[k] = (np.array(p.size()).prod(), p.size())
    return o

# define the LightningModule
class LitAF3Repro(L.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
    
        # self.model = torch.nn.Linear(2, 3).to(device)
        self.model = AF3_structure.Model(**self.config.model)
        print_n_params = False
        if print_n_params:
            logger.info(f'{get_n_params(self.model)=}')
            for k, v in sorted(get_param_sizes(self.model).items(), key=lambda item: item[1]):
                n_param, size = v
                # n_param = np.array(p.size()).prod()
                logger.info(f'{n_param=} {k=} {size=}')

        if self.config.training_params.EMA is not None:
            self.model = EMA(self.model, self.config.training_params.EMA)

        def should_ignore(param_name):
            ignore_regexes = [
                re.compile(r'model\.feature_initializer\.input_feature_embedder\.atom_attention_encoder\.process_s_trunk\..*'),
                re.compile(r'model\.feature_initializer\.input_feature_embedder\.atom_attention_encoder\.process_z\..*'),
                re.compile(r'model\.feature_initializer\.input_feature_embedder\.atom_attention_encoder\.process_r\..*'),
                re.compile(r'model\.feature_initializer\.input_feature_embedder\.atom_attention_encoder\.atom_transformer\.diffusion_transformer\.blocks\.\d+\.attention_pair_bias.ln_1\..*'),
                re.compile(r'model\.recycler\.pairformer_stack\.\d+\.attention_pair_bias\.linear_output_project\..*'),
                re.compile(r'model\.recycler\.pairformer_stack\.\d+\.attention_pair_bias\.ada_ln_1\..*'),
                re.compile(r'model\.diffusion_module\.atom_attention_encoder\.atom_transformer\.diffusion_transformer\.blocks\.\d+\.attention_pair_bias\.ln_1\..*'),
                re.compile(r'model\.diffusion_module\.diffusion_transformer\.blocks\.\d+\.attention_pair_bias\.ln_1\..*'),
                re.compile(r'model\.diffusion_module\.atom_attention_decoder\.atom_transformer\.diffusion_transformer\.blocks\.\d+\.attention_pair_bias\.ln_1\..*'),
            ]
            return any(regex.match(param_name) for regex in ignore_regexes)
        params_to_ignore = []
        for param_name, param in self.model.named_parameters():
            if should_ignore(param_name):
                params_to_ignore.append(param_name)
        torch.nn.parallel.DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(
            self.model,
            params_to_ignore
        )
        assert len(params_to_ignore)
        
        self.loss = AF3_structure.Loss(**self.config.loss)

    def training_step(self, batch, batch_idx):

        logger.debug('batch:\n' + pretty_describe_dict(batch))

        # TODO: move data processing to dataset
        batch = tree.map_structure(lambda x: x.detach().cpu() if hasattr(x, 'cpu') else x, batch)
        network_input, loss_input = prepare_input_af3(
            batch,
            **self.config.af3_data_prep,
        )
        # TODO: move data processing to dataset
        network_input = tree.map_structure(lambda x: x.to(self.device), network_input)
        loss_input = tree.map_structure(lambda x: x.to(self.device), loss_input)

        logger.debug('network_input:\n' + pretty_describe_dict(network_input))
        logger.debug('loss_input:\n' + pretty_describe_dict(loss_input))

        n_cycle = random.randint(1, self.config.recycling.max_cycle)

        X_L = self.model(
            network_input,
            n_cycle,
            no_sync=self.model.no_sync,
        )
        
        loss, loss_dict, loss_dict_batched = self.loss(
            f=network_input['f'],
            t=network_input['t'],
            X_L=X_L,
            X_gt_L=loss_input['X_gt_L'],
        )
        self.log('loss', loss, prog_bar=True)
        return dict(
            loss=loss,
            X_L=X_L,
        ) | loss_dict_batched | network_input | loss_input

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.config.optimizer.type)(
            self.model.parameters(),
            **self.config.optimizer.params,
        )
        scheduler = get_stepwise_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=1000,
            num_steps_decay=5e4,
            decay_rate=0.95,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    
    def configure_callbacks(self):
        return [
            LogMetrics(self.config, **self.config.callbacks.log_metrics),
            NetworkOutputGradSanityCheck(),
            MonitorActivations(),
            LearningRateMonitor(logging_interval='step'),
        ]

class LitDataModule(L.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.init = partial(initialize_chemdata, config.chem_params)
        self.init()

    def train_dataloader(self, rank=None, num_replicas=None):
        train_loader, train_sampler, valid_loaders, valid_samplers = compose_dataset(
            self.init, self.config.dataset_params, self.config.loader_params,
            rank or 0,
            num_replicas or 1,
        )
        return train_loader

@hydra.main(version_base=None, config_path='config/train')
def main(config):
    if config.autograd_detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
    model = LitAF3Repro(config)
    datamodule = LitDataModule(config)
    trainer_logger = LitLogger(**config.logger)

    model_checkpoint = ModelCheckpoint(
        every_n_train_steps=1000,
        dirpath='checkpoints',
    )

    trainer = L.Trainer(
        logger=trainer_logger,
        log_every_n_steps=1,
        gradient_clip_val=10,
        callbacks=[model_checkpoint],
        **config.lightning.trainer
    )
    trainer.fit(model=model, datamodule=datamodule)

if __name__ == "__main__":
    main()
