import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import numpy as np

from functools import partial

import hydra
import os
import time

from rf2aa.data.compose_dataset import compose_dataset, compose_single_item_dataset
from rf2aa.data.data_loader import loader_atomize_pdb
from rf2aa.data.dataloader_adaptor import prepare_input, get_loss_calc_items
from rf2aa.debug import debug_unused_params, debug_used_params, debug_grads
from rf2aa.training.EMA import EMA, count_parameters
from rf2aa.loss.loss_factory import get_loss_and_misc
from rf2aa.training.optimizer import add_weight_decay
from rf2aa.training.recycling import recycle_step_legacy, recycle_step_packed, recycle_sampling
from rf2aa.model.network import RosettaFold
from rf2aa.model.RoseTTAFoldModel import LegacyRoseTTAFoldModule
from rf2aa.training.scheduler import get_stepwise_decay_schedule_with_warmup
import rf2aa.util as util
from rf2aa.util_module import XYZConverter
from rf2aa.chemical import ChemicalData as ChemData
from rf2aa.chemical import initialize_chemdata

#TODO: control environment variables from config
# limit thread counts
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:512"

## To reproduce errors
import random

def seed_all(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

torch.set_num_threads(4)
#torch.autograd.set_detect_anomaly(True)

class Trainer:
    def __init__(self, config) -> None:
        self.config = config

        assert self.config.ddp_params.batch_size == 1, "batch size is assumed to be 1"
        if self.config.experiment.output_dir is not None:
            self.output_dir = self.config.experiment.output_dir 
        else:
            self.output_dir = "models/"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def construct_model(self):
        raise NotImplementedError()

    def construct_optimizer(self):
        if self.config.training_params.weight_decay is not None:
            opt_params = add_weight_decay(self.model, self.config.training_params.weight_decay)
        else:
            opt_params = self.model.parameters()
        self.optimizer = torch.optim.AdamW(opt_params, lr=self.config.training_params.learning_rate)

    def construct_scheduler(self):
        self.scheduler = get_stepwise_decay_schedule_with_warmup(self.optimizer, \
                                **self.config.training_params.learning_rate_schedule)    

    def construct_scaler(self):
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.training_params.use_amp)
    
    def load_checkpoint(self, rank):
        if self.config.training_params.resume_train:
            checkpoint_path = f"{self.output_dir}/{self.config.experiment.name}_last.pt"
        elif self.config.eval_params.checkpoint_path:
            checkpoint_path = self.config.eval_params.checkpoint_path
        map_location = {"cuda:0": f"cuda:{rank}"}
        self.checkpoint = torch.load(checkpoint_path, map_location=map_location)
        print(f"Loading checkpoint from {checkpoint_path} on rank:{rank}")

    def load_model(self):
        torch.cuda.empty_cache()
        if self.config.training_params.resume_train is None:
            raise ValueError("Should not load model when resume_train is True")
        #TODO: check if model should load the final state dict and not the EMA
        self.model.module.model.load_state_dict(self.checkpoint["final_state_dict"], strict=True)
        self.model.module.shadow.load_state_dict(self.checkpoint["model_state_dict"], strict=False)
        print("Checkpoint loaded into model")

    def load_optimizer(self):
        if self.config.training_params.resume_train is None:
            raise ValueError("Should not load optimizer when resume_train is True") 
        self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])

    def load_scheduler(self):
        if self.config.training_params.resume_train is None:
            raise ValueError("Should not load scheduler when resume_train is True") 
        self.scheduler.load_state_dict(self.checkpoint['scheduler_state_dict'])
       
    def load_scaler(self):
        if self.config.training_params.resume_train is None:
            raise ValueError("Should not load scaler when resume_train is True") 
        self.scaler.load_state_dict(self.checkpoint['scaler_state_dict'])

    def construct_dataset(self, init_db, rank, world_size):
        return compose_dataset(
            init_db, self.config.dataset_params, self.config.loader_params, rank, world_size
        )
    
    def construct_loss_function(self):
        raise NotImplementedError() 

    def move_constants_to_device(self, gpu):
        self.fi_dev = ChemData().frame_indices.to(gpu)
        self.xyz_converter = XYZConverter().to(gpu)

        self.l2a = ChemData().long2alt.to(gpu)
        self.aamask = ChemData().allatom_mask.to(gpu)
        self.num_bonds = ChemData().num_bonds.to(gpu)
        self.atom_type_index = ChemData().atom_type_index.to(gpu)
        self.ljlk_parameters = ChemData().ljlk_parameters.to(gpu)
        self.lj_correction_parameters = ChemData().lj_correction_parameters.to(gpu)
        self.hbtypes = ChemData().hbtypes.to(gpu)
        self.hbbaseatoms = ChemData().hbbaseatoms.to(gpu)
        self.hbpolys = ChemData().hbpolys.to(gpu)
        self.cb_len = ChemData().cb_length_t.to(gpu)
        self.cb_ang = ChemData().cb_angle_t.to(gpu)
        self.cb_tor = ChemData().cb_torsion_t.to(gpu)


    def checkpoint_model(self, epoch, metadata={}):
        checkpoint_data = {
                    'epoch'               : epoch,
                    'model_state_dict'    : self.model.module.shadow.state_dict(),
                    'final_state_dict'    : self.model.module.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'scaler_state_dict'   : self.scaler.state_dict(),
                    'training_config'     : dict(self.config),
                    }
        checkpoint_data.update(metadata)
        torch.save(checkpoint_data, f"{self.output_dir}/{self.config.experiment.name}_{epoch}.pt")

    
    def launch_distributed_training(self):
        world_size = torch.cuda.device_count()
        if ('MASTER_ADDR' not in os.environ):
            os.environ['MASTER_ADDR'] = '127.0.0.1' # multinode requires this set in submit script
        if ('MASTER_PORT' not in os.environ):
            os.environ['MASTER_PORT'] = '%d'%self.config.ddp_params.port

        if ("SLURM_NTASKS" in os.environ and "SLURM_PROCID" in os.environ):
            world_size = int(os.environ["SLURM_NTASKS"])
            rank = int (os.environ["SLURM_PROCID"])
            print ("Launched from slurm", rank, world_size)
            self.train_model(rank, world_size)
            #mp.spawn(self.train_model, args=(world_size,), nprocs=world_size, join=True)

        else:
            print ("Launched from interactive")
            world_size = torch.cuda.device_count()

            if world_size == 0:
                print ("Error! No GPUs found!")
            elif world_size == 1:
                # No need for multiple processes with 1 GPU
                self.train_model(0, world_size)
            else:
                mp.spawn(self.train_model, args=(world_size,), nprocs=world_size, join=True)

    def init_process_group(self, rank, world_size):
        gpu = rank % torch.cuda.device_count()
        dist.init_process_group(backend="gloo", world_size=world_size, rank=rank)
        torch.cuda.set_device("cuda:%d"%gpu)
        return gpu
    
    def cleanup(self):
        dist.destroy_process_group()

    def train_model(self, rank, world_size):
        """ runs model training on each gpu """ 
        gpu = self.init_process_group(rank, world_size) 

        #fd initialize chemical data based on input arguments
        #   this needs to be initialized first
        init = partial(initialize_chemdata,self.config.chem_params)
        init()

        train_loader, train_sampler, valid_loaders, valid_samplers = self.construct_dataset(
            init, rank, world_size
        )

        self.train_loader = train_loader
        self.valid_loaders = valid_loaders

        # move global information to device
        self.move_constants_to_device(gpu)

        self.construct_model(device=gpu)
        self.model = DDP(self.model, device_ids=[gpu], find_unused_parameters=False, broadcast_buffers=False)
        if rank == 0:
            print(f"Loading model with {count_parameters(self.model)} parameters")

        self.construct_optimizer()
        self.construct_scheduler()
        self.construct_scaler()
        if self.config.training_params.resume_train:
            self.load_checkpoint(gpu)
            self.load_model()
            self.load_optimizer()
            self.load_scheduler()
            self.load_scaler()
        self.recycle_schedule = recycle_sampling["by_batch"](self.config.loader_params.maxcycle, 
                                                             self.config.experiment.n_epoch,
                                                             self.config.dataset_params.n_train,
                                                             world_size)
        for epoch in range(self.config.experiment.n_epoch):
            train_sampler.set_epoch(epoch) #TODO: need to make sure each gpu gets a different example
            self.train_epoch(epoch, rank, world_size)
            for _, valid_sampler in valid_samplers.items():
                valid_sampler.set_epoch(epoch)

            if (
                self.config.dataset_params.validate_every_n_epochs > 0 
                and epoch % self.config.dataset_params.validate_every_n_epochs==0
            ):
                self.valid_epoch(epoch, rank, world_size)

        self.cleanup() 

    def train_epoch(self, epoch, rank, world_size):
        """ train model """
        # turn on gradients
        self.model.train()

        # clear gradients
        self.optimizer.zero_grad()
        start_time = time.time()
        for train_idx, inputs in enumerate(self.train_loader):
            n_cycle = self.recycle_schedule[epoch, train_idx]  # number of recycling

            # run forward pass and compute loss
            loss, loss_dict = self.train_step(inputs, n_cycle)
            
            # aggregate loss and update parameters
            loss = loss / self.config.ddp_params.accum
            self.scaler.scale(loss).backward()

            if train_idx%self.config.ddp_params.accum == 0:  
                self.update_parameters()
            
                if train_idx % self.config.log_params.log_every_n_examples == 0 and rank == 0:
                    train_time = time.time() - start_time
                    self.log_intermediate_losses(
                        inputs, loss_dict, n_cycle, 
                        (train_idx+1)*world_size, len(self.train_loader)*world_size, train_time
                    ) 
                torch.cuda.empty_cache()

        if rank == 0:
            self.checkpoint_model(epoch)

    def valid_epoch(self, epoch, rank, world_size):
        """ validate model """
        # turn on gradients
        self.model.eval()

        for dataset_name, valid_loader in self.valid_loaders.items():
            valid_loss_dict = None
            for valid_idx, inputs in enumerate(valid_loader):
                n_cycle = self.config.loader_params.maxcycle

                #fd We could make this a separate function call?
                loss, loss_dict = self.train_step(inputs, n_cycle, nograds=True)  

                if valid_loss_dict is None:
                    valid_loss_dict = torch.zeros_like(torch.stack(list(loss_dict.values())))
                valid_loss_dict += torch.stack(list(loss_dict.values()))

            if len(valid_loader) == 0:
                continue

            valid_loss_dict /= float(len(valid_loader)*world_size)
            dist.all_reduce(valid_loss_dict, op=dist.ReduceOp.SUM)

            # reconstruct loss dictionary
            dict_keys = list(loss_dict.keys())
            valid_loss_dict = { 
                dict_keys[i]:valid_loss_dict[i] for i in range(valid_loss_dict.shape[0]) 
            }

            if rank==0:
                self.log_validation_losses(dataset_name, valid_loss_dict)

    def train_step(self, inputs, n_cycle):
        """ take an input from dataloader, run the model and compute a loss """
        raise NotImplementedError()
    
    def valid_step(self, inputs, n_cycle):
        """ take an input from dataloader, run the model and compute a loss.  No grads/checkpointing """
        raise NotImplementedError()
    
    def update_parameters(self):
        """ scale, clip gradients and update parameters """
        # gradient clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training_params.grad_clip)
        self.scaler.step(self.optimizer)
        scale = self.scaler.get_scale()
        self.scaler.update()
        skip_lr_sched = (scale != self.scaler.get_scale())
        self.optimizer.zero_grad()
        if not skip_lr_sched:
            self.scheduler.step()
        self.model.module.update() # apply EMA

    def log_intermediate_losses(self, inputs, loss_dict, n_cycle, Nex, Nepoch, runtime):
        item = inputs[-1]
        max_mem = torch.cuda.max_memory_allocated()/1e9
        print(f"Models: {Nex} of: {Nepoch} Max_Memory: {max_mem:.4f} Runtime: {runtime:.4f}")
        print(f"Example: {item} Recycle:{n_cycle}\n"+
              "\t".join([f"{k}: {v:.4f}" for k,v in loss_dict.items()]))
        torch.cuda.reset_peak_memory_stats()

    def log_validation_losses(self, dataset_name, loss_dict):
        print(f"Dataset: {dataset_name} "+
              "\t".join([f"{k}:{v:.4f}" for k,v in loss_dict.items()]))


class LegacyTrainer(Trainer):
    """ trains Legacy versions of RFAA """
    def __init__(self, config) -> None:
        super().__init__(config)

    def construct_model(self, device="cpu"):
        self.model = LegacyRoseTTAFoldModule(
            **self.config.legacy_model_param,
            aamask = ChemData().allatom_mask.to(device),
            atom_type_index = ChemData().atom_type_index.to(device),
            ljlk_parameters = ChemData().ljlk_parameters.to(device),
            lj_correction_parameters = ChemData().lj_correction_parameters.to(device),
            num_bonds = ChemData().num_bonds.to(device),
            cb_len = ChemData().cb_length_t.to(device),
            cb_ang = ChemData().cb_angle_t.to(device),
            cb_tor = ChemData().cb_torsion_t.to(device),

        ).to(device)
        if self.config.training_params.EMA is not None:
            self.model = EMA(self.model, self.config.training_params.EMA)

    def train_step(self, inputs, n_cycle, nograds=False):
        """ take an input from dataloader, run the model and compute a loss """
        gpu = self.model.device
        # HACK: certain features are constructed during the train step
        # in the future this should only promote the constructed features onto gpu
        task, item, network_input, true_crds, \
            atom_mask, msa, mask_msa, unclamp, negative, symmRs, Lasu, ch_label \
            = prepare_input(inputs, self.xyz_converter, gpu)

        output_i = recycle_step_legacy(self.model, network_input, n_cycle, self.config.training_params.use_amp, nograds=nograds) 
        seq, same_chain, idx_pdb, bond_feats, dist_matrix, atom_frames = get_loss_calc_items(inputs, device=gpu)

        #HACK: indexing into msa and mask msa recycle dimension in arguments of this function
        #HACK: need to promote some inputs to gpu for loss calculation, all promotions should happen together
        msa = msa.to(gpu)
        mask_msa = mask_msa.to(gpu)
        loss, loss_dict = get_loss_and_misc(
            self, # avoid reloading constants to device 
            output_i, true_crds, atom_mask, same_chain,
            seq, msa[:, n_cycle-1], mask_msa[:, n_cycle-1], idx_pdb, bond_feats, dist_matrix, atom_frames, unclamp, negative, task, item, symmRs, Lasu, ch_label, 
            self.config.loss_param
        )
        return loss, loss_dict


class ComposedTrainer(Trainer):
    """ trains composed versions of RFAA """
    def __init__(self, config) -> None:
        super().__init__(config)

    def construct_model(self, device="cpu"):
        self.model = RosettaFold(self.config).to(device)
        if self.config.training_params.EMA is not None:
            self.model = EMA(self.model, self.config.training_params.EMA)

    def train_step(self, inputs, n_cycle, nograds=False):
        """ take an input from dataloader, run the model and compute a loss """
        gpu = self.model.device
        # HACK: certain features are constructed during the train step
        # in the future this should only promote the constructed features onto gpu
        task, item, network_input, true_crds, \
            atom_mask, msa, mask_msa, unclamp, negative, symmRs, Lasu, ch_label \
            = prepare_input(inputs, self.xyz_converter, gpu)

        output_i = recycle_step_packed(self.model, network_input, n_cycle, self.config.training_params.use_amp, nograds=nograds) 
        seq, same_chain, idx_pdb, bond_feats, dist_matrix, atom_frames = get_loss_calc_items(inputs, device=gpu)

        #HACK: indexing into msa and mask msa recycle dimension in arguments of this function
        #HACK: need to promote some inputs to gpu for loss calculation, all promotions should happen together
        msa = msa.to(gpu)
        mask_msa = mask_msa.to(gpu)

        loss, loss_dict = get_loss_and_misc(
            self, # avoid reloading constants to device 
            output_i, true_crds, atom_mask, same_chain,
            seq, msa[:, n_cycle-1], mask_msa[:, n_cycle-1], idx_pdb, bond_feats, dist_matrix, atom_frames, unclamp, negative, task, item, symmRs, Lasu, ch_label, 
            self.config.loss_param
        )
        return loss, loss_dict


@hydra.main(version_base=None, config_path='config/train')
def main(config):
    seed_all()
    trainer = trainer_factory[config.experiment.trainer](config=config)
    trainer.launch_distributed_training()

trainer_factory = {
    "legacy": LegacyTrainer,
    "composed": ComposedTrainer,
}

if __name__ == "__main__":
    main()
