import sys, os, time, datetime, subprocess, shutil
import numpy as np
from copy import deepcopy
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils import data
from data_loader import (
    get_train_valid_set, loader_pdb, loader_fb, loader_complex, loader_na_complex, loader_rna, loader_sm, loader_sm_compl, loader_atomize_pdb,
    Dataset, DatasetComplex, DatasetNAComplex, DatasetRNA, DatasetSM, DatasetSMComplex, DistilledDataset, DistributedWeightedSampler
)
from RoseTTAFoldModel  import RoseTTAFoldModule
from loss import *
from util import *

from train_multi_EMA import Trainer, EMA, count_parameters

# disable openbabel warnings
from openbabel import openbabel as ob
ob.obErrorLog.SetOutputLevel(0)

# distributed data parallel
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
#torch.autograd.set_detect_anomaly(True)
#torch.backends.cudnn.benchmark = False
#torch.backends.cudnn.deterministic = True
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # disable asynchronous execution

## To reproduce errors
import random
random.seed(0)
torch.manual_seed(5924)
np.random.seed(6636)

USE_AMP = False
torch.set_num_threads(4)

LOAD_PARAM = {'shuffle': False,
              'num_workers': 2,
              'pin_memory': True}

class Evaluator(Trainer):
    def __init__(self, model_name='BFF',
                 n_epoch=100, step_lr=100, lr=1.0e-4, l2_coeff=1.0e-2, port=None, interactive=False,
                 model_param={}, loader_param={}, loss_param={}, batch_size=1, accum_step=1, maxcycle=4,
                 eval=True, start_epoch=None, out_dir=None, wandb_prefix=None, model_dir='models/', 
                 n_valid_pdb=None, n_valid_homo=None, n_valid_compl=None, n_valid_na_compl=None, 
                 n_valid_rna=None, n_valid_sm_compl=None, n_valid_sm_compl_ligclus=None, 
                 n_valid_sm_compl_strict=None,n_valid_sm=None):

        super(Evaluator, self).__init__(
            model_name=model_name,
            n_epoch=n_epoch, step_lr=step_lr, l2_coeff=l2_coeff, port=port, interactive=interactive,
            model_param=model_param, loader_param=loader_param, loss_param=loss_param, batch_size=batch_size,
            accum_step=accum_step, maxcycle=maxcycle, eval=eval, out_dir=out_dir,
            wandb_prefix=wandb_prefix, model_dir=model_dir
        )

        self.start_epoch = start_epoch
        self.n_valid_pdb = n_valid_pdb 
        self.n_valid_homo = n_valid_homo 
        self.n_valid_compl = n_valid_compl 
        self.n_valid_na_compl = n_valid_na_compl
        self.n_valid_rna = n_valid_rna
        self.n_valid_sm_compl = n_valid_sm_compl
        self.n_valid_sm_compl_ligclus = n_valid_sm_compl_ligclus
        self.n_valid_sm_compl_strict = n_valid_sm_compl_strict
        self.n_valid_sm = n_valid_sm

    def train_model(self, rank, world_size):
       
        if rank==0: self.record_git_commit()

        gpu = rank % torch.cuda.device_count()
        dist.init_process_group(backend="gloo", world_size=world_size, rank=rank)
        torch.cuda.set_device("cuda:%d"%gpu)

        #define dataset & data loader
        (
            pdb_items, fb_items, compl_items, neg_items, na_compl_items, na_neg_items, rna_items,
            sm_compl_items, sm_items, valid_pdb, valid_homo, valid_compl, valid_neg, valid_na_compl, 
            valid_na_neg, valid_rna, valid_sm_compl, valid_sm_compl_ligclus, valid_sm_compl_strict, 
            valid_sm, homo
        ) = get_train_valid_set(self.loader_param)

        pdb_IDs, pdb_weights, pdb_dict = pdb_items
        fb_IDs, fb_weights, fb_dict = fb_items
        compl_IDs, compl_weights, compl_dict = compl_items
        neg_IDs, neg_weights, neg_dict = neg_items
        na_compl_IDs, na_compl_weights, na_compl_dict = na_compl_items
        na_neg_IDs, na_neg_weights, na_neg_dict = na_neg_items
        rna_IDs, rna_weights, rna_dict = rna_items
        sm_compl_IDs, sm_compl_weights, sm_compl_dict = sm_compl_items
        sm_IDs, sm_weights, sm_dict = sm_items
       
        if self.n_valid_pdb is None: self.n_valid_pdb = len(valid_pdb.keys()) 
        if self.n_valid_homo is None: self.n_valid_homo = len(valid_homo.keys()) 
        if self.n_valid_compl is None: self.n_valid_compl = len(valid_compl.keys())
        if self.n_valid_na_compl is None: self.n_valid_na_compl = len(valid_na_compl.keys())
        if self.n_valid_rna is None: self.n_valid_rna = len(valid_rna.keys())
        if self.n_valid_sm_compl is None: self.n_valid_sm_compl = len(valid_sm_compl.keys())
        if self.n_valid_sm_compl_ligclus is None: self.n_valid_sm_compl_ligclus = len(valid_sm_compl_ligclus.keys())
        if self.n_valid_sm_compl_strict is None: self.n_valid_sm_compl_strict = len(valid_sm_compl_strict.keys())
        if self.n_valid_sm is None: self.n_valid_sm = len(valid_sm.keys())

        if (rank==0):
            print ('Loaded (valid)',
                len(valid_pdb.keys()),'monomers,',
                len(valid_homo.keys()),'homomers,',
                len(valid_compl.keys()),'heteromers,',
                len(valid_na_compl.keys()),'nucleic-acid complexes,',
                len(valid_rna),'RNA structures,',
                len(valid_sm_compl), 'small molecule complexes,',
                len(valid_sm_compl_ligclus), 'small molecule complexes (ligand-clustered),',
                len(valid_sm_compl_strict), 'small molecule complexes (strict),',
                len(valid_sm), 'small molecule crystals.'

            )
            print ('Using',
                self.n_valid_pdb,'monomers,',
                self.n_valid_homo,'homomers,',
                self.n_valid_compl,'heteromers,',
                self.n_valid_na_compl,'nucleic-acid complexes,',
                self.n_valid_rna,'RNA structures,',
                self.n_valid_sm_compl, 'small mol. complexes (fold & dock),',
                self.n_valid_sm_compl_ligclus, 'small molecule complexes (ligand-clustered),',
                self.n_valid_sm_compl_strict, 'small molecule complexes (strict),',
                self.n_valid_sm, "small molecule crystals."
            )

        valid_pdb_set = Dataset(
            list(valid_pdb.keys())[:self.n_valid_pdb],
            loader_pdb, valid_pdb,
            self.loader_param, homo, p_homo_cut=-1.0
        )
        valid_homo_set = Dataset(
            list(valid_homo.keys())[:self.n_valid_homo],
            loader_pdb, valid_homo,
            self.loader_param, homo, p_homo_cut=2.0
        )
        valid_compl_set = DatasetComplex(
            list(valid_compl.keys())[:self.n_valid_compl],
            loader_complex, valid_compl,
            self.loader_param, negative=False
        )
        valid_na_compl_set = DatasetNAComplex(
            list(valid_na_compl.keys())[:self.n_valid_na_compl],
            loader_na_complex, valid_na_compl,
            self.loader_param, negative=False, native_NA_frac=1.0
        )
        valid_na_from_scratch_compl_set = DatasetNAComplex(
            list(valid_na_compl.keys())[:self.n_valid_na_compl],
            loader_na_complex, valid_na_compl,
            self.loader_param, negative=False, native_NA_frac=0.0
        )
        valid_rna_set = DatasetRNA(
            list(valid_rna.keys())[:self.n_valid_rna],
            loader_rna, valid_rna,
            self.loader_param
        )
        valid_sm_compl_set = DatasetSMComplex(
            list(valid_sm_compl.keys())[:self.n_valid_sm_compl],
            loader_sm_compl, valid_sm_compl,
            self.loader_param,
        )
        valid_sm_compl_ligclus_set = DatasetSMComplex(
            list(valid_sm_compl_ligclus.keys())[:self.n_valid_sm_compl_ligclus],
            loader_sm_compl, valid_sm_compl_ligclus,
            self.loader_param, task='sm_compl_ligclus'
        )
        valid_sm_compl_strict_set = DatasetSMComplex(
            list(valid_sm_compl_strict.keys())[:self.n_valid_sm_compl_strict],
            loader_sm_compl, valid_sm_compl_strict,
            self.loader_param, task='sm_compl_strict'
        )
        valid_sm_set = DatasetSM(
            list(valid_sm.keys())[:self.n_valid_sm],
            loader_sm, valid_sm,
            self.loader_param,
        )
        
        valid_pdb_sampler = data.distributed.DistributedSampler(valid_pdb_set, num_replicas=world_size, rank=rank)
        valid_homo_sampler = data.distributed.DistributedSampler(valid_homo_set, num_replicas=world_size, rank=rank)
        valid_compl_sampler = data.distributed.DistributedSampler(valid_compl_set, num_replicas=world_size, rank=rank)
        valid_na_compl_sampler = data.distributed.DistributedSampler(valid_na_compl_set, num_replicas=world_size, rank=rank)
        valid_na_from_scratch_compl_sampler = data.distributed.DistributedSampler(valid_na_from_scratch_compl_set, num_replicas=world_size, rank=rank)
        valid_rna_sampler = data.distributed.DistributedSampler(valid_rna_set, num_replicas=world_size, rank=rank)
        valid_sm_compl_sampler = data.distributed.DistributedSampler(valid_sm_compl_set, num_replicas=world_size, rank=rank)
        valid_sm_compl_ligclus_sampler = data.distributed.DistributedSampler(valid_sm_compl_ligclus_set, num_replicas=world_size, rank=rank)
        valid_sm_compl_strict_sampler = data.distributed.DistributedSampler(valid_sm_compl_strict_set, num_replicas=world_size, rank=rank)
        valid_sm_sampler = data.distributed.DistributedSampler(valid_sm_set, num_replicas=world_size, rank=rank)

        valid_pdb_loader = data.DataLoader(valid_pdb_set, sampler=valid_pdb_sampler, **LOAD_PARAM)
        valid_homo_loader = data.DataLoader(valid_homo_set, sampler=valid_homo_sampler, **LOAD_PARAM)
        valid_compl_loader = data.DataLoader(valid_compl_set, sampler=valid_compl_sampler, **LOAD_PARAM)
        valid_na_compl_loader = data.DataLoader(valid_na_compl_set, sampler=valid_na_compl_sampler, **LOAD_PARAM)
        valid_na_from_scratch_compl_loader = data.DataLoader(valid_na_from_scratch_compl_set, sampler=valid_na_from_scratch_compl_sampler, **LOAD_PARAM)
        valid_rna_loader = data.DataLoader(valid_rna_set, sampler=valid_rna_sampler, **LOAD_PARAM)
        valid_sm_compl_loader = data.DataLoader(valid_sm_compl_set, sampler=valid_sm_compl_sampler, **LOAD_PARAM)
        valid_sm_compl_ligclus_loader = data.DataLoader(valid_sm_compl_ligclus_set, sampler=valid_sm_compl_ligclus_sampler, **LOAD_PARAM)
        valid_sm_compl_strict_loader = data.DataLoader(valid_sm_compl_strict_set, sampler=valid_sm_compl_strict_sampler, **LOAD_PARAM)
        valid_sm_loader = data.DataLoader(valid_sm_set, sampler=valid_sm_sampler, **LOAD_PARAM)
        
        # move some global data to cuda device
        self.ti_dev = self.ti_dev.to(gpu)
        self.ti_flip = self.ti_flip.to(gpu)
        self.ang_ref = self.ang_ref.to(gpu)
        self.fi_dev = self.fi_dev.to(gpu)
        self.l2a = self.l2a.to(gpu)
        self.aamask = self.aamask.to(gpu)
        self.compute_allatom_coords = self.compute_allatom_coords.to(gpu)
        self.num_bonds = self.num_bonds.to(gpu)
        self.atom_type_index = self.atom_type_index.to(gpu)
        self.ljlk_parameters = self.ljlk_parameters.to(gpu)
        self.lj_correction_parameters = self.lj_correction_parameters.to(gpu)
        self.hbtypes = self.hbtypes.to(gpu)
        self.hbbaseatoms = self.hbbaseatoms.to(gpu)
        self.hbpolys = self.hbpolys.to(gpu)
        self.cb_len = self.cb_len.to(gpu)
        self.cb_ang = self.cb_ang.to(gpu)
        self.cb_tor = self.cb_tor.to(gpu)

        # define model
        model = EMA(RoseTTAFoldModule(
            **self.model_param,
            aamask=self.aamask,
            atom_type_index=self.atom_type_index,
            ljlk_parameters=self.ljlk_parameters,
            lj_correction_parameters=self.lj_correction_parameters,
            num_bonds=self.num_bonds,
            cb_len = self.cb_len,
            cb_ang = self.cb_ang,
            cb_tor = self.cb_tor,
            lj_lin=self.loss_param['lj_lin']
        ).to(gpu), 0.999)

        ddp_model = DDP(model, device_ids=[gpu], find_unused_parameters=False)
        if rank == 0:
            print ("# of parameters:", count_parameters(ddp_model))

        for epoch in range(self.start_epoch, self.start_epoch+self.n_epoch):
            valid_pdb_sampler.set_epoch(epoch)
            valid_homo_sampler.set_epoch(epoch)
            valid_compl_sampler.set_epoch(epoch)
            valid_na_compl_sampler.set_epoch(epoch)
            valid_na_from_scratch_compl_sampler.set_epoch(epoch)
            valid_rna_sampler.set_epoch(epoch)
            valid_sm_compl_sampler.set_epoch(epoch)
            valid_sm_compl_ligclus_sampler.set_epoch(epoch)
            valid_sm_compl_strict_sampler.set_epoch(epoch)

            # load this epoch's checkpoint
            loaded_epoch, best_valid_loss = self.load_model(ddp_model, self.model_name, gpu, 
                                                            suffix=str(epoch), resume_train=False)
            if loaded_epoch == -2:
                print(f'Checkpoint doesn\'t exist for epoch {epoch}. Quitting.')
                dist.destroy_process_group()
                return

            _, _, _ = self.valid_pdb_cycle(ddp_model, valid_pdb_loader, rank, gpu, world_size, 
                epoch, verbose = self.eval)
            _, _, _ = self.valid_pdb_cycle(ddp_model, valid_homo_loader, rank, gpu, world_size, 
                epoch, header="Homo", verbose = self.eval)
            _, _, _ = self.valid_pdb_cycle(ddp_model, valid_compl_loader, rank, gpu, world_size, 
                epoch, header="Hetero", verbose = self.eval)
            _, _, _ = self.valid_pdb_cycle(ddp_model, valid_na_compl_loader, rank, gpu, world_size, 
                epoch, header="NA", verbose = self.eval)
            _, _, _ = self.valid_pdb_cycle(ddp_model, valid_na_from_scratch_compl_loader, rank, gpu, 
                world_size, epoch, header="NAfs", verbose = self.eval)
            _, _, _ = self.valid_pdb_cycle(ddp_model, valid_rna_loader, rank, gpu, world_size, 
                epoch, header="RNA", verbose = self.eval)
            valid_tot, valid_loss, valid_acc = self.valid_pdb_cycle(ddp_model, valid_sm_compl_loader, 
                rank, gpu, world_size, epoch, header="SM Compl", verbose = self.eval) 
            _, _, _ = self.valid_pdb_cycle(ddp_model, valid_sm_compl_ligclus_loader, 
                rank, gpu, world_size, epoch, header="SM Compl (lig. clus.)", verbose = self.eval) 
            _, _, _ = self.valid_pdb_cycle(ddp_model, valid_sm_compl_strict_loader, 
                rank, gpu, world_size, epoch, header="SM Compl (strict)", verbose = self.eval) 

        dist.destroy_process_group()


if __name__ == "__main__":
    from arguments import get_args
    args, model_param, loader_param, loss_param = get_args()
    if args.start_epoch is None:
        sys.exit('-start_epoch is required for evaluate.py')

    print (args)

    mp.freeze_support()
    evaluator = Evaluator(model_name=args.model_name,
                    n_epoch=args.num_epochs, step_lr=args.step_lr, lr=args.lr, l2_coeff=1.0e-2,
                    port=args.port, model_param=model_param, loader_param=loader_param, 
                    loss_param=loss_param, 
                    batch_size=args.batch_size,
                    accum_step=args.accum,
                    maxcycle=args.maxcycle,
                    eval=args.eval,
                    start_epoch=args.start_epoch,
                    interactive=args.interactive,
                    out_dir=args.out_dir,
                    wandb_prefix=args.wandb_prefix,
                    model_dir=args.model_dir,
                    n_valid_pdb=args.n_valid_pdb, 
                    n_valid_homo=args.n_valid_homo, 
                    n_valid_compl=args.n_valid_compl, 
                    n_valid_na_compl=args.n_valid_na_compl, 
                    n_valid_rna=args.n_valid_rna,
                    n_valid_sm_compl=args.n_valid_sm_compl,
                    n_valid_sm_compl_ligclus=args.n_valid_sm_compl_ligclus, 
                    n_valid_sm_compl_strict=args.n_valid_sm_compl_strict,
                    n_valid_sm=args.n_valid_sm)
    evaluator.run_model_training(torch.cuda.device_count())
