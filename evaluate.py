import sys, os, time, datetime, subprocess, shutil
import numpy as np
import pandas as pd
from copy import deepcopy
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils import data
from data_loader import (
    get_train_valid_set, loader_pdb, loader_fb, loader_complex, loader_na_complex, loader_rna, loader_sm, loader_sm_compl, loader_sm_compl_covale,
    loader_atomize_pdb, Dataset, DatasetComplex, DatasetNAComplex, DatasetRNA, DatasetSM, DatasetSMComplex, DistilledDataset, DistributedWeightedSampler
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
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # disable asynchronous execution

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
                 model_param={}, dataset_param={}, loader_param={},
                 loss_param={}, batch_size=1, accum_step=1, maxcycle=4,
                 eval=True, start_epoch=None, out_dir=None, wandb_prefix=None, model_dir='models/'):

        super(Evaluator, self).__init__(
            model_name=model_name,
            n_epoch=n_epoch, step_lr=step_lr, l2_coeff=l2_coeff, port=port, interactive=interactive,
            model_param=model_param, loader_param=loader_param, loss_param=loss_param, batch_size=batch_size,
            accum_step=accum_step, maxcycle=maxcycle, eval=eval, out_dir=out_dir,
            wandb_prefix=wandb_prefix, model_dir=model_dir, dataset_param=dataset_param
        )

        self.start_epoch = start_epoch

    def train_model(self, rank, world_size):
       
        if rank==0: self.record_git_commit()

        gpu = rank % torch.cuda.device_count()
        dist.init_process_group(backend="gloo", world_size=world_size, rank=rank)
        torch.cuda.set_device("cuda:%d"%gpu)

        #define dataset & data loader
        train_ID_dict, valid_ID_dict, weights_dict, train_dict, valid_dict, homo = \
            get_train_valid_set(self.loader_param)

        train_ID_dict['atomize_pdb'] = train_ID_dict['pdb']
        valid_ID_dict['atomize_pdb'] = train_ID_dict['pdb']
        weights_dict['atomize_pdb'] = weights_dict['pdb']
        train_dict['atomize_pdb'] = train_dict['pdb']
        valid_ID_dict['atomize_pdb'] = valid_ID_dict['pdb']
        valid_dict['atomize_pdb'] = valid_dict['pdb']

        if self.dataset_param['n_valid_pdb'] is None: 
            self.dataset_param["n_valid_pdb"] = len(valid_dict['pdb']) 
        if self.dataset_param['n_valid_homo'] is None: 
            self.dataset_param["n_valid_homo"] = len(valid_dict['homo']) 
        if self.dataset_param["n_valid_compl"] is None: 
            self.dataset_param["n_valid_compl"] = len(valid_dict['compl'])
        if self.dataset_param["n_valid_na_compl"] is None: 
            self.dataset_param["n_valid_na_compl"] = len(valid_dict['na_compl'])
        if self.dataset_param["n_valid_rna"] is None: 
            self.dataset_param["n_valid_rna"] = len(valid_dict['rna'])
        if self.dataset_param["n_valid_sm_compl"] is None: 
            self.dataset_param["n_valid_sm_compl"] = \
                len(valid_dict['sm_compl']['CLUSTER'].drop_duplicates())
        if self.dataset_param["n_valid_metal_compl"] is None: 
            self.dataset_param["n_valid_metal_compl"] = \
                len(valid_dict['metal_compl']['CLUSTER'].drop_duplicates())
        if self.dataset_param["n_valid_sm_compl_multi"] is None: 
            self.dataset_param["n_valid_sm_compl_multi"] = \
                len(valid_dict['sm_compl_multi']['CLUSTER'].drop_duplicates())
        if self.dataset_param["n_valid_sm_compl_covale"] is None: 
            self.dataset_param["n_valid_sm_compl_covale"] = \
                len(valid_dict['sm_compl_covale']['CLUSTER'].drop_duplicates())
        if self.dataset_param["n_valid_sm_compl_strict"] is None: 
            self.dataset_param["n_valid_sm_compl_strict"] = \
                len(valid_dict['sm_compl_strict'])
        if self.dataset_param["n_valid_sm"] is None: 
            self.dataset_param["n_valid_sm"] = len(valid_dict['sm'])
        if self.dataset_param["n_valid_atomize_pdb"] is None: 
            self.dataset_param["n_valid_atomize_pdb"] = len(valid_dict['pdb'])

        if (rank==0):
            print ('Loaded (training)',
                len(train_ID_dict['pdb']),'monomers/homomers,',
                len(train_ID_dict['fb']),'distilled monomers,',
                len(train_ID_dict['compl']),'heteromers,',
                len(train_ID_dict['neg']),'negative heteromers,',
                len(train_ID_dict['na_compl']),'nucleic-acid complexes,',
                len(train_ID_dict['na_neg']),'negative nucleic-acid complexes,',
                len(train_ID_dict['rna']),'RNA structures,',
                len(train_ID_dict['sm_compl']), 'small molecule complexes, and',
                len(train_ID_dict['metal_compl']), 'metal ion complexes, and',
                len(train_ID_dict['sm_compl_multi']), 'multi-res ligand complexes, and',
                len(train_ID_dict['sm_compl_covale']), 'covalent ligand complexes, and',
                len(train_ID_dict['sm']), "small molecule crystals."
            )
            print ('Loaded (valid)',
                len(valid_ID_dict['pdb']),'monomers,',
                len(valid_ID_dict['homo']),'homomers,',
                len(valid_ID_dict['compl']),'heteromers,',
                len(valid_ID_dict['neg']),'negative heteromers,',
                len(valid_ID_dict['na_compl']),'nucleic-acid complexes,',
                len(valid_ID_dict['na_neg']),'negative nucleic-acid complexes,',
                len(valid_ID_dict['rna']),'RNA structures,',
                len(valid_ID_dict['sm_compl']), 'small molecule complexes,',
                len(valid_ID_dict['metal_compl']), 'metal ion complexes,',
                len(valid_ID_dict['sm_compl_multi']), 'multi-res ligand complexes,',
                len(valid_ID_dict['sm_compl_covale']), 'covalent ligand complexes,',
                len(valid_ID_dict['sm_compl_strict']), 'small molecule complexes (strict),',
                len(valid_ID_dict['sm']), 'small molecule crystals.'
            )

            print ('Using',
                self.dataset_param['n_valid_pdb'],'monomers,',
                self.dataset_param['n_valid_homo'],'homomers,',
                self.dataset_param['n_valid_compl'],'heteromers,',
                self.dataset_param['n_valid_neg'],'negative heteromers,',
                self.dataset_param['n_valid_na_compl'],'nucleic-acid complexes,',
                self.dataset_param['n_valid_na_neg'],'negative nucleic-acid complexes,',
                self.dataset_param['n_valid_rna'],'RNA structures,',
                self.dataset_param['n_valid_sm_compl'], 'small mol. complexes,',
                self.dataset_param['n_valid_metal_compl'], 'metal ion complexes,',
                self.dataset_param['n_valid_sm_compl_multi'], 'multi-res ligand complexes,',
                self.dataset_param['n_valid_sm_compl_covale'], 'covalent ligand complexes,',
                self.dataset_param['n_valid_sm_compl_strict'], 'small molecule complexes (strict),',
                self.dataset_param['n_valid_sm'], "small molecule crystals,",
                self.dataset_param['n_valid_atomize_pdb'],'monomers (atomized)',
            )

        seed = 0 # always draw the same example from each cluster

        valid_sets = dict(
            pdb = Dataset(
                valid_ID_dict['pdb'][:self.dataset_param['n_valid_pdb']],
                loader_pdb, valid_dict['pdb'],
                self.loader_param, homo, p_homo_cut=-1.0
            ),
            homo = Dataset(
                valid_ID_dict['homo'][:self.dataset_param['n_valid_homo']],
                loader_pdb, valid_dict['homo'],
                self.loader_param, homo, p_homo_cut=2.0
            ),
            compl = DatasetComplex(
                valid_ID_dict['compl'][:self.dataset_param['n_valid_compl']],
                loader_complex, valid_dict['compl'],
                self.loader_param, negative=False
            ),
            na_compl = DatasetNAComplex(
                valid_ID_dict['na_compl'][:self.dataset_param['n_valid_na_compl']],
                loader_na_complex, valid_dict['na_compl'],
                self.loader_param, negative=False, native_NA_frac=1.0
            ),
            na_from_scratch_compl = DatasetNAComplex(
                valid_ID_dict['na_compl'][:self.dataset_param['n_valid_na_compl']],
                loader_na_complex, valid_dict['na_compl'],
                self.loader_param, negative=False, native_NA_frac=0.0
            ),
            rna = DatasetRNA(
                valid_ID_dict['rna'][:self.dataset_param['n_valid_rna']],
                loader_rna, valid_dict['rna'],
                self.loader_param
            ),
            sm_compl = DatasetSMComplex(
                valid_ID_dict['sm_compl'][:self.dataset_param['n_valid_sm_compl']],
                loader_sm_compl, valid_dict['sm_compl'],
                self.loader_param,
            ),
            metal_compl = DatasetSMComplex(
                valid_ID_dict['metal_compl'][:self.dataset_param['n_valid_metal_compl']],
                loader_sm_compl, valid_dict['metal_compl'],
                self.loader_param, task='metal_compl'
            ),
            sm_compl_multi = DatasetSMComplex(
                valid_ID_dict['sm_compl_multi'][:self.dataset_param['n_valid_sm_compl_multi']],
                loader_sm_compl, valid_dict['sm_compl_multi'],
                self.loader_param, task='sm_compl_multi'
            ),
            sm_compl_covale = DatasetSMComplex(
                valid_ID_dict['sm_compl_covale'][:self.dataset_param['n_valid_sm_compl_covale']],
                loader_sm_compl_covale, valid_dict['sm_compl_covale'],
                self.loader_param, task='sm_compl_covale'
            ),
            sm_compl_strict = DatasetSMComplex(
                valid_ID_dict['sm_compl_strict'][:self.dataset_param['n_valid_sm_compl_strict']],
                loader_sm_compl, valid_dict['sm_compl_strict'],
                self.loader_param, task='sm_compl_strict'
            ),
            sm = DatasetSM(
                valid_ID_dict['sm'][:self.dataset_param['n_valid_sm']],
                loader_sm, valid_dict['sm'],
                self.loader_param,
            ),
            atomize_pdb = Dataset(
                valid_ID_dict['atomize_pdb'][:self.dataset_param['n_valid_atomize_pdb']],
                loader_atomize_pdb, valid_dict['atomize_pdb'],
                self.loader_param, homo, p_homo_cut=-1.0, n_res_atomize=3, flank=0
            )
        )
        valid_headers = dict(
                pdb = 'Monomer',
                homo = 'Homo',
                compl = 'Hetero',
                na_compl = 'NA',
                na_from_scratch_compl = 'NAfs',
                rna = 'RNA',
                sm_compl = 'SM Compl',
                metal_compl = 'Metal ion',
                sm_compl_multi = 'Multires ligand',
                sm_compl_covale = "Covalent ligand",
                sm_compl_strict = 'SM Compl (strict)',
                sm = 'SM_CSD',
                atomize_pdb = 'Monomer atomize 3'
            )
        valid_samplers = {
            k : data.distributed.DistributedSampler(v, num_replicas=world_size, rank=rank)
            for k,v in valid_sets.items()
        }
        valid_loaders = {
            k : data.DataLoader(v, sampler=valid_samplers[k], **LOAD_PARAM)
            for k,v in valid_sets.items()
        }
        
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
            # always draw the same examples
            seed = 0 # epoch
            for k, sampler in valid_samplers.items():
                sampler.set_epoch(seed)
            
            # load this epoch's checkpoint
            loaded_epoch, best_valid_loss = self.load_model(ddp_model, self.model_name, gpu, 
                                                            suffix=str(epoch), resume_train=False)
            if loaded_epoch == -2:
                print(f'Checkpoint doesn\'t exist for epoch {epoch}. Quitting.')
                dist.destroy_process_group()
                return

            rng = np.random.RandomState(seed=epoch*world_size+rank)

            df_s = []
            for k,v in valid_loaders.items():
                valid_tot_, valid_loss_, valid_acc_, loss_df= self.valid_pdb_cycle(ddp_model, 
                    v, rank, gpu, world_size, epoch, rng, header=valid_headers[k], 
                    verbose = self.eval) 
                df_s.append(loss_df)

            if rank==0:
                loss_df = pd.concat(df_s)
                loss_df.to_csv(self.out_dir+f'/loss_per_ex_valid_ep{epoch}.csv')

        dist.destroy_process_group()


if __name__ == "__main__":
    from arguments import get_args
    args, dataset_param, model_param, loader_param, loss_param = get_args()
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
                    eval=True,
                    start_epoch=args.start_epoch,
                    interactive=args.interactive,
                    out_dir=args.out_dir,
                    wandb_prefix=args.wandb_prefix,
                    model_dir=args.model_dir,
                    dataset_param=dataset_param)
    evaluator.run_model_training(torch.cuda.device_count())
