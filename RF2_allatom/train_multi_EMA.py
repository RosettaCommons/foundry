import sys, os, time, subprocess
import numpy as np
from copy import deepcopy
from collections import OrderedDict
import wandb
import torch
import torch.nn as nn
from torch.utils import data
from functools import partial
from data_loader import (
    get_train_valid_set, loader_pdb, loader_fb, loader_complex, loader_na_complex, loader_rna, loader_small_molecule, loader_sm_compl, loader_atomize_pdb,
    Dataset, DatasetComplex, DatasetNAComplex, DatasetRNA, DatasetSM, DatasetSMComplex, DistilledDataset, DistributedWeightedSampler
)
from kinematics import xyz_to_c6d, c6d_to_bins, xyz_to_t2d, xyz_to_bbtor, get_init_xyz
from RoseTTAFoldModel  import RoseTTAFoldModule
from loss import *
from util import *
from util_module import ComputeAllAtomCoords
from scheduler import get_linear_schedule_with_warmup, get_stepwise_decay_schedule_with_warmup

# disable openbabel warnings
from openbabel import openbabel as ob
ob.obErrorLog.SetOutputLevel(0)

# distributed data parallel
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(5924)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

## To reproduce errors
#import random
np.random.seed(6636)
#random.seed(0)

USE_AMP = False
torch.set_num_threads(4)

N_PRINT_TRAIN = 4
#BATCH_SIZE = 1 * torch.cuda.device_count()

# num structs per epoch
# must be divisible by #GPUs
#N_EXAMPLE_PER_EPOCH = 1208
N_EXAMPLE_PER_EPOCH = 6632 # divisible by 8

LOAD_PARAM = {'shuffle': False,
              'num_workers': 0,
              'pin_memory': True}

DEBUG = False
if DEBUG:
    N_EXAMPLE_PER_EPOCH =8
    #os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # disable asynchronous execution
    LOAD_PARAM['num_workers'] = 0

def add_weight_decay(model, l2_coeff):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        #if len(param.shape) == 1 or name.endswith(".bias"):
        if "norm" in name or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay, 'weight_decay': l2_coeff}]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class EMA(nn.Module):
    def __init__(self, model, decay):
        super().__init__()
        self.decay = decay

        self.model = model
        self.shadow = deepcopy(self.model)

        for param in self.shadow.parameters():
            param.detach_()

    @torch.no_grad()
    def update(self):
        if not self.training:
            print("EMA update should only be called during training", file=stderr, flush=True)
            return

        model_params = OrderedDict(self.model.named_parameters())
        shadow_params = OrderedDict(self.shadow.named_parameters())

        # check if both model contains the same set of keys
        assert model_params.keys() == shadow_params.keys()

        for name, param in model_params.items():
            # see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
            # shadow_variable -= (1 - decay) * (shadow_variable - variable)
            if param.requires_grad:
                shadow_params[name].sub_((1. - self.decay) * (shadow_params[name] - param))

        model_buffers = OrderedDict(self.model.named_buffers())
        shadow_buffers = OrderedDict(self.shadow.named_buffers())

        # check if both model contains the same set of keys
        assert model_buffers.keys() == shadow_buffers.keys()

        for name, buffer in model_buffers.items():
            # buffers are copied
            shadow_buffers[name].copy_(buffer)

    def forward(self, *args, **kwargs):
        if self.training:
            return self.model(*args, **kwargs)
        else:
            return self.shadow(*args, **kwargs)

class Trainer():
    def __init__(self, model_name='BFF',
                 n_epoch=100, step_lr=100, lr=1.0e-4, l2_coeff=1.0e-2, port=None, interactive=False,
                 model_param={}, loader_param={}, loss_param={}, batch_size=1, accum_step=1, maxcycle=4,
                 eval=False, outdir=None, wandb_prefix=None):
        self.model_name = model_name #"BFF"
        #self.model_name = "%s_%d_%d_%d_%d"%(model_name, model_param['n_module'], 
        #                                    model_param['n_module_str'],
        #                                    model_param['d_msa'],
        #                                    model_param['d_pair'])
        #
        self.n_epoch = n_epoch
        self.step_lr = step_lr
        self.init_lr = lr
        self.l2_coeff = l2_coeff
        self.port = port
        self.interactive = interactive
        self.eval = eval
        #
        self.model_param = model_param
        self.loader_param = loader_param
        self.loss_param = loss_param
        self.ACCUM_STEP = accum_step
        self.batch_size = batch_size
        self.outdir = outdir 
        if outdir is not None: 
            os.makedirs(self.outdir, exist_ok=True)
            if outdir[-1] != '/': self.outdir += '/'
        self.wandb_prefix = wandb_prefix

        # for all-atom str loss
        self.ti_dev = torsion_indices
        self.ti_flip = torsion_can_flip
        self.ang_ref = reference_angles
        self.fi_dev = frame_indices
        self.l2a = long2alt
        self.aamask = allatom_mask
        self.num_bonds = num_bonds
        self.atom_type_index = atom_type_index
        self.ljlk_parameters = ljlk_parameters
        self.lj_correction_parameters = lj_correction_parameters
        self.hbtypes = hbtypes
        self.hbbaseatoms = hbbaseatoms
        self.hbpolys = hbpolys
        self.cb_len = cb_length_t
        self.cb_ang = cb_angle_t
        self.cb_tor = cb_torsion_t

        # module torsion -> allatom
        self.compute_allatom_coords = ComputeAllAtomCoords()

        # loss & final activation function
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.active_fn = nn.Softmax(dim=1)

        self.maxcycle = maxcycle

        self.pdb_counter=0
        
    def calc_loss(self, logit_s, label_s,
                  logit_aa_s, label_aa_s, mask_aa_s,
                  pred, pred_tors, pred_allatom, true,
                  mask_crds, mask_BB, mask_2d, same_chain,
                  pred_lddt, idx, atom_frames=None, unclamp=False, negative=False, interface=False,
                  verbose=False, ctr=0,
                  w_dist=1.0, w_aa=1.0, w_str=1.0, w_lddt=1.0, w_bond=1.0, w_clash=0.0, w_hb=0.0, w_dih=0.0,
                  lj_lin=0.85, eps=1e-6
    ):
        # dictionary for keeping track of losses
        loss_dict = {}

        B, L = true.shape[:2]
        seq = label_aa_s[:,0].clone()

        assert (B==1) # fd - code assumes a batch size of 1
        
        loss_s = list()
        tot_loss = 0.0
        
        # c6d loss
        for i in range(4):
            loss = self.loss_fn(logit_s[i], label_s[...,i]) # (B, L, L)
            loss = (mask_2d*loss).sum() / (mask_2d.sum() + eps)
            tot_loss += w_dist*loss
            loss_s.append(loss[None].detach())
            
            loss_dict[f'c6d_{i}'] = float(loss.detach())

        # masked token prediction loss
        loss = self.loss_fn(logit_aa_s, label_aa_s.reshape(B, -1))
        loss = loss * mask_aa_s.reshape(B, -1)
        loss = loss.sum() / (mask_aa_s.sum() + 1e-8)
        tot_loss += w_aa*loss
        loss_s.append(loss[None].detach())

        loss_dict['aa_cce'] = float(loss.detach())

        ### GENERAL LAYERS
        # Structural loss
        dclamp = 300.0 if unclamp else 30.0 
        frames, frame_mask = get_frames(
            pred_allatom[-1,None,...], mask_crds, seq, self.fi_dev, atom_frames)
        # update frames and frames_mask to only include BB frames (have to update both for compatibility with compute_general_FAPE)
        frames_BB = frames.clone()
        frames_BB[..., 1:, :, :] = 0
        frame_mask_BB = frame_mask.clone()
        frame_mask_BB[...,1:] =False

        L1 = same_chain[0,0,:].sum()
        mask_BBA = mask_BB.clone()
        mask_BBA[0, L1:] = False
        if torch.sum(mask_BBA) >0:
            l_fape_A = compute_general_FAPE(
                pred[:,mask_BBA,:,:3],
                true[:,mask_BBA[0],:3],
                mask_crds[:,mask_BBA[0], :3],
                frames_BB[:,mask_BBA[0]],
                frame_mask_BB[:,mask_BBA[0]],
                dclamp=dclamp
            )
            loss_dict['fape_c1'] = float(l_fape_A[-1].detach())
        mask_BBB = mask_BB.clone()
        mask_BBB[0,:L1] = False
        if torch.sum(mask_BBB) >0:
            l_fape_B = compute_general_FAPE(
                pred[:, mask_BBB,:,:3],
                true[:,mask_BBB[0],:3,:3],
                mask_crds[:,mask_BBB[0], :3],
                frames_BB[:,mask_BBB[0]],
                frame_mask_BB[:,mask_BBB[0]],
                Z=4,
                dclamp=4
            )
            loss_dict['fape_c2'] = float(l_fape_B[-1].detach())
        if negative: # inter-chain fapes should be ignored for negative cases
            fracA = float(L1)/len(same_chain[0,0])
            tot_str = fracA*l_fape_A + (1.0-fracA)*l_fape_B

        else:
            tot_str = compute_general_FAPE(
                pred[:,mask_BB,:,:3],
                true[:,mask_BB[0],:3],
                mask_crds[:,mask_BB[0],:3],
                frames_BB[:,mask_BB[0]],
                frame_mask_BB[:,mask_BB[0]],
                dclamp=dclamp
            )
        num_layers = pred.shape[0]
        gamma = 0.99
        w_bb_fape = torch.pow(torch.full((num_layers,), gamma, device=pred.device), torch.arange(num_layers, device=pred.device))
        w_bb_fape = torch.flip(w_bb_fape, (0,))
        w_bb_fape = w_bb_fape / w_bb_fape.sum()
        bb_l_fape = (w_bb_fape*tot_str).sum()
        tot_loss += 0.5*w_str*bb_l_fape
        if "fape_c2" in loss_dict.keys():
            lig_fape = (w_bb_fape*l_fape_B).sum()
            tot_loss += 0.5*w_str*lig_fape
        loss_s.append(tot_str.detach())
        loss_dict['tot_str'] = float(bb_l_fape.detach())
        
        # AllAtom loss
        # get ground-truth torsion angles
        true_tors, true_tors_alt, tors_mask, tors_planar = get_torsions(
            true, seq, self.ti_dev, self.ti_flip, self.ang_ref, mask_in=mask_crds)
        tors_mask *= mask_BB[...,None]

        # get alternative coordinates for ground-truth
        true_alt = torch.zeros_like(true)
        true_alt.scatter_(2, self.l2a[seq,:,None].repeat(1,1,1,3), true)
        natRs_all, _n0 = self.compute_allatom_coords(seq, true[...,:3,:], true_tors)
        natRs_all_alt, _n1 = self.compute_allatom_coords(seq, true_alt[...,:3,:], true_tors_alt)
        predTs = pred[-1,...]
        predRs_all, pred_all = self.compute_allatom_coords(seq, predTs, pred_tors[-1]) 

        #  - resolve symmetry
        xs_mask = self.aamask[seq] # (B, L, 27)
        xs_mask[0,:,14:]=False # (ignore hydrogens except lj loss)
        xs_mask *= mask_crds # mask missing atoms & residues as well
        natRs_all_symm, nat_symm = resolve_symmetry(pred_allatom[-1], natRs_all[0], true[0], natRs_all_alt[0], true_alt[0], xs_mask[0])

        # torsion angle loss
        l_tors = torsionAngleLoss(
            pred_tors,
            true_tors,
            true_tors_alt,
            tors_mask,
            tors_planar,
            eps = 1e-10)
        tot_loss += w_str*l_tors
        loss_s.append(l_tors[None].detach())

        ### FINETUNING LAYERS
        # lddts (CA)
        ca_lddt = calc_lddt(pred[:,:,:,1].detach(), true[:,:,1], mask_BB, mask_2d, same_chain, negative=negative, interface=interface)
        loss_s.append(ca_lddt.detach())

        loss_dict['ca_lddt'] = float(ca_lddt[-1].detach())

        # lddts (allatom) + lddt loss
        lddt_loss, allatom_lddt = calc_allatom_lddt_loss(
            pred_allatom.detach(), nat_symm, pred_lddt, idx, mask_crds, mask_2d, same_chain, negative=negative, interface=interface)
        tot_loss += w_lddt*lddt_loss
        loss_s.append(lddt_loss.detach()[None])
        loss_s.append(allatom_lddt.detach())

        loss_dict['lddt_loss'] = float(lddt_loss.detach())
        loss_dict['allatom_lddt'] = float(allatom_lddt.detach())

        # FAPE losses
        # allatom fape and torsion angle loss
        # frames, frame_mask = get_frames(
        #     pred_allatom[-1,None,...], mask_crds, seq, self.fi_dev, atom_frames)
        if negative: # inter-chain fapes should be ignored for negative cases
            # L1 = same_chain[0,0,:].sum()
            # mask_BBA = mask_BB.clone()
            # mask_BBA[0, L1:] = False
            l_fape_A = compute_general_FAPE(
                pred_allatom[:,mask_BBA[0],:,:3],
                nat_symm[None,mask_BBA[0],:,:3],
                xs_mask[:,mask_BBA[0]],
                frames[:,mask_BBA[0]],
                frame_mask[:,mask_BBA[0]]
            )
            # mask_BBB = mask_BB.clone()
            # mask_BBB[0,:L1] = False
            l_fape_B = compute_general_FAPE(
                pred_allatom[:,mask_BBB[0],:,:3],
                nat_symm[None,mask_BBB[0],:,:3],
                xs_mask[:,mask_BBB[0]],
                frames[:,mask_BBB[0]],
                frame_mask[:,mask_BBB[0]]
            )
            fracA = float(L1)/len(same_chain[0,0])
            l_fape = fracA*l_fape_A + (1.0-fracA)*l_fape_B

        else:
            l_fape = compute_general_FAPE(
                pred_allatom[:,mask_BB[0],:,:3],
                nat_symm[None,mask_BB[0],:,:3],
                xs_mask[:,mask_BB[0]],
                frames[:,mask_BB[0]],
                frame_mask[:,mask_BB[0]]
            )
        loss_s.append(l_fape.detach())
        tot_loss += w_str*l_fape[0]
        loss_dict['fape'] = float(l_fape[0].detach())
        
        # cart bonded (bond geometry)
        bond_loss = calc_BB_bond_geom(seq[0], pred_allatom[0:1], idx)
        if w_bond > 0.0:
            tot_loss += w_bond*bond_loss
        loss_s.append( bond_loss[None].detach() )

        if (pred_allatom.shape[0] > 1):
            bond_loss = calc_cart_bonded(seq, pred_allatom[1:], idx, self.cb_len, self.cb_ang, self.cb_tor)
            if w_bond > 0.0:
                tot_loss += w_bond*bond_loss.mean()
            loss_s.append( bond_loss.detach() )

        loss_dict['bond_loss'] = float(bond_loss.detach())

        # clash [use all atoms not just those in native]
        clash_loss = calc_lj(
            seq[0], pred_allatom, 
            self.aamask, self.ljlk_parameters, self.lj_correction_parameters, self.num_bonds,
            lj_lin=lj_lin
        )
        if w_clash > 0.0:
            tot_loss += w_clash*clash_loss.mean()
        loss_s.append( clash_loss.detach() )

        loss_dict['clash_loss'] = float(clash_loss.detach())

        L0 = same_chain[0,0,:].sum()
        chain1 = torch.zeros_like(same_chain, dtype=bool)
        chain1[:,:L0,:L0] = True
        _, allatom_lddt_c1 = calc_allatom_lddt_loss(
            pred_allatom.detach(), nat_symm, pred_lddt, idx, mask_crds, mask_2d, chain1, negative=True)
        loss_s.append(allatom_lddt_c1.detach())

        loss_dict['allatom_lddt_c1'] = float(allatom_lddt_c1.detach())

        chain2 = torch.zeros_like(same_chain, dtype=bool)
        chain2[:,L0:,L0:] = True
        _, allatom_lddt_c2 = calc_allatom_lddt_loss(
            pred_allatom.detach(), nat_symm, pred_lddt, idx, mask_crds, mask_2d, chain2, negative=True, bin_scaling=0.5)
        loss_s.append(allatom_lddt_c2.detach())

        loss_dict['allatom_lddt_c2'] = float(allatom_lddt_c2.detach())

        _, allatom_lddt_inter = calc_allatom_lddt_loss(
            pred_allatom.detach(), nat_symm, pred_lddt, idx, mask_crds, mask_2d, same_chain, interface=True)
        loss_s.append(allatom_lddt_inter.detach())

        loss_dict['allatom_lddt_inter'] = float(allatom_lddt_inter.detach())
        if float(allatom_lddt_c2.detach()) > .3:
            verbose=True
        # hbond [use all atoms not just those in native]
        #hb_loss = calc_hb(
        #    seq[0], pred_all[0,...,:3], 
        #    self.aamask, self.hbtypes, self.hbbaseatoms, self.hbpolys, 
        #    normalize=(not verbose)
        #)
        #if w_hb > 0.0:
        #    tot_loss += w_hb*hb_loss
        #loss_s.append(torch.stack((hb_loss, clash_loss, bond_loss)).detach())

        if (verbose):
            print (
                ctr,
                tot_str.cpu().detach().numpy(),
                allatom_lddt.cpu().detach().numpy(),
                allatom_lddt_c2.cpu().detach().numpy(),
                l_fape.cpu().detach().numpy(),
                l_fape_B.cpu().detach().numpy(),
                mask_BB[0].sum()
            )
            outdir = self.outdir if self.outdir else './'
            writepdb(outdir+"p_"+self.model_name+"_"+str(ctr)+".pdb", pred_all[-1,mask_BB[0]][:,:23], seq[mask_BB][:])
            writepdb(outdir+"n_"+str(ctr)+".pdb", true[mask_BB][:,:23], seq[mask_BB][:])
            writepdb(outdir+"nre_"+str(ctr)+".pdb", _n0[mask_BB], seq[mask_BB][:])

        loss_dict['total_loss'] = float(tot_loss.detach())

        return tot_loss, torch.cat(loss_s, dim=0), loss_dict


    def calc_acc(self, prob, dist, idx_pdb, mask_2d, return_cnt=False):
        B = idx_pdb.shape[0]
        L = idx_pdb.shape[1] # (B, L)
        seqsep = torch.abs(idx_pdb[:,:,None] - idx_pdb[:,None,:]) + 1
        mask = seqsep > 24
        mask = torch.triu(mask.float())
        mask *= mask_2d
        #
        cnt_ref = dist < 20
        cnt_ref = cnt_ref.float() * mask
        #
        cnt_pred = prob[:,:20,:,:].sum(dim=1) * mask
        #
        top_pred = torch.topk(cnt_pred.view(B,-1), L)
        kth = top_pred.values.min(dim=-1).values
        tmp_pred = list()
        for i_batch in range(B):
            tmp_pred.append(cnt_pred[i_batch] > kth[i_batch])
        tmp_pred = torch.stack(tmp_pred, dim=0)
        tmp_pred = tmp_pred.float()*mask
        #
        condition = torch.logical_and(tmp_pred==cnt_ref, cnt_ref==torch.ones_like(cnt_ref))
        n_good = condition.float().sum()
        n_total = (cnt_ref == torch.ones_like(cnt_ref)).float().sum() + 1e-9
        n_total_pred = (tmp_pred == torch.ones_like(tmp_pred)).float().sum() + 1e-9
        prec = n_good / n_total_pred
        recall = n_good / n_total
        F1 = 2.0*prec*recall / (prec+recall+1e-9)
        if return_cnt:
            return torch.stack([prec, recall, F1]), cnt_pred, cnt_ref

        return torch.stack([prec, recall, F1])

    def load_model(self, model, optimizer, scheduler, scaler, model_name, rank, suffix='last', resume_train=False):
        chk_fn = "models/%s_%s.pt"%(model_name, suffix)
        loaded_epoch = -1
        best_valid_loss = 999999.9
        if not os.path.exists(chk_fn):
            print ('no model found', model_name)
            return -1, best_valid_loss
        print ('loading model', model_name)
        map_location = {"cuda:%d"%0: "cuda:%d"%rank}
        checkpoint = torch.load(chk_fn, map_location=map_location)
        rename_model = False
        new_chk = {}
        msd_src = checkpoint['model_state_dict']
        msd_tgt = model.module.model.state_dict()
        for param in msd_tgt:

            if param not in msd_src:
                print ('missing',param)
                rename_model=True
                #break
            elif (msd_tgt[param].shape == msd_src[param].shape):
                new_chk[param] = msd_src[param]
            else:
                # fd hack for new encoding
                if (msd_src[param].shape[0]==30 and msd_tgt[param].shape[0]==32 and 'compute_allatom_coords' not in param):
                    print ('Fixing',param)
                    new_chk[param] = torch.zeros_like(msd_tgt[param])
                    new_chk[param][:26] =  msd_src[param][:26]
                    new_chk[param][27:31] =  msd_src[param][26:30]

                else:
                    #wrong size latent_emb.emb.weight torch.Size([256, 64]) torch.Size([256, 68])
                    #wrong size templ_emb.emb.weight torch.Size([64, 104]) torch.Size([64, 108])
                    #wrong size full_emb.emb.weight torch.Size([64, 33]) torch.Size([64, 35])

                    print (
                        'wrong size',param,
                        checkpoint['model_state_dict'][param].shape,
                         model.module.model.state_dict()[param].shape )
                    rename_model=True

        #new_chk = checkpoint['model_state_dict']
        model.module.model.load_state_dict(new_chk, strict=False)
        model.module.shadow.load_state_dict(new_chk, strict=False)
        if resume_train and (not rename_model):
            print (' ... loading optimization params')
            loaded_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                print (' ... loading scheduler params')
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            else:
                scheduler.last_epoch = loaded_epoch + 1
            #if 'best_loss' in checkpoint:
            #    best_valid_loss = checkpoint['best_loss']
        return loaded_epoch, best_valid_loss

    def checkpoint_fn(self, model_name, description):
        if not os.path.exists("models"):
            os.mkdir("models")
        name = "%s_%s.pt"%(model_name, description)
        return os.path.join("models", name)
    
    # main entry function of training
    # 1) make sure ddp env vars set
    # 2) figure out if we launched using slurm or interactively
    #   - if slurm, assume 1 job launched per GPU
    #   - if interactive, launch one job for each GPU on node
    def run_model_training(self, world_size):
        if ('MASTER_ADDR' not in os.environ):
            os.environ['MASTER_ADDR'] = '127.0.0.1' # multinode requires this set in submit script
        if ('MASTER_PORT' not in os.environ):
            os.environ['MASTER_PORT'] = '%d'%self.port

        if (not self.interactive and "SLURM_NTASKS" in os.environ and "SLURM_PROCID" in os.environ):
            world_size = int(os.environ["SLURM_NTASKS"])
            rank = int (os.environ["SLURM_PROCID"])
            print ("Launched from slurm", rank, world_size)
            self.train_model(rank, world_size)
        else:
            print ("Launched from interactive")
            world_size = torch.cuda.device_count()
            mp.spawn(self.train_model, args=(world_size,), nprocs=world_size, join=True)

    def train_model(self, rank, world_size):
        
        # save git diff from last commit
        if self.outdir is not None:
            gitdiff_fn = open(f'{self.outdir}/git_diff.txt','w')
            git_diff = subprocess.Popen(['git diff'], cwd = os.getcwd(), shell = True, stdout = gitdiff_fn, stderr = subprocess.PIPE)
            print('Save git diff between current state and last commit')

        # wandb logging
        if self.wandb_prefix is not None and rank == 0:
            print('initializing wandb')
            #wandb.require("service")
            wandb.init(
                project='RF2_allatom',
                entity='bakerlab',
                name=self.wandb_prefix
            )
            all_param = {}
            all_param.update(self.loader_param)
            all_param.update(self.model_param)
            all_param.update(self.loss_param)

            wandb.config = all_param
            wandb.save(os.path.join(os.getcwd(), self.outdir, 'git_diff.txt'))

        #print ("running ddp on rank %d, world_size %d"%(rank, world_size))
        gpu = rank % torch.cuda.device_count()
        dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
        torch.cuda.set_device("cuda:%d"%gpu)

        #define dataset & data loader
        (
            pdb_items, fb_items, compl_items, neg_items, na_compl_items, na_neg_items, rna_items,
            sm_compl_items, valid_pdb, valid_homo, valid_compl, valid_neg, valid_na_compl, 
            valid_na_neg, valid_rna, valid_sm_compl, homo
        ) = get_train_valid_set(self.loader_param)

        pdb_IDs, pdb_weights, pdb_dict = pdb_items
        fb_IDs, fb_weights, fb_dict = fb_items
        compl_IDs, compl_weights, compl_dict = compl_items
        neg_IDs, neg_weights, neg_dict = neg_items
        na_compl_IDs, na_compl_weights, na_compl_dict = na_compl_items
        na_neg_IDs, na_neg_weights, na_neg_dict = na_neg_items
        rna_IDs, rna_weights, rna_dict = rna_items
        sm_compl_IDs, sm_compl_weights, sm_compl_dict = sm_compl_items

        self.n_train = N_EXAMPLE_PER_EPOCH
        self.n_valid_pdb = len(valid_pdb.keys())
        #self.n_valid_pdb = (self.n_valid_pdb // world_size)*world_size
        self.n_valid_homo = len(valid_homo.keys())
        #self.n_valid_homo = (self.n_valid_homo // world_size)*world_size
        self.n_valid_compl = len(valid_compl.keys())
        #self.n_valid_compl = (self.n_valid_compl // world_size)*world_size
        self.n_valid_neg = len(valid_neg.keys())
        #self.n_valid_neg = (self.n_valid_neg // world_size)*world_size
        self.n_valid_na_compl = len(valid_na_compl.keys())
        #self.n_valid_na_compl = (self.n_valid_na_compl // world_size)*world_size
        self.n_valid_na_neg = len(valid_na_neg.keys())
        #self.n_valid_na_neg = (self.n_valid_na_neg // world_size)*world_size
        self.n_valid_rna = len(valid_rna.keys())
        #self.n_valid_rna = (self.n_valid_rna // world_size)*world_size
        self.n_valid_rna = len(valid_rna.keys())
        self.n_valid_sm_compl = len(valid_sm_compl.keys())

        self.n_valid_pdb = 4
        #self.n_valid_homo = 4
        #self.n_valid_compl = 4
        #self.n_valid_neg = 4
        #self.n_valid_na_compl = 4
        #self.n_valid_na_neg = 4
        #self.n_valid_rna = 4
        #self.n_valid_sm_compl = 200

        if (rank==0):
            print ('Loaded (training)',
                len(pdb_IDs),'monomers/homomers,',
                len(fb_IDs),'distilled monomers,',
                len(compl_IDs),'heteromers,',
                len(neg_IDs),'negative heteromers,',
                len(na_compl_IDs),'nucleic-acid complexes,',
                len(na_neg_IDs),'negative nucleic-acid complexes,',
                len(rna_IDs),'RNA structures,  and',
                len(sm_compl_IDs), 'small molecule complexes'
            )
            print ('Loaded (valid)',
                len(valid_pdb.keys()),'monomers,',
                len(valid_homo.keys()),'homomers,',
                len(valid_compl.keys()),'heteromers,',
                len(valid_neg.keys()),'negative heteromers,',
                len(valid_na_compl.keys()),'nucleic-acid complexes,',
                len(valid_na_neg.keys()),'negative nucleic-acid complexes,',
                len(valid_rna),'RNA structures, and',
                len(valid_sm_compl), 'small molecule complexes'
            )
            print ('Using',
                self.n_valid_pdb,'monomers,',
                self.n_valid_homo,'homomers,',
                self.n_valid_compl,'heteromers,',
                self.n_valid_neg,'negative heteromers',
                self.n_valid_na_compl,'nucleic-acid complexes,',
                self.n_valid_na_neg,'negative nucleic-acid complexes,',
                self.n_valid_rna,'RNA structures,  and',
                self.n_valid_sm_compl, 'small molecule complexes'
            )

        train_set = DistilledDataset(
            pdb_IDs, loader_atomize_pdb, pdb_dict,
            compl_IDs, loader_complex, compl_dict,
            neg_IDs, loader_complex, neg_dict,
            na_compl_IDs, loader_na_complex, na_compl_dict,
            na_neg_IDs, loader_na_complex, na_neg_dict,
            fb_IDs, loader_fb, fb_dict,
            rna_IDs, loader_rna, rna_dict,
            sm_compl_IDs, loader_sm_compl, sm_compl_dict,
            sm_compl_IDs, loader_small_molecule, sm_compl_dict,
            homo, 
            self.loader_param,
            native_NA_frac=0.25
        )

        valid_pdb_set = Dataset(
            list(valid_pdb.keys())[:self.n_valid_pdb],
            loader_pdb, valid_pdb,
            self.loader_param, homo, p_homo_cut=-1.0
        )
        #valid_atomize_pdb_set = Dataset(
        #    list(valid_pdb.keys())[:self.n_valid_pdb],
        #    loader_atomize_pdb, valid_pdb,
        #    self.loader_param, homo, p_homo_cut=-1.0)
        # valid_homo_set = Dataset(
        #     list(valid_homo.keys())[:self.n_valid_homo],
        #     loader_pdb, valid_homo,
        #     self.loader_param, homo, p_homo_cut=2.0
        # )
        # valid_compl_set = DatasetComplex(
        #     list(valid_compl.keys())[:self.n_valid_compl],
        #     loader_complex, valid_compl,
        #     self.loader_param, negative=False
        # )
        # valid_neg_set = DatasetComplex(
        #     list(valid_neg.keys())[:self.n_valid_neg],
        #     loader_complex, valid_neg,
        #     self.loader_param, negative=True
        # )
#        valid_na_compl_set = DatasetNAComplex(
#            list(valid_na_compl.keys())[:self.n_valid_na_compl],
#            loader_na_complex, valid_na_compl,
#            self.loader_param, negative=False, native_NA_frac=1.0
#        )
#        valid_na_neg_set = DatasetNAComplex(
#            list(valid_na_neg.keys())[:self.n_valid_na_neg],
#            loader_na_complex, valid_na_neg,
#            self.loader_param, negative=True, native_NA_frac=1.0
#        )
#        valid_na_from_scratch_compl_set = DatasetNAComplex(
#            list(valid_na_compl.keys())[:self.n_valid_na_compl],
#            loader_na_complex, valid_na_compl,
#            self.loader_param, negative=False, native_NA_frac=0.0
#        )
#        valid_na_from_scratch_neg_set = DatasetNAComplex(
#            list(valid_na_neg.keys())[:self.n_valid_na_neg],
#            loader_na_complex, valid_na_neg,
#            self.loader_param, negative=True, native_NA_frac=0.0
#        )
#        valid_rna_set = DatasetRNA(
#            list(valid_rna.keys())[:self.n_valid_rna],
#            loader_rna, valid_rna,
#            self.loader_param
#        )
        valid_sm_compl_set = DatasetSMComplex(
            list(valid_sm_compl.keys())[:self.n_valid_sm_compl],
            loader_sm_compl, valid_sm_compl,
            self.loader_param, p_ligand_dock=0.0,
        )
        valid_sm_compl_rigid_body_set = DatasetSMComplex(
            list(valid_sm_compl.keys())[:self.n_valid_sm_compl],
            loader_sm_compl, valid_sm_compl,
            self.loader_param, p_ligand_dock=1.0)
        valid_sm_set = DatasetSM(
            list(valid_sm_compl.keys())[:self.n_valid_sm_compl],
            loader_small_molecule, valid_sm_compl, self.loader_param
        )
        
        train_sampler = DistributedWeightedSampler(
            train_set, 
            pdb_weights,
            fb_weights,
            compl_weights,
            neg_weights,
            na_compl_weights,
            na_neg_weights,
            rna_weights,
            sm_compl_weights,
            sm_compl_weights,
            num_example_per_epoch=N_EXAMPLE_PER_EPOCH,
            num_replicas=world_size, 
            rank=rank, 
            fraction_fb=0.0,
            fraction_compl=0.0,
            fraction_na_compl=0.0,
            fraction_rna=0.0,
            fraction_sm_compl=0,
            fraction_sm = 0, 
            replacement=True
        )

        valid_pdb_sampler = data.distributed.DistributedSampler(valid_pdb_set, num_replicas=world_size, rank=rank)
        # valid_homo_sampler = data.distributed.DistributedSampler(valid_homo_set, num_replicas=world_size, rank=rank)
        # valid_compl_sampler = data.distributed.DistributedSampler(valid_compl_set, num_replicas=world_size, rank=rank)
        # valid_neg_sampler = data.distributed.DistributedSampler(valid_neg_set, num_replicas=world_size, rank=rank)
#        valid_na_compl_sampler = data.distributed.DistributedSampler(valid_na_compl_set, num_replicas=world_size, rank=rank)
#        valid_na_neg_sampler = data.distributed.DistributedSampler(valid_na_neg_set, num_replicas=world_size, rank=rank)
#        valid_na_from_scratch_compl_sampler = data.distributed.DistributedSampler(valid_na_from_scratch_compl_set, num_replicas=world_size, rank=rank)
#        valid_na_from_scratch_neg_sampler = data.distributed.DistributedSampler(valid_na_from_scratch_neg_set, num_replicas=world_size, rank=rank)
#        valid_rna_sampler = data.distributed.DistributedSampler(valid_rna_set, num_replicas=world_size, rank=rank)

        valid_sm_compl_sampler = data.distributed.DistributedSampler(valid_sm_compl_set, num_replicas=world_size, rank=rank)
        valid_sm_sampler = data.distributed.DistributedSampler(valid_sm_set, num_replicas=world_size, rank=rank)
        valid_sm_compl_rigid_body_sampler = data.distributed.DistributedSampler(valid_sm_compl_rigid_body_set, num_replicas=world_size, rank=rank)

        train_loader = data.DataLoader(train_set, sampler=train_sampler, batch_size=self.batch_size, **LOAD_PARAM)
        valid_pdb_loader = data.DataLoader(valid_pdb_set, sampler=valid_pdb_sampler, **LOAD_PARAM)
        
        #valid_atomize_pdb_loader = data.DataLoader(valid_atomize_pdb_set, sampler=valid_pdb_sampler, **LOAD_PARAM)
        # valid_homo_loader = data.DataLoader(valid_homo_set, sampler=valid_homo_sampler, **LOAD_PARAM)
        # valid_compl_loader = data.DataLoader(valid_compl_set, sampler=valid_compl_sampler, **LOAD_PARAM)
        # valid_neg_loader = data.DataLoader(valid_neg_set, sampler=valid_neg_sampler, **LOAD_PARAM)
#        valid_na_compl_loader = data.DataLoader(valid_na_compl_set, sampler=valid_na_compl_sampler, **LOAD_PARAM)
#        valid_na_neg_loader = data.DataLoader(valid_na_neg_set, sampler=valid_na_neg_sampler, **LOAD_PARAM)
#        valid_na_from_scratch_compl_loader = data.DataLoader(valid_na_from_scratch_compl_set, sampler=valid_na_from_scratch_compl_sampler, **LOAD_PARAM)
#       valid_na_from_scratch_neg_loader = data.DataLoader(valid_na_from_scratch_neg_set, sampler=valid_na_from_scratch_neg_sampler, **LOAD_PARAM)
#        valid_rna_loader = data.DataLoader(valid_rna_set, sampler=valid_rna_sampler, **LOAD_PARAM)

        valid_sm_compl_loader = data.DataLoader(valid_sm_compl_set, sampler=valid_sm_compl_sampler, **LOAD_PARAM)
        valid_sm_compl_rigid_body_loader = data.DataLoader(valid_sm_compl_rigid_body_set, sampler=valid_sm_compl_sampler, **LOAD_PARAM)
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

        #for n,p in model.named_parameters():
        #    if ("finetune_refiner" not in n and "residue_embed" not in n and "allatom_embed" not in n):
        #        p.requires_grad_(False)

        ddp_model = DDP(model, device_ids=[gpu], find_unused_parameters=False)
        if rank == 0:
            print ("# of parameters:", count_parameters(ddp_model))

        # define optimizer and scheduler
        opt_params = add_weight_decay(ddp_model, self.l2_coeff)
        optimizer = torch.optim.AdamW(opt_params, lr=self.init_lr)
        #scheduler = get_stepwise_decay_schedule_with_warmup(optimizer, 1000, 5000, 0.95)
        scheduler = get_stepwise_decay_schedule_with_warmup(optimizer, 0, 5000, 0.95)
        scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
       
        # load model
        loaded_epoch, best_valid_loss = self.load_model(ddp_model, optimizer, scheduler, scaler, 
                                                        self.model_name, gpu, suffix="best", resume_train=True)

        if (self.eval):
#            _, _, _ = self.valid_pdb_cycle(ddp_model, valid_atomize_pdb_loader, rank, gpu, world_size, 0, verbose=True) # for debugging
            # run protein/NA prediction (TEMPLATED)
            #_, _, _ = self.valid_ppi_cycle(
            #    ddp_model, valid_na_compl_loader, valid_na_neg_loader, 
            #    rank, gpu, world_size, 0, header="NA", report_interface=False, verbose=True)

            # run protein/NA prediction (NON-TEMPLATED)
            #_, _, _ = self.valid_ppi_cycle(
            #    ddp_model, valid_na_from_scratch_compl_loader, valid_na_from_scratch_neg_loader, 
            #    rank, gpu, world_size, 0, header="NA", report_interface=False, verbose=True)

            # run RNA prediction
            #_,_,_ = self.valid_pdb_cycle(ddp_model, valid_rna_loader, rank, gpu, world_size, 0, verbose=True)
            #_, _, _ = self.valid_pdb_cycle(ddp_model, valid_sm_compl_loader, rank, gpu, world_size, 0, verbose=True)
            _, _, _ = self.valid_pdb_cycle(ddp_model, valid_sm_compl_rigid_body_loader, rank, gpu, world_size, epoch=0, verbose=True, header='SM Rigid Body')
            dist.destroy_process_group()
            return

        if loaded_epoch >= self.n_epoch:
            DDP_cleanup()
            return

        #_, _, _ = self.valid_pdb_cycle(ddp_model, valid_homo_loader, rank, gpu, world_size, epoch, header="Homo")
        #_, _, _ = self.valid_ppi_cycle(ddp_model, valid_compl_loader, valid_neg_loader, rank, gpu, world_size, epoch, report_interface=True)
        #_, _, _ = self.valid_ppi_cycle(
        #    ddp_model, valid_na_compl_loader, valid_na_neg_loader, 
        #    rank, gpu, world_size, epoch, header="NA", report_interface=False)
        #_, _, _ = self.valid_ppi_cycle(
        #    ddp_model, valid_na_from_scratch_compl_loader, valid_na_from_scratch_neg_loader, 
        #    rank, gpu, world_size, epoch, header="NAfs", report_interface=False)
        #_,_,_ = self.valid_pdb_cycle(ddp_model, valid_rna_loader, rank, gpu, world_size, epoch, header="RNA")

        for epoch in range(loaded_epoch+1, self.n_epoch):
            train_sampler.set_epoch(epoch)
            #valid_pdb_sampler.set_epoch(epoch)
            #valid_homo_sampler.set_epoch(epoch)
            #valid_compl_sampler.set_epoch(epoch)
            #valid_neg_sampler.set_epoch(epoch)
            #valid_sm_compl_sampler.set_epoch(epoch)
            #valid_sm_sampler.set_epoch(epoch)
            #valid_sm_compl_rigid_body_sampler.set_epoch(epoch)

            train_tot, train_loss, train_acc = self.train_cycle(ddp_model, train_loader, optimizer, scheduler, scaler, rank, gpu, world_size, epoch)

            valid_tot, valid_loss, valid_acc = self.valid_pdb_cycle(ddp_model, valid_pdb_loader, rank, gpu, world_size, epoch)
            #_, _, _ = self.valid_pdb_cycle(ddp_model, valid_atomize_pdb_loader, rank, gpu, world_size, epoch, header="Atomize PDB")
            #_, _, _ = self.valid_pdb_cycle(ddp_model, valid_homo_loader, rank, gpu, world_size, epoch, header="Homo")
            #_, _, _ = self.valid_ppi_cycle(ddp_model, valid_compl_loader, valid_neg_loader, rank, gpu, world_size, epoch, report_interface=True)
#            _, _, _ = self.valid_ppi_cycle(
#                ddp_model, valid_na_compl_loader, valid_na_neg_loader, 
#                rank, gpu, world_size, epoch, header="NA", report_interface=False)
#            _, _, _ = self.valid_ppi_cycle(
#                ddp_model, valid_na_from_scratch_compl_loader, valid_na_from_scratch_neg_loader, 
#                rank, gpu, world_size, epoch, header="NAfs", report_interface=False)
#            _,_,_ = self.valid_pdb_cycle(ddp_model, valid_rna_loader, rank, gpu, world_size, epoch, header="RNA")

            valid_tot, valid_loss, valid_acc = self.valid_pdb_cycle(ddp_model, valid_sm_compl_loader, rank, gpu, world_size, epoch, header="SM Compl") 
            # _, _, _ = self.valid_pdb_cycle(ddp_model, valid_sm_compl_rigid_body_loader, rank, gpu, world_size, epoch, header="SM Rigid Body") 
            # _, _, _ = self.valid_pdb_cycle(ddp_model, valid_sm_loader, rank, gpu, world_size, epoch, header="SM Only") 

            if rank == 0: # save model
                if valid_tot < best_valid_loss:
                    best_valid_loss = valid_tot
                    torch.save({'epoch': epoch,
                                #'model_state_dict': ddp_model.state_dict(),
                                'model_state_dict': ddp_model.module.shadow.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                    'scheduler_state_dict': scheduler.state_dict(),
                                    'scaler_state_dict': scaler.state_dict(),
                                    'best_loss': best_valid_loss,
                                    'train_loss': train_loss,
                                    'train_acc': train_acc,
                                    'valid_loss': valid_loss,
                                    'valid_acc': valid_acc},
                                    self.checkpoint_fn(self.model_name, 'best'))
                    wandb.save(self.checkpoint_fn(self.model_name, 'best'))

                
                torch.save({'epoch': epoch,
                            #'model_state_dict': ddp_model.state_dict(),
                            'model_state_dict': ddp_model.module.shadow.state_dict(),
                            'final_state_dict': ddp_model.module.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'scaler_state_dict': scaler.state_dict(),
                            'train_loss': train_loss,
                            'train_acc': train_acc,
                            'valid_loss': valid_loss,
                            'valid_acc': valid_acc,
                            'best_loss': best_valid_loss},
                            self.checkpoint_fn(self.model_name, str(epoch)))
                wandb.save(self.checkpoint_fn(self.model_name, str(epoch)))
        dist.destroy_process_group()

    def train_cycle(self, ddp_model, train_loader, optimizer, scheduler, scaler, rank, gpu, world_size, epoch, verbose=False):
        # Turn on training mode
        ddp_model.train()
        
        # clear gradients
        optimizer.zero_grad()

        start_time = time.time()
        
        # For intermediate logs
        local_tot = 0.0
        local_loss = None
        local_acc = None
        train_tot = 0.0
        train_loss = None
        train_acc = None

        counter = 0
        
        for seq, msa, msa_masked, msa_full, mask_msa, true_crds, atom_mask, idx_pdb, xyz_t, t1d, xyz_prev, same_chain, unclamp, negative, atom_frames, bond_feats, task, item in train_loader:
            
            # skip known bad training examples
            # loader will print warning message with item info for followup later
            if torch.is_tensor(item) and len(item.shape) == 2 and torch.all(item==-1):
                continue

            save_pdbs = np.random.rand()<0.01

            # transfer inputs to device
            B, _, N, L = msa.shape
            idx_pdb = idx_pdb.to(gpu, non_blocking=True) # (B, L)
            true_crds = true_crds.to(gpu, non_blocking=True) # (B, N?, L, Natms, 3)
            atom_mask = atom_mask.to(gpu, non_blocking=True) # (B, L, Natms)
            same_chain = same_chain.to(gpu, non_blocking=True) # (B, L, L)

            xyz_t = xyz_t.to(gpu, non_blocking=True)
            t1d = t1d.to(gpu, non_blocking=True)

            seq = seq.to(gpu, non_blocking=True)
            msa = msa.to(gpu, non_blocking=True)
            msa_masked = msa_masked.to(gpu, non_blocking=True)
            msa_full = msa_full.to(gpu, non_blocking=True)
            mask_msa = mask_msa.to(gpu, non_blocking=True)
            atom_frames = atom_frames.to(gpu, non_blocking=True)
            bond_feats = bond_feats.to(gpu, non_blocking=True)

            # processing template features
            seq_unmasked = msa[:, 0, 0, :] # (B, L)
            xyz_t_frames = xyz_t_to_frame_xyz(xyz_t, seq_unmasked, atom_frames)
            t2d = xyz_to_t2d(xyz_t_frames)

            xyz_t = get_init_xyz(seq[:,0],xyz_t,same_chain)
            xyz_prev = get_init_xyz(seq[:,0],xyz_prev[:,None],same_chain).reshape(B, L, NTOTAL, 3)

            # get torsion angles from templates
            seq_tmpl = t1d[...,:-1].argmax(dim=-1).reshape(-1,L)
            alpha, _, alpha_mask, _ = get_torsions(
                xyz_t.reshape(-1,L,NTOTAL,3), seq_tmpl, self.ti_dev, self.ti_flip, self.ang_ref)
            alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))
            alpha[torch.isnan(alpha)] = 0.0
            alpha = alpha.reshape(B,-1,L,NTOTALDOFS,2)
            alpha_mask = alpha_mask.reshape(B,-1,L,NTOTALDOFS,1)
            alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(B, -1, L, 3*NTOTALDOFS)

            counter += 1

            #if save_pdbs:
               # res_mask = ~((atom_mask[0,0,:,:3].sum(dim=-1) < 3.0) * ~(is_atom(msa[:,i_cycle,0])))
               # writepdb(self.outdir+f'ep{epoch}_{counter}_{item[0][0]}_xyz_prev.pdb', 
               #     torch.nan_to_num(xyz_prev[res_mask][:,:23]), seq_unmasked[res_mask])

            N_cycle = np.random.randint(1, self.maxcycle+1) # number of recycling

            msa_prev = None
            pair_prev = None
            alpha_prev = torch.zeros((B,L,NTOTALDOFS,2)).to(gpu, non_blocking=True)
            state_prev = None

            try:
                with torch.no_grad():
                    for i_cycle in range(N_cycle-1):
                        with ddp_model.no_sync():
                            with torch.cuda.amp.autocast(enabled=USE_AMP):
                                msa_prev, pair_prev, xyz_prev, state_prev, alpha = ddp_model(
                                    msa_masked[:,i_cycle],
                                    msa_full[:,i_cycle],
                                    seq[:,i_cycle],
                                    msa[:,i_cycle,0], # unmasked seq
                                    xyz_prev,
                                    alpha_prev,
                                    idx_pdb,
                                    bond_feats,
                                    t1d=t1d,
                                    t2d=t2d,
                                    xyz_t=xyz_t,
                                    alpha_t=alpha_t,
                                    msa_prev=msa_prev,
                                    pair_prev=pair_prev,
                                    state_prev=state_prev,
                                    return_raw=True,
                                    use_checkpoint=False
                                )

                i_cycle = N_cycle-1

                if counter%self.ACCUM_STEP != 0:
                    with ddp_model.no_sync():
                        with torch.cuda.amp.autocast(enabled=USE_AMP):
                            logit_s, logit_aa_s, pred_crds, alphas, pred_allatom, pred_lddts, _, _, _ = ddp_model(
                                msa_masked[:,i_cycle],
                                msa_full[:,i_cycle],
                                seq[:,i_cycle],
                                msa[:,i_cycle,0], # unmasked seq
                                xyz_prev,
                                alpha_prev,
                                idx_pdb,
                                bond_feats,
                                t1d=t1d,
                                t2d=t2d,
                                xyz_t=xyz_t,
                                alpha_t=alpha_t,
                                msa_prev=msa_prev,
                                pair_prev=pair_prev,
                                state_prev=state_prev,
                                use_checkpoint=True
                            )

                            true_crds, atom_mask = resolve_equiv_natives(pred_crds[-1], true_crds, atom_mask)
                            res_mask = ~((atom_mask[:,:,:3].sum(dim=-1) < 3.0) * ~(is_atom(msa[:,i_cycle,0])))
                            mask_2d = res_mask[:,None,:] * res_mask[:,:,None]

                            true_crds_frame = xyz_to_frame_xyz(true_crds, msa[:, i_cycle, 0],atom_frames)
                            c6d, _ = xyz_to_c6d(true_crds_frame)
                            c6d = c6d_to_bins(c6d, same_chain, negative=negative)

                            prob = self.active_fn(logit_s[0]) # distogram
                            acc_s = self.calc_acc(prob, c6d[...,0], idx_pdb, mask_2d)

                            ctrid = len(train_loader)*rank+counter
                            loss, loss_s, loss_dict = self.calc_loss(
                                logit_s, c6d,
                                logit_aa_s, msa[:, i_cycle], mask_msa[:,i_cycle],
                                pred_crds, alphas, pred_allatom, true_crds, 
                                atom_mask, res_mask, mask_2d, same_chain,
                                pred_lddts, idx_pdb, atom_frames=atom_frames,
                                unclamp=unclamp, negative=negative,
                                verbose=verbose, ctr=ctrid, **self.loss_param
                            )
                        loss = loss / self.ACCUM_STEP
                        scaler.scale(loss).backward()
                else:
                    with torch.cuda.amp.autocast(enabled=USE_AMP):
                        logit_s, logit_aa_s, pred_crds, alphas, pred_allatom, pred_lddts, _, _, _ = ddp_model(
                            msa_masked[:,i_cycle],
                            msa_full[:,i_cycle],
                            seq[:,i_cycle],
                            msa[:,i_cycle,0], # unmasked seq
                            xyz_prev,
                            alpha_prev,
                            idx_pdb,
                            bond_feats,
                            t1d=t1d,
                            t2d=t2d,
                            xyz_t=xyz_t,
                            alpha_t=alpha_t,
                            msa_prev=msa_prev,
                            pair_prev=pair_prev,
                            state_prev=state_prev,
                            use_checkpoint=True
                        )

                        true_crds, atom_mask = resolve_equiv_natives(pred_crds[-1], true_crds, atom_mask)

                        res_mask = ~((atom_mask[:,:,:3].sum(dim=-1) < 3.0) * ~(is_atom(msa[:,i_cycle,0])))
                        mask_2d = res_mask[:,None,:] * res_mask[:,:,None]

                        true_crds_frame = xyz_to_frame_xyz(true_crds, msa[:, i_cycle, 0],atom_frames)
                        c6d, _ = xyz_to_c6d(true_crds_frame)
                        c6d = c6d_to_bins(c6d, same_chain, negative=negative)

                        prob = self.active_fn(logit_s[0]) # distogram
                        acc_s = self.calc_acc(prob, c6d[...,0], idx_pdb, mask_2d)

                        ctrid = len(train_loader)*rank+counter
                        loss, loss_s, loss_dict = self.calc_loss(
                            logit_s, c6d,
                            logit_aa_s, msa[:, i_cycle], mask_msa[:,i_cycle],
                            pred_crds, alphas, pred_allatom, true_crds, 
                            atom_mask, res_mask, mask_2d, same_chain,
                            pred_lddts, idx_pdb, atom_frames=atom_frames, unclamp=unclamp, negative=negative,
                            verbose=verbose, ctr=ctrid, **self.loss_param
                        )
                    loss = loss / self.ACCUM_STEP
                    scaler.scale(loss).backward()
                    # gradient clipping
                    scaler.unscale_(optimizer)

                    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 0.2)

                    scaler.step(optimizer)
                    scale = scaler.get_scale()
                    scaler.update()
                    skip_lr_sched = (scale != scaler.get_scale())
                    optimizer.zero_grad()
                    if not skip_lr_sched:
                        scheduler.step()
                    ddp_model.module.update() # apply EMA
            except Exception as e:
                print('true_crds.shape',true_crds.shape)
                print('msa_masked.shape',msa_masked.shape)
                print('xyz_prev.shape',xyz_prev.shape)
                print('xyz_t.shape',xyz_t.shape)
                raise e
            
            if save_pdbs:
                #res_mask = ~((atom_mask[:,:,:3].sum(dim=-1) < 3.0) * ~(is_atom(msa[:,i_cycle,0])))
                writepdb(self.outdir+f'ep{epoch}_{counter}_{item[0][0]}_xyz_true.pdb', 
                    torch.nan_to_num(true_crds[res_mask][:,:23]), seq_unmasked[res_mask])
                writepdb(self.outdir+f'ep{epoch}_{counter}_{item[0][0]}_xyz_pred.pdb', 
                    torch.nan_to_num(pred_allatom[res_mask][:,:23]), seq_unmasked[res_mask])

            local_tot += loss.detach()*self.ACCUM_STEP
            if local_loss == None:
                local_loss = torch.zeros_like(loss_s.detach())
                local_acc = torch.zeros_like(acc_s.detach())
            local_loss += loss_s.detach()
            local_acc += acc_s.detach()
            
            train_tot += loss.detach()*self.ACCUM_STEP
            if train_loss == None:
                train_loss = torch.zeros_like(loss_s.detach())
                train_acc = torch.zeros_like(acc_s.detach())
            train_loss += loss_s.detach()
            train_acc += acc_s.detach()

            
            if counter % N_PRINT_TRAIN == 0:
                if rank == 0:
                    max_mem = torch.cuda.max_memory_allocated()/1e9
                    train_time = time.time() - start_time
                    local_tot /= float(N_PRINT_TRAIN)
                    local_loss /= float(N_PRINT_TRAIN)
                    local_acc /= float(N_PRINT_TRAIN)
                    
                    local_tot = local_tot.cpu().detach()
                    local_loss = local_loss.cpu().detach().numpy()
                    local_acc = local_acc.cpu().detach().numpy()

                    sys.stdout.write("Local: [%04d/%04d] Batch: [%05d/%05d] Time: %16.1f | total_loss: %8.4f | %s | %.4f %.4f %.4f | Max mem %.4f\n"%(\
                            epoch, self.n_epoch, counter*self.batch_size*world_size, self.n_train, train_time, local_tot, \
                            " ".join(["%8.4f"%l for l in local_loss]),\
                            local_acc[0], local_acc[1], local_acc[2], max_mem))

                    if self.wandb_prefix is not None and rank == 0:
                        loss_dict.update({'total_examples':epoch*len(train_loader)+counter*world_size})
                        log_dict = {f"Train":{task[0]:loss_dict}}
                        wandb.log(log_dict)

                    sys.stdout.flush()
                    local_tot = 0.0
                    local_loss = None 
                    local_acc = None 
                torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        
        # write total train loss
        train_tot /= float(counter * world_size)
        train_loss /= float(counter * world_size)
        train_acc  /= float(counter * world_size)

        dist.all_reduce(train_tot, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_acc, op=dist.ReduceOp.SUM)
        train_tot = train_tot.cpu().detach()
        train_loss = train_loss.cpu().detach().numpy()
        train_acc = train_acc.cpu().detach().numpy()
        if rank == 0:

            train_time = time.time() - start_time
            sys.stdout.write("Train: [%04d/%04d] Batch: [%05d/%05d] Time: %16.1f | total_loss: %8.4f | %s | %.4f %.4f %.4f\n"%(\
                    epoch, self.n_epoch, self.n_train, self.n_train, train_time, train_tot, \
                    " ".join(["%8.4f"%l for l in train_loss]),\
                    train_acc[0], train_acc[1], train_acc[2]))
            sys.stdout.flush()
            
        return train_tot, train_loss, train_acc

    def valid_pdb_cycle(self, ddp_model, valid_loader, rank, gpu, world_size, epoch, header='Monomer', verbose=False):
        valid_tot = 0.0
        valid_loss = None
        valid_acc = None
        counter = 0
        
        start_time = time.time()
        
        with torch.no_grad(): # no need to calculate gradient
            ddp_model.eval() # change it to eval mode
            for seq, msa, msa_masked, msa_full, mask_msa, true_crds, atom_mask, idx_pdb, xyz_t, t1d, xyz_prev, same_chain, unclamp, negative, atom_frames, bond_feats, task, item in valid_loader:
            
                # skip known bad training examples
                # loader will print warning message with item info for followup later
                if torch.is_tensor(item) and len(item.shape) == 2 and torch.all(item==-1):
                    continue

                # transfer inputs to device
                B, _, N, L = msa.shape

                idx_pdb = idx_pdb.to(gpu, non_blocking=True) # (B, L)
                true_crds = true_crds.to(gpu, non_blocking=True) # (B, L, 27, 3)
                atom_mask = atom_mask.to(gpu, non_blocking=True) # (B, L, 27)
                same_chain = same_chain.to(gpu, non_blocking=True)

                xyz_t = xyz_t.to(gpu, non_blocking=True)
                t1d = t1d.to(gpu, non_blocking=True)

                seq = seq.to(gpu, non_blocking=True)
                msa = msa.to(gpu, non_blocking=True)
                msa_masked = msa_masked.to(gpu, non_blocking=True)
                msa_full = msa_full.to(gpu, non_blocking=True)
                mask_msa = mask_msa.to(gpu, non_blocking=True)
                atom_frames = atom_frames.to(gpu, non_blocking=True)
                bond_feats = bond_feats.to(gpu, non_blocking=True)

                # res_mask = ~((atom_mask[:,:,:3].sum(dim=-1) < 3.0) * ~(is_atom(msa[:,i_cycle,0]))) # ignore residues having missing BB atoms for loss calculation
                # mask_2d = res_mask[:,None,:] * res_mask[:,:,None] # ignore pairs having missing residues

                # processing template features
                seq_unmasked = msa[:, 0, 0, :] # (B, L)
                xyz_t_frames = xyz_t_to_frame_xyz(xyz_t, seq_unmasked, atom_frames)
                t2d = xyz_to_t2d(xyz_t_frames)

                xyz_t = get_init_xyz(seq[:,0],xyz_t,same_chain)
                xyz_prev = get_init_xyz(seq[:,0],xyz_prev[:,None],same_chain).reshape(B, L, NTOTAL, 3)

                # get torsion angles from templates
                seq_tmpl = t1d[...,:-1].argmax(dim=-1).reshape(-1,L)
                alpha, _, alpha_mask, _ = get_torsions(xyz_t.reshape(-1,L,NTOTAL,3), seq_tmpl, self.ti_dev, self.ti_flip, self.ang_ref)
                alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))
                alpha[torch.isnan(alpha)] = 0.0
                alpha = alpha.reshape(B,-1,L,NTOTALDOFS,2)
                alpha_mask = alpha_mask.reshape(B,-1,L,NTOTALDOFS,1)
                alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(B, -1, L, 3*NTOTALDOFS)

                # set number of recycles
                N_cycle = self.maxcycle
                msa_prev = None
                pair_prev = None
                alpha_prev = torch.zeros((B,L,NTOTALDOFS,2)).to(gpu, non_blocking=True) #fd we could get this from the template...
                state_prev = None

                for i_cycle in range(N_cycle-1): 
                    msa_prev, pair_prev, xyz_prev, state_prev, alpha = ddp_model(
                        msa_masked[:,i_cycle],
                        msa_full[:,i_cycle],
                        seq[:,i_cycle],
                        msa[:,i_cycle,0], # unmasked seq
                        xyz_prev,
                        alpha_prev,
                        idx_pdb,
                        bond_feats,
                        t1d=t1d,
                        t2d=t2d,
                        xyz_t=xyz_t,
                        alpha_t=alpha_t,
                        msa_prev=msa_prev,
                        pair_prev=pair_prev,
                        state_prev=state_prev,
                        return_raw=True,
                        use_checkpoint=False
                    )

                    #true_crds_i, atom_mask_i = resolve_equiv_natives(xyz_prev, true_crds, atom_mask)

                    #res_mask = ~(atom_mask_i[:,:,:3].sum(dim=-1) < 3.0)
                    #mask_2d = res_mask[:,None,:] * res_mask[:,:,None]

                i_cycle = N_cycle-1
                logit_s, logit_aa_s, pred_crds, alphas, pred_allatom, pred_lddts, _, _, _ = ddp_model(
                    msa_masked[:,i_cycle],
                    msa_full[:,i_cycle],
                    seq[:,i_cycle],
                    msa[:,i_cycle,0], # unmasked seq
                    xyz_prev,
                    alpha_prev,
                    idx_pdb,
                    bond_feats,
                    t1d=t1d,
                    t2d=t2d,
                    xyz_t=xyz_t,
                    alpha_t=alpha_t,
                    msa_prev=msa_prev,
                    pair_prev=pair_prev,
                    state_prev=state_prev,
                    use_checkpoint=False
                )

                true_crds, atom_mask_ = resolve_equiv_natives(pred_crds[-1], true_crds, atom_mask)

                res_mask = ~((atom_mask_[:,:,:3].sum(dim=-1) < 3.0) * ~(is_atom(msa[:,i_cycle,0])))
                mask_2d = res_mask[:,None,:] * res_mask[:,:,None]

                true_crds_frame = xyz_to_frame_xyz(true_crds, msa[:, i_cycle, 0],atom_frames)
                c6d, _ = xyz_to_c6d(true_crds_frame)
                c6d = c6d_to_bins(c6d, same_chain, negative=negative)

                prob = self.active_fn(logit_s[0]) # distogram
                acc_s = self.calc_acc(prob, c6d[...,0], idx_pdb, mask_2d)

                ctrid = len(valid_loader)*rank+counter
                loss, loss_s, loss_dict = self.calc_loss(
                    logit_s, c6d,
                    logit_aa_s, msa[:, i_cycle], mask_msa[:,i_cycle],
                    pred_crds, alphas, pred_allatom, true_crds, 
                    atom_mask_, res_mask, mask_2d, same_chain,
                    pred_lddts, idx_pdb, atom_frames, unclamp=unclamp, negative=negative,
                    verbose=verbose, ctr=ctrid, **self.loss_param
                )

                valid_tot += loss.detach()
                if valid_loss == None:
                    valid_loss = torch.zeros_like(loss_s.detach())
                    valid_acc = torch.zeros_like(acc_s.detach())
                valid_loss += loss_s.detach()
                valid_acc += acc_s.detach()
                counter += 1

        valid_tot /= float(counter*world_size)
        valid_loss /= float(counter*world_size)
        valid_acc /= float(counter*world_size)

        dist.all_reduce(valid_tot, op=dist.ReduceOp.SUM)
        dist.all_reduce(valid_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(valid_acc, op=dist.ReduceOp.SUM)
       
        valid_tot = valid_tot.cpu().detach().numpy()
        valid_loss = valid_loss.cpu().detach().numpy()
        valid_acc = valid_acc.cpu().detach().numpy()
        
        if rank == 0:
            log_dict = {f"Valid_{header}":{task[0]:loss_dict}}
            wandb.log(log_dict)
            train_time = time.time() - start_time
            sys.stdout.write("%s: [%04d/%04d] Batch: [%05d/%05d] Time: %16.1f | total_loss: %8.4f | %s | %.4f %.4f %.4f\n"%(\
                    header, epoch, self.n_epoch, world_size*len(valid_loader), world_size*len(valid_loader), train_time, valid_tot, \
                    " ".join(["%8.4f"%l for l in valid_loss]),\
                    valid_acc[0], valid_acc[1], valid_acc[2])) 
            sys.stdout.flush()
        return valid_tot, valid_loss, valid_acc

    def valid_ppi_cycle(self, ddp_model, valid_pos_loader, valid_neg_loader, rank, gpu, world_size, epoch, header='Protein', report_interface=True, verbose=False):
        valid_tot = 0.0
        valid_loss = None
        valid_acc = None
        valid_inter = None
        counter = 0
        
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        
        start_time = time.time()
        
        with torch.no_grad(): # no need to calculate gradient
            ddp_model.eval() # change it to eval mode
            for seq, msa, msa_masked, msa_full, mask_msa, true_crds, mask_crds, idx_pdb, xyz_t, t1d, xyz_prev, same_chain, unclamp, negative, atom_frames, bond_feats in valid_pos_loader:
                # transfer inputs to device
                B, _, N, L = msa.shape

                idx_pdb = idx_pdb.to(gpu, non_blocking=True) # (B, L)
                true_crds = true_crds.to(gpu, non_blocking=True) # (B, L, 27, 3)
                atom_mask = mask_crds.to(gpu, non_blocking=True) # (B, L, 27)
                same_chain = same_chain.to(gpu, non_blocking=True)

                xyz_t = xyz_t.to(gpu, non_blocking=True)
                t1d = t1d.to(gpu, non_blocking=True)
                
                xyz_prev = xyz_prev.to(gpu, non_blocking=True)

                seq = seq.to(gpu, non_blocking=True)
                msa = msa.to(gpu, non_blocking=True)
                msa_masked = msa_masked.to(gpu, non_blocking=True)
                msa_full = msa_full.to(gpu, non_blocking=True)
                mask_msa = mask_msa.to(gpu, non_blocking=True)
                atom_frames = atom_frames.to(gpu, non_blocking=True)
                bond_feats = bond_feats.to(gpu, non_blocking=True)

                # processing labels for distogram orientograms
                # res_mask = ~((atom_mask[:,:,:3].sum(dim=-1) < 3.0) * ~(is_atom(msa[:,i_cycle,0]))) # ignore residues having missing BB atoms for loss calculation
                # mask_2d = res_mask[:,None,:] * res_mask[:,:,None] # ignore pairs having missing residues

                # processing template features
                # get torsion angles from templates
                seq_tmp = t1d[...,:-1].argmax(dim=-1).reshape(-1,L)
                xyz_t_frames = xyz_t_to_frame_xyz(xyz_t, seq_tmp, atom_frames)
                t2d = xyz_to_t2d(xyz_t_frames)

                alpha, _, alpha_mask, _ = get_torsions(xyz_t.reshape(-1,L,NTOTAL,3), seq_tmp, self.ti_dev, self.ti_flip, self.ang_ref)
                alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))
                alpha[torch.isnan(alpha)] = 0.0
                alpha = alpha.reshape(B,-1,L,NTOTALDOFS,2)
                alpha_mask = alpha_mask.reshape(B,-1,L,NTOTALDOFS,1)
                alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(B, -1, L, 3*NTOTALDOFS)
                # processing template coordinates
                xyz_t = get_init_xyz(seq[:,0],xyz_t,same_chain)
                xyz_prev = get_init_xyz(seq[:,0],xyz_prev[:,None],same_chain).reshape(B, L, NTOTAL, 3)

                N_cycle = self.maxcycle # number of recycling

                msa_prev = None
                pair_prev = None
                alpha_prev = torch.zeros((B,L,NTOTALDOFS,2)).to(gpu, non_blocking=True) #fd we could get this from the template...
                state_prev = None

                for i_cycle in range(N_cycle-1): 
                    msa_prev, pair_prev, xyz_prev, state_prev, alpha = ddp_model(
                        msa_masked[:,i_cycle],
                        msa_full[:,i_cycle],
                        seq[:,i_cycle],
                        msa[:,i_cycle,0], # unmasked seq
                        xyz_prev, 
                        alpha_prev,
                        idx_pdb,
                        bond_feats,
                        t1d=t1d,
                        t2d=t2d,
                        xyz_t=xyz_t,
                        alpha_t=alpha_t,
                        msa_prev=msa_prev,
                        pair_prev=pair_prev,
                        state_prev=state_prev,
                        return_raw=True,
                        use_checkpoint=False
                    )

                    #true_crds_i, atom_mask_i = resolve_equiv_natives(xyz_prev, true_crds, atom_mask)

                    #res_mask = ~(atom_mask_i[:,:,:3].sum(dim=-1) < 3.0)
                    #mask_2d = res_mask[:,None,:] * res_mask[:,:,None]


                i_cycle = N_cycle-1
                logit_s, logit_aa_s, pred_crds, alphas, pred_allatom, pred_lddts, _, _, _ = ddp_model(
                    msa_masked[:,i_cycle],
                    msa_full[:,i_cycle],
                    seq[:,i_cycle], 
                    msa[:,i_cycle,0], # unmasked seq
                    xyz_prev,
                    alpha_prev,
                    idx_pdb,
                    bond_feats,
                    t1d=t1d,
                    t2d=t2d,
                    xyz_t=xyz_t,
                    alpha_t=alpha_t,
                    msa_prev=msa_prev,
                    pair_prev=pair_prev,
                    state_prev=state_prev,
                    use_checkpoint=False
                )

                true_crds, atom_mask = resolve_equiv_natives(pred_crds[-1], true_crds, atom_mask)

                res_mask = ~((atom_mask[:,:,:3].sum(dim=-1) < 3.0) and ~(is_atom(msa[:,i_cycle,0])))
                mask_2d = res_mask[:,None,:] * res_mask[:,:,None]

                true_crds_frame = xyz_to_frame_xyz(true_crds, msa[:, i_cycle, 0],atom_frames)
                c6d, _ = xyz_to_c6d(true_crds_frame)
                c6d = c6d_to_bins(c6d, same_chain, negative=negative)

                prob = self.active_fn(logit_s[0]) # distogram
                acc_s, cnt_pred, cnt_ref = self.calc_acc(prob, c6d[...,0], idx_pdb, mask_2d, return_cnt=True)

                # inter-chain contact prob
                cnt_pred = cnt_pred * (1-same_chain).float()
                cnt_ref = cnt_ref * (1-same_chain).float()
                max_prob = cnt_pred.max()
                if max_prob > 0.5:
                    if (cnt_ref > 0).any():
                        TP += 1.0
                    else:
                        FP += 1.0
                else:
                    if (cnt_ref > 0).any():
                        FN += 1.0
                    else:
                        TN += 1.0
                inter_s = torch.tensor([TP, FP, TN, FN], device=prob.device).float()

                ctrid = len(valid_pos_loader)*rank+counter
                loss, loss_s, loss_dict = self.calc_loss(
                    logit_s, c6d,
                    logit_aa_s, msa[:, i_cycle], mask_msa[:,i_cycle],
                    pred_crds, alphas, pred_allatom, true_crds,
                    atom_mask, res_mask, mask_2d, same_chain,
                    pred_lddts, idx_pdb, atom_frames, unclamp=unclamp, negative=negative, interface=report_interface,
                    verbose=verbose, ctr=ctrid, **self.loss_param
                )

                valid_tot += loss.detach()
                if valid_loss == None:
                    valid_loss = torch.zeros_like(loss_s.detach())
                    valid_acc = torch.zeros_like(acc_s.detach())
                    valid_inter = torch.zeros_like(inter_s.detach())
                valid_loss += loss_s.detach()
                valid_acc += acc_s.detach()
                valid_inter += inter_s.detach()
                counter += 1

            
        valid_tot /= float(counter*world_size)
        valid_loss /= float(counter*world_size)
        valid_acc /= float(counter*world_size)
        
        dist.all_reduce(valid_tot, op=dist.ReduceOp.SUM)
        dist.all_reduce(valid_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(valid_acc, op=dist.ReduceOp.SUM)
       
        valid_tot = valid_tot.cpu().detach().numpy()
        valid_loss = valid_loss.cpu().detach().numpy()
        valid_acc = valid_acc.cpu().detach().numpy()
        
        if rank == 0:
            
            train_time = time.time() - start_time
            sys.stdout.write("%s-interface: [%04d/%04d] Batch: [%05d/%05d] Time: %16.1f | total_loss: %8.4f | %s | %.4f %.4f %.4f\n"%(\
                    header, epoch, self.n_epoch, counter*world_size, counter*world_size, train_time, valid_tot, \
                    " ".join(["%8.4f"%l for l in valid_loss]),\
                    valid_acc[0], valid_acc[1], valid_acc[2])) 
            sys.stdout.flush()
        
        valid_tot = 0.0
        valid_loss = None
        valid_acc = None
        counter = 0

        start_time = time.time()
        
        with torch.no_grad(): # no need to calculate gradient
            ddp_model.eval() # change it to eval mode
            for seq, msa, msa_masked, msa_full, mask_msa, true_crds, mask_crds, idx_pdb, xyz_t, t1d, xyz_prev, same_chain, unclamp, negative, atom_frames in valid_neg_loader:
                # transfer inputs to device
                B, _, N, L = msa.shape

                idx_pdb = idx_pdb.to(gpu, non_blocking=True) # (B, L)
                true_crds = true_crds.to(gpu, non_blocking=True) # (B, L, 27, 3)
                atom_mask = mask_crds.to(gpu, non_blocking=True) # (B, L, 27)
                same_chain = same_chain.to(gpu, non_blocking=True)

                xyz_t = xyz_t.to(gpu, non_blocking=True)
                t1d = t1d.to(gpu, non_blocking=True)
                
                xyz_prev = xyz_prev.to(gpu, non_blocking=True)

                seq = seq.to(gpu, non_blocking=True)
                msa = msa.to(gpu, non_blocking=True)
                msa_masked = msa_masked.to(gpu, non_blocking=True)
                msa_full = msa_full.to(gpu, non_blocking=True)
                mask_msa = mask_msa.to(gpu, non_blocking=True)
                atom_frames = atom_frames.to(gpu, non_blocking=True)
                bond_feats = bond_feats.to(gpu, non_blocking=True)
                
                # processing labels for distogram orientograms
                res_mask = ~((atom_mask[:,:,:3].sum(dim=-1) < 3.0) * ~(is_atom(msa[:,i_cycle,0]))) # ignore residues having missing BB atoms for loss calculation
                mask_2d = res_mask[:,None,:] * res_mask[:,:,None] # ignore pairs having missing residues

                # processing template features
                # get torsion angles from templates
                seq_tmp = t1d[...,:-1].argmax(dim=-1).reshape(-1,L)
                xyz_t_frames = xyz_t_to_frame_xyz(xyz_t, seq_tmp, atom_frames)
                t2d = xyz_to_t2d(xyz_t_frames)

                alpha, _, alpha_mask, _ = get_torsions(xyz_t.reshape(-1,L,NTOTAL,3), seq_tmp, self.ti_dev, self.ti_flip, self.ang_ref)
                alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))
                alpha[torch.isnan(alpha)] = 0.0
                alpha = alpha.reshape(B,-1,L,NTOTALDOFS,2)
                alpha_mask = alpha_mask.reshape(B,-1,L,NTOTALDOFS,1)
                alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(B, -1, L, 3*NTOTALDOFS)
                # processing template coordinates
                xyz_t = get_init_xyz(seq[:,0],xyz_t,same_chain)
                xyz_prev = get_init_xyz(seq[:,0],xyz_prev[:,None],same_chain).reshape(B, L, NTOTAL, 3)

                N_cycle = self.maxcycle # number of recycling

                msa_prev = None
                pair_prev = None
                alpha_prev = torch.zeros((B,L,NTOTALDOFS,2)).to(gpu, non_blocking=True) #fd we could get this from the template...
                state_prev = None
                for i_cycle in range(N_cycle-1): 
                    msa_prev, pair_prev, xyz_prev, state_prev, alpha = ddp_model(
                        msa_masked[:,i_cycle],
                        msa_full[:,i_cycle],
                        seq[:,i_cycle],
                        msa[:,i_cycle,0], # unmasked seq
                        xyz_prev, 
                        alpha_prev,
                        idx_pdb,
                        bond_feats,
                        t1d=t1d,
                        t2d=t2d,
                        xyz_t=xyz_t,
                        alpha_t=alpha_t,
                        msa_prev=msa_prev,
                        pair_prev=pair_prev,
                        state_prev=state_prev,
                        return_raw=True,
                        use_checkpoint=False
                    )

                i_cycle = N_cycle-1
                logit_s, logit_aa_s, pred_crds, alphas, pred_allatom, pred_lddts, _, _, _ = ddp_model(
                    msa_masked[:,i_cycle],
                    msa_full[:,i_cycle],
                    seq[:,i_cycle], 
                    msa[:,i_cycle,0], # unmasked seq
                    xyz_prev,
                    alpha_prev,
                    idx_pdb,
                    bond_feats,
                    t1d=t1d,
                    t2d=t2d,
                    xyz_t=xyz_t,
                    alpha_t=alpha_t,
                    msa_prev=msa_prev,
                    pair_prev=pair_prev,
                    state_prev=state_prev,
                    use_checkpoint=False
                )

                true_crds, atom_mask = resolve_equiv_natives(pred_crds[-1], true_crds, atom_mask)

                res_mask = ~((atom_mask[:,:,:3].sum(dim=-1) < 3.0) * ~(is_atom(msa[:,i_cycle,0])))
                mask_2d = res_mask[:,None,:] * res_mask[:,:,None]

                true_crds_frame = xyz_to_frame_xyz(true_crds, msa[:, i_cycle, 0],atom_frames)
                c6d, _ = xyz_to_c6d(true_crds_frame)
                c6d = c6d_to_bins(c6d, same_chain, negative=negative)

                prob = self.active_fn(logit_s[0]) # distogram
                acc_s, cnt_pred, cnt_ref = self.calc_acc(prob, c6d[...,0], idx_pdb, mask_2d, return_cnt=True)
                
                # inter-chain contact prob
                cnt_pred = cnt_pred * (1-same_chain).float()
                cnt_ref = cnt_ref * (1-same_chain).float()
                max_prob = cnt_pred.max()
                if max_prob > 0.5:
                    if (cnt_ref > 0).any():
                        TP += 1.0
                    else:
                        FP += 1.0
                else:
                    if (cnt_ref > 0).any():
                        FN += 1.0
                    else:
                        TN += 1.0
                inter_s = torch.tensor([TP, FP, TN, FN], device=prob.device).float()

                loss, loss_s, loss_dict = self.calc_loss(
                    logit_s, c6d,
                    logit_aa_s, msa[:, i_cycle], mask_msa[:,i_cycle],
                    pred_crds, alphas, pred_allatom, true_crds,
                    atom_mask, res_mask, mask_2d, same_chain,
                    pred_lddts, idx_pdb, atom_frames, unclamp=unclamp, negative=negative,
                    verbose=verbose, ctr=ctrid, **self.loss_param
                )
                
                valid_tot += loss.detach()
                if valid_loss == None:
                    valid_loss = torch.zeros_like(loss_s.detach())
                    valid_acc = torch.zeros_like(acc_s.detach())
                valid_loss += loss_s.detach()
                valid_acc += acc_s.detach()
                valid_inter += inter_s.detach()
                counter += 1


            
        valid_tot /= float(counter*world_size)
        valid_loss /= float(counter*world_size)
        valid_acc /= float(counter*world_size)
        
        dist.all_reduce(valid_tot, op=dist.ReduceOp.SUM)
        dist.all_reduce(valid_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(valid_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(valid_inter, op=dist.ReduceOp.SUM)
       
        valid_tot = valid_tot.cpu().detach().numpy()
        valid_loss = valid_loss.cpu().detach().numpy()
        valid_acc = valid_acc.cpu().detach().numpy()
        valid_inter = valid_inter.cpu().detach().numpy()
        
        if rank == 0:
            TP, FP, TN, FN = valid_inter 
            prec = TP/(TP+FP+1e-4)
            recall = TP/(TP+FN+1e-4)
            F1 = 2*TP/(2*TP+FP+FN+1e-4)
            
            train_time = time.time() - start_time
            sys.stdout.write("%s-PPI: [%04d/%04d] Batch: [%05d/%05d] Time: %16.1f | total_loss: %8.4f | %s | %.4f %.4f %.4f | %.4f %.4f %.4f\n"%(\
                    header, epoch, self.n_epoch, counter*world_size, counter*world_size, train_time, valid_tot, \
                    " ".join(["%8.4f"%l for l in valid_loss]),\
                    valid_acc[0], valid_acc[1], valid_acc[2],\
                    prec, recall, F1))
            sys.stdout.flush()
        return valid_tot, valid_loss, valid_acc

if __name__ == "__main__":
    from arguments import get_args
    args, model_param, loader_param, loss_param = get_args()

    print (args)

    mp.freeze_support()
    train = Trainer(model_name=args.model_name,
                    n_epoch=args.num_epochs, step_lr=args.step_lr, lr=args.lr, l2_coeff=1.0e-2,
                    port=args.port, model_param=model_param, loader_param=loader_param, 
                    loss_param=loss_param, 
                    batch_size=args.batch_size,
                    accum_step=args.accum,
                    maxcycle=args.maxcycle,
                    eval=args.eval,
                    outdir=args.outdir,
                    wandb_prefix=args.wandb_prefix)
    train.run_model_training(torch.cuda.device_count())
