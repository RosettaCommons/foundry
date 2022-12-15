import sys, os, time, datetime, subprocess, shutil
import numpy as np
import pandas as pd
from copy import deepcopy
from collections import OrderedDict
import wandb
import torch
import torch.nn as nn
from torch.utils import data
from functools import partial
from data_loader import (
    get_train_valid_set, loader_pdb, loader_fb, loader_complex, loader_na_complex, loader_rna, loader_sm, loader_sm_compl, loader_atomize_pdb,
    Dataset, DatasetComplex, DatasetNAComplex, DatasetRNA, DatasetSM, DatasetSMComplex, DistilledDataset, DistributedWeightedSampler
)
from kinematics import xyz_to_c6d, c6d_to_bins, xyz_to_t2d, xyz_to_bbtor
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
              'num_workers': 5,
              'pin_memory': True}

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
                 model_param={}, loader_param={}, loss_param={}, dataset_param={}, batch_size=1, 
                 accum_step=1, maxcycle=4, eval=False, out_dir=None, wandb_prefix=None, 
                 model_dir='models/'):

        self.model_name = model_name 
        self.n_epoch = n_epoch
        self.step_lr = step_lr
        self.init_lr = lr
        self.l2_coeff = l2_coeff
        self.port = port
        self.interactive = interactive
        self.eval = eval
        self.model_param = model_param
        self.loader_param = loader_param
        self.loss_param = loss_param
        self.dataset_param = dataset_param
        self.ACCUM_STEP = accum_step
        self.batch_size = batch_size
        self.out_dir = out_dir 
        if out_dir is not None: 
            os.makedirs(self.out_dir, exist_ok=True)
            if out_dir[-1] != '/': self.out_dir += '/'
        self.wandb_prefix = wandb_prefix
        self.model_dir = model_dir

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
                  logit_aa_s, label_aa_s, mask_aa_s, logit_pae, logit_pde,
                  pred, pred_tors, pred_allatom, true,
                  mask_crds, mask_BB, mask_2d, same_chain,
                  pred_lddt, idx, bond_feats, atom_frames=None, unclamp=False, 
                  negative=False, interface=False,
                  verbose=False, ctr=0,
                  w_dist=1.0, w_aa=1.0, w_str=1.0, w_inter_fape=0.0, w_lig_fape=1.0, w_lddt=1.0, 
                  w_bond=1.0, w_clash=0.0, w_atom_bond=0.0, w_skip_bond=0.0, w_rigid=0.0, w_hb=0.0, w_dih=0.0,
                  w_pae=0.0, w_pde=0.0, lj_lin=0.85, eps=1e-6, item=None, task=None, out_dir='./'
    ):
        gpu = pred.device

        # track losses for printing to local log and uploading to WandB
        loss_dict = OrderedDict()

        B, L = true.shape[:2]
        seq = label_aa_s[:,0].clone()

        assert (B==1) # fd - code assumes a batch size of 1
        
        tot_loss = 0.0
        
        # c6d loss
        for i in range(4):
            loss = self.loss_fn(logit_s[i], label_s[...,i]) # (B, L, L)
            loss = (mask_2d*loss).sum() / (mask_2d.sum() + eps)
            tot_loss += w_dist*loss
            loss_dict[f'c6d_{i}'] = loss.detach()

        # masked token prediction loss
        loss = self.loss_fn(logit_aa_s, label_aa_s.reshape(B, -1))
        loss = loss * mask_aa_s.reshape(B, -1)
        loss = loss.sum() / (mask_aa_s.sum() + 1e-8)
        tot_loss += w_aa*loss
        loss_dict['aa_cce'] = loss.detach()

        ### GENERAL LAYERS
        # Structural loss (layer-wise backbone FAPE)
        dclamp = 300.0 if unclamp else 30.0 # protein & NA FAPE distance cutoffs
        dclamp_sm, Z_sm = 4, 4  # sm mol FAPE distance cutoffs

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
            l_fape_A, _, _ = compute_general_FAPE(
                pred[:,mask_BBA,:,:3],
                true[:,mask_BBA[0],:3],
                mask_crds[:,mask_BBA[0], :3],
                frames_BB[:,mask_BBA[0]],
                frame_mask_BB[:,mask_BBA[0]],
                dclamp=dclamp
            )
        else:
            l_fape_A = torch.tensor([0]).to(gpu)
        loss_dict['bb_fape_c1'] = l_fape_A[-1].detach()

        mask_BBB = mask_BB.clone()
        mask_BBB[0,:L1] = False
        if torch.sum(mask_BBB) >0:
            l_fape_B, _, _ = compute_general_FAPE(
                pred[:, mask_BBB,:,:3],
                true[:,mask_BBB[0],:3,:3],
                mask_crds[:,mask_BBB[0], :3],
                frames_BB[:,mask_BBB[0]],
                frame_mask_BB[:,mask_BBB[0]],
                dclamp=dclamp
            )
        else:
            l_fape_B = torch.tensor([0]).to(gpu)

        loss_dict['bb_fape_c2'] = l_fape_B[-1].detach()
            
        if negative: # inter-chain fapes should be ignored for negative cases
            fracA = float(L1)/len(same_chain[0,0])
            tot_str = fracA*l_fape_A + (1.0-fracA)*l_fape_B
            pae_loss = torch.tensor(0).to(gpu)
            pae_loss = torch.tensor(0).to(gpu)
        else:
            if logit_pae is not None:
                logit_pae = logit_pae[:,:,mask_BB[0]][:,:,:,mask_BB[0]]
            if logit_pde is not None:
                logit_pde = logit_pde[:,:,mask_BB[0]][:,:,:,mask_BB[0]]
            tot_str, pae_loss, pde_loss = compute_general_FAPE(
                pred[:,mask_BB,:,:3],
                true[:,mask_BB[0],:3],
                mask_crds[:,mask_BB[0],:3],
                frames_BB[:,mask_BB[0]],
                frame_mask_BB[:,mask_BB[0]],
                dclamp=dclamp, 
                logit_pae=logit_pae,
                logit_pde=logit_pde
            )
        num_layers = pred.shape[0]
        gamma = 0.99
        w_bb_fape = torch.pow(torch.full((num_layers,), gamma, device=pred.device), torch.arange(num_layers, device=pred.device))
        w_bb_fape = torch.flip(w_bb_fape, (0,))
        w_bb_fape = w_bb_fape / w_bb_fape.sum()
        bb_l_fape = (w_bb_fape*tot_str).sum()

        tot_loss += 0.5*w_str*bb_l_fape
        for i in range(len(tot_str)):
            loss_dict[f'bb_fape_layer{i}'] = tot_str[i].detach()
        loss_dict['bb_fape_full'] = bb_l_fape.detach()

        tot_loss += w_pae*pae_loss + w_pde*pde_loss
        loss_dict['pae_loss'] = pae_loss.detach()
        loss_dict['pde_loss'] = pde_loss.detach()

        sm_res_mask = is_atom(label_aa_s[0,0])*mask_BB[0] # (L,)
        if bool(torch.any(sm_res_mask)):
            # ligand fape (layer-averaged fape on atom coordinates with atom frames)
            l_fape_sm, _, _ = compute_general_FAPE(
                pred[:, sm_res_mask[None],:,:3],
                true[:,sm_res_mask,:3,:3],
                atom_mask = mask_crds[:,sm_res_mask, :3],
                frames = frames_BB[:,sm_res_mask],
                frame_mask = frame_mask_BB[:,sm_res_mask],
                dclamp=dclamp_sm,
                Z=Z_sm
            )
            lig_fape = (w_bb_fape*l_fape_sm).sum()
            tot_loss += 0.5*w_lig_fape*lig_fape
        else:
            lig_fape = torch.tensor(0).to(gpu)
        if not bool(torch.all(sm_res_mask)) and bool(torch.any(sm_res_mask)):    # not all atoms but some atoms    
            # calculate interchain fape 
            # fape of protein coordinates wrt ligand frames 
            mask_crds_protein = mask_crds.clone()
            mask_crds_protein[:, sm_res_mask] = False
            frame_mask_BB_sm = frame_mask_BB.clone()
            frame_mask_BB_sm[:,~sm_res_mask] = False
            l_fape_protein_sm, _, _ = compute_general_FAPE(
                pred[:, mask_BB,:,:3],
                true[:, mask_BB[0],:3,:3],
                atom_mask = mask_crds_protein[:,mask_BB[0], :3],
                frames = frames_BB[:,mask_BB[0]],
                frame_mask = frame_mask_BB_sm[:,mask_BB[0]],
                frame_atom_mask = mask_crds[:,mask_BB[0],:3],
                dclamp=dclamp
            )
            # fape of ligand coordinates wrt protein frames
            mask_crds_sm = mask_crds.clone()
            mask_crds_sm[:, ~sm_res_mask] = False
            frame_mask_BB_protein = frame_mask_BB.clone()
            frame_mask_BB_protein[:,sm_res_mask] = False
            l_fape_sm_protein, _, _ = compute_general_FAPE(
                pred[:, mask_BB,:,:3],
                true[:, mask_BB[0],:3,:3],
                atom_mask = mask_crds_sm[:,mask_BB[0], :3],
                frames = frames_BB[:,mask_BB[0]],
                frame_mask = frame_mask_BB_protein[:,mask_BB[0]],
                frame_atom_mask = mask_crds[:,mask_BB[0],:3],
                dclamp=dclamp
            )
            frac_sm = torch.sum(frame_mask_BB_sm[:,mask_BB[0]])/ torch.sum(frame_mask_BB[:,mask_BB[0]])
            inter_fape = frac_sm*l_fape_protein_sm + (1.0-frac_sm)*l_fape_sm_protein
            bb_l_fape_inter = (w_bb_fape*inter_fape).sum()
            tot_loss += 0.5*w_inter_fape*bb_l_fape_inter
        else:
            bb_l_fape_inter = torch.tensor(0).to(gpu)
            #l_fape_sm = torch.tensor([0]).to(gpu)

        #loss_dict['bb_fape_lig'] = l_fape_sm[-1].detach()
        loss_dict['bb_fape_lig'] = lig_fape.detach()
        loss_dict['bb_fape_inter'] = bb_l_fape_inter.detach()

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
        loss_dict['torsion'] = l_tors.detach()

        ### FINETUNING LAYERS
        # lddts (CA)
        ca_lddt = calc_lddt(pred[:,:,:,1].detach(), true[:,:,1], mask_BB, mask_2d, same_chain, negative=negative, interface=interface)
        loss_dict['ca_lddt'] = ca_lddt[-1].detach()

        # lddts (allatom) + lddt loss
        lddt_loss, allatom_lddt = calc_allatom_lddt_loss(
            pred_allatom.detach(), nat_symm, pred_lddt, idx, mask_crds, mask_2d, same_chain, negative=negative, interface=interface)
        tot_loss += w_lddt*lddt_loss
        loss_dict['lddt_loss'] = lddt_loss.detach()
        loss_dict['allatom_lddt'] = allatom_lddt[0].detach()

        # FAPE losses
        # allatom fape and torsion angle loss
        # frames, frame_mask = get_frames(
        #     pred_allatom[-1,None,...], mask_crds, seq, self.fi_dev, atom_frames)
        if negative: # inter-chain fapes should be ignored for negative cases
            # L1 = same_chain[0,0,:].sum()
            # mask_BBA = mask_BB.clone()
            # mask_BBA[0, L1:] = False
            l_fape_A, _, _ = compute_general_FAPE(
                pred_allatom[:,mask_BBA[0],:,:3],
                nat_symm[None,mask_BBA[0],:,:3],
                xs_mask[:,mask_BBA[0]],
                frames[:,mask_BBA[0]],
                frame_mask[:,mask_BBA[0]]
            )
            # mask_BBB = mask_BB.clone()
            # mask_BBB[0,:L1] = False
            l_fape_B, _, _ = compute_general_FAPE(
                pred_allatom[:,mask_BBB[0],:,:3],
                nat_symm[None,mask_BBB[0],:,:3],
                xs_mask[:,mask_BBB[0]],
                frames[:,mask_BBB[0]],
                frame_mask[:,mask_BBB[0]]
            )
            fracA = float(L1)/len(same_chain[0,0])
            l_fape = fracA*l_fape_A + (1.0-fracA)*l_fape_B

        else:
            l_fape, _, _ = compute_general_FAPE(
                pred_allatom[:,mask_BB[0],:,:3],
                nat_symm[None,mask_BB[0],:,:3],
                xs_mask[:,mask_BB[0]],
                frames[:,mask_BB[0]],
                frame_mask[:,mask_BB[0]]
            )

        tot_loss += w_str*l_fape[0]
        loss_dict['allatom_fape'] = l_fape[0].detach()

        # rmsd loss (for logging only)
        rmsd = calc_crd_rmsd(
            pred_allatom[:,mask_BB[0],:,:3],
            nat_symm[None,mask_BB[0],:,:3],
            xs_mask[:,mask_BB[0]]
            )
        loss_dict["rmsd"] = rmsd[0].detach()
        if torch.any(mask_BBB):
            xs_mask_c1, xs_mask_c2 = xs_mask.clone(), xs_mask.clone()
            xs_mask_c1[:,~mask_BBA[0]] = False
            xs_mask_c2[:,~mask_BBB[0]] = False
            rmsd_c1_c1 = calc_crd_rmsd(
                pred=pred_allatom[:,mask_BB[0],:,:3], true=nat_symm[None,mask_BB[0],:,:3],
                atom_mask=xs_mask_c1[:,mask_BB[0]], rmsd_mask=xs_mask_c1[:,mask_BB[0]]
            )
            rmsd_c1_c2 = calc_crd_rmsd(
                pred=pred_allatom[:,mask_BB[0],:,:3], true=nat_symm[None,mask_BB[0],:,:3],
                atom_mask=xs_mask_c1[:,mask_BB[0]], rmsd_mask=xs_mask_c2[:,mask_BB[0]]
            )
            rmsd_c2_c2 = calc_crd_rmsd(
                pred=pred_allatom[:,mask_BB[0],:,:3], true=nat_symm[None,mask_BB[0],:,:3],
                atom_mask=xs_mask_c2[:,mask_BB[0]], rmsd_mask=xs_mask_c2[:,mask_BB[0]]
            )
            loss_dict["rmsd_c1_c1"]= rmsd_c1_c1[0].detach()
            loss_dict["rmsd_c1_c2"]= rmsd_c1_c2[0].detach()
            loss_dict["rmsd_c2_c2"]= rmsd_c2_c2[0].detach()
        else:
            loss_dict["rmsd_c1_c1"]= loss_dict['rmsd']
            loss_dict["rmsd_c1_c2"]= torch.tensor(0, device=pred.device)
            loss_dict["rmsd_c2_c2"]= torch.tensor(0, device=pred.device)

        # cart bonded (bond geometry)
        bond_loss = calc_BB_bond_geom(seq[0], pred_allatom[0:1], idx)
        if w_bond > 0.0:
            tot_loss += w_bond*bond_loss
        loss_dict['bond_geom'] = bond_loss.detach()

        if (pred_allatom.shape[0] > 1):
            bond_loss = calc_cart_bonded(seq, pred_allatom[1:], idx, self.cb_len, self.cb_ang, self.cb_tor)
            if w_bond > 0.0:
                tot_loss += w_bond*bond_loss.mean()
            oss_dict['clash_loss'] = ( bond_loss.detach() )
        else:
            bond_loss = torch.tensor(0).to(gpu)
        loss_dict['bond_loss'] = bond_loss.detach()

        # clash [use all atoms not just those in native]
        clash_loss = calc_lj(
            seq[0], pred_allatom, 
            self.aamask, bond_feats, self.ljlk_parameters, self.lj_correction_parameters, self.num_bonds,
            lj_lin=lj_lin
        )
        if w_clash > 0.0:
            tot_loss += w_clash*clash_loss.mean()
        loss_dict['clash_loss'] = clash_loss[0].detach()
        atom_bond_loss, skip_bond_loss, rigid_loss = calc_atom_bond_loss(pred_allatom, nat_symm[None], bond_feats, seq)
        if w_atom_bond >= 0.0:
            tot_loss += w_atom_bond*atom_bond_loss
        loss_dict['atom_bond_loss'] = ( atom_bond_loss.detach() )

        if w_skip_bond >= 0.0:
            tot_loss += w_skip_bond*skip_bond_loss
        loss_dict['skip_bond_loss'] = ( skip_bond_loss.detach() )

        if w_rigid >= 0.0:
            tot_loss += w_rigid*rigid_loss
        loss_dict['rigid_loss'] = ( rigid_loss.detach() )
        L0 = same_chain[0,0,:].sum()
        chain1 = torch.zeros_like(same_chain, dtype=bool)
        chain1[:,:L0,:L0] = True
        _, allatom_lddt_c1 = calc_allatom_lddt_loss(
            pred_allatom.detach(), nat_symm, pred_lddt, idx, mask_crds, mask_2d, chain1, negative=True)
        loss_dict['allatom_lddt_c1'] = allatom_lddt_c1[0].detach()

        chain2 = torch.zeros_like(same_chain, dtype=bool)
        chain2[:,L0:,L0:] = True
        _, allatom_lddt_c2 = calc_allatom_lddt_loss(
            pred_allatom.detach(), nat_symm, pred_lddt, idx, mask_crds, mask_2d, chain2, negative=True, bin_scaling=0.5)
        loss_dict['allatom_lddt_c2'] = allatom_lddt_c2[0].detach()

        _, allatom_lddt_inter = calc_allatom_lddt_loss(
            pred_allatom.detach(), nat_symm, pred_lddt, idx, mask_crds, mask_2d, same_chain, interface=True)
        loss_dict['allatom_lddt_inter'] = allatom_lddt_inter[0].detach()
        # hbond [use all atoms not just those in native]
        #hb_loss = calc_hb(
        #    seq[0], pred_all[0,...,:3], 
        #    self.aamask, self.hbtypes, self.hbbaseatoms, self.hbpolys, 
        #    normalize=(not verbose)
        #)
        #if w_hb > 0.0:
        #    tot_loss += w_hb*hb_loss
        #oss_dict['clash_loss'] = (torch.stack((hb_loss, clash_loss, bond_loss)).detach())

        loss_dict['total_loss'] = tot_loss.detach()

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
            #writepdb(out_dir+"p_"+self.model_name+"_"+str(ctr)+".pdb", pred_all[-1,mask_BB[0]][:,:23], seq[mask_BB][:])
            #writepdb(out_dir+"n_"+str(ctr)+".pdb", true[mask_BB][:,:23], seq[mask_BB][:])
            #writepdb(out_dir+"nre_"+str(ctr)+".pdb", _n0[mask_BB], seq[mask_BB][:])

        return tot_loss, loss_dict


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

    def lddt_unbin(self, pred_lddt):
        nbin = pred_lddt.shape[1]
        bin_step = 1.0 / nbin
        lddt_bins = torch.linspace(bin_step, 1.0, nbin, dtype=pred_lddt.dtype, device=pred_lddt.device)
        pred_lddt = torch.nn.Softmax(dim=1)(pred_lddt)
        return torch.sum(lddt_bins[None,:,None]*pred_lddt, dim=1)

    def pae_unbin(self, logits_pae, bin_step=0.5):
        nbin = logits_pae.shape[1]
        bins = torch.linspace(bin_step*0.5, bin_step*nbin-bin_step*0.5, nbin, 
                              dtype=logits_pae.dtype, device=logits_pae.device)
        logits_pae = torch.nn.Softmax(dim=1)(logits_pae)
        return torch.sum(bins[None,:,None,None]*logits_pae, dim=1)

    def pde_unbin(self, logits_pde, bin_step=0.3):
        nbin = logits_pde.shape[1]
        bins = torch.linspace(bin_step*0.5, bin_step*nbin-bin_step*0.5, nbin, 
                              dtype=logits_pde.dtype, device=logits_pde.device)
        logits_pde = torch.nn.Softmax(dim=1)(logits_pde)
        return torch.sum(bins[None,:,None,None]*logits_pde, dim=1)

    def calc_pred_err(self, pred_lddts, logit_pae, logit_pde, seq):
        """Calculates summary metrics on predicted lDDT and distance errors"""
        plddts = self.lddt_unbin(pred_lddts)
        pae = self.pae_unbin(logit_pae) if logit_pae is not None else None
        pde = self.pde_unbin(logit_pde) if logit_pde is not None else None

        sm_mask = is_atom(seq)
        sm_mask_2d = sm_mask[None,:]*sm_mask[:,None]
        prot_mask_2d = (~sm_mask[None,:])*(~sm_mask[:,None])
        inter_mask_2d = sm_mask[None,:]*(~sm_mask[:,None]) + (~sm_mask[None,:])*sm_mask[:,None]

        # assumes B=1
        err_dict = dict(
            plddt = float(plddts.mean()),
            pae = float(pae.mean()) if pae is not None else None,
            pae_lig = float(pae[0,sm_mask_2d].mean()) if pae is not None else None,
            pae_prot = float(pae[0,prot_mask_2d].mean()) if pae is not None else None,
            pae_inter = float(pae[0,inter_mask_2d].mean()) if pae is not None else None,
            pde = float(pde.mean()) if pde is not None else None,
            pde_lig = float(pde[0,sm_mask_2d].mean()) if pde is not None else None,
            pde_prot = float(pde[0,prot_mask_2d].mean()) if pde is not None else None,
            pde_inter = float(pde[0,inter_mask_2d].mean()) if pde is not None else None,
        )
        return err_dict

    def load_model(self, model, model_name, rank, suffix='last', resume_train=False, 
                   optimizer=None, scheduler=None, scaler=None):
        chk_fn = self.model_dir+"/%s_%s.pt"%(model_name, suffix)
        loaded_epoch = -1
        best_valid_loss = 999999.9
        if not os.path.exists(chk_fn):
            print ('no model found', model_name)
            return -1, best_valid_loss
        map_location = {"cuda:%d"%0: "cuda:%d"%rank}
        checkpoint = torch.load(chk_fn, map_location=map_location)
        if rank == 0:
            print ('loading model', model_name, 'from', chk_fn, 'epoch', checkpoint['epoch'])
        new_params = False
        new_chk = {}
        msd_src = checkpoint['model_state_dict']
        msd_tgt = model.module.model.state_dict()
        for param in msd_tgt:
            if param not in msd_src:
                if rank == 0: print ('missing',param)
                new_params = True
                #break
            elif (msd_tgt[param].shape == msd_src[param].shape):
                new_chk[param] = msd_src[param]
            else:
                # fd hack for new encoding
                if (msd_src[param].shape[0]==30 and msd_tgt[param].shape[0]==32 and 'compute_allatom_coords' not in param):
                    if rank == 0: print ('Fixing',param)
                    new_chk[param] = torch.zeros_like(msd_tgt[param])
                    new_chk[param][:26] =  msd_src[param][:26]
                    new_chk[param][27:31] =  msd_src[param][26:30]

                else:
                    #wrong size latent_emb.emb.weight torch.Size([256, 64]) torch.Size([256, 68])
                    #wrong size templ_emb.emb.weight torch.Size([64, 104]) torch.Size([64, 108])
                    #wrong size full_emb.emb.weight torch.Size([64, 33]) torch.Size([64, 35])

                    if rank == 0:
                        print (
                            'wrong size',param,
                             checkpoint['model_state_dict'][param].shape,
                             model.module.model.state_dict()[param].shape )
                    new_params = True

        #new_chk = checkpoint['model_state_dict']
        model.module.model.load_state_dict(new_chk, strict=False)
        model.module.shadow.load_state_dict(new_chk, strict=False)

        #if resume_train and (not rename_model):
        if resume_train:
            loaded_epoch = checkpoint['epoch']
            if not new_params:
                if rank == 0: print (' ... loading optimization params')
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                if rank == 0: print (' ... loading scheduler params')
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            else:
                scheduler.last_epoch = loaded_epoch + 1
            if 'best_loss' in checkpoint:
                best_valid_loss = checkpoint['best_loss']

        return loaded_epoch, best_valid_loss

    def checkpoint_fn(self, model_name, description):
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        name = "%s_%s.pt"%(model_name, description)
        return os.path.join(self.model_dir, name)
    
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
 
    def record_git_commit(self):
        # git hash of current commit
        try:
            commit = subprocess.check_output(f'git --git-dir {script_dir}/.git rev-parse HEAD',
                                                  shell=True).decode().strip()
        except subprocess.CalledProcessError:
            print('WARNING: Failed to determine git commit hash.')
            commit = 'unknown'

        # save git diff from last commit
        git_diff = subprocess.Popen(['git diff'], cwd = os.getcwd(), shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
        out, err = git_diff.communicate()

        git_outdir = self.out_dir if self.out_dir is not None else './'
        datestr = str(datetime.datetime.now()).replace(':','').replace(' ','_') # YYYY-MM-DD_HHMMSS.xxxxxx
        with open(f'{git_outdir}/git_diff_{datestr}.txt','w') as outf:
            if self.eval: 
                print('eval', file=outf)
            else:
                print('train', file=outf)
            print(commit, file=outf)
            print(out.decode(), file=outf)

        print(f'Current date/time: {datestr}')
        print('Saved git diff between current state and last commit')

    def train_model(self, rank, world_size):
       
        if rank==0: self.record_git_commit()

        # wandb logging
        if self.wandb_prefix is not None and rank == 0:
            print('initializing wandb')
            #wandb.require("service")
            wandb.init(
                project='RF2_allatom',
                entity='bakerlab',
                name=self.wandb_prefix,
                resume=True
            )
            all_param = {}
            all_param.update(self.loader_param)
            all_param.update(self.model_param)
            all_param.update(self.loss_param)

            wandb.config = all_param
            wandb.save(os.path.join(os.getcwd(), self.out_dir, 'git_diff.txt'))

        #print ("running ddp on rank %d, world_size %d"%(rank, world_size))
        gpu = rank % torch.cuda.device_count()
        dist.init_process_group(backend="gloo", world_size=world_size, rank=rank)
        torch.cuda.set_device("cuda:%d"%gpu)

        #define dataset & data loader
        (
            pdb_items, fb_items, compl_items, neg_items, na_compl_items, na_neg_items, rna_items,
            sm_compl_items, sm_items, valid_pdb, valid_homo, valid_compl, valid_neg, valid_na_compl, 
            valid_na_neg, valid_rna, valid_sm_compl, valid_sm_compl_ligclus, valid_sm_compl_strict, 
            valid_sm, valid_pep, homo
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

        if self.dataset_param['n_valid_pdb'] is None: self.dataset_param["n_valid_pdb"] = len(valid_pdb.keys()) 
        if self.dataset_param['n_valid_homo'] is None: self.dataset_param["n_valid_homo"] = len(valid_homo.keys()) 
        if self.dataset_param["n_valid_compl"] is None: self.dataset_param["n_valid_compl"] = len(valid_compl.keys())
        if self.dataset_param["n_valid_na_compl"] is None: self.dataset_param["n_valid_na_compl"] = len(valid_na_compl.keys())
        if self.dataset_param["n_valid_rna"] is None: self.dataset_param["n_valid_rna"] = len(valid_rna.keys())
        if self.dataset_param["n_valid_sm_compl"] is None: self.dataset_param["n_valid_sm_compl"] = len(valid_sm_compl.keys())
        if self.dataset_param["n_valid_sm_compl_ligclus"] is None: self.dataset_param["n_valid_sm_compl_ligclus"] = len(valid_sm_compl_ligclus.keys())
        if self.dataset_param["n_valid_sm_compl_strict"] is None: self.dataset_param["n_valid_sm_compl_strict"] = len(valid_sm_compl_strict.keys())
        if self.dataset_param["n_valid_sm"] is None: self.dataset_param["n_valid_sm"] = len(valid_sm.keys())

        if (rank==0):
            print ('Loaded (training)',
                len(pdb_IDs),'monomers/homomers,',
                len(fb_IDs),'distilled monomers,',
                len(compl_IDs),'heteromers,',
                len(neg_IDs),'negative heteromers,',
                len(na_compl_IDs),'nucleic-acid complexes,',
                len(na_neg_IDs),'negative nucleic-acid complexes,',
                len(rna_IDs),'RNA structures,',
                len(sm_compl_IDs), 'small molecule complexes, and',
                len(sm_IDs), "small molecule crystals."
            )
            print ('Loaded (valid)',
                len(valid_pdb.keys()),'monomers,',
                len(valid_homo.keys()),'homomers,',
                len(valid_compl.keys()),'heteromers,',
                len(valid_neg.keys()),'negative heteromers,',
                len(valid_na_compl.keys()),'nucleic-acid complexes,',
                len(valid_na_neg.keys()),'negative nucleic-acid complexes,',
                len(valid_rna),'RNA structures,',
                len(valid_sm_compl), 'small molecule complexes,',
                len(valid_sm_compl_ligclus), 'small molecule complexes (ligand-clustered),',
                len(valid_sm_compl_strict), 'small molecule complexes (strict),',
                len(valid_sm), 'small molecule crystals.'

            )
            print ('Using',
                self.dataset_param['n_valid_pdb'],'monomers,',
                self.dataset_param['n_valid_homo'],'homomers,',
                self.dataset_param['n_valid_compl'],'heteromers,',
                self.dataset_param['n_valid_neg'],'negative heteromers,',
                self.dataset_param['n_valid_na_compl'],'nucleic-acid complexes,',
                self.dataset_param['n_valid_na_neg'],'negative nucleic-acid complexes,',
                self.dataset_param['n_valid_rna'],'RNA structures,',
                self.dataset_param['n_valid_sm_compl'], 'small mol. complexes (fold & dock),',
                self.dataset_param['n_valid_sm_compl_ligclus'], 'small molecule complexes (ligand-clustered),',
                self.dataset_param['n_valid_sm_compl_strict'], 'small molecule complexes (strict),',
                self.dataset_param['n_valid_sm'], "small molecule crystals,",
                self.dataset_param['n_valid_atomize_pdb'],'monomers (atomized)',
            )

        train_set = DistilledDataset(
            pdb_IDs, loader_pdb, pdb_dict,
            compl_IDs, loader_complex, compl_dict,
            #neg_IDs, loader_complex, neg_dict,
            na_compl_IDs, loader_na_complex, na_compl_dict,
            #na_neg_IDs, loader_na_complex, na_neg_dict,
            fb_IDs, loader_fb, fb_dict,
            rna_IDs, loader_rna, rna_dict,
            sm_compl_IDs, loader_sm_compl, sm_compl_dict,
            sm_IDs, loader_sm, sm_dict,
            pdb_IDs, loader_atomize_pdb, pdb_dict,
            homo, 
            self.loader_param,
            native_NA_frac=0.25
        )

        valid_pdb_set = Dataset(
            list(valid_pdb.keys())[:self.dataset_param['n_valid_pdb']],
            loader_pdb, valid_pdb,
            self.loader_param, homo, p_homo_cut=-1.0
        )
        valid_homo_set = Dataset(
            list(valid_homo.keys())[:self.dataset_param['n_valid_homo']],
            loader_pdb, valid_homo,
            self.loader_param, homo, p_homo_cut=2.0
        )
        valid_compl_set = DatasetComplex(
            list(valid_compl.keys())[:self.dataset_param['n_valid_compl']],
            loader_complex, valid_compl,
            self.loader_param, negative=False
        )
        valid_na_compl_set = DatasetNAComplex(
            list(valid_na_compl.keys())[:self.dataset_param['n_valid_na_compl']],
            loader_na_complex, valid_na_compl,
            self.loader_param, negative=False, native_NA_frac=1.0
        )
        valid_na_from_scratch_compl_set = DatasetNAComplex(
            list(valid_na_compl.keys())[:self.dataset_param['n_valid_na_compl']],
            loader_na_complex, valid_na_compl,
            self.loader_param, negative=False, native_NA_frac=0.0
        )
        valid_rna_set = DatasetRNA(
            list(valid_rna.keys())[:self.dataset_param['n_valid_rna']],
            loader_rna, valid_rna,
            self.loader_param
        )
        valid_sm_compl_set = DatasetSMComplex(
            list(valid_sm_compl.keys())[:self.dataset_param['n_valid_sm_compl']],
            loader_sm_compl, valid_sm_compl,
            self.loader_param,
        )
        valid_sm_compl_ligclus_set = DatasetSMComplex(
            list(valid_sm_compl_ligclus.keys())[:self.dataset_param['n_valid_sm_compl_ligclus']],
            loader_sm_compl, valid_sm_compl_ligclus,
            self.loader_param, task='sm_compl_ligclus'
        )
        valid_sm_compl_strict_set = DatasetSMComplex(
            list(valid_sm_compl_strict.keys())[:self.dataset_param['n_valid_sm_compl_strict']],
            loader_sm_compl, valid_sm_compl_strict,
            self.loader_param, task='sm_compl_strict'
        )
        valid_sm_set = DatasetSM(
            list(valid_sm.keys())[:self.dataset_param['n_valid_sm']],
            loader_sm, valid_sm,
            self.loader_param,
        )
        valid_atomize_pdb_set = Dataset(
            list(valid_pdb.keys())[:self.dataset_param['n_valid_atomize_pdb']],
            loader_atomize_pdb, valid_pdb,
            self.loader_param, homo, p_homo_cut=-1.0, n_res_atomize=3, flank=0
        )
        # valid_neg_set = DatasetComplex(
        #     list(valid_neg.keys())[:self.n_valid_neg],
        #     loader_complex, valid_neg,
        #     self.loader_param, negative=True
        # )
#        valid_na_neg_set = DatasetNAComplex(
#            list(valid_na_neg.keys())[:self.n_valid_na_neg],
#            loader_na_complex, valid_na_neg,
#            self.loader_param, negative=True, native_NA_frac=1.0
#        )
#        valid_na_from_scratch_neg_set = DatasetNAComplex(
#            list(valid_na_neg.keys())[:self.n_valid_na_neg],
#            loader_na_complex, valid_na_neg,
#            self.loader_param, negative=True, native_NA_frac=0.0
#        )
        #valid_sm_compl_dock_set = DatasetSMComplex(
        #    list(valid_sm_compl.keys())[:self.n_valid_sm_compl],
        #    loader_sm_compl, valid_sm_compl,
        #    self.loader_param, init_protein_tmpl=True, init_ligand_tmpl=True, 
        #)
        #valid_sm_compl_foldprot_set = DatasetSMComplex(
        #    list(valid_sm_compl.keys())[:self.n_valid_sm_compl],
        #    loader_sm_compl, valid_sm_compl,
        #    self.loader_param, init_protein_tmpl=False, init_ligand_tmpl=True, 
        #)
        #valid_sm_compl_foldlig_set = DatasetSMComplex(
        #    list(valid_sm_compl.keys())[:self.n_valid_sm_compl],
        #    loader_sm_compl, valid_sm_compl,
        #    self.loader_param, init_protein_tmpl=True, init_ligand_tmpl=False, 
        #)
        #valid_sm_set = DatasetSM(
        #    list(valid_sm_compl.keys())[:self.n_valid_sm_compl],
        #    loader_small_molecule, valid_sm_compl, self.loader_param
        #)
        
        train_sampler = DistributedWeightedSampler(
            train_set, 
            pdb_weights,
            fb_weights,
            compl_weights,
            #neg_weights,
            na_compl_weights,
            #na_neg_weights,
            rna_weights,
            sm_compl_weights,
            sm_weights, 
            pdb_weights, # for atomize pdb
            num_example_per_epoch=self.dataset_param['n_train'],
            num_replicas=world_size, 
            rank=rank, 
            fraction_pdb=self.dataset_param['fraction_pdb'],
            fraction_fb=self.dataset_param['fraction_fb'],
            fraction_compl=self.dataset_param['fraction_compl'],
            fraction_na_compl=self.dataset_param['fraction_na_compl'],
            fraction_rna=self.dataset_param['fraction_rna'],
            fraction_sm_compl=self.dataset_param['fraction_sm_compl'],
            fraction_sm=self.dataset_param['fraction_sm'], 
            fraction_atomize_pdb=self.dataset_param['fraction_atomize_pdb'], 
            replacement=True
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
        valid_atomize_pdb_sampler = data.distributed.DistributedSampler(valid_atomize_pdb_set, num_replicas=world_size, rank=rank)
        # valid_neg_sampler = data.distributed.DistributedSampler(valid_neg_set, num_replicas=world_size, rank=rank)
#        valid_na_neg_sampler = data.distributed.DistributedSampler(valid_na_neg_set, num_replicas=world_size, rank=rank)
#        valid_na_from_scratch_neg_sampler = data.distributed.DistributedSampler(valid_na_from_scratch_neg_set, num_replicas=world_size, rank=rank)


        train_loader = data.DataLoader(train_set, sampler=train_sampler, batch_size=self.batch_size, **LOAD_PARAM)
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
        
        valid_atomize_pdb_loader = data.DataLoader(valid_atomize_pdb_set, sampler=valid_atomize_pdb_sampler, **LOAD_PARAM)
        # valid_neg_loader = data.DataLoader(valid_neg_set, sampler=valid_neg_sampler, **LOAD_PARAM)
#        valid_na_neg_loader = data.DataLoader(valid_na_neg_set, sampler=valid_na_neg_sampler, **LOAD_PARAM)
#       valid_na_from_scratch_neg_loader = data.DataLoader(valid_na_from_scratch_neg_set, sampler=valid_na_from_scratch_neg_sampler, **LOAD_PARAM)
        #valid_sm_compl_dock_loader = data.DataLoader(valid_sm_compl_dock_set, sampler=valid_sm_compl_sampler, **LOAD_PARAM)
        #valid_sm_compl_foldprot_loader = data.DataLoader(valid_sm_compl_foldprot_set, sampler=valid_sm_compl_sampler, **LOAD_PARAM)
        #valid_sm_compl_foldlig_loader = data.DataLoader(valid_sm_compl_foldlig_set, sampler=valid_sm_compl_sampler, **LOAD_PARAM)

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
        loaded_epoch, best_valid_loss = self.load_model(ddp_model, self.model_name, gpu, suffix="last", 
                                                        resume_train=True, optimizer=optimizer, 
                                                        scheduler=scheduler, scaler=scaler)
        if loaded_epoch >= self.n_epoch:
            DDP_cleanup()
            return

        for epoch in range(loaded_epoch+1, self.n_epoch):
            train_sampler.set_epoch(epoch)
            valid_pdb_sampler.set_epoch(epoch)
            valid_homo_sampler.set_epoch(epoch)
            valid_compl_sampler.set_epoch(epoch)
            valid_na_compl_sampler.set_epoch(epoch)
            valid_na_from_scratch_compl_sampler.set_epoch(epoch)
            valid_rna_sampler.set_epoch(epoch)
            valid_sm_compl_sampler.set_epoch(epoch)
            valid_sm_compl_ligclus_sampler.set_epoch(epoch)
            valid_sm_compl_strict_sampler.set_epoch(epoch)
            valid_sm_sampler.set_epoch(epoch)
            valid_atomize_pdb_sampler.set_epoch(epoch)
            #valid_neg_sampler.set_epoch(epoch)

            #print('epoch',epoch,'world_size',world_size,'rank',rank)
            rng = np.random.RandomState(seed=epoch*world_size+rank)

            train_tot, train_loss, train_acc = self.train_cycle(ddp_model, train_loader, optimizer, scheduler, scaler, rank, gpu, world_size, epoch, rng)

            _, _, _, _ = self.valid_pdb_cycle(ddp_model, valid_pdb_loader, rank, gpu, world_size, 
                epoch, rng, verbose = self.eval)
            _, _, _, _ = self.valid_pdb_cycle(ddp_model, valid_homo_loader, rank, gpu, world_size, 
                epoch, rng, header="Homo", verbose = self.eval)
            _, _, _, _ = self.valid_pdb_cycle(ddp_model, valid_compl_loader, rank, gpu, world_size, 
                epoch, rng, header="Hetero", verbose = self.eval)
            _, _, _, _ = self.valid_pdb_cycle(ddp_model, valid_na_compl_loader, rank, gpu, 
                world_size, epoch, rng, header="NA", verbose = self.eval)
            _, _, _, _ = self.valid_pdb_cycle(ddp_model, valid_na_from_scratch_compl_loader, rank, gpu, 
                world_size, epoch, rng, header="NAfs", verbose = self.eval)
            _, _, _, _ = self.valid_pdb_cycle(ddp_model, valid_rna_loader, rank, gpu, world_size, 
                epoch, rng, header="RNA", verbose = self.eval)
            valid_tot, valid_loss, valid_acc, _ = self.valid_pdb_cycle(ddp_model, 
                valid_sm_compl_loader, rank, gpu, world_size, epoch, rng, header="SM Compl", 
                verbose = self.eval) 
            _, _, _, _ = self.valid_pdb_cycle(ddp_model, valid_sm_compl_ligclus_loader, 
                rank, gpu, world_size, epoch, rng, header="SM Compl (lig. clus.)", verbose = self.eval) 
            _, _, _, _ = self.valid_pdb_cycle(ddp_model, valid_sm_compl_strict_loader, 
                rank, gpu, world_size, epoch, rng, header="SM Compl (strict)", verbose = self.eval) 
            _, _, _, _ = self.valid_pdb_cycle(ddp_model, valid_sm_loader, 
                rank, gpu, world_size, epoch, rng, header="SM_CSD", verbose = self.eval) 
            _, _, _, _ = self.valid_pdb_cycle(ddp_model, valid_atomize_pdb_loader, rank, gpu, world_size, 
                epoch, rng, header='Monomer atomize 3', verbose = self.eval)
            #_, _, _ = self.valid_ppi_cycle(ddp_model, valid_compl_loader, valid_neg_loader, rank, gpu, 
            #    world_size, epoch, report_interface=True)
#            _, _, _ = self.valid_ppi_cycle(
#                ddp_model, valid_na_compl_loader, valid_na_neg_loader, 
#                rank, gpu, world_size, epoch, header="NA", report_interface=False)
#            _, _, _ = self.valid_ppi_cycle(
#                ddp_model, valid_na_from_scratch_compl_loader, valid_na_from_scratch_neg_loader, 
#                rank, gpu, world_size, epoch, header="NAfs", report_interface=False)

            if self.eval: break

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
                    if self.wandb_prefix is not None:
                        wandb.save(self.checkpoint_fn(self.model_name, 'best'))

                chk_fn = self.checkpoint_fn(self.model_name, str(epoch))
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
                            chk_fn)
                shutil.copy(chk_fn, self.checkpoint_fn(self.model_name, 'last'))

                if self.wandb_prefix is not None:
                    wandb.save(self.checkpoint_fn(self.model_name, str(epoch)))
        dist.destroy_process_group()

    def train_cycle(self, ddp_model, train_loader, optimizer, scheduler, scaler, rank, gpu, world_size, epoch, rng, verbose=False):
        # Turn on training mode
        ddp_model.train()
        
        # clear gradients
        optimizer.zero_grad()

        start_time = time.time()
        
        # save intermediate outputs
        out_dir = self.out_dir+f'/train_ep{epoch}/'
        os.makedirs(out_dir, exist_ok=True)

        # For intermediate logs
        local_tot = 0.0
        local_loss = None
        local_acc = None
        train_tot = 0.0
        train_loss = None
        train_acc = None

        counter = 0
        
        for seq, msa, msa_masked, msa_full, mask_msa, true_crds, atom_mask, idx_pdb, xyz_t, t1d, mask_t, xyz_prev, mask_prev, same_chain, unclamp, negative, atom_frames, bond_feats, chirals, task, item in train_loader:
            # skip known bad training examples
            # loader will print warning message with item info for followup later
            if torch.is_tensor(item) and torch.all(item==-1):
                continue

            r = rng.rand()
            save_pdbs = r<=0.01
            #print('rank',rank, 'counter',counter, 'item',item, 'task',task, 'save_pdbs',save_pdbs, 'r',r)
            
            # transfer inputs to device
            B, _, N, L = msa.shape
            idx_pdb = idx_pdb.to(gpu, non_blocking=True) # (B, L)
            true_crds = true_crds.to(gpu, non_blocking=True) # (B, N?, L, Natms, 3)
            atom_mask = atom_mask.to(gpu, non_blocking=True) # (B, L, Natms)
            same_chain = same_chain.to(gpu, non_blocking=True) # (B, L, L)

            xyz_t = xyz_t.to(gpu, non_blocking=True)
            t1d = t1d.to(gpu, non_blocking=True)
            mask_t = mask_t.to(gpu, non_blocking=True)
            xyz_prev = xyz_prev.to(gpu, non_blocking=True)
            mask_prev = mask_prev.to(gpu, non_blocking=True)
            xyz_prev_orig = xyz_prev.clone()

            seq = seq.to(gpu, non_blocking=True)
            msa = msa.to(gpu, non_blocking=True)
            msa_masked = msa_masked.to(gpu, non_blocking=True)
            msa_full = msa_full.to(gpu, non_blocking=True)
            mask_msa = mask_msa.to(gpu, non_blocking=True)
            atom_frames = atom_frames.to(gpu, non_blocking=True)
            bond_feats = bond_feats.to(gpu, non_blocking=True)
            chirals = chirals.to(gpu, non_blocking=True)
            
            # template masking
            seq_unmasked = msa[:, 0, 0, :] # (B, L)
            mask_t_2d = get_prot_sm_mask(mask_t, seq_unmasked[0]) # (B, T, L)
            mask_t_2d = mask_t_2d[:,:,None]*mask_t_2d[:,:,:,None] # (B, T, L, L)
            mask_t_2d = mask_t_2d.float() * same_chain.float()[:,None] # (ignore inter-chain region)

            mask_recycle = get_prot_sm_mask(mask_prev, seq_unmasked[0])
            mask_recycle = mask_recycle[:,:,None]*mask_recycle[:,None,:] # (B, L, L)
            mask_recycle = same_chain.float()*mask_recycle.float()

            # processing template features
            xyz_t_frames = xyz_t_to_frame_xyz(xyz_t, seq_unmasked, atom_frames)
            t2d = xyz_to_t2d(xyz_t_frames, mask_t_2d)

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

            N_cycle = np.random.randint(1, self.maxcycle+1) # number of recycling

            msa_prev = None
            pair_prev = None
            alpha_prev = torch.zeros((B,L,NTOTALDOFS,2)).to(gpu, non_blocking=True)
            state_prev = None

            with torch.no_grad():
                for i_cycle in range(N_cycle-1):
                    with ddp_model.no_sync():
                        with torch.cuda.amp.autocast(enabled=USE_AMP):
                            msa_prev, pair_prev, xyz_prev, state_prev, alpha_prev, mask_recycle = ddp_model(
                                msa_masked[:,i_cycle],
                                msa_full[:,i_cycle],
                                seq[:,i_cycle],
                                msa[:,i_cycle,0], # unmasked seq
                                xyz_prev,
                                alpha_prev,
                                idx_pdb,
                                bond_feats,
                                chirals,
                                atom_frames=atom_frames,
                                t1d=t1d,
                                t2d=t2d,
                                xyz_t=xyz_t[...,1,:],
                                alpha_t=alpha_t,
                                mask_t=mask_t_2d,
                                same_chain=same_chain,
                                msa_prev=msa_prev,
                                pair_prev=pair_prev,
                                state_prev=state_prev,
                                mask_recycle=mask_recycle,
                                return_raw=True,
                                use_checkpoint=False
                            )

            i_cycle = N_cycle-1

            if counter%self.ACCUM_STEP != 0:
                with ddp_model.no_sync():
                    with torch.cuda.amp.autocast(enabled=USE_AMP):
                        logit_s, logit_aa_s, logit_pae, logit_pde, pred_crds, alphas, pred_allatom, pred_lddts, _, _, _ = ddp_model(
                            msa_masked[:,i_cycle],
                            msa_full[:,i_cycle],
                            seq[:,i_cycle],
                            msa[:,i_cycle,0], # unmasked seq
                            xyz_prev,
                            alpha_prev,
                            idx_pdb,
                            bond_feats,
                            chirals,
                            atom_frames=atom_frames,
                            t1d=t1d,
                            t2d=t2d,
                            xyz_t=xyz_t[...,1,:],
                            alpha_t=alpha_t,
                            mask_t=mask_t_2d,
                            same_chain=same_chain,
                            msa_prev=msa_prev,
                            pair_prev=pair_prev,
                            state_prev=state_prev,
                            mask_recycle=mask_recycle,
                            use_checkpoint=True
                        )

                        true_crds_, atom_mask_ = resolve_equiv_natives(pred_crds[-1], true_crds, atom_mask)
                        res_mask = ~((atom_mask_[:,:,:3].sum(dim=-1) < 3.0) * ~(is_atom(msa[:,i_cycle,0])))
                        mask_2d = res_mask[:,None,:] * res_mask[:,:,None]

                        true_crds_frame = xyz_to_frame_xyz(true_crds_, msa[:, i_cycle, 0],atom_frames)
                        c6d = xyz_to_c6d(true_crds_frame)
                        c6d = c6d_to_bins(c6d, same_chain, negative=negative)

                        prob = self.active_fn(logit_s[0]) # distogram
                        acc_s = self.calc_acc(prob, c6d[...,0], idx_pdb, mask_2d)

                        ctrid = len(train_loader)*rank+counter
                        loss, loss_dict = self.calc_loss(
                            logit_s, c6d,
                            logit_aa_s, msa[:, i_cycle], mask_msa[:,i_cycle], logit_pae, logit_pde,
                            pred_crds, alphas, pred_allatom, true_crds_, 
                            atom_mask_, res_mask, mask_2d, same_chain,
                            pred_lddts, idx_pdb, bond_feats, atom_frames=atom_frames,
                            unclamp=unclamp, negative=negative,
                            verbose=verbose, ctr=ctrid, item=item, task=task, **self.loss_param
                        )
                    loss = loss / self.ACCUM_STEP
                    scaler.scale(loss).backward()
            else:
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    logit_s, logit_aa_s, logit_pae, logit_pde, pred_crds, alphas, pred_allatom, pred_lddts, _, _, _ = ddp_model(
                        msa_masked[:,i_cycle],
                        msa_full[:,i_cycle],
                        seq[:,i_cycle],
                        msa[:,i_cycle,0], # unmasked seq
                        xyz_prev,
                        alpha_prev,
                        idx_pdb,
                        bond_feats,
                        chirals,
                        atom_frames=atom_frames,
                        t1d=t1d,
                        t2d=t2d,
                        xyz_t=xyz_t[...,1,:],
                        alpha_t=alpha_t,
                        mask_t=mask_t_2d,
                        same_chain=same_chain,
                        msa_prev=msa_prev,
                        pair_prev=pair_prev,
                        state_prev=state_prev,
                        mask_recycle=mask_recycle,
                        use_checkpoint=True
                    )

                    true_crds_, atom_mask_ = resolve_equiv_natives(pred_crds[-1], true_crds, atom_mask)

                    res_mask = ~((atom_mask_[:,:,:3].sum(dim=-1) < 3.0) * ~(is_atom(msa[:,i_cycle,0])))
                    mask_2d = res_mask[:,None,:] * res_mask[:,:,None]

                    true_crds_frame = xyz_to_frame_xyz(true_crds_, msa[:, i_cycle, 0],atom_frames)
                    c6d = xyz_to_c6d(true_crds_frame)
                    c6d = c6d_to_bins(c6d, same_chain, negative=negative)

                    prob = self.active_fn(logit_s[0]) # distogram
                    acc_s = self.calc_acc(prob, c6d[...,0], idx_pdb, mask_2d)

                    ctrid = len(train_loader)*rank+counter
                    loss, loss_dict = self.calc_loss(
                        logit_s, c6d,
                        logit_aa_s, msa[:, i_cycle], mask_msa[:,i_cycle], logit_pae, logit_pde,
                        pred_crds, alphas, pred_allatom, true_crds_, 
                        atom_mask_, res_mask, mask_2d, same_chain,
                        pred_lddts, idx_pdb, bond_feats, atom_frames=atom_frames,
                        unclamp=unclamp, negative=negative,
                        verbose=verbose, ctr=ctrid, item=item, task=task, **self.loss_param
                    )
                loss = loss / self.ACCUM_STEP
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer) # gradient clipping

                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 0.2)

                scaler.step(optimizer)
                scale = scaler.get_scale()
                scaler.update()
                skip_lr_sched = (scale != scaler.get_scale())
                optimizer.zero_grad()
                if not skip_lr_sched:
                    scheduler.step()
                ddp_model.module.update() # apply EMA

            if torch.isnan(loss):
                print('nan loss',item)
                save_pdbs = True

            if task[0].startswith('sm_compl'):
                name = item[2][0][0].replace('.mol2','')
            elif task[0]=='sm_only':
                name = item[0]
            else:
                name = item[0][0]

            if save_pdbs:
                writepdb(out_dir+f'ep{epoch}_{task[0]}_{counter}.{rank}_{name}_xyz_prev.pdb', 
                    torch.nan_to_num(xyz_prev_orig[res_mask][:,:23]), seq_unmasked[res_mask],
                    bond_feats=bond_feats[:, res_mask[0]][:, :, res_mask[0]])
                writepdb(out_dir+f'ep{epoch}_{task[0]}_{counter}.{rank}_{name}_xyz_true.pdb', 
                    torch.nan_to_num(true_crds_[res_mask][:,:23]), seq_unmasked[res_mask], 
                    bond_feats=bond_feats[:, res_mask[0]][:, :, res_mask[0]])
                writepdb(out_dir+f'ep{epoch}_{task[0]}_{counter}.{rank}_{name}_xyz_pred.pdb', 
                    torch.nan_to_num(pred_allatom[res_mask][:,:23]), seq_unmasked[res_mask], 
                    bond_feats=bond_feats[:, res_mask[0]][:, :, res_mask[0]])

            local_tot += loss.detach()*self.ACCUM_STEP
            if local_loss is None:
                local_loss = torch.zeros_like(torch.stack(list(loss_dict.values())))
                local_acc = torch.zeros_like(acc_s.detach())
            local_loss += torch.stack(list(loss_dict.values()))
            local_acc += acc_s.detach()
            
            train_tot += loss.detach()*self.ACCUM_STEP
            if train_loss is None:
                train_loss = torch.zeros_like(torch.stack(list(loss_dict.values())))
                train_acc = torch.zeros_like(acc_s.detach())
            train_loss += torch.stack(list(loss_dict.values()))
            train_acc += acc_s.detach()

            # print loss names once at beginning of epoch
            if counter == 1 and rank == 0:
                sys.stdout.write(f'Header: [epoch/num_epochs] Batch: [examples_seen_in_epoch/examples_per_epoch] Time: time | Total_loss: total_loss | {" ".join(loss_dict.keys())} | precision recall F1 | max_mem \n')
            
            if counter % self.ACCUM_STEP == 0:
                if rank == 0:
                    max_mem = torch.cuda.max_memory_allocated()/1e9
                    train_time = time.time() - start_time
                    local_tot /= float(self.ACCUM_STEP)
                    local_loss /= float(self.ACCUM_STEP)
                    local_acc /= float(self.ACCUM_STEP)
                    #local_tot /= float(N_PRINT_TRAIN)
                    #local_loss /= float(N_PRINT_TRAIN)
                    #local_acc /= float(N_PRINT_TRAIN)
                    
                    local_tot = local_tot.cpu().detach().numpy()
                    local_loss = local_loss.cpu().detach().numpy()
                    local_acc = local_acc.cpu().detach().numpy()

                    sys.stdout.write("Local: [%04d/%04d] Batch: [%05d/%05d] Time: %16.1f | total_loss: %8.4f | %s | %.4f %.4f %.4f | Max mem %.4f\n"%(\
                            epoch, self.n_epoch, counter*self.batch_size*world_size, \
                            self.dataset_param['n_train'], train_time, local_tot, \
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
        train_tot = train_tot.cpu().detach().numpy()
        train_loss = train_loss.cpu().detach().numpy()
        train_acc = train_acc.cpu().detach().numpy()

        if rank == 0:
            train_time = time.time() - start_time
            sys.stdout.write("Train: [%04d/%04d] Batch: [%05d/%05d] Time: %16.1f | total_loss: %8.4f | %s | %.4f %.4f %.4f\n"%(\
                    epoch, self.n_epoch, self.dataset_param['n_train'], self.dataset_param['n_train'], \
                    train_time, train_tot, \
                    " ".join(["%8.4f"%l for l in train_loss]),\
                    train_acc[0], train_acc[1], train_acc[2]))
            sys.stdout.flush()
            
        return train_tot, train_loss, train_acc

    def valid_pdb_cycle(self, ddp_model, valid_loader, rank, gpu, world_size, epoch, rng, header='Monomer', verbose=False, print_header=False):
        if len(valid_loader) == 0:
            return None, None, None, None

        valid_tot = 0.0
        valid_loss = None
        valid_acc = None
        counter = 0
        
        start_time = time.time()

        out_dir = self.out_dir+f'/valid_ep{epoch}{"_eval" if verbose else ""}/'
        os.makedirs(out_dir, exist_ok=True)

        if self.eval:
            records = []
        
        with torch.no_grad(): # no need to calculate gradient
            ddp_model.eval() # change it to eval mode
            for seq, msa, msa_masked, msa_full, mask_msa, true_crds, atom_mask, idx_pdb, xyz_t, t1d, mask_t, xyz_prev, mask_prev, same_chain, unclamp, negative, atom_frames, bond_feats, chirals, task, item in valid_loader:
                # skip known bad training examples
                if torch.is_tensor(item) and torch.all(item==-1):
                    continue

                r = rng.rand()
                save_pdbs = r<=0.1 or self.eval
                #print('rank',rank, 'counter',counter, 'item',item, 'task',task, 'save_pdbs',save_pdbs, 'r',r)

                # transfer inputs to device
                B, _, N, L = msa.shape

                idx_pdb = idx_pdb.to(gpu, non_blocking=True) # (B, L)
                true_crds = true_crds.to(gpu, non_blocking=True) # (B, L, 27, 3)
                atom_mask = atom_mask.to(gpu, non_blocking=True) # (B, L, 27)
                same_chain = same_chain.to(gpu, non_blocking=True)

                xyz_t = xyz_t.to(gpu, non_blocking=True)
                t1d = t1d.to(gpu, non_blocking=True)
                mask_t = mask_t.to(gpu, non_blocking=True)
                xyz_prev = xyz_prev.to(gpu, non_blocking=True)
                mask_prev = mask_prev.to(gpu, non_blocking=True)
                xyz_prev_orig = xyz_prev.clone()

                seq = seq.to(gpu, non_blocking=True)
                msa = msa.to(gpu, non_blocking=True)
                msa_masked = msa_masked.to(gpu, non_blocking=True)
                msa_full = msa_full.to(gpu, non_blocking=True)
                mask_msa = mask_msa.to(gpu, non_blocking=True)
                atom_frames = atom_frames.to(gpu, non_blocking=True)
                bond_feats = bond_feats.to(gpu, non_blocking=True)
                chirals = chirals.to(gpu, non_blocking=True)

                # template masking
                seq_unmasked = msa[:, 0, 0, :] # (B, L)
                mask_t_2d = get_prot_sm_mask(mask_t, seq_unmasked[0]) # (B, T, L)
                mask_t_2d = mask_t_2d[:,:,None]*mask_t_2d[:,:,:,None] # (B, T, L, L)
                mask_t_2d = mask_t_2d.float() * same_chain.float()[:,None] # (ignore inter-chain region)

                mask_recycle = get_prot_sm_mask(mask_prev, seq_unmasked[0])
                mask_recycle = mask_recycle[:,:,None]*mask_recycle[:,None,:] # (B, L, L)
                mask_recycle = same_chain.float()*mask_recycle.float()

                # processing template features
                xyz_t_frames = xyz_t_to_frame_xyz(xyz_t, seq_unmasked, atom_frames)
                t2d = xyz_to_t2d(xyz_t_frames, mask_t_2d)

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
                    msa_prev, pair_prev, xyz_prev, state_prev, alpha_prev, mask_recycle = ddp_model(
                        msa_masked[:,i_cycle],
                        msa_full[:,i_cycle],
                        seq[:,i_cycle],
                        msa[:,i_cycle,0], # unmasked seq
                        xyz_prev,
                        alpha_prev,
                        idx_pdb,
                        bond_feats,
                        chirals,
                        atom_frames=atom_frames,
                        t1d=t1d,
                        t2d=t2d,
                        xyz_t=xyz_t[...,1,:],
                        alpha_t=alpha_t,
                        mask_t=mask_t_2d,
                        same_chain=same_chain,
                        msa_prev=msa_prev,
                        pair_prev=pair_prev,
                        state_prev=state_prev,
                        mask_recycle=mask_recycle,
                        return_raw=True,
                        use_checkpoint=False
                    )

                i_cycle = N_cycle-1
                logit_s, logit_aa_s, logit_pae, logit_pde, pred_crds, alphas, pred_allatom, pred_lddts, _, _, _ = ddp_model(
                    msa_masked[:,i_cycle],
                    msa_full[:,i_cycle],
                    seq[:,i_cycle],
                    msa[:,i_cycle,0], # unmasked seq
                    xyz_prev,
                    alpha_prev,
                    idx_pdb,
                    bond_feats,
                    chirals,
                    atom_frames=atom_frames,
                    t1d=t1d,
                    t2d=t2d,
                    xyz_t=xyz_t[...,1,:],
                    alpha_t=alpha_t,
                    mask_t=mask_t_2d,
                    same_chain=same_chain,
                    msa_prev=msa_prev,
                    pair_prev=pair_prev,
                    state_prev=state_prev,
                    mask_recycle=mask_recycle,
                    use_checkpoint=False
                )

                true_crds_, atom_mask_ = resolve_equiv_natives(pred_crds[-1], true_crds, atom_mask)

                res_mask = ~((atom_mask_[:,:,:3].sum(dim=-1) < 3.0) * ~(is_atom(msa[:,i_cycle,0])))
                mask_2d = res_mask[:,None,:] * res_mask[:,:,None]

                true_crds_frame = xyz_to_frame_xyz(true_crds_, msa[:, i_cycle, 0],atom_frames)
                c6d = xyz_to_c6d(true_crds_frame)
                c6d = c6d_to_bins(c6d, same_chain, negative=negative)

                prob = self.active_fn(logit_s[0]) # distogram
                acc_s = self.calc_acc(prob, c6d[...,0], idx_pdb, mask_2d)

                ctrid = len(valid_loader)*rank+counter
                loss, loss_dict = self.calc_loss(
                    logit_s, c6d,
                    logit_aa_s, msa[:, i_cycle], mask_msa[:,i_cycle], logit_pae, logit_pde,
                    pred_crds, alphas, pred_allatom, true_crds_, 
                    atom_mask_, res_mask, mask_2d, same_chain,
                    pred_lddts, idx_pdb, bond_feats, atom_frames, unclamp=unclamp, negative=negative,
                    verbose=verbose, ctr=ctrid, item=item, out_dir=out_dir, **self.loss_param
                )

                valid_tot += loss.detach()
                if valid_loss is None:
                    valid_loss = torch.zeros_like(torch.stack(list(loss_dict.values())))
                    valid_acc = torch.zeros_like(acc_s.detach())
                valid_loss += torch.stack(list(loss_dict.values()))
                valid_acc += acc_s.detach()

                # records results
                if task[0].startswith('sm_compl'):
                    name = item[2][0][0].replace('.mol2','')
                elif task[0]=='sm_only':
                    name = item[0]
                else:
                    name = item[0][0]
                    
                #print('in valid_pdb_cycle', 'save_pdbs=',save_pdbs, header, task[0], counter, name)
                if save_pdbs:
                    #writepdb(out_dir+f'ep{epoch}_{task[0]}_{counter}.{rank}_{name}_xyz_prev.pdb',
                    #    torch.nan_to_num(xyz_prev_orig[res_mask][:,:23]), seq_unmasked[res_mask], 
                    #    bond_feats=bond_feats[:,res_mask[0]][:,:,res_mask[0]])
                    writepdb(out_dir+f'ep{epoch}_{task[0]}_{counter}.{rank}_{name}.pdb',
                        torch.nan_to_num(true_crds_[res_mask][:,:23]), seq_unmasked[res_mask],
                        bond_feats=bond_feats[:,res_mask[0]][:,:,res_mask[0]],
                        chain="A", atom_mask=atom_mask_[res_mask])
                    pred_sup = superimpose(torch.nan_to_num(pred_allatom[:,res_mask[0],:23]),
                                           torch.nan_to_num(true_crds_[:,res_mask[0],:23]),
                                           atom_mask_[:,res_mask[0],:23])
                    writepdb(out_dir+f'ep{epoch}_{task[0]}_{counter}.{rank}_{name}.pdb',
                        pred_sup, seq_unmasked[res_mask],
                        bond_feats=bond_feats[:,res_mask[0]][:,:,res_mask[0]], 
                        chain="B", file_mode='a', atom_mask=atom_mask_[res_mask],
                        atom_idx_offset=atom_mask_[res_mask].sum())

                if self.eval:
                    record = OrderedDict(name = name, Header=header, task = task[0], epoch = epoch)
                    record.update({k:float(v) for k,v in loss_dict.items()})
                    logit_pae_ = logit_pae[...,res_mask[0]][...,res_mask[0],:] if logit_pae is not None else None
                    logit_pde_ = logit_pde[...,res_mask[0]][...,res_mask[0],:] if logit_pde is not None else None
                    pred_err = self.calc_pred_err(pred_lddts, logit_pae_, logit_pde_, 
                                                  seq_unmasked[0,res_mask[0]]) 
                    record.update(pred_err)
                    records.append(record)

                    torch.save({'logits_pae': logit_pae_,
                                'logits_pde': logit_pde_,
                                'pred_lddts': pred_lddts[...,res_mask[0]]},
                               out_dir+f'ep{epoch}_{task[0]}_{counter}.{rank}_{name}_outputs.pt')

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
        
        if self.eval:
            # gather per-example losses
            if rank == 0:
                all_records = [None]*world_size 
                dist.gather_object(records, all_records, dst=0)
            else:
                dist.gather_object(records, dst=0)

        loss_df = None

        if rank == 0:
            if self.wandb_prefix is not None:
                log_dict = {f"Valid_{header}":{task[0]:loss_dict}}
                wandb.log(log_dict)
            train_time = time.time() - start_time

            # print loss names
            if print_header:
                sys.stdout.write(f'Header: [epoch/num_epochs] Batch: [examples_seen_in_epoch/examples_per_epoch] Time: time | Total_loss: total_loss | {" ".join(loss_dict.keys())} | precision recall F1 | max_mem \n')

            sys.stdout.write("%s: [%04d/%04d] Batch: [%05d/%05d] Time: %16.1f | total_loss: %8.4f | %s | %.4f %.4f %.4f\n"%(\
                    header, epoch, self.n_epoch, world_size*len(valid_loader), world_size*len(valid_loader), train_time, valid_tot, \
                    " ".join(["%8.4f"%l for l in valid_loss]),\
                    valid_acc[0], valid_acc[1], valid_acc[2])) 
            sys.stdout.flush()

            if self.eval:
                # save per-example losses
                all_records_ = []
                for records in all_records:
                    all_records_.extend(records)
                loss_df = pd.DataFrame.from_records(all_records_)

        return valid_tot, valid_loss, valid_acc, loss_df

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
            for seq, msa, msa_masked, msa_full, mask_msa, true_crds, mask_crds, idx_pdb, xyz_t, t1d, xyz_prev, same_chain, unclamp, negative, atom_frames, bond_feats, chirals, task, item in valid_pos_loader:
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
                chirals = chirals.to(gpu, non_blocking=True)
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
                        chirals,
                        atom_frames=atom_frames,
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
                    chirals,
                    atom_frames=atom_frames,
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
                c6d = xyz_to_c6d(true_crds_frame)
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
                    pred_lddts, idx_pdb, bond_feats, atom_frames, unclamp=unclamp, negative=negative, interface=report_interface,
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
                        chirals,
                        atom_frames=atom_frames,
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
                    chirals,
                    atom_frames=atom_frames,
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
                c6d = xyz_to_c6d(true_crds_frame)
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
                    pred_lddts, idx_pdb, atom_frames, bond_feats, unclamp=unclamp, negative=negative,
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
    args, dataset_param, model_param, loader_param, loss_param = get_args()

    if int(os.environ["SLURM_PROCID"])==0:
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
                    interactive=args.interactive,
                    out_dir=args.out_dir,
                    wandb_prefix=args.wandb_prefix,
                    model_dir=args.model_dir,
                    dataset_param=dataset_param)
    train.run_model_training(torch.cuda.device_count())
