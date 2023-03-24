import sys, os, time, datetime, subprocess, shutil
from icecream import ic
ic.configureOutput(includeContext=True)

from contextlib import ExitStack, nullcontext

import numpy as np
import pandas as pd
from copy import deepcopy
from collections import OrderedDict
from icecream import ic
import wandb
import torch
import torch.nn as nn
from torch.utils import data
from functools import partial
from tqdm import tqdm
from symmetry import symm_subunit_matrix, find_symm_subs, get_symm_map

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(script_dir)

from rf2aa.data_loader import (
    get_train_valid_set, loader_pdb, loader_fb, loader_complex,
    loader_na_complex, loader_rna, loader_sm, loader_atomize_pdb,
    loader_sm_compl_assembly, loader_sm_compl_assembly_single, 
    Dataset, DatasetComplex, DatasetNAComplex,
    DatasetRNA, DatasetSM, DatasetSMComplex, DatasetSMComplexAssembly,
    DistilledDataset, DistributedWeightedSampler, unbatch_item
)
from rf2aa.kinematics import xyz_to_c6d, c6d_to_bins, xyz_to_t2d, xyz_to_bbtor
from rf2aa.RoseTTAFoldModel  import RoseTTAFoldModule
from rf2aa.loss import *
from rf2aa.util import *
from rf2aa.util_module import XYZConverter
from rf2aa.scheduler import get_linear_schedule_with_warmup, get_stepwise_decay_schedule_with_warmup

# disable openbabel warnings
from openbabel import openbabel as ob
ob.obErrorLog.SetOutputLevel(0)

# distributed data parallel
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
torch.autograd.set_detect_anomaly(True)
#torch.backends.cudnn.benchmark = False
#torch.backends.cudnn.deterministic = True

#os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # disable asynchronous execution

# limit thread counts
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'

## To reproduce errors
import random
random.seed(0)
torch.manual_seed(5924)
np.random.seed(6636)

USE_AMP = False
torch.set_num_threads(4)

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
                 model_dir='models/', dataloader_kwargs = {}, **kwargs):

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
        self.dataloader_kwargs = dataloader_kwargs
        self.write_every_n_steps_train = kwargs.get("write_every_n_steps_train", 64)
        self.write_every_n_steps_valid = kwargs.get("write_every_n_steps_valid", 64)
        self.debug_mode = kwargs.get("debug_mode", False)
        self.skip_valid = kwargs.get("skip_valid", 1)
        self.start_epoch = kwargs.get("start_epoch", 0)

        # for all-atom str loss
        #self.ti_dev = torsion_indices
        #self.ti_flip = torsion_can_flip
        #self.ang_ref = reference_angles
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
        self.xyz_converter = XYZConverter()

        # loss & final activation function
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.active_fn = nn.Softmax(dim=1)

        self.maxcycle = maxcycle

        self.pdb_counter=0
        
    def calc_loss(self, logit_s, label_s,
                  logit_aa_s, label_aa_s, mask_aa_s, logit_pae, logit_pde, p_bind,
                  pred, pred_tors, pred_allatom, true,
                  mask_crds, mask_BB, mask_2d, same_chain,
                  pred_lddt, idx, bond_feats, dist_matrix, atom_frames=None, unclamp=False, 
                  negative=False, interface=False,
                  verbose=False, ctr=0,
                  w_dist=1.0, w_aa=1.0, w_str=1.0, w_inter_fape=0.0, w_lig_fape=1.0, w_lddt=1.0, 
                  w_bond=1.0, w_clash=0.0, w_atom_bond=0.0, w_skip_bond=0.0, w_rigid=0.0, w_hb=0.0, w_bind=0.0,
                  w_pae=0.0, w_pde=0.0, lj_lin=0.85, eps=1e-6, binder_loss_label_smoothing = 0.0, item=None, task=None, out_dir='./'
    ):
        gpu = pred.device

        # track losses for printing to local log and uploading to WandB
        loss_dict = OrderedDict()

        B, L, natoms = true.shape[:3]
        seq = label_aa_s[:,0].clone()

        assert (B==1) # fd - code assumes a batch size of 1
        
        tot_loss = 0.0
        # set up frames
        frames, frame_mask = get_frames(
            pred_allatom[-1,None,...], mask_crds, seq, self.fi_dev, atom_frames)
        # update frames and frames_mask to only include BB frames (have to update both for compatibility with compute_general_FAPE)
        frames_BB = frames.clone()
        frames_BB[..., 1:, :, :] = 0
        frame_mask_BB = frame_mask.clone()
        frame_mask_BB[...,1:] =False
       
        # c6d loss
        for i in range(4):
            loss = self.loss_fn(logit_s[i], label_s[...,i]) # (B, L, L)
            if i==0: # apply distogram loss to all residue pairs with valid BB atoms
                mask_2d_ = mask_2d
            else: 
                # apply anglegram loss only when both residues have valid BB frames (i.e. not metal ions, and not examples with unresolved atoms in frames)
                _, bb_frame_good = mask_unresolved_frames(frames_BB, frame_mask_BB, mask_crds) # (1, L, nframes)
                bb_frame_good = bb_frame_good[...,0] # (1,L)
                loss_mask_2d = bb_frame_good & bb_frame_good[...,None]
                mask_2d_ = mask_2d & loss_mask_2d

            if negative.item():
                # Don't compute inter-chain distogram losses
                # for negative examples.
                mask_2d_ = mask_2d_ * same_chain
            loss = (mask_2d_*loss).sum() / (mask_2d_.sum() + eps)
            tot_loss += w_dist*loss
            loss_dict[f'c6d_{i}'] = loss.detach()

        # masked token prediction loss
        loss = self.loss_fn(logit_aa_s, label_aa_s.reshape(B, -1))
        loss = loss * mask_aa_s.reshape(B, -1)
        loss = loss.sum() / (mask_aa_s.sum() + 1e-8)
        tot_loss += w_aa*loss
        loss_dict['aa_cce'] = loss.detach()

        # col 4: binder loss
        # only apply binding loss to complexes
        # note that this will apply loss to positive sets w/o a corresponding negative set
        #   (e.g., homomers).  Maybe want to change this?
        if (torch.sum(same_chain==0) > 0):
            bce = torch.nn.BCELoss()
            target = torch.tensor(
                [abs(float(not negative) - binder_loss_label_smoothing)],
                device=p_bind.device
            )
            loss = bce(p_bind,target)
        else:
            # avoid unused parameter error
            loss = 0.0 * p_bind.sum()

        tot_loss += w_bind * loss
        loss_dict['binder_bce_loss'] = loss.detach()


        ### GENERAL LAYERS
        # Structural loss (layer-wise backbone FAPE)
        dclamp = 300.0 if unclamp else 30.0 # protein & NA FAPE distance cutoffs
        dclamp_sm, Z_sm = 4, 4  # sm mol FAPE distance cutoffs
        dclamp_prot = 10
        # residue mask for FAPE calculation only masks unresolved protein backbone atoms
        # whereas other losses also maks unresolved ligand atoms (mask_BB)
        # frames with unresolved ligand atoms are masked in compute_general_FAPE
        res_mask = ~((mask_crds[:,:,:3].sum(dim=-1) < 3.0) * ~(is_atom(seq)))
        
        # create 2d masks for intrachain and interchain fape calculations
        nframes = frame_mask.shape[-1]
        # frame_atom_mask_2d = torch.einsum('bfn,bra->bfnra',frame_mask_BB,mask_crds[:, :, :3]) # B, L, nframes, L, natoms
        frame_atom_mask_2d_allatom = torch.einsum('bfn,bra->bfnra', frame_mask_BB, mask_crds) # B, L, nframes, L, natoms
        frame_atom_mask_2d = frame_atom_mask_2d_allatom[:, :, :, :, :3]

        frame_atom_mask_2d_intra_allatom = frame_atom_mask_2d_allatom * same_chain[:, :,None, :, None].expand(-1,-1,nframes,-1, 36)
        frame_atom_mask_2d_intra = frame_atom_mask_2d_intra_allatom[:, :, :, :, :3]
        different_chain = ~same_chain.bool()
        frame_atom_mask_2d_inter = frame_atom_mask_2d*different_chain[:, :,None, :, None].expand(-1,-1,nframes,-1, 3) # B, L, nframes, L, natoms
            
        if negative: # inter-chain fapes should be ignored for negative cases

            if logit_pae is not None:
                logit_pae = logit_pae[:,:,res_mask[0]][:,:,:,res_mask[0]]
            if logit_pde is not None:
                logit_pde = logit_pde[:,:,res_mask[0]][:,:,:,res_mask[0]]
                
            tot_str, pae_loss, pde_loss = compute_general_FAPE(
                pred[:,res_mask,:,:3],
                true[:,res_mask[0],:3],
                mask_crds[:,res_mask[0],:3],
                frames_BB[:,res_mask[0]],
                frame_mask_BB[:,res_mask[0]],
                frame_atom_mask_2d=frame_atom_mask_2d_intra[:, res_mask[0]][:, :, :, res_mask[0]],
                dclamp=dclamp,
                logit_pae=logit_pae,
                logit_pde=logit_pde,
            )

            #fd pae/pde loss not computed correctly, zero for negatives
            # Pascal: I think the above is no longer true. PAE/PDE should
            # be computed correctly for intra chain
            pae_loss *= 0.0
            pde_loss *= 0.0

        else:

            if logit_pae is not None:
                logit_pae = logit_pae[:,:,res_mask[0]][:,:,:,res_mask[0]]
            if logit_pde is not None:
                logit_pde = logit_pde[:,:,res_mask[0]][:,:,:,res_mask[0]]
            
            # change clamp for intra protein to 10, leave rest at 30
            dclamp_2d = torch.full_like(frame_atom_mask_2d_allatom, dclamp, dtype=torch.float32)
            if not unclamp:
                is_prot = is_protein(seq) # (1,L)
                same_chain_clamp_mask = same_chain[:, :, None, :, None].repeat(1,1,nframes,1, natoms)
                # zero out rows and columns with small molecules
                same_chain_clamp_mask[:, ~is_prot[0]] = 0
                same_chain_clamp_mask[:,:, :,  ~is_prot[0]] = 0 
                dclamp_2d *= ~same_chain_clamp_mask.bool()
                dclamp_2d += same_chain_clamp_mask*dclamp_prot

            tot_str, pae_loss, pde_loss = compute_general_FAPE(
                pred[:,res_mask,:,:3],
                true[:,res_mask[0],:3],
                mask_crds[:,res_mask[0],:3],
                frames_BB[:,res_mask[0]],
                frame_mask_BB[:,res_mask[0]],
                dclamp=None,
                dclamp_2d=dclamp_2d[:, res_mask[0]][:, :, :, res_mask[0],:3], 
                logit_pae=logit_pae,
                logit_pde=logit_pde,
            )
        num_layers = pred.shape[0]
        gamma = 1.0 # equal weighting of fape across all layers
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

        # small-molecule ligands
        sm_res_mask = is_atom(label_aa_s[0,0])*res_mask[0] # (L,)
        # frame_atom_mask_2d = torch.einsum('bfn,bra->bfnra',frame_mask_BB,mask_crds[:, :, :3]) # B, L, nframes, L, natoms
        # frame_atom_mask_2d_intra = frame_atom_mask_2d*same_chain[:, :,None, :, None].expand(-1,-1,nframes,-1, 3) # B, L, nframes, L, natoms
        # different_chain = ~same_chain.bool()
        # frame_atom_mask_2d_inter = frame_atom_mask_2d*different_chain[:, :,None, :, None].expand(-1,-1,nframes,-1, 3) # B, L, nframes, L, natoms

        #fd
        frame_allatom_mask_2d = torch.einsum('bfn,bra->bfnra',frame_mask_BB,mask_crds) # B, L, nframes, L, natoms
        frame_allatom_mask_2d_intra = frame_allatom_mask_2d*same_chain[:, :,None, :, None].expand(-1,-1,nframes,-1, NTOTAL) # B, L, nframes, L, natoms

        if not negative and bool(torch.any(~sm_res_mask)) and torch.any(frame_mask_BB[0,~sm_res_mask]):
            # protein fape (layer-averaged fape on protein coordinates with protein frames)
            l_fape_prot_intra, _, _ = compute_general_FAPE(
                pred[:, ~sm_res_mask[None],:,:3],
                true[:,~sm_res_mask,:3,:3],
                atom_mask = mask_crds[:,~sm_res_mask, :3],
                frames = frames_BB[:,~sm_res_mask],
                frame_mask = frame_mask_BB[:,~sm_res_mask],
                frame_atom_mask_2d=frame_atom_mask_2d_intra[:, ~sm_res_mask][:, :, :, ~sm_res_mask],
            )
            prot_fape = l_fape_prot_intra.mean()
            
            l_fape_prot_inter, _, _ = compute_general_FAPE(
                pred[:, ~sm_res_mask[None],:,:3],
                true[:,~sm_res_mask,:3,:3],
                atom_mask = mask_crds[:,~sm_res_mask, :3],
                frames = frames_BB[:,~sm_res_mask],
                frame_mask = frame_mask_BB[:,~sm_res_mask],
                frame_atom_mask_2d=frame_atom_mask_2d_inter[:, ~sm_res_mask][:, :, :, ~sm_res_mask],
            )
            inter_prot_fape = l_fape_prot_inter.mean()
        else:
            prot_fape = torch.tensor(0).to(gpu)
            inter_prot_fape = torch.tensor(0).to(gpu)

        loss_dict['bb_fape_prot_intra'] = prot_fape.detach()
        loss_dict['bb_fape_prot_inter'] = inter_prot_fape.detach()

        if bool(torch.any(sm_res_mask)) and torch.any(frame_mask_BB[0,sm_res_mask]):
            # ligand fape (layer-averaged fape on atom coordinates with atom frames)
            l_fape_sm_intra, _, _ = compute_general_FAPE(
                pred[:, sm_res_mask[None],:,:3],
                true[:,sm_res_mask,:3,:3],
                atom_mask = mask_crds[:,sm_res_mask, :3],
                frames = frames_BB[:,sm_res_mask],
                frame_mask = frame_mask_BB[:,sm_res_mask],
                frame_atom_mask_2d=frame_atom_mask_2d_intra[:, sm_res_mask][:, :, :, sm_res_mask],
                dclamp=dclamp_sm,
                Z=Z_sm
            )
            lig_fape = (w_bb_fape*l_fape_sm_intra).sum()
            tot_loss += 0.5*w_lig_fape*lig_fape
            
            l_fape_sm_inter, _, _ = compute_general_FAPE(
                pred[:, sm_res_mask[None],:,:3],
                true[:,sm_res_mask,:3,:3],
                atom_mask = mask_crds[:,sm_res_mask, :3],
                frames = frames_BB[:,sm_res_mask],
                frame_mask = frame_mask_BB[:,sm_res_mask],
                frame_atom_mask_2d=frame_atom_mask_2d_inter[:, sm_res_mask][:, :, :, sm_res_mask],
                dclamp=dclamp_sm,
                Z=Z_sm
            )
            inter_lig_fape = l_fape_sm_inter.mean()
        else:
            lig_fape = torch.tensor(0).to(gpu)
            inter_lig_fape = torch.tensor(0).to(gpu)

        loss_dict['bb_fape_lig_intra'] = lig_fape.detach()
        loss_dict['bb_fape_lig_inter'] = inter_lig_fape.detach()

        if not bool(torch.all(sm_res_mask)) and bool(torch.any(sm_res_mask)):      
            # calculate interchain fape 
            # fape of protein coordinates wrt ligand frames 
            mask_crds_protein = mask_crds.clone()
            mask_crds_protein[:, sm_res_mask] = False
            frame_mask_BB_sm = frame_mask_BB.clone()
            frame_mask_BB_sm[:,~sm_res_mask] = False
            if torch.any(mask_crds_protein[:,res_mask[0], :3]) and torch.any(frame_mask_BB_sm[:,res_mask[0]]):
                l_fape_protein_sm, _, _ = compute_general_FAPE(
                    pred[:, res_mask,:,:3],
                    true[:, res_mask[0],:3,:3],
                    atom_mask = mask_crds_protein[:,res_mask[0], :3],
                    frames = frames_BB[:,res_mask[0]],
                    frame_mask = frame_mask_BB_sm[:,res_mask[0]],
                    frame_atom_mask = mask_crds[:,res_mask[0],:3],
                    dclamp=dclamp
                )
            else:
                l_fape_protein_sm = torch.tensor(0).to(gpu)
                
            # fape of ligand coordinates wrt protein frames
            mask_crds_sm = mask_crds.clone()
            mask_crds_sm[:, ~sm_res_mask] = False
            frame_mask_BB_protein = frame_mask_BB.clone()
            frame_mask_BB_protein[:,sm_res_mask] = False
            if torch.any(mask_crds_sm[:,res_mask[0], :3]) and torch.any(frame_mask_BB_protein[:,res_mask[0]]):
                l_fape_sm_protein, _, _ = compute_general_FAPE(
                    pred[:, res_mask,:,:3],
                    true[:, res_mask[0],:3,:3],
                    atom_mask = mask_crds_sm[:,res_mask[0], :3],
                    frames = frames_BB[:,res_mask[0]],
                    frame_mask = frame_mask_BB_protein[:,res_mask[0]],
                    frame_atom_mask = mask_crds[:,res_mask[0],:3],
                    dclamp=dclamp
                )
            else:
                l_fape_sm_protein = torch.tensor(0).to(gpu)

            frac_sm = torch.sum(frame_mask_BB_sm[:,res_mask[0]])/ torch.sum(frame_mask_BB[:,res_mask[0]])
            inter_fape = frac_sm*l_fape_protein_sm + (1.0-frac_sm)*l_fape_sm_protein
            bb_l_fape_inter = (w_bb_fape*inter_fape).sum()
            tot_loss += 0.5*w_inter_fape*bb_l_fape_inter
        else:
            bb_l_fape_inter = torch.tensor(0).to(gpu)

        loss_dict['bb_fape_inter'] = bb_l_fape_inter.detach()

        # AllAtom loss
        # get ground-truth torsion angles
        true_tors, true_tors_alt, tors_mask, tors_planar = self.xyz_converter.get_torsions(
            true, seq, mask_in=mask_crds)
        tors_mask *= mask_BB[...,None]

        # get alternative coordinates for ground-truth
        true_alt = torch.zeros_like(true)
        true_alt.scatter_(2, self.l2a[seq,:,None].repeat(1,1,1,3), true)
        natRs_all, _n0 = self.xyz_converter.compute_all_atom(seq, true[...,:3,:], true_tors)
        natRs_all_alt, _n1 = self.xyz_converter.compute_all_atom(seq, true_alt[...,:3,:], true_tors_alt)
        predTs = pred[-1,...]
        predRs_all, pred_all = self.xyz_converter.compute_all_atom(seq, predTs, pred_tors[-1]) 

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
            pred_allatom.detach(), nat_symm, pred_lddt, idx, mask_crds, mask_2d, same_chain, 
            negative=negative, interface=interface, N_stripe=10)
        tot_loss += w_lddt*lddt_loss
        loss_dict['lddt_loss'] = lddt_loss.detach()
        loss_dict['allatom_lddt'] = allatom_lddt[0].detach()
        #print (allatom_lddt[0].detach())

        # FAPE losses
        # allatom fape and torsion angle loss
        # frames, frame_mask = get_frames(
        #     pred_allatom[-1,None,...], mask_crds, seq, self.fi_dev, atom_frames)
        if negative.item(): # inter-chain fapes should be ignored for negative cases
            l_fape, _, _ = compute_general_FAPE(
                pred_allatom[:,res_mask[0],:,:3],
                nat_symm[None,res_mask[0],:,:3],
                xs_mask[:,res_mask[0]],
                frames[:,res_mask[0]],
                frame_mask[:,res_mask[0]],
                frame_atom_mask_2d=frame_allatom_mask_2d_intra[:, res_mask[0]][:, :, :, res_mask[0]]
            )

        else:
            l_fape, _, _ = compute_general_FAPE(
                pred_allatom[:,res_mask[0],:,:3],
                nat_symm[None,res_mask[0],:,:3],
                xs_mask[:,res_mask[0]],
                frames[:,res_mask[0]],
                frame_mask[:,res_mask[0]]
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

        # create protein and not protein masks; not protein could include nucleic acids
        prot_mask_BB = is_protein(label_aa_s[0,0]) #*mask_BB[0] # (L,)
        not_prot_mask_BB  = ~prot_mask_BB.bool()
        xs_mask_prot, xs_mask_lig = xs_mask.clone(), xs_mask.clone()
        xs_mask_prot[:,~prot_mask_BB] = False
        xs_mask_lig[:,~not_prot_mask_BB] = False
        if torch.any(prot_mask_BB):
            rmsd_prot_prot = calc_crd_rmsd(
                pred=pred_allatom[:,mask_BB[0],:,:3], true=nat_symm[None,mask_BB[0],:,:3],
                atom_mask=xs_mask_prot[:,mask_BB[0]], rmsd_mask=xs_mask_prot[:,mask_BB[0]]
            )
        else:
            rmsd_prot_prot = torch.tensor([0], device=pred.device)
        if torch.any(not_prot_mask_BB):
            rmsd_lig_lig = calc_crd_rmsd(
                pred=pred_allatom[:,mask_BB[0],:,:3], true=nat_symm[None,mask_BB[0],:,:3],
                atom_mask=xs_mask_lig[:,mask_BB[0]], rmsd_mask=xs_mask_lig[:,mask_BB[0]]
            )
        else:
            rmsd_lig_lig = torch.tensor([0], device=pred.device)
        if torch.any(prot_mask_BB) and torch.any(not_prot_mask_BB):
            rmsd_prot_lig = calc_crd_rmsd(
                pred=pred_allatom[:,mask_BB[0],:,:3], true=nat_symm[None,mask_BB[0],:,:3],
                atom_mask=xs_mask_prot[:,mask_BB[0]], rmsd_mask=xs_mask_lig[:,mask_BB[0]],
                alignment_radius=10.0
            )
        else:
            rmsd_prot_lig = torch.tensor([0], device=pred.device)
 
        loss_dict["rmsd_prot_prot"]= rmsd_prot_prot[0].detach()
        loss_dict["rmsd_lig_lig"]= rmsd_lig_lig[0].detach()
        loss_dict["rmsd_prot_lig"]= rmsd_prot_lig[0].detach()

        # cart bonded (bond geometry)
        bond_loss = calc_BB_bond_geom(seq[0], pred_allatom[0:1], idx)
        if w_bond > 0.0:
            tot_loss += w_bond*bond_loss
        loss_dict['bond_geom'] = bond_loss.detach()

        if (pred_allatom.shape[0] > 1):
            bond_loss = calc_cart_bonded(seq, pred_allatom[1:], idx, self.cb_len, self.cb_ang, self.cb_tor)
            if w_bond > 0.0:
                tot_loss += w_bond*bond_loss.mean()
            loss_dict['clash_loss'] = ( bond_loss.detach() )
        else:
            bond_loss = torch.tensor(0).to(gpu)
        loss_dict['bond_loss'] = bond_loss.detach()

        # clash [use all atoms not just those in native]
        clash_loss = calc_lj(
            seq[0], pred_allatom, 
            self.aamask, bond_feats, dist_matrix, self.ljlk_parameters, self.lj_correction_parameters, self.num_bonds,
            lj_lin=lj_lin
        )
        if w_clash > 0.0:
            tot_loss += w_clash*clash_loss.mean()
        loss_dict['clash_loss'] = clash_loss[0].detach()
        atom_bond_loss, skip_bond_loss, rigid_loss = calc_atom_bond_loss(
            pred=pred_allatom[:,mask_BB[0]],
            true=nat_symm[None,mask_BB[0]],
            bond_feats=bond_feats[:,mask_BB[0]][:,:,mask_BB[0]],
            seq=seq[:,mask_BB[0]]
        )
        if w_atom_bond >= 0.0:
            tot_loss += w_atom_bond*atom_bond_loss
        loss_dict['atom_bond_loss'] = ( atom_bond_loss.detach() )

        if w_skip_bond >= 0.0:
            tot_loss += w_skip_bond*skip_bond_loss
        loss_dict['skip_bond_loss'] = ( skip_bond_loss.detach() )

        if w_rigid >= 0.0:
            tot_loss += w_rigid*rigid_loss
        loss_dict['rigid_loss'] = ( rigid_loss.detach() )
        chain_prot = same_chain.clone()
        protein_mask_2d = torch.einsum('l,r-> lr', prot_mask_BB, prot_mask_BB)

        _, allatom_lddt_prot_intra = calc_allatom_lddt_loss(
            pred_allatom.detach(), nat_symm, pred_lddt, idx, mask_crds, protein_mask_2d[None], 
            chain_prot, negative=True, N_stripe=10)
        loss_dict['allatom_lddt_prot_intra'] = allatom_lddt_prot_intra[0].detach()

        _, allatom_lddt_prot_inter = calc_allatom_lddt_loss(
            pred_allatom.detach(), nat_symm, pred_lddt, idx, mask_crds, protein_mask_2d[None], 
            chain_prot, interface=True, N_stripe=10)
        loss_dict['allatom_lddt_prot_inter'] = allatom_lddt_prot_inter[0].detach()
        
        chain_lig = same_chain.clone()
        not_protein_mask_2d = torch.einsum('l,r-> lr', not_prot_mask_BB, not_prot_mask_BB)
        _, allatom_lddt_lig_intra = calc_allatom_lddt_loss(
            pred_allatom.detach(), nat_symm, pred_lddt, idx, mask_crds, not_protein_mask_2d[None], 
            chain_lig, negative=True, bin_scaling=0.5, N_stripe=10)
        loss_dict['allatom_lddt_lig_intra'] = allatom_lddt_lig_intra[0].detach()
        
        _, allatom_lddt_lig_inter = calc_allatom_lddt_loss(
            pred_allatom.detach(), nat_symm, pred_lddt, idx, mask_crds, not_protein_mask_2d[None], 
            chain_lig, interface=True, bin_scaling=0.5, N_stripe=10)
        loss_dict['allatom_lddt_lig_inter'] = allatom_lddt_lig_inter[0].detach()

        chain_prot_lig_inter = torch.zeros_like(same_chain, dtype=bool)
        chain_prot_lig_inter += protein_mask_2d
        chain_prot_lig_inter += not_protein_mask_2d
        _, allatom_lddt_inter = calc_allatom_lddt_loss(
            pred_allatom.detach(), nat_symm, pred_lddt, idx, mask_crds, mask_2d, 
            chain_prot_lig_inter, interface=True, N_stripe=10)
        loss_dict['allatom_lddt_prot_lig_inter'] = allatom_lddt_inter[0].detach()
        loss_dict['total_loss'] = tot_loss.detach()

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
        if self.debug_mode:
            return -1, 99999999.9
        chk_fn = self.model_dir+"/%s_%s.pt"%(model_name, suffix)
        loaded_epoch = -1
        best_valid_loss = 999999.9
        if not os.path.exists(chk_fn):
            print ('no model found', model_name)
            return -1, best_valid_loss
        map_location = {"cuda:%d"%0: "cuda:%d"%rank}
        checkpoint = torch.load(chk_fn, map_location=map_location)
        loaded_epoch = checkpoint['epoch']
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

        if self.debug_mode:
            print("Running in DEBUG mode...")
            world_size = 1
            rank = 0
            self.train_model(rank, world_size)
        if (not self.interactive and "SLURM_NTASKS" in os.environ and "SLURM_PROCID" in os.environ):
            world_size = int(os.environ["SLURM_NTASKS"])
            rank = int (os.environ["SLURM_PROCID"])
            print ("Launched from slurm", rank, world_size)
            self.train_model(rank, world_size)
        else:
            print ("Launched from interactive")
            world_size = torch.cuda.device_count()
            if world_size == 1:
                # No need for multiple processes with 1 GPU
                self.train_model(0, world_size)
            else:
                mp.spawn(self.train_model, args=(world_size,), nprocs=world_size, join=True)
 
    def record_git_commit(self):
        # git hash of current commit
        try:
            commit = subprocess.check_output(f'git --git-dir {script_dir}/../.git rev-parse HEAD',
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
        if self.debug_mode: self.wandb_prefix = None
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
        dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
        torch.cuda.set_device("cuda:%d"%gpu)

        # define dataset & data loader
        train_ID_dict, valid_ID_dict, weights_dict, train_dict, valid_dict, homo, chid2hash, chid2taxid = \
            get_train_valid_set(self.loader_param)

        # define atomize_pdb train/valid sets, which use the same examples as pdb set
        train_ID_dict['atomize_pdb'] = train_ID_dict['pdb']
        valid_ID_dict['atomize_pdb'] = valid_ID_dict['pdb']
        weights_dict['atomize_pdb'] = weights_dict['pdb']
        train_dict['atomize_pdb'] = train_dict['pdb']
        valid_dict['atomize_pdb'] = valid_dict['pdb']

        # set number of validation examples being used
        for k in valid_dict:
            if self.dataset_param['n_valid_'+k] is None: 
                self.dataset_param["n_valid_"+k] = len(valid_dict[k]) 

        if (rank==0):
            print('Number of training clusters / examples:')
            for k in train_ID_dict:
                print('  '+k, ':', len(train_ID_dict[k]), '/', len(train_dict[k]))

            print('Number of validation clusters / examples:')
            for k in valid_ID_dict:
                print('  '+k, ':', len(valid_ID_dict[k]), '/', len(valid_dict[k]))

            print('Using number of validation examples:')
            for k in valid_dict:
                print('  '+k, ':', self.dataset_param['n_valid_'+k])

        loader_dict = dict(
            pdb = loader_pdb,
            compl = loader_complex,
            neg_compl = loader_complex,
            na_compl = loader_na_complex,
            neg_na_compl = loader_na_complex,
            fb = loader_fb,
            rna = loader_rna,
            sm_compl = loader_sm_compl_assembly_single,
            metal_compl = loader_sm_compl_assembly_single,
            sm_compl_multi = loader_sm_compl_assembly_single,
            sm_compl_covale = loader_sm_compl_assembly_single,
            sm_compl_asmb = loader_sm_compl_assembly,
            sm = loader_sm,
            atomize_pdb = loader_atomize_pdb,
            sm_compl_furthest_neg = loader_sm_compl_assembly,
            sm_compl_permuted_neg = loader_sm_compl_assembly,
            sm_compl_docked_neg = loader_sm_compl_assembly,
        )

        train_set = DistilledDataset(train_ID_dict, train_dict, loader_dict, homo, chid2hash, chid2taxid,
                                     self.loader_param, native_NA_frac=0.25)

        train_sampler = DistributedWeightedSampler(
            train_set, 
            weights_dict,
            num_example_per_epoch=self.dataset_param['n_train'],
            fractions=OrderedDict([(k, self.dataset_param['fraction_'+k]) for k in train_dict]),
            num_replicas=world_size, 
            rank=rank, 
            replacement=True
        )

        train_loader = data.DataLoader(train_set, sampler=train_sampler, batch_size=self.batch_size, **self.dataloader_kwargs)

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
            rna = DatasetRNA(
                valid_ID_dict['rna'][:self.dataset_param['n_valid_rna']],
                loader_rna, valid_dict['rna'],
                self.loader_param
            ),
            metal_compl = DatasetSMComplexAssembly(
                valid_ID_dict['metal_compl'][:self.dataset_param['n_valid_metal_compl']],
                loader_sm_compl_assembly, valid_dict['metal_compl'],
                chid2hash, chid2taxid, # used for MSA generation of assemblies
                self.loader_param,
                task='metal_compl',
                num_protein_chains=1,
            ),
            sm_compl = DatasetSMComplexAssembly(
                valid_ID_dict['sm_compl'][:self.dataset_param['n_valid_sm_compl']],
                loader_sm_compl_assembly, valid_dict['sm_compl'],
                chid2hash, chid2taxid, # used for MSA generation of assemblies
                self.loader_param,
                task='sm_compl',
                num_protein_chains=1,
            ),
            sm_compl_multi = DatasetSMComplexAssembly(
                valid_ID_dict['sm_compl_multi'][:self.dataset_param['n_valid_sm_compl_multi']],
                loader_sm_compl_assembly, valid_dict['sm_compl_multi'],
                chid2hash, chid2taxid, # used for MSA generation of assemblies
                self.loader_param,
                task='sm_compl_multi',
                num_protein_chains=1,
            ),
            sm_compl_covale = DatasetSMComplexAssembly(
                valid_ID_dict['sm_compl_covale'][:self.dataset_param['n_valid_sm_compl_covale']],
                loader_sm_compl_assembly, valid_dict['sm_compl_covale'],
                chid2hash, chid2taxid, # used for MSA generation of assemblies
                self.loader_param,
                task='sm_compl_covale',
                num_protein_chains=1,
            ),
            sm_compl_asmb = DatasetSMComplexAssembly(
                valid_ID_dict['sm_compl_asmb'][:self.dataset_param['n_valid_sm_compl_asmb']],
                loader_sm_compl_assembly, valid_dict['sm_compl_asmb'],
                chid2hash, chid2taxid, # used for MSA generation of assemblies
                self.loader_param,
                task='sm_compl_asmb'
            ),
            sm_compl_strict = DatasetSMComplexAssembly(
                valid_ID_dict['sm_compl_strict'][:self.dataset_param['n_valid_sm_compl_strict']],
                loader_sm_compl_assembly, valid_dict['sm_compl_strict'],
                chid2hash, chid2taxid, # used for MSA generation of assemblies
                self.loader_param,
                task='sm_compl_strict',
                num_protein_chains=1,
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
            ),
        )

        valid_headers = dict(
            pdb = 'Monomer',
            homo = 'Homo',
            rna = 'RNA',
            sm_compl = 'SM_Compl',
            metal_compl = 'Metal_ion',
            sm_compl_multi = 'Multires_ligand',
            sm_compl_covale = "Covalent_ligand",
            sm_compl_strict = 'SM_Compl_(strict)',
            sm = 'SM_CSD',
            atomize_pdb = 'Monomer_atomize_3',
            sm_compl_asmb = 'SMCompl_Assembly',
        )
        valid_samplers = OrderedDict([
            (k, data.distributed.DistributedSampler(v, num_replicas=world_size, rank=rank))
            for k,v in valid_sets.items()
        ])
        valid_loaders = OrderedDict([
            (k, data.DataLoader(v, sampler=valid_samplers[k], **self.dataloader_kwargs))
            for k,v in valid_sets.items()
        ])

        # PPI validation requires pairs of positive/negative datasets
        # the three SM complex datasets currently result in duplication of positive set
        # this should probably be addressed
        valid_ppi_sets = dict(
            compl = (
                DatasetComplex(
                    valid_ID_dict['compl'][:self.dataset_param['n_valid_compl']],
                    loader_complex, valid_dict['compl'],
                    self.loader_param, negative=False
                ),
                DatasetComplex(
                    valid_ID_dict['neg_compl'][:self.dataset_param['n_valid_neg_compl']],
                    loader_complex, valid_dict['neg_compl'],
                    self.loader_param, negative=True
                ),
            ),
            na_compl = (
                DatasetNAComplex(
                    valid_ID_dict['na_compl'][:self.dataset_param['n_valid_na_compl']],
                    loader_na_complex, valid_dict['na_compl'],
                    self.loader_param, negative=False, native_NA_frac=0.0
                ),
                DatasetNAComplex(
                    valid_ID_dict['neg_na_compl'][:self.dataset_param['n_valid_neg_na_compl']],
                    loader_na_complex, valid_dict['neg_na_compl'],
                    self.loader_param, negative=True, native_NA_frac=0.0
                ),
            ),
            sm_compl_furthest = (
                DatasetSMComplexAssembly(
                    valid_ID_dict['sm_compl'][:self.dataset_param['n_valid_sm_compl']],
                    loader_sm_compl_assembly, valid_dict['sm_compl'],
                    chid2hash, chid2taxid, # used for MSA generation of assemblies
                    self.loader_param,
                    task='sm_compl',
                    num_protein_chains=1, num_ligand_chains=1,
                ),
                DatasetSMComplexAssembly(
                    valid_ID_dict['sm_compl_furthest_neg'][:self.dataset_param['n_valid_sm_compl_furthest_neg']],
                    loader_sm_compl_assembly, valid_dict['sm_compl_furthest_neg'],
                    chid2hash, chid2taxid,
                    self.loader_param, select_farthest_residues=True,
                    task="sm_compl_furthest_neg", num_protein_chains=1, num_ligand_chains=1,
                ),
            ),
            sm_compl_permuted = (
                DatasetSMComplexAssembly(
                    valid_ID_dict['sm_compl'][:self.dataset_param['n_valid_sm_compl']],
                    loader_sm_compl_assembly, valid_dict['sm_compl'],
                    chid2hash, chid2taxid, # used for MSA generation of assemblies
                    self.loader_param,
                    task='sm_compl',
                    num_protein_chains=1, num_ligand_chains=1,
                ),
                DatasetSMComplexAssembly(
                    valid_ID_dict['sm_compl_permuted_neg'][:self.dataset_param['n_valid_sm_compl_permuted_neg']],
                    loader_sm_compl_assembly, valid_dict['sm_compl_permuted_neg'],
                    chid2hash, chid2taxid,
                    self.loader_param, select_negative_ligand=True,
                    task="sm_compl_permuted_neg", num_protein_chains=1, num_ligand_chains=1,
                ),
            ),
            sm_compl_docked = (
                DatasetSMComplexAssembly(
                    valid_ID_dict['sm_compl'][:self.dataset_param['n_valid_sm_compl']],
                    loader_sm_compl_assembly, valid_dict['sm_compl'],
                    chid2hash, chid2taxid, # used for MSA generation of assemblies
                    self.loader_param,
                    task='sm_compl',
                    num_protein_chains=1, num_ligand_chains=1,
                ),
                DatasetSMComplexAssembly(
                    valid_ID_dict['sm_compl_docked_neg'][:self.dataset_param['n_valid_sm_compl_docked_neg']],
                    loader_sm_compl_assembly, valid_dict['sm_compl_docked_neg'],
                    chid2hash, chid2taxid,
                    self.loader_param, select_negative_ligand=True,
                    task="sm_compl_docked_neg", num_protein_chains=1, num_ligand_chains=1,
                ),
            ),
            dude = (
                DatasetSMComplexAssembly(
                    valid_ID_dict['dude_actives'][:self.dataset_param['n_valid_dude_actives']],
                    loader_sm_compl_assembly, valid_dict['dude_actives'],
                    chid2hash, chid2taxid,
                    self.loader_param, load_from_smiles_string=True,
                    task="dude_actives", num_protein_chains=1, num_ligand_chains=1,
                ),
                DatasetSMComplexAssembly(
                    valid_ID_dict['dude_inactives'][:self.dataset_param['n_valid_dude_inactives']],
                    loader_sm_compl_assembly, valid_dict['dude_inactives'],
                    chid2hash, chid2taxid,
                    self.loader_param, load_from_smiles_string=True,
                    task="dude_inactives", num_protein_chains=1, num_ligand_chains=1,
                ),
            )
        )
        valid_ppi_headers = dict(
            compl = 'Complex',
            na_compl = 'P/NA_Complex',
            sm_compl_furthest = 'SM_Complex_(furthest_crop)',
            sm_compl_permuted = 'SM_Complex_(property_matched)',
            sm_compl_docked = "SM_Complex_(docked)",
            dude = "DUD-e",
        )
        valid_ppi_samplers = OrderedDict([
            (k, 
                (
                    data.distributed.DistributedSampler(v, num_replicas=world_size, rank=rank),
                    data.distributed.DistributedSampler(w, num_replicas=world_size, rank=rank),
                )
            )
            for k,(v,w) in valid_ppi_sets.items()
        ])
        valid_ppi_loaders = OrderedDict([
            (k, 
                (
                    data.DataLoader(v, sampler=valid_ppi_samplers[k][0], **self.dataloader_kwargs),
                    data.DataLoader(w, sampler=valid_ppi_samplers[k][1], **self.dataloader_kwargs),
                )
            )
            for k,(v,w) in valid_ppi_sets.items()
        ])

        # move some global data to cuda device
        self.fi_dev = self.fi_dev.to(gpu)
        self.xyz_converter = self.xyz_converter.to(gpu)

        self.l2a = self.l2a.to(gpu)
        self.aamask = self.aamask.to(gpu)
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

        ddp_model = DDP(model, device_ids=[gpu], find_unused_parameters=False, broadcast_buffers=False)
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
        valid_tot = None
        if loaded_epoch >= self.n_epoch:
            DDP_cleanup()
            return

        rng = np.random.RandomState(seed=rank)

        ##fd  Uncomment to run validation set before beginning training
        #for dataset_name, valid_loader in valid_loaders.items():
        #    #print (dataset_name, len(valid_loader))
        #    valid_tot_, valid_loss_, valid_acc_, _ = self.valid_pdb_cycle(ddp_model, 
        #        valid_loader, rank, gpu, world_size, loaded_epoch, rng, 
        #        header=valid_headers[dataset_name], verbose = self.eval) 
        #for dataset_name, (valid_pos_loader, valid_neg_loader) in valid_ppi_loaders.items():
        #    #print (dataset_name, len(valid_pos_loader), len(valid_neg_loader))
        #    valid_tot_, valid_loss_, valid_acc_, _, _ = self.valid_ppi_cycle(ddp_model, 
        #        valid_pos_loader, valid_neg_loader, rank, gpu, world_size, loaded_epoch, rng, 
        #        header=valid_ppi_headers[dataset_name], verbose = self.eval) 

        for epoch in range(loaded_epoch+1, self.n_epoch):
            train_sampler.set_epoch(epoch)
            for k, sampler in valid_samplers.items():
                sampler.set_epoch(epoch)

            rng = np.random.RandomState(seed=epoch*world_size+rank)

            train_tot, train_loss, train_acc = self.train_cycle(ddp_model, train_loader, optimizer, scheduler, scaler, rank, gpu, world_size, epoch, rng)

            if (epoch % self.skip_valid == 0) or (epoch==loaded_epoch+1):
                for dataset_name, valid_loader in valid_loaders.items():
                    valid_tot_, valid_loss_, valid_acc_, _ = self.valid_pdb_cycle(ddp_model, 
                        valid_loader, rank, gpu, world_size, epoch, rng, 
                        header=valid_headers[dataset_name], verbose = self.eval) 

                    if dataset_name == 'sm_compl':
                        valid_tot, valid_loss, valid_acc = valid_tot_, valid_loss_, valid_acc_

                for dataset_name, (valid_pos_loader, valid_neg_loader) in valid_ppi_loaders.items():
                    valid_tot_, valid_loss_, valid_acc_, _, _ = self.valid_ppi_cycle(ddp_model, 
                        valid_pos_loader, valid_neg_loader, rank, gpu, world_size, epoch, rng, 
                        header=valid_ppi_headers[dataset_name], verbose = self.eval) 

            if self.eval: break

            if rank == 0: # save model
                if valid_tot is not None and valid_tot < best_valid_loss:
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

    def _prepare_input(self, inputs, gpu):
        (
            seq, msa, msa_masked, msa_full, mask_msa, true_crds, mask_crds, idx_pdb, 
            xyz_t, t1d, mask_t, xyz_prev, mask_prev, same_chain, unclamp, negative, 
            atom_frames, bond_feats, dist_matrix, chirals, ch_label, symmgp, task, item
        ) = inputs

        # transfer inputs to device
        B, _, N, L = msa.shape

        idx_pdb = idx_pdb.to(gpu, non_blocking=True) # (B, L)
        true_crds = true_crds.to(gpu, non_blocking=True) # (B, L, 27, 3)
        mask_crds = mask_crds.to(gpu, non_blocking=True) # (B, L, 27)
        same_chain = same_chain.to(gpu, non_blocking=True)

        xyz_t = xyz_t.to(gpu, non_blocking=True)
        t1d = t1d.to(gpu, non_blocking=True)
        mask_t = mask_t.to(gpu, non_blocking=True)
        
        xyz_prev = xyz_prev.to(gpu, non_blocking=True)
        mask_prev = mask_prev.to(gpu, non_blocking=True)

        seq = seq.to(gpu, non_blocking=True)
        msa = msa.to(gpu, non_blocking=True)
        msa_masked = msa_masked.to(gpu, non_blocking=True)
        msa_full = msa_full.to(gpu, non_blocking=True)
        mask_msa = mask_msa.to(gpu, non_blocking=True)
        atom_frames = atom_frames.to(gpu, non_blocking=True)
        bond_feats = bond_feats.to(gpu, non_blocking=True)
        dist_matrix = dist_matrix.to(gpu, non_blocking=True)
        chirals = chirals.to(gpu, non_blocking=True)
        assert (len(symmgp)==1)
        symmgp = symmgp[0]

        # symmetry - reprocess (many) inputs
        if (symmgp != 'C1'):
            Lasu = L//2 # msa contains intra/inter block
            symmids, symmRs, symmmeta, symmoffset = symm_subunit_matrix(symmgp)
            symmids = symmids.to(gpu, non_blocking=True)
            symmRs = symmRs.to(gpu, non_blocking=True)
            symmoffset = symmoffset.to(gpu, non_blocking=True)
            symmmeta = (
                [x.to(gpu, non_blocking=True) for x in symmmeta[0]],
                symmmeta[1])
            O = symmids.shape[0]
            xyz_prev = xyz_prev + symmoffset*Lasu**(1/3)

            # find contacting subunits
            xyz_prev, symmsub = find_symm_subs(xyz_prev[:,:Lasu], symmRs, symmmeta)
            symmsub = symmsub.to(gpu, non_blocking=True)
            Osub = symmsub.shape[0]
            mask_prev = mask_prev[:,:L].repeat(1,Osub,1)

            # symmetrize msa
            seq = torch.cat([seq[:,:,:Lasu],*[seq[:,:,Lasu:]]*(Osub-1)], dim=2)
            msa = torch.cat([msa[:,:,:,:Lasu],*[msa[:,:,:,Lasu:]]*(Osub-1)], dim=3)
            msa_masked = torch.cat([msa_masked[:,:,:,:Lasu],*[msa_masked[:,:,:,Lasu:]]*(Osub-1)], dim=3)
            msa_full = torch.cat([msa_full[:,:,:,:Lasu],*[msa_full[:,:,:,Lasu:]]*(Osub-1)], dim=3)
            mask_msa = torch.cat([mask_msa[:,:,:,:Lasu],*[mask_msa[:,:,:,Lasu:]]*(Osub-1)], dim=3)

            # symmetrize templates
            xyz_t = xyz_t[:,:,:Lasu].repeat(1,1,Osub,1,1)
            mask_t = mask_t[:,:,:Lasu].repeat(1,1,Osub,1)
            t1d = t1d[:,:,:Lasu].repeat(1,1,Osub,1)

            # symmetrize atom_frames
            atom_frames = torch.cat([atom_frames[:,:,:Lasu],*[atom_frames[:,:,Lasu:]]*(Osub-1)], dim=2)

            # index, same chain, bond feats
            idx_pdb = torch.arange(Osub*Lasu, device=gpu)[None,:]
            same_chain = torch.zeros((1,Osub*Lasu,Osub*Lasu), device=gpu).long()
            bond_feats_new = torch.zeros((1,Osub*Lasu,Osub*Lasu), device=gpu).long()
            dist_matrix_new = torch.zeros((1,Osub*Lasu,Osub*Lasu), device=gpu).long()
            for o_i in range(Osub):
                same_chain[:,o_i*Lasu:(o_i+1)*Lasu,o_i*Lasu:(o_i+1)*Lasu] = 1
                idx_pdb[:,o_i*Lasu:(o_i+1)*Lasu] += 100*o_i
                bond_feats_new[:,o_i*Lasu:(o_i+1)*Lasu,o_i*Lasu:(o_i+1)*Lasu] = bond_feats
                dist_matrix_new[:,o_i*Lasu:(o_i+1)*Lasu,o_i*Lasu:(o_i+1)*Lasu] = dist_matrix

            bond_feats = bond_feats_new
            dist_matrix = dist_matrix_new

        else:
            Lasu = L
            Osub = 1
            symmids = None
            symmsub = None
            symmRs = None
            symmmeta = None

        # processing template features
        mask_t_2d = mask_t[:,:,:,:3].all(dim=-1) # (B, T, L)
        mask_t_2d = mask_t_2d[:,:,None]*mask_t_2d[:,:,:,None] # (B, T, L, L)
        mask_t_2d = mask_t_2d.float() * same_chain.float()[:,None] # (ignore inter-chain region)
        t2d = xyz_to_t2d(xyz_t, mask_t_2d)

        # get torsion angles from templates
        seq_tmp = t1d[...,:-1].argmax(dim=-1).reshape(-1,Lasu*Osub)
        alpha, _, alpha_mask, _ = self.xyz_converter.get_torsions(xyz_t.reshape(-1,Lasu*Osub,NTOTAL,3), seq_tmp, mask_in=mask_t.reshape(-1,Lasu*Osub,NTOTAL))
        alpha = alpha.reshape(B,-1,Lasu*Osub,NTOTALDOFS,2)
        alpha_mask = alpha_mask.reshape(B,-1,Lasu*Osub,NTOTALDOFS,1)
        alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(B, -1, Lasu*Osub, 3*NTOTALDOFS)
        alpha_prev = torch.zeros((B,Lasu*Osub,NTOTALDOFS,2))

        network_input = {}
        network_input['msa_latent'] = msa_masked
        network_input['msa_full'] = msa_full
        network_input['seq'] = seq
        network_input['seq_unmasked'] = msa[:,0,0]
        network_input['idx'] = idx_pdb
        network_input['t1d'] = t1d
        network_input['t2d'] = t2d
        network_input['xyz_t'] = xyz_t[:,:,:,1]
        network_input['alpha_t'] = alpha_t
        network_input['mask_t'] = mask_t_2d
        network_input['same_chain'] = same_chain
        network_input['bond_feats'] = bond_feats
        network_input['dist_matrix'] = dist_matrix

        network_input['chirals'] = chirals
        network_input['atom_frames'] = atom_frames

        network_input['symmids'] = symmids
        network_input['symmsub'] = symmsub
        network_input['symmRs'] = symmRs
        network_input['symmmeta'] = symmmeta

        mask_recycle = mask_prev[:,:,:3].bool().all(dim=-1)
        mask_recycle = mask_recycle[:,:,None]*mask_recycle[:,None,:] # (B, L, L)
        mask_recycle = same_chain.float()*mask_recycle.float()
        return task, item, network_input, xyz_prev, alpha_prev, mask_recycle, true_crds, mask_crds, msa, mask_msa, unclamp, negative, symmRs, Lasu, ch_label
    
    def _get_model_input(self, network_input, output_i, i_cycle, return_raw=False, use_checkpoint=False):
        input_i = {}
        for key in network_input:
            if key in ['msa_latent', 'msa_full', 'seq']:
                input_i[key] = network_input[key][:,i_cycle]
            else:
                input_i[key] = network_input[key]
        msa_prev, pair_prev, xyz_prev, state_prev, alpha, mask_recycle = output_i
        input_i['msa_prev'] = msa_prev
        input_i['pair_prev'] = pair_prev
        input_i['state_prev'] = state_prev
        input_i['xyz'] = xyz_prev
        input_i['mask_recycle'] = mask_recycle
        input_i['sctors'] = alpha
        input_i['return_raw'] = return_raw
        input_i['use_checkpoint'] = use_checkpoint
        return input_i

    def _get_loss_and_misc(
        self, output_i, true_crds, atom_mask, same_chain,
        seq, msa, mask_msa, idx_pdb, bond_feats, dist_matrix, atom_frames, unclamp, negative, task, item, symmRs, Lasu, ch_label, ctrid=0
    ):
        logit_s, logit_aa_s, logit_pae, logit_pde, p_bind, pred_crds, alphas, pred_allatom, pred_lddts, _, _, _ = output_i

        if (symmRs is not None):
            #print ('a', pred_crds.shape, true_crds.shape, mask_crds.shape)
            ###
            # resolve symmetry
            ###
            true_crds = true_crds[:,0]
            atom_mask = atom_mask[:,0]
            mapT2P = resolve_symmetry_predictions(pred_crds, true_crds, atom_mask, Lasu) # (Nlayer, Ltrue)

            # update all derived data to only include subunits mapping to native
            logit_s_new = []
            for li in logit_s:
                li=torch.gather(li,2,mapT2P[-1][None,None,:,None].repeat(1,li.shape[1],1,li.shape[-1]))
                li=torch.gather(li,3,mapT2P[-1][None,None,None,:].repeat(1,li.shape[1],li.shape[2],1))
                logit_s_new.append(li)
            logit_s = tuple(logit_s_new)

            logit_aa_s = logit_aa_s.view(1,NAATOKENS,msa.shape[-2],msa.shape[-1])
            logit_aa_s = torch.gather(logit_aa_s,3,mapT2P[-1][None,None,None,:].repeat(1,NAATOKENS,logit_aa_s.shape[-2],1))
            logit_aa_s = logit_aa_s.view(1,NAATOKENS,-1)

            msa = torch.gather(msa,2,mapT2P[-1][None,None,:].repeat(1,msa.shape[-2],1))
            mask_msa = torch.gather(mask_msa,2,mapT2P[-1][None,None,:].repeat(1,mask_msa.shape[-2],1))

            logit_pae=torch.gather(logit_pae,2,mapT2P[-1][None,None,:,None].repeat(1,logit_pae.shape[1],1,logit_pae.shape[-1]))
            logit_pae=torch.gather(logit_pae,3,mapT2P[-1][None,None,None,:].repeat(1,logit_pae.shape[1],logit_pae.shape[2],1))

            logit_pde=torch.gather(logit_pde,2,mapT2P[-1][None,None,:,None].repeat(1,logit_pde.shape[1],1,logit_pde.shape[-1]))
            logit_pde=torch.gather(logit_pde,3,mapT2P[-1][None,None,None,:].repeat(1,logit_pde.shape[1],logit_pde.shape[2],1))

            pred_crds = torch.gather(pred_crds,2,mapT2P[:,None,:,None,None].repeat(1,1,1,3,3))
            pred_allatom = torch.gather(pred_allatom,1,mapT2P[-1,None,:,None,None].repeat(1,1,NTOTAL,3))
            alphas = torch.gather(alphas,2,mapT2P[:,None,:,None,None].repeat(1,1,1,NTOTALDOFS,2))

            same_chain=torch.gather(same_chain,1,mapT2P[-1][None,:,None].repeat(1,1,same_chain.shape[-1]))
            same_chain=torch.gather(same_chain,2,mapT2P[-1][None,None,:].repeat(1,same_chain.shape[1],1))

            bond_feats=torch.gather(bond_feats,1,mapT2P[-1][None,:,None].repeat(1,1,bond_feats.shape[-1]))
            bond_feats=torch.gather(bond_feats,2,mapT2P[-1][None,None,:].repeat(1,bond_feats.shape[1],1))

            dist_matrix=torch.gather(dist_matrix,1,mapT2P[-1][None,:,None].repeat(1,1,dist_matrix.shape[-1]))
            dist_matrix=torch.gather(dist_matrix,2,mapT2P[-1][None,None,:].repeat(1,dist_matrix.shape[1],1))

            pred_lddts = torch.gather(pred_lddts,2,mapT2P[-1][None,None,:].repeat(1,pred_lddts.shape[-2],1))
            idx_pdb = torch.gather(idx_pdb,1,mapT2P[-1][None,:])
        elif 'sm_compl' in task[0] or 'metal_compl' in task[0]:
            sm_mask = is_atom(seq[0,0])
            Ls_prot = Ls_from_same_chain_2d(same_chain[:,~sm_mask][:,:,~sm_mask])
            Ls_sm = Ls_from_same_chain_2d(same_chain[:,sm_mask][:,:,sm_mask])

            true_crds, atom_mask = resolve_equiv_natives_asmb(
                pred_allatom, true_crds, atom_mask, ch_label, Ls_prot, Ls_sm)
        else:
            true_crds, atom_mask = resolve_equiv_natives(pred_crds[-1], true_crds, atom_mask)

        res_mask = get_prot_sm_mask(atom_mask, msa[0,0])
        mask_2d = res_mask[:,None,:] * res_mask[:,:,None]

        true_crds_frame = xyz_to_frame_xyz(true_crds, msa[:, 0], atom_frames)
        c6d = xyz_to_c6d(true_crds_frame)
        c6d = c6d_to_bins(c6d, same_chain, negative=negative)

        prob = self.active_fn(logit_s[0]) # distogram
        acc_s = self.calc_acc(prob, c6d[...,0], idx_pdb, mask_2d)

        loss, loss_dict = self.calc_loss(
            logit_s, c6d,
            logit_aa_s, msa, mask_msa, logit_pae, logit_pde, p_bind,
            pred_crds, alphas, pred_allatom, true_crds, 
            atom_mask, res_mask, mask_2d, same_chain,
            pred_lddts, idx_pdb, bond_feats, dist_matrix, atom_frames=atom_frames,
            unclamp=unclamp, negative=negative,
            ctr=ctrid, item=item, task=task, **self.loss_param
        )
        
        return loss, loss_dict, acc_s, p_bind, true_crds, pred_allatom, res_mask


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

        for inputs in train_loader:
            (
                task, item, network_input, xyz_prev, alpha_prev, mask_recycle, 
                true_crds, mask_crds, msa, mask_msa, unclamp, negative, symmRs, Lasu, ch_label
            ) = self._prepare_input(inputs, gpu)
            xyz_prev_orig = xyz_prev.clone()
            
            counter += 1

            N_cycle = np.random.randint(1, self.maxcycle+1) # number of recycling

            output_i = (None, None, xyz_prev, None, alpha_prev, mask_recycle)
            for i_cycle in range(N_cycle):
                with ExitStack() as stack:
                    if i_cycle < N_cycle -1:
                        stack.enter_context(torch.no_grad())
                        stack.enter_context(ddp_model.no_sync())
                        stack.enter_context(torch.cuda.amp.autocast(enabled=USE_AMP))
                        return_raw=True
                        use_checkpoint=False
                    else:
                        stack.enter_context(torch.cuda.amp.autocast(enabled=USE_AMP))
                        return_raw=False
                        use_checkpoint=True
                    input_i = self._get_model_input(network_input, output_i, i_cycle, return_raw=return_raw, use_checkpoint=use_checkpoint)
                    
                    output_i = ddp_model(**input_i)
                    
                    if i_cycle < N_cycle - 1:
                        continue
                    loss, loss_dict, acc_s, _, true_crds, pred_allatom, res_mask = self._get_loss_and_misc(
                        output_i,
                        true_crds, mask_crds, network_input['same_chain'],
                        network_input['seq'], msa[:,i_cycle], mask_msa[:,i_cycle],
                        network_input['idx'], network_input['bond_feats'], network_input['dist_matrix'], network_input['atom_frames'],
                        unclamp, negative, task, item, symmRs, Lasu, ch_label,
                        len(train_loader)*rank+counter
                    )

            loss = loss / self.ACCUM_STEP
            scaler.scale(loss).backward()
            if counter%self.ACCUM_STEP == 0:  
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

            item_ = unbatch_item(item) # remove nested lists to make more readable when printed
            save_pdbs = False
            if torch.isnan(loss):
                print('nan loss',item_)
                save_pdbs = True

            if task[0].startswith('sm_compl') or task[0].startswith('metal_compl'):
                if type(item_['LIGAND'][0]) is list: # multires or covalent ligands
                    lig_str = '_'.join([x[0]+x[1]+'-'+x[2] for x in item_['LIGAND']])[:20]
                else:
                    lig_str = item_['LIGAND'][0]+item_['LIGAND'][1]+'-'+item_['LIGAND'][2]
                name = item_['CHAINID']+'_asm'+str(int(item_['ASSEMBLY']))+'_'+lig_str
            elif task[0]=='sm':
                name = item_['label']
            else:
                name = item_['CHAINID']
            if save_pdbs:
                seq_unmasked = msa[:, 0, 0, :]
                writepdb(out_dir+f'ep{epoch}_{task[0]}_{counter}.{rank}_{name}_xyz_prev.pdb', 
                    torch.nan_to_num(xyz_prev_orig[res_mask][:,:23]), network_input['seq_unmasked'][res_mask],
                    bond_feats=network_input['bond_feats'][:, res_mask[0]][:, :, res_mask[0]])
                writepdb(out_dir+f'ep{epoch}_{task[0]}_{counter}.{rank}_{name}_xyz_true.pdb', 
                    torch.nan_to_num(true_crds[res_mask][:,:23]), network_input['seq_unmasked'][res_mask], 
                    bond_feats=network_input['bond_feats'][:, res_mask[0]][:, :, res_mask[0]])
                writepdb(out_dir+f'ep{epoch}_{task[0]}_{counter}.{rank}_{name}_xyz_pred.pdb', 
                    torch.nan_to_num(pred_allatom[res_mask][:,:23]), network_input['seq_unmasked'][res_mask], 
                    bond_feats=network_input['bond_feats'][:, res_mask[0]][:, :, res_mask[0]])

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

        out_dir = self.out_dir+f'/valid_ep{epoch}/'
        os.makedirs(out_dir, exist_ok=True)

        if self.eval:
            records = []

        if rank == 0 and self.debug_mode:
            valid_iter = tqdm(valid_loader, desc=header)
        else:
            valid_iter = valid_loader
            
        with torch.no_grad(): # no need to calculate gradient
            ddp_model.eval() # change it to eval mode
            for inputs in valid_iter:
                (
                    task, item, network_input, xyz_prev, alpha_prev, mask_recycle, 
                    true_crds, mask_crds, msa, mask_msa, unclamp, negative, symmRs, Lasu, ch_label
                ) = self._prepare_input(inputs, gpu)

                #r = rng.rand()
                save_pdbs = False #r<0.0

                counter += 1

                N_cycle = self.maxcycle # number of recycling

                output_i = (None, None, xyz_prev, None, alpha_prev, mask_recycle)
                for i_cycle in range(N_cycle):
                    with ExitStack() as stack:
                        stack.enter_context(torch.cuda.amp.autocast(enabled=USE_AMP))
                        stack.enter_context(ddp_model.no_sync())
                        use_checkpoint=False
                        if i_cycle < N_cycle -1:
                            return_raw=True
                        else:
                            return_raw=False
                        
                        input_i = self._get_model_input(network_input, output_i, i_cycle, return_raw=return_raw)
                        output_i = ddp_model(**input_i)

                        if i_cycle < N_cycle - 1:
                            continue

                loss, loss_dict, acc_s, _, true_crds, pred_allatom, res_mask = self._get_loss_and_misc(
                    output_i,
                    true_crds, mask_crds, network_input['same_chain'],
                    network_input['seq'], msa[:,i_cycle], mask_msa[:,i_cycle],
                    network_input['idx'], network_input['bond_feats'], network_input['dist_matrix'], network_input['atom_frames'],
                    unclamp, negative, task, item, symmRs, Lasu, ch_label,
                    len(valid_loader)*rank+counter
                )
                
                valid_tot += loss.detach()
                if valid_loss is None:
                    valid_loss = torch.zeros_like(torch.stack(list(loss_dict.values())))
                    valid_acc = torch.zeros_like(acc_s.detach())
                valid_loss += torch.stack(list(loss_dict.values()))
                valid_acc += acc_s.detach()
                
                # record results
                item_ = unbatch_item(item)
                if task[0].startswith('sm_compl') or task[0].startswith('metal_compl'):
                    if type(item_['LIGAND'][0]) is list: # multires or covalent ligands
                        lig_str = '_'.join([x[0]+x[1]+'-'+x[2] for x in item_['LIGAND']])[:20]
                    else:
                        lig_str = item_['LIGAND'][0]+item_['LIGAND'][1]+'-'+item_['LIGAND'][2]
                    name = item_['CHAINID']+'_asm'+str(int(item_['ASSEMBLY']))+'_'+lig_str
                elif task[0]=='sm':
                    name = item_['label']
                else:
                    name = item_['CHAINID']
                    
                if save_pdbs:
                    seq_unmasked = msa[:, 0, 0, :]
                    writepdb(out_dir+f'ep{epoch}_{task[0]}_{counter}.{rank}_{name}.pdb',
                        torch.nan_to_num(true_crds[res_mask][:,:23]), seq_unmasked[res_mask],
                        bond_feats=bond_feats[:,res_mask[0]][:,:,res_mask[0]],
                        chain="A", atom_mask=atom_mask[res_mask])

                    pred_sup = superimpose(torch.nan_to_num(pred_allatom[:,res_mask[0],:23]),
                                           torch.nan_to_num(true_crds_[:,res_mask[0],:23]),
                                           atom_mask[:,res_mask[0],:23])
                    writepdb(out_dir+f'ep{epoch}_{task[0]}_{counter}.{rank}_{name}.pdb',
                        pred_sup, seq_unmasked[res_mask],
                        bond_feats=bond_feats[:,res_mask[0]][:,:,res_mask[0]], 
                        chain="B", file_mode='a', atom_mask=atom_mask[res_mask],
                        atom_idx_offset=int(atom_mask[res_mask].sum().item()))

                if self.eval:
                    record = OrderedDict(name = name, Header=header, task = task[0], epoch = epoch)
                    record.update({k:float(v) for k,v in loss_dict.items()})
                    logit_pae_ = logit_pae[...,res_mask[0]][...,res_mask[0],:] if logit_pae is not None else None
                    logit_pde_ = logit_pde[...,res_mask[0]][...,res_mask[0],:] if logit_pde is not None else None
                    pred_err = self.calc_pred_err(pred_lddts, logit_pae_, logit_pde_, 
                                                  seq_unmasked[0,res_mask[0]]) 
                    record.update(pred_err)

                    is_binder_label = int(not negative)
                    record["is_binder_label"] = is_binder_label
                    record["binding_probability"] = binding_probabilities.item()

                    records.append(record)

                    #torch.save({'logits_pae': logit_pae_,
                    #            'logits_pde': logit_pde_,
                    #            'pred_lddts': pred_lddts[...,res_mask[0]]},
                    #           out_dir+f'ep{epoch}_{task[0]}_{counter}.{rank}_{name}_outputs.pt')

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

    def valid_ppi_cycle(self, ddp_model, valid_pos_loader, valid_neg_loader, rank, gpu, world_size, epoch, rng, header='Protein', verbose=False, print_header=False):
        if len(valid_pos_loader) == 0 or len(valid_neg_loader) == 0:
            # Note: you need both your positive and negative
            # validation sets to have examples, otherwise this
            # function does not make sense!
            return None, None, None, None, None
        
        valid_tot = 0.0
        valid_loss = None
        valid_acc = None
        valid_inter = None
        counter = 0
        
        start_time = time.time()

        if rank == 0 and self.debug_mode:
            valid_pos_iter = tqdm(valid_pos_loader, desc=f"{header} (positives)")
        else:
            valid_pos_iter = valid_pos_loader

        with torch.no_grad(): # no need to calculate gradient
            ddp_model.eval() # change it to eval mode
            for inputs in valid_pos_iter:
                (
                    task, item, network_input, xyz_prev, alpha_prev, mask_recycle, 
                    true_crds, mask_crds, msa, mask_msa, unclamp, negative, symmRs, Lasu, ch_label
                ) = self._prepare_input(inputs, gpu)

                counter += 1

                N_cycle = self.maxcycle # number of recycling

                output_i = (None, None, xyz_prev, None, alpha_prev, mask_recycle)
                for i_cycle in range(N_cycle): 
                    with ExitStack() as stack:
                        stack.enter_context(torch.cuda.amp.autocast(enabled=USE_AMP))
                        stack.enter_context(ddp_model.no_sync())
                        use_checkpoint=False
                        if i_cycle < N_cycle - 1:
                            return_raw=True
                        else:
                            return_raw=False
                        
                        input_i = self._get_model_input(network_input, output_i, i_cycle, return_raw=return_raw)
                        output_i = ddp_model(**input_i)

                        if i_cycle < N_cycle-1:
                            continue

                        loss, loss_dict, acc_s, p_bind, _, _, _ = self._get_loss_and_misc(
                            output_i,
                            true_crds, mask_crds, network_input['same_chain'],
                            network_input['seq'], msa[:,i_cycle], mask_msa[:,i_cycle],
                            network_input['idx'], network_input['bond_feats'], network_input['dist_matrix'], network_input['atom_frames'],
                            unclamp, negative, task, item, symmRs, Lasu, ch_label,
                            len(valid_pos_loader)*rank+counter
                        )

                        if (p_bind>0.5):
                            TP,FN = 1.0, 0.0
                        else:
                            TP,FN = 0.0, 1.0
                        inter_s = torch.tensor([TP, 0.0, 0.0, FN], device=p_bind.device).float()

                valid_tot += loss.detach()
                if valid_loss == None:
                    valid_loss = torch.zeros_like(torch.stack(list(loss_dict.values())))
                    valid_acc = torch.zeros_like(acc_s.detach())
                    valid_inter = torch.zeros_like(inter_s.detach())
                valid_loss += torch.stack(list(loss_dict.values()))
                valid_acc += acc_s.detach()
                valid_inter += inter_s.detach()
            
        valid_tot /= float(counter*world_size)
        valid_loss /= float(counter*world_size)
        valid_acc /= float(counter*world_size)
        
        dist.all_reduce(valid_tot, op=dist.ReduceOp.SUM)
        dist.all_reduce(valid_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(valid_acc, op=dist.ReduceOp.SUM)
       
        valid_tot = valid_tot.cpu().detach().numpy()
        valid_loss = valid_loss.cpu().detach().numpy()
        valid_acc = valid_acc.cpu().detach().numpy()
        
        loss_df_pos = None

        if rank == 0:
            if self.wandb_prefix is not None:
                log_dict = {f"Valid_{header}":{task[0]:loss_dict}}
                wandb.log(log_dict)
            train_time = time.time() - start_time

            # print loss names
            if print_header:
                sys.stdout.write(f'Header: [epoch/num_epochs] Batch: [examples_seen_in_epoch/examples_per_epoch] Time: time | Total_loss: total_loss | {" ".join(loss_dict.keys())} | precision recall F1 | max_mem \n')

            train_time = time.time() - start_time
            sys.stdout.write("%s: [%04d/%04d] Batch: [%05d/%05d] Time: %16.1f | total_loss: %8.4f | %s | %.4f %.4f %.4f\n"%(\
                    header, epoch, self.n_epoch, len(valid_pos_loader)*world_size, len(valid_pos_loader)*world_size, train_time, valid_tot, \
                    " ".join(["%8.4f"%l for l in valid_loss]),\
                    valid_acc[0], valid_acc[1], valid_acc[2])) 
            sys.stdout.flush()

            if self.eval:
                # save per-example losses
                all_records_ = []
                for records in all_records:
                    all_records_.extend(records)
                loss_df_pos = pd.DataFrame.from_records(all_records_)

        valid_tot = 0.0
        valid_loss = None
        valid_acc = None
        counter = 0

        if rank == 0 and self.debug_mode:
            valid_neg_iter = tqdm(valid_neg_loader, desc=f"{header} (negatives)")
        else:
            valid_neg_iter = valid_neg_loader
        
        start_time = time.time()

        with torch.no_grad(): # no need to calculate gradient
            ddp_model.eval() # change it to eval mode
            for inputs in valid_neg_iter: 
                (
                    task, item, network_input, xyz_prev, alpha_prev, mask_recycle, 
                    true_crds, mask_crds, msa, mask_msa, unclamp, negative, symmRs, Lasu, ch_label
                ) = self._prepare_input(inputs, gpu)

                counter += 1

                N_cycle = self.maxcycle # number of recycling
                
                output_i = (None, None, xyz_prev, None, alpha_prev, mask_recycle)
                for i_cycle in range(N_cycle): 
                    with ExitStack() as stack:
                        stack.enter_context(torch.cuda.amp.autocast(enabled=USE_AMP))
                        stack.enter_context(ddp_model.no_sync())
                        if i_cycle < N_cycle - 1:
                            return_raw=True
                        else:
                            return_raw=False

                        input_i = self._get_model_input(network_input, output_i, i_cycle, return_raw=return_raw)
                        output_i = ddp_model(**input_i)

                        if i_cycle < N_cycle - 1:
                            continue

                        loss, loss_dict, acc_s, p_bind, _, _, _ = self._get_loss_and_misc(
                            output_i,
                            true_crds, mask_crds, network_input['same_chain'],
                            network_input['seq'], msa[:,i_cycle], mask_msa[:,i_cycle],
                            network_input['idx'], network_input['bond_feats'], network_input['dist_matrix'], network_input['atom_frames'],
                            unclamp, negative, task, item, symmRs, Lasu, ch_label,
                            len(valid_pos_loader)*rank+counter
                        )
                        if (p_bind>0.5):
                            FP,TN = 1.0, 0.0
                        else:
                            FP,TN = 0.0, 1.0
                        inter_s = torch.tensor([0.0, FP, TN, 0.0], device=p_bind.device).float()

                valid_tot += loss.detach()
                if valid_loss == None:
                    valid_loss = torch.zeros_like(torch.stack(list(loss_dict.values())))
                    valid_acc = torch.zeros_like(acc_s.detach())
                valid_loss += torch.stack(list(loss_dict.values()))
                valid_acc += acc_s.detach()
                valid_inter += inter_s.detach()

            
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

        loss_df_neg = None

        if rank == 0:
            TP, FP, TN, FN = valid_inter
            prec = TP/(TP+FP+1e-4)
            recall = TP/(TP+FN+1e-4)
            F1 = 2*TP/(2*TP+FP+FN+1e-4)
            
            train_time = time.time() - start_time
            sys.stdout.write("%s-discrim: [%04d/%04d] Batch: [%05d/%05d] Time: %16.1f | total_loss: %8.4f | %s | %.4f %.4f %.4f | %.4f %.4f %.4f\n"%(\
                    header, epoch, self.n_epoch, counter*world_size, counter*world_size, train_time, valid_tot, \
                    " ".join(["%8.4f"%l for l in valid_loss]),\
                    valid_acc[0], valid_acc[1], valid_acc[2],\
                    prec, recall, F1))
            sys.stdout.flush()

            if self.eval:
                # save per-example losses
                all_records_ = []
                for records in all_records:
                    all_records_.extend(records)
                loss_df_neg = pd.DataFrame.from_records(all_records_)

        return valid_tot, valid_loss, valid_acc, loss_df_pos, loss_df_neg



if __name__ == "__main__":
    from arguments import get_args
    args, dataset_param, model_param, loader_param, loss_param = get_args()

    if args.debug:
        DEBUG = True
        args.dataloader_num_workers = 0

    if "SLURM_PROCID" in os.environ:
        if int(os.environ["SLURM_PROCID"])==0:
            print (args)

    mp.freeze_support()

    dataloader_kwargs = {
        "shuffle": args.shuffle_dataloader,
        "num_workers": args.dataloader_num_workers,
        "pin_memory": not args.dont_pin_memory,
    }

    trainer_object = Trainer(
        model_name=args.model_name,
        n_epoch=args.num_epochs,
        step_lr=args.step_lr,
        lr=args.lr,
        l2_coeff=1.0e-2,
        port=args.port,
        model_param=model_param,
        loader_param=loader_param,
        loss_param=loss_param,
        batch_size=args.batch_size,
        accum_step=args.accum,
        maxcycle=args.maxcycle,
        eval=args.eval,
        interactive=args.interactive,
        out_dir=args.out_dir,
        wandb_prefix=args.wandb_prefix,
        model_dir=args.model_dir,
        dataset_param=dataset_param,
        dataloader_kwargs=dataloader_kwargs,
        debug_mode=args.debug,
        skip_valid=args.skip_valid,
    )
    trainer_object.run_model_training(torch.cuda.device_count())
