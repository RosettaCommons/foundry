import sys, os, json, pickle, glob
import time
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(script_dir)

import rf2aa.parsers as parsers
from rf2aa.RoseTTAFoldModel  import RoseTTAFoldModule
import rf2aa.util as util
from rf2aa.util import *
from rf2aa.loss import *
from collections import namedtuple, OrderedDict
from rf2aa.ffindex import *
from rf2aa.data_loader import MSAFeaturize, MSABlockDeletion, merge_a3m_homo, merge_a3m_hetero
from rf2aa.kinematics import xyz_to_c6d, c6d_to_bins, xyz_to_t2d, get_chirals
from rf2aa.util_module import ComputeAllAtomCoords
from rf2aa.chemical import NTOTAL, NTOTALDOFS, NAATOKENS, INIT_CRDS
from rf2aa.parsers import read_templates
from rf2aa.memory import mem_report
from scipy.interpolate import Akima1DInterpolator

def get_args():
    DB = "/projects/ml/TrRosetta/pdb100_2022Apr19/pdb100_2022Apr19"
    import argparse
    parser = argparse.ArgumentParser(description="RoseTTAFold: Protein structure prediction with 3-track attentions on 1D, 2D, and 3D features")
    parser.add_argument("-checkpoint", 
        #default="/home/jue/git/rf2a-fd3/folddock3_20221125/models/rf2a_fd3_20221125_199.pt",
        default='/home/jue/git/rf2a-fd3-ph3/folddock3_20221125/models/rf2a_fd3_20221125_469.pt',
        help="Path to model weights")

    parser.add_argument("-msa", help='Input sequence/MSA to predict structure from, in fasta/a3m format')
    parser.add_argument("-pdb", help='PDB of sequence to predict structure from')
    parser.add_argument("-hhr", help='Input hhr file.')
    parser.add_argument("-atab", help='Input atab file.')
    parser.add_argument("-pt", help='PyTorch cached version of PDB')
    parser.add_argument("-mol2", help='mol2 of small molecule to predict structure from')
    parser.add_argument("-smiles", help='smiles string of small molecule to predict structure from')
    parser.add_argument("-db", default=DB, required=False, help="HHsearch database [%s]"%DB)

    parser.add_argument("-list", help='list of PDB inputs')
    parser.add_argument("-folder", help='folder with PDB inputs')

    parser.add_argument("-outcsv", default='rfaa_scores.csv', help='output CSV for losses')
    parser.add_argument("-out", help='prefix of output files')

    parser.add_argument("-dump_extra_pdbs", action='store_true', default=False, help='output initial and final prediction in addition to best prediction')
    parser.add_argument("-dump_traj", action='store_true', default=False, help='output trajectory pdb')
    parser.add_argument("-dump_aux", action='store_true', default=False, help='output distograms/anglegrams and confidence estimates')
    parser.add_argument("-init_protein_tmpl", action='store_true', default=False, help='initialize protein template structure to ground truth')
    parser.add_argument("-init_ligand_tmpl", action='store_true', default=False, help='initialize ligand template structure to ground truth')
    parser.add_argument("-init_protein_xyz", action='store_true', default=False, help='initialize protein coordinates to ground truth')
    parser.add_argument("-init_ligand_xyz", action='store_true', default=False, help='initialize ligand coordinates to ground truth')
    parser.add_argument("-num_interp", type=int, default=5, help='number of interpolation frames for trajectory output')
    parser.add_argument("-parse_hetatm", action="store_true", default=False, help="parse ligand information from input pdb")
    parser.add_argument("-n_pred", type=int, default=1, help='number of repeat predictions')
    parser.add_argument("-n_cycle", type=int, default=10, help='number of recycles')
    parser.add_argument("-trunc_N", type=int, default=0, help='residues to truncate at N-term on MSA to match PDB')
    parser.add_argument("-trunc_C", type=int, default=0, help='residues to truncate at C-term on MSA to match PDB')
    parser.add_argument("-no_extra_l1", dest='use_extra_l1', default='True', action='store_false',
            help="Turn off chirality and LJ grad inputs to SE3 layers (for backwards compatibility).")
    parser.add_argument("-no_atom_frames", dest='use_atom_frames', default='True', action='store_false',
            help="Turn off l1 features from atom frames in SE3 layers (for backwards compatibility).")

    args = parser.parse_args()

    return args


MODEL_PARAM ={
        "n_extra_block"   : 4,
        "n_main_block"    : 32,
        "n_ref_block"     : 4,
        "d_msa"           : 256,
        "d_pair"          : 192,
        "d_templ"         : 64,
        "n_head_msa"      : 8,
        "n_head_pair"     : 6,
        "n_head_templ"    : 4,
        "d_hidden"        : 32,
        "d_hidden_templ"  : 64,
        "p_drop"       : 0.0,
        "lj_lin"       : 0.7,
        'symmetrize_repeats': False,
        'repeat_length': float('nan'),
        'symmsub_k': float('nan'),
        'sym_method': float('nan'),
        'main_block': float('nan'),
        'copy_main_block_template': False
        }

SE3_param = {
        "num_layers"    : 1,
        "num_channels"  : 32,
        "num_degrees"   : 2,
        "l0_in_features": 64,
        "l0_out_features": 64,
        "l1_in_features": 3,
        "l1_out_features": 2,
        "num_edge_features": 64,
        "div": 4,
        "n_heads": 4
        }
SE3_ref_param = {
        "num_layers"    : 2,
        "num_channels"  : 32,
        "num_degrees"   : 2,
        "l0_in_features": 64,
        "l0_out_features": 64,
        "l1_in_features": 3,
        "l1_out_features": 2,
        "num_edge_features": 64,
        "div": 4,
        "n_heads": 4
        }
MODEL_PARAM['SE3_param'] = SE3_param
MODEL_PARAM['SE3_ref_param'] = SE3_ref_param


# compute expected value from binned lddt
def lddt_unbin(pred_lddt):
    nbin = pred_lddt.shape[1]
    bin_step = 1.0 / nbin
    lddt_bins = torch.linspace(bin_step, 1.0, nbin, dtype=pred_lddt.dtype, device=pred_lddt.device)
    
    pred_lddt = nn.Softmax(dim=1)(pred_lddt)
    return torch.sum(lddt_bins[None,:,None]*pred_lddt, dim=1)


def get_msa(a3mfilename):                                                                       
    msa,ins, _ = parsers.parse_a3m(a3mfilename, unzip='.gz' in a3mfilename)
    return {'msa':torch.tensor(msa), 'ins':torch.tensor(ins)}


class Predictor():
    def __init__(self, args, device="cuda:0"):
        # define model name
        self.device = device
        self.active_fn = nn.Softmax(dim=1)

        FFindexDB = namedtuple("FFindexDB", "index, data")
        self.ffdb = FFindexDB(read_index(args.db+'_pdb.ffindex'),
                              read_data(args.db+'_pdb.ffdata'))

        # define model & load model
        MODEL_PARAM['use_extra_l1'] = args.use_extra_l1
        MODEL_PARAM['use_atom_frames'] = args.use_atom_frames
        self.model = RoseTTAFoldModule(
            **MODEL_PARAM,
            aamask = util.allatom_mask.to(self.device),
            atom_type_index = util.atom_type_index.to(self.device),
            ljlk_parameters = util.ljlk_parameters.to(self.device),
            lj_correction_parameters = util.lj_correction_parameters.to(self.device),
            num_bonds = util.num_bonds.to(self.device),
            cb_len = util.cb_length_t.to(self.device),
            cb_ang = util.cb_angle_t.to(self.device),
            cb_tor = util.cb_torsion_t.to(self.device),
        ).to(self.device)

        checkpoint = torch.load(args.checkpoint, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.compute_allatom_coords = ComputeAllAtomCoords().to(self.device)

        # loss & final activation function
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.active_fn = nn.Softmax(dim=1)

        # move some global data to cuda device
        self.ti_dev = torsion_indices.to(device)
        self.ti_flip = torsion_can_flip.to(device)
        self.ang_ref = reference_angles.to(device)
        self.fi_dev = frame_indices.to(device)
        self.l2a = long2alt.to(device)
        self.aamask = allatom_mask.to(device)
        self.atom_type_index = atom_type_index.to(device)
        self.ljlk_parameters = ljlk_parameters.to(device)
        self.lj_correction_parameters = lj_correction_parameters.to(device)
        self.hbtypes = hbtypes.to(device)
        self.hbbaseatoms = hbbaseatoms.to(device)
        self.hbpolys = hbpolys.to(device)
        self.num_bonds = num_bonds.to(self.device),
        self.cb_len = cb_length_t.to(self.device),
        self.cb_ang = cb_angle_t.to(self.device),
        self.cb_tor = cb_torsion_t.to(self.device),

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
            else: # apply anglegram loss only when both residues have valid BB frames (i.e. not metal ions)
                bb_frame_good = frame_mask[:,:,0]
                loss_mask_2d = bb_frame_good & bb_frame_good[...,None]
                mask_2d_ = mask_2d & loss_mask_2d
            loss = (mask_2d_*loss).sum() / (mask_2d_.sum() + eps)
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
        # residue mask for FAPE calculation only masks unresolved protein backbone atoms
        # whereas other losses also maks unresolved ligand atoms (mask_BB)
        # frames with unresolved ligand atoms are masked in compute_general_FAPE
        res_mask = ~((mask_crds[:,:,:3].sum(dim=-1) < 3.0) * ~(is_atom(seq)))

        L1 = same_chain[0,0,:].sum()
        res_mask_A = res_mask.clone()
        res_mask_A[0, L1:] = False
        if torch.sum(res_mask_A)>0 and torch.sum(frame_mask_BB[:,res_mask_A[0]])>0:
            l_fape_A, _, _ = compute_general_FAPE(
                pred[:,res_mask_A,:,:3],
                true[:,res_mask_A[0],:3],
                mask_crds[:,res_mask_A[0], :3],
                frames_BB[:,res_mask_A[0]],
                frame_mask_BB[:,res_mask_A[0]],
                dclamp=dclamp
            )
        else:
            l_fape_A = torch.tensor([0]).to(gpu)
        loss_dict['bb_fape_c1'] = l_fape_A[-1].detach()

        res_mask_B = res_mask.clone()
        res_mask_B[0,:L1] = False
        if torch.sum(res_mask_B)>0 and torch.sum(frame_mask_BB[:,res_mask_B[0]])>0:
            l_fape_B, _, _ = compute_general_FAPE(
                pred[:, res_mask_B,:,:3],
                true[:,res_mask_B[0],:3,:3],
                mask_crds[:,res_mask_B[0], :3],
                frames_BB[:,res_mask_B[0]],
                frame_mask_BB[:,res_mask_B[0]],
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
                logit_pae = logit_pae[:,:,res_mask[0]][:,:,:,res_mask[0]]
            if logit_pde is not None:
                logit_pde = logit_pde[:,:,res_mask[0]][:,:,:,res_mask[0]]
            tot_str, pae_loss, pde_loss = compute_general_FAPE(
                pred[:,res_mask,:,:3],
                true[:,res_mask[0],:3],
                mask_crds[:,res_mask[0],:3],
                frames_BB[:,res_mask[0]],
                frame_mask_BB[:,res_mask[0]],
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

        # small-molecule ligands
        sm_res_mask = is_atom(label_aa_s[0,0])*res_mask[0] # (L,)
        if bool(torch.any(sm_res_mask)) and torch.any(frame_mask_BB[0,sm_res_mask]):
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
        loss_dict['bb_fape_lig'] = lig_fape.detach()

        if not bool(torch.all(sm_res_mask)) and bool(torch.any(sm_res_mask)):    # not all atoms but some atoms
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
            # res_mask_A = mask_BB.clone()
            # res_mask_A[0, L1:] = False
            l_fape_A, _, _ = compute_general_FAPE(
                pred_allatom[:,res_mask_A[0],:,:3],
                nat_symm[None,res_mask_A[0],:,:3],
                xs_mask[:,res_mask_A[0]],
                frames[:,res_mask_A[0]],
                frame_mask[:,res_mask_A[0]]
            )
            # res_mask_B = mask_BB.clone()
            # res_mask_B[0,:L1] = False
            l_fape_B, _, _ = compute_general_FAPE(
                pred_allatom[:,res_mask_B[0],:,:3],
                nat_symm[None,res_mask_B[0],:,:3],
                xs_mask[:,res_mask_B[0]],
                frames[:,res_mask_B[0]],
                frame_mask[:,res_mask_B[0]]
            )
            fracA = float(L1)/len(same_chain[0,0])
            l_fape = fracA*l_fape_A + (1.0-fracA)*l_fape_B

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
        try:
            rmsd = calc_crd_rmsd(
                pred_allatom[:,mask_BB[0],:,:3],
                nat_symm[None,mask_BB[0],:,:3],
                xs_mask[:,mask_BB[0]]
                )
            loss_dict["rmsd"] = rmsd[0].detach()
        except Exception as e:
            print('calc_crd_rmsd failed on ',item)
            rmsd = torch.tensor([0])
            loss_dict['rmsd'] = torch.tensor([0])

        if torch.any(res_mask_B):
            xs_mask_c1, xs_mask_c2 = xs_mask.clone(), xs_mask.clone()
            xs_mask_c1[:,~res_mask_A[0]] = False
            xs_mask_c2[:,~res_mask_B[0]] = False
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
            loss_dict['clash_loss'] = ( bond_loss.detach() )
        else:
            bond_loss = torch.tensor(0).to(gpu)
        loss_dict['bond_loss'] = bond_loss.detach()

        # clash [use all atoms not just those in native]
        #clash_loss = calc_lj(
        #    seq[0], pred_allatom,
        #    self.aamask, bond_feats, self.ljlk_parameters, self.lj_correction_parameters, self.num_bonds,
        #    lj_lin=lj_lin
        #)
        #if w_clash > 0.0:
        #    tot_loss += w_clash*clash_loss.mean()
        #loss_dict['clash_loss'] = clash_loss[0].detach()
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

        #if (verbose):
            #print (
            #    ctr,
            #    tot_str.cpu().detach().numpy(),
            #    allatom_lddt.cpu().detach().numpy(),
            #    allatom_lddt_c2.cpu().detach().numpy(),
            #    l_fape.cpu().detach().numpy(),
            #    l_fape_B.cpu().detach().numpy(),
            #    mask_BB[0].sum()
            #)
            #writepdb(out_dir+"p_"+self.model_name+"_"+str(ctr)+".pdb", pred_all[-1,mask_BB[0]][:,:23], seq[mask_BB][:])
            #writepdb(out_dir+"n_"+str(ctr)+".pdb", true[mask_BB][:,:23], seq[mask_BB][:])
            #writepdb(out_dir+"nre_"+str(ctr)+".pdb", _n0[mask_BB], seq[mask_BB][:])

        return tot_loss, loss_dict

    def predict(self, out_prefix, msa_fn=None, pdb_fn=None, pt_fn=None, 
        a3m_fn=None, hhr_fn=None, atab_fn=None, mol2_fn=None,
        init_protein_tmpl=False, init_ligand_tmpl=False, init_protein_xyz=False,
        init_ligand_xyz=False, n_cycle=10, n_templ=4, random_noise=5.0, trunc_N=0,
        trunc_C=0):

        has_ligand = False
        if pdb_fn is not None:
            xyz_prot, mask_prot, idx_prot, seq_prot = parsers.parse_pdb(pdb_fn, seq=True)
            xyz_prot[:,14:] = 0 # remove hydrogens
            mask_prot[:,14:] = False
            xyz_prot = torch.tensor(xyz_prot)
            mask_prot = torch.tensor(mask_prot)
            protein_L, nprotatoms, _ = xyz_prot.shape
            msa_prot = torch.tensor(seq_prot)[None].long()
            ins_prot = torch.zeros(msa_prot.shape).long()
            a3m_prot = {"msa": msa_prot, "ins": ins_prot}
            idx_prot = torch.tensor(idx_prot)

            stream = [l for l in open(pdb_fn) if "HETATM" in l or "CONECT" in l]
            if len(stream)>0:
                mol, msa_sm, ins_sm, xyz_sm, mask_sm = parsers.parse_mol("".join(stream), filetype="pdb", string=True)
                a3m_sm = {"msa": msa_sm.unsqueeze(0), "ins": ins_sm.unsqueeze(0)}
                G = util.get_nxgraph(mol)
                atom_frames = util.get_atom_frames(msa_sm, G)
                N_symmetry, sm_L, _ = xyz_sm.shape
                Ls = [protein_L, sm_L]
                a3m = merge_a3m_hetero(a3m_prot, a3m_sm, Ls)
                msa = a3m['msa'].long()
                ins = a3m['ins'].long()
                chirals = get_chirals(mol, xyz_sm[0]) 
                has_ligand = True

        if pt_fn is not None:
            pdbA = torch.load(pt_fn)
            xyz_prot, mask_prot = pdbA["xyz"], pdbA["mask"]
            alphabet = 'ARNDCQEGHILKMFPSTWYV'
            aa_1_N = dict(zip(list(alphabet),range(len(alphabet))))
            msa_prot = torch.tensor([aa_1_N[a] for a in pdbA['seq']])[None]
            ins_prot = torch.zeros(msa_prot.shape).long()

        if msa_fn is not None:
            a3m = get_msa(msa_fn)
            msa_prot = a3m['msa'].clone().long()
            qlen = msa_prot.shape[1]
            msa_prot = msa_prot[:,trunc_N:qlen-trunc_C]
            ins_prot = a3m['ins'].clone().long()[:,trunc_N:qlen-trunc_C]
            protein_L = msa_prot.shape[-1]
            idx_prot = torch.arange(protein_L)

        if mol2_fn is not None:
            a3m_prot = {"msa": msa_prot, "ins": ins_prot}
            mol, msa_sm, ins_sm, xyz_sm, mask_sm = parsers.parse_mol(mol2_fn)
            a3m_sm = {"msa": msa_sm.unsqueeze(0), "ins": ins_sm.unsqueeze(0)}
            G = util.get_nxgraph(mol)
            atom_frames = util.get_atom_frames(msa_sm, G)
            N_symmetry, sm_L, _ = xyz_sm.shape

            Ls = [protein_L, sm_L]
            a3m = merge_a3m_hetero(a3m_prot, a3m_sm, Ls)
            msa = a3m['msa'].long()
            ins = a3m['ins'].long()
            chirals = get_chirals(mol, xyz_sm[0])
            has_ligand = True

        if not has_ligand:
            Ls = [msa_prot.shape[-1], 0]
            N_symmetry = 1
            msa = msa_prot
            ins = ins_prot
            chirals = torch.Tensor()
            atom_frames = torch.zeros(msa[:,0].shape)

        xyz = torch.full((N_symmetry, sum(Ls), NTOTAL, 3), np.nan).float()
        mask = torch.full(xyz.shape[:-1], False).bool()
        if pdb_fn is not None:
            xyz[:, :Ls[0], :nprotatoms, :] = xyz_prot.expand(N_symmetry, Ls[0], nprotatoms, 3)
            mask[:, :protein_L, :nprotatoms] = mask_prot.expand(N_symmetry, Ls[0], nprotatoms)
        if has_ligand:
            xyz[:, Ls[0]:, 1, :] = xyz_sm
            mask[:, protein_L:, 1] = mask_sm
        idx_sm = torch.arange(max(idx_prot),max(idx_prot)+Ls[1])+200
        idx_pdb = torch.concat([idx_prot.clone(), idx_sm])
        
        seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, 
            p_mask=0.0, params={'MAXLAT': 128, 'MAXSEQ': 1024, 'MAXCYCLE': n_cycle}, tocpu=True)

        chain_idx = torch.zeros((sum(Ls), sum(Ls))).long()
        chain_idx[:Ls[0], :Ls[0]] = 1
        chain_idx[Ls[0]:, Ls[0]:] = 1
        bond_feats = torch.zeros((sum(Ls), sum(Ls))).long()
        bond_feats[:Ls[0], :Ls[0]] = util.get_protein_bond_feats(Ls[0])
        if has_ligand:
            bond_feats[Ls[0]:, Ls[0]:] = util.get_bond_feats(mol)

        if init_protein_tmpl or init_ligand_tmpl:
            # make blank features for 2 templates
            xyz_t = torch.full((2,sum(Ls),NTOTAL,3),np.nan).float()
            f1d_t = torch.cat((
                torch.nn.functional.one_hot(
                    torch.full((2, sum(Ls)), 20).long(),
                    num_classes=NAATOKENS-1).float(), # all gaps (no mask token)
                torch.zeros((2, sum(Ls), 1)).float()
            ), -1) # (2, L_protein + L_sm, NAATOKENS)
            mask_t = torch.full((2, sum(Ls), NTOTAL), False)

            if init_protein_tmpl: # input true protein xyz as template 0
                xyz_t[0, :Ls[0], :14] = xyz[0, :Ls[0], :14]
                f1d_t[0, :Ls[0]] = torch.cat((
                    torch.nn.functional.one_hot(msa[0, :Ls[0] ], num_classes=NAATOKENS-1).float(),
                    torch.ones((Ls[0], 1)).float()
                ), -1) # (1, L_protein, NAATOKENS)
                mask_t[0, :Ls[0], :nprotatoms] = mask_prot

            if init_ligand_tmpl: # input true s.m. xyz as template 1
                xyz_t[1, Ls[0]:, :14] = xyz[0, Ls[0]:, :14]
                f1d_t[1, Ls[0]:] = torch.cat((
                    torch.nn.functional.one_hot(msa[0, Ls[0]: ]-1, num_classes=NAATOKENS-1).float(),
                    torch.ones((Ls[1], 1)).float()
                ), -1) # (1, L_sm, NAATOKENS)
                mask_t[1, Ls[0]:, 1] = mask_sm[0] # all symmetry variants have same mask
        elif hhr_fn is not None:
            # templates from file
            xyz_t_prot, mask_t_prot, t1d_prot = read_templates(qlen, self.ffdb, hhr_fn,
                                                               atab_fn, n_templ=n_templ)
            xyz_t_prot = xyz_t_prot[:,trunc_N:qlen-trunc_C]
            mask_t_prot = mask_t_prot[:,trunc_N:qlen-trunc_C]
            t1d_prot = t1d_prot[:,trunc_N:qlen-trunc_C]

            # blank templates to include ligand
            xyz_t = torch.full((n_templ,sum(Ls),NTOTAL,3),np.nan).float()
            f1d_t = torch.cat((
                torch.nn.functional.one_hot(
                    torch.full((n_templ, sum(Ls)), 20).long(),
                    num_classes=NAATOKENS-1).float(), # all gaps (no mask token)
                torch.zeros((n_templ, sum(Ls), 1)).float()
            ), -1) # (n_templ, L_protein + L_sm, NAATOKENS)
            mask_t = torch.full((n_templ, sum(Ls), NTOTAL), False)

            xyz_t[:, :Ls[0], :14] = xyz_t_prot[:, :, :14]
            mask_t[:, :Ls[0], :14] = mask_t_prot[:, :, :14]
            f1d_t[:, :Ls[0]] = t1d_prot
        else:
            # blank template
            xyz_t = INIT_CRDS.reshape(1,1,NTOTAL,3).repeat(1,sum(Ls),1,1) \
                + torch.rand(1,sum(Ls),1,3)*random_noise - random_noise/2
            f1d_t = torch.nn.functional.one_hot(torch.full((1, sum(Ls)), 20).long(), num_classes=NAATOKENS-1).float() # all gaps
            conf = torch.zeros((1, sum(Ls), 1)).float()
            f1d_t = torch.cat((f1d_t, conf), -1)
            mask_t = torch.full((1,sum(Ls),NTOTAL), False)

        if init_protein_xyz or init_ligand_xyz:
            # initialize coords to ground truth
            xyz_prev = torch.full((sum(Ls), NTOTAL, 3), np.nan).float()
            mask_prev = torch.full((sum(Ls), NTOTAL), False)

            com = xyz[0,:,1].nanmean(0)
            if init_protein_xyz:
                xyz1 = xyz[0, :Ls[0]]
                xyz_prev[:Ls[0]] = xyz1 - com
                mask_prev[:Ls[0]] = mask[0,:Ls[0]]
            if init_ligand_xyz:
                xyz2 = xyz[0, Ls[0]:]
                xyz_prev[Ls[0]:] = xyz2 - com
                mask_prev[Ls[0]:] = mask[0,Ls[0]:]

            # initialize missing positions in ground truth structures
            init = INIT_CRDS.reshape(1,NTOTAL,3).repeat(sum(Ls),1,1)
            init = init + torch.rand(sum(Ls),1,3)*random_noise - random_noise/2
            xyz_prev = torch.where(mask_prev[:,:,None], xyz_prev, init).contiguous()

        else:
            init = INIT_CRDS.reshape(1,NTOTAL,3).repeat(sum(Ls),1,1) + \
                   torch.rand(sum(Ls),1,3)*random_noise - random_noise/2
            mask_prev = mask_t[0].clone()
            xyz_prev = torch.where(mask_prev[:,:,None], xyz_t[0].clone(), init).contiguous()

        xyz = torch.nan_to_num(xyz)
        xyz_t = torch.nan_to_num(xyz_t)

        seq = seq[None].to(self.device, non_blocking=True)
        msa = msa_seed_orig[None].to(self.device, non_blocking=True)
        msa_masked = msa_seed[None].to(self.device, non_blocking=True)
        msa_full = msa_extra[None].to(self.device, non_blocking=True)
        true_crds = xyz[None].to(self.device, non_blocking=True) # (B, L, 27, 3)
        atom_mask = mask[None].to(self.device, non_blocking=True) # (B, L, 27)
        idx_pdb = idx_pdb[None].to(self.device, non_blocking=True) # (B, L)
        xyz_t = xyz_t[None].to(self.device, non_blocking=True)
        mask_t = mask_t[None].to(self.device, non_blocking=True)
        t1d = f1d_t[None].to(self.device, non_blocking=True)
        xyz_prev = xyz_prev[None].to(self.device, non_blocking=True)
        mask_prev = mask_prev[None].to(self.device, non_blocking=True)
        same_chain = chain_idx[None].to(self.device, non_blocking=True)
        atom_frames = atom_frames[None].to(self.device, non_blocking=True)
        bond_feats = bond_feats[None].to(self.device, non_blocking=True)
        chirals = chirals[None].to(self.device, non_blocking=True)
        xyz_prev_orig = xyz_prev.clone()

        # transfer inputs to device
        B, _, N, L = msa.shape

        # processing template features
        seq_unmasked = msa[:, 0, 0, :] # (B, L)
        mask_t_2d = util.get_prot_sm_mask(mask_t, seq_unmasked[0]) # (B, T, L)
        mask_t_2d = mask_t_2d[:,:,None]*mask_t_2d[:,:,:,None] # (B, T, L, L)
        mask_t_2d = mask_t_2d.float() * same_chain.float()[:,None] # (ignore inter-chain region)
        mask_recycle = util.get_prot_sm_mask(mask_prev, seq_unmasked[0])
        mask_recycle = mask_recycle[:,:,None]*mask_recycle[:,None,:] # (B, L, L)
        mask_recycle = same_chain.float()*mask_recycle.float()

        xyz_t_frames = util.xyz_t_to_frame_xyz(xyz_t, seq_unmasked, atom_frames)
        t2d = xyz_to_t2d(xyz_t_frames, mask_t_2d)

        seq_tmp = t1d[...,:-1].argmax(dim=-1).reshape(-1,sum(Ls))

        alpha, _, alpha_mask, _ = util.get_torsions(
            xyz_t.reshape(-1,sum(Ls),NTOTAL,3),
            seq_tmp,
            util.torsion_indices.to(self.device),
            util.torsion_can_flip.to(self.device),
            util.reference_angles.to(self.device)
        )
        alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))
        alpha[torch.isnan(alpha)] = 0.0
        alpha = alpha.reshape(1,-1,sum(Ls),NTOTALDOFS,2)
        alpha_mask = alpha_mask.reshape(1,-1,sum(Ls),NTOTALDOFS,1)
        alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(1, -1, sum(Ls), 3*NTOTALDOFS).to(self.device)

        start = time.time()
        torch.cuda.reset_peak_memory_stats()
        self.model.eval()
        all_pred = []
        all_pred_allatom = []
        with torch.no_grad():
            msa_prev = None
            pair_prev = None
            alpha_prev = torch.zeros((1,L,NTOTALDOFS,2), device=seq.device)
            state_prev = None

            best_lddt = torch.tensor([-1.0], device=seq.device)
            best_xyz = None
            best_logit = None
            best_aa = None
            best_pae = None
            best_pde = None

            for i_cycle in range(n_cycle):
                logit_s, logit_aa_s, logit_pae, logit_pde, pred_crds, alpha, pred_allatom, pred_lddt_binned, \
                    msa_prev, pair_prev, state_prev, _ = self.model(
                    msa_masked[:,i_cycle], 
                    msa_full[:,i_cycle],
                    seq[:,i_cycle], 
                    msa[:,i_cycle,0], 
                    xyz_prev, 
                    alpha_prev,
                    idx_pdb,
                    bond_feats=bond_feats,
                    chirals=chirals,
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
                    mask_recycle=mask_recycle
                )

                logit_aa = logit_aa_s.reshape(B,-1,N,L)[:,:,0].permute(0,2,1)
                xyz_prev = pred_allatom[-1].unsqueeze(0)
                mask_recycle = None

                all_pred.append(pred_crds)
                all_pred_allatom.append(pred_allatom[-1])

                pred_lddt = lddt_unbin(pred_lddt_binned)
                if pred_lddt.mean() > best_lddt.mean():
                    best_xyz = xyz_prev.clone()
                    best_logit = logit_s
                    best_aa = logit_aa
                    best_lddt = pred_lddt.clone()
                    best_pae = logit_pae.detach().cpu().numpy()
                    best_pde = logit_pde.detach().cpu().numpy()

                print(f'RECYCLE {i_cycle}\tcurrent lddt: {pred_lddt.mean():.3f}\t'\
                      f'best lddt: {best_lddt.mean():.3f}')

            prob_s = list()
            for logit in logit_s:
                prob = self.active_fn(logit.float()) # distogram
                prob = prob.reshape(-1, L, L) #.permute(1,2,0).cpu().numpy()
                prob = prob / (torch.sum(prob, dim=0)[None]+1e-8)
                prob_s.append(prob)
        
        end = time.time()

        max_mem = torch.cuda.max_memory_allocated()/1e9
        print ("max mem", max_mem)
        print ("runtime", end-start)

        # output pdbs
        util.writepdb(out_prefix+".pdb", best_xyz[0], seq[0, -1], bfacts=100*best_lddt[0].float(), 
                      bond_feats=bond_feats)
        if args.dump_extra_pdbs:
            util.writepdb(out_prefix+"_last.pdb", xyz_prev[0], seq[0, -1], bfacts=100*best_lddt[0].float(),
                          bond_feats=bond_feats)
            util.writepdb(out_prefix+"_init.pdb", xyz_prev_orig[0], seq[0, -1], bond_feats=bond_feats)

        # output losses, model confidence
        xyz = xyz[None].to(self.device)
        mask = mask[None].to(self.device)
        mask_msa = mask_msa[None].to(self.device)

        true_crds_, atom_mask_ = resolve_equiv_natives(pred_crds[-1], xyz, mask)
        res_mask = get_prot_sm_mask(atom_mask_, msa[0,i_cycle,0])
        mask_2d = res_mask[:,None,:] * res_mask[:,:,None]

        true_crds_frame = xyz_to_frame_xyz(true_crds_, msa[:, i_cycle, 0],atom_frames)
        c6d = xyz_to_c6d(true_crds_frame)
        c6d = c6d_to_bins(c6d, same_chain, negative=False).to(self.device)
        loss, loss_dict = self.calc_loss(
            logit_s, c6d,
            logit_aa_s, msa[:, i_cycle], mask_msa[:,i_cycle], logit_pae, logit_pde,
            pred_crds, alpha, pred_allatom, true_crds_,
            atom_mask_, res_mask, mask_2d, same_chain,
            pred_lddt_binned, idx_pdb, bond_feats, atom_frames=atom_frames,
            unclamp=False, negative=False
        )
        loss_dict = OrderedDict([(k,float(v)) for k,v in loss_dict.items()])
        loss_dict['pae'] = float(best_pae.mean())
        loss_dict['pde'] = float(best_pde.mean())
        loss_dict['plddt'] = float(best_lddt[0].mean())

        if args.dump_aux:
            prob_s = [prob.permute(1,2,0).detach().cpu().numpy().astype(np.float16) for prob in prob_s]
            with open("%s.pkl"%(out_prefix), 'wb') as outf:
                pickle.dump(dict(
                    dist = prob_s[0].astype(np.float16), \
                    omega = prob_s[1].astype(np.float16),\
                    theta = prob_s[2].astype(np.float16),\
                    phi = prob_s[3].astype(np.float16),\
                    loss = dict(loss_dict)
                ), outf)

        # output folding trajectory
        if args.dump_traj:
            all_pred = torch.cat([xyz_prev_orig[0:1,None,:,:3]]+all_pred, dim=0)
            is_prot = ~util.is_atom(seq[0,0,:])
            T = all_pred.shape[0]
            t = np.arange(T)
            n_frames = args.num_interp*(T-1)+1
            Y = np.zeros((n_frames,L,3,3))
            for i_res in range(L):
                for i_atom in range(3):
                    for i_coord in range(3):
                        interp = Akima1DInterpolator(t,all_pred[:,0,i_res,i_atom,i_coord].detach().cpu().numpy())
                        Y[:,i_res,i_atom,i_coord] = interp(np.arange(n_frames)/args.num_interp)
            Y = torch.from_numpy(Y).float()

            # 1st frame is final pred so pymol renders bonds correctly
            util.writepdb(out_prefix+"_traj.pdb", Y[-1], seq[0,-1], 
                modelnum=0, bond_feats=bond_feats, file_mode="w")
            for i in range(Y.shape[0]):
                util.writepdb(out_prefix+"_traj.pdb", Y[i], seq[0,-1], 
                    modelnum=i+1, bond_feats=bond_feats, file_mode="a")

        return loss_dict


if __name__ == "__main__":
    args = get_args()

    pred = Predictor(args)

    if args.out is None:
        if args.msa is not None: in_name = args.msa
        elif args.pdb is not None: in_name = args.pdb
        args.out = '.'.join(os.path.basename(in_name).split('.')[:-1])+'_pred'

    # single prediction mode
    if args.list is None and args.folder is None:
        for n in range(args.n_pred):
            print(f'Making prediction {n}...')
            pred.predict(args.out+f'_{n}', 
                         msa_fn=args.msa,
                         pdb_fn=args.pdb,
                         pt_fn=args.pt,
                         hhr_fn=args.hhr, 
                         atab_fn=args.atab, 
                         mol2_fn=args.mol2, 
                         init_protein_tmpl=args.init_protein_tmpl,
                         init_ligand_tmpl=args.init_ligand_tmpl,
                         init_protein_xyz=args.init_protein_xyz,
                         init_ligand_xyz=args.init_ligand_xyz,
                         parse_hetatm=args.parse_hetatm,
                         n_cycle=args.n_cycle,
                         trunc_N=args.trunc_N,
                         trunc_C=args.trunc_C)

    # scoring a list of inputs
    else:
        if args.list is not None:
            with open(args.list) as f:
                filenames = [line.strip() for line in f.readlines()]
        elif args.folder is not None:
            filenames = glob.glob(args.folder+'/*.pdb')

        print(f'Scoring {len(filenames)} files')
        outdir = os.path.dirname(args.out) + '/'
        os.makedirs(outdir, exist_ok=True)

        records = []
        for fn in filenames:
            name = os.path.basename(fn).replace('.pdb','')
            print(f'Processing {fn}')
            for n in range(args.n_pred):
                print(f'Making prediction {n}...')
                loss_dict = pred.predict(
                    outdir+name+f'_pred_{n}', 
                    pdb_fn=fn,
                    n_cycle=args.n_cycle
                )
                loss_dict['name'] = name+f'_{n}'
                print(f'rmsd_c1_c1: {loss_dict["rmsd_c1_c1"]:.3f}\t'\
                      f'rmsd_c1_c2: {loss_dict["rmsd_c1_c2"]:.3f}\t'\
                      f'rmsd_c2_c2: {loss_dict["rmsd_c2_c2"]:.3f}')
                records.append(loss_dict)

        df = pd.DataFrame.from_records(records)
        print(f'Outputting scores to {args.outcsv}')
        df.to_csv(args.outcsv)

