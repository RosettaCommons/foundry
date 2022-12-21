import sys, os, json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import parsers
from RoseTTAFoldModel  import RoseTTAFoldModule
import util
from collections import namedtuple
from ffindex import *
from data_loader import MSAFeaturize, MSABlockDeletion, merge_a3m_homo, merge_a3m_hetero
from kinematics import xyz_to_c6d, c6d_to_bins, xyz_to_t2d, get_chirals
from util_module import ComputeAllAtomCoords
from chemical import NTOTAL, NTOTALDOFS, NAATOKENS, INIT_CRDS
from memory import mem_report
from scipy.interpolate import Akima1DInterpolator

MODEL_PARAM ={
        "n_extra_block"   : 4,
        "n_main_block"    : 32,
        "n_ref_block"     : 4,
        "d_msa"           : 256,
        "d_pair"          : 128,
        "d_templ"         : 64,
        "n_head_msa"      : 8,
        "n_head_pair"     : 4,
        "n_head_templ"    : 4,
        "d_hidden"        : 32,
        "d_hidden_templ"  : 64,
        "p_drop"       : 0.0,
        "lj_lin"       : 0.7
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
    msa,ins = parsers.parse_a3m(a3mfilename, unzip='.gz' in a3mfilename)
    return {'msa':torch.tensor(msa), 'ins':torch.tensor(ins)}


class Predictor():
    def __init__(self, args, device="cuda:0"):
        # define model name
        self.device = device
        self.active_fn = nn.Softmax(dim=1)

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

    def predict(self, out_prefix, fasta_fn=None, pdb_fn=None, pt_fn=None, mol2_fn=None, 
        init_protein_tmpl=False, init_ligand_tmpl=False, init_protein_xyz=False, init_ligand_xyz=False,
        parse_hetatm=False, n_cycle=10, random_noise=5.0):

        if pdb_fn is not None:
            xyz_prot, mask_prot, idx_prot, seq_prot = parsers.parse_pdb(pdb_fn,seq=True)
            xyz_prot[:,14:] = 0 # remove hydrogens
            mask_prot[:,14:] = False
            xyz_prot = torch.tensor(xyz_prot)
            mask_prot = torch.tensor(mask_prot)
            protein_L, nprotatoms, _ = xyz_prot.shape
            msa_prot = torch.tensor(seq_prot)[None].long()
            ins_prot = torch.zeros(msa_prot.shape).long()
            a3m_prot = {"msa": msa_prot, "ins": ins_prot}
            if parse_hetatm:
                stream = [l for l in open(pdb_fn) if "HETATM" in l or "CONECT" in l]
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
        if pt_fn is not None:
            pdbA = torch.load(pt_fn)
            xyz_prot, mask_prot = pdbA["xyz"], pdbA["mask"]
            alphabet = 'ARNDCQEGHILKMFPSTWYV'
            aa_1_N = dict(zip(list(alphabet),range(len(alphabet))))
            msa_prot = torch.tensor([aa_1_N[a] for a in pdbA['seq']])[None]
            ins_prot = torch.zeros(msa_prot.shape).long()

        if fasta_fn is not None:
            a3m = get_msa(fasta_fn)
            msa_prot = a3m['msa'].clone().long()
            ins_prot = a3m['ins'].clone().long()
            protein_L = msa_prot.shape[-1]
            idx_prot = torch.arange(protein_L)

        if mol2_fn is not None:
            a3m_prot = {"msa": msa_prot, "ins": ins_prot}
            mol, msa_sm, ins_sm, xyz_sm, mask_sm = parsers.parse_mol(args.mol2)
            a3m_sm = {"msa": msa_sm.unsqueeze(0), "ins": ins_sm.unsqueeze(0)}
            G = util.get_nxgraph(mol)
            atom_frames = util.get_atom_frames(msa_sm, G)
            N_symmetry, sm_L, _ = xyz_sm.shape

            Ls = [protein_L, sm_L]
            a3m = merge_a3m_hetero(a3m_prot, a3m_sm, Ls)
            msa = a3m['msa'].long()
            ins = a3m['ins'].long()
            chirals = get_chirals(mol, xyz_sm[0])
        if mol2_fn is None and not parse_hetatm:
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
        if mol2_fn is not None:
            xyz[:, Ls[0]:, 1, :] = xyz_sm
            mask[:, protein_L:, 1] = mask_sm
        idx_sm = torch.arange(max(idx_prot),max(idx_prot)+Ls[1])+200
        idx_pdb = torch.concat([torch.tensor(idx_prot), idx_sm])
        
        seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, 
            p_mask=0.0, params={'MAXLAT': 128, 'MAXSEQ': 1024, 'MAXCYCLE': n_cycle}, tocpu=True)

        chain_idx = torch.zeros((sum(Ls), sum(Ls))).long()
        chain_idx[:Ls[0], :Ls[0]] = 1
        chain_idx[Ls[0]:, Ls[0]:] = 1
        bond_feats = torch.zeros((sum(Ls), sum(Ls))).long()
        bond_feats[:Ls[0], :Ls[0]] = util.get_protein_bond_feats(Ls[0])
        if mol2_fn is not None or parse_hetatm:
            bond_feats[Ls[0]:, Ls[0]:] = util.get_bond_feats(mol, G)

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
            init = INIT_CRDS.reshape(1,NTOTAL,3).repeat(sum(Ls),1,1)
            xyz_prev = init + torch.rand(sum(Ls),1,3)*random_noise - random_noise/2
            mask_prev = mask.clone()

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
        mask_prev = mask_prev.to(self.device, non_blocking=True)
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
                    msa_prev, pair_prev, state_prev = self.model(
                    msa_masked[:,i_cycle], 
                    msa_full[:,i_cycle],
                    seq[:,i_cycle], 
                    seq[:,i_cycle], 
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

                logit_aa_s = logit_aa_s.reshape(B,-1,N,L)[:,:,0].permute(0,2,1)
                xyz_prev = pred_allatom[-1].unsqueeze(0)
                mask_recycle = None

                all_pred.append(pred_crds)
                all_pred_allatom.append(pred_allatom[-1])

                pred_lddt = lddt_unbin(pred_lddt_binned)
                if pred_lddt.mean() > best_lddt.mean():
                    best_xyz = xyz_prev.clone()
                    best_logit = logit_s
                    best_aa = logit_aa_s
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

        # output pdbs
        util.writepdb(out_prefix+".pdb", best_xyz[0], seq[0, -1], bfacts=100*best_lddt[0].float(), 
                      bond_feats=bond_feats)
        if args.dump_extra_pdbs:
            util.writepdb(out_prefix+"_last.pdb", xyz_prev[0], seq[0, -1], bfacts=100*best_lddt[0].float(),
                          bond_feats=bond_feats)
            util.writepdb(out_prefix+"_init.pdb", xyz_prev_orig[0], seq[0, -1], bond_feats=bond_feats)

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

        if args.dump_aux:
            prob_s = [prob.permute(1,2,0).detach().cpu().numpy().astype(np.float16) for prob in prob_s]
            np.savez_compressed("%s.npz"%(out_prefix), 
                                dist = prob_s[0].astype(np.float16), \
                                omega = prob_s[1].astype(np.float16),\
                                theta = prob_s[2].astype(np.float16),\
                                phi = prob_s[3].astype(np.float16),\
                                lddt = best_lddt[0].detach().cpu().numpy().astype(np.float16),
                                pae = best_pae,
                                pde = best_pde)
        max_mem = torch.cuda.max_memory_allocated()/1e9
        print ("max mem", max_mem)
        print ("runtime", end-start)

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="RoseTTAFold: Protein structure prediction with 3-track attentions on 1D, 2D, and 3D features")
    parser.add_argument("-checkpoint", 
        default="/home/jue/git/rf2a-big/big_pair128_20221004/models/rf2a_big_pair128_20221004_208.pt",
        help="Path to model weights")
    parser.add_argument("-fasta", help='FASTA of sequence/MSA to predict structure from')
    parser.add_argument("-pdb", help='PDB of sequence to predict structure from')
    parser.add_argument("-pt", help='PyTorch cached version of PDB')
    parser.add_argument("-mol2", help='mol2 of small molecule to predict structure from')
    parser.add_argument('-input_json', help='json file containing a list of '
        'dictionaries with sets of arguments for multiple prediction runs.')
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
    parser.add_argument("-n_cycle", type=int, default=10, help='number of recycles')
    parser.add_argument("-no_extra_l1", dest='use_extra_l1', default='True', action='store_false',
            help="Turn off chirality and LJ grad inputs to SE3 layers (for backwards compatibility).")
    parser.add_argument("-no_atom_frames", dest='use_atom_frames', default='True', action='store_false',
            help="Turn off l1 features from atom frames in SE3 layers (for backwards compatibility).")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()

    pred = Predictor(args)

    if args.input_json is not None:
        with open(args.input_json) as f:
            argdict_list = json.load(f)

    for argdict in argdict_list:
        for k,v in argdict.items():
            if hasattr(args, k): setattr(args, k, v)

        if args.out is None:
            if args.fasta is not None: in_name = args.fasta
            elif args.pdb is not None: in_name = args.pdb
            args.out = '.'.join(os.path.basename(in_name).split('.')[:-1])+'_pred'

        pred.predict(args.out, 
                     fasta_fn=args.fasta,
                     pdb_fn=args.pdb,
                     pt_fn=args.pt,
                     mol2_fn=args.mol2, 
                     init_protein_tmpl=args.init_protein_tmpl,
                     init_ligand_tmpl=args.init_ligand_tmpl,
                     init_protein_xyz=args.init_protein_xyz,
                     init_ligand_xyz=args.init_ligand_xyz,
                     parse_hetatm=args.parse_hetatm,
                     n_cycle=args.n_cycle)
