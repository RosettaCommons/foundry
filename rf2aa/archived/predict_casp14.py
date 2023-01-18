import sys, os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from parsers import parse_a3m, read_templates
from RoseTTAFoldModel  import RoseTTAFoldModule
import util
from collections import namedtuple
from ffindex import *
from data_loader import MSAFeaturize, MSABlockDeletion
from kinematics import xyz_to_c6d, c6d_to_bins2, xyz_to_t2d, get_init_xyz
from util_module import ComputeAllAtomCoords

from memory import mem_report

MAX_CYCLE = 20
NREPLICATES = 5
NBIN = [37, 37, 37, 19]

MODEL_PARAM ={
        "n_extra_block"    : 4,
        "n_main_block"     : 32,
        "n_ref_block"      : 0,
        "n_finetune_block" : 4,
        "d_msa"            : 256 ,
        "d_pair"           : 128,
        "d_templ"          : 64,
        "n_head_msa"       : 8,
        "n_head_pair"      : 4,
        "n_head_templ"     : 4,
        "d_hidden"         : 32,
        "d_hidden_templ"   : 64,
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
MODEL_PARAM['SE3_param'] = SE3_param


# params for the folding protocol
fold_params = {
    "SG7"     : np.array([[[-2,3,6,7,6,3,-2]]])/21,
    "SG9"     : np.array([[[-21,14,39,54,59,54,39,14,-21]]])/231,
    "DCUT"    : 19.5,
    "ALPHA"   : 1.57,
    
    # TODO: add Cb to the motif
    "NCAC"    : np.array([[-0.676, -1.294,  0.   ],
                          [ 0.   ,  0.   ,  0.   ],
                          [ 1.5  , -0.174,  0.   ]], dtype=np.float32),
    "CLASH"   : 2.0,
    "PCUT"    : 0.5,
    "DSTEP"   : 0.5,
    "ASTEP"   : np.deg2rad(10.0),
    "XYZRAD"  : 7.5,
    "WANG"    : 0.1,
    "WCST"    : 0.1
}

fold_params["SG"] = fold_params["SG9"]

# compute expected value from binned lddt
def lddt_unbin(pred_lddt):
    nbin = pred_lddt.shape[1]
    bin_step = 1.0 / nbin
    lddt_bins = torch.linspace(bin_step, 1.0, nbin, dtype=pred_lddt.dtype, device=pred_lddt.device)
    
    pred_lddt = nn.Softmax(dim=1)(pred_lddt)
    return torch.sum(lddt_bins[None,:,None]*pred_lddt, dim=1)

class Predictor():
    def __init__(self, model_name="BFF", model_dir=None, device="cuda:0"):
        if model_dir == None:
            self.model_dir = "%s/models"%(os.path.dirname(os.path.abspath(__file__)))
        else:
            self.model_dir = model_dir
        #
        # define model name
        self.model_name = model_name
        self.device = device
        self.active_fn = nn.Softmax(dim=1)

        # define model & load model
        self.model = RoseTTAFoldModule(
            **MODEL_PARAM,
            aamask=util.allatom_mask.to(self.device),
            atom_type_index=util.atom_type_index.to(self.device),
            ljlk_parameters=util.ljlk_parameters.to(self.device),
            lj_correction_parameters=util.lj_correction_parameters.to(self.device),
            num_bonds=util.num_bonds.to(self.device),
            cb_len = util.cb_length_t.to(self.device),
            cb_ang = util.cb_angle_t.to(self.device),
            cb_tor = util.cb_torsion_t.to(self.device),
        ).to(self.device)

        could_load = self.load_model(self.model_name)
        if not could_load:
            print ("ERROR: failed to load model")
            sys.exit()

        self.compute_allatom_coords = ComputeAllAtomCoords().to(self.device)

    def load_model(self, model_name, suffix='last'):
        chk_fn = "%s/%s_%s.pt"%(self.model_dir, model_name, suffix)
        if not os.path.exists(chk_fn):
            return False
        checkpoint = torch.load(chk_fn, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return True
    
    def predict(self, a3m_fn, out_prefix, hhr_fn=None, atab_fn=None, window=1e9, shift=50, n_latent=256):
        msa_orig, ins_orig, _ = parse_a3m(a3m_fn, unzip=False)
        N, L = msa_orig.shape
        #
        if os.path.exists(hhr_fn):
            xyz_t, t1d = read_templates(L, ffdb, hhr_fn, atab_fn, n_templ=4)
        else:
            xyz_t = torch.full((1,L,3,3),np.nan).float()
            t1d = torch.nn.functional.one_hot(torch.full((1, L), 20).long(), num_classes=21).float() # all gaps
            t1d = torch.cat((t1d, torch.zeros((1,L,1)).float()), -1)
        #
        # template features
        xyz_t = xyz_t.float().unsqueeze(0)
        t1d = t1d.float().unsqueeze(0)
        t2d = xyz_to_t2d(xyz_t)
        xyz_t = get_init_xyz(xyz_t) # initialize coordinates with first template

        seq_tmp = t1d[...,:-1].argmax(dim=-1).reshape(-1,L)

        alpha, _, alpha_mask, _ = util.get_torsions(
            xyz_t.reshape(-1,L,27,3),
            seq_tmp,
            util.torsion_indices,
            util.torsion_can_flip,
            util.reference_angles
        )
        alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))
        alpha[torch.isnan(alpha)] = 0.0
        alpha = alpha.reshape(1,-1,L,10,2)
        alpha_mask = alpha_mask.reshape(1,-1,L,10,1)
        alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(1, -1, L, 30)

        self.model.eval()
        for i_trial in range(NREPLICATES):
            if os.path.exists("%s_%02d_init.pdb"%(out_prefix, i_trial)):
                continue
            self.run_prediction(msa_orig, ins_orig, t1d, t2d, xyz_t, xyz_t[:,0], alpha_t, "%s_%02d"%(out_prefix, i_trial), n_latent=n_latent)
            torch.cuda.empty_cache()

    def run_prediction(self, msa_orig, ins_orig, t1d, t2d, xyz_t, xyz, alpha_t, out_prefix, n_latent=256):
        start = time.time()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            #
            msa = torch.tensor(msa_orig).long().to(self.device) # (N, L)
            ins = torch.tensor(ins_orig).long().to(self.device)
            if msa_orig.shape[0] > 4096:
                msa, ins = MSABlockDeletion(msa, ins)
            print (msa_orig.shape, msa.shape)
            #
            seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, p_mask=0.0)
            _, N, L = msa_seed.shape[:3]
            B = 1
            #
            idx_pdb = torch.arange(L).long().view(1, L)
            #
            seq = seq.unsqueeze(0)
            msa_seed = msa_seed.unsqueeze(0)
            msa_extra = msa_extra.unsqueeze(0)
            t1d = t1d.to(self.device)
            t2d = t2d.to(self.device)
            idx_pdb = idx_pdb.to(self.device)
            xyz_t = xyz_t.to(self.device)
            alpha_t = alpha_t.to(self.device)
            xyz = xyz.to(self.device)

            self.write_pdb(seq[0, -1], xyz[0], prefix="%s_templ"%(out_prefix))
            
            msa_prev = None
            pair_prev = None
            alpha_prev = torch.zeros((1,L,10,2), device=seq.device)
            xyz_prev=xyz
            state_prev = None

            best_lddt = torch.tensor([-1.0], device=seq.device)
            best_xyz = None
            best_logit = None
            best_aa = None
            for i_cycle in range(MAX_CYCLE):
                with torch.cuda.amp.autocast(True):
                    logit_s, logit_aa_s, init_crds, alpha_prev, init_allatom, pred_lddt_binned, msa_prev, pair_prev, state_prev = self.model(
                        msa_seed[:,i_cycle], 
                        msa_extra[:,i_cycle],
                        seq[:,i_cycle], 
                        xyz_prev, 
                        alpha_prev,
                        idx_pdb,
                        t1d=t1d, 
                        t2d=t2d,
                        xyz_t=xyz_t,
                        alpha_t=alpha_t,
                        msa_prev=msa_prev,
                        pair_prev=pair_prev,
                        state_prev=state_prev
                    )

                    logit_aa_s = logit_aa_s.reshape(B,-1,N,L)[:,:,0].permute(0,2,1)

                #xyz_prev = init_crds[-1]
                xyz_prev = init_allatom[-1].unsqueeze(0)
                #msa_prev = msa_prev[:,0]
                alpha_prev = alpha_prev[-1]
                pred_lddt = lddt_unbin(pred_lddt_binned)

                print ("RECYCLE", i_cycle, pred_lddt.mean(), best_lddt.mean())
                #_, all_crds = self.compute_allatom_coords(seq[:,i_cycle], init_crds[-1], alpha_prev)
                self.write_pdb(seq[0, -1], init_allatom[-1], Bfacts=pred_lddt[0], prefix="%s_cycle_%02d"%(out_prefix, i_cycle))

                if pred_lddt.mean() < best_lddt.mean():
                    continue
                best_xyz = init_allatom[-1].clone()
                best_logit = logit_s
                best_aa = logit_aa_s
                best_lddt = pred_lddt.clone()

            prob_s = list()
            for logit in logit_s:
                prob = self.active_fn(logit.float()) # distogram
                prob = prob.reshape(-1, L, L) #.permute(1,2,0).cpu().numpy()
                prob_s.append(prob)
        
        end = time.time()

        for prob in prob_s:
            prob += 1e-8
            prob = prob / torch.sum(prob, dim=0)[None]
        self.write_pdb(seq[0, -1], best_xyz, Bfacts=best_lddt[0], prefix="%s_init"%(out_prefix))
        prob_s = [prob.permute(1,2,0).detach().cpu().numpy().astype(np.float16) for prob in prob_s]
        np.savez_compressed("%s.npz"%(out_prefix), dist=prob_s[0].astype(np.float16), \
                            omega=prob_s[1].astype(np.float16),\
                            theta=prob_s[2].astype(np.float16),\
                            phi=prob_s[3].astype(np.float16),\
                            lddt=best_lddt[0].detach().cpu().numpy().astype(np.float16))

        max_mem = torch.cuda.max_memory_allocated()/1e9
        print ("max mem", max_mem)
        print ("runtime", end-start)

                    
    def write_pdb(self, seq, atoms, Bfacts=None, prefix=None):
        L = len(seq)
        filename = "%s.pdb"%prefix
        ctr = 1
        with open(filename, 'wt') as f:
            if Bfacts == None:
                Bfacts = np.zeros(L)
            else:
                Bfacts = torch.clamp( Bfacts, 0, 1)
            
            for i,s in enumerate(seq):
                if (len(atoms.shape)==2):
                    f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                            "ATOM", ctr, " CA ", util.num2aa[s], 
                            "A", i+1, atoms[i,0], atoms[i,1], atoms[i,2],
                            1.0, Bfacts[i] ) )
                    ctr += 1

                elif atoms.shape[1]==3:
                    for j,atm_j in enumerate((" N  "," CA "," C  ")):
                        f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                                "ATOM", ctr, atm_j, util.num2aa[s], 
                                "A", i+1, atoms[i,j,0], atoms[i,j,1], atoms[i,j,2],
                                1.0, Bfacts[i] ) )
                        ctr += 1                
                else:
                    atms = util.aa2long[s]
                    for j,atm_j in enumerate(atms):
                        if (atm_j is not None):
                            f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                                "ATOM", ctr, atm_j, util.num2aa[s], 
                                "A", i+1, atoms[i,j,0], atoms[i,j,1], atoms[i,j,2],
                                1.0, Bfacts[i] ) )
                            ctr += 1
        
def get_args():
    #DB="/home/robetta/rosetta_server_beta/external/databases/trRosetta/pdb100_2021Mar03/pdb100_2021Mar03"
    DB = "/projects/ml/TrRosetta/pdb100_2020Mar11/pdb100_2020Mar11"
    import argparse
    parser = argparse.ArgumentParser(description="RoseTTAFold: Protein structure prediction with 3-track attentions on 1D, 2D, and 3D features")
    parser.add_argument("-db", default=DB, required=False, 
                        help="HHsearch database [%s]"%DB)
    parser.add_argument("-model_name", default="BFF", required=False, 
                        help="Prefix for model. The model under models/[model_name]_best.pt will be used. [BFF]")
    parser.add_argument("-i", default=1, type=int, required=False, help="Parallelize i of j [1]")
    parser.add_argument("-j", default=1, type=int, required=False, help="Parallelize i of j [1]")
    args = parser.parse_args()
    return args

casp14_home = "/home/minkbaek/CASP14.bench"
if __name__ == "__main__":
    args = get_args()
    #tar_s = [line.strip() for line in open("%s/tar_s"%casp14_home)]
    tar_s = [line.strip() for line in open("./tar_s")]
    FFDB = args.db
    FFindexDB = namedtuple("FFindexDB", "index, data")
    ffdb = FFindexDB(read_index(FFDB+'_pdb.ffindex'),
                     read_data(FFDB+'_pdb.ffdata'))
    pred = Predictor(model_name=args.model_name)

    for i_str,tar in enumerate(tar_s):
        if (i_str % args.j == args.i % args.j):
            print (tar)
            if not os.path.exists(tar):
                os.mkdir(tar)
            out_prefix = "%s/%s"%(tar, tar)
            a3m_fn = "%s/a3m.final/%s.a3m"%(casp14_home,tar)
            hhr_fn = "%s/hhr.final/%s.hhr"%(casp14_home,tar)
            atab_fn = "%s/hhr.final/%s.atab"%(casp14_home, tar)
            if not os.path.exists("%s_04.npz"%out_prefix):
                pred.predict(a3m_fn, out_prefix, hhr_fn, atab_fn)
