import sys, os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
#from parsers import parse_a3m, read_templates
from RoseTTAFoldModel  import RoseTTAFoldModule
import util
from collections import namedtuple
#from ffindex import *
from data_loader import *
from kinematics import xyz_to_c6d, c6d_to_bins2, xyz_to_t2d, get_init_xyz
from util_module import ComputeAllAtomCoords
from loss import *

MAX_CYCLE = 4
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
        "lj_lin"       : 0.6
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

LOAD_PARAM = {'shuffle': False,
              'num_workers': 4,
              'pin_memory': True}

fb_dir = "/projects/ml/TrRosetta/fb_af"
base_dir = "/projects/ml/TrRosetta/PDB30-20FEB17"

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

        self.aamask = util.allatom_mask.to(self.device)
        self.atom_type_index = util.atom_type_index.to(self.device)
        self.ljlk_parameters = util.ljlk_parameters.to(self.device)
        self.lj_correction_parameters = util.lj_correction_parameters.to(self.device)
        self.num_bonds = util.num_bonds.to(self.device)
        self.cb_len = util.cb_length_t.to(self.device)
        self.cb_ang = util.cb_angle_t.to(self.device)
        self.cb_tor = util.cb_torsion_t.to(self.device)

        # define model & load model
        self.model = RoseTTAFoldModule(
            **MODEL_PARAM,
            aamask=self.aamask,
            atom_type_index = self.atom_type_index,
            ljlk_parameters = self.ljlk_parameters,
            lj_correction_parameters = self.lj_correction_parameters,
            num_bonds = self.num_bonds,
            cb_len = self.cb_len,
            cb_ang = self.cb_ang,
            cb_tor = self.cb_tor
        ).to(self.device)

        could_load = self.load_model(self.model_name)
        if not could_load:
            print ("ERROR: failed to load model")
            sys.exit()
        
        self.compute_allatom_coords = ComputeAllAtomCoords().to(self.device)

        self.ti_dev = util.torsion_indices.to(self.device)
        self.ti_flip = util.torsion_can_flip.to(self.device)
        self.ang_ref = util.reference_angles.to(self.device)
        self.l2a = util.long2alt.to(self.device)
        self.aamask = util.allatom_mask.to(self.device)

    def load_model(self, model_name, suffix='last'):
        chk_fn = "%s/%s_%s.pt"%(self.model_dir, model_name, suffix)
        if not os.path.exists(chk_fn):
            return False
        checkpoint = torch.load(chk_fn, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return True
    
    def run_prediction(self, seq, msa_seed, msa_extra, true_crds, res_mask, atom_mask, idx_pdb, xyz_t, t1d, alpha_t, tag):
        self.model.eval()
        with torch.no_grad():
            # transfer inputs to device
            B, _, N, L, _ = msa_seed.shape

            idx_pdb = idx_pdb.to(self.device, non_blocking=True) # (B, L)
            true_crds = true_crds.to(self.device, non_blocking=True) # (B, L, 27, 3)
            res_mask = res_mask.to(self.device, non_blocking=True) # (B, L)
            atom_mask = atom_mask.to(self.device, non_blocking=True) # (B, L, 27)

            xyz_t = xyz_t.to(self.device, non_blocking=True)
            t1d = t1d.to(self.device, non_blocking=True)
            alpha_t = alpha_t.to(self.device, non_blocking=True)

            seq = seq.to(self.device, non_blocking=True)
            msa_seed = msa_seed.to(self.device, non_blocking=True)
            msa_extra = msa_extra.to(self.device, non_blocking=True)
            
            # processing labels & template features
            c6d, _ = xyz_to_c6d(true_crds)
            c6d = c6d_to_bins2(c6d)
            t2d = xyz_to_t2d(xyz_t)
            xyz_t = get_init_xyz(xyz_t)
            xyz_prev = xyz_t[:,0]

            # set number of recycles
            msa_prev = None
            pair_prev = None
            alpha_prev = torch.zeros((1,L,10,2)).to(self.device, non_blocking=True) #fd we could get this from the template...
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

                #print ("RECYCLE", i_cycle, pred_lddt.mean(), best_lddt.mean())
                #_, all_crds = self.compute_allatom_coords(seq[:,i_cycle], xyz_prev, alpha_prev)
                #self.write_pdb(seq[0, -1], all_crds[0], Bfacts=pred_lddt[0], prefix="%s_cycle_%02d"%(out_prefix, i_cycle))

                if pred_lddt.mean() < best_lddt.mean():
                    continue
                best_xyz = init_allatom[-1].clone()
                best_logit = logit_s
                best_aa = logit_aa_s
                best_lddt = pred_lddt.clone()

            # lddt to native
            seq = seq[:,0]
            res_mask = res_mask[0]
            true_tors, true_tors_alt, tors_mask, tors_planar = util.get_torsions(
                true_crds, seq, self.ti_dev, self.ti_flip, self.ang_ref, mask_in=atom_mask)

            # get alternative coordinates for ground-truth
            true_alt = torch.zeros_like(true_crds)
            true_alt.scatter_(2, self.l2a[seq,:,None].repeat(1,1,1,3), true_crds)
            natRs_all, _n0 = self.compute_allatom_coords(seq, true_crds[...,:3,:], true_tors)
            natRs_all_alt, _n1 = self.compute_allatom_coords(seq, true_alt[...,:3,:], true_tors_alt)

            #  - resolve symmetry
            xs_mask = self.model.simulator.aamask[seq] # (B, L, 27)
            xs_mask[0,:,14:]=False # (ignore hydrogens except lj loss)
            xs_mask *= atom_mask # mask missing atoms & residues as well
            natRs_all_symm, nat_symm = resolve_symmetry(best_xyz, natRs_all[0], true_crds[0], natRs_all_alt[0], true_alt[0], xs_mask[0])

            atom_mask_trim = atom_mask[0,res_mask]
            true_lddt = calc_allatom_lddt(best_xyz.unsqueeze(0), nat_symm, idx_pdb, atom_mask)

            ljE = calc_lj(
                seq[0], init_allatom, 
                self.aamask, 
                self.ljlk_parameters,
                self.lj_correction_parameters,
                self.num_bonds
            )
            cbE = calc_cart_bonded(
                seq, init_allatom, idx_pdb, self.cb_len, self.cb_ang, self.cb_tor)

            print (tag[0],tag[1],true_lddt.mean().cpu().numpy(), ljE.cpu().numpy(), cbE.cpu().numpy())
            self.write_pdb(seq[0], best_xyz, Bfacts=best_lddt[0], prefix="preds/%s_pred"%(tag[0]))

            #if (true_lddt.mean()<0.9 or true_lddt.mean()>0.97):
            #    self.write_pdb(seq[0, -1], all_crds[0], Bfacts=best_lddt[0], prefix="%s_pred"%(tag[1]))
            #    self.write_pdb(seq[0, -1], true_crds[0,...,:14,:], Bfacts=best_lddt[0], prefix="%s_native"%(tag[1]))

            #prob_s = list()
            #for logit in logit_s:
            #    prob = self.active_fn(logit.float()) # distogram
            #    prob = prob.reshape(-1, L, L) #.permute(1,2,0).cpu().numpy()
            #    prob_s.append(prob)
        #for prob in prob_s:
        #    prob += 1e-8
        #    prob = prob / torch.sum(prob, dim=0)[None]
        #self.write_pdb(seq[0, -1], best_xyz[0], Bfacts=best_lddt[0], prefix="%s_init"%(out_prefix))
        #prob_s = [prob.permute(1,2,0).detach().cpu().numpy().astype(np.float16) for prob in prob_s]
        #np.savez_compressed("%s.npz"%(out_prefix), dist=prob_s[0].astype(np.float16), \
        #                    omega=prob_s[1].astype(np.float16),\
        #                    theta=prob_s[2].astype(np.float16),\
        #                    phi=prob_s[3].astype(np.float16),\
        #                    lddt=best_lddt[0].detach().cpu().numpy().astype(np.float16))

                    
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
                    natoms = atoms.shape[1]
                    atms = util.aa2long[s]
                    # his prot hack
                    if (s==8 and torch.linalg.norm( atoms[i,9,:]-atoms[i,5,:] ) < 1.7):
                        atms = (
                            " N  "," CA "," C  "," O  "," CB "," CG "," NE2"," CD2"," CE1"," ND1",
                              None,  None,  None,  None," H  "," HA ","1HB ","2HB "," HD2"," HE1",
                            " HD1",  None,  None,  None,  None,  None,  None) # his_d

                    for j,atm_j in enumerate(atms):
                        if (j<natoms and atm_j is not None): # and not torch.isnan(atomscpu[i,j,:]).any()):
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
    parser.add_argument("-model_name", default="BFF", required=False, 
                        help="Prefix for model. The model under models/[model_name]_best.pt will be used. [BFF]")
    parser.add_argument("-i", default=1, required=False, type=int,
                        help="parallelize i of j")
    parser.add_argument("-j", default=1, required=False, type=int,
                        help="parallelize i of j")
    args = parser.parse_args()
    return args

LOADER_PARAMS = {
        "FB_LIST" : "%s/list_b1-3.csv"%fb_dir,
        "FB_DIR"  : fb_dir,
        "PLDDTCUT": 70.0,
        #"seqID"   : 50.0,
        "MAXLAT"  : 256,
        "MAXSEQ"  : 2048,
        "MAXCYCLE": 4,
        "SCCUT"   : 90.0
    }

if __name__ == "__main__":
    args = get_args()
    pred = Predictor(model_name=args.model_name)

    # compile facebook model sets
    with open(LOADER_PARAMS['FB_LIST'], 'r') as f:
        reader = csv.reader(f)
        next(reader)
        rows = [[r[0],r[2],int(r[3]),len(r[-1].strip())] for r in reader
                 if float(r[1]) > 85.0 and
                 len(r[-1].strip()) > 200]
    fb = {}
    for r in rows:
        if r[2] in fb.keys():
            fb[r[2]].append((r[:2], r[-1]))
        else:
            fb[r[2]] = [(r[:2], r[-1])]

    for i, (id_i, key_i) in enumerate(fb.items()):
        if (i%args.j != args.i%args.j):
            continue

        item = key_i[0][0]
        a3m = get_msa(os.path.join(LOADER_PARAMS["FB_DIR"], "a3m", item[-1][:2], item[-1][2:], item[0]+".a3m.gz"), item[0])
        pdb = get_pdb(os.path.join(LOADER_PARAMS["FB_DIR"], "pdb", item[-1][:2], item[-1][2:], item[0]+".pdb"),
                      os.path.join(LOADER_PARAMS["FB_DIR"], "pdb", item[-1][:2], item[-1][2:], item[0]+".plddt.npy"),
                      item[0], LOADER_PARAMS['PLDDTCUT'], LOADER_PARAMS['SCCUT'])
        idx = pdb['idx']
        l = a3m['msa'].shape[-1]

        msa = a3m['msa'].long()
        ins = a3m['ins'].long()
        #if len(msa) > 5:
        #    msa, ins = MSABlockDeletion(msa, ins)
        seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, LOADER_PARAMS, p_mask=0.0)

        # No templates
        xyz_t = torch.full((1,l,27,3),np.nan).float()
        alpha_t = torch.full((1,l,30),0.0).float()
        f1d_t = torch.nn.functional.one_hot(torch.full((1, l), 20).long(), num_classes=21).float() # all gaps
        f1d_t = torch.cat((f1d_t, torch.zeros((1,l,1)).float()), -1)

        true_crds = torch.full((len(idx),27,3),np.nan).float()
        true_crds[:,:,:] = pdb['xyz']
        mask_atoms = torch.zeros((len(idx),27), dtype=torch.bool)
        mask_atoms[:,:] = pdb['mask']
        mask_res = mask_atoms.sum(dim=-1)>=3

        idx_pdb = torch.arange(l).long()

        pred.run_prediction(
            seq[None,...], msa_seed[None,...], msa_extra[None,...],
            true_crds[None,...], mask_res[None,...], mask_atoms[None,...], 
            idx_pdb[None,...], xyz_t[None,...], f1d_t[None,...], alpha_t[None,...],
            item)

