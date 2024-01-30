import torch
import numpy as np
import unittest, random, os, sys

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(script_dir)

from rf2aa.model.RoseTTAFoldModel import LegacyRoseTTAFoldModule
import rf2aa.data.parsers as parsers
import rf2aa.util as util 
from rf2aa.chemical import load_pdb_ideal_sdf_strings, NTOTALDOFS
from rf2aa.data.data_loader import MSAFeaturize, TemplFeaturize, \
     center_and_realign_missing, generate_xyz_prev, get_bond_distances
from rf2aa.data.compose_dataset import default_dataloader_params
from rf2aa.kinematics import get_chirals, xyz_to_t2d
from rf2aa.tensor_util import assert_equal, cmp
MAXLAT=256
MAXSEQ=2048

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
params =  default_dataloader_params

def make_deterministic(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def featurize_input(three_letter="HEM", pickle_input=None):
    """
    make dummy features for the model
    """
    if pickle_input is not None:
        if os.path.exists(pickle_input):
            rf_input = torch.load(pickle_input)
            return rf_input
    ligands = load_pdb_ideal_sdf_strings(return_only_sdf_strings=True)
    sdf = ligands[three_letter]
    mol, msa_sm, ins_sm, xyz_sm, mask_sm = parsers.parse_mol(sdf, filetype="sdf", string=True)
    a3m_sm = {"msa": msa_sm.unsqueeze(0), "ins": ins_sm.unsqueeze(0)}
    G = util.get_nxgraph(mol)
    N_symmetry, sm_L, _ = xyz_sm.shape
    Ls = [ sm_L]
    a3m = a3m_sm
    msa = a3m['msa'].long()
    ins = a3m['ins'].long()
    bond_feats = util.get_bond_feats(mol)
    chirals = get_chirals(mol, xyz_sm[0]) 
    atom_frames = util.get_atom_frames(msa_sm, G)
    idx = torch.arange(sm_L)
    same_chain = torch.ones((sm_L, sm_L)).long()
    dist_matrix = get_bond_distances(bond_feats)
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, 
            p_mask=0.0, params={'MAXLAT': MAXLAT, 'MAXSEQ': MAXSEQ, 'MAXCYCLE': 1}, tocpu=True)

    xyz_t, f1d_t, mask_t, _ = TemplFeaturize({"ids":[]}, sm_L, params, offset=0,
        npick=0, pick_top=True)
    xyz_t_frames = util.xyz_t_to_frame_xyz(xyz_t[None], seq, atom_frames[None])
    mask_t_2d = util.get_prot_sm_mask(mask_t, seq[0])[None] # (B, T, L)
    mask_t_2d = mask_t_2d[:,:,None]*mask_t_2d[:,:,:,None] # (B, T, L, L)
    t2d = xyz_to_t2d(xyz_t_frames, mask_t_2d[None])
    alpha_t = torch.zeros(1, sum(Ls), NTOTALDOFS*3)
    ntempl = xyz_t.shape[0]
    xyz_t = torch.stack(
        [center_and_realign_missing(xyz_t[i], mask_t[i], same_chain=same_chain) for i in range(ntempl)]
    )
    L = sum(Ls)
    xyz_prev, _ = generate_xyz_prev(xyz_t, mask_t, params)
    alpha_prev = torch.zeros((1,L,NTOTALDOFS,2))
    rf_input = {
        "msa_clust": msa_seed,
        "msa_extra": msa_extra,
        "seq": seq,
        "seq_unmasked": msa_seed_orig[:, 0],
        "xyz_prev": xyz_prev[None],
        "alpha_prev": alpha_prev,
        "idx_pdb": idx[None],
        "bond_feats": bond_feats[None],
        "dist_matrix": dist_matrix[None],
        "chirals": chirals[None],
        "atom_frames": atom_frames[None],
        "t1d": f1d_t[None],
        "t2d": t2d[0],
        "xyz_t": xyz_t[...,1,:][None],
        "alpha_t": alpha_t[None],
        "mask_t": mask_t_2d,
        "same_chain": same_chain[None],
        "msa_prev": None,
        "pair_prev": None,
        "state_prev": None,
        "mask_recycle": None

    }
    if pickle_input is not None:
        torch.save(rf_input, pickle_input)
    return rf_input


class FoldAndDock3TestCase(unittest.TestCase):
    
    def setUp(self):
        super(FoldAndDock3TestCase, self).__init__()
        self.name = "fd3_inference"
        pickle_input = str(os.path.join("test_pickles", f"{self.name}_input"))
        #MODEL_PARAM['use_extra_l1'] = True
        #MODEL_PARAM['use_atom_frames'] = True
        
        # refactored model param allowing for backwards compatibility
        MODEL_PARAM['use_chiral_l1'] = True
        MODEL_PARAM['use_lj_l1'] = True
        MODEL_PARAM['use_atom_frames'] = True
        MODEL_PARAM['use_same_chain'] = True
        MODEL_PARAM['recycling_type'] = 'all'
        self.model = LegacyRoseTTAFoldModule(
            **MODEL_PARAM,
            aamask = util.allatom_mask,
            atom_type_index = util.atom_type_index,
            ljlk_parameters = util.ljlk_parameters,
            lj_correction_parameters = util.lj_correction_parameters,
            num_bonds = util.num_bonds,
            cb_len = util.cb_length_t,
            cb_ang = util.cb_angle_t,
            cb_tor = util.cb_torsion_t,
        )
        self.rf_input = featurize_input(pickle_input=pickle_input)

    def test_inference(self):
        make_deterministic()
        out = self.model(
            msa_latent = self.rf_input["msa_clust"].float(),
            msa_full = self.rf_input["msa_extra"].float(),
            seq = self.rf_input["seq"],
            seq_unmasked = self.rf_input["seq_unmasked"],
            xyz = self.rf_input["xyz_prev"],
            sctors = self.rf_input["alpha_prev"],
            idx = self.rf_input["idx_pdb"],
            bond_feats = self.rf_input["bond_feats"],
            dist_matrix = self.rf_input["dist_matrix"],
            chirals = self.rf_input["chirals"],
            atom_frames = self.rf_input["atom_frames"],
            t1d = self.rf_input["t1d"],
            t2d = self.rf_input["t2d"],
            xyz_t = self.rf_input["xyz_t"],
            alpha_t = self.rf_input["alpha_t"],
            mask_t = self.rf_input["mask_t"],
            same_chain = self.rf_input["same_chain"],
            msa_prev = self.rf_input["msa_prev"],
            pair_prev = self.rf_input["pair_prev"],
            state_prev = self.rf_input["state_prev"],
            mask_recycle = self.rf_input["mask_recycle"]

        )
        output_names = ("logits_c6d", "logits_aa", "logits_pae", \
                        "logits_pde", "p_bind", "xyz", "alpha", "xyz_allatom", \
                        "lddt", "seq", "pair", "state")
        output_dict = dict(zip(output_names, out))
        test_out_path = os.path.join("test_pickles", f"{self.name}_out") 
        if not os.path.exists(test_out_path):
            torch.save(output_dict, test_out_path)
            print(f"saved output at {test_out_path}")
        else:
            reference_output_dict = torch.load(test_out_path)
            for output in output_names:
                
                want = reference_output_dict[output]
                got = output_dict[output]
                if output == "logits_c6d":
                    want = want[0]
                    got = got[0]
                print(output)
                cmp(got, want) 

if __name__ == "__main__":
    unittest.main()