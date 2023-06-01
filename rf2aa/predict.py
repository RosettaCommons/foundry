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
from rf2aa.util_module import XYZConverter
from rf2aa.chemical import NTOTAL, NTOTALDOFS, NAATOKENS, INIT_CRDS
from rf2aa.parsers import read_templates, parse_multichain_fasta, parse_mixed_fasta
from rf2aa.memory import mem_report
from rf2aa.symmetry import symm_subunit_matrix, find_symm_subs, update_symm_subs, get_symm_map, get_symmetry

from scipy.interpolate import Akima1DInterpolator

def get_bond_distances(bond_feats):
    atom_bonds = (bond_feats > 0)*(bond_feats<5)
    dist_matrix = scipy.sparse.csgraph.shortest_path(atom_bonds.long().numpy(), directed=False)
    # dist_matrix = torch.tensor(np.nan_to_num(dist_matrix, posinf=4.0)) # protein portion is inf and you don't want to mask it out
    return torch.from_numpy(dist_matrix).float()

def get_args():
    DB = "/projects/ml/TrRosetta/pdb100_2022Apr19/pdb100_2022Apr19"
    import argparse
    parser = argparse.ArgumentParser(description="RoseTTAFold: Protein structure prediction with 3-track attentions on 1D, 2D, and 3D features")
    parser.add_argument("-checkpoint", 
        default='./models/RF2_25c_last.pt',
        help="Path to model weights")

    parser.add_argument("-msa", help='Input sequence/MSA to predict structure from, in fasta/a3m format')
    parser.add_argument("-pdb", help='PDB of sequence to predict structure from')
    parser.add_argument("-multi", help='RFNA-formatted list of fasta/a3m files to predict structure')
    parser.add_argument("-oligo", help='homo-oligomeric state for each chain, when using multi',default=None, nargs='+')
    parser.add_argument("-tmpl_chain", help='ChainID of PDB to use as template')
    parser.add_argument("-hhr", help='Input hhr file.')
    parser.add_argument("-atab", help='Input atab file.')
    parser.add_argument("-pt", help='PyTorch cached version of PDB') # fd unused
    parser.add_argument("-mol2", help='mol2 of small molecule to predict structure from')
    #parser.add_argument("-smiles", help='smiles string of small molecule to predict structure from') # fd unused
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
    parser.add_argument("-n_pred", type=int, default=1, help='number of repeat predictions')
    parser.add_argument("-n_cycle", type=int, default=10, help='number of recycles')
    parser.add_argument("-trunc_N", type=int, default=0, help='residues to truncate at N-term on MSA to match PDB')
    parser.add_argument("-trunc_C", type=int, default=0, help='residues to truncate at C-term on MSA to match PDB')
    parser.add_argument("-no_extra_l1", dest='use_extra_l1', default='True', action='store_false',
            help="Turn off chirality and LJ grad inputs to SE3 layers (for backwards compatibility).")
    parser.add_argument("-no_atom_frames", dest='use_atom_frames', default='True', action='store_false',
            help="Turn off l1 features from atom frames in SE3 layers (for backwards compatibility).")

    # fd: atomization & forced disulfides
    parser.add_argument("-atomize_residues", default=None, required=False, help="Residues to atomize")
    parser.add_argument("-disulfidize_residues", default=None, required=False, help="Residues to atomize")

    # fd: symmetry
    parser.add_argument("-symm", required=False, default="C1", help="Model with symmetry")
    parser.add_argument("-symm_fit", required=False, default=False, action='store_true',
        help="Use beta method for 3D updates with symmetry")
    parser.add_argument("-symm_scale", required=False, default=1.0, 
        help="When symm_fit is enabled, a scalefactor on translation versus rotation")

    args = parser.parse_args()

    return args


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


# compute expected value from binned lddt
def lddt_unbin(pred_lddt):
    # calculate lddt prediction loss
    nbin = pred_lddt.shape[1]
    bin_step = 1.0 / nbin
    lddt_bins = torch.linspace(bin_step, 1.0, nbin, dtype=pred_lddt.dtype, device=pred_lddt.device)

    pred_lddt = nn.Softmax(dim=1)(pred_lddt)
    return torch.sum(lddt_bins[None,:,None]*pred_lddt, dim=1)

def pae_unbin(pred_pae):
    # calculate pae loss
    nbin = pred_pae.shape[1]
    bin_step = 0.5
    pae_bins = torch.linspace(bin_step, bin_step*(nbin-1), nbin, dtype=pred_pae.dtype, device=pred_pae.device)

    pred_pae = nn.Softmax(dim=1)(pred_pae)
    return torch.sum(pae_bins[None,:,None,None]*pred_pae, dim=1)

def get_msa(a3mfilename):                                                                       
    msa,ins, _ = parsers.parse_a3m(a3mfilename)
    return {'msa':torch.tensor(msa), 'ins':torch.tensor(ins)}


def get_bond_feats(resids, seq, ra, dslf=None):
    ra2ind = {}
    for i, two_d in enumerate(ra):
        ra2ind[tuple(two_d.numpy())] = i
    N = len(ra2ind.keys())
    bond_feats = torch.zeros((N, N))
    prot_atom_conns = []
    for i, res in enumerate(seq[resids]):
        for j, bond in enumerate(aabonds[res]):
            start_idx = aa2long[res].index(bond[0])
            end_idx = aa2long[res].index(bond[1])
            if (i, start_idx) not in ra2ind or (i, end_idx) not in ra2ind:
                #skip bonds with atoms that aren't observed in the structure
                continue
            start_idx = ra2ind[(i, start_idx)]
            end_idx = ra2ind[(i, end_idx)]

            # maps the 2d index of the start and end indices to btype
            bond_feats[start_idx, end_idx] = aabtypes[res][j]
            bond_feats[end_idx, start_idx] = aabtypes[res][j]

        #1 peptide bonds
        if resids[i]+1 in resids:
            start_idx = ra2ind[(i, 2)]
            end_idx = ra2ind[(resids.index(resids[i]+1), 0)]
            bond_feats[start_idx, end_idx] = aabtypes[res][j]
            bond_feats[end_idx, start_idx] = aabtypes[res][j]
        else:
            if resids[i]<seq.shape[0]-1: #TO DO: check if chain break
                prot_atom_conns.append( (resids[i]+1,ra2ind[(i, 2)]) )

        if resids[i]-1 not in resids:
            if resids[i]>0:
                prot_atom_conns.append( (resids[i]-1,ra2ind[(i, 0)]) )

    #2 disulfides
    if dslf is not None:
        for i,j in dslf:
            start_idx = ra2ind[(resids.index(i), 5)]
            end_idx = ra2ind[(resids.index(j), 5)]
            bond_feats[start_idx, end_idx] = 1
            bond_feats[end_idx, start_idx] = 1

    return bond_feats, prot_atom_conns

def atomize_protein(resids, seq, dslf=None):
    residues_atomize = seq[resids]
    residues_atom_types = [aa2elt[num][:14] for num in residues_atomize]
    residue_atomize_mask = util.allatom_mask[residues_atomize][:,:14]

    ra = residue_atomize_mask.nonzero()
    lig_seq = torch.tensor([aa2num[residues_atom_types[r][a]] if residues_atom_types[r][a] in aa2num else aa2num["ATM"] for r,a in ra])
    ins = torch.zeros_like(lig_seq)

    bond_feats, prot_atom_conns = get_bond_feats(resids, seq, ra, dslf)

    #HACK: use networkx graph to make the atom frames, correct implementation will include frames with "residue atoms"
    G = nx.from_numpy_matrix(bond_feats.numpy())
    frames = get_atom_frames(lig_seq, G)

    angle = np.arcsin(1/3**0.5) # perfect tetrahedral geometry
    chiral_atoms = aachirals[residues_atomize]

    r,a = ra.T
    chiral_atoms = chiral_atoms[r,a].nonzero().squeeze(1) #num_chiral_centers
    num_chiral_centers = chiral_atoms.shape[0]
    chiral_bonds = bond_feats[chiral_atoms] # find bonds to each chiral atom
    chiral_bonds_idx = chiral_bonds.nonzero().reshape(num_chiral_centers, 3, 2)
    
    chirals = torch.zeros((num_chiral_centers, 5))
    chirals[:,0] = chiral_atoms.long()
    chirals[:, 1:-1] = chiral_bonds_idx[...,-1].long()
    chirals[:, -1] = angle

    #if n>0:
    #    chirals = chirals.repeat(3,1).float()
    #    chirals[n:2*n,1:-1] = torch.roll(chirals[n:2*n,1:-1],1,1)
    #    chirals[2*n: ,1:-1] = torch.roll(chirals[2*n: ,1:-1],2,1)
    #    dih = get_dih(*lig_xyz[chirals[:,:4].long()].split(split_size=1,dim=1))[:,0]
    #    chirals[dih<0.0,-1] = -angle

    return lig_seq, ins, frames, bond_feats, chirals, prot_atom_conns


def load_prot_na_data(fasta_fn, oligo=None):
    # load multiple fastas/a3ms for different molecules as in RFNA

    print('fasta_fn',fasta_fn)
    print('oligo',oligo)

    if oligo is None:
        oligo = [1 for _ in fasta_fn]
    else:
        oligo = [int(x) for x in oligo]
    while len(oligo) < len(fasta_fn):
        oligo.append(1)

    Ls, f_Ls, msas, inss = [], [], [], []
    n_p = 0
    n_n = 0
    for i,seq_i in enumerate(fasta_fn):
        print('reading input:',seq_i)
        fseq_i =  seq_i.split(':')
        if (len(fseq_i)==2):
            ftype,fseq_i = fseq_i
        else:
            ftype='P'
            fseq_i = fseq_i[0]

        if ftype.upper()=='PR':
            msa_i, ins_i, L = parse_mixed_fasta(fseq_i, MAXSEQ)
        else:
            msa_i, ins_i, L = parse_multichain_fasta(fseq_i, rna_alphabet=(ftype.upper()=='R'), dna_alphabet=(ftype.upper()=='D'))
        msa_i = torch.tensor(msa_i).long()
        ins_i = torch.tensor(ins_i).long()
        if (oligo[i] > 1):
            mono_L = L
            for k in range(1,oligo[i]):
                L += mono_L
            msa_i, ins_i = merge_a3m_homo(msa_i, ins_i, oligo[i])
        Ls += L
        f_Ls.append(sum(L))
        if (msa_i.shape[0] > MAXSEQ):
            idxs_tokeep = np.random.permutation(msa_i.shape[0])[:MAXSEQ]
            idxs_tokeep[0] = 0
            msa_i = msa_i[idxs_tokeep]
            ins_i = ins_i[idxs_tokeep]
        msas.append(msa_i)
        inss.append(ins_i)
        if ftype.upper() == 'P':
            n_p += len(L)
        elif ftype.upper() in ['D','R']:
            n_n += len(L)
        elif ftype.upper() == 'PR':
            n_p += 1
            n_n += 1

    msa_orig = {'msa':msas[0],'ins':inss[0]}
    for i in range(1,len(f_Ls)):
        msa_orig = merge_a3m_hetero(msa_orig, {'msa':msas[i],'ins':inss[i]}, [sum(f_Ls[:i]),f_Ls[i]])
    msa_orig, ins_orig = msa_orig['msa'], msa_orig['ins']

    L = sum(Ls)
    same_chain = torch.zeros((1,L,L), dtype=torch.bool)
    same_chain[:,:sum(Ls[:n_p]),:sum(Ls[:n_p])] = 1
    same_chain[:,sum(Ls[:n_p]):,sum(Ls[:n_p]):] = 1

    idx_pdb = torch.arange(L).long().view(1, L)
    for i in range(len(Ls)-1):
        idx_pdb[ :, sum(Ls[:(i+1)]): ] += 200

    idx_pdb = idx_pdb[0]

    print('Ls:',Ls)
    print('msa shape:',msa_orig.shape)

    Ls = [sum(Ls[:n_p]), sum(Ls[n_p:])]
    print('Ls:',Ls)
    return msa_orig, ins_orig, same_chain, idx_pdb, Ls


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
            fit=args.symm_fit, 
            tscale=args.symm_scale
        ).to(self.device)

        checkpoint = torch.load(args.checkpoint, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.xyz_converter = XYZConverter().to(self.device)

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

    def predict(self, out_prefix, msa_fn=None, pdb_fn=None, tmpl_chain=None, pt_fn=None, 
        a3m_fn=None, hhr_fn=None, atab_fn=None, mol2_fn=None, fasta_fn=None, oligo=None,
        init_protein_tmpl=False, init_ligand_tmpl=False, init_protein_xyz=False,
        init_ligand_xyz=False, atomize_res=None, disulfidize_res=None,
        n_cycle=10, n_templ=4, random_noise=5.0, trunc_N=0,
        trunc_C=0, templ_conf=0.5, 
        symm="C1"
    ):

        if atomize_res is not None:
            atomize_res = [int(x) for x in atomize_res.split(',')]
        else:
            atomize_res = []

        xyz = None
        mask_prot = None

        if disulfidize_res is not None:
            alldslf = []
            for dslf in disulfidize_res.split(','):
                x,y = dslf.split(':')
                x,y = int(x),int(y)
                alldslf.append( (x,y) )
            disulfidize_res = alldslf
        else:
            disulfidize_res = []

        has_ligand = False
        if pdb_fn is not None:
            msa_prot, ins_prot, Ls_prot, xyz, mask_prot, _, dslf = parsers.read_multichain_pdb(
                pdb_fn, tmpl_chain=tmpl_chain
            )

            #if (len(dslf)>0):
            #    if (len(disulfidize_res) == 0):
            #        disulfidize_res = dslf # if provided, use that instead

            xyz[:,:,14:] = 0 # remove hydrogens
            mask_prot[:,:,14:] = False
            a3m_prot = {"msa": msa_prot, "ins": ins_prot}

            idx_prot = torch.arange(sum(Ls_prot))
            ctr = 0
            for l in Ls_prot:
                ctr += l
                idx_prot[ctr:] += 100

            stream = [l for l in open(pdb_fn) if "HETATM" in l or "CONECT" in l]
            if len(stream)>0:
                mol, msa_sm, ins_sm, xyz_sm, mask_sm = parsers.parse_mol("".join(stream), filetype="pdb", string=True)
                a3m_sm = {"msa": msa_sm.unsqueeze(0), "ins": ins_sm.unsqueeze(0)}
                G = util.get_nxgraph(mol)
                atom_frames = util.get_atom_frames(msa_sm, G)
                N_symmetry, sm_L, _ = xyz_sm.shape
                Ls = Ls_prot + [ sm_L]
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
            Ls_prot = [qlen]
            msa_prot = msa_prot[:,trunc_N:qlen-trunc_C]
            ins_prot = a3m['ins'].clone().long()[:,trunc_N:qlen-trunc_C]
            protein_L = msa_prot.shape[-1]
            idx_prot = torch.arange(protein_L)

        if fasta_fn is not None:
            msa_prot, ins_prot, chain_idx, idx_pdb, Ls = load_prot_na_data(fasta_fn.split(','),oligo)
            protein_L = msa_prot.shape[-1]
            prot_na = True
        else:
            prot_na = False

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
            if not prot_na:
                Ls = [msa_prot.shape[-1], 0]
            N_symmetry = 1
            msa = msa_prot
            ins = ins_prot
            chirals = torch.Tensor()
            atom_frames = torch.zeros(msa[:,0].shape)

        if len(atomize_res) >0 or len(disulfidize_res) >0:
            for x,y in disulfidize_res:
                if x not in atomize_res:
                    atomize_res.append(x)
                if y not in atomize_res:
                    atomize_res.append(y)

            atomize_res.sort()
            print ('atomizing:',atomize_res)
            print ('disulfidizing:',disulfidize_res)
            lig_seq, lig_ins, atom_frames, bond_feats_sm, chirals, prot_atom_conns = atomize_protein(
                atomize_res, msa_prot[0], dslf=disulfidize_res
            )

            # combine msa
            LprotOrig = msa_prot.shape[-1]
            Lprot = LprotOrig - len(atomize_res)
            Ls = [LprotOrig, lig_seq.shape[0]]
            a3m_prot = {"msa": msa_prot, "ins": ins_prot}
            a3m_sm = {"msa": lig_seq.unsqueeze(0), "ins": lig_ins.unsqueeze(0)}
            a3m = merge_a3m_hetero(a3m_prot, a3m_sm, Ls)
            msa = a3m['msa'].long()
            ins = a3m['ins'].long()

            # combine bond features
            bond_feats = torch.zeros((sum(Ls), sum(Ls))).long()
            bond_feats[:Ls[0], :Ls[0]] = util.get_protein_bond_feats_from_idx(LprotOrig,idx_prot)
            bond_feats[Ls[0]:, Ls[0]:] = bond_feats_sm
            for res,atm in prot_atom_conns:
                bond_feats[res, Ls[0]+atm] = 6
                bond_feats[Ls[0]+atm, res] = 6

            # remove atomized residues
            mask=torch.ones(msa.shape[-1],dtype=torch.bool)
            mask[atomize_res]=False
            msa = msa[:,mask]
            ins = ins[:,mask]
            bond_feats = bond_feats[mask][:,mask]

            print (bond_feats)

            idx_prot = idx_prot[mask[:LprotOrig]]
            if xyz is not None:
                xyz = xyz[:,mask[:LprotOrig]]
                mask_prot = mask_prot[:,mask[:LprotOrig]]

            Ls = [Lprot, lig_seq.shape[0]]

            # shift chirals
            chirals[:, :-1] = chirals[:, :-1] + Lprot
            has_ligand = True
            print ('Ls',Ls)
        else:
            bond_feats = torch.zeros((sum(Ls), sum(Ls))).long()
            bond_feats[:Ls[0], :Ls[0]] = util.get_protein_bond_feats(Ls[0])
            if has_ligand:
                bond_feats[Ls[0]:, Ls[0]:] = util.get_bond_feats(mol)
            if prot_na and (Ls[1] > 0):
                bond_feats[Ls[0]:, Ls[0]:] = util.get_protein_bond_feats(Ls[1])

        #maxindex = 0
        #if len(idx_prot)>0:
        #    maxindex = max(idx_prot)
        #idx_sm = torch.arange(maxindex,maxindex+Ls[1])+200
        #idx_pdb = torch.concat([idx_prot.clone(), idx_sm])

        ## symmetry 1: load and get offsets
        symmids,symmRs,symmmeta,symmoffset = symm_subunit_matrix(symm)
        O = symmids.shape[0]

        chain_idx = torch.zeros((sum(Ls), sum(Ls))).long()
        chain_idx[:Ls[0], :Ls[0]] = 1
        chain_idx[Ls[0]:, Ls[0]:] = 1

        dist_matrix = get_bond_distances(bond_feats)

        SYMM_OFFSET_SCALE = 1.0
        xyz_t_cloud = torch.zeros((1,sum(Ls),NTOTAL,3))
        NAmask = torch.zeros(sum(Ls),dtype=bool)
        Nprot = msa_prot.shape[-1]
        NAmask[:Nprot] = util.is_nucleic(msa_prot[0])
        xyz_t_cloud[:,NAmask] = (
            INIT_NA_CRDS.reshape(1,1,NTOTAL,3).repeat(1,NAmask.sum(),1,1)
            + torch.rand(1,NAmask.sum(),1,3)*random_noise - random_noise/2
            + SYMM_OFFSET_SCALE*symmoffset*NAmask.sum()**(1/2)  # note: offset based on symmgroup
        )

        Pmask = ~NAmask
        xyz_t_cloud[:,Pmask] = (
            INIT_CRDS.reshape(1,1,NTOTAL,3).repeat(1,Pmask.sum(),1,1)
            + torch.rand(1,Pmask.sum(),1,3)*random_noise - random_noise/2
            + SYMM_OFFSET_SCALE*symmoffset*Pmask.sum()**(1/2)  # note: offset based on symmgroup
        )

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
                    templ_conf*torch.ones((Ls[0], 1)).float()
                ), -1) # (1, L_protein, NAATOKENS)
                mask_t[0, :Ls[0], :14] = mask_prot[:,:,:14]

            if init_ligand_tmpl: # input true s.m. xyz as template 1
                xyz_t[1, Ls[0]:, :14] = xyz[0, Ls[0]:, :14]
                f1d_t[1, Ls[0]:] = torch.cat((
                    torch.nn.functional.one_hot(msa[0, Ls[0]: ]-1, num_classes=NAATOKENS-1).float(),
                    templ_conf*torch.ones((Ls[1], 1)).float()
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
            xyz_t = xyz_t_cloud
            f1d_t = torch.nn.functional.one_hot(torch.full((1, sum(Ls)), 20).long(), num_classes=NAATOKENS-1).float() # all gaps
            conf = torch.zeros((1, sum(Ls), 1)).float()
            f1d_t = torch.cat((f1d_t, conf), -1)
            mask_t = torch.full((1,sum(Ls),NTOTAL), False)

        xyz_t[torch.isnan(xyz_t)] = xyz_t_cloud[torch.isnan(xyz_t)]

        ## symmetry 2: find contacting subunits and symmetrize inputs
        xyz_t, symmsub = find_symm_subs(xyz_t[:,:sum(Ls)],symmRs,symmmeta)
        Osub = symmsub.shape[0]
        mask_t = mask_t.repeat(1,Osub,1)
        f1d_t = f1d_t.repeat(1,Osub,1)

        atom_frames = atom_frames.repeat(Osub,1,1)

        mask_prev = mask_t[0].clone()
        xyz_prev = xyz_t[0].clone()

        if init_protein_xyz or init_ligand_xyz:
            com = xyz[0,:,1].nanmean(0)
            if init_protein_xyz:
                xyz1 = xyz[0, :Ls[0]]
                xyz_prev[:Ls[0]] = xyz1 - com
                mask_prev[:Ls[0]] = mask_prot[0]
            if init_ligand_xyz:
                xyz2 = xyz[0, Ls[0]:]
                xyz_prev[Ls[0]:] = xyz2 - com
                mask_prev[Ls[0]:] = mask_sm[0]

            # initialize missing positions in ground truth structures
            xyz_prev[:sum(Ls)] = torch.where(mask_prev[:,:,None], xyz_prev[:sum(Ls)], xyz_t_cloud).contiguous()

        if not prot_na: # RM symmetry not compatible with protein-NA for now
            #   a) symmetrize msa
            effL = Osub*sum(Ls)
            if (Osub>1):
                msa, ins = merge_a3m_homo(msa, ins, Osub)

            #   b) symmetrize index
            idx_pdb = torch.arange(effL)
            chain_idx = torch.zeros((effL,effL), device=self.device).long()
            bond_feats_new = torch.zeros((effL,effL), device=self.device).long()
            dist_matrix_new = torch.full((effL,effL), np.inf, device=self.device)
            i_start = 0
            for o_i in range(Osub):
                i_stop = i_start + sum(Ls)
                bond_feats_new[i_start:i_stop,i_start:i_stop] = bond_feats
                dist_matrix_new[i_start:i_stop,i_start:i_stop] = dist_matrix
                for li in Ls:
                    i_stop = i_start + li
                    idx_pdb[i_stop:] += 100
                    chain_idx[i_start:i_stop,i_start:i_stop] = 1
                    i_start = i_stop
            bond_feats = bond_feats_new
            dist_matrix = dist_matrix_new

        seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, 
            p_mask=0.0, params={'MAXLAT': MAXLAT, 'MAXSEQ': MAXSEQ, 'MAXCYCLE': n_cycle}, tocpu=True)

        seq = seq[None].to(self.device, non_blocking=True)
        msa = msa_seed_orig[None].to(self.device, non_blocking=True)
        msa_masked = msa_seed[None].to(self.device, non_blocking=True)
        msa_full = msa_extra[None].to(self.device, non_blocking=True)
        idx_pdb = idx_pdb[None].to(self.device, non_blocking=True) # (B, L)
        xyz_t = xyz_t[None].to(self.device, non_blocking=True)
        mask_t = mask_t[None].to(self.device, non_blocking=True)
        t1d = f1d_t[None].to(self.device, non_blocking=True)
        xyz_prev = xyz_prev[None].to(self.device, non_blocking=True)
        mask_prev = mask_prev[None].to(self.device, non_blocking=True)
        same_chain = chain_idx[None].to(self.device, non_blocking=True)
        atom_frames = atom_frames[None].to(self.device, non_blocking=True)
        bond_feats = bond_feats[None].to(self.device, non_blocking=True)
        dist_matrix = dist_matrix[None].to(self.device, non_blocking=True)
        chirals = chirals[None].to(self.device, non_blocking=True)
        xyz_prev_orig = xyz_prev.clone()

        symmids = symmids.to(self.device)
        symmsub = symmsub.to(self.device)
        symmRs = symmRs.to(self.device)
        subsymms, _ = symmmeta
        for i in range(len(subsymms)):
            subsymms[i] = subsymms[i].to(self.device)

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

        alpha, _, alpha_mask, _ = self.xyz_converter.get_torsions(
            xyz_t.reshape(-1,sum(Ls),NTOTAL,3),
            seq_tmp
        )
        alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))
        alpha[torch.isnan(alpha)] = 0.0
        alpha = alpha.reshape(1,-1,Osub*sum(Ls),NTOTALDOFS,2)
        alpha_mask = alpha_mask.reshape(1,-1,Osub*sum(Ls),NTOTALDOFS,1)
        alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(1, -1, Osub*sum(Ls), 3*NTOTALDOFS).to(self.device)

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

            if prot_na:
                print ("           PAE_p/p PAE_p/n PAE_n/n  p_bind   plddt    best")

            for i_cycle in range(n_cycle):
                logit_s, logit_aa_s, logit_pae, logit_pde, p_bind, pred_crds, alpha, pred_allatom, pred_lddt_binned, \
                msa_prev, pair_prev, state_prev = self.model(
                    msa_masked[:,i_cycle], 
                    msa_full[:,i_cycle],
                    seq[:,i_cycle], 
                    msa[:,i_cycle,0], 
                    xyz_prev, 
                    alpha_prev,
                    idx_pdb,
                    bond_feats=bond_feats,
                    dist_matrix=dist_matrix,
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
                    mask_recycle=mask_recycle,
                    symmids=symmids,
                    symmsub=symmsub,
                    symmRs=symmRs,
                    symmmeta=symmmeta 
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
                    best_p_bind = p_bind

                if prot_na:
                    L1 = Ls[0]
                    pae = pae_unbin(logit_pae)
                    pae_pp = pae[:,:L1,:L1].mean().cpu().numpy()
                    pae_pd = pae[:,:L1,L1:].mean().cpu().numpy()
                    pae_dd = pae[:,L1:,L1:].mean().cpu().numpy()
                    print ("RECYCLE %2d %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f"%(
                        i_cycle,
                        pae_pp,
                        pae_pd,
                        pae_dd,
                        p_bind,
                        pred_lddt.mean().cpu().numpy(),
                        best_lddt.mean().cpu().numpy()
                    ) )

                else:
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
        #  - make full complex
        Lasu = sum(Ls)
        best_xyzfull = torch.zeros( (B,O*Lasu,NTOTAL,3),device=best_xyz.device )
        best_xyzfull[:,:Lasu] = best_xyz[:,:Lasu]
        seq_full = torch.zeros( (B,1,O*Lasu),dtype=seq.dtype, device=seq.device )
        seq_full[:,0,:Lasu] = seq[:,0,:Lasu]
        best_lddtfull = torch.zeros( (B,O*Lasu),device=best_lddt.device )
        best_lddtfull[:,:Lasu] = best_lddt[:,:Lasu]
        bond_featsfull = torch.zeros( (B,O*Lasu,O*Lasu),device=best_lddt.device )
        bond_featsfull[:,:Lasu,:Lasu] = bond_feats[:,:Lasu,:Lasu]
        Lsfull = Ls*O
        for i in range(1,O):
            best_xyzfull[:,(i*Lasu):((i+1)*Lasu)] = torch.einsum('ij,braj->brai', symmRs[i], best_xyz[:,:Lasu])
            bond_featsfull[:,(i*Lasu):((i+1)*Lasu),(i*Lasu):((i+1)*Lasu)] = bond_feats[:,:Lasu,:Lasu]
            seq_full[:,0,(i*Lasu):((i+1)*Lasu)] = seq[:,0,:Lasu]

        util.writepdb(out_prefix+".pdb", best_xyzfull[0], seq_full[0, -1], bfacts=100*best_lddtfull[0].float(), 
                      bond_feats=bond_featsfull, chain_Ls=Lsfull)

        if args.dump_extra_pdbs:
            util.writepdb(out_prefix+"_last.pdb", xyz_prev[0], seq[0, -1], bfacts=100*best_lddtfull[0].float(),
                          bond_feats=bond_featsfull)
            util.writepdb(out_prefix+"_init.pdb", xyz_prev_orig[0], seq[0, -1], bond_feats=bond_featsfull)

        # output losses, model confidence
        loss_dict = {}
        loss_dict['pae'] = float(best_pae.mean())
        loss_dict['pde'] = float(best_pde.mean())
        loss_dict['plddt'] = float(best_lddt[0].mean())
        loss_dict['p_bind'] = float(best_p_bind)

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
        elif args.multi is not None: 
            in_name = '_'.join([os.path.basename(x.split(':')[-1]) for x in args.multi.split(',')])
        args.out = '.'.join(os.path.basename(in_name).split('.')[:-1])+'_pred'

    # single prediction mode
    if args.list is None and args.folder is None:
        for n in range(args.n_pred):
            print(f'Making prediction {n}...')
            pred.predict(args.out+f'_{n}', 
                         msa_fn=args.msa,
                         pdb_fn=args.pdb,
                         fasta_fn=args.multi,
                         oligo=args.oligo,
                         tmpl_chain=args.tmpl_chain,
                         pt_fn=args.pt,
                         hhr_fn=args.hhr, 
                         atab_fn=args.atab, 
                         mol2_fn=args.mol2, 
                         init_protein_tmpl=args.init_protein_tmpl,
                         init_ligand_tmpl=args.init_ligand_tmpl,
                         init_protein_xyz=args.init_protein_xyz,
                         init_ligand_xyz=args.init_ligand_xyz,
                         atomize_res=args.atomize_residues,
                         disulfidize_res=args.disulfidize_residues,
                         n_cycle=args.n_cycle,
                         trunc_N=args.trunc_N,
                         trunc_C=args.trunc_C,
                         symm=args.symm
            )

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

