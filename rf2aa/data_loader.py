import torch
import warnings
import time
import deepdiff
from icecream import ic
from torch.utils import data
import os, csv, random, pickle, gzip, itertools, time, ast, sys
from dateutil import parser
from collections import OrderedDict
from itertools import permutations

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
sys.path.append(script_dir+'/../')

import numpy as np
import pandas as pd
import torch
from torch.utils import data
import scipy
from scipy.sparse.csgraph import shortest_path
import networkx as nx

from rf2aa.parsers import parse_a3m, parse_pdb, parse_fasta_if_exists, parse_mol
from rf2aa.chemical import INIT_CRDS, INIT_NA_CRDS, NAATOKENS, MASKINDEX,\
    NTOTAL, NBTYPES, CHAIN_GAP, num2aa
from rf2aa.kinematics import get_chirals
from rf2aa.util import get_nxgraph, get_atom_frames, get_bond_feats, get_protein_bond_feats, \
    atomize_protein, center_and_realign_missing, random_rot_trans, allatom_mask, cif_prot_to_xyz, \
    cif_ligand_to_xyz, cif_ligand_to_obmol, get_automorphs, get_ligand_atoms_bonds, get_alt_query_ligand, \
    remove_unresolved_substructures, map_identical_prot_chains, cartprodcat, idx_from_Ls, \
    same_chain_2d_from_Ls, get_prot_sm_mask, reindex_protein_feats_after_atomize

# faster for remote/tukwila nodes 
#base_dir = "/databases/TrRosetta/PDB-2021AUG02" 
#compl_dir = "/databases/TrRosetta/RoseTTAComplex"
#na_dir = "/databases/TrRosetta/nucleic"
#sm_compl_dir = "/databases/TrRosetta/RF2_allatom"
#mol_dir = "/databases/TrRosetta/RF2_allatom/by-pdb"
csd_dir = "/databases/csd543"

# older paths, still good but best for local/UW nodes
base_dir = "/projects/ml/TrRosetta/PDB-2021AUG02"  
compl_dir = "/projects/ml/RoseTTAComplex"
na_dir = "/projects/ml/nucleic"
na_dir = "/home/dimaio/TrRosetta/nucleic"
fb_dir = "/projects/ml/TrRosetta/fb_af"
sm_compl_dir = "/projects/ml/RF2_allatom"
mol_dir = "/projects/ml/RF2_allatom/rcsb/pkl" # for phase 3 dataloaders -- switch over when ready
# mol_dir = "/projects/ml/RF2_allatom/isdf" # for legacy datasets

if not os.path.exists(base_dir):
    # training on AWS
    base_dir = "/data/databases/PDB-2021AUG02"
    compl_dir = "/data/databases/RoseTTAComplex"
    na_dir = "/data/databases/nucleic"
    fb_dir = "/data/databases/fb_af"
    sm_compl_dir = "/data/databases/RF2_allatom"
    mol_dir = "/data/databases/RF2_allatom/isdf"
    csd_dir = "/data/databases/csd543"

if not os.path.exists(base_dir):
    # training on blue
    base_dir = "/gscratch2/PDB-2021AUG02"
    compl_dir = "/gscratch2/RoseTTAComplex"
    na_dir = "/gscratch2/nucleic"
    fb_dir = "/gscratch2/fb_af1"
    sm_compl_dir = "/gscratch2/RF2_allatom"
    mol_dir = "/gscratch2/RF2_allatom/rcsb/pkl"
    csd_dir = "/gscratch2/RF2_allatom/csd543"

default_dataloader_params = {
        "COMPL_LIST"       : "%s/list.hetero.csv"%compl_dir,
        "HOMO_LIST"        : "%s/list.homo.csv"%compl_dir,
        "NEGATIVE_LIST"    : "%s/list.negative.csv"%compl_dir,
        "RNA_LIST"         : "%s/list.rnaonly.csv"%na_dir,
        "NA_COMPL_LIST"    : "%s/list.nucleic.NODIMERS.csv"%sm_compl_dir,
        "NEG_NA_COMPL_LIST": "%s/list.na_negatives.csv"%na_dir,
        "SM_LIST"          : "%s/sm_compl_20230130.csv"%sm_compl_dir, 
        "MET_LIST"         : "%s/metal_compl_20230130.csv"%sm_compl_dir, 
        "SM_MULTI_LIST"    : "%s/sm_compl_multi_20230130.csv"%sm_compl_dir, 
        "SM_COVALE_LIST"   : "%s/sm_compl_covalent_20230130.csv"%sm_compl_dir,
        "SM_ASMB_LIST"     : "%s/sm_compl_asmb_20230130.csv"%sm_compl_dir,
        "PDB_LIST"         : "%s/list_v02_w_taxid.csv"%base_dir, # on digs
        "FB_LIST"          : "%s/list_b1-3.csv"%fb_dir,
        "CSD_LIST"         : "%s/csd543_cleaned01.csv"%csd_dir, 
        "VAL_PDB"          : "%s/valid_remapped"%sm_compl_dir,
        "VAL_RNA"          : "%s/rna_valid.csv"%na_dir,
        "VAL_COMPL"        : "%s/val_lists/xaa"%compl_dir,
        "VAL_NEG"          : "%s/val_lists/xaa.neg"%compl_dir,
        "VAL_SM_STRICT"    : "%s/sm_compl_valid_strict_20230130.csv"%sm_compl_dir, 
        "TEST_SM"          : "%s/sm_test_heldout_test_clusters.txt"%sm_compl_dir,
        "DATAPKL"          : "%s/dataset_20230130.pkl"%sm_compl_dir, # cache for faster loading 
        "PDB_DIR"          : base_dir,
        "FB_DIR"           : fb_dir,
        "COMPL_DIR"        : compl_dir,
        "NA_DIR"           : na_dir,
        "MOL_DIR"          : mol_dir,
        "CSD_DIR"          : csd_dir,
        "MINTPLT"          : 0,
        "MAXTPLT"          : 5,
        "MINSEQ"           : 1,
        "MAXSEQ"           : 1024,
        "MAXLAT"           : 128, 
        "CROP"             : 384,
        "DATCUT"           : "2020-Apr-30",
        "RESCUT"           : 4.5,
        "BLOCKCUT"         : 5,
        "PLDDTCUT"         : 70.0,
        "SCCUT"            : 90.0,
        "ROWS"             : 1,
        "SEQID"            : 95.0,
        "MAXCYCLE"         : 4,
        "RMAX"             : 5.0,
        "MAXRES"           : 1,
        "MINATOMS"         : 5,
        "MAXATOMS"         : 100,
        "MAXSIM"           : 0.85,
        "MAXNSYMM"         : 1024,
        "NRES_ATOMIZE_MIN" : 1,
        "NRES_ATOMIZE_MAX" : 5,
        "ATOMIZE_FLANK"    : 0,
        "CLUSTER_LIGANDS"  : False
    }

def set_data_loader_params(args):
    for param in default_dataloader_params:
        if hasattr(args, param.lower()):
            default_dataloader_params[param] = getattr(args, param.lower())
    return default_dataloader_params

def MSABlockDeletion(msa, ins, nb=5):
    '''
    Input: MSA having shape (N, L)
    output: new MSA with block deletion
    '''
    N, L = msa.shape
    block_size = max(int(N*0.3), 1)
    block_start = np.random.randint(low=1, high=N, size=nb) # (nb)
    to_delete = block_start[:,None] + np.arange(block_size)[None,:]
    to_delete = np.unique(np.clip(to_delete, 1, N-1))
    #
    mask = np.ones(N, np.bool)
    mask[to_delete] = 0

    return msa[mask], ins[mask]

def cluster_sum(data, assignment, N_seq, N_res):
    csum = torch.zeros(N_seq, N_res, data.shape[-1], device=data.device).scatter_add(0, assignment.view(-1,1,1).expand(-1,N_res,data.shape[-1]), data.float())
    return csum

def MSAFeaturize(msa, ins, params, p_mask=0.15, eps=1e-6, nmer=1, L_s=[], tocpu=False, fixbb=False):
    '''
    Input: full MSA information (after Block deletion if necessary) & full insertion information
    Output: seed MSA features & extra sequences
    
    Seed MSA features:
        - aatype of seed sequence (20 regular aa + 1 gap/unknown + 1 mask)
        - profile of clustered sequences (22)
        - insertion statistics (2)
        - N-term or C-term? (2)
    extra sequence features:
        - aatype of extra sequence (22)
        - insertion info (1)
        - N-term or C-term? (2)
    '''
    if fixbb:
        p_mask = 0
        msa = msa[:1]
        ins = ins[:1]
    N, L = msa.shape
    
    term_info = torch.zeros((L,2), device=msa.device).float()
    if len(L_s) < 1:
        term_info[0,0] = 1.0 # flag for N-term
        term_info[-1,1] = 1.0 # flag for C-term
    else:
        start = 0
        for L_chain in L_s:
            term_info[start, 0] = 1.0 # flag for N-term
            term_info[start+L_chain-1,1] = 1.0 # flag for C-term
            start += L_chain
    #binding_site = torch.zeros((L,1), device=msa.device).float()
    binding_site = torch.zeros((L,0), device=msa.device).float() # keeping this off for now (Jue 12/19)
        
    # raw MSA profile
    raw_profile = torch.nn.functional.one_hot(msa, num_classes=NAATOKENS)
    raw_profile = raw_profile.float().mean(dim=0) 

    # Select Nclust sequence randomly (seed MSA or latent MSA)
    Nclust = (min(N, params['MAXLAT'])-1) // nmer 
    Nclust = Nclust*nmer + 1
    
    if N > Nclust*2:
        Nextra = N - Nclust
    else:
        Nextra = N
    Nextra = min(Nextra, params['MAXSEQ']) // nmer
    Nextra = max(1, Nextra * nmer)
    #
    b_seq = list()
    b_msa_clust = list()
    b_msa_seed = list()
    b_msa_extra = list()
    b_mask_pos = list()
    for i_cycle in range(params['MAXCYCLE']):
        sample_mono = torch.randperm((N-1)//nmer, device=msa.device)
        sample = [sample_mono + imer*((N-1)//nmer) for imer in range(nmer)]
        sample = torch.stack(sample, dim=-1)
        sample = sample.reshape(-1)
        msa_clust = torch.cat((msa[:1,:], msa[1:,:][sample[:Nclust-1]]), dim=0)
        ins_clust = torch.cat((ins[:1,:], ins[1:,:][sample[:Nclust-1]]), dim=0)

        # 15% random masking 
        # - 10%: aa replaced with a uniformly sampled random amino acid
        # - 10%: aa replaced with an amino acid sampled from the MSA profile
        # - 10%: not replaced
        # - 70%: replaced with a special token ("mask")
        random_aa = torch.tensor([[0.05]*20 + [0.0]*(NAATOKENS-20)], device=msa.device)
        same_aa = torch.nn.functional.one_hot(msa_clust, num_classes=NAATOKENS)
        probs = 0.1*random_aa + 0.1*raw_profile + 0.1*same_aa
        #probs = torch.nn.functional.pad(probs, (0, 1), "constant", 0.7)
        probs[...,MASKINDEX]=0.7

        sampler = torch.distributions.categorical.Categorical(probs=probs)
        mask_sample = sampler.sample()

        mask_pos = torch.rand(msa_clust.shape, device=msa_clust.device) < p_mask
        mask_pos[msa_clust>MASKINDEX]=False # no masking on NAs
        
        use_seq = msa_clust
        msa_masked = torch.where(mask_pos, mask_sample, use_seq)
        b_seq.append(msa_masked[0].clone())

        ## get extra sequenes
        if N > Nclust*2:  # there are enough extra sequences
            msa_extra = msa[1:,:][sample[Nclust-1:]]
            ins_extra = ins[1:,:][sample[Nclust-1:]]
            extra_mask = torch.full(msa_extra.shape, False, device=msa_extra.device)
        elif N - Nclust < 1:
            msa_extra = msa_masked.clone()
            ins_extra = ins_clust.clone()
            extra_mask = mask_pos.clone()
        else:
            msa_add = msa[1:,:][sample[Nclust-1:]]
            ins_add = ins[1:,:][sample[Nclust-1:]]
            mask_add = torch.full(msa_add.shape, False, device=msa_add.device)
            msa_extra = torch.cat((msa_masked, msa_add), dim=0)
            ins_extra = torch.cat((ins_clust, ins_add), dim=0)
            extra_mask = torch.cat((mask_pos, mask_add), dim=0)
        N_extra = msa_extra.shape[0]
        
        # clustering (assign remaining sequences to their closest cluster by Hamming distance
        msa_clust_onehot = torch.nn.functional.one_hot(msa_masked, num_classes=NAATOKENS)
        msa_extra_onehot = torch.nn.functional.one_hot(msa_extra, num_classes=NAATOKENS)
        count_clust = torch.logical_and(~mask_pos, msa_clust != 20).float() # 20: index for gap, ignore both masked & gaps
        count_extra = torch.logical_and(~extra_mask, msa_extra != 20).float()
        agreement = torch.matmul((count_extra[:,:,None]*msa_extra_onehot).view(N_extra, -1), (count_clust[:,:,None]*msa_clust_onehot).view(Nclust, -1).T)
        assignment = torch.argmax(agreement, dim=-1)

        # seed MSA features
        # 1. one_hot encoded aatype: msa_clust_onehot
        # 2. cluster profile
        count_extra = ~extra_mask
        count_clust = ~mask_pos
        msa_clust_profile = cluster_sum(count_extra[:,:,None]*msa_extra_onehot, assignment, Nclust, L)
        msa_clust_profile += count_clust[:,:,None]*msa_clust_profile
        count_profile = cluster_sum(count_extra[:,:,None], assignment, Nclust, L).view(Nclust, L)
        count_profile += count_clust
        count_profile += eps
        msa_clust_profile /= count_profile[:,:,None]
        # 3. insertion statistics
        msa_clust_del = cluster_sum((count_extra*ins_extra)[:,:,None], assignment, Nclust, L).view(Nclust, L)
        msa_clust_del += count_clust*ins_clust
        msa_clust_del /= count_profile
        ins_clust = (2.0/np.pi)*torch.arctan(ins_clust.float()/3.0) # (from 0 to 1)
        msa_clust_del = (2.0/np.pi)*torch.arctan(msa_clust_del.float()/3.0) # (from 0 to 1)
        ins_clust = torch.stack((ins_clust, msa_clust_del), dim=-1)
        #
        if fixbb:
            assert params['MAXCYCLE'] == 1
            msa_clust_profile = msa_clust_onehot
            msa_extra_onehot = msa_clust_onehot
            ins_clust[:] = 0
            ins_extra[:] = 0
            # This is how it is done in rfdiff, but really it seems like it should be all 0.
            # Keeping as-is for now for consistency, as it may be used in downstream masking done
            # by apply_masks.
            mask_pos = torch.full_like(msa_clust, 1).bool()
        msa_seed = torch.cat((msa_clust_onehot, msa_clust_profile, ins_clust, term_info[None].expand(Nclust,-1,-1)), dim=-1)

        # extra MSA features
        ins_extra = (2.0/np.pi)*torch.arctan(ins_extra[:Nextra].float()/3.0) # (from 0 to 1)
        msa_extra = torch.cat((msa_extra_onehot[:Nextra], ins_extra[:,:,None], term_info[None].expand(Nextra,-1,-1)), dim=-1)

        if (tocpu):
            b_msa_clust.append(msa_clust.cpu())
            b_msa_seed.append(msa_seed.cpu())
            b_msa_extra.append(msa_extra.cpu())
            b_mask_pos.append(mask_pos.cpu())
        else:
            b_msa_clust.append(msa_clust)
            b_msa_seed.append(msa_seed)
            b_msa_extra.append(msa_extra)
            b_mask_pos.append(mask_pos)
    
    b_seq = torch.stack(b_seq)
    b_msa_clust = torch.stack(b_msa_clust)
    b_msa_seed = torch.stack(b_msa_seed)
    b_msa_extra = torch.stack(b_msa_extra)
    b_mask_pos = torch.stack(b_mask_pos)

    return b_seq, b_msa_clust, b_msa_seed, b_msa_extra, b_mask_pos

def blank_template(n_tmpl, L, random_noise=5.0):
    xyz = INIT_CRDS.reshape(1,1,NTOTAL,3).repeat(n_tmpl,L,1,1) \
        + torch.rand(n_tmpl,L,1,3)*random_noise - random_noise/2
    t1d = torch.nn.functional.one_hot(torch.full((n_tmpl, L), 20).long(), num_classes=NAATOKENS-1).float() # all gaps
    conf = torch.zeros((n_tmpl, L, 1)).float()
    t1d = torch.cat((t1d, conf), -1)
    mask_t = torch.full((n_tmpl,L,NTOTAL), False)
    return xyz, t1d, mask_t


def TemplFeaturize(tplt, qlen, params, offset=0, npick=1, npick_global=None, pick_top=True, same_chain=None, random_noise=5):

    seqID_cut = params['SEQID']

    if npick_global == None:
        npick_global=max(npick, 1)

    ntplt = len(tplt['ids'])
    if (ntplt < 1) or (npick < 1): #no templates in hhsearch file or not want to use templ
        return blank_template(npick_global, qlen, random_noise)
    
    # ignore templates having too high seqID
    if seqID_cut <= 100.0:
        tplt_valid_idx = torch.where(tplt['f0d'][0,:,4] < seqID_cut)[0]
        tplt['ids'] = np.array(tplt['ids'])[tplt_valid_idx]
    else:
        tplt_valid_idx = torch.arange(len(tplt['ids']))
    
    # check again if there are templates having seqID < cutoff
    ntplt = len(tplt['ids'])
    npick = min(npick, ntplt)
    if npick<1: # no templates
        return blank_template(npick_global, qlen, random_noise)

    if not pick_top: # select randomly among all possible templates
        sample = torch.randperm(ntplt)[:npick]
    else: # only consider top 50 templates
        sample = torch.randperm(min(50,ntplt))[:npick]

    xyz = INIT_CRDS.reshape(1,1,NTOTAL,3).repeat(npick_global,qlen,1,1) + torch.rand(1,qlen,1,3)*random_noise
    mask_t = torch.full((npick_global,qlen,NTOTAL),False) # True for valid atom, False for missing atom
    t1d = torch.full((npick_global, qlen), 20).long()
    t1d_val = torch.zeros((npick_global, qlen)).float()

    for i,nt in enumerate(sample):
        tplt_idx = tplt_valid_idx[nt]
        sel = torch.where(tplt['qmap'][0,:,1]==tplt_idx)[0]
        pos = tplt['qmap'][0,sel,0] + offset

        ntmplatoms = tplt['xyz'].shape[2] # will be bigger for NA templates
        xyz[i,pos,:ntmplatoms] = tplt['xyz'][0,sel]
        mask_t[i,pos,:ntmplatoms] = tplt['mask'][0,sel].bool()

        # 1-D features: alignment confidence 
        t1d[i,pos] = tplt['seq'][0,sel]
        t1d_val[i,pos] = tplt['f1d'][0,sel,2] # alignment confidence
        xyz[i] = center_and_realign_missing(xyz[i], mask_t[i], same_chain=same_chain)

    t1d = torch.nn.functional.one_hot(t1d, num_classes=NAATOKENS-1).float() # (no mask token)
    t1d = torch.cat((t1d, t1d_val[...,None]), dim=-1)

    return xyz, t1d, mask_t

def merge_hetero_templates(xyz_t_prot, f1d_t_prot, mask_t_prot, Ls_prot):
    """Diagonally tiles template coordinates, 1d input features, and masks across
    template and residue dimensions. 1st template is concatenated directly on residue
    dimension after a random rotation & translation."""
    N_tmpl_tot = sum([x.shape[0] for x in xyz_t_prot])
    N_f1d_feat = f1d_t_prot[0].shape[2]
    protein_Ls = [xyz_.shape[1] for xyz_ in xyz_t_prot]

    xyz_t_out, f1d_t_out, mask_t_out = blank_template(N_tmpl_tot, sum(Ls_prot))

    i_tmpl = 0
    i_res = 0
    for xyz_, f1d_, mask_ in zip(xyz_t_prot, f1d_t_prot, mask_t_prot):
        N_tmpl, L_tmpl = xyz_.shape[:2]
        if i_tmpl == 0:
            i1, i2 = 1, N_tmpl
        else:
            i1, i2 = i_tmpl, i_tmpl+N_tmpl - 1

        # 1st template is concatenated directly
        xyz_t_out[0, i_res:i_res+L_tmpl] = random_rot_trans(xyz_[0:1])
        f1d_t_out[0, i_res:i_res+L_tmpl] = f1d_[0]
        mask_t_out[0, i_res:i_res+L_tmpl] = mask_[0]

        # remaining templates are diagonally tiled
        xyz_t_out[i1:i2, i_res:i_res+L_tmpl] = xyz_[1:]
        f1d_t_out[i1:i2, i_res:i_res+L_tmpl] = f1d_[1:]
        mask_t_out[i1:i2, i_res:i_res+L_tmpl] = mask_[1:]

        if i_tmpl == 0:
            i_tmpl += N_tmpl
        else:
            i_tmpl += N_tmpl-1
        i_res += L_tmpl

    return xyz_t_out, f1d_t_out, mask_t_out

def get_train_valid_set(params, NEG_CLUSID_OFFSET=1000000, no_match_okay=False, legacy_datapkl=True):
    """Loads training/validation sets as pandas DataFrames and returns them in
    dictionaries keyed by their dataset names.

    Parameters
    ----------
    params : dict
        Config info with paths to various data csv files
    NEG_CLUSID_OFFSET : int
        Offset to add to cluster IDs of negative (compl, NA compl) examples to
        make them distinct from positive examples

    Returns
    ------
    train_ID_dict : dict
        keys are names of datasets, values are np.arrays of cluster IDs to sample
    valid_ID_dict : dict 
        keys are names of datasets, values are np.arrays of cluster IDs to sample
    weights_dict : dict
        keys are names of datasets, values are np.arrays of weights for
        sampling the IDs in train_ID_dict 
    train_set_dict : dict
        keys are names of datasets, values are pandas DataFrames
    valid_set_dict : dict
        keys are names of datasets, values are pandas DataFrames
    """
    ignore = ['DATASETS', 'DATASET_PROB', 'DIFF_MASK_PROBS']
    params = {k:v for k,v in params.items() if k not in ignore}
    # hack to load the right number of outputs for cached diffusion training data
    # remove this once new datasets are online
    if os.path.exists(params['DATAPKL']) and legacy_datapkl:
        with open(params["DATAPKL"], "rb") as f:
            print ('Loading',params["DATAPKL"],'...')
            (
                pdb_IDs, pdb_weights, train_pdb,
                fb_IDs, fb_weights, fb,
                compl_IDs, compl_weights, train_compl,
                neg_IDs, neg_weights, train_neg,
                na_compl_IDs, na_compl_weights, train_na_compl,
                na_neg_IDs, na_neg_weights, train_na_neg,
                rna_IDs, rna_weights, train_rna,
                sm_compl_IDs, sm_compl_weights, train_sm_compl,
                sm_IDs, sm_weights, train_sm,
                valid_pdb, valid_homo,
                valid_compl, valid_neg,
                valid_na_compl, valid_na_neg,
                valid_rna, valid_sm_compl, valid_sm_compl_ligclus,
                valid_sm_compl_strict, valid_sm, valid_pep,
                homo, gen_params
            ) = pickle.load(f)
            diff = deepdiff.DeepDiff(params, gen_params, ignore_order=True)
            ic(diff)
            if diff and 'values_changed' in diff:
                changed = set(diff['values_changed'])
                ic(changed)
                for ig in ignore:
                    changed.discard(f"root['{ig}']")
                if changed:
                    ic(changed)
                    print(f'cache miss: dataset generation parameters passed to train multi differ from those in the dataset pkl:')
                    print(diff)
                    if not no_match_okay:
                        raise Exception(diff)

        return (
            (pdb_IDs, torch.tensor(pdb_weights).float(), train_pdb), \
            (fb_IDs, torch.tensor(fb_weights).float(), fb), \
            (compl_IDs, torch.tensor(compl_weights).float(), train_compl), \
            (neg_IDs, torch.tensor(neg_weights).float(), train_neg),\
            (na_compl_IDs, torch.tensor(na_compl_weights).float(), train_na_compl),\
            (na_neg_IDs, torch.tensor(na_neg_weights).float(), train_na_neg),\
            (rna_IDs, torch.tensor(rna_weights).float(), train_rna),\
            (sm_compl_IDs, torch.tensor(sm_compl_weights).float(), train_sm_compl), \
            (sm_IDs, torch.tensor(sm_weights).float(), train_sm), \
            valid_pdb, valid_homo,
            valid_compl, valid_neg,
            valid_na_compl, valid_na_neg,
            valid_rna, valid_sm_compl, valid_sm_compl_ligclus, valid_sm_compl_strict, valid_sm, valid_pep,
            homo
        )

    # try to load cached datasets 
    if os.path.exists(params['DATAPKL']):
        with open(params["DATAPKL"], "rb") as f:
            print ('Loading',params["DATAPKL"],'...')
            train_ID_dict, valid_ID_dict, weights_dict, \
                train_dict, valid_dict, homo, chid2hash, chid2taxid = pickle.load(f)
            print ('...done')
        return train_ID_dict, valid_ID_dict, weights_dict, train_dict, valid_dict, homo, chid2hash, chid2taxid

    t0 = time.time()
    print(f'cached train/valid datasets {params["DATAPKL"]} not found. '\
          f're-parsing train/valid metadata...')
    
    # helper functions
    def _load_df(filename, pad_hash=True, eval_cols=[]):
        """load dataframe, zero-pad hash string, parse columns as python objects"""
        df = pd.read_csv(filename, na_filter=False) # prevents chain "NA" loading as NaN
        if pad_hash: # restore leading zeros, make into string
            df['HASH'] = df['HASH'].apply(lambda x: f'{x:06d}') 
        for col in eval_cols:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x)) # interpret as list of strings
        return df

    def _apply_date_res_cutoffs(df):
        """filter dataframe by date and resolution cutoffs"""
        return df[(df.RESOLUTION <= params['RESCUT']) & 
                  (df.DEPOSITION.apply(lambda x: parser.parse(x)) <= parser.parse(params['DATCUT']))]
    
    def _get_IDs_weights(df):
        """return unique cluster IDs and AF2-style sampling weights based on seq length"""
        tmp_df = df.drop_duplicates('CLUSTER')
        IDs = tmp_df.CLUSTER.values
        weights = (1/512.)*np.clip(tmp_df.LEN_EXIST.values, 256, 512)
        return IDs, torch.tensor(weights)
    
    # containers for returning the training data/metadata
    train_dict, valid_dict, train_ID_dict, valid_ID_dict, weights_dict = \
        OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
    
    # validation IDs for PDB set
    val_pdb_ids = set([int(l) for l in open(params['VAL_PDB']).readlines()])
    val_compl_ids = set([int(l) for l in open(params['VAL_COMPL']).readlines()])
    val_neg_ids = set([int(l)+NEG_CLUSID_OFFSET for l in open(params['VAL_NEG']).readlines()])
    val_rna_pdb_ids = set([l.rstrip() for l in open(params['VAL_RNA']).readlines()])
    test_sm_ids = set([int(l) for l in open(params['TEST_SM']).readlines()])

    # pdb monomers
    pdb = _load_df(params['PDB_LIST'])
    pdb = _apply_date_res_cutoffs(pdb)
    train_dict['pdb'] = pdb[(~pdb.CLUSTER.isin(val_pdb_ids)) & (~pdb.CLUSTER.isin(test_sm_ids))]
    valid_dict['pdb'] = pdb[pdb.CLUSTER.isin(val_pdb_ids) & (~pdb.CLUSTER.isin(test_sm_ids))]
    val_hash = set(valid_dict['pdb'].HASH.values)
    chid2hash = dict(zip(pdb.CHAINID, pdb.HASH))
    chid2taxid = dict(zip(pdb.CHAINID, pdb.TAXID))
    train_ID_dict['pdb'], weights_dict['pdb'] = _get_IDs_weights(train_dict['pdb'])
    valid_ID_dict['pdb'] = valid_dict['pdb'].CLUSTER.drop_duplicates().values    

    # homo-oligomers
    homo = pd.read_csv(params['HOMO_LIST'])
    tmp_df = pdb[pdb.CLUSTER.isin(val_pdb_ids) & 
                 (pdb.CHAINID.isin(homo['CHAIN_A'])) & 
                 (~pdb.CLUSTER.isin(test_sm_ids))]
    valid_dict['homo'] = homo.merge(tmp_df[['CHAINID','HASH','CLUSTER']], 
                                    left_on='CHAIN_A', right_on='CHAINID', how='right')
    valid_ID_dict['homo'] = valid_dict['homo'].CLUSTER.drop_duplicates().values

    # facebook AF2 distillation set
    fb = pd.read_csv(params['FB_LIST'])
    fb = fb.rename(columns={'#CHAINID':'CHAINID'})
    fb = fb[(fb.plDDT>80) & (fb.SEQUENCE.apply(len) > 200)]
    fb['LEN_EXIST'] = fb.SEQUENCE.apply(len)
    train_dict['fb'] = fb    
    train_ID_dict['fb'], weights_dict['fb'] = _get_IDs_weights(train_dict['fb'])

    # pdb hetero complexes
    compl = pd.read_csv(params['COMPL_LIST'],skiprows=1,header=None)
    compl.columns = ['CHAINID','DEPOSITION','RESOLUTION','HASH','CLUSTER',
                     'LENA:B','TAXONOMY','ASSM_A','OP_A','ASSM_B','OP_B','HETERO']
    compl = _apply_date_res_cutoffs(compl)
    compl['HASH_A'] = compl.HASH.apply(lambda x: x.split('_')[0])
    compl['HASH_B'] = compl.HASH.apply(lambda x: x.split('_')[1])
    compl['LEN'] = compl['LENA:B'].apply(lambda x: [int(y) for y in x.split(':')])
    compl['LEN_EXIST'] = compl['LEN'].apply(lambda x: sum(x)) # total length, for computing weights
    
    valid_dict['compl'] = compl[compl.CLUSTER.isin(val_compl_ids)]
    train_dict['compl'] = compl[(~compl.CLUSTER.isin(val_compl_ids)) &
                                (~compl.HASH_A.isin(val_hash)) &
                                (~compl.HASH_B.isin(val_hash))]
    train_ID_dict['compl'], weights_dict['compl'] = _get_IDs_weights(train_dict['compl'])
    valid_ID_dict['compl'] = valid_dict['compl'].CLUSTER.drop_duplicates().values

    # negative complexes
    #neg = pd.read_csv(params['NEGATIVE_LIST'])
    #neg = _apply_date_res_cutoffs(neg)
    #neg['CLUSTER'] = neg.CLUSTER + NEG_CLUSID_OFFSET
    #neg['HASH_A'] = neg.HASH.apply(lambda x: x.split('_')[0])
    #neg['HASH_B'] = neg.HASH.apply(lambda x: x.split('_')[1])
    #neg['LEN'] = neg['LENA:B'].apply(lambda x: [int(y) for y in x.split(':')])
    #neg['LEN_EXIST'] = neg['LEN'].apply(lambda x: sum(x))

    #valid_dict['neg'] = neg[neg.CLUSTER.isin(val_neg_ids)]
    #train_dict['neg'] = neg[(~neg.CLUSTER.isin(val_neg_ids)) &
    #                        (~neg.HASH_A.isin(val_hash)) &
    #                        (~neg.HASH_B.isin(val_hash))]
    #train_ID_dict['neg'], weights_dict['neg'] = _get_IDs_weights(train_dict['neg'])
    #valid_ID_dict['neg'] = valid_dict['neg'].CLUSTER.drop_duplicates().values

    # nucleic acid complexes
    na = _load_df(params['NA_COMPL_LIST'])
    na = _apply_date_res_cutoffs(na)
    na['LEN'] = na['LENA:B:C'].apply(lambda x: [int(y) for y in x.split(':')])
    na['LEN_EXIST'] = na['LEN'].apply(lambda x: sum(x))

    valid_dict['na_compl'] = na[na.CLUSTER.isin(val_compl_ids)]
    train_dict['na_compl'] = na[(~na.CLUSTER.isin(val_compl_ids))]
    train_ID_dict['na_compl'], weights_dict['na_compl'] = _get_IDs_weights(train_dict['na_compl'])
    valid_ID_dict['na_compl'] = valid_dict['na_compl'].CLUSTER.drop_duplicates().values

    # negative nucleic acid complexes
    #na_neg = _load_df(params['NEG_NA_COMPL_LIST'])
    #na_neg = _apply_date_res_cutoffs(na)
    #na_neg['CLUSTER'] = na_neg.CLUSTER + NEG_CLUSID_OFFSET

    #na_neg['LEN'] = na_neg['LENA:B:C'].apply(lambda x: [int(y) for y in x.split(':')])
    #na_neg['LEN_EXIST'] = na_neg['LEN'].apply(lambda x: sum(x))

    #valid_dict['na_neg'] = na_neg[na_neg.CLUSTER.isin(val_neg_ids)]
    #train_dict['na_neg'] = na_neg[(~na_neg.CLUSTER.isin(val_neg_ids))]
    #train_ID_dict['na_neg'], weights_dict['na_neg'] = _get_IDs_weights(train_dict['na_neg'])
    #valid_ID_dict['na_neg'] = valid_dict['na_neg'].CLUSTER.drop_duplicates().values

    # rna
    rna = pd.read_csv(params['RNA_LIST'])
    rna = _apply_date_res_cutoffs(rna)
    rna['LEN'] = rna['LENA:B:C'].apply(lambda x: [int(y) for y in x.split(':')])
    rna['CLUSTER'] = range(len(rna)) # for unweighted sampling

    in_val = rna['CHAINID'].apply(lambda x: any([y in val_rna_pdb_ids for y in x.split(':')]))
    train_dict['rna'] = rna[~in_val]
    valid_dict['rna'] = rna[in_val]
    train_ID_dict['rna'] = train_dict['rna'].CLUSTER.values # all unique
    weights_dict['rna'] = torch.ones(len(train_ID_dict['rna']))
    valid_ID_dict['rna'] = valid_dict['rna'].CLUSTER.values

    # protein-small molecule complexes
    def _prep_sm_compl_data(df):
        """repeated operations for protein / small molecule datasets"""
        train_df = df[~df.CLUSTER.isin(val_pdb_ids)]
        valid_df = df[df.CLUSTER.isin(val_pdb_ids)]

        seq_len_factor = (1/512.)*np.clip(df.LEN_EXIST, 256, 512) # standard seq length weighting
        df['WEIGHT'] = seq_len_factor # can potentially include other factors (ligand cluster size, etc)
        df_clus = df[['CLUSTER','WEIGHT']].groupby('CLUSTER').mean().reset_index()
        clus2weight = dict(zip(df_clus.CLUSTER, df_clus.WEIGHT))

        train_IDs = train_df.CLUSTER.drop_duplicates().values
        weights = [clus2weight[i] for i in train_IDs]
        
        valid_IDs = valid_df.CLUSTER.drop_duplicates().values

        return train_df, valid_df, train_IDs, valid_IDs, torch.tensor(weights)

    # protein / small molecule complexes
    df = _load_df(params['SM_LIST'], eval_cols=['LIGAND','LIGXF','PARTNERS'])
    df = _apply_date_res_cutoffs(df)
    df = df[
        ~((df['CHAINID']=='1q9x_K') & (df['LIGAND'].apply(lambda x: x[0][0]=='S'))) &
        ~((df['CHAINID']=='4s0n_A') & (df['LIGAND'].apply(lambda x: x[0][0]=='J'))) &
        ~((df['CHAINID']=='3agv_A') & (df['LIGAND'].apply(lambda x: x[0][0]=='F'))) &
        ~((df['CHAINID']=='5l6x_B') & (df['LIGAND'].apply(lambda x: x[0][0]=='O'))) &
        ~((df['CHAINID']=='5l6x_A') & (df['LIGAND'].apply(lambda x: x[0][0]=='I'))) &
        ~(df['CHAINID'].isin([
            '1khz_B', '1g9q_A', '1g9q_B', # cuda indexing errors during forward pass
            '4u9i_B', '4u9h_B', '4jhq_A', '4jhq_B', '5myq_A', '5myq_B', # error during loading
            '5myq_C', '5myq_D', '6g7r_D', '6g7r_B', '6gal_D', '6fpi_B', # error during loading
            '6fpi_D', '6fpw_B', '6fpw_D', # error during loading
        ]))
    ]
    train_dict['sm_compl'], valid_dict['sm_compl'], train_ID_dict['sm_compl'], \
        valid_ID_dict['sm_compl'], weights_dict['sm_compl'] = _prep_sm_compl_data(df)

    # protein / metal ion complexes
    df = _load_df(params['MET_LIST'], eval_cols=['LIGAND','LIGXF','PARTNERS'])
    df = _apply_date_res_cutoffs(df)
    train_dict['metal_compl'], valid_dict['metal_compl'], train_ID_dict['metal_compl'], \
        valid_ID_dict['metal_compl'], weights_dict['metal_compl'] = _prep_sm_compl_data(df)
    
    # protein / multi-residue ligand complexes
    df = _load_df(params['SM_MULTI_LIST'], eval_cols=['LIGAND','LIGXF','PARTNERS'])
    df = _apply_date_res_cutoffs(df)
    df = df[df['LIGATOMS']<=params['CROP']//2]
    train_dict['sm_compl_multi'], valid_dict['sm_compl_multi'], train_ID_dict['sm_compl_multi'], \
        valid_ID_dict['sm_compl_multi'], weights_dict['sm_compl_multi'] = _prep_sm_compl_data(df)

    # protein / covalent ligand complexes
    df = _load_df(params['SM_COVALE_LIST'], eval_cols=['COVALENT', 'LIGAND', 'LIGXF', 'PARTNERS'])
    df = _apply_date_res_cutoffs(df)
    df = df[~df['CHAINID'].isin([
        '1adl_A', '1bs3_A', '1bs3_B', '1btx_A', '1bxw_A', '1etu_A', '1gjm_A',
        '1h3v_B', '1jkj_B', '1l0i_A', '1q1k_A', '1qga_A', '1qga_B', '1nte_A',
        '1x83_B', '2b4b_B', '3dpm_A', '3dpm_B', '4ztt_F', '5kxd_A', '6mhb_F'
    ])]
    train_dict['sm_compl_covale'], valid_dict['sm_compl_covale'], train_ID_dict['sm_compl_covale'], \
        valid_ID_dict['sm_compl_covale'], weights_dict['sm_compl_covale'] = _prep_sm_compl_data(df)

    # protein / ligand assemblies (more than 2 chains)
    df = _load_df(params['SM_ASMB_LIST'], eval_cols=['COVALENT', 'LIGAND', 'LIGXF', 'PARTNERS'])
    df = _apply_date_res_cutoffs(df)

    # these filters are blindly copied from sm_compl and sm_compl_covale above based on
    # experience in training phase 2. these may work now in the re-curated phase 3 datasets,
    # try them at some point
    df = df[
        ~((df['CHAINID']=='1q9x_K') & (df['LIGAND'].apply(lambda x: x[0][0]=='S'))) &
        ~((df['CHAINID']=='4s0n_A') & (df['LIGAND'].apply(lambda x: x[0][0]=='J'))) &
        ~((df['CHAINID']=='3agv_A') & (df['LIGAND'].apply(lambda x: x[0][0]=='F'))) &
        ~((df['CHAINID']=='5l6x_B') & (df['LIGAND'].apply(lambda x: x[0][0]=='O'))) &
        ~((df['CHAINID']=='5l6x_A') & (df['LIGAND'].apply(lambda x: x[0][0]=='I'))) &
        ~(df['CHAINID'].isin([
            '1khz_B', '1g9q_A', '1g9q_B', # cuda indexing errors during forward pass
            '4u9i_B', '4u9h_B', '4jhq_A', '4jhq_B', '5myq_A', '5myq_B', # error during loading
            '5myq_C', '5myq_D', '6g7r_D', '6g7r_B', '6gal_D', '6fpi_B', # error during loading
            '6fpi_D', '6fpw_B', '6fpw_D', # error during loading
        ]))
    ]
    df = df[~df['CHAINID'].isin([
        '1adl_A', '1bs3_A', '1bs3_B', '1btx_A', '1bxw_A', '1etu_A', '1gjm_A',
        '1h3v_B', '1jkj_B', '1l0i_A', '1q1k_A', '1qga_A', '1qga_B', '1nte_A',
        '1x83_B', '2b4b_B', '3dpm_A', '3dpm_B', '4ztt_F', '5kxd_A', '6mhb_F'
    ])]
    train_dict['sm_compl_asmb'], valid_dict['sm_compl_asmb'], train_ID_dict['sm_compl_asmb'], \
        valid_ID_dict['sm_compl_asmb'], weights_dict['sm_compl_asmb'] = _prep_sm_compl_data(df)

    # strict protein / ligand validation set
    val_df = _load_df(params['VAL_SM_STRICT'], params, eval_cols=['LIGAND','LIGXF','PARTNERS'])
    val_df = _apply_date_res_cutoffs(val_df)
    valid_dict['sm_compl_strict'] = val_df
    valid_ID_dict['sm_compl_strict'] = val_df.CLUSTER.drop_duplicates().values

    # remove sm compl protein chains from pdb set
    df = train_dict['pdb']
    sm_compl_chains = np.concatenate([
        train_dict['sm_compl']['CHAINID'].values,
        train_dict['metal_compl']['CHAINID'].values,
        train_dict['sm_compl_multi']['CHAINID'].values,
        train_dict['sm_compl_covale']['CHAINID'].values,
        train_dict['sm_compl_asmb']['CHAINID'].values
    ])
    train_dict['pdb'] = df[~df['CHAINID'].isin(sm_compl_chains)]
    train_ID_dict['pdb'], weights_dict['pdb'] = _get_IDs_weights(train_dict['pdb'])

    # cambridge small molecule database
    sm = _load_df(params['CSD_LIST'], pad_hash=False, eval_cols=['sim','sim_valid','sim_test'])
    sim_idx = int(params["MAXSIM"]*100-50)
    sm = sm[
        (sm['r_factor'] <= params['RMAX']) &
        (sm['nres'] <= params['MAXRES']) &
        (sm['nheavy'] <= params['MAXATOMS']) &
        (sm['nheavy'] >= params['MINATOMS']) &
        (sm['sim_test'].apply(lambda x: x[sim_idx]==0))
    ]
    sm['CLUSTER'] = range(len(sm)) # for unweighted sampling
    sm['train_sim'] = sm['sim'].apply(lambda x: x[sim_idx])
    sm['valid_sim'] = sm['sim_valid'].apply(lambda x: x[sim_idx])
    sm = sm.drop(['sim','sim_test','sim_valid'],axis=1) # drop these memory-intensive columns

    train_dict['sm'] = sm[sm['valid_sim'] == 0]
    valid_dict['sm'] = sm[sm['valid_sim'] > 0]
    train_ID_dict['sm'] = train_dict['sm'].CLUSTER.values
    valid_ID_dict['sm'] = valid_dict['sm'].CLUSTER.values
    weights_dict['sm'] = torch.ones(len(valid_ID_dict['sm']))

    print(f'Done loading datasets in {time.time()-t0} seconds')

    # cache datasets for faster loading next time
    with open(params["DATAPKL"], "wb") as f:
        print ('Writing',params["DATAPKL"],'...')
        pickle.dump((train_ID_dict, valid_ID_dict, weights_dict, 
                     train_dict, valid_dict, homo, chid2hash, chid2taxid), f)
        print ('...done')

    return train_ID_dict, valid_ID_dict, weights_dict, train_dict, valid_dict, \
        homo, chid2hash, chid2taxid

# slice long chains
def get_crop(l, mask, device, crop_size, unclamp=False):
    sel = torch.arange(l,device=device)
    if l <= crop_size:
        return sel

    size = crop_size

    mask = ~(mask[:,:3].sum(dim=-1) < 3.0)
    exists = mask.nonzero()[0]

    if unclamp: # bias it toward N-term.. (follow what AF did.. but don't know why)
        x = np.random.randint(len(exists)) + 1
        res_idx = exists[torch.randperm(x)[0]].item()
    else:
        res_idx = exists[torch.randperm(len(exists))[0]].item()
    lower_bound = max(0, res_idx-size+1)
    upper_bound = min(l-size, res_idx+1)
    start = np.random.randint(lower_bound, upper_bound)
    return sel[start:start+size]

# devide crop between multiple (2+) chains
#   >20 res / chain
def rand_crops(ls, maxlen, minlen=20):
    base = [min(minlen,l) for l in ls ]
    nremain = [max(0,l-minlen) for l in ls ]

    # this must be inefficient...
    pool = []
    for i in range(len(ls)):
        pool.extend([i]*nremain[i])
    pool = random.sample(pool,maxlen-sum(base))
    chosen = [base[i] + sum(p==i for p in pool) for i in range(len(ls))]
    return torch.tensor(chosen)


def get_complex_crop(len_s, mask, device, params):
    tot_len = sum(len_s)
    sel = torch.arange(tot_len, device=device)

    crops = rand_crops(len_s, params['CROP'])

    offset = 0
    sel_s = list()
    for k in range(len(len_s)):
        mask_chain = ~(mask[offset:offset+len_s[k],:3].sum(dim=-1) < 3.0)
        exists = mask_chain.nonzero()[0]
        res_idx = exists[torch.randperm(len(exists))[0]].item()
        lower_bound = max(0, res_idx - crops[k] + 1)
        upper_bound = min(len_s[k]-crops[k], res_idx) + 1
        start = np.random.randint(lower_bound, upper_bound) + offset
        sel_s.append(sel[start:start+crops[k]])
        offset += len_s[k]
    return torch.cat(sel_s)

def get_spatial_crop(xyz, mask, sel, len_s, params, label, cutoff=10.0, eps=1e-6):
    device = xyz.device

    # get interface residues
    #   interface defined as chain 1 versus all other chains
    cond = torch.cdist(xyz[:len_s[0],1], xyz[len_s[0]:,1]) < cutoff
    cond = torch.logical_and(cond, mask[:len_s[0],None,1]*mask[None,len_s[0]:,1]) 
    i,j = torch.where(cond)
    ifaces = torch.cat([i,j+len_s[0]])
    if len(ifaces) < 1:
        print ("ERROR: no iface residue????", label)
        return get_complex_crop(len_s, mask, device, params)
    cnt_idx = ifaces[np.random.randint(len(ifaces))]

    dist = torch.cdist(xyz[:,1], xyz[cnt_idx,1][None]).reshape(-1) + torch.arange(len(xyz), device=xyz.device)*eps
    cond = mask[:,1]*mask[cnt_idx,1]
    dist[~cond] = 999999.9
    _, idx = torch.topk(dist, params['CROP'], largest=False)

    sel, _ = torch.sort(sel[idx])
    return sel


# this is a bit of a mess...
def get_na_crop(seq, xyz, mask, sel, len_s, params, negative=False, incl_protein=True, cutoff=12.0, bp_cutoff=4.0, eps=1e-6):
    device = xyz.device

    # get base pairing NA bases
    repatom = torch.zeros(sum(len_s), dtype=torch.long, device=xyz.device)
    repatom[seq==22] = 15 # DA - N1
    repatom[seq==23] = 14 # DC - N3
    repatom[seq==24] = 15 # DG - N1
    repatom[seq==25] = 14 # DT - N3
    repatom[seq==27] = 12 # A - N1
    repatom[seq==28] = 15 # C - N3
    repatom[seq==29] = 12 # G - N1
    repatom[seq==30] = 15 # U - N3

    if not incl_protein:
        if len(len_s)==2:
            # 2 RNA chains
            xyz_na1_rep = torch.gather(xyz[:len_s[0]], 1, repatom[:len_s[0],None,None].repeat(1,1,3)).squeeze(1)
            xyz_na2_rep = torch.gather(xyz[len_s[0]:], 1, repatom[len_s[0]:,None,None].repeat(1,1,3)).squeeze(1)
            cond = torch.cdist(xyz_na1_rep, xyz_na2_rep) < bp_cutoff

            mask_na1_rep = torch.gather(mask[:len_s[0]], 1, repatom[:len_s[0],None]).squeeze(1)
            mask_na2_rep = torch.gather(mask[len_s[0]:], 1, repatom[len_s[0]:,None]).squeeze(1)
            cond = torch.logical_and(cond, mask_na1_rep[:,None]*mask_na2_rep[None,:]) 
        else:
            # 1 RNA chains
            xyz_na_rep = torch.gather(xyz, 1, repatom[:,None,None].repeat(1,1,3)).squeeze(1)
            cond = torch.cdist(xyz_na_rep, xyz_na_rep) < bp_cutoff
            mask_na_rep = torch.gather(mask, 1, repatom[:,None]).squeeze(1)
            cond = torch.logical_and(cond, mask_na_rep[:,None]*mask_na_rep[None,:])

        if (torch.sum(cond)==0):
            i= np.random.randint(len_s[0]-1)
            while (not mask[i,1] or not mask[i+1,1]):
                i = np.random.randint(len_s[0])
            cond[i,i+1] = True

    else:
        if len(len_s)==3:
            xyz_na1_rep = torch.gather(xyz[len_s[0]:(len_s[0]+len_s[1])], 1, repatom[len_s[0]:(len_s[0]+len_s[1]),None,None].repeat(1,1,3)).squeeze(1)
            xyz_na2_rep = torch.gather(xyz[(len_s[0]+len_s[1]):], 1, repatom[(len_s[0]+len_s[1]):,None,None].repeat(1,1,3)).squeeze(1)
            cond_bp = torch.cdist(xyz_na1_rep, xyz_na2_rep) < bp_cutoff

            mask_na1_rep = torch.gather(mask[len_s[0]:(len_s[0]+len_s[1])], 1, repatom[len_s[0]:(len_s[0]+len_s[1]),None]).squeeze(1)
            mask_na2_rep = torch.gather(mask[(len_s[0]+len_s[1]):], 1, repatom[(len_s[0]+len_s[1]):,None]).squeeze(1)
            cond_bp = torch.logical_and(cond_bp, mask_na1_rep[:,None]*mask_na2_rep[None,:]) 

        if (not negative):
            # get interface residues
            #   interface defined as chain 1 versus all other chains
            xyz_na_rep = torch.gather(xyz[len_s[0]:], 1, repatom[len_s[0]:,None,None].repeat(1,1,3)).squeeze(1)
            cond = torch.cdist(xyz[:len_s[0],1], xyz_na_rep) < cutoff
            mask_na_rep = torch.gather(mask[len_s[0]:], 1, repatom[len_s[0]:,None]).squeeze(1)
            cond = torch.logical_and(
                cond, 
                mask[:len_s[0],None,1] * mask_na_rep[None,:]
            )

        if (negative or torch.sum(cond)==0):
            # pick a random pair of residues
            cond = torch.zeros( (len_s[0], sum(len_s[1:])), dtype=torch.bool )
            i,j = np.random.randint(len_s[0]), np.random.randint(sum(len_s[1:]))
            while (not mask[i,1]):
                i = np.random.randint(len_s[0])
            while (not mask[len_s[0]+j,1]):
                j = np.random.randint(sum(len_s[1:]))
            cond[i,j] = True

    # a) build a graph of costs:
    #     cost (i,j in same chain) = abs(i-j)
    #     cost (i,j in different chains) = { 0 if i,j are an interface
    #                                    = { 999 if i,j are NOT an interface
    if len(len_s)==3:
        int_1_2 = np.full((len_s[0],len_s[1]),999)
        int_1_3 = np.full((len_s[0],len_s[2]),999)
        int_2_3 = np.full((len_s[1],len_s[2]),999)
        int_1_2[cond[:,:len_s[1]]]=1
        int_1_3[cond[:,len_s[1]:]]=1
        int_2_3[cond_bp] = 0
        inter = np.block([
            [np.abs(np.arange(len_s[0])[:,None]-np.arange(len_s[0])[None,:]),int_1_2,int_1_3],
            [int_1_2.T,np.abs(np.arange(len_s[1])[:,None]-np.arange(len_s[1])[None,:]),int_2_3],
            [int_1_3.T,int_2_3.T,np.abs(np.arange(len_s[2])[:,None]-np.arange(len_s[2])[None,:])]
        ])
    elif len(len_s)==2:
        int_1_2 = np.full((len_s[0],len_s[1]),999)
        int_1_2[cond]=1
        inter = np.block([
            [np.abs(np.arange(len_s[0])[:,None]-np.arange(len_s[0])[None,:]),int_1_2],
            [int_1_2.T,np.abs(np.arange(len_s[1])[:,None]-np.arange(len_s[1])[None,:])]
        ])
    else:
        inter = np.abs(np.arange(len_s[0])[:,None]-np.arange(len_s[0])[None,:])
        inter[cond] = 1

    # b) pick a random interface residue
    intface,_ = torch.where(cond)
    startres = intface[np.random.randint(len(intface))]

    # c) traverse graph starting from chosen residue
    d_res = shortest_path(inter,directed=False,indices=startres)
    _, idx = torch.topk(torch.from_numpy(d_res).to(device=device), params['CROP'], largest=False)

    sel, _ = torch.sort(sel[idx])

    return sel

def find_msa_hashes(protein_chain_info, params):
    """
    given a list of protein chains, this function searches through all the pregenerated MSAs and identifies the correct MSA hashes/metadata to load for each protein chain
    it returns a list of dictionaries with msa hash and other relevant metadata for constructing a paired MSA for multiple chains
    """
    updated_protein_chain_info = []
    msas_to_load = []  
    # handles checking all pairs of chains if they have paired MSAs
    for item1, item2 in itertools.permutations(protein_chain_info, 2):
        # if you already have a MSA for item1 skip the other pairings
        if item1 in updated_protein_chain_info:
            continue

        if item1["hash"] != item2["hash"] and item1["query_taxid"] == item2["query_taxid"]: # different hashes but same tax id, means there is a pMSA generated
            msaA_id = item1["hash"]
            msaB_id = item2["hash"]
            pMSA_hash = "_".join([msaA_id, msaB_id])
            pMSA_fn = params['COMPL_DIR'] + '/pMSA/' + msaA_id[:3] + '/' + msaB_id[:3] + '/' + pMSA_hash + '.a3m.gz'
            if os.path.exists(pMSA_fn):
                updated_protein_chain_info.append(item1)
                msas_to_load.append({"path": pMSA_fn, 
                                     "hash": msaA_id, 
                                     "seq_range": (0, item1["len"]),
                                     "paired": True})
            else: 
                # check if the sequence is the second sequence in the paired MSA
                # msaA_id = item2["hash"]
                # msaB_id = item1["hash"]
                pMSA_hash = "_".join([msaB_id, msaA_id])
                pMSA_fn = params['COMPL_DIR'] + '/pMSA/' + msaB_id[:3] + '/' + msaA_id[:3] + '/' + pMSA_hash + '.a3m.gz'
                if os.path.exists(pMSA_fn):
                    updated_protein_chain_info.append(item1)
                    msas_to_load.append({"path": pMSA_fn, 
                                         "hash": msaA_id, 
                                         "seq_range": (item2["len"], item1["len"]+item2["len"]), # store sequence indices to only pull out second chain
                                         "paired": True}) 

    # add in information from remaining chains
    unpaired_items = [item for item in protein_chain_info if item not in updated_protein_chain_info]
    unpaired_msas = [{"path": params['PDB_DIR'] + '/a3m/' + info["hash"][:3] + '/' + info["hash"] + '.a3m.gz',
                      "hash": info["hash"], 
                      "seq_range": (0,info["len"]), 
                      "paired": False} for info in unpaired_items]
    updated_protein_chain_info.extend(unpaired_items) # maps the order of the chains to the order of loaded MSAs so coordinates and msa match
    msas_to_load.extend(unpaired_msas) # msas_to_load will be the same length as updated_protein_chain_info

    # currently updated_protein_chain_info and msas_to_load have items in the same order
    # explicitly update the order of msas_to_load to match the initial input protein_chain_info which will match the xyz coordinates generated in the dataloader
    try:
        original_pci_order = [updated_protein_chain_info.index(info) for info in protein_chain_info]
    except Exception as e:
        print(f"ERROR: there is a protein chain that was supposed to be loaded that was not: input chains: {str(protein_chain_info)}   output_chains: {str(updated_protein_chain_info)}")
        raise e
    msas_to_load = [msas_to_load[i] for i in original_pci_order]

    assert len(protein_chain_info) == len(msas_to_load), f"not all protein chains had corresponding MSAs: {str(protein_chain_info)} "
    return msas_to_load


def get_assembly_msa(protein_chain_info, params):
    """
    takes a list of dictionaries containing relevant information about protein chains and returns an MSA (paired if possible)
    for those chains

    WARNING: this code is the general case that can make Nmer assembly chain MSAs from the currently generated MSAs (single 
    chain and two paired chains) but a preferable approach would be to regenerate all the MSAs from scratch using hhblits and 
    pair them before filtering
    """
    msas_to_load = find_msa_hashes(protein_chain_info, params)
    msa_hashes = [msa["hash"] for msa in msas_to_load]
    # merge msas
    a3m = None
    if len(msa_hashes) == 0:
        raise NotImplementedError(f"No MSAs were found for these protein chains {str(protein_chain_info)}")
    elif len(set(msa_hashes)) == 1: # monomer/homomer case (all same msas)
        msa_vals = msas_to_load[0]
        num_copies = len(msa_hashes)
        a3m = get_msa(msa_vals["path"], msa_vals["hash"])
        msa = a3m['msa'].long()
        ins = a3m['ins'].long()
        L_s = [msa.shape[1]]*num_copies
        # check if monomer or homomer
        if num_copies >1:
            msa, ins = merge_a3m_homo(msa, ins, num_copies)
            a3m = {"msa": msa, "ins": ins}
    else: # heteromer case (at least two different MSAs will handle things like AB, AAB, ABC...)
        a3m_list = []
        L_s = []
        for i in range(len(msa_hashes)):
            msa_vals = msas_to_load[i]
            msa, ins, taxID = parse_a3m(msa_vals["path"], paired=msa_vals["paired"])
            msa = msa[:, msa_vals["seq_range"][0]:msa_vals["seq_range"][1]]
            ins = ins[:, msa_vals["seq_range"][0]:msa_vals["seq_range"][1]]
            a3m_list.append({"msa":msa, "ins":ins, "taxID":taxID, "hash":msa_vals["hash"]})
            L_s.append(msa_vals["seq_range"][1]-msa_vals["seq_range"][0])
        msaA, insA = merge_msas(a3m_list, L_s)
        a3m = {"msa": torch.tensor(msaA), "ins": torch.tensor(insA)}
    return a3m

# merge msa & insertion statistics of two proteins having different taxID
def merge_a3m_hetero(a3mA, a3mB, L_s):
    # merge msa
    query = torch.cat([a3mA['msa'][0], a3mB['msa'][0]]).unsqueeze(0) # (1, L)
    msa = [query]
    if a3mA['msa'].shape[0] > 1:
        extra_A = torch.nn.functional.pad(a3mA['msa'][1:], (0,sum(L_s[1:])), "constant", 20) # pad gaps
        msa.append(extra_A)
    if a3mB['msa'].shape[0] > 1:
        extra_B = torch.nn.functional.pad(a3mB['msa'][1:], (L_s[0],0), "constant", 20)
        msa.append(extra_B)
    msa = torch.cat(msa, dim=0)

    # merge ins
    query = torch.cat([a3mA['ins'][0], a3mB['ins'][0]]).unsqueeze(0) # (1, L)
    ins = [query]
    if a3mA['ins'].shape[0] > 1:
        extra_A = torch.nn.functional.pad(a3mA['ins'][1:], (0,sum(L_s[1:])), "constant", 0) # pad gaps
        ins.append(extra_A)
    if a3mB['ins'].shape[0] > 1:
        extra_B = torch.nn.functional.pad(a3mB['ins'][1:], (L_s[0],0), "constant", 0)
        ins.append(extra_B)
    ins = torch.cat(ins, dim=0)
    return {'msa': msa, 'ins': ins}

# merge msa & insertion statistics of units in homo-oligomers
def merge_a3m_homo(msa_orig, ins_orig, nmer):
    N, L = msa_orig.shape[:2]
    msa = torch.full((1+(N-1)*nmer, L*nmer), 20, dtype=msa_orig.dtype, device=msa_orig.device)
    ins = torch.full((1+(N-1)*nmer, L*nmer), 0, dtype=ins_orig.dtype, device=msa_orig.device)
    start=0
    start2 = 1
    for i_c in range(nmer):
        msa[0, start:start+L] = msa_orig[0] 
        msa[start2:start2+(N-1), start:start+L] = msa_orig[1:]
        ins[0, start:start+L] = ins_orig[0]
        ins[start2:start2+(N-1), start:start+L] = ins_orig[1:]
        start += L
        start2 += (N-1)
    return msa, ins

def merge_msas(a3m_list, L_s):
    """
    takes a list of a3m dictionaries with keys msa, ins and a list of protein lengths and creates a
    combined MSA 
    """
    seen = set()
    taxIDs = []
    a3mA = a3m_list[0]
    taxIDs.extend(a3mA["taxID"])
    seen.update(a3mA["hash"])
    msaA, insA = a3mA["msa"], a3mA["ins"]
    for i in range(1, len(a3m_list)):
        a3mB = a3m_list[i]
        pair_taxIDs = set(taxIDs).intersection(set(a3mB["taxID"]))
        if a3mB["hash"] in seen or len(pair_taxIDs) < 5: #homomer/not enough pairs 
            a3mA = {"msa": msaA, "ins": insA}
            L_s_to_merge = [sum(L_s[:i])+1, L_s[i]]
            a3mA = merge_a3m_hetero(a3mA, a3mB, L_s_to_merge)
            msaA, insA = a3mA["msa"], a3mA["ins"]
            taxIDs.extend(a3mB["taxID"])
        else:
            final_pairsA = []
            final_pairsB = []
            msaB, insB = a3mB["msa"], a3mB["ins"]
            for pair in pair_taxIDs:
                pair_a3mA = np.where(np.array(taxIDs)==pair)
                pair_a3mB = np.where(a3mB["taxID"]==pair)
                msaApair = np.argsort(np.sum(msaA[pair_a3mA, :] == msaA[0, :],axis=-1))
                msaBpair = np.argsort(np.sum(msaB[pair_a3mB, :] == msaB[0, :],axis=-1))
                      
                final_pairsA.append(pair_a3mA[0][msaApair[0]][0])
                final_pairsB.append(pair_a3mB[0][msaBpair[0]][0])
            paired_msaB = np.full((msaA.shape[0], L_s[i]), 20) #num_sequences protein A, L protein B
            paired_msaB[np.array(final_pairsA).ravel().astype(np.uint8)] = msaB[np.array(final_pairsB).ravel().astype(np.uint8)]
            msaA = np.append(msaA, paired_msaB, axis=1)
            insA = np.zeros_like(msaA) #paired MSAs in our dataset dont have insertions so setting to all 0s
        seen.update(a3mB["hash"])
        
    return msaA, insA

# Generate input features for single-chain
def featurize_single_chain(msa, ins, tplt, pdb, params, unclamp=False, pick_top=True, random_noise=5.0, fixbb=False):
    msa_featurization_kwargs = {}
    if fixbb:
        ic('setting msa feat kwargs')
        msa_featurization_kwargs['p_mask'] = 0.0
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params, fixbb=fixbb, **msa_featurization_kwargs)

    # get ground-truth structures
    idx = torch.arange(len(pdb['xyz'])) 
    xyz = torch.full((len(idx),NTOTAL,3),np.nan).float()
    xyz[:,:14,:] = pdb['xyz']
    mask = torch.full((len(idx), NTOTAL), False)
    mask[:,:14] = pdb['mask']
    xyz = torch.nan_to_num(xyz)

    # get template features
    ntempl = np.random.randint(params['MINTPLT'], params['MAXTPLT']+1)
    xyz_t, f1d_t, mask_t = TemplFeaturize(tplt, msa.shape[1], params, npick=ntempl, offset=0, pick_top=pick_top, random_noise=random_noise)

    # Residue cropping
    crop_idx = get_crop(len(idx), mask, msa_seed_orig.device, params['CROP'], unclamp=unclamp)
    seq = seq[:,crop_idx]
    msa_seed_orig = msa_seed_orig[:,:,crop_idx]
    msa_seed = msa_seed[:,:,crop_idx]
    msa_extra = msa_extra[:,:,crop_idx]
    mask_msa = mask_msa[:,:,crop_idx]
    xyz_t = xyz_t[:,crop_idx]
    f1d_t = f1d_t[:,crop_idx]
    mask_t = mask_t[:,crop_idx]
    xyz = xyz[crop_idx]
    mask = mask[crop_idx]
    idx = idx[crop_idx]

    # get initial coordinates
    xyz_prev = xyz_t[0].clone()
    mask_prev = mask_t[0].clone()
    chain_idx = torch.ones((len(crop_idx), len(crop_idx))).long()
    bond_feats = get_protein_bond_feats(len(crop_idx)).long()
    chirals = torch.Tensor()
    #print ("loader_single", mask.shape, xyz_t.shape, f1d_t.shape, xyz_prev.shape)

    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa, \
           xyz.float(), mask, idx.long(),\
           xyz_t.float(), f1d_t.float(), mask_t, \
           xyz_prev.float(), mask_prev, \
           chain_idx, unclamp, False, torch.zeros(seq.shape), bond_feats, chirals

# Generate input features for homo-oligomers
def featurize_homo(msa_orig, ins_orig, tplt, pdbA, pdbid, interfaces, params, pick_top=True, random_noise=5.0, fixbb=False):
    L = msa_orig.shape[1]
    
    msa, ins = merge_a3m_homo(msa_orig, ins_orig, 2) # make unpaired alignments, for training, we always use two chains
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params, nmer=2, L_s=[L,L], fixbb=fixbb)

    # get template features
    ntempl = np.random.randint(params['MINTPLT'], params['MAXTPLT']+1)
    if ntempl < 1:
        xyz_t, f1d_t, mask_t = TemplFeaturize(tplt, 2*L, params, npick=ntempl, offset=0, pick_top=pick_top, random_noise=random_noise)
    else:
        xyz_t_single, f1d_t_single, mask_t_single = TemplFeaturize(tplt, L, params, npick=ntempl, offset=0, pick_top=pick_top, random_noise=random_noise)
        # duplicate
        xyz_t = torch.cat((xyz_t_single, random_rot_trans(xyz_t_single)), dim=1) # (ntempl, 2*L, natm, 3)
        f1d_t = torch.cat((f1d_t_single, f1d_t_single), dim=1) # (ntempl, 2*L, 21)
        mask_t = torch.cat((mask_t_single, mask_t_single), dim=1) # (ntempl, 2*L, natm)

    # get initial coordinates
    xyz_prev = xyz_t[0].clone()
    mask_prev = mask_t[0].clone()

    # get ground-truth structures
    # load metadata
    PREFIX = "%s/torch/pdb/%s/%s"%(params['PDB_DIR'],pdbid[1:3],pdbid)
    meta = torch.load(PREFIX+".pt")

    npairs = len(interfaces)
    xyz = torch.full((npairs, 2*L, NTOTAL, 3), np.nan).float()
    mask = torch.full((npairs, 2*L, NTOTAL), False)
    for i_int,interface in enumerate(interfaces):
        pdbB = torch.load(params['PDB_DIR']+'/torch/pdb/'+interface['CHAIN_B'][1:3]+'/'+interface['CHAIN_B']+'.pt')
        xformA = meta['asmb_xform%d'%interface['ASSM_A']][interface['OP_A']]
        xformB = meta['asmb_xform%d'%interface['ASSM_B']][interface['OP_B']]
        xyzA = torch.einsum('ij,raj->rai', xformA[:3,:3], pdbA['xyz']) + xformA[:3,3][None,None,:]
        xyzB = torch.einsum('ij,raj->rai', xformB[:3,:3], pdbB['xyz']) + xformB[:3,3][None,None,:]
        xyz[i_int,:,:14] = torch.cat((xyzA, xyzB), dim=0)
        mask[i_int,:,:14] = torch.cat((pdbA['mask'], pdbB['mask']), dim=0)
    xyz = torch.nan_to_num(xyz)

    idx = torch.arange(L*2)
    idx[L:] += CHAIN_GAP # to let network know about chain breaks

    # indicator for which residues are in same chain
    chain_idx = torch.zeros((2*L, 2*L)).long()
    chain_idx[:L, :L] = 1
    chain_idx[L:, L:] = 1
    bond_feats = torch.zeros((2*L, 2*L)).long()
    bond_feats[:L, :L] = get_protein_bond_feats(L)
    bond_feats[L:, L:] = get_protein_bond_feats(L)

    # Residue cropping
    if 2*L > params['CROP']:
        if np.random.rand() < 0.5: # 50% --> interface crop
            spatial_crop_tgt = np.random.randint(0, npairs)
            crop_idx = get_spatial_crop(xyz[spatial_crop_tgt], mask[spatial_crop_tgt], torch.arange(L*2), [L,L], params, interfaces[spatial_crop_tgt]['CHAIN_B'])
        else: # 50% --> have same cropped regions across all copies
            crop_idx = get_crop(L, mask[0,:L], msa_seed_orig.device, params['CROP']//2, unclamp=False) # cropped region for first copy
            crop_idx = torch.cat((crop_idx, crop_idx+L)) # get same crops
        seq = seq[:,crop_idx]
        msa_seed_orig = msa_seed_orig[:,:,crop_idx]
        msa_seed = msa_seed[:,:,crop_idx]
        msa_extra = msa_extra[:,:,crop_idx]
        mask_msa = mask_msa[:,:,crop_idx]
        xyz_t = xyz_t[:,crop_idx]
        f1d_t = f1d_t[:,crop_idx]
        mask_t = mask_t[:,crop_idx]
        xyz = xyz[:,crop_idx]
        mask = mask[:,crop_idx]
        idx = idx[crop_idx]
        chain_idx = chain_idx[crop_idx][:,crop_idx]
        bond_feats = bond_feats[crop_idx][:,crop_idx]
        xyz_prev = xyz_prev[crop_idx]
        mask_prev = mask_prev[crop_idx]
    chirals = torch.Tensor()
    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa, \
           xyz.float(), mask, idx.long(),\
           xyz_t.float(), f1d_t.float(), mask_t, \
           xyz_prev.float(), mask_prev, \
           chain_idx, False, False, torch.zeros(seq.shape), bond_feats, chirals


def get_pdb(pdbfilename, plddtfilename, item, lddtcut, sccut):
    xyz, mask, res_idx = parse_pdb(pdbfilename)
    plddt = np.load(plddtfilename)
    
    # update mask info with plddt (ignore sidechains if plddt < 90.0)
    mask_lddt = np.full_like(mask, False)
    mask_lddt[plddt > sccut] = True
    mask_lddt[:,:5] = True
    mask = np.logical_and(mask, mask_lddt)
    mask = np.logical_and(mask, (plddt > lddtcut)[:,None])
    
    return {'xyz':torch.tensor(xyz), 'mask':torch.tensor(mask), 'idx': torch.tensor(res_idx), 'label':item}

def get_msa(a3mfilename, item, unzip=True):
    msa,ins, taxIDs = parse_a3m(a3mfilename, unzip=unzip)
    return {'msa':torch.tensor(msa), 'ins':torch.tensor(ins), 'taxIDs':taxIDs, 'label':item}

# Load PDB examples
def loader_pdb(item, params, homo, unclamp=False, pick_top=True, p_homo_cut=0.5, fixbb=False):
    # load MSA, PDB, template info
    pdb_chain, pdb_hash = item['CHAINID'], item['HASH']
    pdb = torch.load(params['PDB_DIR']+'/torch/pdb/'+pdb_chain[1:3]+'/'+pdb_chain+'.pt')
    a3m = get_msa(params['PDB_DIR'] + '/a3m/' + pdb_hash[:3] + '/' + pdb_hash + '.a3m.gz', pdb_hash)
    tplt = torch.load(params['PDB_DIR']+'/torch/hhr/'+pdb_hash[:3]+'/'+pdb_hash+'.pt')

    # get msa features
    msa = a3m['msa'].long()
    ins = a3m['ins'].long()
    if len(msa) > params['BLOCKCUT']:
        msa, ins = MSABlockDeletion(msa, ins)

    if pdb_chain in homo['CHAIN_A'].values: # Target is homo-oligomer
        p_homo = np.random.rand()
        if p_homo < p_homo_cut: # model as homo-oligomer with p_homo_cut prob
            pdbid = pdb_chain.split('_')[0]
            interfaces = homo[homo['CHAIN_A']==pdb_chain].to_dict(orient='records') # list of dicts
            feats = featurize_homo(msa, ins, tplt, pdb, pdbid, interfaces, params, pick_top=pick_top, fixbb=fixbb)
            return feats + (torch.zeros((msa.shape[1],)), "homo",item,)
        else:
            return featurize_single_chain(msa, ins, tplt, pdb, params, unclamp=unclamp, pick_top=pick_top, fixbb=fixbb) \
                   + (torch.zeros((msa.shape[1],)), "monomer",item,)
    else:
        return featurize_single_chain(msa, ins, tplt, pdb, params, unclamp=unclamp, pick_top=pick_top, fixbb=fixbb) \
               + (torch.zeros((msa.shape[1],)), "monomer",item,)

    
def loader_fb(item, params, unclamp=False, fixbb=False):
    
    # loads sequence/structure/plddt information
    pdb_chain, hashstr = item['CHAINID'], item['HASH']
    a3m = get_msa(os.path.join(params["FB_DIR"], "a3m", hashstr[:2], hashstr[2:], pdb_chain+".a3m.gz"), pdb_chain)
    pdb = get_pdb(os.path.join(params["FB_DIR"], "pdb", hashstr[:2], hashstr[2:], pdb_chain+".pdb"),
                  os.path.join(params["FB_DIR"], "pdb", hashstr[:2], hashstr[2:], pdb_chain+".plddt.npy"),
                  pdb_chain, params['PLDDTCUT'], params['SCCUT'])
   
    # get msa features
    msa = a3m['msa'].long()
    ins = a3m['ins'].long()
    l_orig = msa.shape[1]
    if len(msa) > params['BLOCKCUT']:
        msa, ins = MSABlockDeletion(msa, ins)
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params, fixbb=fixbb)
    
    # get template features -- None
    tplt_blank = {"ids":[]}
    xyz_t, f1d_t, mask_t = TemplFeaturize(tplt_blank, l_orig, params, offset=0, npick=0)  

    idx = pdb['idx']
    xyz = torch.full((len(idx),NTOTAL,3),np.nan).float()
    xyz[:,:27,:] = pdb['xyz'][:,:27]
    mask = torch.full((len(idx),NTOTAL), False)
    mask[:,:27] = pdb['mask'][:,:27]

    # Residue cropping
    crop_idx = get_crop(len(idx), mask, msa_seed_orig.device, params['CROP'], unclamp=unclamp)
    seq = seq[:,crop_idx]
    msa_seed_orig = msa_seed_orig[:,:,crop_idx]
    msa_seed = msa_seed[:,:,crop_idx]
    msa_extra = msa_extra[:,:,crop_idx]
    mask_msa = mask_msa[:,:,crop_idx]
    xyz_t = xyz_t[:,crop_idx]
    f1d_t = f1d_t[:,crop_idx]
    mask_t = mask_t[:, crop_idx]
    xyz = xyz[crop_idx]
    mask = mask[crop_idx]
    idx = idx[crop_idx]

    # initial structure
    xyz_prev = xyz_t[0].clone()
    mask_prev = mask_t[0].clone()
    chain_idx = torch.ones((len(crop_idx), len(crop_idx))).long()
    bond_feats = get_protein_bond_feats(len(crop_idx)).long()
    chirals = torch.Tensor()
    ch_label = torch.zeros(seq[0].shape)
    #print ("loader_fb", mask.shape, xyz_t.shape, f1d_t.shape, xyz_prev.shape)
    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa, \
           xyz.float(), mask, idx.long(),\
           xyz_t.float(), f1d_t.float(), mask_t, \
           xyz_prev.float(), mask_prev, \
           chain_idx, unclamp, False, torch.zeros(seq.shape), bond_feats, chirals, ch_label, "fb", item

def loader_complex(item, params, negative=False, pick_top=True, random_noise=5.0, fixbb=False):

    pdb_pair, pMSA_hash, L_s, taxID = item['CHAINID'], item['HASH'], item['LEN'], item['TAXONOMY']
    msaA_id, msaB_id = pMSA_hash.split('_')
    
    if len(set(taxID.split(':'))) == 1: # two proteins have same taxID -- use paired MSA
        # read pMSA
        if negative:
            pMSA_fn = params['COMPL_DIR'] + '/pMSA.negative/' + msaA_id[:3] + '/' + msaB_id[:3] + '/' + pMSA_hash + '.a3m.gz'
        else:
            pMSA_fn = params['COMPL_DIR'] + '/pMSA/' + msaA_id[:3] + '/' + msaB_id[:3] + '/' + pMSA_hash + '.a3m.gz'
        a3m = get_msa(pMSA_fn, pMSA_hash, unzip=True)
    else:
        # read MSA for each subunit & merge them
        a3mA_fn = params['PDB_DIR'] + '/a3m/' + msaA_id[:3] + '/' + msaA_id + '.a3m.gz'
        a3mB_fn = params['PDB_DIR'] + '/a3m/' + msaB_id[:3] + '/' + msaB_id + '.a3m.gz'
        a3mA = get_msa(a3mA_fn, msaA_id)
        a3mB = get_msa(a3mB_fn, msaB_id)
        a3m = merge_a3m_hetero(a3mA, a3mB, L_s)

    # get MSA features
    msa = a3m['msa'].long()
    if negative: # Qian's paired MSA for true-pairs have no insertions... (ignore insertion to avoid any weird bias..) 
        ins = torch.zeros_like(msa)
    else:
        ins = a3m['ins'].long()
    if len(msa) > params['BLOCKCUT']:
        msa, ins = MSABlockDeletion(msa, ins)
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params, L_s=L_s, fixbb=fixbb)

    # read template info
    tpltA_fn = params['PDB_DIR'] + '/torch/hhr/' + msaA_id[:3] + '/' + msaA_id + '.pt'
    tpltB_fn = params['PDB_DIR'] + '/torch/hhr/' + msaB_id[:3] + '/' + msaB_id + '.pt'
    tpltA = torch.load(tpltA_fn)
    tpltB = torch.load(tpltB_fn)

    ntemplA = np.random.randint(params['MINTPLT'], params['MAXTPLT']+1)
    ntemplB = np.random.randint(0, params['MAXTPLT']+1-ntemplA)
    xyz_t_A, f1d_t_A, mask_t_A = TemplFeaturize(tpltA, L_s[0], params, offset=0, npick=ntemplA, npick_global=max(1,max(ntemplA, ntemplB)), pick_top=pick_top, random_noise=random_noise)
    xyz_t_B, f1d_t_B, mask_t_B = TemplFeaturize(tpltB, L_s[1], params, offset=0, npick=ntemplB, npick_global=max(1,max(ntemplA, ntemplB)), pick_top=pick_top, random_noise=random_noise)
    xyz_t = torch.cat((xyz_t_A, random_rot_trans(xyz_t_B)), dim=1) # (T, L1+L2, natm, 3)
    f1d_t = torch.cat((f1d_t_A, f1d_t_B), dim=1) # (T, L1+L2, natm, 3)
    mask_t = torch.cat((mask_t_A, mask_t_B), dim=1) # (T, L1+L2, natm, 3)

    # get initial coordinates
    xyz_prev = xyz_t[0].clone()
    mask_prev = mask_t[0].clone()

    # read PDB
    pdbA_id, pdbB_id = pdb_pair.split(':')
    pdbA = torch.load(params['PDB_DIR']+'/torch/pdb/'+pdbA_id[1:3]+'/'+pdbA_id+'.pt')
    pdbB = torch.load(params['PDB_DIR']+'/torch/pdb/'+pdbB_id[1:3]+'/'+pdbB_id+'.pt')

    if not negative:
        # read metadata
        pdbid = pdbA_id.split('_')[0]
        meta = torch.load(params['PDB_DIR']+'/torch/pdb/'+pdbid[1:3]+'/'+pdbid+'.pt')

        # get transform
        xformA = meta['asmb_xform%d'%item['ASSM_A']][item['OP_A']]
        xformB = meta['asmb_xform%d'%item['ASSM_B']][item['OP_B']]    
        
        # apply transform
        xyzA = torch.einsum('ij,raj->rai', xformA[:3,:3], pdbA['xyz']) + xformA[:3,3][None,None,:]
        xyzB = torch.einsum('ij,raj->rai', xformB[:3,:3], pdbB['xyz']) + xformB[:3,3][None,None,:]
        xyz = torch.full((sum(L_s), NTOTAL, 3), np.nan).float()
        xyz[:,:14] = torch.cat((xyzA, xyzB), dim=0)
        mask = torch.full((sum(L_s), NTOTAL), False)
        mask[:,:14] = torch.cat((pdbA['mask'], pdbB['mask']), dim=0)
    else:
        xyz = torch.full((sum(L_s), NTOTAL, 3), np.nan).float()
        xyz[:,:14] = torch.cat((pdbA['xyz'], pdbB['xyz']), dim=0)
        mask = torch.full((sum(L_s), NTOTAL), False)
        mask[:,:14] = torch.cat((pdbA['mask'], pdbB['mask']), dim=0)
    xyz = torch.nan_to_num(xyz)

    idx = torch.arange(sum(L_s))
    idx[L_s[0]:] += CHAIN_GAP

    chain_idx = torch.zeros((sum(L_s), sum(L_s))).long()
    chain_idx[:L_s[0], :L_s[0]] = 1
    chain_idx[L_s[0]:, L_s[0]:] = 1
    bond_feats = torch.zeros((sum(L_s), sum(L_s))).long()
    bond_feats[:L_s[0], :L_s[0]] = get_protein_bond_feats(L_s[0])
    bond_feats[L_s[0]:, L_s[0]:] = get_protein_bond_feats(sum(L_s[1:]))

    # Do cropping
    if sum(L_s) > params['CROP']:
        if negative:
            sel = get_complex_crop(L_s, mask, seq.device, params)
        else:
            sel = get_spatial_crop(xyz, mask, torch.arange(sum(L_s)), L_s, params, pdb_pair)
        #
        seq = seq[:,sel]
        msa_seed_orig = msa_seed_orig[:,:,sel]
        msa_seed = msa_seed[:,:,sel]
        msa_extra = msa_extra[:,:,sel]
        mask_msa = mask_msa[:,:,sel]
        xyz = xyz[sel]
        mask = mask[sel]
        xyz_t = xyz_t[:,sel]
        f1d_t = f1d_t[:,sel]
        mask_t = mask_t[:,sel]
        xyz_prev = xyz_prev[sel]
        mask_prev = mask_prev[sel]
        #
        idx = idx[sel]
        chain_idx = chain_idx[sel][:,sel]
        bond_feats = bond_feats[sel][:,sel]
    chirals = torch.Tensor()
    L1 = chain_idx[0,:].sum()
    ch_label = torch.zeros(seq[0].shape)
    ch_label[L1:] = 1
    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa,\
           xyz.float(), mask, idx.long(), \
           xyz_t.float(), f1d_t.float(), mask_t, \
           xyz_prev.float(), mask_prev, \
           chain_idx, False, negative, torch.zeros(seq.shape), bond_feats, chirals, ch_label, "compl", item

def loader_na_complex(item, params, native_NA_frac=0.25, negative=False, pick_top=True, random_noise=5.0):
    pdb_set = item['CHAINID']
    msa_id = item['HASH']
    Ls = item['LEN']

    # read MSA for protein
    a3mA = get_msa(params['PDB_DIR'] + '/a3m/' + msa_id[:3] + '/' + msa_id + '.a3m.gz', msa_id)

    # read PDBs
    pdb_ids = pdb_set.split(':')
    pdbA = torch.load(params['PDB_DIR']+'/torch/pdb/'+pdb_ids[0][1:3]+'/'+pdb_ids[0]+'.pt')
    pdbB = torch.load(params['NA_DIR']+'/torch/'+pdb_ids[1][1:3]+'/'+pdb_ids[1]+'.pt')
    pdbC = None
    if (len(pdb_ids)==3):
        pdbC = torch.load(params['NA_DIR']+'/torch/'+pdb_ids[2][1:3]+'/'+pdb_ids[2]+'.pt')

    # msa for NA is sequence only
    msaB,insB = parse_fasta_if_exists(
        pdbB['seq'], params['NA_DIR']+'/torch/'+pdb_ids[1][1:3]+'/'+pdb_ids[1]+'.afa', 
        maxseq=5000,
        rmsa_alphabet=True
    )
    a3mB = {'msa':torch.from_numpy(msaB), 'ins':torch.from_numpy(insB)}
    NMDLS=1
    if (len(pdb_ids)==3):
        msaC,insC = parse_fasta_if_exists(
            pdbC['seq'], params['NA_DIR']+'/torch/'+pdb_ids[2][1:3]+'/'+pdb_ids[2]+'.afa', 
            maxseq=5000,
            rmsa_alphabet=True
        )
        a3mC = {'msa':torch.from_numpy(msaC), 'ins':torch.from_numpy(insC)}
        a3mB = merge_a3m_hetero(a3mB, a3mC, Ls[1:])
        if (pdbB['seq']==pdbC['seq']):
            NMDLS=2 # flip B and C
    a3m = merge_a3m_hetero(a3mA, a3mB, [Ls[0],sum(Ls[1:])])

    # note: the block below is due to differences in the way RNA and DNA structures are processed
    # to support NMR, RNA structs return multiple states
    # For protein/NA complexes get rid of the 'NMODEL' dimension (if present)
    # NOTE there are a very small number of protein/NA NMR models:
    #       - ideally these should return the ensemble, but that requires reprocessing of PDBs
    if (len(pdbB['xyz'].shape) > 3):
         pdbB['xyz'] = pdbB['xyz'][0,...]
         pdbB['mask'] = pdbB['mask'][0,...]
    if (pdbC is not None and len(pdbC['xyz'].shape) > 3):
         pdbC['xyz'] = pdbC['xyz'][0,...]
         pdbC['mask'] = pdbC['mask'][0,...]

    # read template info
    tpltA = torch.load(params['PDB_DIR'] + '/torch/hhr/' + msa_id[:3] + '/' + msa_id + '.pt')
    ntempl = np.random.randint(params['MINTPLT'], params['MAXTPLT']-1)
    xyz_t, f1d_t, mask_t = TemplFeaturize(tpltA, sum(Ls), params, offset=0, npick=ntempl, pick_top=pick_top, random_noise=random_noise) 
    xyz_t[:,Ls[0]:] = INIT_NA_CRDS.reshape(1,1,NTOTAL,3).repeat(1,sum(Ls[1:]),1,1) + torch.rand(1,sum(Ls[1:]),1,3)*random_noise - random_noise/2

    if (np.random.rand()<=native_NA_frac):
        natNA_templ = pdbB['xyz']
        maskNA_templ = pdbB['mask']

        if pdbC is not None:
            natNA_templ = torch.cat((pdbB['xyz'], pdbC['xyz']), dim=0)
            maskNA_templ =  torch.cat((pdbB['mask'], pdbC['mask']), dim=0)

        # construct template from NA
        xyz_t_B = INIT_CRDS.reshape(1,1,NTOTAL,3).repeat(1,sum(Ls),1,1) + torch.rand(1,sum(Ls),1,3)*random_noise - random_noise/2
        #xyz_t_B[:,Ls[0]:,:23] = natNA_templ
        mask_t_B = torch.full((1,sum(Ls),NTOTAL), False)
        mask_t_B[:,Ls[0]:,:23] = maskNA_templ
        xyz_t_B[mask_t_B] = natNA_templ[maskNA_templ]

        seq_t_B = torch.cat( (torch.full((1, Ls[0]), 20).long(),  a3mB['msa'][0:1]), dim=1)
        seq_t_B[seq_t_B>21] -= 1 # remove mask token
        f1d_t_B = torch.nn.functional.one_hot(seq_t_B, num_classes=NAATOKENS-1).float()
        conf_B = torch.cat( (
            torch.zeros((1,Ls[0],1)),
            torch.full((1,sum(Ls[1:]),1),1.0),
        ),dim=1).float()
        f1d_t_B = torch.cat((f1d_t_B, conf_B), -1)

        xyz_t = torch.cat((xyz_t,xyz_t_B),dim=0)
        f1d_t = torch.cat((f1d_t,f1d_t_B),dim=0)
        mask_t = torch.cat((mask_t,mask_t_B),dim=0)

    # get MSA features
    msa = a3m['msa'].long()
    ins = a3m['ins'].long()
    if len(msa) > params['BLOCKCUT']:
        msa, ins = MSABlockDeletion(msa, ins)
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params, L_s=Ls)

    xyz = torch.full((NMDLS, sum(Ls), NTOTAL, 3), np.nan).float()
    mask = torch.full((NMDLS, sum(Ls), NTOTAL), False)
    if (len(pdb_ids)==3):
        xyz[:,:Ls[0],:14] = pdbA['xyz'][None,...]
        xyz[0,Ls[0]:,:23] = torch.cat((pdbB['xyz'], pdbC['xyz']), dim=0)
        mask[:,:Ls[0],:14] = pdbA['mask'][None,...]
        mask[0,Ls[0]:,:23] = torch.cat((pdbB['mask'], pdbC['mask']), dim=0)
        if (NMDLS==2): # B & C are identical
            xyz[1,Ls[0]:,:23] = torch.cat((pdbC['xyz'], pdbB['xyz']), dim=0)
            mask[1,Ls[0]:,:23] = torch.cat((pdbC['mask'], pdbB['mask']), dim=0)
    else:
        xyz[0,:Ls[0],:14] = pdbA['xyz']
        xyz[0,Ls[0]:,:23] = pdbB['xyz']
        mask[0,:Ls[0],:14] = pdbA['mask']
        mask[0,Ls[0]:,:23] = pdbB['mask']
    xyz = torch.nan_to_num(xyz)

    # other features
    idx = idx_from_Ls(Ls)
    same_chain = same_chain_2d_from_Ls(Ls)

    bond_feats = torch.zeros((sum(Ls), sum(Ls))).long()
    offset = 0
    for L_ in Ls:
        bond_feats[offset:offset+L_, offset:offset+L_] = get_protein_bond_feats(L_)
        offset += L_

    ch_label = torch.cat([torch.full((L_,), i) for i,L_ in enumerate(Ls)]).long()

    # Do cropping
    if sum(Ls) > params['CROP']:
        cropref = np.random.randint(xyz.shape[0])
        sel = get_na_crop(seq[0], xyz[cropref], mask[cropref], torch.arange(sum(Ls)), Ls, params, negative)

        seq = seq[:,sel]
        msa_seed_orig = msa_seed_orig[:,:,sel]
        msa_seed = msa_seed[:,:,sel]
        msa_extra = msa_extra[:,:,sel]
        mask_msa = mask_msa[:,:,sel]
        xyz = xyz[:,sel]
        mask = mask[:,sel]
        xyz_t = xyz_t[:,sel]
        f1d_t = f1d_t[:,sel]
        mask_t = mask_t[:,sel]
        #
        idx = idx[sel]
        same_chain = same_chain[sel][:,sel]
        bond_feats = bond_feats[sel][:,sel]
        ch_label = ch_label[sel]

    xyz_prev = xyz_t[0].clone()
    mask_prev = mask_t[0].clone()
    chirals = torch.Tensor()

    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa,\
           xyz.float(), mask, idx.long(), \
           xyz_t.float(), f1d_t.float(), mask_t, \
           xyz_prev.float(), mask_prev, \
           same_chain, False, negative, torch.zeros(seq.shape), bond_feats, chirals, ch_label, "na_compl", item


def loader_rna(item, params, random_noise=5.0):
    # read PDBs
    pdb_ids = item['CHAINID'].split(':')
    Ls = item['LEN']

    pdbA = torch.load(params['NA_DIR']+'/torch/'+pdb_ids[0][1:3]+'/'+pdb_ids[0]+'.pt')
    pdbB = None
    if (len(pdb_ids)==2):
        pdbB = torch.load(params['NA_DIR']+'/torch/'+pdb_ids[1][1:3]+'/'+pdb_ids[1]+'.pt')

    # msa for NA is sequence only
    msaA,insA = parse_fasta_if_exists(pdbA['seq'], params['NA_DIR']+'/torch/'+pdb_ids[0][1:3]+'/'+pdb_ids[0]+'.afa', rmsa_alphabet=True)
    a3m = {'msa':torch.from_numpy(msaA), 'ins':torch.from_numpy(insA)}
    if (len(pdb_ids)==2):
        msaB,insB = parse_fasta_if_exists(pdbB['seq'], params['NA_DIR']+'/torch/'+pdb_ids[1][1:3]+'/'+pdb_ids[1]+'.afa', rmsa_alphabet=True)
        a3mB = {'msa':torch.from_numpy(msaB), 'ins':torch.from_numpy(insB)}
        a3m = merge_a3m_hetero(a3m, a3mB, Ls)

    # get template features -- None
    L = sum(Ls)
    xyz_t = INIT_NA_CRDS.reshape(1,1,NTOTAL,3).repeat(1,L,1,1) + torch.rand(1,L,1,3)*random_noise
    f1d_t = torch.nn.functional.one_hot(torch.full((1, L), 20).long(), num_classes=NAATOKENS-1).float() # all gaps
    mask_t = torch.full((1,L,NTOTAL), False)
    conf = torch.zeros((1,L,1)).float() # zero confidence
    f1d_t = torch.cat((f1d_t, conf), -1)

    NMDLS = pdbA['xyz'].shape[0]

    # get MSA features
    msa = a3m['msa'].long()
    ins = a3m['ins'].long()
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params, L_s=Ls)

    xyz = torch.full((NMDLS, L, NTOTAL, 3), np.nan).float()
    mask = torch.full((NMDLS, L, NTOTAL), False)
    if (len(pdb_ids)==2):
        xyz[:,:,:23] = torch.cat((pdbA['xyz'], pdbB['xyz']), dim=1)
        mask[:,:,:23] = torch.cat((pdbA['mask'], pdbB['mask']), dim=1)
    else:
        xyz[:,:,:23] = pdbA['xyz']
        mask[:,:,:23] = pdbA['mask']

    idx = torch.arange(L)
    if (len(pdb_ids)==2):
        idx[Ls[0]:] += CHAIN_GAP

    same_chain = same_chain_2d_from_Ls(Ls)
    bond_feats = torch.zeros((L, L)).long()
    bond_feats[:Ls[0],:Ls[0]] = get_protein_bond_feats(Ls[0]) # assumes 2 chains
    bond_feats[Ls[0]:,Ls[0]:] = get_protein_bond_feats(Ls[1]) # assumes 2 chains

    # Do cropping
    if sum(Ls) > params['CROP']:
        cropref = np.random.randint(xyz.shape[0])
        sel = get_na_crop(seq[0], xyz[cropref], mask[cropref], torch.arange(L), Ls, params, incl_protein=False)

        seq = seq[:,sel]
        msa_seed_orig = msa_seed_orig[:,:,sel]
        msa_seed = msa_seed[:,:,sel]
        msa_extra = msa_extra[:,:,sel]
        mask_msa = mask_msa[:,:,sel]
        xyz = xyz[:,sel]
        mask = mask[:,sel]
        xyz_t = xyz_t[:,sel]
        f1d_t = f1d_t[:,sel]
        mask_t = mask_t[:,sel]
        #
        idx = idx[sel]
        chain_idx = chain_idx[sel][:,sel]
        bond_feats = bond_feats[sel][:, sel]

    xyz_prev = xyz_t[0].clone()
    mask_prev = mask_t[0].clone()   
    chirals = torch.Tensor()
    ch_label = torch.zeros((L,)).long()
    ch_label[Ls[0]:] = 1

    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa,\
           xyz.float(), mask, idx.long(), \
           xyz_t.float(), f1d_t.float(), mask_t, \
           xyz_prev.float(), mask_prev, \
           same_chain, False, False, torch.zeros(seq.shape), bond_feats.long(), chirals, ch_label, "rna",item

def loader_sm_compl(item, params, pick_top=True,
    init_protein_tmpl=False, init_ligand_tmpl=False,
    init_protein_xyz=False, init_ligand_xyz=False, random_noise=5.0, fixbb=False):
    """Load protein/SM complex with mixed residue and atom tokens. Also,
    compute frames for atom FAPE loss calc"""

    pdb_chain, pdb_hash, ligands = item
    #with open(f'sm_compl_item_params_{pdb_chain}.pkl', 'wb') as f:
    #    pickle.dump([item, params], f)

    # Load protein information
    pdbA = torch.load(params['PDB_DIR']+'/torch/pdb/'+pdb_chain[1:3]+'/'+pdb_chain+'.pt')
    a3mA = get_msa(params['PDB_DIR'] + '/a3m/'+pdb_hash[:3] + '/'+ pdb_hash + '.a3m.gz', pdb_hash)
    tpltA = torch.load(params['PDB_DIR']+'/torch/hhr/'+pdb_hash[:3]+'/'+pdb_hash+'.pt')
   
    # get msa features
    msa_prot = a3mA['msa'].long()
    ins_prot = a3mA['ins'].long()

    if len(msa_prot) > params['BLOCKCUT']:
        msa_prot, ins_prot = MSABlockDeletion(msa_prot, ins_prot)
    a3m_prot = {"msa": msa_prot, "ins": ins_prot}

    # load pre-parsed cif assembly - requires cifutils.py in path for object definitions
    chains, asmb, covale, modres = pickle.load(gzip.open(params['MOL_DIR']+f'/{pdb_id[1:3]}/{pdb_id}.pkl.gz'))

    # coordinate transforms to recreate this bio-assembly
    i_a = str(item['ASSEMBLY'])
    asmb_xfs = asmb[i_a]

    # 1st protein chain in list is the main binding partner
    for p in item['PARTNERS']:
        if p[0]==i_ch_prot:
            i_xf_prot = p[1]
            break

    # load protein chain
    ch = chains[i_ch_prot]
    ch_xf = asmb_xfs[i_xf_prot]
    xyz_prot, mask_prot, seq_prot, chid_prot, resi_prot, _ = cif_prot_to_xyz(ch, ch_xf, modres)
    protein_L, nprotatoms, _ = xyz_prot.shape

    if len(ligands):
        # Load small molecule
        i_lig = np.random.randint(len(ligands))
        ligname = ligands[i_lig].split('_')[1]
        filename = params["MOL_DIR"]+'/'+ligname[0]+'/'+ligname+'/'+ligands[i_lig].replace('.mol2','.isdf')
        mol, msa_sm, ins_sm, xyz_sm, mask_sm = parse_mol(filename, filetype="sdf")
        for alt_lig in ligands[:i_lig]+ligands[i_lig+1:]:
            ligname = alt_lig.split('_')[1]
            filename = params["MOL_DIR"]+'/'+ligname[0]+'/'+ligname+'/'+alt_lig.replace('.mol2','.isdf')
            mol2, msa_sm2, ins_sm2, xyz_sm2, mask_sm2 = parse_mol(filename, filetype='sdf')
            if (msa_sm2.shape == msa_sm.shape) and all(msa_sm2==msa_sm):
                xyz_sm = torch.concat([xyz_sm, xyz_sm2],dim=0) # (N_symm1 + N_symm2, Natoms, 3)
                mask_sm = torch.concat([mask_sm, mask_sm2],dim=0)
            else:
                print(f'WARNING [loader_sm_compl]: Ligands at different bindings sites don\'t have same '\
                      f'atom order: {item[0]}: {ligands[i_lig]} vs {alt_lig}. Skipping latter ligand.')

        USENSYM = params['MAXNSYMM']
        if fixbb:
            USENSYM = 1
        # clamp number of symmetry variants to save GPU memory
        if xyz_sm.shape[0] > USENSYM: 
            xyz_sm = xyz_sm[:USENSYM]
            mask_sm = mask_sm[:USENSYM]      

        if xyz_sm.shape[0] ==0:
            print(f'ERROR [loader_sm_compl]: {item[0]} had no xyz coords')
            return (torch.tensor([-1]),)*21

        a3m_sm = {"msa": msa_sm.unsqueeze(0), "ins": ins_sm.unsqueeze(0)}
        G = get_nxgraph(mol)
        frames = get_atom_frames(msa_sm, G)
        chirals = get_chirals(mol, xyz_sm[0])
        if chirals.shape[0] == 0:
            chirals = torch.zeros((0,4))
    else:
        xyz_sm = torch.zeros((1,0,1))
        mask_sm = torch.zeros((1,0))
        chirals = torch.zeros((0,4)) # 4 might not be right but it's empty

    # Generate ground truth structure: account for ligand symmetry
    N_symmetry, sm_L, _ = xyz_sm.shape
    xyz = torch.full((N_symmetry, protein_L+sm_L, NTOTAL, 3), np.nan).float()
    mask = torch.full(xyz.shape[:-1], False).bool()
    xyz[:, :protein_L, :nprotatoms, :] = xyz_prot.expand(N_symmetry, protein_L, nprotatoms, 3)
    xyz[:, protein_L:, 1, :] = xyz_sm
    mask[:, :protein_L, :nprotatoms] = mask_prot.expand(N_symmetry, protein_L, nprotatoms)
    mask[:, protein_L:, 1] = mask_sm

    Ls = [xyz_prot.shape[0], xyz_sm.shape[1]]

    a3m_to_check = [('protein', a3m_prot)]
    if len(ligands):
        a3m_to_check.append(('ligand', a3m_sm))
    else:
        frames = torch.zeros((0,3,2))

    for i, (a3m_name, a3m) in enumerate(a3m_to_check):
        if not (a3m['msa'].shape[1]==Ls[i]):
            print(f'WARNING [loader_sm_compl]: {a3m_name} XYZ and MSA lengths don\'t match: {item}. Skipping.')
            return (torch.tensor([-1]),)*21

    if len(ligands):
        a3m = merge_a3m_hetero(a3m_prot, a3m_sm, Ls)
    else:
        a3m = a3m_prot
    msa = a3m['msa'].long()
    ins = a3m['ins'].long()

    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params, fixbb=fixbb)

    idx = torch.arange(sum(Ls))
    idx[Ls[0]:] += CHAIN_GAP

    chain_idx = torch.zeros((sum(Ls), sum(Ls))).long()
    chain_idx[:Ls[0], :Ls[0]] = 1
    chain_idx[Ls[0]:, Ls[0]:] = 1
    bond_feats = torch.zeros((sum(Ls), sum(Ls))).long()
    bond_feats[:Ls[0], :Ls[0]] = get_protein_bond_feats(Ls[0])
    if len(ligands):
        bond_feats[Ls[0]:, Ls[0]:] = get_bond_feats(mol)

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
            xyz_t[0, :Ls[0], :3] = xyz[0, :Ls[0], :3]
            f1d_t[0, :Ls[0]] = torch.cat((
                torch.nn.functional.one_hot(msa_seed_orig[0,0, :Ls[0] ], num_classes=NAATOKENS-1).float(),
                torch.ones((Ls[0], 1)).float()
            ), -1) # (1, L_protein, NAATOKENS)
            mask_t[0, :Ls[0], :nprotatoms] = mask_prot

        if init_ligand_tmpl: # input true s.m. xyz as template 1
            xyz_t[1, Ls[0]:, :3] = xyz[0, Ls[0]:, :3]
            f1d_t[1, Ls[0]:] = torch.cat((
                torch.nn.functional.one_hot(msa_seed_orig[0,0, Ls[0]: ]-1, num_classes=NAATOKENS-1).float(),
                torch.ones((Ls[1], 1)).float()
            ), -1) # (1, L_sm, NAATOKENS)
            mask_t[1, Ls[0]:, 1] = mask_sm[0] # all symmetry variants have same mask
    else:
        # standard template featurization
        # same_chain argument prevents sm. mol from being initialized at one end of protein
        ntempl = np.random.randint(params['MINTPLT'], params['MAXTPLT']-1)
        if not fixbb:
            xyz_t, f1d_t, mask_t = TemplFeaturize(tpltA, sum(Ls), params, offset=0,
                npick=ntempl, pick_top=pick_top, same_chain=chain_idx, random_noise=random_noise)
            if msa.shape[1] != xyz_t.shape[1]:
                print(f'WARNING [loader_sm_compl]: MSA and template lengths do not match: {item}. Skipping.')
                return (torch.tensor([-1]),)*21
        if fixbb:
            xyz_t = torch.clone(xyz[0])[None]
            mask_t = torch.clone(mask)
            seq_mask_shifted = torch.clone(seq)
            seq_mask_shifted[seq_mask_shifted>=MASKINDEX] -= 1
            f1d_t = torch.nn.functional.one_hot(seq_mask_shifted, num_classes=NAATOKENS-1)
            conf = torch.ones_like(seq[:1])[...,None]
            f1d_t = torch.cat((f1d_t, conf), dim=-1)


    if init_protein_xyz or init_ligand_xyz:
        # initialize coords to ground truth, move to origin, rotate randomly
        xyz_prev = torch.full((sum(Ls), NTOTAL, 3), np.nan).float()
        mask_prev = torch.full((sum(Ls), NTOTAL), False)
        R = scipy.spatial.transform.Rotation.random(2).as_matrix()
        R = torch.tensor(R).float()
        if init_protein_xyz:
            xyz1 = xyz[0, :Ls[0], :3]
            xyz1 = xyz1 - xyz1[:,1].nanmean(0)
            xyz_prev[:Ls[0], :3] = xyz1 @ R[0].T
            mask_prev[:Ls[0]] = mask[0,:Ls[0]]
        if init_ligand_xyz:
            xyz2 = xyz[0, Ls[0]:, :3]
            xyz2 = xyz2 - xyz2[:,1].nanmean(0)
            xyz_prev[Ls[0]:, :3] = xyz2 @ R[1].T
            mask_prev[Ls[0]:] = mask[0,Ls[0]:]

        # initialize missing positions in ground truth structures
        init = INIT_CRDS.reshape(1,NTOTAL,3).repeat(sum(Ls),1,1)
        init = init + torch.rand(sum(Ls),1,3)*random_noise - random_noise/2
        xyz_prev = torch.where(mask_prev[:,:,None], xyz_prev, init).contiguous()

    else:
        xyz_prev = xyz_t[0].clone()
        if not fixbb:
            xyz_prev = torch.nan_to_num(xyz_prev)
        mask_prev = mask_t[0].clone()

    xyz = torch.nan_to_num(xyz)
    xyz_t = torch.nan_to_num(xyz_t)

    if sum(Ls) > params["CROP"]:
        if len(ligands):
            sel = crop_sm_compl(xyz_prot, xyz_sm[0], Ls, params)
        else:
            sel = get_crop(len(idx), mask[0], msa_seed_orig.device, params["CROP"], unclamp=False)
        seq = seq[:,sel]
        msa_seed_orig = msa_seed_orig[:,:,sel]
        msa_seed = msa_seed[:,:,sel]
        msa_extra = msa_extra[:,:,sel]
        mask_msa = mask_msa[:,:,sel]
        xyz = xyz[:,sel] 
        mask = mask[:,sel]
        xyz_t = xyz_t[:,sel]
        f1d_t = f1d_t[:,sel]
        mask_t = mask_t[:,sel]
        xyz_prev = xyz_prev[sel]
        mask_prev = mask_prev[sel] 
        idx = idx[sel]
        chain_idx = chain_idx[sel][:,sel]
        bond_feats = bond_feats[sel][:, sel]
    # need to reindex the chiral atom positions - assumes they are the second chain
    if chirals.shape[0]>0:
        L1 = chain_idx[0,:].sum()
        chirals[:, :-1] = chirals[:, :-1] +L1
    if fixbb:
        # Remove symmetry from ground-truth since we are currently not predicting ligands
        xyz = xyz[0]
        mask = mask[0]

    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa,\
           xyz.float(), mask, idx.long(), \
           xyz_t.float(), f1d_t.float(), mask_t, \
           xyz_prev.float(), mask_prev, \
           chain_idx, False, False, frames, bond_feats, chirals, "sm_compl", item


def loader_sm_compl_covale(item, params, pick_top=True, 
    init_protein_tmpl=False, init_ligand_tmpl=False,
    init_protein_xyz=False, init_ligand_xyz=False, task='sm_compl_covale', random_noise=5.0):
    """
    dataloader for covalently linked small molecule protein complexes
    """
    pdb_chain, pdb_hash = item['CHAINID'], item['HASH'] 
    ligand = item['LIGAND'] # list of (lig_chain, lig_res_num, lig_name)
    covalent = item["COVALENT"] 
    # Check if any of the covalent bonds are to hyodrgens, these are sent to the loader_sm_compl
    for bond in covalent:
        for atom in bond:
            if atom[3][0] == "H":
                return loader_sm_compl(item, params, pick_top, 
                    init_protein_tmpl=init_protein_tmpl, 
                    init_ligand_tmpl=init_ligand_tmpl, 
                    init_protein_xyz=init_protein_xyz, 
                    init_ligand_xyz=init_ligand_xyz, 
                    task=task, random_noise=random_noise)
    
    pdb_id, i_ch_prot = pdb_chain.split('_')

    ### Load protein
    a3mA = get_msa(params['PDB_DIR'] + '/a3m/'+pdb_hash[:3] + '/'+ pdb_hash + '.a3m.gz', pdb_hash)
    tpltA = torch.load(params['PDB_DIR']+'/torch/hhr/'+pdb_hash[:3]+'/'+pdb_hash+'.pt')
    
    # get msa features
    msa_prot = a3mA['msa'].long()
    ins_prot = a3mA['ins'].long()

    if len(msa_prot) > params['BLOCKCUT']:
        msa_prot, ins_prot = MSABlockDeletion(msa_prot, ins_prot)
    # a3m_prot = {"msa": msa_prot, "ins": ins_prot}
    # xyz_prot, mask_prot = pdbA["xyz"], pdbA["mask"]

    pdb_fn = params['MOL_DIR']+f'/{pdb_id[1:3]}/{pdb_id}.pkl.gz'
    chains, asmb, covale, modres = pickle.loads(gzip.open(pdb_fn.strip(), "rb").read())
    
    # coordinate transforms to recreate this bio-assembly
    i_a = str(item['ASSEMBLY'])
    asmb_xfs = asmb[i_a]

    # 1st protein chain in list is the main binding partner
    for p in item['PARTNERS']:
        if p[0]==i_ch_prot:
            prot_ch2xf = dict([p[:2]])
            break

    # load protein chain
    ch = chains[i_ch_prot]
    ch_xf = asmb_xfs[prot_ch2xf[i_ch_prot]]
    xyz_prot, mask_prot, seq_prot, chid_prot, resi_prot, _ = cif_prot_to_xyz(ch, ch_xf, modres)
    protein_L, nprotatoms, _ = xyz_prot.shape

    # prepare atomized residue and small molecule (first 20 elements of num2aa are amino acids)
    residues_to_atomize = list(itertools.chain.from_iterable([[a[:3] for a in b if a[2] in num2aa[:20]] for b in covalent ])) 
    ligand.extend(residues_to_atomize)

    lig_atoms, lig_bonds = get_ligand_atoms_bonds(ligand, chains, covale)
    lig_ch2xf = dict(item['LIGXF'])
    lig_ch2xf.update(prot_ch2xf) # add protein transform for atomized portion

    xyz_sm, occ_sm, msa_sm, _, akeys = cif_ligand_to_xyz(lig_atoms, asmb_xfs, lig_ch2xf)
    mask_sm = (occ_sm > 0) # fractionally occupied atom positions considered valid
    # remove atoms that are not resolved and do not have any bonds to resolved atoms
    atoms_without_resolved_neighbors = remove_unresolved_substructures(akeys, lig_bonds, mask_sm)
    # update indexing of all lists, tensors to remove unresolved substructures
    akeys = [akey for i, akey in enumerate(akeys) if atoms_without_resolved_neighbors[i].item() != 0]
    xyz_sm = xyz_sm[atoms_without_resolved_neighbors]
    mask_sm = mask_sm[atoms_without_resolved_neighbors]
    msa_sm = msa_sm[atoms_without_resolved_neighbors]

    mol, bond_feats_sm = cif_ligand_to_obmol(xyz_sm, akeys, lig_atoms, lig_bonds)
    xyz_sm, mask_sm = get_automorphs(mol, xyz_sm, mask_sm)

    if xyz_sm.shape[0] > params['MAXNSYMM']: 
        xyz_sm = xyz_sm[:params['MAXNSYMM']]
        mask_sm = mask_sm[:params['MAXNSYMM']]

    if xyz_sm.shape[0] == 0:
        print(f'ERROR [loader_sm_compl]: {item[0]} had no xyz coords')
        return (torch.tensor([-1]),)*21 

    ins_sm = torch.zeros_like(msa_sm)
    a3m_sm = {"msa": msa_sm.unsqueeze(0), "ins": ins_sm.unsqueeze(0)}
    G = get_nxgraph(mol)
    frames = get_atom_frames(msa_sm, G)
    chirals_sm = get_chirals(mol, xyz_sm[0])

    Ls = [xyz_prot.shape[0], xyz_sm.shape[1]]
    bond_feats = torch.zeros((sum(Ls), sum(Ls))).long()
    # remove residues that are going to be atomized from msa
    if residues_to_atomize:
        
        for residue in residues_to_atomize:
            atomize_N = residue + ("N",)
            atomize_C = residue + ("C",)
            N_index = akeys.index(atomize_N) + Ls[0]
            C_index = akeys.index(atomize_C) + Ls[0]
            residue_index = int(residue[1]) - 1 # residues are 1 indexed in the cif files 

            if residue_index != 0: # if first residue in chain, no extra bond feats to previous residue
                bond_feats[residue_index-1, N_index] = 6
                bond_feats[N_index, residue_index-1] = 6
            if residue_index != Ls[0]: #if residue is last in chain, no extra bonds feats to following residue
                bond_feats[residue_index+1, C_index] = 6
                bond_feats[C_index,residue_index+1] = 6
        for residue in residues_to_atomize:
            residue_index = int(residue[1]) - 1 # residues are 1 indexed in the cif files
            msa_prot = torch.cat((msa_prot[:, :residue_index], msa_prot[:, residue_index+1:]), dim=1)
            ins_prot = torch.cat((ins_prot[:, :residue_index], ins_prot[:, residue_index+1:]), dim=1)

            xyz_prot = torch.cat((xyz_prot[:residue_index], xyz_prot[residue_index+1:]), dim=0)
            mask_prot = torch.cat((mask_prot[:residue_index],mask_prot[residue_index+1:]), dim=0)
            bond_feats = torch.cat((bond_feats[ :residue_index], bond_feats[residue_index+1:]), dim=0)
            bond_feats = torch.cat((bond_feats[ :, :residue_index], bond_feats[:, residue_index+1:]), dim=1)
            
    
    a3m_prot = {"msa": msa_prot, "ins": ins_prot}
    protein_L, nprotatoms, _ = xyz_prot.shape

    # Generate ground truth structure: account for ligand symmetry
    N_symmetry, sm_L, _ = xyz_sm.shape
    xyz = torch.full((N_symmetry, protein_L+sm_L, NTOTAL, 3), np.nan).float()
    mask = torch.full(xyz.shape[:-1], False).bool()
    xyz[:, :protein_L, :nprotatoms, :] = xyz_prot.expand(N_symmetry, protein_L, nprotatoms, 3)
    xyz[:, protein_L:, 1, :] = xyz_sm
    mask[:, :protein_L, :nprotatoms] = mask_prot.expand(N_symmetry, protein_L, nprotatoms)
    mask[:, protein_L:, 1] = mask_sm

    Ls = [xyz_prot.shape[0], xyz_sm.shape[1]]
    
    if not ((a3m_prot['msa'].shape[1]==Ls[0]) and (a3m_sm['msa'].shape[1]==Ls[1])):
        print(f'WARNING [loader_sm_compl]: Sm. mol. XYZ and MSA lengths don\'t match: {item}. Skipping.')
        return (torch.tensor([-1]),)*21

    a3m = merge_a3m_hetero(a3m_prot, a3m_sm, Ls)
    msa = a3m['msa'].long()
    ins = a3m['ins'].long()

    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params)
    idx = torch.arange(sum(Ls))
    idx[Ls[0]:] += CHAIN_GAP
    
    node_type = torch.zeros((sum(Ls), sum(Ls))).long() # holds 2D information on whether atom to atom interaction or residue to residue
    node_type[:Ls[0], :Ls[0]] = 1
    node_type[Ls[0]:, Ls[0]:] = 1 
    chain_idx = torch.ones_like(node_type)

    bond_feats[:Ls[0], :Ls[0]] = get_protein_bond_feats(Ls[0])
    bond_feats[Ls[0]:, Ls[0]:] = bond_feats_sm
            
            
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
            xyz_t[0, :Ls[0], :3] = xyz[0, :Ls[0], :3]
            f1d_t[0, :Ls[0]] = torch.cat((
                torch.nn.functional.one_hot(msa_seed_orig[0,0, :Ls[0] ], num_classes=NAATOKENS-1).float(),
                torch.ones((Ls[0], 1)).float()
            ), -1) # (1, L_protein, NAATOKENS)
            mask_t[0, :Ls[0], :nprotatoms] = mask_prot

        if init_ligand_tmpl: # input true s.m. xyz as template 1
            xyz_t[1, Ls[0]:, :3] = xyz[0, Ls[0]:, :3]
            f1d_t[1, Ls[0]:] = torch.cat((
                torch.nn.functional.one_hot(msa_seed_orig[0,0, Ls[0]: ]-1, num_classes=NAATOKENS-1).float(),
                torch.ones((Ls[1], 1)).float()
            ), -1) # (1, L_sm, NAATOKENS)
            mask_t[1, Ls[0]:, 1] = mask_sm[0] # all symmetry variants have same mask
    else:
        # standard template featurization
        # same_chain argument prevents sm. mol from being initialized at one end of protein
        ntempl = np.random.randint(params['MINTPLT'], params['MAXTPLT']-1)
        xyz_t, f1d_t, mask_t = TemplFeaturize(tpltA, sum(Ls), params, offset=0,
            npick=ntempl, pick_top=pick_top, same_chain=chain_idx, random_noise=random_noise) 

        if msa.shape[1] != xyz_t.shape[1]:
            print(f'WARNING [loader_sm_compl]: MSA and template lengths do not match: {item}. Skipping.')
            return (torch.tensor([-1]),)*21
    if init_protein_xyz or init_ligand_xyz:
        # initialize coords to ground truth, move to origin, rotate randomly
        xyz_prev = torch.full((sum(Ls), NTOTAL, 3), np.nan).float()
        mask_prev = torch.full((sum(Ls), NTOTAL), False)
        R = scipy.spatial.transform.Rotation.random(2).as_matrix()
        R = torch.tensor(R).float()
        if init_protein_xyz:
            xyz1 = xyz[0, :Ls[0], :3]
            xyz1 = xyz1 - xyz1[:,1].nanmean(0)
            xyz_prev[:Ls[0], :3] = xyz1 @ R[0].T
            mask_prev[:Ls[0]] = mask[0,:Ls[0]]
        if init_ligand_xyz:
            xyz2 = xyz[0, Ls[0]:, :3]
            xyz2 = xyz2 - xyz2[:,1].nanmean(0)
            xyz_prev[Ls[0]:, :3] = xyz2 @ R[1].T
            mask_prev[Ls[0]:] = mask[0,Ls[0]:]

        # initialize missing positions in ground truth structures
        init = INIT_CRDS.reshape(1,NTOTAL,3).repeat(sum(Ls),1,1)
        init = init + torch.rand(sum(Ls),1,3)*random_noise - random_noise/2
        xyz_prev = torch.where(mask_prev[:,:,None], xyz_prev, init).contiguous()

    else:
        xyz_prev = xyz_t[0].clone()
        xyz_prev = torch.nan_to_num(xyz_prev)
        mask_prev = mask_t[0].clone()

    xyz = torch.nan_to_num(xyz)
    xyz_t = torch.nan_to_num(xyz_t)
      
    if sum(Ls) > params["CROP"]:
        sel = crop_sm_compl(xyz_prot, xyz_sm[0], Ls, params)
        seq = seq[:,sel]
        msa_seed_orig = msa_seed_orig[:,:,sel]
        msa_seed = msa_seed[:,:,sel]
        msa_extra = msa_extra[:,:,sel]
        mask_msa = mask_msa[:,:,sel]
        xyz = xyz[:,sel] 
        mask = mask[:,sel]
        xyz_t = xyz_t[:,sel]
        f1d_t = f1d_t[:,sel]
        mask_t = mask_t[:,sel]
        xyz_prev = xyz_prev[sel]
        mask_prev = mask_prev[sel] 
        idx = idx[sel]
        node_type = node_type[sel][:,sel]
        chain_idx = chain_idx[sel][:,sel]
        bond_feats = bond_feats[sel][:, sel]
    # need to reindex the chiral atom positions - assumes they are the second chain
    L1 = node_type[0,:].sum()
    if chirals_sm.shape[0]>0:
        chirals_sm[:, :-1] = chirals_sm[:, :-1] +L1 
    ch_label = torch.zeros(seq[0].shape)
    ch_label[L1:] = 1
    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa,\
           xyz.float(), mask, idx.long(), \
           xyz_t.float(), f1d_t.float(), mask_t, \
           xyz_prev.float(), mask_prev, \
           chain_idx, False, False, frames, bond_feats, chirals_sm, ch_label, task, item

def find_residues_to_atomize_covale(partners, covale):
    """
    Updates partner lists to have atomized residues when residues are making
    covalent bonds with small molecules.  Also returns list of atomized
    residues so the other features, MSA, templates etc can be removed.

    Parameters
    ----------
    partners : list of 4-tuples (partner, transform_index, num_contacts, partner_type)
    covale : list
        List of cifutils.Bond objects representing inter-chain bonds in this PDB entry.
    """
    if not covale:
        return partners, []
    
    residues_to_atomize = []
    for bond in covale:
        lig_present = []
        prot_present = []
        res_key = None

        # check if the protein and ligand with the covalent linkage is present in the curated item
        for partner in partners:
            #placeholder variable for residue information of the residue that will be atomized
            if partner[-1]=='nonpoly':
                lig_present.append(any([bond.a[:3] == p or bond.b[:3] == p for p in partner[0]])) # matching chain IDs, residue num and residue name
            else:
                #track absolute index of lig and protein in partners list 
                lig_present.append(False)
            
            if  partner[-1]=="polypeptide(L)":
                if bond.a[0] == partner[0]:
                    res_key = bond.a
                    prot_present.append(True)
                elif bond.b[0] == partner[0]:
                    res_key = bond.b
                    prot_present.append(True)
                else: 
                    prot_present.append(False)
            else:
                #track absolute index of lig and protein in partners list 
                prot_present.append(False)

        if any(lig_present) and any(prot_present):
            lig_idx = lig_present.index(True)
            prot_idx = prot_present.index(True)
            
            lig_partner = partners[lig_idx]
            prot_partner = partners[prot_idx]
            lig_partner[0].append(res_key[:3])
            lig_partner[1].append(prot_partner[:2])
            residues_to_atomize.append((res_key[:3], (prot_partner[:2]))) # residue key, transform

    return partners, residues_to_atomize


def featurize_asmb_prot(pdb_id, partners, params, chains, asmb_xfs, modres, chid2hash=None, 
    pick_top=True, random_noise=5.0):
    """Loads multiple protein chains from parsed CIF assembly into tensors.
    Outputs will contain chains roughly in the order that they appear in
    `partners` (decreasing number of contacts to query ligand), except that
    chains with different letters but the same sequence (homo-oligomers) are
    placed contiguously in the residue dimension. All homo-oligomer chain swaps
    are enumerated and stored in the leading dimension ("permutation
    dimension"). Chain swap permutations of different sets of homo-oligomers
    are combined by a cartesian product (e.g. a complex with 2 copies of chain
    A and 3 copies of chain B, where A and B have distinct sequences, will have
    2 (# chain swaps of A) * 6 (# chain swaps of B) = 12 total chain-swap
    permutations.

    Parameters
    ----------
    pdb_id : string
        PDB accession of example. Used to load the pre-parsed CIF data.
    partners : list of 4-tuples (partner, transform_index, num_contacts, partner_type)
        Protein chains to featurize. All elements should have `partner_type =
        'polypeptide(L)'`. `partner` contains the chain letter.
        `transform_index` is an integer index of the coordinate transform for
        each partner chain.
    params : dict
        Parameters for the data loader
    chains : dict
        Dictionary mapping chain letters to cifutils.Chain objects representing
        the chains in a PDB entry.
    asmb_xfs : list of 2-tuples (chain_id, torch.Tensor(4,4))
        Coordinate transforms for the current assembly
    modres : dict
        Maps modified residue names to their canonical equivalents. Any
        modified residue will be converted to its standard equivalent and
        coordinates for atoms with matching names will be saved.
    chid2hash : dict
        Maps chainid to msa hash which is used to get templates for the 
        Maps chain ids (<pdbid>_<chain_letter>) to hash strings used to name homology
        template and MSA files. If None, no templates are loaded.

    Returns
    -------
    xyz_prot : tensor (N_chain_permutation, L_total, N_atoms, 3)
        Atom coordinates of all the protein chains
    mask_prot : tensor (N_chain_permutation, L_total, N_atoms)
        Boolean mask for whether an atom exists in `xyz_prot`
    seq_prot : tensor (L_total,)
        Integer-coded sequence of the protein chains
    ch_label_prot : tensor (L_total,)
        Integer-coded chain identity for each residue. Differs from chain letter
        in that different-lettered chains with the same sequence will have the
        same integer code
    xyz_t_prot : tensor (N_templates, L_total, N_atoms, 3)
        Atom coordinates of the templates
    f1d_t_prot : tensor (N_templates, L_total, N_t1d_features)
        1D template features
    mask_t_prot : tensor (N_templates, L_total, N_atoms)
        Boolean mask for whether template atoms exist
    Ls_prot : list
        Length of each protein chain
    mod_residues_to_atomize : list
        List of tuples `((chain_letter, residue_num, residue_name),
        (chain_letter, xform_index))` representing chemically modified residues
        that should be atomized.
    """
    # assign number to each unique sequence, irrespective of chain letter
    chnum2chlet, chlet2chnum = map_identical_prot_chains(partners, chains, modres)
    # protein true coords
    xyz_prot, mask_prot, Ls_prot, ch_label_prot, seq_prot = [], [], [], [], []
    xyz_t_prot, f1d_t_prot, mask_t_prot = [], [], []
    for chnum, chlet_set in chnum2chlet.items():
        ## protein coordinates
        # every location of this chain
        partners = [p for p in partners if (p[-1]=='polypeptide(L)') and (p[0] in chlet_set)]
        N_mer = len(partners)
        xyz_chxf, mask_chxf, seq_chxf, mod_residues_to_atomize = [], [], [], []
        for p in partners:
            xyz_, mask_, seq_, _, _, residues_to_atomize = cif_prot_to_xyz(chains[p[0]], asmb_xfs[p[1]], modres)
            residues_to_atomize = [(residue, (residue[0], p[1])) for residue in residues_to_atomize]
            xyz_chxf.append(xyz_) # (L, N_atoms, 3)
            mask_chxf.append(mask_)
            seq_chxf.append(seq_)
            mod_residues_to_atomize.extend(residues_to_atomize)
            Ls_prot.append(xyz_.shape[0])

        # concatenate all locations, repeat for every permutation of locations
        xyz_ch, mask_ch, seq_ch = [], [], []
        for idx in permutations(range(len(xyz_chxf))):
            xyz_ch.append(torch.cat([xyz_chxf[i] for i in idx], dim=0))
            mask_ch.append(torch.cat([mask_chxf[i] for i in idx], dim=0))
        xyz_ch = torch.stack(xyz_ch, dim=0) # (perm(N_mer), L*N_mer, N_atoms, 3)
        mask_ch = torch.stack(mask_ch, dim=0) # (perm(N_mer), L*N_mer, N_atoms)

        seq_ch = torch.cat(seq_chxf, dim=0)

        # save results for each chain
        xyz_prot.append(xyz_ch)
        mask_prot.append(mask_ch)
        seq_prot.append(seq_ch)
        ch_label_prot.append(torch.full((xyz_ch.shape[1],),chnum))
        chnum += 1

        ## protein templates
        ntempl = np.random.randint(params['MINTPLT'], params['MAXTPLT']+1)
        if chid2hash is None or ntempl < 1:
            xyz_t_ch, f1d_t_ch, mask_t_ch = \
                blank_template(n_tmpl=1, L=xyz_ch.shape[1], random_noise=random_noise)
        else:
            pdb_hash = chid2hash[pdb_id+'_'+list(chlet_set)[0]] # chlet_set all have same hash
            tplt = torch.load(params['PDB_DIR']+'/torch/hhr/'+pdb_hash[:3]+'/'+pdb_hash+'.pt')
            xyz_t_, f1d_t_, mask_t_ = TemplFeaturize(tplt, Ls_prot[-1], params, npick=ntempl, 
                offset=0, pick_top=pick_top, random_noise=random_noise)
            xyz_t_ch = torch.cat([xyz_t_]+[random_rot_trans(xyz_t_) for i in range(N_mer-1)], dim=1) # (ntempl, L*N_mer, natm, 3)
            f1d_t_ch = torch.cat([f1d_t_]*N_mer, dim=1) # (ntempl, L*N_mer, 21)
            mask_t_ch = torch.cat([mask_t_]*N_mer, dim=1) # (ntempl, L*N_mer, natm)

        xyz_t_prot.append(xyz_t_ch)
        f1d_t_prot.append(f1d_t_ch)
        mask_t_prot.append(mask_t_ch)

    # cartesian product over each chain's location permutations
    xyz_prot = cartprodcat(xyz_prot) # (prod_i(N_perm_i), sum_i(L_i*N_mer_i), N_atoms, 3)
    mask_prot = cartprodcat(mask_prot) # (prod_i(N_perm_i), sum_i(L_i*N_mer_i), N_atoms)

    xyz_t_prot, f1d_t_prot, mask_t_prot = \
        merge_hetero_templates(xyz_t_prot, f1d_t_prot, mask_t_prot, Ls_prot)

    ch_label_prot = torch.cat(ch_label_prot, dim=0)
    seq_prot = torch.cat(seq_prot, dim=0)

    return xyz_prot, mask_prot.bool(), seq_prot, ch_label_prot, xyz_t_prot, f1d_t_prot, \
           mask_t_prot, Ls_prot, mod_residues_to_atomize

def featurize_single_ligand(ligand, chains, covale, lig_xf_s, asmb_xfs, offset, params):
    """Loads a single ligand in a specific assembly from a parsed CIF file into
    tensors. If more than one coordinate transform exists for the ligand, the
    additional copies of the molecule are concatenated into the symmetry
    permutation dimension if atom occupancy is fractional (between 0 and 1) and
    appended to a list (for later concatenation along the residue dimension) if
    atom occupancy is equal to 1.0.

    Parameters
    ----------
    ligand : list of tuples (chain_letter, res_num, res_name)
    chains : dict
        Dictionary mapping chain letters to cifutils.Chain objects representing 
        the chains in a PDB entry.
    covale : list
        List of cifutils.Bond objects representing inter-chain bonds in this
        PDB entry.
    lig_xf_s : list of tuples (chain_letter, transform_index)
    asmb_xfs : list of tuples (chain_letter, np.array(4,4))
        Coordinate transforms for this assembly
    offset : int
        Offset on residue dimension for updating chiral features indexes
    params : dict
        Data loader parameters. 

    Returns
    -------
    All outputs will be a list of length `N_lig` (number of copies of this
    ligand in the assembly):

    xyz_lig : list of torch.Tensor (N_permutation, L, 3), float
        Atom coordinates of this ligand. This list will have length > 1 if
        multiple coordinate transforms exist and atom occupancy is 1. If
        multiple transforms exist and atom occupancy is between 0 and 1, this
        will be a list with one coordinate tensor containing multiple sets of
        coordinates in the symmetry dimension.
    mask_lig : list of torch.Tensor (N_permutation, L), bool
        Mask that is true if a certain atom exists.
    msa_lig : list of torch.Tensor (L_total,)
    bond_feats_lig : list of torch.Tensor (L, L)
    akeys_lig : list of tuples (chain_id, residue_num, residue_name, atom_name)
    Ls_lig : list of int
    frames_lig : list of torch.Tensor (L, 3, 2)
    chirals_lig : list of torch.Tensor (N_chirals, 5)
    resnames : list
        Residue names of each ligand in the returned lists
    """
    lig_atoms, lig_bonds = get_ligand_atoms_bonds(ligand, chains, covale)
    
    xyz_lig, mask_lig, msa_lig, bond_feats_lig, akeys_lig, Ls_lig = [], [], [], [], [], []
    frames_lig, chirals_lig, resname_lig = [], [], [] 

    for lig_xf in lig_xf_s: # all possible locations for this ligand
        ch2xf = dict(lig_xf)
        xyz_, occ_, msa_, chid_, akeys_ = cif_ligand_to_xyz(lig_atoms, asmb_xfs, ch2xf)
        if (occ_==0).all(): continue # no valid atom positions
        mask_ = (occ_ > 0) # partially occupied atoms are considered valid
        mol_, bond_feats_ = cif_ligand_to_obmol(xyz_, akeys_, lig_atoms, lig_bonds)
        xyz_, mask_ = get_automorphs(mol_, xyz_, mask_)

        # clamp number of atom permutations to save GPU memory
        if xyz_.shape[0] > params['MAXNSYMM']:
            print('WARNING: Too many atom permutations ({xyz_.shape[0]}) in {item["CHAINID"]} {ligand}. '\
                  'Keeping only {params["MAXSYMM"]}.')
            xyz_ = xyz_[:params['MAXNSYMM']]
            mask_ = mask_[:params['MAXNSYMM']]

        G = get_nxgraph(mol_)
        frames_ = get_atom_frames(msa_, G)
        chirals_ = get_chirals(mol_, xyz_[0])
        if chirals_.shape[0]>0:
            chirals_[:,:-1] = chirals_[:,:-1] + offset

        if ((occ_<1) & (occ_>0)).any(): 
            # partial occupancy, add to permutation dimension
            if not ((occ_<1) & (occ_>0)).all():
                print('WARNING: Partial occupancy for a subset of atoms in ligand', ligand)
                print('         Adding to permutation dimension as alternate coordinates.')

            if len(xyz_lig) == 0:
                xyz_lig = [xyz_]
                mask_lig = [mask_]
                msa_lig = [msa_]
                bond_feats_lig = [bond_feats_]
                akeys_lig = [akeys_]
                Ls_lig = [xyz_.shape[1]]
                frames_lig = [frames_]
                chirals_lig = [chirals_]
                resname_lig = ['_'.join([res[2] for res in ligand])]
            else:
                xyz_lig[0] = torch.cat([xyz_lig[0], xyz_],dim=0)
                mask_lig[0] = torch.cat([mask_lig[0], mask_],dim=0)
        else: 
            # full occupancy, add as new chain
            xyz_lig.append(xyz_)
            mask_lig.append(mask_)
            msa_lig.append(msa_)
            bond_feats_lig.append(bond_feats_)
            akeys_lig.append(akeys_)
            Ls_lig.append(xyz_.shape[1])
            frames_lig.append(frames_)
            chirals_lig.append(chirals_)
            resname_lig.append('_'.join([res[2] for res in ligand]))
            offset += xyz_.shape[1]

    return xyz_lig, mask_lig, msa_lig, bond_feats_lig, akeys_lig, Ls_lig, frames_lig, \
           chirals_lig, resname_lig

def featurize_asmb_ligands(partners, params, chains, asmb_xfs, covale):
    """Loads multiple ligands chains from parsed CIF assembly into tensors.
    Outputs will contain ligands in the order that they appear in
    `partners` (decreasing number of contacts to query ligand). Leading
    dimension of output coordinates contains atom position permutations for
    each ligand.  Atom permutations between different ligands that are
    identical are not generated here, so loss must be calculated in a way that
    accounts for inter-ligand swap permutations.

    Parameters
    ----------
    partners : list of 4-tuples (partner, transform_index, num_contacts, partner_type)
        Ligands to featurize. All elements should have `partner_type =
        'nonpoly'` and `partner` is a list of tuples (chain_letter, res_num,
        res_name) corresponding to the residues that make up this ligand.
        `transform_index` will be a list of tuples (chain_letter, idx)
        indicating the index of the coordinate transform for each chain in the
        ligand.
    params : dict
        Parameters for the data loader
    chains : dict
        Dictionary mapping chain letters to cifutils.Chain objects representing
        the chains in a PDB entry.
    asmb_xfs : list of 2-tuples (chain_id, torch.Tensor(4,4))
        Coordinate transforms for the current assembly
    covale : dict
        List of cifutils.Bond objects representing inter-chain bonds in this
        PDB entry.

    Returns
    -------
    xyz_sm : tensor (N_atom_permutation, L_total, N_atoms, 3)
        Atom coordinates of all the ligands.
    mask_sm : tensor (N_atom_permutation, L_total, N_atoms)
        Boolean mask for whether an atom exists in `xyz_sm`.
    msa_sm : tensor (L_total,)
        Integer-coded "sequence" (atom types) of the ligands
    bond_feats_sm : list of tensors (L_chain, L_chain)
        List of bond feature matrices for each ligand chain
    frames : (L_total, 3, 2)
        Frame atom relative indices for each ligand atom
    chirals : (N_chiral_atoms, 5)
        Chiral features (4 residue indices and 1 ideal dihedral angle) for each
        chiral center
    sm_Ls : list
        Length of each ligand
    ch_label_sm : tensor (L_total,)
        Integer-coded chain identity for each ligand. Ligands are assigned the
        same code if their representation as an ordered list of tuples
        (residue_name, atom_name) is the same.
    akeys_sm : list 
        list of tuples (chid, resnum, resname, atomtype). Used downstream to map atom identities to index in xyz tensors
    lig_names : list
        Name of each ligand (residue name(s) joined by '_')
    """
    # group ligands with identical chain & residue numbers
    # these may represent alternate locations of the same molecule
    # and need to be loaded into permutation dimension
    ligands = []
    lig2xf = OrderedDict()
    for p in partners:
        if p[-1] != 'nonpoly': continue
        ligands.append(p[0])
        k = str(p[0]) # make multires ligand into string for using as dict key
        if k not in lig2xf:
            lig2xf[k] = []
        lig2xf[k].append(p[1])

    # load all ligands
    xyz_sm, mask_sm, = [], []
    msa_sm, bond_feats_sm, akeys_sm, Ls_sm, frames, chirals, resnames = [], [], [], [], [], [], []
    for ligkey, lig_xf_s in lig2xf.items():

        ligand = ast.literal_eval(ligkey)

        offset = sum(Ls_sm) # residue numbering offset for chirals
        xyz_lig, mask_lig, msa_lig, bond_feats_lig, akeys_lig, Ls_lig, frames_lig, \
        chirals_lig, resname_lig = \
            featurize_single_ligand(ligand, chains, covale, lig_xf_s, asmb_xfs, offset, params)

        xyz_sm.extend(xyz_lig)
        mask_sm.extend(mask_lig)
        msa_sm.extend(msa_lig)
        bond_feats_sm.extend(bond_feats_lig)
        akeys_sm.extend(akeys_lig)
        Ls_sm.extend(Ls_lig)
        frames.extend(frames_lig)
        chirals.extend(chirals_lig)
        resnames.extend(resname_lig)

    # concatenate features
    msa_sm = torch.cat(msa_sm, dim=0)
    frames = torch.cat(frames, dim=0)
    chirals = torch.cat(chirals, dim=0)

    # concatenate coordinates with enough room for the largest symmetry permutation dimension
    N_symm = max([xyz_.shape[0] for xyz_ in xyz_sm])
    xyz_out = torch.full((N_symm,sum(Ls_sm),3), np.nan)
    mask_out = torch.full((N_symm,sum(Ls_sm)), False)
    i_res = 0
    for xyz_, mask_ in zip(xyz_sm, mask_sm):
        N_symm_, L_ = xyz_.shape[:2]
        xyz_out[:N_symm_, i_res:i_res+L_] = xyz_
        mask_out[:N_symm_, i_res:i_res+L_] = mask_
        i_res += L_
    xyz_sm, mask_sm = xyz_out, mask_out

    # detect which ligands are the same
    # ligands are considered identical if they have identical string representations
    # of an ordered list of (lig name, atom name) tuples
    lig_dict = dict()
    for i in range(len(akeys_sm)):
        ak = str(sorted([x[2:] for x in akeys_sm[i]])) # [(lig_name, atom_name), ...]
        if ak not in lig_dict:
            lig_dict[ak] = []
        lig_dict[ak].append(i)

    keymap = dict(zip(lig_dict.keys(),range(len(lig_dict))))
    idx2label = dict()
    for k,v in lig_dict.items():
        for idx in v:
            idx2label[idx] = keymap[k]

    ch_label_sm = [torch.full((L_,), idx2label[i]) for i,L_ in enumerate(Ls_sm)]
    ch_label_sm = torch.cat(ch_label_sm, dim=0)

    return xyz_sm, mask_sm, msa_sm[None], bond_feats_sm, frames, chirals, Ls_sm, \
           ch_label_sm, akeys_sm, resnames


def loader_sm_compl_assembly(item, params, chid2hash=None, chid2taxid=None, task='sm_compl_asmb', 
    num_protein_chains=None, num_ligand_chains=None, pick_top=True, fixbb=False, random_noise=5.0):
    """Load protein/ligand assembly from pre-parsed CIF files. Outputs can
    represent multiple chains, which are ordered from most to least contacts
    with query ligand.  Protein chains all come before ligand chains, and
    protein chains with identical sequences are grouped contiguously.

    `all_partners` is a list of 5-tuples representing ligands and protein
    chains near the query ligand that should be featurized as part of the
    assembly. The 5-tuple has the form

        (partner, xforms, num_contacts, min_dist, partner_type)

    `num_contacts` is the number of heavy atoms within 5A of the query ligand.
    `min_dist` is the minimum distance in angstroms between a heavy atom and
    the ligand. If `partner_type` is "polypeptide", then `partner` is the chain
    letter and `xforms` is an integer index of a coordinate transform in
    `asmb_xfs`.  If `partner_type` is "nonpoly", then `partner` is a list of
    tuples `(chain_letter, res_num, res_name)` representing a ligand and
    `xforms` is a list of tuples `(chain_letter, xform_index)` representing
    transforms.  
    """
    pdb_chain = item['CHAINID'] 
    pdb_id = pdb_chain.split('_')[0]

    # load pre-parsed cif assembly - requires cifutils.py in path for object definitions
    chains, asmb, covale, modres = \
        pickle.load(gzip.open(params['MOL_DIR']+f'/{pdb_id[1:3]}/{pdb_id}.pkl.gz'))

    # list of proteins and ligands to featurize
    all_partners = [(item['LIGAND'], item['LIGXF'], -1, 'nonpoly')] + item["PARTNERS"]

    # update partners to atomize residues that are covalently linked to proteins
    all_partners, residues_to_atomize = find_residues_to_atomize_covale(all_partners, covale) 

    # get list of coordinate transforms to recreate this bio-assembly
    i_a = str(item['ASSEMBLY'])
    asmb_xfs = asmb[i_a]

    # load protein chains
    prot_partners = [p for p in all_partners if p[-1]=='polypeptide(L)']
    if num_protein_chains is not None:
        prot_partners = prot_partners[:num_protein_chains]

    xyz_prot, mask_prot, seq_prot, ch_label_prot, xyz_t_prot, f1d_t_prot, \
    mask_t_prot, Ls_prot, mod_residues_to_atomize = \
        featurize_asmb_prot(pdb_id, prot_partners, params, chains, asmb_xfs, modres, chid2hash,
                            pick_top=pick_top, random_noise=random_noise)

    # update partners and residues_to_atomize with modified residues to be atomized
    all_partners.extend([([res_tuple], [ch_xf], -1, "nonpoly",) # multi-res ligand format
                         for (res_tuple, ch_xf) in mod_residues_to_atomize])
    residues_to_atomize.extend(mod_residues_to_atomize)

    # load ligands
    lig_partners = [p for p in all_partners if p[-1]=='nonpoly']
    if num_ligand_chains is not None:
        lig_partners = lig_partners[:num_ligand_chains]
    xyz_sm, mask_sm, msa_sm, bond_feats_sm, frames, chirals, Ls_sm, ch_label_sm, akeys_sm, lig_names = \
        featurize_asmb_ligands(lig_partners, params, chains, asmb_xfs, covale)

    # combine protein & ligand coordinates
    N_symm_prot = xyz_prot.shape[0]
    N_symm_sm = xyz_sm.shape[0]
    L_total = sum(Ls_prot)+sum(Ls_sm)

    xyz = torch.full((max(N_symm_prot, N_symm_sm), L_total, NTOTAL, 3), np.nan).float()
    xyz[:N_symm_prot, :sum(Ls_prot)] = xyz_prot
    xyz[:N_symm_sm, sum(Ls_prot):, 1, :] = xyz_sm

    mask = torch.full((max(N_symm_prot, N_symm_sm), L_total, NTOTAL), False).bool()
    mask[:N_symm_prot, :sum(Ls_prot)] = mask_prot
    mask[:N_symm_sm, sum(Ls_prot):, 1] = mask_sm

    # combine protein & ligand templates
    N_tmpl = xyz_t_prot.shape[0]
    xyz_t_sm, f1d_t_sm, mask_t_sm = blank_template(N_tmpl, sum(Ls_sm), random_noise)
    xyz_t = torch.cat([xyz_t_prot, xyz_t_sm],dim=1)
    f1d_t = torch.cat([f1d_t_prot, f1d_t_sm],dim=1)
    mask_t = torch.cat([mask_t_prot, mask_t_sm],dim=1)

    # bond features
    bond_feats_prot = [get_protein_bond_feats(L) for L in Ls_prot]
    bond_feats = torch.zeros((L_total, L_total)).long()
    offset = 0
    for bf in bond_feats_prot+bond_feats_sm:
        L = bf.shape[0]
        bond_feats[offset:offset+L, offset:offset+L] = bf
        offset += L

    # other features
    idx = idx_from_Ls(Ls_prot+Ls_sm)
    same_chain = same_chain_2d_from_Ls(Ls_prot+Ls_sm)
    ch_label = torch.cat([ch_label_prot, ch_label_sm+ch_label_prot.max()+1])

    # load msa
    if chid2hash is not None: 
        prot_chains = [
            chlet for (chlet, xform, num_contacts, min_dist, partner_type) in prot_partners 
            if partner_type =="polypeptide(L)"
        ]
        protein_chain_info = [{
            "chid": f"{pdb_id}_{chid}", 
            "hash": chid2hash[f"{pdb_id}_{chid}"], 
            "len": Ls_prot[i],
            "query_taxid": chid2taxid[f"{pdb_id}_{chid}"]
        } for i, chid in enumerate(prot_chains)]
        a3m_prot = get_assembly_msa(protein_chain_info, params)
        a3m_sm = dict(msa=msa_sm, ins=torch.zeros_like(msa_sm))
        a3m = merge_a3m_hetero(a3m_prot, a3m_sm, [sum(Ls_prot), sum(Ls_sm)])
        msa, ins = a3m['msa'].long(), a3m['ins'].long()
    else:
        # no msa hash provided, return query sequence as msa
        msa = torch.cat([seq_prot[None], msa_sm],dim=1)
        ins = torch.zeros_like(msa)
    assert msa.shape[1] == xyz.shape[1], "msa shape and xyz shape don't match"
    
    if residues_to_atomize:
        msa, ins, xyz, mask, bond_feats, idx, xyz_t, f1d_t, mask_t, same_chain, ch_label, \
        Ls_prot, Ls_sm \
            = reindex_protein_feats_after_atomize(
                residues_to_atomize,
                prot_partners,
                msa, 
                ins,
                xyz,
                mask,
                bond_feats,
                idx,
                xyz_t,
                f1d_t,
                mask_t,
                same_chain,
                ch_label,
                Ls_prot,
                Ls_sm,
                akeys_sm
            )
        
    xyz_prev = xyz_t[0].clone()
    mask_prev = mask_t[0].clone()

    xyz_prev = torch.nan_to_num(xyz_prev)
    xyz = torch.nan_to_num(xyz)
    xyz_t = torch.nan_to_num(xyz_t)

    # keep track of protein positions for reindexing chirals after crop
    L_total = sum(Ls_prot)+sum(Ls_sm)
    is_prot = torch.zeros(L_total) 
    is_prot[:sum(Ls_prot)] = 1

    # crop around query ligand (1st sm chain)
    if L_total > params["CROP"]:
        sel = crop_sm_compl_assembly(xyz[0], mask[0], Ls_prot, Ls_sm, params['CROP'])
        msa = msa[:, sel]
        ins = ins[:, sel]
        xyz = xyz[:,sel]
        mask = mask[:,sel]
        xyz_t = xyz_t[:,sel]
        f1d_t = f1d_t[:,sel]
        mask_t = mask_t[:,sel]
        xyz_prev = xyz_prev[sel]
        mask_prev = mask_prev[sel]
        idx = idx[sel]
        same_chain = same_chain[sel][:,sel]
        bond_feats = bond_feats[sel][:,sel]
        ch_label = ch_label[sel]
        is_prot = is_prot[sel]

        # crop small molecule features, assumes all sm chains are after all protein chains
        atom_sel = sel[sel>=sum(Ls_prot)] - sum(Ls_prot) # 0 index all the selected atoms
        frames = frames[atom_sel]
        chirals = crop_chirals(chirals, atom_sel)

    # reindex chiral atom positions - assumes all sm chains are after all protein chains
    if chirals.shape[0]>0:
        L1 = is_prot.sum()
        chirals[:, :-1] = chirals[:, :-1] + L1

    # create MSA features from cropped msa and insertions
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params, fixbb=fixbb)
    
    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa,\
           xyz.float(), mask, idx.long(), \
           xyz_t.float(), f1d_t.float(), mask_t, \
           xyz_prev.float(), mask_prev, \
           same_chain, False, False, frames, bond_feats, chirals, ch_label, task, item


def loader_atomize_pdb(item, params, homo, n_res_atomize, flank, unclamp=False, 
    pick_top=True, p_homo_cut=0.5, random_noise=5.0):
    """ load pdb with portions represented as atoms instead of residues """
    pdb_chain, pdb_hash = item['CHAINID'], item['HASH']
    pdb = torch.load(params['PDB_DIR']+'/torch/pdb/'+pdb_chain[1:3]+'/'+pdb_chain+'.pt')
    a3m = get_msa(params['PDB_DIR'] + '/a3m/' + pdb_hash[:3] + '/' + pdb_hash + '.a3m.gz', pdb_hash)
    # get msa features
    msa = a3m['msa'].long()
    ins = a3m['ins'].long()
    if len(msa) > params['BLOCKCUT']:
        msa, ins = MSABlockDeletion(msa, ins)
    
    idx = torch.arange(len(pdb['xyz'])) 
    xyz = torch.full((len(idx),NTOTAL,3), np.nan).float()
    xyz[:,:14,:] = pdb['xyz']
    mask = torch.full((len(idx), NTOTAL), False)
    mask[:,:14] = pdb['mask']
    
    # handle template features
    ntempl = np.random.randint(params['MINTPLT'], params['MAXTPLT']-1)
    xyz_t_prot, f1d_t_prot, mask_t_prot = TemplFeaturize(tplt, len(pdb['xyz']), params, offset=0, 
        npick=ntempl, pick_top=pick_top, random_noise=random_noise)

    crop_idx = get_crop(len(idx), mask, msa.device, params['CROP'], unclamp=unclamp)
    msa_prot = msa[:, crop_idx]
    ins_prot = ins[:, crop_idx]
    xyz_prot = xyz[crop_idx]
    mask_prot = mask[crop_idx]
    idx = idx[crop_idx]
    xyz_t_prot = xyz_t_prot[:, crop_idx]
    f1d_t_prot = f1d_t_prot[:, crop_idx]
    mask_t_prot = mask_t_prot[:, crop_idx]
    protein_L, nprotatoms, _ = xyz_prot.shape

    # choose region to atomize
    can_atomize_mask = torch.ones((protein_L,))

    idx_missing_N = torch.where(~mask_prot[1:,0])[0]+1 # residues missing bb N, excluding 1st residue
    idx_missing_C = torch.where(~mask_prot[:-1,2])[0] # residues missing bb C, excluding last residue
    can_atomize_mask[idx_missing_N-1] = 0 # can't atomize residues before a missing N
    can_atomize_mask[idx_missing_C+1] = 0 # can't atomize residues after a missing C

    num_atoms_per_res = allatom_mask[msa_prot[0],:14].sum(dim=-1) # how many atoms should each residue have?
    num_atoms_exist = mask_prot.sum(dim=-1) # how many atoms have coords in each residue?
    can_atomize_mask[(num_atoms_per_res != num_atoms_exist)] = 0
    can_atomize_idx = torch.where(can_atomize_mask)[0]

    # not enough valid residues to atomize and have space for flanks, treat as monomer example
    if flank + 1 >= can_atomize_idx.shape[0]-(n_res_atomize+flank+1):
        return featurize_single_chain(msa, ins, tplt, pdb, params, random_noise=random_noise) \
            + ("atomize_pdb", item,)

    i_start = torch.randint(flank+1, can_atomize_idx.shape[0]-(n_res_atomize+flank+1),(1,))
    i_start = can_atomize_idx[i_start] # index of the first residue to be atomized

    for i_end in range(i_start+1, i_start + n_res_atomize):
        if i_end not in can_atomize_idx:
            n_res_atomize = int(i_end-i_start)
            print(f'WARNING: n_res_atomize set to {n_res_atomize} due to not enough consecutive '\
                  f'fully-resolved residues to atomize. {item} i_start={i_start}')
            break

    msa_sm, ins_sm, xyz_sm, mask_sm, frames, bond_feats_sm, last_C, chirals = atomize_protein(i_start, msa_prot, xyz_prot, mask_prot, n_res_atomize=n_res_atomize)
        
    # generate blank template for atoms
    tplt_sm = {"ids":[]}
    xyz_t_sm, f1d_t_sm, mask_t_sm = TemplFeaturize(tplt_sm, xyz_sm.shape[1], params, offset=0, npick=0, pick_top=pick_top)
    ntempl = xyz_t_prot.shape[0]
    xyz_t = torch.cat((xyz_t_prot, xyz_t_sm.repeat(ntempl,1,1,1)), dim=1)
    f1d_t = torch.cat((f1d_t_prot, f1d_t_sm.repeat(ntempl,1,1)), dim=1)
    mask_t = torch.cat((mask_t_prot, mask_t_sm.repeat(ntempl,1,1)), dim=1)

    # Generate ground truth structure: account for ligand symmetry
    N_symmetry, sm_L, _ = xyz_sm.shape
    xyz = torch.full((N_symmetry, protein_L+sm_L, NTOTAL, 3), np.nan).float()
    mask = torch.full(xyz.shape[:-1], False).bool()
    xyz[:, :protein_L, :nprotatoms, :] = xyz_prot.expand(N_symmetry, protein_L, nprotatoms, 3)
    xyz[:, protein_L:, 1, :] = xyz_sm
    mask[:, :protein_L, :nprotatoms] = mask_prot.expand(N_symmetry, protein_L, nprotatoms)
    mask[:, protein_L:, 1] = mask_sm
    
    Ls = [xyz_prot.shape[0], xyz_sm.shape[1]]
    a3m_prot = {"msa": msa_prot, "ins": ins_prot}
    a3m_sm = {"msa": msa_sm.unsqueeze(0), "ins": ins_sm.unsqueeze(0)}
    a3m = merge_a3m_hetero(a3m_prot, a3m_sm, Ls)
    msa = a3m['msa'].long()
    ins = a3m['ins'].long()

    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params)

    # handle bond features
    bond_feats = torch.zeros((sum(Ls), sum(Ls))).long()
    bond_feats[:Ls[0], :Ls[0]] = get_protein_bond_feats(Ls[0])
    bond_feats[Ls[0]:, Ls[0]:] = bond_feats_sm
    bond_feats[i_start-1, Ls[0]] = 6
    bond_feats[Ls[0], i_start-1] = 6
    if len(last_C.numpy())==1:
        bond_feats[i_start+n_res_atomize+flank, Ls[0]+int(last_C.numpy())] = 6
        bond_feats[Ls[0]+int(last_C.numpy()), i_start+n_res_atomize+flank] = 6
    else:
        print(f"ERROR: {item} has multiple values for last_C, {last_C.numpy()} with i_start= {i_start}")

    # handle res_idx
    last_res = idx[-1]
    idx_sm = torch.arange(Ls[1]) + last_res
    idx = torch.cat((idx, idx_sm))

    # handle chain_idx
    chain_idx = torch.ones((sum(Ls), sum(Ls))).long()
    node_type = torch.zeros((sum(Ls), sum(Ls))).long() # holds 2D information on whether atom to atom interaction or residue to residue
    node_type[:Ls[0], :Ls[0]] = 1
    node_type[Ls[0]:, Ls[0]:] = 1 
    
    # remove msa features for atomized portion
    i1 = i_start - flank
    i2 = i_start + n_res_atomize + flank
    seq = torch.cat((seq[:, :i1], seq[:, i2:]), dim=1)
    msa_seed_orig = torch.cat((msa_seed_orig[:, :, :i1], msa_seed_orig[:, :, i2:]), dim=2)
    msa_seed = torch.cat((msa_seed[:, :, :i1], msa_seed[:, :, i2:]), dim=2)
    msa_extra = torch.cat((msa_extra[:, :, :i1], msa_extra[:, :, i2:]), dim=2)
    mask_msa = torch.cat((mask_msa[:, :, :i1], mask_msa[:, :, i2:]), dim=2)
    xyz = torch.cat((xyz[:, :i1], xyz[:, i2:]), dim=1)
    mask = torch.cat((mask[:, :i1], mask[:, i2:]), dim=1)

    idx = torch.cat((idx[:i1], idx[i2:]), dim=0)
    xyz_t = torch.cat((xyz_t[:, :i1], xyz_t[:, i2:]), dim=1)
    f1d_t = torch.cat((f1d_t[:, :i1], f1d_t[:, i2:]), dim=1)
    mask_t = torch.cat((mask_t[:, :i1], mask_t[:, i2:]), dim=1)
    chain_idx = torch.cat((chain_idx[ :i1], chain_idx[i2:]), dim=0)
    chain_idx = torch.cat((chain_idx[ :, :i1], chain_idx[:, i2:]), dim=1)
    node_type = torch.cat((node_type[ :i1], node_type[i2:]), dim=0)
    node_type = torch.cat((node_type[ :, :i1], node_type[:, i2:]), dim=1)
    bond_feats = torch.cat((bond_feats[ :i1], bond_feats[i2:]), dim=0)
    bond_feats = torch.cat((bond_feats[ :, :i1], bond_feats[:, i2:]), dim=1)

    xyz_prev = xyz_t[0].clone()
    xyz_prev[Ls[0]:] = xyz_prev[i_start]
    mask_prev = mask_t[0].clone()
    xyz = torch.nan_to_num(xyz)

    if chirals.shape[0]>0:
        L1 = node_type[0,:].sum()
        chirals[:, :-1] = chirals[:, :-1] +L1
    ch_label = torch.zeros(seq[0].shape)
    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa,\
           xyz.float(), mask, idx.long(), \
           xyz_t.float(), f1d_t.float(), mask_t, \
           xyz_prev.float(), mask_prev, \
           chain_idx, False, False, frames, bond_feats,chirals, ch_label, "atomize_pdb", item

def loader_sm(item, params, pick_top=True):
    """Load small molecule with atom tokens. Also, compute frames for atom FAPE loss calc"""
    # Load small molecule
    fname = params['CSD_DIR']+'/torch/'+item['label'][:2]+'/'+item['label']+'.pt'
    data = torch.load(fname)

    mol, msa_sm, ins_sm, xyz_sm, mask_sm = parse_mol(data["mol2"], string=True)
    a3m = {"msa": msa_sm.unsqueeze(0), "ins": ins_sm.unsqueeze(0)}
    G = get_nxgraph(mol)
    frames = get_atom_frames(msa_sm, G)

    if xyz_sm.shape[0] > params['MAXNSYMM']: # clip no. of symmetry variants to save GPU memory
        xyz_sm = xyz_sm[:params['MAXNSYMM']]
        mask_sm = mask_sm[:params['MAXNSYMM']]

    chirals = get_chirals(mol, xyz_sm[0])
    N_symmetry, sm_L, _ = xyz_sm.shape

    if sm_L < 2:
        print(f'WARNING [loader_sm]: Sm mol. {item} only has one atom. Skipping.')
        return [torch.tensor([-1])]*20 # flag for bad example

    # Generate ground truth structure: account for ligand symmetry
    xyz = torch.full((N_symmetry, sm_L, NTOTAL, 3), np.nan).float()
    xyz[:, :, 1, :] = xyz_sm

    mask = torch.full(xyz.shape[:-1], False).bool()
    mask[:, :, 1] = True # CAs

    msa = a3m['msa'].long()
    ins = a3m['ins'].long()
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params)

    idx = torch.arange(sm_L)
    chain_idx = torch.ones((sm_L, sm_L)).long()
    bond_feats = get_bond_feats(mol)

    xyz_t, f1d_t, mask_t = TemplFeaturize({"ids":[]}, sm_L, params, offset=0,
        npick=0, pick_top=pick_top)

    xyz_prev = xyz_t[0]
    mask_prev = mask_t[0].clone()

    xyz = torch.nan_to_num(xyz)
    ch_label = torch.zeros(seq[0].shape)
    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa,\
           xyz.float(), mask, idx.long(), \
           xyz_t.float(), f1d_t.float(), mask_t, \
           xyz_prev.float(), mask_prev, \
           chain_idx, False, False, frames, bond_feats, chirals, ch_label, "sm", item

def crop_sm_compl(prot_xyz, lig_xyz,Ls, params):
    """choose residues with calphas close to a random ligand atom"""
    # ligand_com = torch.nanmean(lig_xyz, dim=[0,1]).expand(1,3)
    i_face_xyz = lig_xyz[np.random.randint(len(lig_xyz))]
    dist = torch.cdist(prot_xyz[:,1].unsqueeze(0), i_face_xyz.unsqueeze(0)).flatten()
    dist = torch.nan_to_num(dist, nan=999999)
    n_ligand_atoms = len(lig_xyz)
    # select the whole ligand
    lig_sel = torch.arange(n_ligand_atoms)+Ls[0]
    if n_ligand_atoms >= params["CROP"]:
        warnings.warn(f'# of ligand atoms ({n_ligand_atoms}) >= ({params["CROP"]}), using entire ligand')
        return lig_sel
    _, idx = torch.topk(dist, params["CROP"]-len(lig_xyz), largest=False)
    sel, _ = torch.sort(idx)
    return torch.cat((sel, lig_sel))

def crop_sm_compl_assembly(all_xyz, all_mask, Ls_prot, Ls_sm, n_crop):
    """Choose residues with the `n_crop` closest C-alphas to a random atom on
    query ligand. Operates on multi-chain assemblies. Nearby ligands are
    included if all of their (unmasked) atoms are in the crop. Otherwise none
    of the atoms of that ligand are included in crop. Nearby protein chains are
    excluded if they are too short and have too few contacts to ligands or
    other protein chains. 
    
    Parameters
    ----------
    all_xyz : torch.Tensor (L_total, N_atoms, 3)
        Coordinates of full assembly with all protein chains, followed by all ligand chains. 
        1st ligand chain is assumed to be query ligand.
    all_mask : torch.Tensor (L_total, N_atoms)
        Boolean mask for whether each atom in `all_xyz` is valid
    res_mask : torch.Tensor (L_total,) bool
        Boolean mask for which residues/ligand atoms exist.
    Ls_prot : list (N_prot_chains,)
        Lengths of protein chains
    Ls_sm : list (N_lig_chains,)
        Lengths of ligand chains
    n_crop : int
        Number of nearest residues or ligand atoms to include in crop
    
    Returns
    -------
    sel : torch.Tensor (N_residues, )
        Indices of positions inside crop. Will always include entire query ligand and whole 
        ligands that are inside crop. Ligands partially inside crop will be removed, so 
        length of `sel` may be less than `n_crop`
    """
    ca_xyz = all_xyz[:,1]
    query_atom = ca_xyz[sum(Ls_prot)+np.random.randint(Ls_sm[0])] # random atom in query ligand (1st sm chain)
    dist = torch.cdist(ca_xyz.unsqueeze(0), query_atom.unsqueeze(0)).flatten()
    dist = torch.nan_to_num(dist, nan=999999)

    res_mask = torch.cat([all_mask[:sum(Ls_prot),:3].all(dim=-1), all_mask[sum(Ls_prot):,1]])

    idx = torch.argsort(dist)
    idx = idx[torch.isin(idx, torch.where(res_mask)[0])] # exclude invalid residues from crop
    idx = idx[:n_crop]

    # always include every query ligand atom, regardless of if they're in topk
    query_lig_idx = np.arange(Ls_sm[0]) + sum(Ls_prot)
    sel = np.unique(np.concatenate([idx.numpy(), query_lig_idx]))

    # remove any ligands who don't have all of their atoms in the topk
    offset = sum(Ls_prot)
    for L in Ls_sm:
        # only look for unmasked atoms to be in crop
        curr_lig_idx = np.arange(L) + offset
        curr_lig_idx = np.where(res_mask[curr_lig_idx])[0]+offset 

        if not np.isin(curr_lig_idx, sel).all():
            sel = np.setdiff1d(sel,curr_lig_idx)
        offset += L

    # remove protein chains that are short and don't contact other proteins or ligands
    # distance between protein C-alphas
    dist_prot_ca = torch.cdist(ca_xyz[:sum(Ls_prot)], ca_xyz[:sum(Ls_prot)]) # (L_prot, L_prot)

    # distance between closest heavy atom on each residue and ligand atoms
    dist_prot_lig = torch.cdist(all_xyz[:sum(Ls_prot)], ca_xyz[sum(Ls_prot):])
    dist_prot_lig[~all_mask[:sum(Ls_prot)]] = 99999
    dist_prot_lig, _ = dist_prot_lig.min(dim=1) # (L_prot, L_sm)

    res_mask_prot = res_mask[:sum(Ls_prot)]
    res_mask_lig = res_mask[sum(Ls_prot):]

    offset = 0
    for L in Ls_prot:
        # protein contacts (C-alpha within 10A)
        #dist_nonself = torch.cat([dist_prot_ca[:offset, offset:offset+L],
        #                          dist_prot_ca[offset+L:, offset:offset+L]],dim=0) # (L_prot - L, L)
        #res_mask_nonself = torch.cat([res_mask_prot[:offset], res_mask_prot[offset+L:]]) # (L_prot - L,)
        #dist_nonself = dist_nonself[res_mask_nonself][:,res_mask_prot[offset:offset+L]]
        #num_prot_contacts = (dist_nonself<10).sum()

        # protein-ligand contacts (heavy atom within 5A)
        dist_ = dist_prot_lig[offset:offset+L]
        dist_ = dist_[res_mask_prot[offset:offset+L]][:,res_mask_lig]
        num_lig_contacts = (dist_<5).sum()

        # number of residues in crop
        curr_chain_idx = np.where(res_mask)[0]
        curr_chain_idx = curr_chain_idx[(curr_chain_idx>offset) & (curr_chain_idx<offset+L)]
        num_residues = np.isin(curr_chain_idx, sel).sum()

        #if (num_residues < 8) and (num_prot_contacts < 10) and (num_lig_contacts < 10):
        if (num_residues < 8) and (num_lig_contacts < 10):
            curr_chain_idx = np.arange(L) + offset
            sel = np.setdiff1d(sel, curr_chain_idx)
            print(f'removed chain from crop: (num_residues={num_residues} '\
                  f'num_lig_contacts={num_lig_contacts})')

        offset += L

    return torch.from_numpy(sel)

def crop_chirals(chirals, atom_sel):
    """
    this function returns only chiral centers that appear in molecules that are chosen after cropping
    chirals (nchirals, 5) first four indices in the second dimension are indices and the fifth is the angle that chiral center forms
    atom_sel: 1D tensor of small molecule atoms chosen to include in the crop
    """
    if chirals.numel() == 0: # no chirals in this selection
        return chirals

    ncrop_atoms  = atom_sel.shape[0]
    # update absolute indexing if there are ligands "cropped". There will never be part of a ligand that is cropped, only entire ligands
    idx_update = dict(zip(atom_sel.numpy().tolist(), range(ncrop_atoms))) 
    cropped_chirals = []
    for chiral_center in chirals:
        if all([chiral_neighbor in atom_sel for chiral_neighbor in chiral_center[:-1]]):
            updated_chiral_center = np.array([idx_update[chiral_neighbor.item()] for chiral_neighbor in chiral_center[:-1]] + [chiral_center[-1]])
            updated_chiral_center = torch.from_numpy(updated_chiral_center)
            cropped_chirals.append(updated_chiral_center)
    if not cropped_chirals: # all the chiral centers were in ligands that were removed
        return torch.Tensor()
    return torch.stack(cropped_chirals)

def adaptor_fix_bb(out):
    """
    adapts the outputs of RF2-allatom phase 3 dataloaders into fixed bb outputs
    takes in a tuple with 22 items representing the RF2-allatom data outputs and returns a tuple of 22 items updated for the 
    fixedbb tasks
    """
    assert len(out) == 22, f"found {len(out)} elements in RF2-allatom output"
    (seq, msa, msa_masked, msa_full, mask_msa, true_crds, atom_mask, idx_pdb, xyz_t, t1d, mask_t, xyz_prev,
        mask_prev, same_chain, unclamp, negative, atom_frames, bond_feats, chirals, ch_label, dataset_name, item) = out
    #remove permutation symmetry dimension if present
    if len(true_crds.shape) == 4 and len(atom_mask.shape) == 3:
        true_crds = true_crds[0]
        atom_mask = atom_mask[0]
    
    #update template features
    xyz_t = torch.clone(true_crds)[None]
    mask_t = torch.clone(atom_mask)[None]
    seq_mask_shifted = torch.clone(seq)
    seq_mask_shifted[seq_mask_shifted>=MASKINDEX] -= 1
    f1d_t = torch.nn.functional.one_hot(seq_mask_shifted, num_classes=NAATOKENS-1)
    conf = torch.ones_like(seq[:1])[...,None]
    f1d_t = torch.cat((f1d_t, conf), dim=-1)
    t1d = f1d_t.float()

    # our dataloaders return torch.zeros(L...) for atom frames and chirals when there are none, this updates it to use common shape 
    if torch.sum(atom_frames) == 0:
        atom_frames = torch.zeros((0,3,2))
    if torch.sum(chirals) == 0:
        chirals = torch.zeros((0,5))
    return seq, msa, msa_masked, msa_full, mask_msa, true_crds, atom_mask, idx_pdb, xyz_t, t1d, mask_t, xyz_prev, \
        mask_prev, same_chain, unclamp, negative, atom_frames, bond_feats, chirals, ch_label, dataset_name, item

def unbatch_item(item):
    """
    Flattens batched dictionaries returned from dataloaders to remove unecessary nested lists
    Only used for SM compl datasets where item is a dictionary
    """
    def flatten_value(v):
        if type(v) is list and len(v)==1:
            v = v[0]
        elif type(v) is torch.Tensor and len(v.shape)>0 and v.shape[0]==1:
            v = v[0].item()
        elif type(v) is torch.Tensor and len(v.shape)==0:
            v = v.item()
        if (type(v) is list and len(v)>1):
            for i,x in enumerate(v):
                v[i] = flatten_value(x)
        return v

    new_item = dict()
    for k in item:
        new_item[k] = flatten_value(item[k])
    return new_item

def sample_item(df, ID, rng=None):
    """Sample a training example from a sequence cluster `ID` from the dataset
    represented by DataFrame `df`"""
    clus_df = df[df['CLUSTER']==ID]
    return clus_df.sample(1, random_state=rng).to_dict(orient='records')[0]

def sample_item_sm_compl(df, ID, dedup_ligand=True):
    """Sample a protein-ligand training example from sequence cluster `ID` from
    the dataset represented by DataFrame `df`"""
    # get all examples in this cluster
    tmp_df = df[df.CLUSTER==ID]

    # uniformly sample from unique PDB chains
    chid = np.random.choice(tmp_df.CHAINID.drop_duplicates().values)
    tmp_df = tmp_df[tmp_df.CHAINID==chid]

    if dedup_ligand:
        # uniform sample from unique ligands
        lignames = list(set([x[0][2] for x in tmp_df['LIGAND']]))
        chosen_lig = np.random.choice(lignames)
        tmp_df = tmp_df[tmp_df['LIGAND'].apply(lambda x: x[0][2]==chosen_lig)]

    item = tmp_df.sample(1).to_dict(orient='records')[0] # choose 1 random row

    return item


class Dataset(data.Dataset):
    def __init__(self, IDs, loader, data_df, params, homo, unclamp_cut=0.9, pick_top=True, p_homo_cut=-1.0, n_res_atomize=0, flank=0, seed=None):
        self.IDs = IDs
        self.data_df = data_df
        self.loader = loader
        self.params = params
        self.homo = homo
        self.pick_top = pick_top
        self.unclamp_cut = unclamp_cut
        self.p_homo_cut = p_homo_cut
        self.n_res_atomize = n_res_atomize
        self.flank = flank
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        item = sample_item(self.data_df, ID, self.rng)
        kwargs = dict()
        if self.n_res_atomize > 0:
            kwargs['n_res_atomize'] = self.n_res_atomize
            kwargs['flank'] = self.flank
        out = self.loader(item, self.params, self.homo,
                          unclamp = (self.rng.rand() > self.unclamp_cut),
                          pick_top = self.pick_top, 
                          p_homo_cut = self.p_homo_cut,
                          **kwargs)
        return out

class DatasetComplex(data.Dataset):
    def __init__(self, IDs, loader, data_df, params, pick_top=True, negative=False, seed=None):
        self.IDs = IDs
        self.data_df = data_df
        self.loader = loader
        self.params = params
        self.pick_top = pick_top
        self.negative = negative
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        item = sample_item(self.data_df, ID, self.rng)
        out = self.loader(item,
                          self.params,
                          pick_top = self.pick_top,
                          negative = self.negative)
        return out

class DatasetNAComplex(data.Dataset):
    def __init__(self, IDs, loader, data_df, params, pick_top=True, negative=False, native_NA_frac=0.0, seed=None):
        self.IDs = IDs
        self.data_df = data_df
        self.loader = loader
        self.params = params
        self.pick_top = pick_top
        self.negative = negative
        self.native_NA_frac = native_NA_frac
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        item = sample_item(self.data_df, ID, self.rng)
        out = self.loader(item,
                          self.params,
                          pick_top = self.pick_top,
                          negative = self.negative,
                          native_NA_frac = self.native_NA_frac
        )
        return out

class DatasetRNA(data.Dataset):
    def __init__(self, IDs, loader, data_df, params, seed=None):
        self.IDs = IDs
        self.data_df = data_df
        self.loader = loader
        self.params = params
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        item = sample_item(self.data_df, ID, self.rng)
        out = self.loader(item, self.params)
        return out

class DatasetSMComplex(data.Dataset):
    def __init__(self, IDs, loader, data_df, params, init_protein_tmpl=False, init_ligand_tmpl=False,
                 init_protein_xyz=False, init_ligand_xyz=False, task='sm_compl', seed=None):
        self.IDs = IDs
        self.data_df = data_df
        self.loader = loader
        self.params = params
        self.init_protein_tmpl = init_protein_tmpl
        self.init_ligand_tmpl = init_ligand_tmpl
        self.init_protein_xyz = init_protein_xyz
        self.init_ligand_xyz = init_ligand_xyz
        self.task = task
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        item = sample_item_sm_compl(self.data_df, ID)
        try:
            out = self.loader(
                item,
                self.params,
                init_protein_tmpl = self.init_protein_tmpl,
                init_ligand_tmpl = self.init_ligand_tmpl,
                init_protein_xyz = self.init_protein_xyz,
                init_ligand_xyz = self.init_ligand_xyz,
                task = self.task
            )
        except Exception as e:
            print('error in DatasetSMComplex',item)
            #raise e
        return out

class DatasetSMComplexAssembly(data.Dataset):
    def __init__(self, IDs, loader, data_df, chid2hash, chid2taxid, params, task, num_protein_chains=None, 
    seed=None):
        self.IDs = IDs
        self.data_df = data_df
        self.loader = loader
        self.chid2hash = chid2hash
        self.chid2taxid = chid2taxid
        self.params = params
        self.task = task
        self.num_protein_chains = num_protein_chains
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.IDs)
        
    def __getitem__(self, index):
        ID = self.IDs[index]
        item = sample_item_sm_compl(self.data_df, ID)
        try:
            out = self.loader(
                item,
                self.params,
                self.chid2hash,
                self.chid2taxid,
                task=self.task,
                num_protein_chains=self.num_protein_chains,
            )
        except Exception as e:
            print('error in DatasetSMComplexAssembly',item)
            #raise e
        return out

class DatasetSM(data.Dataset):
    def __init__(self, IDs, loader, data_df, params, seed=None):
        self.IDs = IDs
        self.data_df = data_df
        self.loader = loader
        self.params = params
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        item = sample_item(self.data_df, ID, self.rng)
        out = self.loader(item, self.params)
        return out

class DistilledDataset(data.Dataset):
    def __init__(self, ID_dict, dataset_dict, loader_dict, homo, chid2hash, chid2taxid, params, 
                 native_NA_frac=0.25, unclamp_cut=0.9):

        self.ID_dict = ID_dict
        self.dataset_dict = dataset_dict
        self.loader_dict = loader_dict
        self.homo = homo
        self.chid2hash = chid2hash
        self.chid2taxid = chid2taxid
        self.params = params
        self.unclamp_cut = unclamp_cut
        self.native_NA_frac = native_NA_frac
        self.index_dict = OrderedDict([
            (k, np.arange(len(self.ID_dict[k]))) for k in self.dataset_dict.keys()
        ])

    def __len__(self):
        return sum([len(v) for k,v in self.index_dict.items()])

    def __getitem__(self, index):
        p_unclamp = np.random.rand()

        # order of datasets here must match key order in self.dataset_dict
        offset = 0
        if index >= offset and index < offset + len(self.index_dict['pdb']):
            ID = self.ID_dict['pdb'][index-offset]
            item = sample_item(self.dataset_dict['pdb'], ID)
            out = self.loader_dict['pdb'](item, self.params, self.homo, unclamp=(p_unclamp > self.unclamp_cut))
        offset += len(self.index_dict['pdb'])

        if index >= offset and index < offset + len(self.index_dict['fb']):
            ID = self.ID_dict['fb'][index-offset]
            item = sample_item(self.dataset_dict['fb'], ID)
            out = self.loader_dict['fb'](item, self.params, unclamp=(p_unclamp > self.unclamp_cut))
        offset += len(self.index_dict['fb'])

        if index >= offset and index < offset + len(self.index_dict['compl']):
            ID = self.ID_dict['compl'][index-offset]
            item = sample_item(self.dataset_dict['compl'], ID)
            out = self.loader_dict['compl'](item, self.params, negative=False)
        offset += len(self.index_dict['compl'])

        #if index >= offset and index < offset + len(self.neg_inds):
        #   ID = self.neg_IDs[index-offset]
        #   sel_idx = np.random.randint(0, len(self.neg_dict[ID]))
        #   out = self.neg_loader(
        #       self.neg_dict[ID][sel_idx][0],
        #       self.neg_dict[ID][sel_idx][1],
        #       self.neg_dict[ID][sel_idx][2],
        #       self.neg_dict[ID][sel_idx][3],
        #       self.params,
        #       negative=True
        #   )
        #offset += len(self.neg_inds)

        if index >= offset and index < offset + len(self.index_dict['na_compl']):
            ID = self.ID_dict['na_compl'][index-offset]
            item = sample_item(self.dataset_dict['na_compl'], ID)
            out = self.loader_dict['na_compl'](item, self.params, negative=False, native_NA_frac=self.native_NA_frac)
        offset += len(self.index_dict['na_compl'])

        #if index >= offset and index < offset + len(self.na_neg_inds):
        #   ID = self.na_neg_IDs[index-offset]
        #   sel_idx = np.random.randint(0, len(self.na_neg_dict[ID]))
        #   out = self.na_neg_loader(
        #       self.na_neg_dict[ID][sel_idx][0],
        #       self.na_neg_dict[ID][sel_idx][1],
        #       self.params,
        #       negative=True,
        #       native_NA_frac=self.native_NA_frac
        #   )
        #offset += len(self.na_neg_inds)

        try:
            if index >= offset and index < offset + len(self.index_dict['rna']):
                ID = self.ID_dict['rna'][index-offset]
                item = sample_item(self.dataset_dict['rna'], ID)
                out = self.loader_dict['rna'](item, self.params)
            offset += len(self.index_dict['rna'])

            if index >= offset and index < offset + len(self.index_dict['sm_compl']):
                ID = self.ID_dict['sm_compl'][index-offset]
                item = sample_item_sm_compl(self.dataset_dict['sm_compl'], ID)
                out = self.loader_dict['sm_compl'](item, self.params, self.chid2hash, 
                self.chid2taxid, task='sm_compl', num_protein_chains=1)
            offset += len(self.index_dict['sm_compl'])

            if index >= offset and index < offset + len(self.index_dict['metal_compl']):
                ID = self.ID_dict['metal_compl'][index-offset]
                item = sample_item_sm_compl(self.dataset_dict['metal_compl'], ID)
                out = self.loader_dict['metal_compl'](item, self.params, self.chid2hash, 
                self.chid2taxid, task='metal_compl', num_protein_chains=1)
            offset += len(self.index_dict['metal_compl'])

            if index >= offset and index < offset + len(self.index_dict['sm_compl_multi']):
                ID = self.ID_dict['sm_compl_multi'][index-offset]
                item = sample_item_sm_compl(self.dataset_dict['sm_compl_multi'], ID)
                out = self.loader_dict['sm_compl_multi'](item, self.params, self.chid2hash, 
                self.chid2taxid, task='sm_compl_multi', num_protein_chains=1)
            offset += len(self.index_dict['sm_compl_multi'])

            if index >= offset and index < offset + len(self.index_dict['sm_compl_covale']):
                ID = self.ID_dict['sm_compl_covale'][index-offset]
                item = sample_item_sm_compl(self.dataset_dict['sm_compl_covale'], ID)
                out = self.loader_dict['sm_compl_covale'](item, self.params, self.chid2hash, 
                self.chid2taxid, task='sm_compl_covale')
            offset += len(self.index_dict['sm_compl_covale'])

            if index >= offset and index < offset + len(self.index_dict['sm_compl_asmb']):
                ID = self.ID_dict['sm_compl_asmb'][index-offset]
                item = sample_item_sm_compl(self.dataset_dict['sm_compl_asmb'], ID)
                out = self.loader_dict['sm_compl_asmb'](item, self.params, self.chid2hash, 
                self.chid2taxid)
            offset += len(self.index_dict['sm_compl_asmb'])

            if index >= offset and index < offset + len(self.index_dict['sm']):
                ID = self.ID_dict['sm'][index-offset]
                item = sample_item(self.dataset_dict['sm'], ID)
                out = self.loader_dict['sm'](item, self.params)
            offset += len(self.index_dict['sm'])

            if index >= offset and index < offset + len(self.index_dict['atomize_pdb']):
                ID = self.ID_dict['atomize_pdb'][index-offset]
                item = sample_item(self.dataset_dict['atomize_pdb'], ID)
                n_res_atomize = np.random.randint(self.params['NRES_ATOMIZE_MIN'], 
                                                  self.params['NRES_ATOMIZE_MAX']+1)
                out = self.loader_dict['atomize_pdb'](item,
                    self.params, self.homo, n_res_atomize, self.params['ATOMIZE_FLANK'], 
                    unclamp=(p_unclamp > self.unclamp_cut))
            offset += len(self.index_dict['atomize_pdb'])
        except Exception as e:
            # print(int(item['CLUSTER']), item['CHAINID'], task)
            print('error loading',item)
            raise e

        return out

class DistributedWeightedSampler(data.Sampler):
    def __init__(
        self,
        dataset,
        weights_dict,
        num_example_per_epoch=25600,
        fractions = OrderedDict(
            pdb=1.,
            fb=0,
            compl=0,
            na_compl=0,
            rna=0,
            sm_compl=0,
            metal_compl=0,
            sm_compl_multi=0,
            sm_compl_covale=0,
            sm_compl_asmb=0,
            sm=0,
            atomize_pdb=0,
        ),
        num_replicas=None,
        rank=None,
        replacement=False
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        assert num_example_per_epoch % num_replicas == 0
        assert (np.allclose(sum([v for k,v in fractions.items()]), 1.0)), \
            f"Fractions of datasets add up to {sum([v for k,v in fractions.items()])}, should add up to 1.0"

        self.dataset = dataset
        self.weights_dict = weights_dict
        self.num_replicas = num_replicas
        self.num_per_epoch_dict = OrderedDict([
            (dataset_name, int(round(num_example_per_epoch * fractions[dataset_name]))) 
            for dataset_name in self.dataset.dataset_dict.keys()
        ])
        self.total_size = num_example_per_epoch
        self.num_samples = self.total_size // self.num_replicas
        self.rank = rank
        self.epoch = 0
        self.replacement = replacement

        if (rank==0):
            print(f"Training examples per epoch ({self.total_size} total):")
            for k,v in self.num_per_epoch_dict.items():
                print('  '+k, ':', v)

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        # get indices (fb + pdb models)
        indices = torch.arange(len(self.dataset))

        # weighted subsampling
        # order of datasets in this loop should match order in DistilledDataset.__getitem__()
        offset = 0
        sel_indices = torch.tensor((),dtype=int)
        for dataset_name in self.dataset.dataset_dict.keys():
            if (self.num_per_epoch_dict[dataset_name]> 0):
                sampled_idx = torch.multinomial(self.weights_dict[dataset_name], 
                                                self.num_per_epoch_dict[dataset_name], 
                                                self.replacement, 
                                                generator=g)
                sel_indices = torch.cat((sel_indices, indices[sampled_idx + offset]))
            offset += len(self.dataset.ID_dict[dataset_name])

        # shuffle indices
        indices = sel_indices[torch.randperm(len(sel_indices), generator=g)]

        # per each gpu
        indices = indices[self.rank:self.total_size:self.num_replicas]
        print('rank',self.rank,': expecting',self.num_samples,'examples, drew',len(indices),'examples')
        #assert len(indices) == self.num_samples

        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

