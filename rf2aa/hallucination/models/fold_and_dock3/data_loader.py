import torch
from torch.utils import data
import os, csv, random, pickle
from dateutil import parser
import numpy as np
import pandas as pd
import ast
import scipy
from scipy.sparse.csgraph import shortest_path

from parsers import parse_a3m, parse_pdb, parse_fasta_if_exists, parse_mol
from chemical import INIT_CRDS, INIT_NA_CRDS, NAATOKENS, MASKINDEX, NTOTAL, NBTYPES, CHAIN_GAP
from kinematics import get_chirals
from util import get_nxgraph, get_atom_frames, get_bond_feats, get_protein_bond_feats, \
    atomize_protein, center_and_realign_missing, random_rot_trans, allatom_mask

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
mol_dir = "/projects/ml/RF2_allatom/isdf"

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
    mol_dir = "/gscratch2/RF2_allatom/isdf"
    csd_dir = "/gscratch2/RF2_allatom/csd543"

def set_data_loader_params(args):
    params = {
        "COMPL_LIST"       : "%s/list.hetero.csv"%compl_dir,
        "HOMO_LIST"        : "%s/list.homo.csv"%compl_dir,
        "NEGATIVE_LIST"    : "%s/list.negative.csv"%compl_dir,
        "RNA_LIST"         : "%s/list.rnaonly.csv"%na_dir,
        "NA_COMPL_LIST"    : "%s/list.nucleic.NODIMERS.csv"%sm_compl_dir,
        "NEG_NA_COMPL_LIST": "%s/list.na_negatives.csv"%na_dir,
        "SM_LIST"          : "%s/list_v02_smcompl_20221123.csv"%sm_compl_dir, 
        "PDB_LIST"         : "%s/list_v02.csv"%base_dir, # on digs
        "FB_LIST"          : "%s/list_b1-3.csv"%fb_dir,
        "CSD_LIST"         : "%s/csd543_cleaned01.csv"%csd_dir, 
        "VAL_PDB"          : "%s/valid_remapped"%sm_compl_dir,
        "VAL_RNA"          : "%s/rna_valid.csv"%na_dir,
        "VAL_COMPL"        : "%s/val_lists/xaa"%compl_dir,
        "VAL_NEG"          : "%s/val_lists/xaa.neg"%compl_dir,
        "VAL_SM_LIGCLUS"   : "%s/list_v02_smcompl_ligclusvalid_20221117.csv"%sm_compl_dir, 
        "VAL_SM_STRICT"    : "%s/list_v02_smcompl_validstrict_20221102.csv"%sm_compl_dir, 
        "VAL_PEP"          : "%s/list_v02_peptide_benchmark_valid.csv"%sm_compl_dir,
        "TEST_SM"          : "%s/sm_test_heldout_test_clusters.txt"%sm_compl_dir,
        "DATAPKL"          : "%s/dataset_20221123.pkl"%sm_compl_dir, # cache for faster loading
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
        "CROP"             : 256,
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
    for param in params:
        if hasattr(args, param.lower()):
            params[param] = getattr(args, param.lower())
    return params

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

def get_term_feats(L,Ls):
    """Creates N/C-terminus binary features"""
    term_info = torch.zeros((L,2)).float()
    if len(Ls) < 1:
        term_info[0,0] = 1.0 # flag for N-term
        term_info[-1,1] = 1.0 # flag for C-term
    else:
        start = 0
        for L_chain in Ls:
            term_info[start, 0] = 1.0 # flag for N-term
            term_info[start+L_chain-1,1] = 1.0 # flag for C-term
            start += L_chain
    return term_info

def cluster_sum(data, assignment, N_seq, N_res):
    csum = torch.zeros(N_seq, N_res, data.shape[-1], device=data.device).scatter_add(0, assignment.view(-1,1,1).expand(-1,N_res,data.shape[-1]), data.float())
    return csum

def MSAFeaturize(msa, ins, params, p_mask=0.15, eps=1e-6, nmer=1, L_s=[], tocpu=False):
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
    N, L = msa.shape
    
    term_info = get_term_feats(L, L_s).to(msa.device)

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

        msa_masked = torch.where(mask_pos, mask_sample, msa_clust)
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

def TemplFeaturize(tplt, qlen, params, offset=0, npick=1, npick_global=None, pick_top=True, same_chain=None, random_noise=5):

    def _blank_template():
        xyz = INIT_CRDS.reshape(1,1,NTOTAL,3).repeat(npick_global,qlen,1,1) \
            + torch.rand(npick_global,qlen,1,3)*random_noise - random_noise/2
        t1d = torch.nn.functional.one_hot(torch.full((npick_global, qlen), 20).long(), num_classes=NAATOKENS-1).float() # all gaps
        conf = torch.zeros((npick_global, qlen, 1)).float()
        t1d = torch.cat((t1d, conf), -1)
        mask_t = torch.full((npick_global,qlen,NTOTAL), False)
        return xyz, t1d, mask_t

    seqID_cut = params['SEQID']

    if npick_global == None:
        npick_global=max(npick, 1)

    ntplt = len(tplt['ids'])
    if (ntplt < 1) or (npick < 1): #no templates in hhsearch file or not want to use templ
        return _blank_template()
    
    # ignore templates having too high seqID
    if seqID_cut <= 100.0:
        tplt_valid_idx = torch.where(tplt['f0d'][0,:,4] < seqID_cut)[0]
        tplt['ids'] = np.array(tplt['ids'])[sel]
    else:
        tplt_valid_idx = torch.arange(len(tplt['ids']))
    
    # check again if there are templates having seqID < cutoff
    ntplt = len(tplt['ids'])
    npick = min(npick, ntplt)
    if npick<1: # no templates
        return _blank_template()

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


def get_train_valid_set(params, OFFSET=1000000):
    if (not os.path.exists(params['DATAPKL'])):
        print(f'cached train/valid datasets {params["DATAPKL"]} not found. '\
              f're-parsing train/valid metadata...')

        # read validation IDs for PDB set
        val_pdb_ids = set([int(l) for l in open(params['VAL_PDB']).readlines()])
        val_compl_ids = set([int(l) for l in open(params['VAL_COMPL']).readlines()])
        val_neg_ids = set([int(l)+OFFSET for l in open(params['VAL_NEG']).readlines()])
        val_rna_pdb_ids = set([l.rstrip() for l in open(params['VAL_RNA']).readlines()])
        test_sm_ids = set([int(l) for l in open(params['TEST_SM']).readlines()])

        # read & clean RNA list
        with open(params['RNA_LIST'], 'r') as f:
            reader = csv.reader(f)
            next(reader)
            rows = [[r[0],[int(clid) for clid in r[3].split(':')], [int(plen) for plen in r[4].split(':')]] for r in reader
                    if float(r[2]) <= params['RESCUT'] and
                    parser.parse(r[1]) <= parser.parse(params['DATCUT'])]

        # compile training and validation sets
        train_rna = {}
        valid_rna = {}
        for i,r in enumerate(rows):
            if any([x in val_rna_pdb_ids for x in r[0].split(":")]):
                valid_rna[i] = [(r[0], r[-1])]
            else:
                train_rna[i] = [(r[0], r[-1])]

        # parse protein-small molecule complexes
        def _load_df(filename, eval_cols=[]):
            """Loads CSV into dataframe and applies general filters"""
            df = pd.read_csv(filename)
            df['HASH'] = df['HASH'].apply(lambda x: f'{x:06d}') # restore leading zeros, make into string
            for col in eval_cols:
                df[col] = df[col].apply(lambda x: ast.literal_eval(x)) # interpret as list of strings
            df = df[
                (df.RESOLUTION<=params['RESCUT']) &
                (df.DEPOSITION.apply(lambda x: parser.parse(x))<=parser.parse(params['DATCUT']))
            ]
            return df

        def _make_example_dict(df, make_entry_func, cluster_key='CLUSTER'):
            """Converts a dataframe of training examples to dictionary where keys are cluster IDs and values
               are lists of training examples formatted according to the function `make_entry_func`."""
            data_dict = {}
            for i,row in df.iterrows():
                entry = make_entry_func(row)
                cluster = row[cluster_key] # cluster of this example
                if cluster in data_dict.keys():
                    data_dict[cluster].append(entry)
                else:
                    data_dict[cluster] = [entry]
            return data_dict

        df = _load_df(params['SM_LIST'], eval_cols=['LIGANDS'])
        df = df[
            ~df.LIGANDS.apply(lambda x: '1fcv_GCU_1_A_405__B___.mol2' in x) # tanimoto neighbors=0, weight=Inf
        ]

        # weight each example by various factors
        seq_len_factor = (1/512.)*np.clip(df.LEN_EXIST, 256, 512) # sample longer sequences more often
        lig_copy_num = df.LIGANDS.apply(lambda x: len(x)).astype(float) # num copies of this ligand in this structure
        multi_lig_factor = lig_copy_num/df.NUM_LIGANDS # correct for multiple unique ligands per protein
        if params['CLUSTER_LIGANDS']:
            ligand_cluster_factor = lig_copy_num/df.LIGAND_CLUSTER_SIZE # how many sm mol have tanimoto > 0.85 to this?
        else:
            ligand_cluster_factor = 1.0
        seq_cluster_factor = 1./df.CLUSTER_SIZE # protein seq similarity cluster size
        df['WEIGHT'] = seq_len_factor * multi_lig_factor * ligand_cluster_factor * seq_cluster_factor

        # compile protein-sm. mol training & validation sets
        def _make_entry(row):
            return ((row.CHAINID, row.HASH, row.LIGANDS), row.LEN_EXIST, row.WEIGHT)
        train_sm_compl = _make_example_dict(df[~df.CLUSTER.isin(val_pdb_ids)], make_entry_func=_make_entry)
        valid_sm_compl = _make_example_dict(df[df.CLUSTER.isin(val_pdb_ids)], make_entry_func=_make_entry)

        # protein-small mol cluster weights are the sum of cluster member weights
        sm_compl_IDs = list(train_sm_compl.keys())
        df_clus = df[['CLUSTER','WEIGHT']].groupby('CLUSTER').sum().reset_index()
        clus2weight = dict(zip(df_clus.CLUSTER, df_clus.WEIGHT))
        sm_compl_weights = [clus2weight[i] for i in sm_compl_IDs]

        # stricter versions of protein-small mol validation set
        df = _load_df(params['VAL_SM_LIGCLUS'], eval_cols=['LIGANDS']) # deduplicated ligand clusters
        df['WEIGHT'] = 1 # examples have already been redundancy-reduced
        valid_sm_compl_ligclus = _make_example_dict(df, make_entry_func=_make_entry)

        df = _load_df(params['VAL_SM_STRICT'], eval_cols=['LIGANDS']) # removed ligand cluster overlap with train
        df['WEIGHT'] = 1 # examples have already been redundancy-reduced
        valid_sm_compl_strict = _make_example_dict(df, make_entry_func=_make_entry, cluster_key='ID') # use all examples regardless of sequence cluster

        # read homo-oligomer list
        homo = {}
        with open(params['HOMO_LIST'], 'r') as f:
            reader = csv.reader(f)
            next(reader)
            # read pdbA, pdbB, bioA, opA, bioB, opB
            rows = [[r[0], r[1], int(r[2]), int(r[3]), int(r[4]), int(r[5])] for r in reader]
        for r in rows:
            if r[0] in homo.keys():
                homo[r[0]].append(r[1:])
            else:
                homo[r[0]] = [r[1:]]

        # read & clean list.csv
        with open(params['PDB_LIST'], 'r') as f:
            reader = csv.reader(f)
            next(reader)
            rows = [[r[0],r[3],int(r[4]), int(r[-1].strip())] for r in reader
                    if float(r[2])<=params['RESCUT'] and
                    parser.parse(r[1])<=parser.parse(params['DATCUT'])]

        # compile training and validation sets
        val_hash = list()
        train_pdb = {}
        valid_pdb = {}
        valid_homo = {}
        for r in rows:
            if r[2] in test_sm_ids:
                continue # completely held out test set examples
            if r[2] in val_pdb_ids: 
                val_hash.append(r[1])
                if r[2] in valid_pdb.keys():
                    valid_pdb[r[2]].append((r[:2], r[-1]))
                else:
                    valid_pdb[r[2]] = [(r[:2], r[-1])]
                #
                if r[0] in homo:
                    if r[2] in valid_homo.keys():
                        valid_homo[r[2]].append((r[:2], r[-1]))
                    else:
                        valid_homo[r[2]] = [(r[:2], r[-1])]
            else:
                if r[2] in train_pdb.keys():
                    train_pdb[r[2]].append((r[:2], r[-1]))
                else:
                    train_pdb[r[2]] = [(r[:2], r[-1])]

        # compile facebook model sets
        with open(params['FB_LIST'], 'r') as f:
            reader = csv.reader(f)
            next(reader)
            rows = [[r[0],r[2],int(r[3]),len(r[-1].strip())] for r in reader
                     if float(r[1]) > 80.0 and
                     len(r[-1].strip()) > 200]
        fb = {}
        for r in rows:
            if r[2] in fb.keys():
                fb[r[2]].append((r[:2], r[-1]))
            else:
                fb[r[2]] = [(r[:2], r[-1])]

        # compile complex sets
        with open(params['COMPL_LIST'], 'r') as f:
            reader = csv.reader(f)
            next(reader)
            # read complex_pdb, pMSA_hash, complex_cluster, length, taxID, assembly (bioA,opA,bioB,opB)
            rows = [[r[0], r[3], int(r[4]), [int(plen) for plen in r[5].split(':')], r[6] , [int(r[7]), int(r[8]), int(r[9]), int(r[10])]] for r in reader
                    if float(r[2]) <= params['RESCUT'] and
                    parser.parse(r[1]) <= parser.parse(params['DATCUT'])]

        train_compl = {}
        valid_compl = {}
        for r in rows:
            if r[2] in val_compl_ids:
                if r[2] in valid_compl.keys():
                    valid_compl[r[2]].append((r[:2], r[-3], r[-2], r[-1])) # ((pdb, hash), length, taxID, assembly, negative?)
                else:
                    valid_compl[r[2]] = [(r[:2], r[-3], r[-2], r[-1])]
            else:
                # if subunits are included in PDB validation set, exclude them from training
                hashA, hashB = r[1].split('_')
                if hashA in val_hash:
                    continue
                if hashB in val_hash:
                    continue
                if r[2] in train_compl.keys():
                    train_compl[r[2]].append((r[:2], r[-3], r[-2], r[-1]))
                else:
                    train_compl[r[2]] = [(r[:2], r[-3], r[-2], r[-1])]

        # compile negative examples
        # remove pairs if any of the subunits are included in validation set
        with open(params['NEGATIVE_LIST'], 'r') as f:
            reader = csv.reader(f)
            next(reader)
            # read complex_pdb, pMSA_hash, complex_cluster, length, taxonomy
            rows = [[r[0],r[3],OFFSET+int(r[4]),[int(plen) for plen in r[5].split(':')],r[6]] for r in reader
                    if float(r[2])<=params['RESCUT'] and
                    parser.parse(r[1])<=parser.parse(params['DATCUT'])]

        train_neg = {}
        valid_neg = {}
        for r in rows:
            if r[2] in val_neg_ids:
                if r[2] in valid_neg.keys():
                    valid_neg[r[2]].append((r[:2], r[-2], r[-1], []))
                else:
                    valid_neg[r[2]] = [(r[:2], r[-2], r[-1], [])]
            else:
                hashA, hashB = r[1].split('_')
                if hashA in val_hash:
                    continue
                if hashB in val_hash:
                    continue
                if r[2] in train_neg.keys():
                    train_neg[r[2]].append((r[:2], r[-2], r[-1], []))
                else:
                    train_neg[r[2]] = [(r[:2], r[-2], r[-1], [])]

        # compile NA complex sets
        # use PDB validation set as validation set
        with open(params['NA_COMPL_LIST'], 'r') as f:
            reader = csv.reader(f)
            next(reader)
            # read complex_pdb, pMSA_hash, complex_cluster, length
            rows = [[r[0], r[3], int(r[4]), [int(plen) for plen in r[5].split(':')]] for r in reader
                    if float(r[2]) <= params['RESCUT'] and
                    parser.parse(r[1]) <= parser.parse(params['DATCUT'])]

        train_na_compl = {}
        valid_na_compl = {}
        for r in rows:
            if r[2] in val_compl_ids:
                if r[2] in valid_na_compl.keys():
                    valid_na_compl[r[2]].append((r[:2], r[-1])) # ((pdb, hash), length)
                else:
                    valid_na_compl[r[2]] = [(r[:2], r[-1])]
            else:
                if r[2] in train_na_compl.keys():
                    train_na_compl[r[2]].append((r[:2], r[-1]))
                else:
                    train_na_compl[r[2]] = [(r[:2], r[-1])]

        # compile negative examples
        # remove pairs if any of the subunits are included in validation set
        with open(params['NEG_NA_COMPL_LIST'], 'r') as f:
            reader = csv.reader(f)
            next(reader)
            # read complex_pdb, pMSA_hash, complex_cluster, length, taxonomy
            rows = [[r[0],r[3],OFFSET+int(r[4]),[int(plen) for plen in r[5].split(':')]] for r in reader
                    if float(r[2])<=params['RESCUT'] and
                    parser.parse(r[1])<=parser.parse(params['DATCUT'])]

        train_na_neg = {}
        valid_na_neg = {}
        for r in rows:
            if r[2] in val_neg_ids:
                if r[2] in valid_na_neg.keys():
                    valid_na_neg[r[2]].append((r[:2], r[-1]))
                else:
                    valid_na_neg[r[2]] = [(r[:2], r[-1])]
            else:
                if r[2] in train_na_neg.keys():
                    train_na_neg[r[2]].append((r[:2], r[-1]))
                else:
                    train_na_neg[r[2]] = [(r[:2], r[-1])]
        
        # cambridge small molecule crystals
        sim_idx = int(params["MAXSIM"]*100-50)
        with open(params["CSD_LIST"], 'r') as f:
            reader = csv.reader(f)
            next(reader)
            rows = [[r[1], # name
                    ast.literal_eval(r[11])[sim_idx], # train similarity 
                    ast.literal_eval(r[13])[sim_idx]] # valid similarity
                    for r in reader if float(r[4]) <= params["RMAX"] 
                    and float(r[8]) <= params["MAXRES"] 
                    and float(r[9]) <= params["MAXATOMS"] 
                    and float(r[9]) >= params["MINATOMS"] 
                    and ast.literal_eval(r[12])[sim_idx]==0] # not in test set

        train_sm = {}
        valid_sm = {}
        for i, row in enumerate(rows):
            if row[2] != 0:
                valid_sm[i] = row[:2]
            else:
                train_sm[i] = row[:2]

        # protein-peptide complexes (validation only)
        def _make_entry(row):
            return ((row.CHAINID, row.HASH, row.PEP_CHAINID), row.LEN_EXIST)
        df = _load_df(params['VAL_PEP'])
        valid_pep = _make_example_dict(df, make_entry_func=_make_entry)

        # Get average chain length in each cluster and calculate weights
        # protein-small mol complex weights are done separately above
        pdb_IDs = list(train_pdb.keys())
        fb_IDs = list(fb.keys())
        compl_IDs = list(train_compl.keys())
        neg_IDs = list(train_neg.keys())
        na_compl_IDs = list(train_na_compl.keys())
        na_neg_IDs = list(train_na_neg.keys())
        rna_IDs = list(train_rna.keys())
        sm_IDs = list(train_sm.keys())

        pdb_weights = np.array([train_pdb[key][0][1] for key in pdb_IDs])
        pdb_weights = (1/512.)*np.clip(pdb_weights, 256, 512)
        fb_weights = np.array([fb[key][0][1] for key in fb_IDs])
        fb_weights = (1/512.)*np.clip(fb_weights, 256, 512)
        compl_weights = np.array([sum(train_compl[key][0][1]) for key in compl_IDs])
        compl_weights = (1/512.)*np.clip(compl_weights, 256, 512)
        neg_weights = np.array([sum(train_neg[key][0][1]) for key in neg_IDs])
        neg_weights = (1/512.)*np.clip(neg_weights, 256, 512)
        na_compl_weights = np.array([sum(train_na_compl[key][0][1]) for key in na_compl_IDs])
        na_compl_weights = (1/512.)*np.clip(na_compl_weights, 256, 512)
        na_neg_weights = np.array([sum(train_na_neg[key][0][1]) for key in na_neg_IDs])
        na_neg_weights = (1/512.)*np.clip(na_neg_weights, 256, 512)
        rna_weights = np.ones(len(rna_IDs)) # no weighing
        sm_weights = np.array([train_sm[key][1] for key in sm_IDs])

        # save
        obj = (
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
            homo
        )
        with open(params["DATAPKL"], "wb") as f:
            print ('Writing',params["DATAPKL"],'...')
            pickle.dump(obj, f)
            print ('...done')
    else:
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
                homo
            ) = pickle.load(f)
            print ('...done')

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

# Generate input features for single-chain
def featurize_single_chain(msa, ins, tplt, pdb, params, unclamp=False, pick_top=True, random_noise=5.0):
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params)
    
    # get template features
    ntempl = np.random.randint(params['MINTPLT'], params['MAXTPLT']+1)
    xyz_t, f1d_t, mask_t = TemplFeaturize(tplt, msa.shape[1], params, npick=ntempl, offset=0, pick_top=pick_top, random_noise=random_noise)
    
    # get ground-truth structures
    idx = torch.arange(len(pdb['xyz'])) 
    xyz = torch.full((len(idx),NTOTAL,3),np.nan).float()
    xyz[:,:14,:] = pdb['xyz']
    mask = torch.full((len(idx), NTOTAL), False)
    mask[:,:14] = pdb['mask']
    xyz = torch.nan_to_num(xyz)

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
def featurize_homo(msa_orig, ins_orig, tplt, pdbA, pdbid, interfaces, params, pick_top=True, random_noise=5.0):
    L = msa_orig.shape[1]
    
    msa, ins = merge_a3m_homo(msa_orig, ins_orig, 2) # make unpaired alignments, for training, we always use two chains
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params, nmer=2, L_s=[L,L])

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
        pdbB = torch.load(params['PDB_DIR']+'/torch/pdb/'+interface[0][1:3]+'/'+interface[0]+'.pt')
        xformA = meta['asmb_xform%d'%interface[1]][interface[2]]
        xformB = meta['asmb_xform%d'%interface[3]][interface[4]]
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
            crop_idx = get_spatial_crop(xyz[spatial_crop_tgt], mask[spatial_crop_tgt], torch.arange(L*2), [L,L], params, interfaces[spatial_crop_tgt][0])
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
    msa,ins = parse_a3m(a3mfilename, unzip=unzip)
    return {'msa':torch.tensor(msa), 'ins':torch.tensor(ins), 'label':item}

# Load PDB examples
def loader_pdb(item, params, homo, unclamp=False, pick_top=True, p_homo_cut=0.5):
    # load MSA, PDB, template info
    pdb = torch.load(params['PDB_DIR']+'/torch/pdb/'+item[0][1:3]+'/'+item[0]+'.pt')
    a3m = get_msa(params['PDB_DIR'] + '/a3m/' + item[1][:3] + '/' + item[1] + '.a3m.gz', item[1])
    tplt = torch.load(params['PDB_DIR']+'/torch/hhr/'+item[1][:3]+'/'+item[1]+'.pt')
   
    # get msa features
    msa = a3m['msa'].long()
    ins = a3m['ins'].long()
    if len(msa) > params['BLOCKCUT']:
        msa, ins = MSABlockDeletion(msa, ins)

    if item[0] in homo: # Target is homo-oligomer
        p_homo = np.random.rand()
        if p_homo < p_homo_cut: # model as homo-oligomer with p_homo_cut prob
            pdbid = item[0].split('_')[0]
            interfaces = homo[item[0]]
            feats = featurize_homo(msa, ins, tplt, pdb, pdbid, interfaces, params, pick_top=pick_top)
            return feats + ("homo",item,)
        else:
            return featurize_single_chain(msa, ins, tplt, pdb, params, unclamp=unclamp, pick_top=pick_top) \
                   + ("monomer",item,)
    else:
        return featurize_single_chain(msa, ins, tplt, pdb, params, unclamp=unclamp, pick_top=pick_top) \
               + ("monomer",item,)

    
def loader_fb(item, params, unclamp=False):
    
    # loads sequence/structure/plddt information 
    a3m = get_msa(os.path.join(params["FB_DIR"], "a3m", item[-1][:2], item[-1][2:], item[0]+".a3m.gz"), item[0])
    pdb = get_pdb(os.path.join(params["FB_DIR"], "pdb", item[-1][:2], item[-1][2:], item[0]+".pdb"),
                  os.path.join(params["FB_DIR"], "pdb", item[-1][:2], item[-1][2:], item[0]+".plddt.npy"),
                  item[0], params['PLDDTCUT'], params['SCCUT'])
    
    # get msa features
    msa = a3m['msa'].long()
    ins = a3m['ins'].long()
    l_orig = msa.shape[1]
    if len(msa) > params['BLOCKCUT']:
        msa, ins = MSABlockDeletion(msa, ins)
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params)
    
    # get template features -- None
    xyz_t = torch.full((1,l_orig,NTOTAL,3),np.nan).float()
    f1d_t = torch.nn.functional.one_hot(torch.full((1, l_orig), 20).long(), num_classes=NAATOKENS-1).float() # all gaps
    conf = torch.zeros((1,l_orig,1)).float() # zero confidence
    f1d_t = torch.cat((f1d_t, conf), -1)
    
    idx = pdb['idx']
    xyz = torch.full((len(idx),NTOTAL,3),np.nan).float()
    xyz[:,:27,:] = pdb['xyz']
    mask = torch.full((len(idx),NTOTAL), False)
    mask[:,:27] = pdb['mask']

    # Residue cropping
    crop_idx = get_crop(len(idx), mask, msa_seed_orig.device, params['CROP'], unclamp=unclamp)
    seq = seq[:,crop_idx]
    msa_seed_orig = msa_seed_orig[:,:,crop_idx]
    msa_seed = msa_seed[:,:,crop_idx]
    msa_extra = msa_extra[:,:,crop_idx]
    mask_msa = mask_msa[:,:,crop_idx]
    xyz_t = xyz_t[:,crop_idx]
    f1d_t = f1d_t[:,crop_idx]
    xyz = xyz[crop_idx]
    mask = mask[crop_idx]
    idx = idx[crop_idx]

    # initial structure
    xyz_prev = xyz_t[0].clone()
    chain_idx = torch.ones((len(crop_idx), len(crop_idx))).long()
    bond_feats = get_protein_bond_feats(len(crop_idx)).long()
    chirals = torch.Tensor()
    #print ("loader_fb", mask.shape, xyz_t.shape, f1d_t.shape, xyz_prev.shape)

    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa, \
           xyz.float(), mask, idx.long(),\
           xyz_t.float(), f1d_t.float(), xyz_prev.float(), \
           chain_idx, unclamp, False, torch.zeros(seq.shape), bond_feats,chirals,"fb", item


def loader_complex(item, L_s, taxID, assem, params, negative=False, pick_top=True, random_noise=5.0):
    pdb_pair = item[0]
    pMSA_hash = item[1]
    
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
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params, L_s=L_s)

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
    
    if len(assem) > 0:
        # read metadata
        pdbid = pdbA_id.split('_')[0]
        meta = torch.load(params['PDB_DIR']+'/torch/pdb/'+pdbid[1:3]+'/'+pdbid+'.pt')

        # get transform
        xformA = meta['asmb_xform%d'%assem[0]][assem[1]]
        xformB = meta['asmb_xform%d'%assem[2]][assem[3]]
        
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
    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa,\
           xyz.float(), mask, idx.long(), \
           xyz_t.float(), f1d_t.float(), mask_t, \
           xyz_prev.float(), mask_prev, \
           chain_idx, False, negative, torch.zeros(seq.shape), bond_feats, chirals,"compl", item


def loader_na_complex(item, Ls, params, native_NA_frac=0.25, negative=False, pick_top=True, random_noise=5.0):
    pdb_set = item[0]
    msa_id = item[1]

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

    idx = torch.arange(sum(Ls))
    idx[Ls[0]:] += CHAIN_GAP
    if (len(pdb_ids)==3):
        idx[Ls[1]:] += CHAIN_GAP

    chain_idx = torch.zeros((sum(Ls), sum(Ls))).long()
    chain_idx[:Ls[0], :Ls[0]] = 1
    chain_idx[Ls[0]:, Ls[0]:] = 1  # fd - "negatives" still predict DNA double helix
    bond_feats = torch.zeros((sum(Ls), sum(Ls))).long()
    bond_feats[:Ls[0], :Ls[0]] = get_protein_bond_feats(Ls[0])
    bond_feats[Ls[0]:, Ls[0]:] = get_protein_bond_feats(sum(Ls[1:]))

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
        chain_idx = chain_idx[sel][:,sel]
        bond_feats = bond_feats[sel][:,sel]

    xyz_prev = xyz_t[0].clone()
    mask_prev = mask_t[0].clone()
    chirals = torch.Tensor()
    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa,\
           xyz.float(), mask, idx.long(), \
           xyz_t.float(), f1d_t.float(), mask_t, \
           xyz_prev.float(), mask_prev, \
           chain_idx, False, negative, torch.zeros(seq.shape), bond_feats, chirals, "na_compl", item

def loader_rna(item, Ls, params, random_noise=5.0):
    # read PDBs
    pdb_ids = item.split(':')
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

    chain_idx = torch.ones(L,L).long()
    bond_feats = get_protein_bond_feats(L)

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
    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa,\
           xyz.float(), mask, idx.long(), \
           xyz_t.float(), f1d_t.float(), mask_t, \
           xyz_prev.float(), mask_prev, \
           chain_idx, False, False, torch.zeros(seq.shape), bond_feats.long(), chirals, "rna",item

def loader_sm_compl(item, params, pick_top=True,
    init_protein_tmpl=False, init_ligand_tmpl=False,
    init_protein_xyz=False, init_ligand_xyz=False, random_noise=5.0):
    """Load protein/SM complex with mixed residue and atom tokens. Also,
    compute frames for atom FAPE loss calc"""

    pdb_chain, pdb_hash, ligands = item

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
    xyz_prot, mask_prot = pdbA["xyz"], pdbA["mask"]
    protein_L, nprotatoms, _ = xyz_prot.shape
 
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

    # clamp number of symmetry variants to save GPU memory
    if xyz_sm.shape[0] > params['MAXNSYMM']: 
        xyz_sm = xyz_sm[:params['MAXNSYMM']]
        mask_sm = mask_sm[:params['MAXNSYMM']]

    if xyz_sm.shape[0] ==0:
        print(f'ERROR [loader_sm_compl]: {item[0]} had no xyz coords')
        return (torch.tensor([-1]),)*21 

    a3m_sm = {"msa": msa_sm.unsqueeze(0), "ins": ins_sm.unsqueeze(0)}
    G = get_nxgraph(mol)
    frames = get_atom_frames(msa_sm, G)
    chirals = get_chirals(mol, xyz_sm[0])

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
    
    chain_idx = torch.zeros((sum(Ls), sum(Ls))).long()
    chain_idx[:Ls[0], :Ls[0]] = 1
    chain_idx[Ls[0]:, Ls[0]:] = 1 
    bond_feats = torch.zeros((sum(Ls), sum(Ls))).long()
    bond_feats[:Ls[0], :Ls[0]] = get_protein_bond_feats(Ls[0])
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
        sel = crop_sm(xyz_prot, xyz_sm[0], Ls, params)
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
    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa,\
           xyz.float(), mask, idx.long(), \
           xyz_t.float(), f1d_t.float(), mask_t, \
           xyz_prev.float(), mask_prev, \
           chain_idx, False, False, frames, bond_feats, chirals, "sm_compl", item


def loader_atomize_pdb(item, params, homo, n_res_atomize, flank, unclamp=False, 
    pick_top=True, p_homo_cut=0.5, random_noise=5.0):
    """ load pdb with portions represented as atoms instead of residues """
    pdb = torch.load(params['PDB_DIR']+'/torch/pdb/'+item[0][1:3]+'/'+item[0]+'.pt')
    a3m = get_msa(params['PDB_DIR'] + '/a3m/' + item[1][:3] + '/' + item[1] + '.a3m.gz', item[1])
    tplt = torch.load(params['PDB_DIR']+'/torch/hhr/'+item[1][:3]+'/'+item[1]+'.pt')
    
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
                  f'fully-resolved residues to atomize. {item[0]} i_start={i_start}')
            break

    try:
        msa_sm, ins_sm, xyz_sm, mask_sm, frames, bond_feats_sm, last_C, chirals = atomize_protein(i_start, msa_prot, xyz_prot, mask_prot, n_res_atomize=n_res_atomize)
    except Exception as e:
        print('atomize_protein failed on ', item, 'with n_res_atomize', n_res_atomize,', flank',flank, ', i_start',i_start)
        raise e
        
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
    chain_idx = torch.zeros((sum(Ls), sum(Ls))).long()
    chain_idx[:Ls[0], :Ls[0]] = 1
    chain_idx[Ls[0]:, Ls[0]:] = 1 
    
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
    bond_feats = torch.cat((bond_feats[ :i1], bond_feats[i2:]), dim=0)
    bond_feats = torch.cat((bond_feats[ :, :i1], bond_feats[:, i2:]), dim=1)

    xyz_prev = xyz_t[0].clone()
    xyz_prev[Ls[0]:] = xyz_prev[i_start]
    mask_prev = mask_t[0].clone()
    xyz = torch.nan_to_num(xyz)
    if chirals.shape[0]>0:
        L1 = chain_idx[0,:].sum()
        chirals[:, :-1] = chirals[:, :-1] +L1
    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa,\
           xyz.float(), mask, idx.long(), \
           xyz_t.float(), f1d_t.float(), mask_t, \
           xyz_prev.float(), mask_prev, \
           chain_idx, False, False, frames, bond_feats,chirals,"atomize_pdb", item

def loader_sm(item, params, pick_top=True):
    """Load small molecule with atom tokens. Also, compute frames for atom FAPE loss calc"""
    # Load small molecule
    fname = params['CSD_DIR']+'/torch/'+item[:2]+'/'+item+'.pt'
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
    
    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa,\
           xyz.float(), mask, idx.long(), \
           xyz_t.float(), f1d_t.float(), mask_t, \
           xyz_prev.float(), mask_prev, \
           chain_idx, False, False, frames, bond_feats, chirals, "sm_only", item

def crop_sm(prot_xyz, lig_xyz,Ls, params):
    """choose residues with calphas close to a random ligand atom"""
    # ligand_com = torch.nanmean(lig_xyz, dim=[0,1]).expand(1,3)
    i_face_xyz = lig_xyz[np.random.randint(len(lig_xyz))]
    dist = torch.cdist(prot_xyz[:,1].unsqueeze(0), i_face_xyz.unsqueeze(0)).flatten()
    dist = torch.nan_to_num(dist, nan=999999)
    _, idx = torch.topk(dist, params["CROP"]-len(lig_xyz), largest=False)
    sel, _ = torch.sort(idx)
    # select the whole ligand
    lig_sel = torch.arange(lig_xyz.shape[0])+Ls[0]
    return torch.cat((sel, lig_sel))


class Dataset(data.Dataset):
    def __init__(self, IDs, loader, item_dict, params, homo, unclamp_cut=0.9, pick_top=True, p_homo_cut=-1.0, n_res_atomize=0, flank=0, seed=None):
        self.IDs = IDs
        self.item_dict = item_dict
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
        sel_idx = self.rng.randint(0, len(self.item_dict[ID]))
        p_unclamp = self.rng.rand()
        kwargs = dict()
        if self.n_res_atomize > 0:
            kwargs['n_res_atomize'] = self.n_res_atomize
            kwargs['flank'] = self.flank
        if p_unclamp > self.unclamp_cut:
            out = self.loader(self.item_dict[ID][sel_idx][0], self.params, self.homo,
                              unclamp=True, 
                              pick_top=self.pick_top, 
                              p_homo_cut=self.p_homo_cut,
                              **kwargs)
        else:
            out = self.loader(self.item_dict[ID][sel_idx][0], self.params, self.homo, 
                              pick_top=self.pick_top,
                              p_homo_cut=self.p_homo_cut,
                              **kwargs)
        return out

class DatasetComplex(data.Dataset):
    def __init__(self, IDs, loader, item_dict, params, pick_top=True, negative=False, seed=None):
        self.IDs = IDs
        self.item_dict = item_dict
        self.loader = loader
        self.params = params
        self.pick_top = pick_top
        self.negative = negative
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        sel_idx = self.rng.randint(0, len(self.item_dict[ID]))
        out = self.loader(self.item_dict[ID][sel_idx][0],
                          self.item_dict[ID][sel_idx][1],
                          self.item_dict[ID][sel_idx][2],
                          self.item_dict[ID][sel_idx][3],
                          self.params,
                          pick_top = self.pick_top,
                          negative = self.negative)
        return out

class DatasetNAComplex(data.Dataset):
    def __init__(self, IDs, loader, item_dict, params, pick_top=True, negative=False, native_NA_frac=0.0, seed=None):
        self.IDs = IDs
        self.item_dict = item_dict
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
        sel_idx = self.rng.randint(0, len(self.item_dict[ID]))
        out = self.loader(
                self.item_dict[ID][sel_idx][0],
            self.item_dict[ID][sel_idx][1],
            self.params,
            pick_top = self.pick_top,
            negative = self.negative,
            native_NA_frac = self.native_NA_frac
        )
        return out

class DatasetRNA(data.Dataset):
    def __init__(self, IDs, loader, item_dict, params, seed=None):
        self.IDs = IDs
        self.item_dict = item_dict
        self.loader = loader
        self.params = params
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        sel_idx = self.rng.randint(0, len(self.item_dict[ID]))
        out = self.loader(
            self.item_dict[ID][sel_idx][0],
            self.item_dict[ID][sel_idx][1],
            self.params
        )
        return out


class DatasetSMComplex(data.Dataset):
    def __init__(self, IDs, loader, item_dict, params, init_protein_tmpl=False, init_ligand_tmpl=False,
                 init_protein_xyz=False, init_ligand_xyz=False, task=None, seed=None):
        self.IDs = IDs
        self.item_dict = item_dict
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
        sel_idx = self.rng.randint(0, len(self.item_dict[ID])) # no weighting of samples during validation
        out = self.loader(
            self.item_dict[ID][sel_idx][0],
            self.params,
            init_protein_tmpl = self.init_protein_tmpl,
            init_ligand_tmpl = self.init_ligand_tmpl,
            init_protein_xyz = self.init_protein_xyz,
            init_ligand_xyz = self.init_ligand_xyz
        )
        if self.task is not None:
            out = out[:-2]+(self.task,)+out[-1:] # custom task name
        return out

class DatasetSM(data.Dataset):
    def __init__(self, IDs, loader, item_dict, params):
        self.IDs = IDs
        self.item_dict = item_dict
        self.loader = loader
        self.params = params

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        out = self.loader(
            self.item_dict[ID][0],
            self.params
        )
        return out

class DistilledDataset(data.Dataset):
    def __init__(
        self,
        pdb_IDs, pdb_loader, pdb_dict,
        compl_IDs, compl_loader, compl_dict,
        #neg_IDs, neg_loader, neg_dict,
        na_compl_IDs, na_compl_loader, na_compl_dict,
        #na_neg_IDs, na_neg_loader, na_neg_dict,
        fb_IDs, fb_loader, fb_dict,
        rna_IDs, rna_loader, rna_dict,
        sm_compl_IDs, sm_compl_loader, sm_compl_dict, 
        sm_IDs, sm_loader, sm_dict,
        atomize_pdb_IDs, atomize_pdb_loader, atomize_pdb_dict,
        homo, 
        params,
        native_NA_frac=0.25,
        unclamp_cut=0.9,
    ):
        #
        self.pdb_IDs = pdb_IDs
        self.pdb_dict = pdb_dict
        self.pdb_loader = pdb_loader
        self.compl_IDs = compl_IDs
        self.compl_loader = compl_loader
        self.compl_dict = compl_dict
        #self.neg_IDs = neg_IDs
        #self.neg_loader = neg_loader
        #self.neg_dict = neg_dict
        self.na_compl_IDs = na_compl_IDs
        self.na_compl_loader = na_compl_loader
        self.na_compl_dict = na_compl_dict
        #self.na_neg_IDs = na_neg_IDs
        #self.na_neg_loader = na_neg_loader
        #self.na_neg_dict = na_neg_dict
        self.fb_IDs = fb_IDs
        self.fb_dict = fb_dict
        self.fb_loader = fb_loader
        self.rna_IDs = rna_IDs
        self.rna_dict = rna_dict
        self.rna_loader = rna_loader
        self.sm_compl_IDs = sm_compl_IDs
        self.sm_compl_loader = sm_compl_loader
        self.sm_compl_dict = sm_compl_dict
        self.sm_IDs = sm_IDs
        self.sm_loader = sm_loader
        self.sm_dict = sm_dict
        self.atomize_pdb_IDs = atomize_pdb_IDs
        self.atomize_pdb_dict = atomize_pdb_dict
        self.atomize_pdb_loader = atomize_pdb_loader

        self.homo = homo
        self.params = params
        self.unclamp_cut = unclamp_cut
        self.native_NA_frac = native_NA_frac

        self.pdb_inds = np.arange(len(self.pdb_IDs))
        self.fb_inds = np.arange(len(self.fb_IDs))
        self.compl_inds = np.arange(len(self.compl_IDs))
        #self.neg_inds = np.arange(len(self.neg_IDs))
        self.na_compl_inds = np.arange(len(self.na_compl_IDs))
        #self.na_neg_inds = np.arange(len(self.na_neg_IDs))
        self.rna_inds = np.arange(len(self.rna_IDs))
        self.sm_compl_inds = np.arange(len(self.sm_compl_IDs))
        self.sm_inds = np.arange(len(self.sm_IDs))
        self.atomize_pdb_inds = np.arange(len(self.atomize_pdb_IDs))

    def __len__(self):
        return (
            len(self.fb_inds)
            + len(self.pdb_inds)
            + len(self.compl_inds)
            + len(self.na_compl_inds)
            + len(self.rna_inds)
            + len(self.sm_compl_inds)
            + len(self.sm_inds)
            + len(self.atomize_pdb_inds)
            #+ len(self.neg_inds)
            #+ len(self.na_neg_inds)
        )

    # order:
    #    0            - nfb-1        = FB
    #    nfb          - nfb+npdb-1   = PDB
    #    "+npdb       - "+ncmpl-1    = COMPLEX
    #    "+ncmpl      - "+nneg-1     = COMPLEX NEGATIVES
    #    "+nneg       - "+nna_cmpl-1 = NA COMPLEX
    #    "+nna_cmpl   - "+nrna-1     = NA COMPLEX NEGATIVES
    #    "+nrna-1     - "nsm_compl-1 = RNA
    #    nsm_compl -1 -              = SM COMPLEX
    def __getitem__(self, index):
        p_unclamp = np.random.rand()

        if index < len(self.fb_inds):
            ID = self.fb_IDs[index]
            sel_idx = np.random.randint(0, len(self.fb_dict[ID]))
            out = self.fb_loader(self.fb_dict[ID][sel_idx][0], self.params, unclamp=(p_unclamp > self.unclamp_cut))
        offset = len(self.fb_inds)

        if index >= offset and index < offset + len(self.pdb_inds):
            ID = self.pdb_IDs[index-offset]
            sel_idx = np.random.randint(0, len(self.pdb_dict[ID]))
            out = self.pdb_loader(self.pdb_dict[ID][sel_idx][0], self.params, self.homo, unclamp=(p_unclamp > self.unclamp_cut))
        offset += len(self.pdb_inds)

        if index >= offset and index < offset + len(self.compl_inds):
            ID = self.compl_IDs[index-offset]
            sel_idx = np.random.randint(0, len(self.compl_dict[ID]))
            out = self.compl_loader(
                self.compl_dict[ID][sel_idx][0], 
                self.compl_dict[ID][sel_idx][1],
                self.compl_dict[ID][sel_idx][2], 
                self.compl_dict[ID][sel_idx][3], 
                self.params,
                negative=False
            )
        offset += len(self.compl_inds)

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

        if index >= offset and index < offset + len(self.na_compl_inds):
            ID = self.na_compl_IDs[index-offset]
            sel_idx = np.random.randint(0, len(self.na_compl_dict[ID]))
            out = self.na_compl_loader(
                self.na_compl_dict[ID][sel_idx][0],
                self.na_compl_dict[ID][sel_idx][1],
                self.params,
                negative=False,
                native_NA_frac=self.native_NA_frac
            )

        offset += len(self.na_compl_inds)

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

        if index >= offset and index < offset + len(self.rna_inds):
            ID = self.rna_IDs[index-offset]
            sel_idx = np.random.randint(0, len(self.rna_dict[ID]))
            out = self.rna_loader(
                self.rna_dict[ID][sel_idx][0],
                self.rna_dict[ID][sel_idx][1],
                self.params
            )
        offset += len(self.rna_inds)

        if index >= offset and index < offset + len(self.sm_compl_inds):
            ID = self.sm_compl_IDs[index-offset]

            # choose one of 4 protein-sm tasks
            #i_task = np.random.randint(4)
            i_task = 0
            if i_task==0: # fold-and-dock
                kwargs = {}
                task = 'sm_compl_fold_dock'
            elif i_task==1: # rigid body dock
                kwargs = dict(
                    init_protein_tmpl = True, init_ligand_tmpl = True,
                )
                task = 'sm_compl_dock'
            elif i_task==2: # fold protein
                kwargs = dict(
                    init_ligand_tmpl = True, 
                )
                task = 'sm_compl_foldprot'
            elif i_task==3: # fold ligand
                kwargs = dict(
                    init_protein_tmpl = True, 
                )
                task = 'sm_compl_foldsm'

            weights = torch.tensor([item[2] for item in self.sm_compl_dict[ID]])
            sel_idx = torch.multinomial(weights, 1)
            out = self.sm_compl_loader(
                self.sm_compl_dict[ID][sel_idx][0],
                self.params,
                **kwargs
            )
            out = out[:-2]+(task,)+out[-1:]
        offset += len(self.sm_compl_inds)

        if index >= offset and index < offset + len(self.sm_inds):
            ID = self.sm_IDs[index-offset]
            out = self.sm_loader(
                self.sm_dict[ID][0],
                self.params,
            )
        offset += len(self.sm_inds)

        if index >= offset and index < offset + len(self.atomize_pdb_inds):
            ID = self.atomize_pdb_IDs[index-offset]
            sel_idx = np.random.randint(0, len(self.atomize_pdb_dict[ID]))
            n_res_atomize = np.random.randint(self.params['NRES_ATOMIZE_MIN'], 
                                              self.params['NRES_ATOMIZE_MAX']+1)
            out = self.atomize_pdb_loader(self.atomize_pdb_dict[ID][sel_idx][0], 
                self.params, self.homo, n_res_atomize, self.params['ATOMIZE_FLANK'], 
                unclamp=(p_unclamp > self.unclamp_cut))
        offset += len(self.atomize_pdb_inds)

        return out

class DistributedWeightedSampler(data.Sampler):
    def __init__(
        self,
        dataset,
        pdb_weights,
        fb_weights,
        compl_weights,
        #neg_weights,
        na_compl_weights,
        #neg_na_compl_weights,
        rna_weights,
        sm_compl_weights,
        sm_weights,
        atomize_pdb_weights,
        num_example_per_epoch=25600,
        fraction_pdb=0.125,
        fraction_fb=0.125,
        fraction_compl=0.125,  # half neg, half pos
        fraction_na_compl=0.125, # half neg, half pos
        fraction_rna=0.125,
        fraction_sm_compl=0.125, 
        fraction_sm=0.125,
        fraction_atomize_pdb=0.125,
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
        assert (fraction_pdb + fraction_fb + fraction_compl + fraction_na_compl + \
                fraction_rna + fraction_sm_compl + fraction_sm + fraction_atomize_pdb == 1.0)

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.num_fb_per_epoch = int(round(num_example_per_epoch*fraction_fb))
        self.num_compl_per_epoch = int(round(num_example_per_epoch*fraction_compl))
        #self.num_neg_per_epoch = 0 #manually set to 0
        self.num_na_compl_per_epoch = int(round(num_example_per_epoch*fraction_na_compl))
        #self.num_neg_na_compl_per_epoch = 0 # manually set to 0
        self.num_rna_per_epoch = int(round(num_example_per_epoch*fraction_rna))
        self.num_sm_compl_per_epoch = int(round(num_example_per_epoch*fraction_sm_compl))
        self.num_sm_per_epoch = int(round(num_example_per_epoch*fraction_sm))
        self.num_atomize_pdb_per_epoch = int(round(num_example_per_epoch*fraction_atomize_pdb))

        self.num_pdb_per_epoch = num_example_per_epoch - (
            self.num_fb_per_epoch 
            + self.num_compl_per_epoch
            + self.num_na_compl_per_epoch
            + self.num_rna_per_epoch
            + self.num_sm_compl_per_epoch
            + self.num_sm_per_epoch
            + self.num_atomize_pdb_per_epoch
        #    + self.num_neg_per_epoch
        #    + self.num_neg_na_compl_per_epoch
        )

        if (rank==0):
            print (
                "Per epoch:",
                self.num_pdb_per_epoch,"pdb,",
                self.num_fb_per_epoch,"fb,",
                self.num_compl_per_epoch,"compl,",
        #        self.num_neg_per_epoch,"neg,",
                self.num_na_compl_per_epoch,"NA compl,",
        #        self.num_neg_na_compl_per_epoch,"NA neg,",
                self.num_rna_per_epoch,"RNA,",
                self.num_sm_compl_per_epoch, "SM Compl,",
                self.num_sm_per_epoch, "SM crystals."
            )


        self.total_size = num_example_per_epoch
        self.num_samples = self.total_size // self.num_replicas
        self.rank = rank
        self.epoch = 0
        self.replacement = replacement

        self.pdb_weights = pdb_weights
        self.fb_weights = fb_weights
        self.compl_weights = compl_weights
        #self.neg_weights = neg_weights
        self.na_compl_weights = na_compl_weights
        #self.neg_na_compl_weights = neg_na_compl_weights
        self.rna_weights = rna_weights
        self.sm_compl_weights = sm_compl_weights
        self.sm_weights = sm_weights
        self.atomize_pdb_weights = atomize_pdb_weights

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        # get indices (fb + pdb models)
        indices = torch.arange(len(self.dataset))

        # weighted subsampling
        # order:
        #    0          - nfb-1        = FB
        #    nfb        - nfb+npdb-1   = PDB
        #    "+npdb     - "+ncmpl-1    = COMPLEX
        #    "+ncmpl    - "+nneg-1     = COMPLEX NEGATIVES
        #    "+nneg     - "+nna_cmpl-1 = NA COMPLEX
        #    "+nna_cmpl - "+nrna-1     = NA COMPLEX NEGATIVES
        #    "+nrna-1   -              = RNA
        offset = 0
        sel_indices = torch.tensor((),dtype=int)
        if (self.num_fb_per_epoch>0):
            fb_sampled = torch.multinomial(self.fb_weights, self.num_fb_per_epoch, self.replacement, generator=g)
            sel_indices = torch.cat((sel_indices, indices[fb_sampled]))
        offset += len(self.dataset.fb_IDs)

        if (self.num_pdb_per_epoch>0):
            pdb_sampled = torch.multinomial(self.pdb_weights, self.num_pdb_per_epoch, self.replacement, generator=g)
            sel_indices = torch.cat((sel_indices, indices[pdb_sampled + offset]))
        offset += len(self.dataset.pdb_IDs)

        if (self.num_compl_per_epoch>0):
            compl_sampled = torch.multinomial(self.compl_weights, self.num_compl_per_epoch, self.replacement, generator=g)
            sel_indices = torch.cat((sel_indices, indices[compl_sampled + offset]))
        offset += len(self.dataset.compl_IDs)
        
        #if (self.num_neg_per_epoch>0):
        #   neg_sampled = torch.multinomial(self.neg_weights, self.num_neg_per_epoch, self.replacement, generator=g)
        #   sel_indices = torch.cat((sel_indices, indices[neg_sampled + offset]))
        #offset += len(self.dataset.neg_IDs)

        if (self.num_na_compl_per_epoch>0):
            na_compl_sampled = torch.multinomial(self.na_compl_weights, self.num_na_compl_per_epoch, self.replacement, generator=g)
            sel_indices = torch.cat((sel_indices, indices[na_compl_sampled + offset]))
        offset += len(self.dataset.na_compl_IDs)

        #if (self.num_neg_na_compl_per_epoch>0):
        #   neg_na_sampled = torch.multinomial(self.neg_na_compl_weights, self.num_neg_na_compl_per_epoch, self.replacement, generator=g)
        #   sel_indices = torch.cat((sel_indices, indices[neg_na_sampled + offset]))
        #offset += len(self.dataset.na_neg_IDs)

        if (self.num_rna_per_epoch>0):
            rna_sampled = torch.multinomial(self.rna_weights, self.num_rna_per_epoch, self.replacement, generator=g)
            sel_indices = torch.cat((sel_indices, indices[rna_sampled + offset]))
        offset += len(self.dataset.rna_IDs)

        if (self.num_sm_compl_per_epoch>0):
            sm_compl_sampled = torch.multinomial(self.sm_compl_weights, self.num_sm_compl_per_epoch, self.replacement, generator=g)
            sel_indices = torch.cat((sel_indices, indices[sm_compl_sampled + offset]))
        offset += len(self.dataset.sm_compl_IDs)

        if (self.num_sm_per_epoch>0):
            sm_sampled = torch.multinomial(self.sm_weights, self.num_sm_per_epoch, self.replacement, generator=g)
            sel_indices = torch.cat((sel_indices, indices[sm_sampled + offset]))
        offset += len(self.dataset.sm_IDs)

        if (self.num_atomize_pdb_per_epoch>0):
            atomize_pdb_sampled = torch.multinomial(self.atomize_pdb_weights, self.num_atomize_pdb_per_epoch, self.replacement, generator=g)
            sel_indices = torch.cat((sel_indices, indices[atomize_pdb_sampled + offset]))
        offset += len(self.dataset.atomize_pdb_IDs)

        # shuffle indices
        indices = sel_indices[torch.randperm(len(sel_indices), generator=g)]

        # per each gpu
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

