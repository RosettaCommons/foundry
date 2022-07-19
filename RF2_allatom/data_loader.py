import torch
from torch.utils import data
import os
import csv
from dateutil import parser
import numpy as np
from parsers import parse_a3m, parse_pdb, parse_fasta_if_exists, parse_mol, get_ligand_xyz
from chemical import INIT_CRDS, INIT_NA_CRDS, NAATOKENS, MASKINDEX, NTOTAL, NBTYPES
from util import get_nxgraph, get_atom_frames, get_bond_feats, get_protein_bond_feats
import pickle
import random
import ast

from scipy.sparse.csgraph import shortest_path

base_dir = "/projects/ml/TrRosetta/PDB-2021AUG02"
compl_dir = "/projects/ml/RoseTTAComplex"
#na_dir = "/projects/ml/nucleic"
na_dir = "/home/dimaio/TrRosetta/nucleic"
fb_dir = "/projects/ml/TrRosetta/fb_af"
mol_dir = "/projects/ml/ligand_datasets/mmcif_parse_wlig"

if not os.path.exists(base_dir):
    # training on AWS
    #base_dir = "/gscratch2/PDB-2021AUG02"
    #compl_dir = "/gscratch2/RoseTTAComplex"
    #na_dir = "/gscratch2/nucleic"
    #fb_dir = "/gscratch2/fb_af1"
    base_dir = "/data/databases/PDB-2021AUG02"
    fb_dir = "/data/databases/fb_af"
    compl_dir = "/data/databases/RoseTTAComplex"
    mol_dir = "/home/rohith"

def set_data_loader_params(args):
    PARAMS = {
        "COMPL_LIST"       : "%s/list.hetero.csv"%compl_dir,
        "HOMO_LIST"        : "%s/list.homo.csv"%compl_dir,
        "NEGATIVE_LIST"    : "%s/list.negative.csv"%compl_dir,
        "RNA_LIST"         : "%s/list.rnaonly.csv"%na_dir,
        "NA_COMPL_LIST"    : "%s/list.nucleic.csv"%na_dir,
        "NEG_NA_COMPL_LIST": "%s/list.na_negatives.csv"%na_dir,
        "SM_LIST"          : "%s/list_v02_ligonly_notest.csv"%base_dir, 
        "PDB_LIST"         : "%s/list_v02.csv"%base_dir, # on digs
        #"PDB_LIST"        : "/gscratch2/list_2021AUG02.csv", # on blue
        "FB_LIST"          : "%s/list_b1-3.csv"%fb_dir,
        "VAL_PDB"          : "./valid_remapped",
        "VAL_RNA"          : "%s/rna_valid.csv"%na_dir,
        "VAL_COMPL"        : "%s/val_lists/xaa"%compl_dir,
        "VAL_NEG"          : "%s/val_lists/xaa.neg"%compl_dir,
        "TEST_SM"          : "./lig_test",
        "DATAPKL"          : "./dataset.pkl", # cache for faster loading
        "PDB_DIR"          : base_dir,
        "FB_DIR"           : fb_dir,
        "COMPL_DIR"        : compl_dir,
        "NA_DIR"           : na_dir,
        "MOL_DIR"          : mol_dir, 
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
        "MAXCYCLE"         : 4
    }
    for param in PARAMS:
        if hasattr(args, param.lower()):
            PARAMS[param] = getattr(args, param.lower())
    return PARAMS

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

def TemplFeaturize(tplt, qlen, params, offset=0, npick=1, pick_top=True):
    seqID_cut = params['SEQID']

    ntplt = len(tplt['ids'])
    if (ntplt < 1) or (npick < 1): #no templates in hhsearch file or not want to use templ
        xyz = torch.full((1, qlen, NTOTAL, 3), np.nan).float()
        t1d = torch.nn.functional.one_hot(
            torch.full((1, qlen), 20).long(), num_classes=NAATOKENS-1).float() # all gaps (no mask token)
        conf = torch.zeros((1, qlen, 1)).float()
        t1d = torch.cat((t1d, conf), -1)
        return xyz, t1d
    
    # ignore templates having too high seqID
    if seqID_cut <= 100.0:
        sel = torch.where(tplt['f0d'][0,:,4] < seqID_cut)[0]
        tplt['ids'] = np.array(tplt['ids'])[sel]
        tplt['qmap'] = tplt['qmap'][:,sel]
        tplt['xyz'] = tplt['xyz'][:, sel]
        tplt['seq'] = tplt['seq'][:, sel]
        tplt['f1d'] = tplt['f1d'][:, sel]
    
    # check again if there are templates having seqID < cutoff
    ntplt = len(tplt['ids'])
    npick = min(npick, ntplt)
    if npick<1: # no templates
        xyz = torch.full((1,qlen,NTOTAL,3),np.nan).float()
        t1d = torch.nn.functional.one_hot(
            torch.full((1, qlen), 20).long(), num_classes=NAATOKENS-1).float() # all gaps (no mask token)
        conf = torch.zeros((1, qlen, 1)).float()
        t1d = torch.cat((t1d, conf), -1)
        return xyz, t1d

    if not pick_top: # select randomly among all possible templates
        sample = torch.randperm(ntplt)[:npick]
    else: # only consider top 50 templates
        sample = torch.randperm(min(50,ntplt))[:npick]

    xyz = torch.full((npick,qlen,NTOTAL,3),np.nan).float()
    mask = torch.full((npick,qlen,NTOTAL),False)
    t1d = torch.full((npick, qlen), 20).long()  # all gaps
    t1d_val = torch.zeros((npick, qlen)).float()

    for i,nt in enumerate(sample):
        ntmplatoms = tplt['xyz'].shape[2] # will be bigger for NA templates
        sel = torch.where(tplt['qmap'][0,:,1]==nt)[0]
        pos = tplt['qmap'][0,sel,0] + offset
        xyz[i,pos,:ntmplatoms] = tplt['xyz'][0,sel]
        mask[i,pos,:ntmplatoms] = tplt['mask'][0,sel]
        # 1-D features: alignment confidence 
        t1d[i,pos] = tplt['seq'][0,sel]
        t1d_val[i,pos] = tplt['f1d'][0,sel,2] # alignment confidence

    t1d = torch.nn.functional.one_hot(t1d, num_classes=NAATOKENS-1).float() # (no mask token)
    t1d = torch.cat((t1d, t1d_val[...,None]), dim=-1)

    xyz = torch.where(mask[...,None], xyz.float(),torch.full((npick,qlen,NTOTAL,3),np.nan).float())

    return xyz, t1d


def get_train_valid_set(params, OFFSET=1000000):
    if (not os.path.exists(params['DATAPKL'])):
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

        with open(params["SM_LIST"], 'r') as f:
            reader = csv.reader(f)
            next(reader)
            rows = [[r[0],r[3],int(r[4]), int(r[6]), ast.literal_eval(r[-2].strip())] for r in reader
                            if float(r[2])<=params['RESCUT'] and
                            parser.parse(r[1])<=parser.parse(params['DATCUT'])]

        train_sm_compl = {}
        valid_sm_compl = {}
        for r in rows:
            if r[2] in val_pdb_ids:
                if r[2] in valid_sm_compl.keys():
                    valid_sm_compl[r[2]].append((r[:2], r[3], r[-1]))
                else:
                    valid_sm_compl[r[2]] = [(r[:2], r[3], r[-1])]
            else:
                if r[2] in train_sm_compl.keys():
                    train_sm_compl[r[2]].append((r[:2], r[3], r[-1]))
                else:
                    train_sm_compl[r[2]] = [(r[:2], r[3], r[-1])]

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
            if r[2] in val_pdb_ids or r[2] in test_sm_ids:
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

        # Get average chain length in each cluster and calculate weights
        pdb_IDs = list(train_pdb.keys())
        fb_IDs = list(fb.keys())
        compl_IDs = list(train_compl.keys())
        neg_IDs = list(train_neg.keys())
        na_compl_IDs = list(train_na_compl.keys())
        na_neg_IDs = list(train_na_neg.keys())
        rna_IDs = list(train_rna.keys())
        sm_compl_IDs = list(train_sm_compl.keys())

        #
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
        sm_compl_weights = np.array([train_sm_compl[key][0][1] for key in sm_compl_IDs])
        sm_compl_weights = (1/512.)*np.clip(sm_compl_weights, 256, 512)  

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
            valid_pdb, valid_homo, 
            valid_compl, valid_neg,
            valid_na_compl, valid_na_neg,
            valid_rna, valid_sm_compl,
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
                valid_pdb, valid_homo, 
                valid_compl, valid_neg,
                valid_na_compl, valid_na_neg,
                valid_rna, valid_sm_compl,
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
        (sm_compl_IDs, torch.tensor(sm_compl_weights).float(), train_sm_compl),
        valid_pdb, valid_homo, 
        valid_compl, valid_neg,
        valid_na_compl, valid_na_neg,
        valid_rna, valid_sm_compl,
        homo
    )


# slice long chains
def get_crop(l, mask, device, params, unclamp=False):

    sel = torch.arange(l,device=device)
    if l <= params['CROP']:
        return sel

    size = params['CROP']

    mask = ~(mask[:,:3].sum(dim=-1) < 3.0)
    exists = mask.nonzero()[0]
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

def get_spatial_crop(xyz, mask, sel, len_s, params, cutoff=10.0, eps=1e-6):
    device = xyz.device

    # get interface residues
    #   interface defined as chain 1 versus all other chains
    cond = torch.cdist(xyz[:len_s[0],1], xyz[len_s[0]:,1]) < cutoff
    cond = torch.logical_and(cond, mask[:len_s[0],None,1]*mask[None,len_s[0]:,1]) 
    i,j = torch.where(cond)
    ifaces = torch.cat([i,j+len_s[0]])
    if len(ifaces) < 1:
        print ("ERROR: no iface residue????")
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
def featurize_single_chain(msa, ins, tplt, pdb, params, unclamp=False, pick_top=True):
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params)
    
    # get template features
    ntempl = np.random.randint(params['MINTPLT'], params['MAXTPLT']+1)
    xyz_t,f1d_t = TemplFeaturize(tplt, msa.shape[1], params, npick=ntempl, offset=0, pick_top=pick_top)
    
    # get ground-truth structures
    idx = torch.arange(len(pdb['xyz'])) 
    xyz = torch.full((len(idx),NTOTAL,3),np.nan).float()
    xyz[:,:14,:] = pdb['xyz']
    mask = torch.full((len(idx), NTOTAL), False)
    mask[:,:14] = pdb['mask']

    # Residue cropping
    crop_idx = get_crop(len(idx), mask, msa_seed_orig.device, params, unclamp=unclamp)
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

    # get initial coordinates
    xyz_prev = xyz_t[0]
    chain_idx = torch.ones((len(crop_idx), len(crop_idx))).long()
    bond_feats = get_protein_bond_feats(len(crop_idx)).long()
    bond_feats = torch.nn.functional.one_hot(bond_feats, num_classes=NBTYPES)
    # replace missing with blackholes & conovert NaN to zeros to avoid any NaN problems during loss calculation
    init = INIT_CRDS.reshape(1, NTOTAL, 3).repeat(len(xyz), 1, 1)
    xyz = torch.where(mask[...,None], xyz, init).contiguous()
    xyz = torch.nan_to_num(xyz)
    
    #print ("loader_single", mask.shape, xyz_t.shape, f1d_t.shape, xyz_prev.shape)

    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa, \
           xyz.float(), mask, idx.long(),\
           xyz_t.float(), f1d_t.float(), xyz_prev.float(), \
           chain_idx, unclamp, False, torch.zeros(seq.shape), bond_feats

# Generate input features for homo-oligomers
def featurize_homo(msa_orig, ins_orig, tplt, pdbA, pdbid, interfaces, params, pick_top=True):
    L = msa_orig.shape[1]
    
    msa, ins = merge_a3m_homo(msa_orig, ins_orig, 2) # make unpaired alignments, for training, we always use two chains
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params, nmer=2, L_s=[L,L])

    # get template features
    ntempl = np.random.randint(params['MINTPLT'], params['MAXTPLT']//2+1)
    xyz_t_single, f1d_t_single = TemplFeaturize(tplt, L, params, npick=ntempl, offset=0, pick_top=pick_top)
    ntempl = max(1, ntempl)
    # duplicate
    xyz_t = torch.full((2*ntempl, L*2, NTOTAL, 3), np.nan).float()
    f1d_t = torch.full((2*ntempl, L*2), 20).long()
    f1d_t = torch.cat((torch.nn.functional.one_hot(f1d_t, num_classes=NAATOKENS-1).float(), torch.zeros((2*ntempl, L*2, 1)).float()), dim=-1)
    xyz_t[:ntempl,:L] = xyz_t_single
    xyz_t[ntempl:,L:] = xyz_t_single
    f1d_t[:ntempl,:L] = f1d_t_single
    f1d_t[ntempl:,L:] = f1d_t_single
    
    # get initial coordinates
    xyz_prev = torch.cat((xyz_t_single[0], xyz_t_single[0]), dim=0)

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

    idx = torch.arange(L*2)
    idx[L:] += 200 # to let network know about chain breaks

    # indicator for which residues are in same chain
    chain_idx = torch.zeros((2*L, 2*L)).long()
    chain_idx[:L, :L] = 1
    chain_idx[L:, L:] = 1
    bond_feats = torch.zeros((2*L, 2*L)).long()
    bond_feats[:L, :L] = get_protein_bond_feats(L)
    bond_feats[L:, L:] = get_protein_bond_feats(L)

    # Residue cropping
    if 2*L > params['CROP']:
        # crop so there are contacts in AT LEAST ONE of the interfaces
        spatial_crop_tgt = np.random.randint(0, npairs)
        crop_idx = get_spatial_crop(
            xyz[spatial_crop_tgt], mask[spatial_crop_tgt], torch.arange(L*2), [L,L], params)
        seq = seq[:,crop_idx]
        msa_seed_orig = msa_seed_orig[:,:,crop_idx]
        msa_seed = msa_seed[:,:,crop_idx]
        msa_extra = msa_extra[:,:,crop_idx]
        mask_msa = mask_msa[:,:,crop_idx]
        xyz_t = xyz_t[:,crop_idx]
        f1d_t = f1d_t[:,crop_idx]
        xyz = xyz[:,crop_idx]
        mask = mask[:,crop_idx]
        idx = idx[crop_idx]
        chain_idx = chain_idx[crop_idx][:,crop_idx]
        bond_feats = bond_feats[crop_idx][:,crop_idx]
        xyz_prev = xyz_prev[crop_idx]

    bond_feats = torch.nn.functional.one_hot(bond_feats, num_classes=NBTYPES)
    # replace missing with blackholes & conovert NaN to zeros to avoid any NaN problems during loss calculation
    init = INIT_CRDS.reshape(1, 1, NTOTAL, 3).repeat(npairs, xyz.shape[1], 1, 1)

    xyz = torch.where(mask[...,None], xyz, init).contiguous()
    xyz = torch.nan_to_num(xyz)

    #print ("loader_homo", mask.shape, xyz_t.shape, f1d_t.shape, xyz_prev.shape)

    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa, \
           xyz.float(), mask, idx.long(),\
           xyz_t.float(), f1d_t.float(), xyz_prev.float(), \
           chain_idx, False, False, torch.zeros(seq.shape), bond_feats


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
            # choose one from all possible dimer copies of original homomers
            #sel_idx = np.random.randint(0, len(homo[item[0]]))
            #homo_item = homo[item[0]][sel_idx]
            interfaces = homo[item[0]]
            feats = featurize_homo(msa, ins, tplt, pdb, pdbid, interfaces, params, pick_top=pick_top)
            return feats
        else:
            return featurize_single_chain(msa, ins, tplt, pdb, params, unclamp=unclamp, pick_top=pick_top)
    else:
        return featurize_single_chain(msa, ins, tplt, pdb, params, unclamp=unclamp, pick_top=pick_top)
    
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
    crop_idx = get_crop(len(idx), mask, msa_seed_orig.device, params, unclamp=unclamp)
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
    xyz_prev = xyz_t[0]
    chain_idx = torch.ones((len(crop_idx), len(crop_idx))).long()
    bond_feats = get_protein_bond_feats(len(crop_idx)).long()
    bond_feats = torch.nn.functional.one_hot(bond_feats, num_classes=NBTYPES)

    #print ("loader_fb", mask.shape, xyz_t.shape, f1d_t.shape, xyz_prev.shape)

    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa, \
           xyz.float(), mask, idx.long(),\
           xyz_t.float(), f1d_t.float(), xyz_prev.float(), \
           chain_idx, unclamp, False, torch.zeros(seq.shape), bond_feats


def loader_complex(item, L_s, taxID, assem, params, negative=False, pick_top=True):
    pdb_pair = item[0]
    pMSA_hash = item[1]
    
    msaA_id, msaB_id = pMSA_hash.split('_')
    if len(set(taxID.split(':'))) == 1: # two proteins have same taxID -- use paired MSA
        # read pMSA
        if negative:
            pMSA_fn = params['COMPL_DIR'] + '/pMSA.negative/' + msaA_id[:3] + '/' + msaB_id[:3] + '/' + pMSA_hash + '.a3m'
        else:
            pMSA_fn = params['COMPL_DIR'] + '/pMSA/' + msaA_id[:3] + '/' + msaB_id[:3] + '/' + pMSA_hash + '.a3m'
        a3m = get_msa(pMSA_fn, pMSA_hash, unzip=False)
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
    ntempl = np.random.randint(params['MINTPLT'], params['MAXTPLT']//2+1)
    xyz_t_A, f1d_t_A = TemplFeaturize(tpltA, sum(L_s), params, offset=0, npick=ntempl, pick_top=pick_top) 
    xyz_t_B, f1d_t_B = TemplFeaturize(tpltB, sum(L_s), params, offset=L_s[0], npick=ntempl, pick_top=pick_top) 
    xyz_t = torch.cat((xyz_t_A, xyz_t_B), dim=0)
    f1d_t = torch.cat((f1d_t_A, f1d_t_B), dim=0)

    # get initial coordinates
    xyz_prev = torch.cat((xyz_t_A[0][:L_s[0]], xyz_t_B[0][L_s[0]:]), dim=0)

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
    idx = torch.arange(sum(L_s))
    idx[L_s[0]:] += 200

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
            sel = get_spatial_crop(xyz, mask, torch.arange(sum(L_s)), L_s, params)
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
        xyz_prev = xyz_prev[sel]
        #
        idx = idx[sel]
        chain_idx = chain_idx[sel][:,sel]
        bond_feats = bond_feats[sel][:,sel]
    bond_feats = torch.nn.functional.one_hot(bond_feats, num_classes=NBTYPES)

    # replace missing with blackholes & conovert NaN to zeros to avoid any NaN problems during loss calculation
    init = INIT_CRDS.reshape(1, NTOTAL, 3).repeat(len(xyz), 1, 1)
    xyz = torch.where(mask[...,None], xyz, init).contiguous()
    xyz = torch.nan_to_num(xyz)
    
    #print ("loader_compl", mask.shape, xyz_t.shape, f1d_t.shape, xyz_prev.shape, negative)

    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa,\
           xyz.float(), mask, idx.long(), \
           xyz_t.float(), f1d_t.float(), xyz_prev.float(), \
           chain_idx, False, negative, torch.zeros(seq.shape), bond_feats

def loader_na_complex(item, Ls, params, native_NA_frac=0.25, negative=False, pick_top=True):
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
    #alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-0acgtxbdhuy"), dtype='|S1').view(np.uint8) # -0 are UNK/mask
    #if (len(pdb_ids)==2):
    #    a3mB = np.array([list(pdbB['seq'])], dtype='|S1').view(np.uint8)
    #else:
    #    a3mB = np.array([list(pdbB['seq']+pdbC['seq'])], dtype='|S1').view(np.uint8)  # separate entries?
    #for i in range(alphabet.shape[0]):
    #    a3mB[a3mB == alphabet[i]] = i
    #a3mB = {
    #    'msa':torch.from_numpy(a3mB),
    #    'ins':torch.zeros(a3mB.shape, dtype=torch.uint8),
    #}
    msaB,insB = parse_fasta_if_exists(pdbB['seq'], params['NA_DIR']+'/torch/'+pdb_ids[1][1:3]+'/'+pdb_ids[1]+'.afa', rmsa_alphabet=True)
    a3mB = {'msa':torch.from_numpy(msaB), 'ins':torch.from_numpy(insB)}
    if (len(pdb_ids)==3):
        msaC,insC = parse_fasta_if_exists(pdbC['seq'], params['NA_DIR']+'/torch/'+pdb_ids[2][1:3]+'/'+pdb_ids[2]+'.afa', rmsa_alphabet=True)
        a3mC = {'msa':torch.from_numpy(msaC), 'ins':torch.from_numpy(insC)}
        a3mB = merge_a3m_hetero(a3mB, a3mC, Ls[1:])
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
    xyz_t, f1d_t = TemplFeaturize(tpltA, sum(Ls), params, offset=0, npick=ntempl, pick_top=pick_top) 

    xyz_prev = xyz_t[0]

    if (np.random.rand()<=native_NA_frac):
        natNA_templ = pdbB['xyz']
        if pdbC is not None:
            natNA_templ = torch.cat((pdbB['xyz'], pdbC['xyz']), dim=0)

        # construct template from NA
        xyz_t_B = torch.full((1,sum(Ls),NTOTAL,3),np.nan).float()
        xyz_t_B[:,Ls[0]:sum(Ls),:23] = natNA_templ
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

        xyz_prev = xyz_t_B[0] # initialize NA only

    # get MSA features
    msa = a3m['msa'].long()
    ins = a3m['ins'].long()
    if len(msa) > params['BLOCKCUT']:
        msa, ins = MSABlockDeletion(msa, ins)
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params, L_s=Ls)

    xyz = torch.full((sum(Ls), NTOTAL, 3), np.nan).float()
    mask = torch.full((sum(Ls), NTOTAL), False)

    if (len(pdb_ids)==3):
        xyz[:Ls[0],:14] = pdbA['xyz']
        xyz[Ls[0]:,:23] = torch.cat((pdbB['xyz'], pdbC['xyz']), dim=0)
        mask[:Ls[0],:14] = pdbA['mask']
        mask[Ls[0]:,:23] = torch.cat((pdbB['mask'], pdbC['mask']), dim=0)
    else:
        xyz[:Ls[0],:14] = pdbA['xyz']
        xyz[Ls[0]:,:23] = pdbB['xyz']
        mask[:Ls[0],:14] = pdbA['mask']
        mask[Ls[0]:,:23] = pdbB['mask']

    idx = torch.arange(sum(Ls))
    idx[Ls[0]:] += 200
    if (len(pdb_ids)==3):
        idx[Ls[1]:] += 200

    chain_idx = torch.zeros((sum(Ls), sum(Ls))).long()
    chain_idx[:Ls[0], :Ls[0]] = 1
    chain_idx[Ls[0]:, Ls[0]:] = 1  # fd - "negatives" still predict DNA double helix
    bond_feats = torch.zeros((sum(Ls), sum(Ls))).long()
    bond_feats[:Ls[0], :Ls[0]] = get_protein_bond_feats(L_s[0])
    bond_feats[Ls[0]:, Ls[0]:] = get_protein_bond_feats(sum(Ls[1:]))

    init = torch.cat((
        INIT_CRDS.reshape(1, NTOTAL, 3).repeat(Ls[0], 1, 1),
        INIT_NA_CRDS.reshape(1, NTOTAL, 3).repeat(sum(Ls[1:]), 1, 1)
    ), dim=0)

    # Do cropping
    #print (item)
    if sum(Ls) > params['CROP']:
        sel = get_na_crop(seq[0], xyz, mask, torch.arange(sum(Ls)), Ls, params, negative)

        seq = seq[:,sel]
        msa_seed_orig = msa_seed_orig[:,:,sel]
        msa_seed = msa_seed[:,:,sel]
        msa_extra = msa_extra[:,:,sel]
        mask_msa = mask_msa[:,:,sel]
        xyz = xyz[sel]
        mask = mask[sel]
        xyz_t = xyz_t[:,sel]
        f1d_t = f1d_t[:,sel]
        xyz_prev = xyz_prev[sel]
        #
        idx = idx[sel]
        chain_idx = chain_idx[sel][:,sel]
        bond_feats = bond_feats[sel][:,sel]
        init = init[sel]
    bond_feats = torch.nn.functional.one_hot(bond_feats, num_classes=NBTYPES)
    # replace missing with blackholes & conovert NaN to zeros to avoid any NaN problems during loss calculation
    xyz = torch.where(mask[...,None], xyz, init).contiguous()
    xyz = torch.nan_to_num(xyz)

    #print ("loader_na_complex", mask.shape, xyz_t.shape, f1d_t.shape, xyz_prev.shape)

    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa,\
           xyz.float(), mask, idx.long(), \
           xyz_t.float(), f1d_t.float(), xyz_prev.float(), \
           chain_idx, False, negative, torch.zeros(seq.shape), bond_feats

def loader_rna(pdb_set, Ls, params):
    # read PDBs
    pdb_ids = pdb_set.split(':')
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
    xyz_t = torch.full((1,L,NTOTAL,3),np.nan).float()
    f1d_t = torch.nn.functional.one_hot(torch.full((1, L), 20).long(), num_classes=NAATOKENS-1).float() # all gaps
    conf = torch.zeros((1,L,1)).float() # zero confidence
    f1d_t = torch.cat((f1d_t, conf), -1)

    xyz_prev = xyz_t[0]

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
        idx[Ls[0]:] += 200

    chain_idx = torch.ones(L,L).long()
    bond_feats = get_protein_bond_feats(L)
    init = INIT_NA_CRDS.reshape(1, NTOTAL, 3).repeat(L, 1, 1)

    # Do cropping
    #print (item)
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
        xyz_prev = xyz_prev[sel]
        #
        idx = idx[sel]
        chain_idx = chain_idx[sel][:,sel]
        bond_feats = bond_feats[sel][:, sel]
        init = init[sel]
    bond_feats = torch.nn.functional.one_hot(bond_feats, num_classes=NBTYPES)
    # replace missing with blackholes & conovert NaN to zeros to avoid any NaN problems during loss calculation
    xyz = torch.where(mask[...,None], xyz, init).contiguous()
    xyz = torch.nan_to_num(xyz)

    #print ("loader_rna", mask.shape, xyz_t.shape, f1d_t.shape, xyz_prev.shape)

    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa,\
           xyz.float(), mask, idx.long(), \
           xyz_t.float(), f1d_t.float(), xyz_prev.float(), \
           chain_idx, False, False, torch.zeros(seq.shape), bond_feats

def loader_sm_compl(item, sm_chains, params, pick_top=True):
    """Load protein/SM complex with mixed residue and atom tokens. Also, compute frames for atom FAPE loss calc"""
    # Load protein information
    pdbA = torch.load(params['PDB_DIR']+'/torch/pdb/'+item[0][1:3]+'/'+item[0]+'.pt')
    a3mA = get_msa(params['PDB_DIR'] + '/a3m/'+item[1][:3] + '/'+ item[1] + '.a3m.gz', item[1])
    tpltA = torch.load(params['PDB_DIR']+'/torch/hhr/'+item[1][:3]+'/'+item[1]+'.pt')
   
    # get msa features
    msa_prot = a3mA['msa'].long()
    ins_prot = a3mA['ins'].long()

    if len(msa_prot) > params['BLOCKCUT']:
        msa_prot, ins_prot = MSABlockDeletion(msa_prot, ins_prot)
    a3m_prot = {"msa": msa_prot, "ins": ins_prot}
    xyz_prot, mask_prot = pdbA["xyz"], pdbA["mask"]
    protein_L, nprotatoms, _ = xyz_prot.shape
    # Load small molecule

    mol, msa_sm, ins_sm = parse_mol(params["MOL_DIR"]+"/mol2/"+item[0][1:3]+"/"+item[0][:-1]+random.choice(sm_chains)+".mol2")
    a3m_sm = {"msa": msa_sm.unsqueeze(0), "ins": ins_sm.unsqueeze(0)}
    G = get_nxgraph(mol)
    frames = get_atom_frames(msa_sm, mol, G)
    xyz_sm, mask_sm = get_ligand_xyz(mol)

    N_symmetry, sm_L, _ = xyz_sm.shape
    # Generate ground truth structure: account for ligand symmetry
    xyz = torch.full((N_symmetry, protein_L+sm_L, NTOTAL, 3), np.nan).float()
    mask = torch.full(xyz.shape[:-1], False).bool()
    xyz[:, :protein_L, :nprotatoms, :] = xyz_prot.expand(N_symmetry, protein_L, nprotatoms, 3)
    xyz[:, protein_L:, 1, :] = xyz_sm
    mask[:, :protein_L, :nprotatoms] = mask_prot.expand(N_symmetry, protein_L, nprotatoms)
    mask[:, protein_L:, 1] = mask_sm
    
    Ls = [xyz_prot.shape[0], xyz_sm.shape[1]]
    a3m = merge_a3m_hetero(a3m_prot, a3m_sm, Ls)
    msa = a3m['msa'].long()
    ins = a3m['ins'].long()

    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params)
    
    idx = torch.arange(sum(Ls))
    idx[Ls[0]:] += 200
    
    chain_idx = torch.zeros((sum(Ls), sum(Ls))).long()
    chain_idx[:Ls[0], :Ls[0]] = 1
    chain_idx[Ls[0]:, Ls[0]:] = 1 
    bond_feats = torch.zeros((sum(Ls), sum(Ls))).long()
    bond_feats[:Ls[0], :Ls[0]] = get_protein_bond_feats(Ls[0])
    bond_feats[Ls[0]:, Ls[0]:] = get_bond_feats(mol, G)
    
    ntempl = np.random.randint(params['MINTPLT'], params['MAXTPLT']-1)
    xyz_t, f1d_t = TemplFeaturize(tpltA, sum(Ls), params, offset=0, npick=ntempl, pick_top=pick_top) 
    # give template of native backbone if none exists
    
    #generate initial coordinates
    xyz_prev = xyz_t[0]

    if sum(Ls) > params["CROP"]:
        sel = crop_small_molecule(xyz_prot, xyz_sm[0], Ls, params)
        
        seq = seq[:,sel]
        msa_seed_orig = msa_seed_orig[:,:,sel]
        msa_seed = msa_seed[:,:,sel]
        msa_extra = msa_extra[:,:,sel]
        mask_msa = mask_msa[:,:,sel]
        xyz = xyz[:,sel] 
        mask = mask[:,sel]
        xyz_t = xyz_t[:,sel]
        f1d_t = f1d_t[:,sel]
        xyz_prev = xyz_prev[sel] # need to initialize ligand atoms
        #
        idx = idx[sel]
        chain_idx = chain_idx[sel][:,sel]
        bond_feats = bond_feats[sel][:, sel]
    bond_feats = torch.nn.functional.one_hot(bond_feats, num_classes=NBTYPES)
    # replace missing with blackholes & conovert NaN to zeros to avoid any NaN problems during loss calculation
    init = INIT_CRDS.reshape(1, NTOTAL, 3).repeat(xyz.shape[0], xyz.shape[1], 1, 1)
    xyz = torch.where(mask[...,None], xyz, init).contiguous()
    xyz = torch.nan_to_num(xyz)

    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa,\
           xyz.float(), mask, idx.long(), \
           xyz_t.float(), f1d_t.float(), xyz_prev.float(), \
           chain_idx, False, False, frames, bond_feats

def crop_small_molecule(prot_xyz, lig_xyz,Ls, params):
    """choose residues with calphas close to the ligand center of mass"""
    ligand_com = torch.nanmean(lig_xyz, dim=[0,1]).expand(1,3)
    dist = torch.cdist(prot_xyz[:,1].double(), ligand_com).flatten()
    _, idx = torch.topk(dist, params["CROP"]-len(lig_xyz), largest=False)
    sel, _ = torch.sort(idx)
    # select the whole ligand
    lig_sel = torch.arange(lig_xyz.shape[0])+Ls[0]
    return torch.cat((sel, lig_sel))


class Dataset(data.Dataset):
    def __init__(self, IDs, loader, item_dict, params, homo, unclamp_cut=0.9, pick_top=True, p_homo_cut=-1.0):
        self.IDs = IDs
        self.item_dict = item_dict
        self.loader = loader
        self.params = params
        self.homo = homo
        self.pick_top = pick_top
        self.unclamp_cut = unclamp_cut
        self.p_homo_cut = p_homo_cut

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        sel_idx = np.random.randint(0, len(self.item_dict[ID]))
        p_unclamp = np.random.rand()
        if p_unclamp > self.unclamp_cut:
            out = self.loader(self.item_dict[ID][sel_idx][0], self.params, self.homo,
                              unclamp=True, 
                              pick_top=self.pick_top, 
                              p_homo_cut=self.p_homo_cut)
        else:
            out = self.loader(self.item_dict[ID][sel_idx][0], self.params, self.homo, 
                              pick_top=self.pick_top,
                              p_homo_cut=self.p_homo_cut)
        return out

class DatasetComplex(data.Dataset):
    def __init__(self, IDs, loader, item_dict, params, pick_top=True, negative=False):
        self.IDs = IDs
        self.item_dict = item_dict
        self.loader = loader
        self.params = params
        self.pick_top = pick_top
        self.negative = negative

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        sel_idx = np.random.randint(0, len(self.item_dict[ID]))
        out = self.loader(self.item_dict[ID][sel_idx][0],
                          self.item_dict[ID][sel_idx][1],
                          self.item_dict[ID][sel_idx][2],
                          self.item_dict[ID][sel_idx][3],
                          self.params,
                          pick_top = self.pick_top,
                          negative = self.negative)
        return out

class DatasetNAComplex(data.Dataset):
    def __init__(self, IDs, loader, item_dict, params, pick_top=True, negative=False, native_NA_frac=0.0):
        self.IDs = IDs
        self.item_dict = item_dict
        self.loader = loader
        self.params = params
        self.pick_top = pick_top
        self.negative = negative
        self.native_NA_frac = native_NA_frac

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        sel_idx = np.random.randint(0, len(self.item_dict[ID]))
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
    def __init__(self, IDs, loader, item_dict, params):
        self.IDs = IDs
        self.item_dict = item_dict
        self.loader = loader
        self.params = params

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        sel_idx = np.random.randint(0, len(self.item_dict[ID]))
        out = self.loader(
            self.item_dict[ID][sel_idx][0],
            self.item_dict[ID][sel_idx][1],
            self.params
        )
        return out


class DatasetSMComplex(data.Dataset):
    def __init__(self, IDs, loader, item_dict, params):
        self.IDs = IDs
        self.item_dict = item_dict
        self.loader = loader
        self.params = params

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        sel_idx = np.random.randint(0, len(self.item_dict[ID]))
        out = self.loader(
            self.item_dict[ID][sel_idx][0],
            self.item_dict[ID][sel_idx][2],
            self.params
        )
        return out


class DistilledDataset(data.Dataset):
    def __init__(
        self,
        pdb_IDs, pdb_loader, pdb_dict,
        compl_IDs, compl_loader, compl_dict,
        neg_IDs, neg_loader, neg_dict,
        na_compl_IDs, na_compl_loader, na_compl_dict,
        na_neg_IDs, na_neg_loader, na_neg_dict,
        fb_IDs, fb_loader, fb_dict,
        rna_IDs, rna_loader, rna_dict,
        sm_compl_IDs, sm_compl_loader, sm_compl_dict, 
        homo, 
        params,
        native_NA_frac=0.25,
        unclamp_cut=0.9
    ):
        #
        self.pdb_IDs = pdb_IDs
        self.pdb_dict = pdb_dict
        self.pdb_loader = pdb_loader
        self.compl_IDs = compl_IDs
        self.compl_loader = compl_loader
        self.compl_dict = compl_dict
        self.neg_IDs = neg_IDs
        self.neg_loader = neg_loader
        self.neg_dict = neg_dict
        self.na_compl_IDs = na_compl_IDs
        self.na_compl_loader = na_compl_loader
        self.na_compl_dict = na_compl_dict
        self.na_neg_IDs = na_neg_IDs
        self.na_neg_loader = na_neg_loader
        self.na_neg_dict = na_neg_dict
        self.fb_IDs = fb_IDs
        self.fb_dict = fb_dict
        self.fb_loader = fb_loader
        self.rna_IDs = rna_IDs
        self.rna_dict = rna_dict
        self.rna_loader = rna_loader
        self.sm_compl_IDs = sm_compl_IDs
        self.sm_compl_loader = sm_compl_loader
        self.sm_compl_dict = sm_compl_dict
        self.homo = homo
        self.params = params
        self.unclamp_cut = unclamp_cut
        self.native_NA_frac = native_NA_frac

        self.compl_inds = np.arange(len(self.compl_IDs))
        self.neg_inds = np.arange(len(self.neg_IDs))
        self.na_compl_inds = np.arange(len(self.na_compl_IDs))
        self.na_neg_inds = np.arange(len(self.na_neg_IDs))
        self.fb_inds = np.arange(len(self.fb_IDs))
        self.pdb_inds = np.arange(len(self.pdb_IDs))
        self.rna_inds = np.arange(len(self.rna_IDs))
        self.sm_compl_inds = np.arange(len(self.sm_compl_IDs))

    def __len__(self):
        return (
            len(self.fb_inds)
            + len(self.pdb_inds)
            + len(self.compl_inds)
            + len(self.neg_inds)
            + len(self.na_compl_inds)
            + len(self.na_neg_inds)
            + len(self.rna_inds)
            + len(self.sm_compl_inds)
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
        if index >= offset and index < offset + len(self.neg_inds):
            ID = self.neg_IDs[index-offset]
            sel_idx = np.random.randint(0, len(self.neg_dict[ID]))
            out = self.neg_loader(
                self.neg_dict[ID][sel_idx][0],
                self.neg_dict[ID][sel_idx][1],
                self.neg_dict[ID][sel_idx][2],
                self.neg_dict[ID][sel_idx][3],
                self.params,
                negative=True
            )

        offset += len(self.neg_inds)
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
        if index >= offset and index < offset + len(self.na_neg_inds):
            ID = self.na_neg_IDs[index-offset]
            sel_idx = np.random.randint(0, len(self.na_neg_dict[ID]))
            out = self.na_neg_loader(
                self.na_neg_dict[ID][sel_idx][0],
                self.na_neg_dict[ID][sel_idx][1],
                self.params,
                negative=True,
                native_NA_frac=self.native_NA_frac
            )

        offset += len(self.na_neg_inds)
        if index >= offset and index < offset + len(self.rna_inds):
            ID = self.rna_IDs[index-offset]
            sel_idx = np.random.randint(0, len(self.rna_dict[ID]))
            out = self.rna_loader(
                self.rna_dict[ID][sel_idx][0],
                self.rna_dict[ID][sel_idx][1],
                self.params
            )
        offset += len(self.rna_inds)
        if index >= offset:
            ID = self.sm_compl_IDs[index-offset]
            sel_idx = np.random.randint(0, len(self.sm_compl_dict[ID]))
            out = self.sm_compl_loader(
                self.sm_compl_dict[ID][sel_idx][0],
                self.sm_compl_dict[ID][sel_idx][2],
                self.params
            )
        return out

class DistributedWeightedSampler(data.Sampler):
    def __init__(
        self,
        dataset,
        pdb_weights,
        fb_weights,
        compl_weights,
        neg_weights,
        na_compl_weights,
        neg_na_compl_weights,
        rna_weights,
        sm_compl_weights,
        num_example_per_epoch=25600,
        fraction_fb=0.16,
        fraction_compl=0.16,  # half neg, half pos
        fraction_na_compl=0.16, # half neg, half pos
        fraction_rna=0.16,
        fraction_sm_compl=0.16, 
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
        assert (fraction_fb+fraction_compl+fraction_na_compl+fraction_rna+fraction_sm_compl<= 1.0)

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.num_fb_per_epoch = int(round(num_example_per_epoch*fraction_fb))
        self.num_compl_per_epoch = int(round(0.5*num_example_per_epoch*fraction_compl))
        self.num_neg_per_epoch = self.num_compl_per_epoch
        self.num_na_compl_per_epoch = int(round(0.5*num_example_per_epoch*fraction_na_compl))
        self.num_neg_na_compl_per_epoch = self.num_na_compl_per_epoch
        self.num_rna_per_epoch = int(round(num_example_per_epoch*fraction_rna))
        self.num_sm_compl_per_epoch = int(round(num_example_per_epoch*fraction_sm_compl))

        self.num_pdb_per_epoch = num_example_per_epoch - (
            self.num_fb_per_epoch 
            + self.num_compl_per_epoch
            + self.num_neg_per_epoch
            + self.num_na_compl_per_epoch
            + self.num_neg_na_compl_per_epoch
            + self.num_rna_per_epoch
            + self.num_sm_compl_per_epoch
        )

        if (rank==0):
            print (
                "Per epoch:",
                self.num_pdb_per_epoch,"pdb,",
                self.num_fb_per_epoch,"fb,",
                self.num_compl_per_epoch,"compl,",
                self.num_neg_per_epoch,"neg,",
                self.num_na_compl_per_epoch,"NA compl,",
                self.num_neg_na_compl_per_epoch,"NA neg,",
                self.num_rna_per_epoch,"RNA,",
                self.num_sm_compl_per_epoch, "SM Compl."
            )


        self.total_size = num_example_per_epoch
        self.num_samples = self.total_size // self.num_replicas
        self.rank = rank
        self.epoch = 0
        self.replacement = replacement

        self.pdb_weights = pdb_weights
        self.fb_weights = fb_weights

        self.compl_weights = compl_weights
        self.neg_weights = neg_weights

        self.na_compl_weights = na_compl_weights
        self.neg_na_compl_weights = neg_na_compl_weights

        self.rna_weights = rna_weights
        self.sm_compl_weights = sm_compl_weights

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
        sel_indices = torch.tensor((),dtype=int)
        if (self.num_fb_per_epoch>0):
            fb_sampled = torch.multinomial(self.fb_weights, self.num_fb_per_epoch, self.replacement, generator=g)
            sel_indices = torch.cat((sel_indices, indices[fb_sampled]))

        if (self.num_pdb_per_epoch>0):
            offset = len(self.dataset.fb_IDs)
            pdb_sampled = torch.multinomial(self.pdb_weights, self.num_pdb_per_epoch, self.replacement, generator=g)
            sel_indices = torch.cat((sel_indices, indices[pdb_sampled + offset]))

        if (self.num_compl_per_epoch>0):
            offset = len(self.dataset.fb_IDs) + len(self.dataset.pdb_IDs)
            compl_sampled = torch.multinomial(self.compl_weights, self.num_compl_per_epoch, self.replacement, generator=g)
            sel_indices = torch.cat((sel_indices, indices[compl_sampled + offset]))
        
        if (self.num_neg_per_epoch>0):
            offset = len(self.dataset.fb_IDs) + len(self.dataset.pdb_IDs) + len(self.dataset.compl_IDs)
            neg_sampled = torch.multinomial(self.neg_weights, self.num_neg_per_epoch, self.replacement, generator=g)
            sel_indices = torch.cat((sel_indices, indices[neg_sampled + offset]))

        if (self.num_na_compl_per_epoch>0):
            offset = (
                len(self.dataset.fb_IDs) 
                + len(self.dataset.pdb_IDs) 
                + len(self.dataset.compl_IDs)
                + len(self.dataset.neg_IDs)
            )
            na_compl_sampled = torch.multinomial(self.na_compl_weights, self.num_na_compl_per_epoch, self.replacement, generator=g)
            sel_indices = torch.cat((sel_indices, indices[na_compl_sampled + offset]))

        if (self.num_neg_na_compl_per_epoch>0):
            offset = (
                len(self.dataset.fb_IDs) 
                + len(self.dataset.pdb_IDs) 
                + len(self.dataset.compl_IDs)
                + len(self.dataset.neg_IDs)
                + len(self.dataset.na_compl_IDs)
            )
            neg_na_sampled = torch.multinomial(self.neg_na_compl_weights, self.num_neg_na_compl_per_epoch, self.replacement, generator=g)
            sel_indices = torch.cat((sel_indices, indices[neg_na_sampled + offset]))

        if (self.num_rna_per_epoch>0):
            offset = (
                len(self.dataset.fb_IDs) 
                + len(self.dataset.pdb_IDs) 
                + len(self.dataset.compl_IDs)
                + len(self.dataset.neg_IDs)
                + len(self.dataset.na_compl_IDs)
                + len(self.dataset.na_neg_IDs)
            )
            rna_sampled = torch.multinomial(self.rna_weights, self.num_rna_per_epoch, self.replacement, generator=g)
            sel_indices = torch.cat((sel_indices, indices[rna_sampled + offset]))

        if (self.num_sm_compl_per_epoch>0):
            offset = (
                len(self.dataset.fb_IDs) 
                + len(self.dataset.pdb_IDs) 
                + len(self.dataset.compl_IDs)
                + len(self.dataset.neg_IDs)
                + len(self.dataset.na_compl_IDs)
                + len(self.dataset.na_neg_IDs)
                + len(self.dataset.rna_IDs)
            )
            sm_compl_sampled = torch.multinomial(self.sm_compl_weights, self.num_sm_compl_per_epoch, self.replacement, generator=g)
            sel_indices = torch.cat((sel_indices, indices[sm_compl_sampled + offset]))
        

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

