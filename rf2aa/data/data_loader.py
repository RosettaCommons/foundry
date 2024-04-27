import torch
import warnings
import time
from icecream import ic
from torch.utils import data
import os, csv, random, pickle, gzip, itertools, time, ast, copy, sys
from dateutil import parser
from collections import OrderedDict, Counter
from itertools import permutations
from typing import Dict, Optional, Tuple, List, Set, Any
from pathlib import Path
from os.path import exists

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
sys.path.append(script_dir+'/../')

import numpy as np
import pandas as pd
import scipy
from scipy.sparse.csgraph import shortest_path
import networkx as nx

import rf2aa.cifutils as cifutils
from rf2aa.data.parsers import parse_a3m, parse_pdb, parse_fasta_if_exists, parse_mol, parse_mixed_fasta, get_dislf
from rf2aa.data.chain_crop import get_complex_crop, get_crop, get_discontiguous_crop, get_na_crop, get_spatial_crop, \
    crop_sm_compl, crop_sm_compl_asmb_contig, crop_sm_compl_assembly, crop_chirals
from rf2aa.chemical import ChemicalData as ChemData
from rf2aa.chemical import load_tanimoto_sim_matrix

from rf2aa.kinematics import get_chirals
from rf2aa.symmetry import get_symmetry
from rf2aa.set_seed import seed_all
from rf2aa.data.identical_ligands import get_extra_identical_copies_from_chains
from rf2aa.util import get_nxgraph, get_atom_frames, get_bond_feats, get_protein_bond_feats, \
    center_and_realign_missing, random_rot_trans, cif_prot_to_xyz, \
    cif_ligand_to_xyz, cif_ligand_to_obmol, get_automorphs, get_ligand_atoms_bonds, \
    map_identical_prot_chains, cartprodcat, idx_from_Ls, same_chain_2d_from_Ls, bond_feats_from_Ls, \
    reindex_protein_feats_after_atomize, get_residue_contacts, atomize_discontiguous_residues, pop_protein_feats, \
    is_atom, get_atom_template_indices, reassign_symmetry_after_cropping, expand_xyz_sm_to_ntotal, Ls_from_same_chain_2d, \
    is_protein, is_nucleic, is_RNA, is_DNA, is_atom
from rf2aa.data.cluster_dataset import cluster_factory
assert "rf2aa" in os.path.abspath(cifutils.__file__)


# fd NA structures are in a different order internally than they are stored
# fd on disk.  This function remaps the loaded order->model order
# old:
#       0      1      2      3      4      5      6      7      8      9      10
#    (" OP1"," P  "," OP2"," O5'"," C5'"," C4'"," O4'"," C3'"," O3'"," C1'"," C2'", ... #   A
#    (" OP1"," P  "," OP2"," O5'"," C5'"," C4'"," O4'"," C3'"," O3'"," C1'"," C2'", ... #   C
#    (" OP1"," P  "," OP2"," O5'"," C5'"," C4'"," O4'"," C3'"," O3'"," C1'"," C2'", ... #   G
#    (" OP1"," P  "," OP2"," O5'"," C5'"," C4'"," O4'"," C3'"," O3'"," C1'"," C2'", ... #   U

# new:
#    (" O4'"," C1'"," C2'"," OP1"," P  "," OP2"," O5'"," C5'"," C4'"," C3'"," O3'", ... #27   A
#    (" O4'"," C1'"," C2'"," OP1"," P  "," OP2"," O5'"," C5'"," C4'"," C3'"," O3'", ... #28   C
#    (" O4'"," C1'"," C2'"," OP1"," P  "," OP2"," O5'"," C5'"," C4'"," C3'"," O3'", ... #29   G
#    (" O4'"," C1'"," C2'"," OP1"," P  "," OP2"," O5'"," C5'"," C4'"," C3'"," O3'", ... #30   U

def remap_NA_xyz_tensors(xyz,mask,seq):
    if ChemData().params.use_phospate_frames_for_NA:
        return xyz,mask

    dna_mask = is_DNA(seq)
    DNAMAP = (6,10,9,0,1,2,3,4,5,7,8)
    xyz[:,dna_mask,:11] = xyz[:,dna_mask][...,DNAMAP,:]
    mask[:,dna_mask,:11] = mask[:,dna_mask][...,DNAMAP]

    rna_mask = is_RNA(seq)
    RNAMAP = (6,9,10,0,1,2,3,4,5,7,8)
    xyz[:,rna_mask,:11] = xyz[:,rna_mask][...,RNAMAP,:]
    mask[:,rna_mask,:11] = mask[:,rna_mask][...,RNAMAP]

    return xyz,mask

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
    mask = np.ones(N, bool)
    mask[to_delete] = 0

    return msa[mask], ins[mask]

def subsample_MSA(msa, ins, num_seqs_to_sample):
    """
    subsample MSA. this is distinct from block deletion which attempts to cut off a full clade 
    because this is intended to make the MSA very shallow to force the model to condition on information 
    in xyz_prev
    Args:
        msa (torch.Tensor): msa pulled from a3m file
        ins (torch.Tensor): insertions from a3m file
        num_seqs_to_sample (int): number of sequences to select from MSA
    """
    num_seqs_in_msa = msa.shape[0] - 1 # don't include query sequence
    samples = torch.randperm(num_seqs_in_msa)[:num_seqs_to_sample]
    samples = torch.cat([torch.tensor([0]), samples]) # add query sequence back in 
    return msa[samples], ins[samples]

def cluster_sum(data, assignment, N_seq, N_res, cast_to_float: bool = True):
    if cast_to_float:
        data = data.float()

    csum = torch.zeros(N_seq, N_res, data.shape[-1], device=data.device).scatter_add(
        0, assignment.view(-1, 1, 1).expand(-1, N_res, data.shape[-1]), data
    )
    return csum

def get_term_feats(Ls):
    """Creates N/C-terminus binary features"""
    term_info = torch.zeros((sum(Ls),2)).float()
    start = 0
    for L_chain in Ls:
        term_info[start, 0] = 1.0 # flag for N-term
        term_info[start+L_chain-1,1] = 1.0 # flag for C-term
        start += L_chain
    return term_info


def get_sample(msa, nmer, i_cycle, N, seed_msa_clus=None):
    sample_mono = torch.randperm((N - 1) // nmer, device=msa.device)
    sample = [sample_mono + imer * ((N - 1) // nmer) for imer in range(nmer)]
    sample = torch.stack(sample, dim=-1)
    sample = sample.reshape(-1)

    # add MSA clusters pre-chosen before calling this function
    if seed_msa_clus is not None:
        sample_seed = seed_msa_clus[i_cycle]
        sample_more = torch.tensor([i for i in sample if i not in sample_seed])
        N_sample_more = len(sample) - len(sample_seed)
        if N_sample_more > 0:
            sample_more = sample_more[torch.randperm(len(sample_more))[:N_sample_more]]
            sample = torch.cat([sample_seed, sample_more])
        else:
            sample = sample_seed[
                : len(sample)
            ]  # take all clusters from pre-chosen ones
    return sample


def get_masked_msa(
    msa, msa_clust_indices, p_mask, seq, msa_onehot, raw_profile, msa_clust
):
    random_aa = torch.tensor(
        [[0.05] * 20 + [0.0] * (ChemData().NAATOKENS - 20)], device=msa.device
    )
    same_aa = msa_onehot[msa_clust_indices]
    # explicitly remove ] from nucleic acids and atoms
    same_aa[..., ChemData().NPROTAAS :] = 0
    raw_profile[..., ChemData().NPROTAAS :] = 0
    probs = 0.1 * random_aa + 0.1 * raw_profile + 0.1 * same_aa
    # probs = torch.nn.functional.pad(probs, (0, 1), "constant", 0.7)

    # explicitly set the probability of masking for nucleic acids and atoms
    probs[..., is_protein(seq), ChemData().MASKINDEX] = 0.7
    probs[..., ~is_protein(seq), :] = (
        0  # probably overkill but set all none protein elements to 0
    )
    probs[1:, ~is_protein(seq), 20] = 1.0  # want to leave the gaps as gaps
    probs[0, is_nucleic(seq), ChemData().MASKINDEX] = 1.0
    probs[0, is_atom(seq), ChemData().aa2num["ATM"]] = 1.0

    sampler = torch.distributions.categorical.Categorical(probs=probs)
    mask_sample = sampler.sample()

    mask_pos = torch.rand(msa_clust.shape, device=msa_clust.device) < p_mask
    # mask_pos[msa_clust>MASKINDEX]=False # no masking on NAs
    use_seq = msa_clust
    msa_masked = torch.where(mask_pos, mask_sample, use_seq)
    return msa_masked, mask_pos


def get_extra_msa(
    N,
    Nclust,
    sample,
    msa,
    ins,
    msa_onehot_float,
    msa_clust_indices,
    msa_clust_onehot,
    mask_pos,
    L,
    N_extra,
):
    if N > Nclust * 2:  # there are enough extra sequences
        msa_extra_indices = sample[Nclust - 1 :] + 1
        extra_mask = torch.full((N_extra, L), False, device=msa.device)
        msa_extra_onehot = msa_onehot_float[msa_extra_indices]
    elif N - Nclust < 1:
        msa_extra_indices = msa_clust_indices
        extra_mask = mask_pos.clone()
        msa_extra_onehot = msa_clust_onehot.clone()
    else:
        msa_extra_indices = torch.cat(
            [
                msa_clust_indices,
                sample[Nclust - 1 :] + 1,
            ]
        )
        msa_extra_onehot = torch.cat(
            [msa_clust_onehot, msa_onehot_float[sample[Nclust - 1 :] + 1]],
            dim=0,
        )
        mask_add = torch.full((N_extra, L), False, device=msa.device)
        extra_mask = torch.cat((mask_pos, mask_add), dim=0)

    ins_extra = ins[msa_extra_indices]
    return msa_extra_indices, msa_extra_onehot, ins_extra, extra_mask


def compute_assignment(msa_extra_indices, extra_mask, msa_float, msa_clust, mask_pos):
    # Note: float cast does implicit copy, no need to worry about
    # the overwritten values for the clust tensor
    msa_extra_for_agreement = msa_float[msa_extra_indices]
    msa_clust_for_agreement = msa_clust.float()

    count_clust = torch.logical_and(
        ~mask_pos, msa_clust != 20
    )  # 20: index for gap, ignore both masked & gaps
    count_extra = torch.logical_and(~extra_mask, msa_extra_for_agreement != 20)

    # Things that are masked should not compute to the agreement sum,
    # hence choosing two negative numbers here that are not equal.
    overwritten_extra = msa_extra_for_agreement[~count_extra]
    msa_extra_for_agreement[~count_extra] = -1.0
    msa_clust_for_agreement[~count_clust] = -2.0

    # Uses 0 norm cdist to compute sequence identity percentage,
    # which is equivalent to hamming distance,
    # then inverts to get the number of equal positions.
    agreement = torch.cdist(msa_extra_for_agreement, msa_clust_for_agreement, p=0.0)
    agreement = msa_extra_for_agreement.shape[1] - agreement
    assignment = torch.argmax(agreement, dim=-1)

    # Have to replace the re-written values because what is in the seed
    # MSA changes per recycle
    msa_float[msa_extra_indices][~count_extra] = overwritten_extra
    return assignment


def compute_seed_msa(
    extra_mask,
    msa_extra_onehot,
    ins_extra,
    ins_clust,
    mask_pos,
    Nclust,
    L,
    assignment,
    eps,
):
    # seed MSA features
    # 1. one_hot encoded aatype: msa_clust_onehot
    # 2. cluster profile
    count_extra = ~extra_mask
    count_clust = ~mask_pos
    msa_clust_profile = cluster_sum(
        count_extra[:, :, None] * msa_extra_onehot,
        assignment,
        Nclust,
        L,
        cast_to_float=False,
    )
    msa_clust_profile += count_clust[:, :, None] * msa_clust_profile

    count_profile = cluster_sum(count_extra[:, :, None], assignment, Nclust, L).view(
        Nclust, L
    )
    count_profile += count_clust
    count_profile += eps
    msa_clust_profile /= count_profile[:, :, None]
    # 3. insertion statistics

    msa_clust_del = cluster_sum(
        (count_extra * ins_extra)[:, :, None], assignment, Nclust, L
    ).view(Nclust, L)

    msa_clust_del += count_clust * ins_clust
    msa_clust_del /= count_profile
    ins_clust = (2.0 / np.pi) * torch.arctan(ins_clust.float() / 3.0)  # (from 0 to 1)
    msa_clust_del = (2.0 / np.pi) * torch.arctan(
        msa_clust_del.float() / 3.0
    )  # (from 0 to 1)
    ins_clust = torch.stack((ins_clust, msa_clust_del), dim=-1)
    return ins_clust, msa_clust_profile


def MSAFeaturize(
    msa,
    ins,
    params,
    p_mask=0.15,
    eps=1e-6,
    nmer=1,
    L_s=[],
    term_info=None,
    tocpu=False,
    fixbb=False,
    seed_msa_clus=None,
    deterministic=False
):
    """
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
    """
    if deterministic:
        seed_all()
    # Truncate MSA (for efficiency when pre-computing lengths)
    if params.get("MSA_LIMIT") is not None:
        # Raise a warning that we are truncating the MSA
        warnings.warn(
            f"Truncating MSA to {params['MSA_LIMIT']} sequences. Only to be used for length pre-computation, NOT training."
        )
        msa = msa[: params["MSA_LIMIT"]]
        seed_msa_clus = None

    if fixbb:
        p_mask = 0
        msa = msa[:1]
        ins = ins[:1]
    N, L = msa.shape

    if term_info is None:
        if len(L_s) == 0:
            L_s = [L]
        term_info = get_term_feats(L_s)
    term_info = term_info.to(msa.device)

    # raw MSA profile
    msa_float = msa.float()
    msa_onehot = torch.nn.functional.one_hot(msa, num_classes=ChemData().NAATOKENS)
    msa_onehot_float = msa_onehot.float()
    raw_profile = msa_onehot_float.mean(dim=0)

    # Select Nclust sequence randomly (seed MSA or latent MSA)
    Nclust = (min(N, params["MAXLAT"]) - 1) // nmer
    Nclust = Nclust * nmer + 1

    if N > Nclust * 2:
        Nextra = N - Nclust
    else:
        Nextra = N
    Nextra = min(Nextra, params["MAXSEQ"]) // nmer
    Nextra = max(1, Nextra * nmer)
    #
    b_seq = list()
    b_msa_clust = list()
    b_msa_seed = list()
    b_msa_extra = list()
    b_mask_pos = list()
    for i_cycle in range(params["MAXCYCLE"]):
        sample = get_sample(msa, nmer, i_cycle, N, seed_msa_clus)

        msa_clust_indices = torch.cat(
            [
                torch.zeros((1,), device=msa.device, dtype=torch.int64),
                sample[: Nclust - 1] + 1,
            ]
        )
        msa_clust = msa[msa_clust_indices]
        ins_clust = ins[msa_clust_indices]

        # 15% random masking
        # - 10%: aa replaced with a uniformly sampled random amino acid
        # - 10%: aa replaced with an amino acid sampled from the MSA profile
        # - 10%: not replaced
        # - 70%: replaced with a special token ("mask")
        seq = msa_clust[0]

        msa_masked, mask_pos = get_masked_msa(
            msa, msa_clust_indices, p_mask, seq, msa_onehot, raw_profile, msa_clust
        )
        msa_clust_onehot = torch.nn.functional.one_hot(
            msa_masked, num_classes=ChemData().NAATOKENS
        ).float()

        b_seq.append(msa_masked[0].clone())

        ## get extra sequences
        N_extra = sample.shape[0] - Nclust + 1

        msa_extra_indices, msa_extra_onehot, ins_extra, extra_mask = get_extra_msa(
            N,
            Nclust,
            sample,
            msa,
            ins,
            msa_onehot_float,
            msa_clust_indices,
            msa_clust_onehot,
            mask_pos,
            L,
            N_extra,
        )

        # clustering (assign remaining sequences to their closest cluster by Hamming distance
        assignment = compute_assignment(
            msa_extra_indices, extra_mask, msa_float, msa_clust, mask_pos
        )
        ins_clust, msa_clust_profile = compute_seed_msa(
            extra_mask,
            msa_extra_onehot,
            ins_extra,
            ins_clust,
            mask_pos,
            Nclust,
            L,
            assignment,
            eps,
        )

        if fixbb:
            assert params["MAXCYCLE"] == 1
            msa_clust_profile = msa_clust_onehot
            msa_extra_onehot = msa_clust_onehot
            ins_clust[:] = 0
            ins_extra[:] = 0
            # This is how it is done in rfdiff, but really it seems like it should be all 0.
            # Keeping as-is for now for consistency, as it may be used in downstream masking done
            # by apply_masks.
            mask_pos = torch.full_like(msa_clust, 1, dtype=torch.bool)
        msa_seed = torch.cat(
            (
                msa_clust_onehot,
                msa_clust_profile,
                ins_clust,
                term_info[None].expand(Nclust, -1, -1),
            ),
            dim=-1,
        )

        # extra MSA features
        ins_extra = (2.0 / np.pi) * torch.arctan(
            ins_extra[:Nextra].float() / 3.0
        )  # (from 0 to 1)
        try:
            msa_extra = torch.cat(
                (
                    msa_extra_onehot[:Nextra],
                    ins_extra[:, :, None],
                    term_info[None].expand(Nextra, -1, -1),
                ),
                dim=-1,
            )
        except Exception as e:
            print("msa_extra.shape", msa_extra.shape)
            print("ins_extra.shape", ins_extra.shape)

        if tocpu:
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
    xyz = ChemData().INIT_CRDS.reshape(1,1,ChemData().NTOTAL,3).repeat(n_tmpl,L,1,1) \
        + torch.rand(n_tmpl,L,1,3)*random_noise - random_noise/2
    t1d = torch.nn.functional.one_hot(torch.full((n_tmpl, L), 20).long(), num_classes=ChemData().NAATOKENS-1).float() # all gaps
    conf = torch.zeros((n_tmpl, L, 1)).float()
    t1d = torch.cat((t1d, conf), -1)
    mask_t = torch.full((n_tmpl,L,ChemData().NTOTAL), False)
    return xyz, t1d, mask_t, np.full((n_tmpl), "")


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

    xyz = ChemData().INIT_CRDS.reshape(1,1,ChemData().NTOTAL,3).repeat(npick_global,qlen,1,1) + torch.rand(1,qlen,1,3)*random_noise
    mask_t = torch.full((npick_global,qlen,ChemData().NTOTAL),False) # True for valid atom, False for missing atom
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
        # xyz[i] = center_and_realign_missing(xyz[i], mask_t[i], same_chain=same_chain)

    t1d = torch.nn.functional.one_hot(t1d, num_classes=ChemData().NAATOKENS-1).float() # (no mask token)
    t1d = torch.cat((t1d, t1d_val[...,None]), dim=-1)

    tplt_ids = np.array(tplt["ids"])[sample].flatten() # np.array of chain ids (ordered)
    return xyz, t1d, mask_t, tplt_ids

def merge_hetero_templates(xyz_t_prot, f1d_t_prot, mask_t_prot, tplt_ids, Ls_prot):
    """Diagonally tiles template coordinates, 1d input features, and masks across
    template and residue dimensions. 1st template is concatenated directly on residue
    dimension after a random rotation & translation.
    """
    N_tmpl_tot = sum([x.shape[0] for x in xyz_t_prot])

    xyz_t_out, f1d_t_out, mask_t_out, _ = blank_template(N_tmpl_tot, sum(Ls_prot))
    tplt_ids_out = np.full((N_tmpl_tot),"", dtype=object) # rk bad practice.. should fix
    i_tmpl = 0
    i_res = 0
    for xyz_, f1d_, mask_, ids in zip(xyz_t_prot, f1d_t_prot, mask_t_prot, tplt_ids):
        N_tmpl, L_tmpl = xyz_.shape[:2]
        if i_tmpl == 0:
            i1, i2 = 1, N_tmpl
        else:
            i1, i2 = i_tmpl, i_tmpl+N_tmpl - 1
 
        # 1st template is concatenated directly, so that all atoms are set in xyz_prev
        xyz_t_out[0, i_res:i_res+L_tmpl] = random_rot_trans(xyz_[0:1])
        f1d_t_out[0, i_res:i_res+L_tmpl] = f1d_[0]
        mask_t_out[0, i_res:i_res+L_tmpl] = mask_[0]

        if not tplt_ids_out[0]: # only add first template
            tplt_ids_out[0] = ids[0]
        # remaining templates are diagonally tiled
        xyz_t_out[i1:i2, i_res:i_res+L_tmpl] = xyz_[1:]
        f1d_t_out[i1:i2, i_res:i_res+L_tmpl] = f1d_[1:]
        mask_t_out[i1:i2, i_res:i_res+L_tmpl] = mask_[1:]
        tplt_ids_out[i1:i2] = ids[1:] 
        if i_tmpl == 0:
            i_tmpl += N_tmpl
        else:
            i_tmpl += N_tmpl-1
        i_res += L_tmpl

    return xyz_t_out, f1d_t_out, mask_t_out, tplt_ids_out

def spoof_template(xyz, seq, mask, is_motif=None, template_conf=1, random_noise=5):
    """
    generate template features from an arbitrary xyz, seq and mask
    is_motif indicates which residues from the input xyz should be templated
    """
    if len(xyz.shape) == 4: # template ignores symmetry dimension
        xyz = xyz[0]
    if len(mask.shape) == 3:
        mask = mask[0]

    L = xyz.shape[0]
    if is_motif is None:
        is_motif = torch.arange(L)

    xyz_t = ChemData().INIT_CRDS.reshape(1,1,ChemData().NTOTAL,3).repeat(1,L,1,1) + torch.rand(1,L,1,3)*random_noise

    t1d = torch.cat((
        torch.nn.functional.one_hot(
            torch.full((1, L), 20).long(),
            num_classes=ChemData().NAATOKENS-1).float(), # all gaps (no mask token)
        torch.zeros((1, L, 1)).float()
    ), -1) # (1, L_protein + L_sm, NAATOKENS)
    mask_t = torch.full((1, L, ChemData().NTOTAL), False)

    xyz_t[0, is_motif, :14] = xyz[is_motif, :14]
    xyz_t = torch.nan_to_num(xyz_t) # xyz has NaNs
    t1d[0, is_motif] = torch.cat((
        torch.nn.functional.one_hot(seq[is_motif], num_classes=ChemData().NAATOKENS-1).float(),
        torch.full((len(is_motif), 1), template_conf).float()
    ), -1) # (1, L_protein, NAATOKENS)
    mask_t[0, is_motif, :14] = mask[is_motif, :14]
    return xyz_t, t1d, mask_t

def generate_sm_template_feats(tplt_ids, resnames, akeys, Ls_sm, chid2smpartners, params):
    """
    based on the templates chosen for the protein, give templates for the small molecule
    there are 2 cases:
        1. The ligand in the template is identical to the query ligand. in this case we provide 
        the full coordinates of the template, with full template confidence
        2. The ligand in the template is not identical. In this case, the closest tanimoto hit is taken.
        the t1d features for seq are set to the ATM token and the template confidence is scaled to the 
        tanimoto similarity of the morgan fingerprint
    """
    sim, names = load_tanimoto_sim_matrix(base_path=params['SM_COMPL_DIR']) # could load this earlier...
    name2idx = dict(zip(names,range(len(names))))
    
    xyz_t_all_template = []
    f1d_t_all_template = []
    mask_t_all_template = []
    for chid in tplt_ids:
        # if chain does not have a small molecule, generate blank template
        if chid not in chid2smpartners:
            xyz_t, f1d_t, mask_t, _ = blank_template(1, sum(Ls_sm))
            xyz_t_all_template.append(xyz_t)
            f1d_t_all_template.append(f1d_t)
            mask_t_all_template.append(mask_t)
            continue
        
        chains, _, covale, _ = \
            pickle.load(gzip.open(params['MOL_DIR']+f'/{chid[1:3]}/{chid.split("_")[0]}.pkl.gz'))
        template_partners = chid2smpartners[chid]
        # only include partners in the ligands category, this will exclude metals 
        # TODO: add templating for metals
        template_partner_names = [ligand[0][2] for ligand in template_partners if ligand[0][2] in names \
            and ligand[0][2] is not None]
        
        if len(template_partner_names) == 0: # this is the case where all the template partners are metals or sugars
            xyz_t, f1d_t, mask_t, _ = blank_template(1, sum(Ls_sm))
            xyz_t_all_template.append(xyz_t)
            f1d_t_all_template.append(f1d_t)
            mask_t_all_template.append(mask_t)
            continue
        
        template_partner_sim_idxs = [name2idx[name] for name in template_partner_names]
        assert len(resnames) == len(Ls_sm), \
            f"length of ligand residue names and small molecule length do not match, length resnames \
                = {len(resnames)}, length Ls_sm = {len(Ls_sm)}"
        xyz_t_all_lig = []
        f1d_t_all_lig = []
        mask_t_all_lig = []

        for i, (lig_name, L) in enumerate(zip(resnames, Ls_sm)):            
            # lookup pairwise tanimoto sim
            # for each lig_partner, choose the closest tanimoto similar ligand in the template, 
            # without replacement
            
            if lig_name in template_partner_names:
                template_partner_idx = template_partner_names.index(lig_name)
                max_tanimoto = 1
                input_akeys = akeys[i]
            elif lig_name in names:
                lig_sim_all = sim[name2idx[lig_name]]
                lig_sim_template = lig_sim_all[template_partner_sim_idxs]
                template_partner_sorted_idxs = np.argsort(lig_sim_template)[::-1]
                template_partner_idx = None
                # really inelegant... done to sample without replacement and handle edge cases
                for idx in template_partner_sorted_idxs:
                    if template_partner_names[idx] is not None:
                        template_partner_idx = idx
                        max_tanimoto = lig_sim_template[idx]
                        break
                input_akeys = None
                if template_partner_idx is None: # case where more ligands in query than template
                    xyz_t, f1d_t, mask_t, _ = blank_template(1, L)
                    xyz_t_all_lig.append(xyz_t)
                    f1d_t_all_lig.append(f1d_t)
                    mask_t_all_lig.append(mask_t)
                    continue
                
            else: # query ligand not in the tanimoto db
                xyz_t, f1d_t, mask_t, _ = blank_template(1, L)
                xyz_t_all_lig.append(xyz_t)
                f1d_t_all_lig.append(f1d_t)
                mask_t_all_lig.append(mask_t)
                input_akeys = None
                continue
            
            # load that ligand from the cif file and create the xyz_t, t1d and mask_t
            ligand = template_partners[template_partner_idx]
            # remove that template partner for future iterations to sample without replacement
            template_partner_names[template_partner_idx] = None
            
            lig_atoms, lig_bonds = get_ligand_atoms_bonds(ligand, chains, covale)
            
            # templates are from asymmetric unit so we do not want to apply a transfrom
            # set up transforms to just be the identity matrix
            asmb_xfs = [(ligand[0], torch.eye(4))]
            ch2xf = {ligand[0]:0}
            #HACK: need to reindex the akeys to be the chain id, res id of the template ligand
            if input_akeys is not None:
                input_akeys = [tuple(list(ligand[0][:2]) + list(key[2:])) for key in input_akeys]

            try:
                xyz_, occ_, msa_, chid_, akeys_ = cif_ligand_to_xyz(lig_atoms, asmb_xfs, ch2xf, input_akeys=input_akeys)
            except Exception as e:
                # this is expected to fail if the template ligand is on multiple chains
                print(e)
                xyz_t, f1d_t, mask_t, _ = blank_template(1, L)
                xyz_t_all_lig.append(xyz_t)
                f1d_t_all_lig.append(f1d_t)
                mask_t_all_lig.append(mask_t)
                input_akeys = None
                continue
            
            # if we did not supply input_akeys we do not want to use the order of the template
            # we will collapse the coordinates to their unweighted com and set the sequence to be ATM
            if input_akeys is None or len(akeys_) != len(input_akeys): # length of template is different from ground truth, need to remake tensors to match
                ligand_com = torch.mean(xyz_, dim=0)
                xyz_ = torch.zeros((L, 3))
                xyz_[:] = ligand_com + torch.rand(3) # add a little noise to avoid learning templates are at the com

                occ_ = torch.full((L,), True)
                #HACK templates have NTOKENS-1 classes (no mask token) but the ATM token appears after the 
                # mask token so need to decrement the token number by 1
                msa_ = torch.full((L,), ChemData().aa2num["ATM"] - 1 )
            else:
                assert input_akeys == akeys_, "if provided input akeys, output akeys must match"

            # convert coordinates into L,36,3 and mask into L,36 to feed into spoof template
            xyz_sm, mask_sm = expand_xyz_sm_to_ntotal(xyz_[None], occ_[None])

            xyz_t, f1d_t, mask_t = spoof_template(xyz_sm, msa_.long(), mask_sm, template_conf=max_tanimoto)
            xyz_t_all_lig.append(xyz_t)
            f1d_t_all_lig.append(f1d_t)
            mask_t_all_lig.append(mask_t)
            
        # cat in length dimension (1)
        xyz_t_all_template.append(torch.cat(xyz_t_all_lig, dim=1))
        f1d_t_all_template.append(torch.cat(f1d_t_all_lig, dim=1))
        mask_t_all_template.append(torch.cat(mask_t_all_lig, dim=1))
    
    # cat in template dimension (0)
    xyz_t_all_template = torch.cat(xyz_t_all_template, dim=0)
    f1d_t_all_template = torch.cat(f1d_t_all_template, dim=0)
    mask_t_all_template = torch.cat(mask_t_all_template, dim=0)
    return xyz_t_all_template, f1d_t_all_template, mask_t_all_template

def generate_xyz_prev(xyz_t, mask_t, params):
    """
    allows you to use different initializations for the coordinate track specified in params
    """
    L = xyz_t.shape[1]
    if params["BLACK_HOLE_INIT"]:
        xyz_t, _, mask_t = blank_template(1, L)
    return xyz_t[0].clone(), mask_t[0].clone()


def _load_df(filename, pad_hash=True, eval_cols=[]):
    """load dataframe, zero-pad hash string, parse columns as python objects"""
    df = pd.read_csv(filename, na_filter=False)  # prevents chain "NA" loading as NaN
    if pad_hash:  # restore leading zeros, make into string
        df["HASH"] = df["HASH"].apply(lambda x: f"{x:06d}")
    for col in eval_cols:
        df[col] = df[col].apply(
            lambda x: ast.literal_eval(x)
        )  # interpret as list of strings
    return df

def params_match_pickle(
    loader_params: Dict[str, Any],
    data: Dict[str, Any],
    match_keys: List[str] = ["ligands_to_remove", "weight_sm_compl_by_seq_len", "sm_compl_cluster_method"],
) -> bool:
    """
    Check if the parameters used to generate the data in the pickle file match the
    parameters in the current run. This is useful for checking if the data in the pickle
    file is still valid for the current run.

    Args:
        loader_params (Dict[str, Any]): The parameters used to load the data.
        data (Dict[str, Any]): The data loaded from the pickle file.
        match_keys (List[str], optional): The keys to check for matching. Defaults to ["ligands_to_remove"].

    Returns:
        bool: True if the parameters match, False otherwise.
    """
    for key in match_keys:
        if key not in data and key not in loader_params:
            continue
        elif key not in data or key not in loader_params:
            return False
        elif data[key] != loader_params[key]:
            return False
    return True


def get_train_valid_set(loader_params, NEG_CLUSID_OFFSET=1000000, no_match_okay=False, diffusion_training=False):
    """Loads training/validation sets as pandas DataFrames and returns them in
    dictionaries keyed by their dataset names.

    Parameters
    ----------
    params : dict
        Config info with paths to various data csv files
    NEG_CLUSID_OFFSET : int
        Offset to add to cluster IDs of negative (compl, NA compl) examples to
        make them distinct from positive examples
    no_match_okay : bool
        If True, will not check that data pickle was loaded using the same
        parameters as current training run.
    diffusion_training : bool
        Modifies loaded datasets for diffusion training (as opposed to
        structure prediction). 

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
    loader_params = {k:v for k,v in loader_params.items() if k not in ignore}

    # try to load cached datasets
    if os.path.exists(loader_params["DATAPKL"]):
        with open(loader_params["DATAPKL"], "rb") as f:
            if "SLURM_PROCID" in os.environ and int(os.environ["SLURM_PROCID"]) == 0:
                print(f"Loading cached dataset from {loader_params['DATAPKL']}...")
            data = pickle.load(f)
            if type(data) is dict:
                if no_match_okay or params_match_pickle(loader_params, data):
                    return (
                        data["train_ID_dict"],
                        data["valid_ID_dict"],
                        data["weights_dict"],
                        data["train_dict"],
                        data["valid_dict"],
                        data["homo"],
                        data["chid2hash"],
                        data["chid2taxid"],
                        data["chid2smpartners"],
                    )
                else:
                    print("Stored dataset does not match config. Regenerating...")
            elif isinstance(data, tuple):
                train_ID_dict, valid_ID_dict, weights_dict, \
                    train_dict, valid_dict, homo, chid2hash, chid2taxid, *extra = data
                return train_ID_dict, valid_ID_dict, weights_dict, train_dict, valid_dict, homo, chid2hash, chid2taxid, *extra
            else:
                print(
                    "Stored dataset is not a dictionary or tuple, which means you are probably working with an outdated version of the dataset. Regenerating..."
                )
    else:
        print(
            f"Cached train/valid datasets {loader_params['DATAPKL']} not found. Re-parsing train/valid metadata..."
        )

    t0 = time.time()

    # helper functions
    def _apply_date_res_cutoffs(df):
        """filter dataframe by date and resolution cutoffs"""
        return df[(df.RESOLUTION <= loader_params['RESCUT']) & 
                  (df.DEPOSITION.apply(lambda x: parser.parse(x)) <= parser.parse(loader_params['DATCUT']))]

    def _get_IDs_weights(df):
        """return unique cluster IDs and AF2-style sampling weights based on seq length"""
        tmp_df = df.drop_duplicates('CLUSTER')
        IDs = tmp_df.CLUSTER.values
        weights = (1/512.)*np.clip(tmp_df.LEN_EXIST.values, 256, 512)
        return IDs, torch.tensor(weights)

    # fd remove "bad" ligands from the training/validation sets
    def _apply_lig_exclusions(df, excl):
        """filter dataframe by residue exclusions.  if ANY res in multires is excluded, all is."""
        ids=[tuple(y[-1] for y in x) for x in df['LIGAND'].tolist()]
        mask=[not any([x in excl for x in I]) for I in ids]
        return df[mask]

    # containers for returning the training data/metadata
    train_dict, valid_dict, train_ID_dict, valid_ID_dict, weights_dict = \
        OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()

    # validation IDs for PDB set
    val_pdb_ids = set([int(l) for l in open(loader_params['VAL_PDB']).readlines()])
    val_compl_ids = set([int(l) for l in open(loader_params['VAL_COMPL']).readlines()])
    val_neg_ids = set([int(l)+NEG_CLUSID_OFFSET for l in open(loader_params['VAL_NEG']).readlines()])
    val_rna_pdb_ids = set([l.rstrip() for l in open(loader_params['VAL_RNA']).readlines()])
    val_dna_pdb_ids = set([l.rstrip() for l in open(loader_params['VAL_DNA']).readlines()])
    val_tf_ids = set([int(l) for l in open(loader_params['VAL_TF']).readlines()])
    test_sm_ids = set([int(l) for l in open(loader_params['TEST_SM']).readlines()])

    # pdb monomers
    pdb = _load_df(loader_params['PDB_LIST'])
    pdb = _apply_date_res_cutoffs(pdb)
    if loader_params['MAXMONOMERLENGTH'] is not None:
        pdb = pdb[pdb["LEN_EXIST"] < loader_params['MAXMONOMERLENGTH']]
        pdb = pdb[pdb["LEN_EXIST"]>60]
    train_dict['pdb'] = pdb[(~pdb.CLUSTER.isin(val_pdb_ids)) & (~pdb.CLUSTER.isin(test_sm_ids))]
    valid_dict['pdb'] = pdb[pdb.CLUSTER.isin(val_pdb_ids) & (~pdb.CLUSTER.isin(test_sm_ids))]
    val_hash = set(valid_dict['pdb'].HASH.values)
    train_ID_dict['pdb'], weights_dict['pdb'] = _get_IDs_weights(train_dict['pdb'])
    valid_ID_dict['pdb'] = valid_dict['pdb'].CLUSTER.drop_duplicates().values    

    pdb_metadata = _load_df(loader_params['PDB_METADATA'])
    chid2hash = dict(zip(pdb_metadata.CHAINID, pdb_metadata.HASH))
    tmp = pdb_metadata.dropna(subset=['TAXID'])
    chid2taxid = dict(zip(tmp.CHAINID, tmp.TAXID))

    # short dslf loops
    dslf = pd.read_csv(loader_params['DSLF_LIST'])
    tmp_df = pdb[ pdb.CHAINID.isin(dslf.CHAIN_A)]
    valid_dict['dslf'] = dslf.merge(tmp_df[['CHAINID','HASH','CLUSTER']], 
                                    left_on='CHAIN_A', right_on='CHAINID', how='right')
    valid_ID_dict['dslf'] = valid_dict['dslf'].CLUSTER.drop_duplicates().values

    dslf_fb = pd.read_csv(loader_params['DSLF_FB_LIST'])

    # homo-oligomers
    homo = pd.read_csv(loader_params['HOMO_LIST'])
    tmp_df = pdb[pdb.CLUSTER.isin(val_pdb_ids) & 
                 (pdb.CHAINID.isin(homo['CHAIN_A'])) & 
                 (~pdb.CLUSTER.isin(test_sm_ids))]
    valid_dict['homo'] = homo.merge(tmp_df[['CHAINID','HASH','CLUSTER']], 
                                    left_on='CHAIN_A', right_on='CHAINID', how='right')
    valid_ID_dict['homo'] = valid_dict['homo'].CLUSTER.drop_duplicates().values

    # facebook AF2 distillation set
    fb = pd.read_csv(loader_params['FB_LIST'])
    fb = fb.rename(columns={'#CHAINID':'CHAINID'})
    fb = fb[(fb.plDDT>80) & (fb.SEQUENCE.apply(len) > 200)]
    fb['LEN_EXIST'] = fb.SEQUENCE.apply(len)

    # upweight clusters containing disulfide loop cases
    dslf_loops = fb[fb.CHAINID.isin(dslf_fb.CHAIN_A)]
    dslf_loops_clusters = dslf_loops.CLUSTER.unique()
    to_upweight = fb.CLUSTER.isin(dslf_loops_clusters)
    fb['HAS_DSLF_LOOP'] = to_upweight
    train_dict['fb'] = fb    
    train_ID_dict['fb'], weights_dict['fb'] = _get_IDs_weights(train_dict['fb'])

    # pdb hetero complexes
    compl = pd.read_csv(loader_params['COMPL_LIST'],skiprows=1,header=None)
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
    neg = pd.read_csv(loader_params['NEGATIVE_LIST'])
    neg = _apply_date_res_cutoffs(neg)
    neg['CLUSTER'] = neg.CLUSTER + NEG_CLUSID_OFFSET
    neg['HASH_A'] = neg.HASH.apply(lambda x: x.split('_')[0])
    neg['HASH_B'] = neg.HASH.apply(lambda x: x.split('_')[1])
    neg['LEN'] = neg['LENA:B'].apply(lambda x: [int(y) for y in x.split(':')])
    neg['LEN_EXIST'] = neg['LEN'].apply(lambda x: sum(x))

    valid_dict['neg_compl'] = neg[neg.CLUSTER.isin(val_neg_ids)]
    train_dict['neg_compl'] = neg[(~neg.CLUSTER.isin(val_neg_ids)) &
                            (~neg.HASH_A.isin(val_hash)) &
                            (~neg.HASH_B.isin(val_hash))]
    train_ID_dict['neg_compl'], weights_dict['neg_compl'] = _get_IDs_weights(train_dict['neg_compl'])
    valid_ID_dict['neg_compl'] = valid_dict['neg_compl'].CLUSTER.drop_duplicates().values

    # nucleic acid complexes
    na = _load_df(loader_params['NA_COMPL_LIST'])
    na = _apply_date_res_cutoffs(na)
    na['LEN'] = na['LENA:B:C:D'].apply(lambda x: [int(y) for y in x.split(':')])
    na['LEN_EXIST'] = na['LEN'].apply(lambda x: sum(x))
    na['TOPAD?'] = na['TOPAD?'].apply(lambda x: bool(x))

    train_dict['na_compl'] = na[(~na.CLUSTER.isin(val_compl_ids))]
    valid_dict['na_compl'] = na[na.CLUSTER.isin(val_compl_ids)]
    train_ID_dict['na_compl'], weights_dict['na_compl'] = _get_IDs_weights(train_dict['na_compl'])
    valid_ID_dict['na_compl'] = valid_dict['na_compl'].CLUSTER.drop_duplicates().values

    # negative nucleic acid complexes
    na_neg = _load_df(loader_params['NEG_NA_COMPL_LIST'])
    na_neg = _apply_date_res_cutoffs(na_neg)
    na_neg['CLUSTER'] = na_neg.CLUSTER + NEG_CLUSID_OFFSET

    na_neg['LEN'] = na_neg['LENA:B:C:D'].apply(lambda x: [int(y) for y in x.split(':')])
    na_neg['LEN_EXIST'] = na_neg['LEN'].apply(lambda x: sum(x))

    train_dict['neg_na_compl'] = na_neg[(~na_neg.CLUSTER.isin(val_neg_ids))]
    valid_dict['neg_na_compl'] = na_neg[na_neg.CLUSTER.isin(val_neg_ids)]
    train_ID_dict['neg_na_compl'], weights_dict['neg_na_compl'] = _get_IDs_weights(train_dict['neg_na_compl'])
    valid_ID_dict['neg_na_compl'] = valid_dict['neg_na_compl'].CLUSTER.drop_duplicates().values

    # dna-protein distillation (from TF data) (RM)
    distil_tf = _load_df(loader_params['TF_DISTIL_LIST'])
    distil_tf['CLUSTER'] = distil_tf['cluster_id']
    distil_tf['LEN'] = [ 
            [int(row['Domain size']), int(row['DNA size']), int(row['DNA size'])] if row['oligo'] == 'monomer' 
            else [int(row['Domain size']), int(row['Domain size']), int(row['DNA size']), int(row['DNA size'])]
            for _, row in distil_tf.iterrows() 
            ]
    distil_tf['LEN_EXIST'] = distil_tf['LEN'].apply(lambda x: sum(x))

    train_dict['distil_tf'] = distil_tf[~distil_tf.CLUSTER.isin(val_tf_ids)]
    valid_dict['distil_tf'] = distil_tf[distil_tf.CLUSTER.isin(val_tf_ids)]
    train_ID_dict['distil_tf'], weights_dict['distil_tf'] = _get_IDs_weights(train_dict['distil_tf']) 
    valid_ID_dict['distil_tf'] = valid_dict['distil_tf'].CLUSTER.drop_duplicates().values

    # sequence-only DNA/protein complexes (TF data) (RM)
    tf = _load_df(loader_params['TF_COMPL_LIST'])
    tf['CLUSTER'] = tf['cluster_id']
    tf['LEN'] = [
            [int(row['Domain size']), int(row['DNA size']), int(row['DNA size'])]
            for _, row in tf.iterrows()
            ]
    tf['LEN_EXIST'] = tf['LEN'].apply(lambda x: sum(x))

    train_dict['tf'] = tf[~tf.CLUSTER.isin(val_tf_ids)]
    valid_dict['tf'] = tf[tf.CLUSTER.isin(val_tf_ids)]
    train_ID_dict['tf'], weights_dict['tf'] = _get_IDs_weights(train_dict['tf'])
    valid_ID_dict['tf'] = valid_dict['tf'].CLUSTER.drop_duplicates().values

    train_dict['neg_tf'] = tf[~tf.CLUSTER.isin(val_tf_ids)]
    valid_dict['neg_tf'] = tf[tf.CLUSTER.isin(val_tf_ids)]
    train_ID_dict['neg_tf'], weights_dict['neg_tf'] = _get_IDs_weights(train_dict['neg_tf']) 
    valid_ID_dict['neg_tf'] = valid_dict['neg_tf'].CLUSTER.drop_duplicates().values

    # rna
    rna = pd.read_csv(loader_params['RNA_LIST'])
    rna = _apply_date_res_cutoffs(rna)
    rna['LEN'] = rna['LENA:B'].apply(lambda x: [int(y) for y in x.split(':')])
    rna['LEN_EXIST'] = rna['LEN'].apply(lambda x: sum(x))

    in_val = rna['CHAINID'].apply(lambda x: any([y in val_rna_pdb_ids for y in x.split(':')]))
    train_dict['rna'] = rna[~in_val]
    valid_dict['rna'] = rna[in_val]
    train_ID_dict['rna'], weights_dict['rna'] = _get_IDs_weights(train_dict['rna'])
    valid_ID_dict['rna'] = valid_dict['rna'].CLUSTER.drop_duplicates().values #fd

    # dna
    dna = pd.read_csv(loader_params['DNA_LIST'])
    dna = _apply_date_res_cutoffs(dna)
    dna['LEN'] = dna['LENA:B'].apply(lambda x: [int(y) for y in x.split(':')])
    dna['CLUSTER'] = range(len(dna)) # for unweighted sampling
    dna['LEN_EXIST'] = dna['LEN'].apply(lambda x: sum(x))

    in_val = dna['CHAINID'].apply(lambda x: any([y in val_dna_pdb_ids for y in x.split(':')]))
    train_dict['dna'] = dna[~in_val]
    valid_dict['dna'] = dna[in_val]
    train_ID_dict['dna'], weights_dict['dna'] = _get_IDs_weights(train_dict['dna'])
    valid_ID_dict['dna'] = valid_dict['dna'].CLUSTER.drop_duplicates().values #fd

    # protein-small molecule complexes
    def _prep_sm_compl_data(df):
        """repeated operations for protein / small molecule datasets"""
        # don't use partially unresolved ligands for diffusion training
        if diffusion_training:
            df = df[df['LIGATOMS']==df['LIGATOMS_RESOLVED']]

        train_df = df[~df.CLUSTER.isin(val_pdb_ids)]
        valid_df = df[df.CLUSTER.isin(val_pdb_ids)]

        if loader_params.get("weight_sm_compl_by_seq_len", True):
            seq_len_factor = (1/512.)*np.clip(df.LEN_EXIST, 256, 512) # standard seq length weighting
            df.loc[:,'WEIGHT'] = seq_len_factor # can potentially include other factors (ligand cluster size, etc)
        else:
            df["WEIGHT"] = 1.0

        df_clus = df[['CLUSTER','WEIGHT']].groupby('CLUSTER').mean().reset_index()
        clus2weight = dict(zip(df_clus.CLUSTER, df_clus.WEIGHT))

        train_IDs = train_df.CLUSTER.drop_duplicates().values
        weights = [clus2weight[i] for i in train_IDs]

        valid_IDs = valid_df.CLUSTER.drop_duplicates().values

        return train_df, valid_df, train_IDs, valid_IDs, torch.tensor(weights)

    # protein / small molecule complexes
    df_sm = _load_df(loader_params['SM_LIST'], eval_cols=['COVALENT','LIGAND','LIGXF','PARTNERS'])
    df_sm = _apply_date_res_cutoffs(df_sm)
    df_sm = _apply_lig_exclusions(df_sm, loader_params['ligands_to_remove'])
    # remove very big things
    #  (fd: only 80 examples are larger than 196 atoms, the majority are "not useful cases")
    df_sm = df_sm[df_sm['LIGATOMS']<=196] 

    df = df_sm[df_sm['SUBSET']=='organic']
    # optionally recluster the protein/small molecule complex examples

    cluster_type = loader_params.get("sm_compl_cluster_method", "by_protein_sequence")
    cluster_fn = cluster_factory[cluster_type]
    df = cluster_fn(df)

    train_dict['sm_compl'], valid_dict['sm_compl'], train_ID_dict['sm_compl'], \
        valid_ID_dict['sm_compl'], weights_dict['sm_compl'] = _prep_sm_compl_data(df)

    # protein / metal ion complexes
    df = df_sm[df_sm['SUBSET']=='metal']
    train_dict['metal_compl'], valid_dict['metal_compl'], train_ID_dict['metal_compl'], \
        valid_ID_dict['metal_compl'], weights_dict['metal_compl'] = _prep_sm_compl_data(df)

    # protein / multi-residue ligand complexes
    df = df_sm[df_sm['SUBSET']=='multi']
    train_dict['sm_compl_multi'], valid_dict['sm_compl_multi'], train_ID_dict['sm_compl_multi'], \
        valid_ID_dict['sm_compl_multi'], weights_dict['sm_compl_multi'] = _prep_sm_compl_data(df)

    # protein / covalent ligand complexes
    df = df_sm[df_sm['SUBSET']=='covale']
    train_dict['sm_compl_covale'], valid_dict['sm_compl_covale'], train_ID_dict['sm_compl_covale'], \
        valid_ID_dict['sm_compl_covale'], weights_dict['sm_compl_covale'] = _prep_sm_compl_data(df)

    # protein / ligand assemblies (more than 2 chains)
    df = df_sm[df_sm['SUBSET']=='asmb']
    train_dict['sm_compl_asmb'], valid_dict['sm_compl_asmb'], train_ID_dict['sm_compl_asmb'], \
        valid_ID_dict['sm_compl_asmb'], weights_dict['sm_compl_asmb'] = _prep_sm_compl_data(df)

    # strict protein / ligand validation set
    val_df = _load_df(loader_params['VAL_SM_STRICT'], loader_params, eval_cols=['LIGAND','LIGXF','PARTNERS'])
    val_df = _apply_date_res_cutoffs(val_df)
    valid_dict['sm_compl_strict'] = val_df
    valid_ID_dict['sm_compl_strict'] = val_df.CLUSTER.drop_duplicates().values

    # rk want to provide ligand context in templates
    # for each unique protein chain map to all the query ligand partners in the dataset
    chid2smpartners = df_sm.groupby("CHAINID").agg(lambda x: [val for val in x])["LIGAND"].to_dict()

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
    sm = _load_df(loader_params['CSD_LIST'], pad_hash=False, eval_cols=['sim','sim_valid','sim_test'])
    sim_idx = int(loader_params["MAXSIM"]*100-50)
    sm = sm[
        (sm['r_factor'] <= loader_params['RMAX']) &
        (sm['nres'] <= loader_params['MAXRES']) &
        (sm['nheavy'] <= loader_params['MAXATOMS']) &
        (sm['nheavy'] >= loader_params['MINATOMS']) &
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
    with open(loader_params['DATAPKL'], "wb") as f:
        print ('Writing',loader_params['DATAPKL'],'...')
        data = {
            'train_ID_dict':train_ID_dict, 
            'valid_ID_dict':valid_ID_dict, 
            'weights_dict':weights_dict, 
            'train_dict':train_dict, 
            'valid_dict':valid_dict, 
            'homo':homo, 
            'chid2hash':chid2hash, 
            'chid2taxid':chid2taxid, 
            'chid2smpartners':chid2smpartners,
        }
        data.update(loader_params)
        pickle.dump(data, f)
        print ('...done')

    return train_ID_dict, valid_ID_dict, weights_dict, train_dict, valid_dict, \
        homo, chid2hash, chid2taxid, chid2smpartners


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

    elif all([not x['paired'] for x in msas_to_load]): # all unpaired, tile diagonally
        a3m = dict(msa=torch.tensor([[]]), ins=torch.tensor([[]]))
        for msa_vals in msas_to_load:
            a3m_ = get_msa(msa_vals["path"], msa_vals["hash"])
            L_s = [a3m['msa'].shape[1], a3m_['msa'].shape[1]]
            a3m = merge_a3m_hetero(a3m, a3m_, L_s)

    else: # heteromer case (at least two different MSAs will handle things like AB, AAB, ABC...)
        a3m_list = []
        L_s = []
        for i in range(len(msa_hashes)):
            msa_vals = msas_to_load[i]
            msa, ins, taxID = parse_a3m(msa_vals["path"], paired=msa_vals["paired"])
            msa = msa[:, msa_vals["seq_range"][0]:msa_vals["seq_range"][1]]
            ins = ins[:, msa_vals["seq_range"][0]:msa_vals["seq_range"][1]]
            a3m_list.append({"msa":torch.tensor(msa).long(), "ins":torch.tensor(ins).long(), 
                             "taxID":taxID, "hash":msa_vals["hash"]})
            L_s.append(msa_vals["seq_range"][1]-msa_vals["seq_range"][0])
        msaA, insA = merge_msas(a3m_list, L_s)
        a3m = {"msa": msaA, "ins": insA}
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

    a3m = {'msa': msa, 'ins': ins}

    # merge taxids
    if 'taxid' in a3mA and 'taxid' in a3mB:
        a3m['taxid'] = np.concatenate([np.array(a3mA['taxid']), np.array(a3mB['taxid'])[1:]])

    return a3m

# merge msa & insertion statistics of units in homo-oligomers
def merge_a3m_homo(msa_orig, ins_orig, nmer, mode="default"):
     N, L = msa_orig.shape[:2]
     if mode == "repeat":

         # AAAAAA
         # AAAAAA

         msa = torch.tile(msa_orig,(1,nmer))
         ins = torch.tile(ins_orig,(1,nmer))

     elif mode == "diag":

         # AAAAAA
         # A-----
         # -A----
         # --A---
         # ---A--
         # ----A-
         # -----A

         N = N - 1
         new_N = 1 + N * nmer
         new_L = L * nmer
         msa = torch.full((new_N, new_L), 20, dtype=msa_orig.dtype, device=msa_orig.device)
         ins = torch.full((new_N, new_L), 0, dtype=ins_orig.dtype, device=msa_orig.device)

         start_L = 0
         start_N = 1
         for i_c in range(nmer):
             msa[0, start_L:start_L+L] = msa_orig[0] 
             msa[start_N:start_N+N, start_L:start_L+L] = msa_orig[1:]
             ins[0, start_L:start_L+L] = ins_orig[0]
             ins[start_N:start_N+N, start_L:start_L+L] = ins_orig[1:]
             start_L += L
             start_N += N
     else:

         # AAAAAA
         # A-----
         # -AAAAA

         msa = torch.full((2*N-1, L*nmer), 20, dtype=msa_orig.dtype, device=msa_orig.device)
         ins = torch.full((2*N-1, L*nmer), 0, dtype=ins_orig.dtype, device=msa_orig.device)

         msa[:N, :L] = msa_orig
         ins[:N, :L] = ins_orig
         start = L

         for i_c in range(1,nmer):
             msa[0, start:start+L] = msa_orig[0] 
             msa[N:, start:start+L] = msa_orig[1:]
             ins[0, start:start+L] = ins_orig[0]
             ins[N:, start:start+L] = ins_orig[1:]
             start += L        

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
            L_s_to_merge = [sum(L_s[:i]), L_s[i]]
            a3mA = merge_a3m_hetero(a3mA, a3mB, L_s_to_merge)
            msaA, insA = a3mA["msa"], a3mA["ins"]
            taxIDs.extend(a3mB["taxID"])
        else:
            final_pairsA = []
            final_pairsB = []
            msaB, insB = a3mB["msa"], a3mB["ins"]
            for pair in pair_taxIDs:
                pair_a3mA = np.where(np.array(taxIDs)==pair)[0]
                pair_a3mB = np.where(a3mB["taxID"]==pair)[0]
                msaApair = torch.argmin(torch.sum(msaA[pair_a3mA, :] == msaA[0, :],axis=-1))
                msaBpair = torch.argmin(torch.sum(msaB[pair_a3mB, :] == msaB[0, :],axis=-1))
                final_pairsA.append(pair_a3mA[msaApair])
                final_pairsB.append(pair_a3mB[msaBpair])
            paired_msaB = torch.full((msaA.shape[0], L_s[i]), 20).long() # (N_seq_A, L_B)
            paired_msaB[final_pairsA] = msaB[final_pairsB]
            msaA = torch.cat([msaA, paired_msaB], dim=1)
            insA = torch.zeros_like(msaA) # paired MSAs in our dataset dont have insertions 
        seen.update(a3mB["hash"])
        
    return msaA, insA

def remove_all_gap_seqs(a3m):
    """Removes sequences that are all gaps from an MSA represented as `a3m` dictionary"""
    idx_seq_keep = ~(a3m['msa']==ChemData().UNKINDEX).all(dim=1)
    a3m['msa'] = a3m['msa'][idx_seq_keep]
    a3m['ins'] = a3m['ins'][idx_seq_keep]
    return a3m

def join_msas_by_taxid(a3mA, a3mB, idx_overlap=None):
    """Joins (or "pairs") 2 MSAs by matching sequences with the same
    taxonomic ID. If more than 1 sequence exists in both MSAs with the same tax
    ID, only the sequence with the highest sequence identity to the query (1st
    sequence in MSA) will be paired.
    
    Sequences that aren't paired will be padded and added to the bottom of the
    joined MSA.  If a subregion of the input MSAs overlap (represent the same
    chain), the subregion residue indices can be given as `idx_overlap`, and
    the overlap region of the unpaired sequences will be included in the joined
    MSA.
    
    Parameters
    ----------
    a3mA : dict
        First MSA to be joined, with keys `msa` (N_seq, L_seq), `ins` (N_seq,
        L_seq), `taxid` (N_seq,), and optionally `is_paired` (N_seq,), a
        boolean tensor indicating whether each sequence is fully paired. Can be
        a multi-MSA (contain >2 sub-MSAs).
    a3mB : dict
        2nd MSA to be joined, with keys `msa`, `ins`, `taxid`, and optionally
        `is_paired`. Can be a multi-MSA ONLY if not overlapping with 1st MSA.
    idx_overlap : tuple or list (optional)
        Start and end indices of overlap region in 1st MSA, followed by the
        same in 2nd MSA.

    Returns
    -------
    a3m : dict
        Paired MSA, with keys `msa`, `ins`, `taxid` and `is_paired`.
    """
    # preprocess overlap region
    L_A, L_B = a3mA['msa'].shape[1], a3mB['msa'].shape[1]
    if idx_overlap is not None:
        i1A, i2A, i1B, i2B = idx_overlap
        i1B_new, i2B_new = (0, i1B) if i2B==L_B else (i2B, L_B) # MSA B residues that don't overlap MSA A
        assert((i1B==0) or (i2B==a3mB['msa'].shape[1])), \
            "When overlapping with 1st MSA, 2nd MSA must comprise at most 2 sub-MSAs "\
            "(i.e. residue range should include 0 or a3mB['msa'].shape[1])"
    else:
        i1B_new, i2B_new = (0, L_B)
        
    # pair sequences
    taxids_shared = a3mA['taxid'][np.isin(a3mA['taxid'],a3mB['taxid'])]
    i_pairedA, i_pairedB = [], []
    
    for taxid in taxids_shared:
        i_match = np.where(a3mA['taxid']==taxid)[0]
        i_match_best = torch.argmin(torch.sum(a3mA['msa'][i_match]==a3mA['msa'][0], axis=1))
        i_pairedA.append(i_match[i_match_best])

        i_match = np.where(a3mB['taxid']==taxid)[0]
        i_match_best = torch.argmin(torch.sum(a3mB['msa'][i_match]==a3mB['msa'][0], axis=1))
        i_pairedB.append(i_match[i_match_best])

    # unpaired sequences
    i_unpairedA = np.setdiff1d(np.arange(a3mA['msa'].shape[0]), i_pairedA)
    i_unpairedB = np.setdiff1d(np.arange(a3mB['msa'].shape[0]), i_pairedB)
    N_paired, N_unpairedA, N_unpairedB = len(i_pairedA), len(i_unpairedA), len(i_unpairedB)

    # handle overlap region
    # if msa A consists of sub-MSAs 1,2,3 and msa B of 2,4 (i.e overlap region is 2),
    # this diagram shows how the variables below make up the final multi-MSA
    # (* denotes nongaps, - denotes gaps)
    #  1 2 3 4
    # |*|*|*|*|   msa_paired
    # |*|*|*|-|   msaA_unpaired
    # |-|*|-|*|   msaB_unpaired
    if idx_overlap is not None:
        assert((a3mA['msa'][i_pairedA, i1A:i2A]==a3mB['msa'][i_pairedB, i1B:i2B]) |
               (a3mA['msa'][i_pairedA, i1A:i2A]==ChemData().UNKINDEX)).all(),\
            'Paired MSAs should be identical (or 1st MSA should be all gaps) in overlap region'

        # overlap region gets sequences from 2nd MSA bc sometimes 1st MSA will be all gaps here
        msa_paired = torch.cat([a3mA['msa'][i_pairedA, :i1A],
                                a3mB['msa'][i_pairedB, i1B:i2B],
                                a3mA['msa'][i_pairedA, i2A:],
                                a3mB['msa'][i_pairedB, i1B_new:i2B_new] ], dim=1)
        msaA_unpaired = torch.cat([a3mA['msa'][i_unpairedA],
                                 torch.full((N_unpairedA, i2B_new-i1B_new), ChemData().UNKINDEX) ], dim=1)
        msaB_unpaired = torch.cat([torch.full((N_unpairedB, i1A), ChemData().UNKINDEX),
                                 a3mB['msa'][i_unpairedB, i1B:i2B],
                                 torch.full((N_unpairedB, L_A-i2A), ChemData().UNKINDEX),
                                 a3mB['msa'][i_unpairedB, i1B_new:i2B_new] ], dim=1)
    else:
        # no overlap region, simple offset pad & stack
        # this code is actually a special case of "if" block above, but writing
        # this out explicitly here to make the logic more clear
        msa_paired = torch.cat([a3mA['msa'][i_pairedA], a3mB['msa'][i_pairedB, i1B_new:i2B_new]], dim=1)
        msaA_unpaired = torch.cat([a3mA['msa'][i_unpairedA],
                                 torch.full((N_unpairedA, L_B), ChemData().UNKINDEX)], dim=1) # pad with gaps
        msaB_unpaired = torch.cat([torch.full((N_unpairedB, L_A), ChemData().UNKINDEX),
                                 a3mB['msa'][i_unpairedB]], dim=1) # pad with gaps

    # stack paired & unpaired
    msa = torch.cat([msa_paired, msaA_unpaired, msaB_unpaired], dim=0)
    taxids = np.concatenate([a3mA['taxid'][i_pairedA], a3mA['taxid'][i_unpairedA], a3mB['taxid'][i_unpairedB]])

    # label "fully paired" sequences (a row of MSA that was never padded with gaps)
    # output seq is fully paired if seqs A & B both started out as paired and were paired to
    # each other on tax ID. 
    # NOTE: there is a rare edge case that is ignored here for simplicity: if
    # pMSA 0+1 and 1+2 are joined and then joined to 2+3, a seq that exists in
    # 0+1 and 2+3 but NOT 1+2 will become fully paired on the last join but
    # will not be labeled as such here
    is_pairedA = a3mA['is_paired'] if 'is_paired' in a3mA else torch.ones((a3mA['msa'].shape[0],)).bool()
    is_pairedB = a3mB['is_paired'] if 'is_paired' in a3mB else torch.ones((a3mB['msa'].shape[0],)).bool()
    is_paired = torch.cat([is_pairedA[i_pairedA] & is_pairedB[i_pairedB],
                           torch.zeros((N_unpairedA + N_unpairedB,)).bool()])

    # insertion features in paired MSAs are assumed to be zero
    a3m = dict(msa=msa, ins=torch.zeros_like(msa), taxid=taxids, is_paired=is_paired)
    return a3m


def load_minimal_multi_msa(hash_list, taxid_list, Ls, params):
    """Load a multi-MSA, which is a MSA that is paired across more than 2
    chains. This loads the MSA for unique chains. Use 'expand_multi_msa` to
    duplicate portions of the MSA for homo-oligomer repeated chains.

    Given a list of unique MSA hashes, loads all MSAs (using paired MSAs where
    it can) and pairs sequences across as many sub-MSAs as possible by matching
    taxonomic ID. For details on how pairing is done, see
    `join_msas_by_taxid()`

    Parameters
    ----------
    hash_list : list of str 
        Hashes of MSAs to load and join. Must not contain duplicates.
    taxid_list : list of str
        Taxonomic IDs of query sequences of each input MSA.
    Ls : list of int
        Lengths of the chains corresponding to the hashes.

    Returns
    -------
    a3m_out : dict
        Multi-MSA with all input MSAs. Keys: `msa`,`ins` [torch.Tensor (N_seq, L)], 
        `taxid` [np.array (Nseq,)], `is_paired` [torch.Tensor (N_seq,)]
    hashes_out : list of str
        Hashes of MSAs in the order that they are joined in `a3m_out`.
        Contains the same elements as the input `hash_list` but may be in a
        different order.
    Ls_out : list of int
        Lengths of each chain in `a3m_out`
    """
    assert(len(hash_list)==len(set(hash_list))), 'Input MSA hashes must be unique'

    # the lists below are constructed such that `a3m_list[i_a3m]` is a multi-MSA
    # comprising sub-MSAs whose indices in the input lists are 
    # `i_in = idx_list_groups[i_a3m][i_submsa]`, i.e. the sub-MSA hashes are
    # `hash_list[i_in]` and lengths are `Ls[i_in]`.
    # Each sub-MSA spans a region of its multi-MSA `a3m_list[i_a3m][:,i_start:i_end]`, 
    # where `(i_start,i_end) = res_range_groups[i_a3m][i_submsa]`
    a3m_list = []         # list of multi-MSAs
    idx_list_groups = []  # list of lists of indices of input chains making up each multi-MSA
    res_range_groups = [] # list of lists of start and end residues of each sub-MSA in multi-MSA

    # iterate through all pairs of hashes and look for paired MSAs (pMSAs)
    # NOTE: in the below, if pMSAs are loaded for hashes 0+1 and then 2+3, and
    # later a pMSA is found for 0+2, the last MSA will not be loaded. The 0+1
    # and 2+3 pMSAs will still be joined on taxID at the end, but sequences
    # only present in the 0+2 pMSA pMSAs will be missed. this is probably very
    # rare and so is ignored here for simplicity.
    N = len(hash_list)
    for i1, i2 in itertools.permutations(range(N),2):

        idx_list = [x for group in idx_list_groups for x in group] # flattened list of loaded hashes
        if i1 in idx_list and i2 in idx_list: continue # already loaded
        if i1 == '' or i2 == '': continue # no taxID means no pMSA

        # a paired MSA exists
        if taxid_list[i1]==taxid_list[i2]:
            
            h1, h2 = hash_list[i1], hash_list[i2]
            fn = params['COMPL_DIR']+'/pMSA/'+h1[:3]+'/'+h2[:3]+'/'+h1+'_'+h2+'.a3m.gz'

            if os.path.exists(fn):
                msa, ins, taxid = parse_a3m(fn, paired=True)
                a3m_new = dict(msa=torch.tensor(msa), ins=torch.tensor(ins), taxid=taxid,
                               is_paired=torch.ones(msa.shape[0]).bool())
                res_range1 = (0,Ls[i1])
                res_range2 = (Ls[i1],msa.shape[1])

                # both hashes are new, add paired MSA to list
                if i1 not in idx_list and i2 not in idx_list:
                    a3m_list.append(a3m_new)
                    idx_list_groups.append([i1,i2])
                    res_range_groups.append([res_range1, res_range2])

                # one of the hashes is already in a multi-MSA
                # find that multi-MSA and join the new pMSA to it
                elif i1 in idx_list:
                    # which multi-MSA & sub-MSA has the hash with index `i1`?
                    i_a3m = np.where([i1 in group for group in idx_list_groups])[0][0]
                    i_submsa = np.where(np.array(idx_list_groups[i_a3m])==i1)[0][0]
                    
                    idx_overlap = res_range_groups[i_a3m][i_submsa] + res_range1
                    a3m_list[i_a3m] = join_msas_by_taxid(a3m_list[i_a3m], a3m_new, idx_overlap)
                    
                    idx_list_groups[i_a3m].append(i2)
                    L = res_range_groups[i_a3m][-1][1] # length of current multi-MSA
                    L_new = res_range2[1] - res_range2[0]
                    res_range_groups[i_a3m].append((L, L+L_new))

                elif i2 in idx_list:
                    # which multi-MSA & sub-MSA has the hash with index `i2`?
                    i_a3m = np.where([i2 in group for group in idx_list_groups])[0][0]
                    i_submsa = np.where(np.array(idx_list_groups[i_a3m])==i2)[0][0]
                    
                    idx_overlap = res_range_groups[i_a3m][i_submsa] + res_range2
                    a3m_list[i_a3m] = join_msas_by_taxid(a3m_list[i_a3m], a3m_new, idx_overlap)
                    
                    idx_list_groups[i_a3m].append(i1)
                    L = res_range_groups[i_a3m][-1][1] # length of current multi-MSA
                    L_new = res_range1[1] - res_range1[0]
                    res_range_groups[i_a3m].append((L, L+L_new))
                    
    # add unpaired MSAs
    # ungroup hash indices now, since we're done making multi-MSAs
    idx_list = [x for group in idx_list_groups for x in group]
    for i in range(N):
        if i not in idx_list:
            fn = params['PDB_DIR'] + '/a3m/' + hash_list[i][:3] + '/' + hash_list[i] + '.a3m.gz'
            msa, ins, taxid = parse_a3m(fn)
            a3m_new = dict(msa=torch.tensor(msa), ins=torch.tensor(ins), 
                           taxid=taxid, is_paired=torch.ones(msa.shape[0]).bool())
            a3m_list.append(a3m_new)
            idx_list.append(i)
            
    Ls_out = [Ls[i] for i in idx_list]
    hashes_out = [hash_list[i] for i in idx_list]
            
    # join multi-MSAs & unpaired MSAs
    a3m_out = a3m_list[0]
    for i in range(1, len(a3m_list)):
        a3m_out = join_msas_by_taxid(a3m_out, a3m_list[i])

    return a3m_out, hashes_out, Ls_out    


def expand_multi_msa(a3m, hashes_in, hashes_out, Ls_in, Ls_out, params):
    """Expands a multi-MSA of unique chains into an MSA of a
    hetero-homo-oligomer in which some chains appear more than once. The query
    sequences (1st sequence of MSA) are concatenated directly along the
    residue dimention. The remaining sequences are offset-tiled (i.e. "padded &
    stacked") so that exact repeat sequences aren't paired.

    For example, if the original multi-MSA contains unique chains 1,2,3 but
    the final chain order is 1,2,1,3,3,1, this function will output an MSA like
    (where - denotes a block of gap characters):

        1 2 - 3 - -
        - - 1 - 3 -
        - - - - - 1

    Parameters
    ----------
    a3m : dict
        Contains torch.Tensors `msa` and `ins` (N_seq, L) and np.array `taxid` (Nseq,),
        representing the multi-MSA of unique chains.
    hashes_in : list of str
        Unique MSA hashes used in `a3m`.
    hashes_out : list of str
        Non-unique MSA hashes desired in expanded MSA.
    Ls_in : list of int
        Lengths of each chain in `a3m`
    Ls_out : list of int
        Lengths of each chain desired in expanded MSA.
    params : dict
        Data loading parameters

    Returns
    -------
    a3m : dict
        Contains torch.Tensors `msa` and `ins` of expanded MSA. No
        taxids because no further joining needs to be done.
    """
    assert(len(hashes_out)==len(Ls_out))
    assert(set(hashes_in)==set(hashes_out))
    assert(a3m['msa'].shape[1]==sum(Ls_in))

    # figure out which oligomeric repeat is represented by each hash in `hashes_out`
    # each new repeat will be offset in sequence dimension of final MSA
    counts = dict()
    n_copy = [] # n-th copy of this hash in `hashes`
    for h in hashes_out:
        if h in counts:
            counts[h] += 1
        else:
            counts[h] = 1
        n_copy.append(counts[h])

    # num sequences in source & destination MSAs
    N_in = a3m['msa'].shape[0]
    N_out = (N_in-1)*max(n_copy)+1 # concatenate query seqs, pad&stack the rest

    # source MSA
    msa_in, ins_in = a3m['msa'], a3m['ins']

    # initialize destination MSA to gap characters
    msa_out = torch.full((N_out, sum(Ls_out)), ChemData().UNKINDEX)
    ins_out = torch.full((N_out, sum(Ls_out)), 0)

    # for each destination chain
    for i_out, h_out in enumerate(hashes_out):
        # identify index of source chain
        i_in = np.where(np.array(hashes_in)==h_out)[0][0]

        # residue indexes
        i1_res_in = sum(Ls_in[:i_in])
        i2_res_in = sum(Ls_in[:i_in+1])
        i1_res_out = sum(Ls_out[:i_out])
        i2_res_out = sum(Ls_out[:i_out+1])

        # copy over query sequence
        msa_out[0, i1_res_out:i2_res_out] = msa_in[0, i1_res_in:i2_res_in]
        ins_out[0, i1_res_out:i2_res_out] = ins_in[0, i1_res_in:i2_res_in]

        # offset non-query sequences along sequence dimension based on repeat number of a given hash
        i1_seq_out = 1+(n_copy[i_out]-1)*(N_in-1)
        i2_seq_out = 1+n_copy[i_out]*(N_in-1)
        # copy over non-query sequences
        msa_out[i1_seq_out:i2_seq_out, i1_res_out:i2_res_out] = msa_in[1:, i1_res_in:i2_res_in]
        ins_out[i1_seq_out:i2_seq_out, i1_res_out:i2_res_out] = ins_in[1:, i1_res_in:i2_res_in]

    # only 1st oligomeric repeat can be fully paired
    is_paired_out = torch.cat([a3m['is_paired'], torch.zeros((N_out-N_in,)).bool()]) 

    a3m_out = dict(msa=msa_out, ins=ins_out, is_paired=is_paired_out)
    a3m_out = remove_all_gap_seqs(a3m_out)

    return a3m_out

def load_multi_msa(chain_ids, Ls, chid2hash, chid2taxid, params):
    """Loads multi-MSA for an arbitrary number of protein chains. Tries to
    locate paired MSAs and pair sequences across all chains by taxonomic ID.
    Unpaired sequences are padded and stacked on the bottom.
    """
    # get MSA hashes (used to locate a3m files) and taxonomic IDs (used to determine pairing)
    hashes = []
    hashes_unique = []
    taxids_unique = []
    Ls_unique = []
    for chid,L_ in zip(chain_ids, Ls):
        hashes.append(chid2hash[chid])
        if chid2hash[chid] not in hashes_unique:
            hashes_unique.append(chid2hash[chid])
            taxids_unique.append(chid2taxid.get(chid))
            Ls_unique.append(L_)

    # loads multi-MSA for unique chains
    a3m_prot, hashes_unique, Ls_unique = \
        load_minimal_multi_msa(hashes_unique, taxids_unique, Ls_unique, params)

    # expands multi-MSA to repeat chains of homo-oligomers
    a3m_prot = expand_multi_msa(a3m_prot, hashes_unique, hashes, Ls_unique, Ls, params)

    return a3m_prot

def choose_multimsa_clusters(msa_seq_is_paired, params):
    """Returns indices of fully-paired sequences in a multi-MSA to use as seed
    clusters during MSA featurization.
    """
    frac_paired = msa_seq_is_paired.float().mean()
    if frac_paired > 0.25: # enough fully paired sequences, just let MSAFeaturize choose randomly
        return None
    else:
        # ensure that half of the clusters are fully-paired sequences,
        # and let the rest be chosen randomly
        N_seed = params['MAXLAT']//2
        msa_seed_clus = []
        for i_cycle in range(params['MAXCYCLE']):
            idx_paired = torch.where(msa_seq_is_paired)[0]
            msa_seed_clus.append(idx_paired[torch.randperm(len(idx_paired))][:N_seed])
        return msa_seed_clus


# fd
def get_bond_distances(bond_feats):
    atom_bonds = (bond_feats > 0)*(bond_feats<5)
    dist_matrix = scipy.sparse.csgraph.shortest_path(atom_bonds.long().numpy(), directed=False)
    # dist_matrix = torch.tensor(np.nan_to_num(dist_matrix, posinf=4.0)) # protein portion is inf and you don't want to mask it out
    return torch.from_numpy(dist_matrix).float()

# Generate input features for single-chain
def featurize_single_chain(msa, ins, tplt, pdb, params, unclamp=False, pick_top=True, random_noise=5.0, fixbb=False, p_short_crop=0.0, p_dslf_crop=0.0):
    msa_featurization_kwargs = {}
    if fixbb:
#        ic('setting msa feat kwargs')
        msa_featurization_kwargs['p_mask'] = 0.0

    # get ground-truth structures
    idx = torch.arange(len(pdb['xyz'])) 
    xyz = torch.full((len(idx),ChemData().NTOTAL,3),np.nan).float()
    xyz[:,:14,:] = pdb['xyz']
    mask = torch.full((len(idx), ChemData().NTOTAL), False)
    mask[:,:14] = pdb['mask']
    xyz = torch.nan_to_num(xyz)

    # get template features
    ntempl = np.random.randint(params['MINTPLT'], params['MAXTPLT']+1)
    xyz_t, f1d_t, mask_t, _ = TemplFeaturize(tplt, msa.shape[1], params, npick=ntempl, offset=0, pick_top=pick_top, random_noise=random_noise)
    
    # Residue cropping
    croplen = params['CROP']
    disulf_crop = False
    disulfs = get_dislf(msa[0],xyz,mask)
    if (len(disulfs)>1) and (np.random.rand() < p_dslf_crop):
        start,stop,clen = min ([(x,y,y-x) for x,y in disulfs], key=lambda x:x[2])
        if (clen<=20):
            crop_idx = torch.arange(start,stop+1,device=msa.device)
            disulf_crop = True

    if (not disulf_crop):
        if (np.random.rand() < p_short_crop):
            croplen = np.random.randint(8,16)

        crop_function = get_crop
        if params.get('DISCONTIGUOUS_CROP', False):
            crop_function = get_discontiguous_crop
        crop_idx = crop_function(len(idx), mask, msa.device, croplen, unclamp=unclamp)

    if (disulf_crop):
        ###
        # Atomize disulfide
        msa_prot = msa[:, crop_idx]
        ins_prot = ins[:, crop_idx]
        xyz_prot = xyz[crop_idx]
        mask_prot = mask[crop_idx]
        idx = idx[crop_idx]
        xyz_t_prot = xyz_t[:, crop_idx]
        f1d_t_prot = f1d_t[:, crop_idx]
        mask_t_prot = mask_t[:, crop_idx]
        protein_L, nprotatoms, _ = xyz_prot.shape

        bond_feats = get_protein_bond_feats(len(crop_idx)).long()
        same_chain = torch.ones((len(crop_idx), len(crop_idx))).long()

        res_idxs_to_atomize = torch.tensor([0,len(crop_idx)-1], device=msa.device)
        dslfs = [(0,len(crop_idx)-1)]
        seq_atomize_all, ins_atomize_all, xyz_atomize_all, mask_atomize_all, frames_atomize_all, chirals_atomize_all, \
            bond_feats, same_chain = atomize_discontiguous_residues(res_idxs_to_atomize, msa_prot, xyz_prot, mask_prot, bond_feats, same_chain, dslfs=dslfs)

        # Generate ground truth structure: account for ligand symmetry
        N_symmetry, sm_L, _ = xyz_atomize_all.shape
        xyz = torch.full((N_symmetry, protein_L+sm_L, ChemData().NTOTAL, 3), np.nan).float()
        mask = torch.full(xyz.shape[:-1], False).bool()
        xyz[:, :protein_L, :nprotatoms, :] = xyz_prot.expand(N_symmetry, protein_L, nprotatoms, 3)
        xyz[:, protein_L:, 1, :] = xyz_atomize_all
        mask[:, :protein_L, :nprotatoms] = mask_prot.expand(N_symmetry, protein_L, nprotatoms)
        mask[:, protein_L:, 1] = mask_atomize_all

        # generate (empty) template for atoms
        tplt_sm = {"ids":[]}
        xyz_t_sm, f1d_t_sm, mask_t_sm,_ = TemplFeaturize(tplt_sm, xyz_atomize_all.shape[1], params, offset=0, npick=0, pick_top=pick_top)
        ntempl = xyz_t_prot.shape[0]    
        xyz_t = torch.cat((xyz_t_prot, xyz_t_sm.repeat(ntempl,1,1,1)), dim=1)
        f1d_t = torch.cat((f1d_t_prot, f1d_t_sm.repeat(ntempl,1,1)), dim=1)
        mask_t = torch.cat((mask_t_prot, mask_t_sm.repeat(ntempl,1,1)), dim=1)

        Ls = [xyz_prot.shape[0], xyz_atomize_all.shape[1]]
        a3m_prot = {"msa": msa_prot, "ins": ins_prot}
        a3m_sm = {"msa": seq_atomize_all.unsqueeze(0), "ins": ins_atomize_all.unsqueeze(0)}

        a3m = merge_a3m_hetero(a3m_prot, a3m_sm, Ls)
        msa = a3m['msa'].long()
        ins = a3m['ins'].long()

        # handle res_idx
        last_res = idx[-1]
        idx_sm = torch.arange(Ls[1]) + last_res
        idx = torch.cat((idx, idx_sm))

        ch_label = torch.zeros(sum(Ls))
        # remove msa features for atomized portion
        msa, ins, xyz, mask, bond_feats, idx, xyz_t, f1d_t, mask_t, same_chain, ch_label = \
            pop_protein_feats(res_idxs_to_atomize, msa, ins, xyz, mask, bond_feats, idx, xyz_t, f1d_t, mask_t, same_chain, ch_label, Ls)
        # N/C-terminus features for MSA features (need to generate before cropping)
        # term_info = get_term_feats(Ls)
        # term_info[protein_L:, :] = 0 # ligand chains don't get termini features
        # msa_featurization_kwargs["term_info"] = term_info
        seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params, fixbb=fixbb, **msa_featurization_kwargs)

        xyz_prev, mask_prev = generate_xyz_prev(xyz_t, mask_t, params)

        xyz = torch.nan_to_num(xyz)

        dist_matrix = get_bond_distances(bond_feats)

        if chirals_atomize_all.shape[0]>0:
            L1 = torch.sum(~is_atom(seq[0]))
            chirals_atomize_all[:, :-1] = chirals_atomize_all[:, :-1] +L1

    else:
        ###
        # Normal
        seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params, fixbb=fixbb, **msa_featurization_kwargs)

        seq = seq[:,crop_idx]
        same_chain = torch.ones((len(crop_idx), len(crop_idx))).long()
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

        xyz_prev, mask_prev = generate_xyz_prev(xyz_t, mask_t, params)

        bond_feats = get_protein_bond_feats(len(crop_idx)).long()
        dist_matrix = get_bond_distances(bond_feats)

        ch_label = torch.zeros(seq[0].shape)

        chirals_atomize_all = torch.zeros(0,5)
        frames_atomize_all = torch.zeros(0,3,2)

    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa, \
           xyz.float(), mask, idx.long(),\
           xyz_t.float(), f1d_t.float(), mask_t, \
           xyz_prev.float(), mask_prev, \
           same_chain, unclamp, False, frames_atomize_all, bond_feats.long(), dist_matrix, chirals_atomize_all, \
           ch_label, "C1"

# Generate input features for homo-oligomers
def featurize_homo(msa_orig, ins_orig, tplt, pdbA, pdbid, interfaces, params, pick_top=True, random_noise=5.0, fixbb=False):
    L = msa_orig.shape[1]

    # msa always over 2 subunits (higher-order symms expand this)
    msa, ins = merge_a3m_homo(msa_orig, ins_orig, 2) # make unpaired alignments, for training, we always use two chains
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params, L_s=[L,L])

    # get ground-truth structures
    # load metadata
    PREFIX = "%s/torch/pdb/%s/%s"%(params['PDB_DIR'],pdbid[1:3],pdbid)
    meta = torch.load(PREFIX+".pt")

    # get all possible pairs
    npairs = len(interfaces)
    xyz = ChemData().INIT_CRDS.reshape(1,1,ChemData().NTOTAL,3).repeat(npairs, 2*L, 1, 1)
    mask = torch.full((npairs, 2*L, ChemData().NTOTAL), False)
    #print ("featurize_homo",pdbid,interfaces)
    for i_int,interface in enumerate(interfaces):
        pdbB = torch.load(params['PDB_DIR']+'/torch/pdb/'+interface['CHAIN_B'][1:3]+'/'+interface['CHAIN_B']+'.pt')
        xformA = meta['asmb_xform%d'%interface['ASSM_A']][interface['OP_A']]
        xformB = meta['asmb_xform%d'%interface['ASSM_B']][interface['OP_B']]
        xyzA = torch.einsum('ij,raj->rai', xformA[:3,:3], pdbA['xyz']) + xformA[:3,3][None,None,:]
        xyzB = torch.einsum('ij,raj->rai', xformB[:3,:3], pdbB['xyz']) + xformB[:3,3][None,None,:]
        xyz[i_int,:,:14] = torch.cat((xyzA, xyzB), dim=0)
        mask[i_int,:,:14] = torch.cat((pdbA['mask'], pdbB['mask']), dim=0)
    xyz = torch.nan_to_num(xyz)

    # detect any point symmetries
    symmgp, symmsubs = get_symmetry(xyz,mask)
    nsubs = len(symmsubs)+1

    #print ('symmgp',symmgp)
    # build full native complex (for loss calcs)
    if (symmgp != 'C1'):
        xyzfull = torch.zeros((1,nsubs*L,ChemData().NTOTAL,3))
        maskfull = torch.full((1,nsubs*L,ChemData().NTOTAL), False)
        xyzfull[0,:L] = xyz[0,:L]
        maskfull[0,:L] = mask[0,:L]
        for i in range(1,nsubs):
            xyzfull[0,i*L:(i+1)*L] = xyz[symmsubs[i-1],L:]
            maskfull[0,i*L:(i+1)*L] = mask[symmsubs[i-1],L:]
        xyz = xyzfull
        mask = maskfull

    # get template features
    ntempl = np.random.randint(params['MINTPLT'], params['MAXTPLT']+1)
    if ntempl < 1:
        xyz_t, f1d_t, mask_t, _ = TemplFeaturize(tplt, L, params, npick=ntempl, offset=0, pick_top=pick_top, random_noise=random_noise)
    else:
        xyz_t, f1d_t, mask_t, _ = TemplFeaturize(tplt, L, params, npick=ntempl, offset=0, pick_top=pick_top, random_noise=random_noise)
        # duplicate

    if (symmgp != 'C1'):
        # everything over ASU
        idx = torch.arange(L)
        same_chain = torch.ones((L, L)).long()
        nsub = len(symmsubs)+1
        bond_feats = get_protein_bond_feats(L)
    else:  # either asymmetric dimer or (usually) helical symmetry...
        # everything over 2 copies
        xyz_t = torch.cat([xyz_t, random_rot_trans(xyz_t)], dim=1)
        f1d_t = torch.cat([f1d_t]*2, dim=1)
        mask_t = torch.cat([mask_t]*2, dim=1)
        idx = torch.arange(L*2)
        idx[L:] += 100 # to let network know about chain breaks

        same_chain = torch.zeros((2*L, 2*L)).long()
        same_chain[:L, :L] = 1
        same_chain[L:, L:] = 1
        bond_feats = torch.zeros((2*L, 2*L)).long()
        bond_feats[:L, :L] = get_protein_bond_feats(L)
        bond_feats[L:, L:] = get_protein_bond_feats(L)

        nsub = 2
    
    ntempl = xyz_t.shape[0]
    xyz_t = torch.stack(
        [center_and_realign_missing(xyz_t[i], mask_t[i], same_chain=same_chain) for i in range(ntempl)]
    )
    # get initial coordinates
    xyz_prev, mask_prev = generate_xyz_prev(xyz_t, mask_t, params)

    # figure out crop
    if (symmgp =='C1'):
        cropsub = 2
    elif (symmgp[0]=='C'):
        cropsub = min(3, int(symmgp[1:]))
    elif (symmgp[0]=='D'):
        cropsub = min(5, 2*int(symmgp[1:]))
    else:
        cropsub = 6

    # Residue cropping
    if cropsub*L > params['CROP']:
        #if np.random.rand() < 0.5: # 50% --> interface crop
        #    spatial_crop_tgt = np.random.randint(0, npairs)
        #    crop_idx = get_spatial_crop(xyz[spatial_crop_tgt], mask[spatial_crop_tgt], torch.arange(L*2), [L,L], params, interfaces[spatial_crop_tgt][0])
        #else: # 50% --> have same cropped regions across all copies
        #    crop_idx = get_crop(L, mask[0,:L], msa_seed_orig.device, params['CROP']//2, unclamp=False) # cropped region for first copy
        #    crop_idx = torch.cat((crop_idx, crop_idx+L)) # get same crops
        #    #print ("check_crop", crop_idx, crop_idx.shape)

        # fd: always use same cropped regions across all copies
        crop_idx = get_crop(L, mask[0,:L], msa_seed_orig.device, params['CROP']//cropsub, unclamp=False) # cropped region for first copy
        crop_idx_full = torch.cat([crop_idx,crop_idx+L])
        if (symmgp == 'C1'):
            crop_idx = crop_idx_full
            crop_idx_complete = crop_idx_full
        else:
            crop_idx_complete = []
            for i in range(nsub):
                crop_idx_complete.append(crop_idx+i*L)
            crop_idx_complete = torch.cat(crop_idx_complete)

        # over 2 copies
        seq = seq[:,crop_idx_full]
        msa_seed_orig = msa_seed_orig[:,:,crop_idx_full]
        msa_seed = msa_seed[:,:,crop_idx_full]
        msa_extra = msa_extra[:,:,crop_idx_full]
        mask_msa = mask_msa[:,:,crop_idx_full]

        # over 1 copy (symmetric) or 2 copies (asymmetric)
        xyz_t = xyz_t[:,crop_idx]
        f1d_t = f1d_t[:,crop_idx]
        mask_t = mask_t[:,crop_idx]
        idx = idx[crop_idx]
        same_chain = same_chain[crop_idx][:,crop_idx]
        bond_feats = bond_feats[crop_idx][:,crop_idx]
        xyz_prev = xyz_prev[crop_idx]
        mask_prev = mask_prev[crop_idx]

        # over >=2 copies
        xyz = xyz[:,crop_idx_complete]
        mask = mask[:,crop_idx_complete]

    dist_matrix = get_bond_distances(bond_feats)
    chirals = torch.Tensor()
    ch_label = torch.zeros(seq[0].shape)

    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa, \
           xyz.float(), mask, idx.long(),\
           xyz_t.float(), f1d_t.float(), mask_t, \
           xyz_prev.float(), mask_prev, \
           same_chain, False, False, torch.zeros(seq.shape), bond_feats, dist_matrix, chirals, ch_label, symmgp


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

def get_msa(a3mfilename, item, maxseq=5000):
    msa,ins, taxIDs = parse_a3m(a3mfilename, maxseq=5000)
    return {'msa':torch.tensor(msa), 'ins':torch.tensor(ins), 'taxIDs':taxIDs, 'label':item}

# Load PDB examples
def loader_pdb(item, params, homo, unclamp=False, pick_top=True, p_homo_cut=0.5, p_short_crop=0.0, p_dslf_crop=0.0, fixbb=False):
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

    # when target is homo-oligomer, model as homo-oligomer with probability p_homo_cut
    if pdb_chain in homo['CHAIN_A'].values and np.random.rand() < p_homo_cut: 
        pdbid = pdb_chain.split('_')[0]
        interfaces = homo[homo['CHAIN_A']==pdb_chain].to_dict(orient='records') # list of dicts
        feats = featurize_homo(msa, ins, tplt, pdb, pdbid, interfaces, params, pick_top=pick_top, fixbb=fixbb)
        return feats + ("homo",item,)

    # only short crop monomers
    feats = featurize_single_chain(
        msa, ins, tplt, pdb, params, unclamp=unclamp, pick_top=pick_top, fixbb=fixbb, p_short_crop=p_short_crop, p_dslf_crop=p_dslf_crop
    )
    return feats + ("monomer",item,)


def loader_fb(item, params, unclamp=False, p_short_crop=0.0, p_dslf_crop=0.0, fixbb=False):
    # loads sequence/structure/plddt information
    pdb_chain, hashstr = item['CHAINID'], item['HASH']
    a3m = get_msa(os.path.join(params["FB_DIR"], "a3m", hashstr[:2], hashstr[2:], pdb_chain+".a3m.gz"), pdb_chain)
    pdb = get_pdb(os.path.join(params["FB_DIR"], "pdb", hashstr[:2], hashstr[2:], pdb_chain+".pdb"),
                  os.path.join(params["FB_DIR"], "pdb", hashstr[:2], hashstr[2:], pdb_chain+".plddt.npy"),
                  pdb_chain, params['PLDDTCUT'], params['SCCUT'])

    # get msa features
    msa = a3m['msa'].long()
    ins = a3m['ins'].long()
    if len(msa) > params['BLOCKCUT']:
        msa, ins = MSABlockDeletion(msa, ins)
    L = msa.shape[1]

    # get ground-truth structures
    idx = pdb['idx']
    xyz = torch.full((len(idx),ChemData().NTOTAL,3),np.nan).float()
    xyz[:,:27,:] = pdb['xyz'][:,:27]
    mask = torch.full((len(idx),ChemData().NTOTAL), False)
    mask[:,:27] = pdb['mask'][:,:27]

    # get template features -- None
    tplt_blank = {"ids":[]}
    xyz_t, f1d_t, mask_t, _ = TemplFeaturize(tplt_blank, L, params, offset=0, npick=0)  

    # Residue cropping
    croplen = params['CROP']
    disulf_crop = False
    # random disulfide loop
    disulfs = get_dislf(msa[0],xyz,mask)
    if (len(disulfs)>1) and (np.random.rand() < p_dslf_crop):
        start,stop,clen = min ([(x,y,y-x) for x,y in disulfs], key=lambda x:x[2])
        if (clen<=20):
            crop_idx = torch.arange(start,stop+1,device=msa.device)
            disulf_crop = True
            #print ('loader_fb crop',crop_idx)

    if (not disulf_crop):
        if (np.random.rand() < p_short_crop):
            croplen = np.random.randint(8,16)
        crop_idx = get_crop(len(idx), mask, msa.device, croplen, unclamp=unclamp)

    if (disulf_crop):
        ###
        # Atomize disulfide
        msa_prot = msa[:, crop_idx]
        ins_prot = ins[:, crop_idx]
        xyz_prot = xyz[crop_idx]
        mask_prot = mask[crop_idx]
        idx = idx[crop_idx]
        xyz_t_prot = xyz_t[:, crop_idx]
        f1d_t_prot = f1d_t[:, crop_idx]
        mask_t_prot = mask_t[:, crop_idx]
        protein_L, nprotatoms, _ = xyz_prot.shape

        bond_feats = get_protein_bond_feats(len(crop_idx)).long()
        same_chain = torch.ones((len(crop_idx), len(crop_idx))).long()

        res_idxs_to_atomize = torch.tensor([0,len(crop_idx)-1], device=msa.device)
        dslfs = [(0,len(crop_idx)-1)]
        seq_atomize_all, ins_atomize_all, xyz_atomize_all, mask_atomize_all, frames_atomize_all, chirals_atomize_all, \
            bond_feats, same_chain = atomize_discontiguous_residues(res_idxs_to_atomize, msa_prot, xyz_prot, mask_prot, bond_feats, same_chain, dslfs=dslfs)
        atom_template_motif_idxs = get_atom_template_indices(msa,res_idxs_to_atomize)

        # Generate ground truth structure: account for ligand symmetry
        N_symmetry, sm_L, _ = xyz_atomize_all.shape
        xyz = torch.full((N_symmetry, protein_L+sm_L, ChemData().NTOTAL, 3), np.nan).float()
        mask = torch.full(xyz.shape[:-1], False).bool()
        xyz[:, :protein_L, :nprotatoms, :] = xyz_prot.expand(N_symmetry, protein_L, nprotatoms, 3)
        xyz[:, protein_L:, 1, :] = xyz_atomize_all
        mask[:, :protein_L, :nprotatoms] = mask_prot.expand(N_symmetry, protein_L, nprotatoms)
        mask[:, protein_L:, 1] = mask_atomize_all

        # generate (empty) template for atoms
        tplt_sm = {"ids":[]}
        xyz_t_sm, f1d_t_sm, mask_t_sm, _ = TemplFeaturize(tplt_sm, xyz_atomize_all.shape[1], params, offset=0, npick=0)
        ntempl = xyz_t_prot.shape[0]
        xyz_t = torch.cat((xyz_t_prot, xyz_t_sm.repeat(ntempl,1,1,1)), dim=1)
        f1d_t = torch.cat((f1d_t_prot, f1d_t_sm.repeat(ntempl,1,1)), dim=1)
        mask_t = torch.cat((mask_t_prot, mask_t_sm.repeat(ntempl,1,1)), dim=1)

        Ls = [xyz_prot.shape[0], xyz_atomize_all.shape[1]]
        a3m_prot = {"msa": msa_prot, "ins": ins_prot}
        a3m_sm = {"msa": seq_atomize_all.unsqueeze(0), "ins": ins_atomize_all.unsqueeze(0)}

        a3m = merge_a3m_hetero(a3m_prot, a3m_sm, Ls)
        msa = a3m['msa'].long()
        ins = a3m['ins'].long()

        # handle res_idx
        last_res = idx[-1]
        idx_sm = torch.arange(Ls[1]) + last_res
        idx = torch.cat((idx, idx_sm))

        ch_label = torch.zeros(sum(Ls))
        # remove msa features for atomized portion
        msa, ins, xyz, mask, bond_feats, idx, xyz_t, f1d_t, mask_t, same_chain, ch_label = \
            pop_protein_feats(res_idxs_to_atomize, msa, ins, xyz, mask, bond_feats, idx, xyz_t, f1d_t, mask_t, same_chain, ch_label, Ls)
        # N/C-terminus features for MSA features (need to generate before cropping)
        # term_info = get_term_feats(Ls)
        # term_info[protein_L:, :] = 0 # ligand chains don't get termini features
        seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params, 
        #term_info=term_info
        )

        xyz_prev, mask_prev = generate_xyz_prev(xyz_t, mask_t, params)

        xyz = torch.nan_to_num(xyz)

        dist_matrix = get_bond_distances(bond_feats)

        if chirals_atomize_all.shape[0]>0:
            L1 = torch.sum(~is_atom(seq[0]))
            chirals_atomize_all[:, :-1] = chirals_atomize_all[:, :-1] +L1

    else:

        ###
        # Normal
        seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params)

        seq = seq[:,crop_idx]
        same_chain = torch.ones((len(crop_idx), len(crop_idx))).long()
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

        xyz_prev, mask_prev = generate_xyz_prev(xyz_t, mask_t, params)

        bond_feats = get_protein_bond_feats(len(crop_idx)).long()
        dist_matrix = get_bond_distances(bond_feats)

        ch_label = torch.zeros(seq[0].shape)

        chirals_atomize_all = torch.Tensor()
        frames_atomize_all = torch.zeros(seq.shape)

    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa, \
           xyz.float(), mask, idx.long(),\
           xyz_t.float(), f1d_t.float(), mask_t, \
           xyz_prev.float(), mask_prev, \
           same_chain, unclamp, False, frames_atomize_all, bond_feats.long(), dist_matrix, chirals_atomize_all, \
           ch_label, "C1", "fb", item


def loader_complex(item, params, negative=False, pick_top=True, random_noise=5.0, fixbb=False):

    pdb_pair, pMSA_hash, L_s, taxID = item['CHAINID'], item['HASH'], item['LEN'], item['TAXONOMY']
    msaA_id, msaB_id = pMSA_hash.split('_')
    
    if len(set(taxID.split(':'))) == 1: # two proteins have same taxID -- use paired MSA
        # read pMSA
        if negative:
            pMSA_fn = params['COMPL_DIR'] + '/pMSA.negative/' + msaA_id[:3] + '/' + msaB_id[:3] + '/' + pMSA_hash + '.a3m.gz'
        else:
            pMSA_fn = params['COMPL_DIR'] + '/pMSA/' + msaA_id[:3] + '/' + msaB_id[:3] + '/' + pMSA_hash + '.a3m.gz'
        a3m = get_msa(pMSA_fn, pMSA_hash)
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
    xyz_t_A, f1d_t_A, mask_t_A, _ = TemplFeaturize(tpltA, L_s[0], params, offset=0, npick=ntemplA, npick_global=max(1,max(ntemplA, ntemplB)), pick_top=pick_top, random_noise=random_noise)
    xyz_t_B, f1d_t_B, mask_t_B, _ = TemplFeaturize(tpltB, L_s[1], params, offset=0, npick=ntemplB, npick_global=max(1,max(ntemplA, ntemplB)), pick_top=pick_top, random_noise=random_noise)
    xyz_t = torch.cat((xyz_t_A, random_rot_trans(xyz_t_B)), dim=1) # (T, L1+L2, natm, 3)
    f1d_t = torch.cat((f1d_t_A, f1d_t_B), dim=1) # (T, L1+L2, natm, 3)
    mask_t = torch.cat((mask_t_A, mask_t_B), dim=1) # (T, L1+L2, natm, 3)

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
        xyz = torch.full((sum(L_s), ChemData().NTOTAL, 3), np.nan).float()
        xyz[:,:14] = torch.cat((xyzA, xyzB), dim=0)
        mask = torch.full((sum(L_s), ChemData().NTOTAL), False)
        mask[:,:14] = torch.cat((pdbA['mask'], pdbB['mask']), dim=0)
    else:
        xyz = torch.full((sum(L_s), ChemData().NTOTAL, 3), np.nan).float()
        xyz[:,:14] = torch.cat((pdbA['xyz'], pdbB['xyz']), dim=0)
        mask = torch.full((sum(L_s), ChemData().NTOTAL), False)
        mask[:,:14] = torch.cat((pdbA['mask'], pdbB['mask']), dim=0)
    xyz = torch.nan_to_num(xyz)

    idx = torch.arange(sum(L_s))
    idx[L_s[0]:] += ChemData().CHAIN_GAP

    same_chain = torch.zeros((sum(L_s), sum(L_s))).long()
    same_chain[:L_s[0], :L_s[0]] = 1
    same_chain[L_s[0]:, L_s[0]:] = 1
    bond_feats = torch.zeros((sum(L_s), sum(L_s))).long()
    bond_feats[:L_s[0], :L_s[0]] = get_protein_bond_feats(L_s[0])
    bond_feats[L_s[0]:, L_s[0]:] = get_protein_bond_feats(sum(L_s[1:]))

    ntempl = xyz_t.shape[0]
    xyz_t = torch.stack(
        [center_and_realign_missing(xyz_t[i], mask_t[i], same_chain=same_chain) for i in range(ntempl)]
    )    
    # get initial coordinates
    xyz_prev, mask_prev = generate_xyz_prev(xyz_t, mask_t, params)
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
        same_chain = same_chain[sel][:,sel]
        bond_feats = bond_feats[sel][:,sel]

    dist_matrix = get_bond_distances(bond_feats)
    chirals = torch.Tensor()
    L1 = same_chain[0,:].sum()
    ch_label = torch.zeros(seq[0].shape)
    ch_label[L1:] = 1
    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa,\
           xyz.float(), mask, idx.long(), \
           xyz_t.float(), f1d_t.float(), mask_t, \
           xyz_prev.float(), mask_prev, \
           same_chain, False, negative, torch.zeros(seq.shape), bond_feats, dist_matrix, chirals, ch_label, 'C1', "compl", item

def loader_na_complex(item, params, native_NA_frac=0.05, negative=False, pick_top=True, random_noise=5.0):
    pdb_set = item['CHAINID']
    msa_id = item['HASH']
    #Ls = item['LEN']  #fd this is not reported correctly....

    if negative:
        padding = (item['DNA1'],item['DNA2'])
    else:
        padding = item['TOPAD?']
    
    # read PDBs
    pdb_ids = pdb_set.split(':')

    # read protein MSA
    a3mA = get_msa(params['PDB_DIR'] + '/a3m/' + msa_id[:3] + '/' + msa_id + '.a3m.gz', msa_id, maxseq=5000)

    # protein + NA
    NMDLS = 1
    if (len(pdb_ids)==2):
        pdbA = [ torch.load(params['PDB_DIR']+'/torch/pdb/'+pdb_ids[0][1:3]+'/'+pdb_ids[0]+'.pt') ]
        
        filenameB = params['NA_DIR']+'/torch/'+pdb_ids[1][1:3]+'/'+pdb_ids[1]+'.pt'
        if os.path.exists(filenameB+".v3"):
            filenameB = filenameB+".v3"
        pdbB = [ torch.load(filenameB) ]

        msaB,insB = parse_fasta_if_exists(
            pdbB[0]['seq'], params['NA_DIR']+'/torch/'+pdb_ids[1][1:3]+'/'+pdb_ids[1]+'.afa', 
            maxseq=5000,
            rmsa_alphabet=True
        )
        a3mB = {'msa':torch.from_numpy(msaB), 'ins':torch.from_numpy(insB)}

        Ls = [a3mA['msa'].shape[1], a3mB['msa'].shape[1]]
    # protein + NA duplex
    elif (len(pdb_ids)==3):
        pdbA = [ torch.load(params['PDB_DIR']+'/torch/pdb/'+pdb_ids[0][1:3]+'/'+pdb_ids[0]+'.pt') ]
        filenameB1 = params['NA_DIR']+'/torch/'+pdb_ids[1][1:3]+'/'+pdb_ids[1]+'.pt'
        filenameB2 = params['NA_DIR']+'/torch/'+pdb_ids[2][1:3]+'/'+pdb_ids[2]+'.pt'
        if os.path.exists(filenameB1+".v3"):
            filenameB1 = filenameB1+".v3"
        if os.path.exists(filenameB2+".v3"):
            filenameB2 = filenameB2+".v3"
        pdbB = [ torch.load(filenameB1), torch.load(filenameB2) ]

        msaB1,insB1 = parse_fasta_if_exists(
            pdbB[0]['seq'], params['NA_DIR']+'/torch/'+pdb_ids[1][1:3]+'/'+pdb_ids[1]+'.afa', 
            maxseq=5000,
            rmsa_alphabet=True
        )
        msaB2,insB2 = parse_fasta_if_exists(
            pdbB[1]['seq'], params['NA_DIR']+'/torch/'+pdb_ids[2][1:3]+'/'+pdb_ids[2]+'.afa', 
            maxseq=5000,
            rmsa_alphabet=True
        )
        if (pdbB[0]['seq']==pdbB[1]['seq']):
            NMDLS=2 # flip B0 and B1

        a3mB1 = {'msa':torch.from_numpy(msaB1), 'ins':torch.from_numpy(insB1)}
        a3mB2 = {'msa':torch.from_numpy(msaB2), 'ins':torch.from_numpy(insB2)}
        Ls = [a3mA['msa'].shape[1], a3mB1['msa'].shape[1], a3mB2['msa'].shape[1]]
        a3mB = merge_a3m_hetero(a3mB1, a3mB2, Ls[1:])

    # homodimer + NA duplex
    elif (len(pdb_ids)==4):
        pdbA = [
            torch.load(params['PDB_DIR']+'/torch/pdb/'+pdb_ids[0][1:3]+'/'+pdb_ids[0]+'.pt'),
            torch.load(params['PDB_DIR']+'/torch/pdb/'+pdb_ids[1][1:3]+'/'+pdb_ids[1]+'.pt')
        ]
        filenameB1 = params['NA_DIR']+'/torch/'+pdb_ids[2][1:3]+'/'+pdb_ids[2]+'.pt'
        filenameB2 = params['NA_DIR']+'/torch/'+pdb_ids[3][1:3]+'/'+pdb_ids[3]+'.pt'
        if os.path.exists(filenameB1+".v3"):
            filenameB1 = filenameB1+".v3"
        if os.path.exists(filenameB2+".v3"):
            filenameB2 = filenameB2+".v3"
        pdbB = [ torch.load(filenameB1), torch.load(filenameB2) ]
        msaB1,insB1 = parse_fasta_if_exists(
            pdbB[0]['seq'], params['NA_DIR']+'/torch/'+pdb_ids[2][1:3]+'/'+pdb_ids[2]+'.afa', 
            maxseq=5000,
            rmsa_alphabet=True
        )
        msaB2,insB2 = parse_fasta_if_exists(
            pdbB[1]['seq'], params['NA_DIR']+'/torch/'+pdb_ids[3][1:3]+'/'+pdb_ids[3]+'.afa', 
            maxseq=5000,
            rmsa_alphabet=True
        )
        a3mB1 = {'msa':torch.from_numpy(msaB1), 'ins':torch.from_numpy(insB1)}
        a3mB2 = {'msa':torch.from_numpy(msaB2), 'ins':torch.from_numpy(insB2)}
        Ls = [a3mA['msa'].shape[1], a3mA['msa'].shape[1], a3mB1['msa'].shape[1], a3mB2['msa'].shape[1]]
        a3mB = merge_a3m_hetero(a3mB1, a3mB2, Ls[2:])


        NMDLS=2 # flip A0 and A1
        if (pdbB[0]['seq']==pdbB[1]['seq']):
            NMDLS=4 # flip B0 and B1

    else:
        assert False

    # apply padding
    if (not negative and padding):
        assert (len(pdbB)==2)
        lpad = np.random.randint(6)
        rpad = np.random.randint(6)
        lseq1 = torch.randint(4,(1,lpad))
        rseq1 = torch.randint(4,(1,rpad))
        lseq2 = 3-torch.flip(rseq1,(1,))
        rseq2 = 3-torch.flip(lseq1,(1,))

        # pad seqs -- hacky, DNA indices 22-25
        msaB1 = torch.cat((22+lseq1,a3mB1['msa'],22+rseq1), dim=1)
        msaB2 = torch.cat((22+lseq2,a3mB2['msa'],22+rseq2), dim=1)
        insB1 = torch.cat((torch.zeros_like(lseq1),a3mB1['ins'],torch.zeros_like(rseq1)), dim=1)
        insB2 = torch.cat((torch.zeros_like(lseq2),a3mB2['ins'],torch.zeros_like(rseq2)), dim=1)
        a3mB1 = {'msa':msaB1, 'ins':insB1}
        a3mB2 = {'msa':msaB2, 'ins':insB2}

        # update lengths
        Ls = Ls.copy()
        Ls[-2] = msaB1.shape[1]
        Ls[-1] = msaB2.shape[1]

        a3mB = merge_a3m_hetero(a3mB1, a3mB2, Ls[-2:])

        # pad PDB
        pdbB[0]['xyz'] = torch.nn.functional.pad(pdbB[0]['xyz'], (0,0,0,0,lpad,rpad), "constant", 0.0)
        pdbB[0]['mask'] = torch.nn.functional.pad(pdbB[0]['mask'], (0,0,lpad,rpad), "constant", False)
        pdbB[1]['xyz'] = torch.nn.functional.pad(pdbB[1]['xyz'], (0,0,0,0,rpad,lpad), "constant", 0.0)
        pdbB[1]['mask'] = torch.nn.functional.pad(pdbB[1]['mask'], (0,0,rpad,lpad), "constant", False)

    # rewrite seq if negative
    if (negative):
        alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-Xacgtxbdhuy"), dtype='|S1').view(np.uint8)
        seqA = np.array( [list(padding[0])], dtype='|S1').view(np.uint8)
        seqB = np.array( [list(padding[1])], dtype='|S1').view(np.uint8)
        for i in range(alphabet.shape[0]):
            seqA[seqA == alphabet[i]] = i
            seqB[seqB == alphabet[i]] = i
        seqA = torch.tensor(seqA)
        seqB = torch.tensor(seqB)

        # scramble seq
        diff = (a3mB1['msa'] != seqA)
        shift = torch.randint(1,4, (torch.sum(diff),), dtype=torch.uint8)
        seqA[diff] = ((a3mB1['msa'][diff]-22)+shift)%4+22
        seqB = torch.flip(25-seqA+22, dims=(-1,))

        a3mB1 = {'msa':seqA, 'ins':torch.zeros(seqA.shape)}
        a3mB2 = {'msa':seqB, 'ins':torch.zeros(seqB.shape)}
        a3mB = merge_a3m_hetero(a3mB1, a3mB2, Ls[-2:])

    ## look for shared MSA
    a3m=None
    NAchn = pdb_ids[1].split('_')[1]
    sharedMSA = params['NA_DIR']+'/msas/'+pdb_ids[0][1:3]+'/'+pdb_ids[0][:4]+'/'+pdb_ids[0]+'_'+NAchn+'_paired.a3m'
    if (len(pdb_ids)==2 and exists(sharedMSA)):
        msa,ins = parse_mixed_fasta(sharedMSA)
        if (msa.shape[1] != sum(Ls)):
            print ("Error loading shared MSA",pdb_ids, msa.shape, Ls)
        else:
            a3m = {'msa':torch.from_numpy(msa),'ins':torch.from_numpy(ins)}

    if a3m is None:
        if (len(pdbA)==2):
            msa = a3mA['msa'].long()
            ins = a3mA['ins'].long()
            msa,ins = merge_a3m_homo(msa, ins, 2)
            a3mA = {'msa':msa,'ins':ins}

        if (len(pdb_ids)==4):
            a3m = merge_a3m_hetero(a3mA, a3mB, [Ls[0]+Ls[1],sum(Ls[2:])])
        else:
            a3m = merge_a3m_hetero(a3mA, a3mB, [Ls[0],sum(Ls[1:])])


    # the block below is due to differences in the way RNA and DNA structures are processed
    # to support NMR, RNA structs return multiple states
    # For protein/NA complexes get rid of the 'NMODEL' dimension (if present)
    # NOTE there are a very small number of protein/NA NMR models:
    #       - ideally these should return the ensemble, but that requires reprocessing of proteins
    for pdb in pdbB:
        if (len(pdb['xyz'].shape) > 3):
            pdb['xyz'] = pdb['xyz'][0,...]
            pdb['mask'] = pdb['mask'][0,...]

    # read template info
    tpltA = torch.load(params['PDB_DIR'] + '/torch/hhr/' + msa_id[:3] + '/' + msa_id + '.pt')
    ntempl = np.random.randint(params['MINTPLT'], params['MAXTPLT']-1)
    if (len(pdb_ids)==4):
        if ntempl < 1:
            xyz_t, f1d_t, mask_t, _ = TemplFeaturize(tpltA, 2*Ls[0], params, npick=ntempl, offset=0, pick_top=pick_top, random_noise=random_noise)
        else:
            xyz_t_single, f1d_t_single, mask_t_single, _ = TemplFeaturize(tpltA, Ls[0], params, npick=ntempl, offset=0, pick_top=pick_top, random_noise=random_noise)
            # duplicate
            xyz_t = torch.cat((xyz_t_single, random_rot_trans(xyz_t_single)), dim=1) # (ntempl, 2*L, natm, 3)
            f1d_t = torch.cat((f1d_t_single, f1d_t_single), dim=1) # (ntempl, 2*L, 21)
            mask_t = torch.cat((mask_t_single, mask_t_single), dim=1) # (ntempl, 2*L, natm)

        ntmpl = xyz_t.shape[0]
        nNA = sum(Ls[2:])
        xyz_t = torch.cat( 
            (xyz_t, ChemData().INIT_NA_CRDS.reshape(1,1,ChemData().NTOTAL,3).repeat(ntmpl,nNA,1,1) + torch.rand(ntmpl,nNA,1,3)*random_noise), dim=1)
        f1d_t = torch.cat(
            (f1d_t, torch.nn.functional.one_hot(torch.full((ntmpl,nNA), 20).long(), num_classes=ChemData().NAATOKENS).float()), dim=1) # add extra class for 0 confidence
        mask_t = torch.cat( 
            (mask_t, torch.full((ntmpl,nNA,ChemData().NTOTAL), False)), dim=1)

        NAstart = 2*Ls[0]
    else:
        xyz_t, f1d_t, mask_t, _ = TemplFeaturize(tpltA, sum(Ls), params, offset=0, npick=ntempl, pick_top=pick_top, random_noise=random_noise)
        xyz_t[:,Ls[0]:] = ChemData().INIT_NA_CRDS.reshape(1,1,ChemData().NTOTAL,3).repeat(1,sum(Ls[1:]),1,1) + torch.rand(1,sum(Ls[1:]),1,3)*random_noise
        NAstart = Ls[0]

    # seed with native NA
    if (np.random.rand()<=native_NA_frac):
        natNA_templ = torch.cat( [x['xyz'] for x in pdbB], dim=0)
        maskNA_templ = torch.cat( [x['mask'] for x in pdbB], dim=0)

        # construct template from NA
        xyz_t_B = ChemData().INIT_CRDS.reshape(1,1,ChemData().NTOTAL,3).repeat(1,sum(Ls),1,1) + torch.rand(1,sum(Ls),1,3)*random_noise
        mask_t_B = torch.full((1,sum(Ls),ChemData().NTOTAL), False)
        mask_t_B[:,NAstart:,:23] = maskNA_templ
        xyz_t_B[mask_t_B] = natNA_templ[maskNA_templ]

        seq_t_B = torch.cat( (torch.full((1, NAstart), 20).long(),  a3mB['msa'][0:1]), dim=1)
        seq_t_B[seq_t_B>21] -= 1 # remove mask token
        f1d_t_B = torch.nn.functional.one_hot(seq_t_B, num_classes=ChemData().NAATOKENS-1).float()
        conf_B = torch.cat( (
            torch.zeros((1,NAstart,1)),
            torch.full((1,sum(Ls)-NAstart,1),1.0),
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

    # build native from components
    xyz = torch.full((NMDLS, sum(Ls), ChemData().NTOTAL, 3), np.nan)
    mask = torch.full((NMDLS, sum(Ls), ChemData().NTOTAL), False)
    if (len(pdb_ids)==2):
        xyz[0,:NAstart,:14] = pdbA[0]['xyz']
        xyz[0,NAstart:,:23] = pdbB[0]['xyz']
        mask[0,:NAstart,:14] = pdbA[0]['mask']
        mask[0,NAstart:,:23] = pdbB[0]['mask']
    elif (len(pdb_ids)==3):
        xyz[:,:NAstart,:14] = pdbA[0]['xyz'][None,...]
        xyz[0,NAstart:,:23] = torch.cat((pdbB[0]['xyz'], pdbB[1]['xyz']), dim=0)
        mask[:,:NAstart,:14] = pdbA[0]['mask'][None,...]
        mask[0,NAstart:,:23] = torch.cat((pdbB[0]['mask'], pdbB[1]['mask']), dim=0)
        if (NMDLS==2): # B & C are identical
            xyz[1,NAstart:,:23] = torch.cat((pdbB[1]['xyz'], pdbB[0]['xyz']), dim=0)
            mask[1,NAstart:,:23] = torch.cat((pdbB[1]['mask'], pdbB[0]['mask']), dim=0)
    else:
        xyz[0,:NAstart,:14] = torch.cat( (pdbA[0]['xyz'], pdbA[1]['xyz']), dim=0)
        xyz[1,:NAstart,:14] = torch.cat( (pdbA[1]['xyz'], pdbA[0]['xyz']), dim=0)
        xyz[:2,NAstart:,:23] = torch.cat((pdbB[0]['xyz'], pdbB[1]['xyz']), dim=0)[None,...]
        mask[0,:NAstart,:14] = torch.cat( (pdbA[0]['mask'], pdbA[1]['mask']), dim=0)
        mask[1,:NAstart,:14] = torch.cat( (pdbA[1]['mask'], pdbA[0]['mask']), dim=0)
        mask[:2,NAstart:,:23] = torch.cat( (pdbB[0]['mask'], pdbB[1]['mask']), dim=0)[None,...]
        if (NMDLS==4): # B & C are identical
            xyz[2,:NAstart,:14] = torch.cat( (pdbA[0]['xyz'], pdbA[1]['xyz']), dim=0)
            xyz[3,:NAstart,:14] = torch.cat( (pdbA[1]['xyz'], pdbA[0]['xyz']), dim=0)
            xyz[2:,NAstart:,:23] = torch.cat((pdbB[1]['xyz'], pdbB[0]['xyz']), dim=0)[None,...]
            mask[2,:NAstart,:14] = torch.cat( (pdbA[0]['mask'], pdbA[1]['mask']), dim=0)
            mask[3,:NAstart,:14] = torch.cat( (pdbA[1]['mask'], pdbA[0]['mask']), dim=0)
            mask[2:,NAstart:,:23] = torch.cat( (pdbB[1]['mask'], pdbB[0]['mask']), dim=0)[None,...]
    xyz = torch.nan_to_num(xyz)

    xyz, mask = remap_NA_xyz_tensors(xyz,mask,msa[0])

    # other features
    idx = idx_from_Ls(Ls)
    same_chain = same_chain_2d_from_Ls(Ls)
    bond_feats = bond_feats_from_Ls(Ls)
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

    ntempl = xyz_t.shape[0]
    xyz_t = torch.stack(
        [center_and_realign_missing(xyz_t[i], mask_t[i], same_chain=same_chain) for i in range(ntempl)]
    )
    xyz_prev, mask_prev = generate_xyz_prev(xyz_t, mask_t, params)

    atom_frames = torch.zeros(0,3,2)
    chirals = torch.zeros(0,5)
    dist_matrix = get_bond_distances(bond_feats)

    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa,\
           xyz.float(), mask, idx.long(), \
           xyz_t.float(), f1d_t.float(), mask_t, \
           xyz_prev.float(), mask_prev, \
           same_chain, False, negative, atom_frames, bond_feats, dist_matrix, chirals, ch_label, 'C1', "na_compl", item

def loader_tf_complex(item, params, negative=False, pick_top=True, random_noise=5.0):
#    ic(item, negative)

    gene_id = item["gene_id"]
    HASH = item['HASH']

    # read protein MSA from a3m file
    a3mA = get_msa(params["TF_DIR"]+f'/a3m_v2/{gene_id[:2]}/{gene_id}_aligned_domain.a3m', HASH)
    L_prot = a3mA['msa'].shape[1]

    # pick a DNA sequence to use
    tf_bind = 'neg' if negative else 'pos'
    seqs_fn = params["TF_DIR"]+f'/train_seqs/{gene_id[:2]}/{gene_id}_{tf_bind}.afa'
    with open(seqs_fn, 'r') as f_seqs:
        seqs = [line.strip() for line in f_seqs]
    # for positives, and in 20% of negatives, just pick a random sequence
    if (not negative) or (np.random.rand() < 0.2):
        seq = seqs[np.random.randint(len(seqs))]
        _, nmer = choose_matching_seq([seq],seq)

    # for the other 80% of negatives, look at a positive sequence and match its subseq symmetry
    # e.g. if pos is a partial palindrome (GCACGTGG), neg must also be one (AGCCGGCG) 
    else:
        # choose a positive seq to use as reference
        pos_seqs_fn = params["TF_DIR"]+f'/train_seqs/{gene_id[:2]}/{gene_id}_pos.afa'
        with open(pos_seqs_fn, 'r') as f_seqs:
            pos_seqs = [line.strip() for line in f_seqs]
        pos_seq = pos_seqs[np.random.randint(len(pos_seqs))]

        seq, nmer = choose_matching_seq(seqs, pos_seq)
        if seq is None:
            # no repetitions found in positive or no matches found in negatives
            # revert to default and pick a random sequence
            seq = seqs[np.random.randint(len(seqs))]
    

    # add padding from negative sequences
    pad_options = np.array(['NONE','BOTH','LEFT','RIGHT'])
    pad_weights = np.array([   2  ,   0  ,   0  ,   0   ])
    pad_choice = np.random.choice(pad_options, 1, p=pad_weights/sum(pad_weights))
    
    if negative:
        neg_seqs = seqs
    elif pad_choice != 'NONE':
        neg_seqs_fn = params["TF_DIR"]+f'/train_seqs/{gene_id[:2]}/{gene_id}_neg.afa'
        with open(neg_seqs_fn, 'r') as f_seqs:
            neg_seqs = [line.strip() for line in f_seqs]
    
    def get_pad(neg_seqs,MIN_PER=1,MAX_PER=8):
        pad_seq = np.random.choice(neg_seqs,1)[0]
        l_pad = np.random.randint(MIN_PER,MAX_PER+1)
        pad_idx = np.random.randint(0, len(pad_seq) - l_pad + 1)
        return pad_seq[pad_idx : (pad_idx+l_pad)]
        
    if pad_choice in ['LEFT','BOTH']:
        seq = get_pad(neg_seqs) + seq
    if pad_choice in ['RIGHT','BOTH']:
        seq = seq + get_pad(neg_seqs)

    # add sequence-unknown padding to DNA sequence for dimer predictions
    LEN_OFFSET = np.random.randint(-1,5)
    while len(seq) < 6 * nmer + LEN_OFFSET:
        if random.random() < 0.5:
            seq = seq + 'D'
        else:
            seq = 'D' + seq

    Ls = [L_prot, len(seq), len(seq)] 

    # oligomerize protein
    if nmer > 1:
        msaA, insA = merge_a3m_homo(a3mA['msa'].long(), a3mA['ins'].long(), nmer)
        a3mA['msa'] = msaA
        a3mA['ins'] = insA
    while len(Ls) < nmer + 2:
        Ls = [Ls[0]] + Ls

    # compute reverse sequence
    DNAPAIRS = {'A':'T','T':'A','C':'G','G':'C','D':'D'}
    rseq = ''.join([DNAPAIRS[x] for x in seq][::-1])

    # convert sequence to numbers and merge
    alphabet = np.array(list("00000000000000000000-0ACGTD00000"), dtype='|S1').view(np.uint8)
    msaB = np.array([list(seq)], dtype='|S1').view(np.uint8)
    msaC = np.array([list(rseq)], dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        msaB[msaB == alphabet[i]] = i
        msaC[msaC == alphabet[i]] = i
    insB = np.zeros((1,Ls[-2]))
    insC = np.zeros((1,Ls[-1]))
    a3mB = {'msa': torch.from_numpy(msaB), 'ins': torch.from_numpy(insB), 'label': HASH}
    a3mC = {'msa': torch.from_numpy(msaC), 'ins': torch.from_numpy(insC), 'label': HASH}

    a3mB = merge_a3m_hetero(a3mB, a3mC, [Ls[-2], Ls[-1]])
#    ic(a3mA['msa'].shape,a3mB['msa'].shape,Ls,gene_id)
    LA = a3mA['msa'].shape[1]
    LB = a3mB['msa'].shape[1]
    a3m  = merge_a3m_hetero(a3mA, a3mB, [LA,LB])
    L = sum(Ls)
    assert L == a3m['msa'].shape[1]

    # read template info (no template)
    ntempl = 0
    tpltA = {'ids':[]} # a fake tpltA
    xyz_t, f1d_t, mask_t, _ = TemplFeaturize(tpltA, L, params, offset=0, npick=ntempl, pick_top=pick_top, random_noise=random_noise)
    xyz_t[:,LA:] = ChemData().INIT_NA_CRDS.reshape(1,1,ChemData().NTOTAL,3).repeat(1,LB,1,1) + torch.rand(1,LB,1,3)*random_noise

    # get MSA features
    msa = a3m['msa'].long()
    ins = a3m['ins'].long()
    if len(msa) > params['BLOCKCUT']:
        msa, ins = MSABlockDeletion(msa, ins)
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params, L_s=Ls)

    # build dummy "native" in case a loss function expects it 
    xyz = torch.full((1, L, ChemData().NTOTAL, 3), np.nan)
    mask = torch.full((1, L, ChemData().NTOTAL), False)

    is_NA = is_nucleic(msa[0])
    xyz[:,is_NA] = ChemData().NIT_NA_CRDS.reshape(1,1,ChemData().NTOTAL,3).repeat(1,is_NA.sum(),1,1) + torch.rand(1,is_NA.sum(),1,3)*random_noise
    is_prot = ~is_NA
    xyz[:,is_prot] = ChemData().INIT_CRDS.reshape(1,1,ChemData().NTOTAL,3).repeat(1,is_prot.sum(),1,1) + torch.rand(1,is_prot.sum(),1,3)*random_noise

    xyz = torch.nan_to_num(xyz)

    # adjust residue indices for chain breaks
    idx = torch.arange(L)
    for i in range(1,len(Ls)):
        idx[sum(Ls[:i]):] += 100

    # determine which residue pairs are on the same chain
    chain_idx = torch.zeros((L,L)).long() # AKA "same_chain" in other places
    chain_idx[:LA, :LA] = 1

    chain_idx[LA:, LA:] = 1

    # other features
    bond_feats = bond_feats_from_Ls(Ls).long()
    chirals = torch.Tensor()
    ch_label = torch.zeros((L,)).long()
    for i in range(len(Ls)):
        ch_label[sum(Ls[:i]):sum(Ls[:i+1])] = i

    # Do cropping
    if sum(Ls) > params['CROP']:
#        print (f'started cropping ({item["gene_id"]})')

        sel = torch.full((L,), False)
        # use all DNA
        sel[LA:] = torch.full((LB,), True)

        # use a random continous stretch of protein (same for each monomer)
        pcrop = params['CROP'] - torch.sum(sel)
        pcrop_per, pcrop_rem = pcrop // nmer, pcrop % nmer
        remainder_places = np.array([np.random.randint(nmer) for _ in range(pcrop_rem)])
        prop_bonuses = [sum(remainder_places==n) for n in range(nmer)]

        cropbegin = np.random.randint(Ls[0]-pcrop_per+1)
        for n in range(nmer):
            start = sum(Ls[:n]) + cropbegin
            end = start + pcrop_per
            while prop_bonuses[n] > 0:
                prop_bonuses[n] -= 1
                if random.random() < 0.5 and end < sum(Ls[:n+1]):
                    end += 1
                elif start > 0:
                    start -= 1
            sel[start:end] = torch.full((end-start,), True)

#        print (f'got crop sele w/ total size {torch.sum(sel)} ({item["gene_id"]})')

        seq = seq[:,sel]
        msa_seed_orig = msa_seed_orig[:,:,sel]
        msa_seed = msa_seed[:,:,sel]
        msa_extra = msa_extra[:,:,sel]
        mask_msa = mask_msa[:,:,sel]
        mask_t = mask_t[:,sel]
        xyz = xyz[:,sel]
        mask = mask[:,sel]
        xyz_t = xyz_t[:,sel]
        f1d_t = f1d_t[:,sel]
        #
        idx = idx[sel]
        chain_idx = chain_idx[sel][:,sel]
        bond_feats = bond_feats[sel][:, sel]

    xyz_prev = xyz_t[0].clone()
    mask_prev = mask_t[0].clone()

    dist_matrix = get_bond_distances(bond_feats)

    if negative:
        task = 'neg_tf'
    else:
        task = 'tf'

    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa,\
           xyz.float(), mask, idx.long(), \
           xyz_t.float(), f1d_t.float(), mask_t, \
           xyz_prev.float(), mask_prev, \
           chain_idx, False, negative, \
           torch.zeros(seq.shape), bond_feats, dist_matrix, chirals, ch_label, 'C1', task, item

def loader_distil_tf(item, params, random_noise=5.0, pick_top=True, native_NA_frac=0.0, negative=False):
    # collect info
    gene_id = item['gene_id']
    Ls = item['LEN']
    oligo = item['oligo']
    dnaseq = item['DNA sequence']
    HASH = item['HASH']

    nmer = 2 if oligo == 'dimer' else 1

    ##################################
    # Load and prepare sequence data #
    ##################################
    # protein MSA from an a3m file
    a3mA = get_msa(params["TF_DIR"]+f'/a3m_v2/{gene_id[:2]}/{gene_id}_aligned_domain.a3m', HASH)

    # oligomerize protein
    if nmer > 1:
        msaA, insA = merge_a3m_homo(a3mA['msa'].long(), a3mA['ins'].long(), nmer)
        a3mA['msa'] = msaA
        a3mA['ins'] = insA
        fseq = 'DD' + dnaseq + 'DD'
    else:
        fseq = dnaseq
    
    # DNA from a single sequence
    DNAPAIRS = {'A':'T','T':'A','C':'G','G':'C','D':'D'}
    rseq = ''.join([DNAPAIRS[x] for x in fseq][::-1])

    # NOTE: padding?

    # convert sequence to numbers and merge
    alphabet = np.array(list("00000000000000000000-0ACGTD00000"), dtype='|S1').view(np.uint8)
    msaB = np.array([list(fseq)], dtype='|S1').view(np.uint8)
    msaC = np.array([list(rseq)], dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        msaB[msaB == alphabet[i]] = i
        msaC[msaC == alphabet[i]] = i
    insB = np.zeros((1,Ls[-2]))
    insC = np.zeros((1,Ls[-1]))
    a3mB = {'msa': torch.from_numpy(msaB), 'ins': torch.from_numpy(insB), 'label': HASH}
    a3mC = {'msa': torch.from_numpy(msaC), 'ins': torch.from_numpy(insC), 'label': HASH}

    a3mB = merge_a3m_hetero(a3mB, a3mC, [Ls[-2], Ls[-1]])
    a3m  = merge_a3m_hetero(a3mA, a3mB, [sum(Ls[:nmer]),sum(Ls[nmer:])])

    # get MSA features
    msa = a3m['msa'].long()
    ins = a3m['ins'].long()
    if len(msa) > params['BLOCKCUT']:
        msa, ins = MSABlockDeletion(msa, ins)
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params, L_s=Ls)

    ###################################
    # Load and prepare structure data #
    ###################################
    # load predicted structure as "truth"
    xyz, mask, _, pdbseq = parse_pdb(
            params["TF_DIR"]+f'/distill_v2/filtered/{gene_id[:2]}/{gene_id}_{dnaseq}.pdb',
            seq=True,
            lddt_mask=True
            )

    xyz = torch.from_numpy(xyz)
    mask = torch.from_numpy(mask)
    pdbseq = torch.from_numpy(pdbseq)

    # Don't need to remap because we load directly from .pdb with re-mapped chemical.py
    # read template info (no template)
    # NOTE: use templates?
    ntempl = 0
    tpltA = {'ids':[]} # a fake tpltA
    xyz_t, f1d_t, mask_t, _ = TemplFeaturize(tpltA, sum(Ls), params, offset=0, npick=ntempl, pick_top=True, random_noise=random_noise)
    NAstart = sum(Ls[:nmer])
    xyz_t[:,NAstart:] = ChemData().INIT_NA_CRDS.reshape(1,1,ChemData().NTOTAL,3).repeat(1,sum(Ls[-2:]),1,1) + torch.rand(1,sum(Ls[-2:]),1,3)*random_noise

    # other features
    idx = idx_from_Ls(Ls)
    same_chain = same_chain_2d_from_Ls(Ls)
    bond_feats = bond_feats_from_Ls(Ls).long()
    ch_label = torch.cat([torch.full((L_,), i) for i,L_ in enumerate(Ls)]).long()

    ###############
    # Do cropping #
    ###############
    
    if sum(Ls) > params['CROP']:
        sel = get_na_crop(seq[0], xyz, mask, torch.arange(sum(Ls)), Ls, params, negative=False)

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
        #
        idx = idx[sel]
        same_chain = same_chain[sel][:,sel]
        bond_feats = bond_feats[sel][:,sel]
        ch_label = ch_label[sel]

    chirals = torch.Tensor()
    dist_matrix = get_bond_distances(bond_feats)
    xyz_prev, mask_prev = generate_xyz_prev(xyz_t, mask_t, params)

    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa,\
           xyz.float(), mask, idx.long(), \
           xyz_t.float(), f1d_t.float(), mask_t, \
           xyz_prev.float(), mask_prev, \
           same_chain, False, False, \
           torch.zeros(seq.shape), bond_feats, dist_matrix, chirals, \
           ch_label, 'C1', "distil_tf", item

def choose_matching_seq(seqs, pos_seq):
    t0 = time.time()
    
    # convert all sequences to numerical msa format
    alphabet = np.array(list("ACTG"), dtype='|S1').view(np.uint8)
    N, L = len(seqs), len(seqs[0])
    t03 = time.time()
    msa = np.array(list(''.join(seqs)), dtype='|S1').view(np.uint8).reshape((N,L))
    pos_msa = np.array([list(s) for s in [pos_seq]], dtype='|S1').view(np.uint8)
    t04 = time.time()

    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i
        pos_msa[pos_msa == alphabet[i]] = i

    t05 = time.time()

    # efficiently get complement sequences based on the alphabet indexes
    pos_rc_msa = (pos_msa + 2) % 4
    pos_rc_msa = pos_rc_msa[:,::-1]
    pos_r_seq = ''.join(["ACTG"[i] for i in pos_rc_msa[0]])
    rc_msa = (msa + 2) % 4
    rc_msa = rc_msa[:,::-1]

    t06 = time.time()

    # identify length and placement of longest duplicate subsequence in positive sequence
    N, L = msa.shape
    # scan through all subseqs starting from 10 bp, down to 3 bp
    for l in range(min(L,10),2,-1):
        pos_counter = Counter(tuple(pos_msa[0,i:i+l]) for i in range(L + 1 - l))
        pos_counter.update(tuple(pos_rc_msa[0,i:i+l]) for i in range(L + 1 - l))

        # if all subseqs are unique, continue to next shorter length
        nrep = max(pos_counter.values())
        if nrep == 1:
            continue

        # else, find the count and sequence indexes of the most common subseq
        pos_subseq = pos_counter.most_common(1)[0][0]

        idxs  = tuple(i for i in range(L + 1 - l) if tuple(pos_msa[0,i:i+l]) == pos_subseq)
        idxs += tuple(i+L for i in range(L + 1 - l) if tuple(pos_rc_msa[0,i:i+l]) == pos_subseq)
        assert len(idxs) == nrep

        break
    else:
        # if all subseqs down to 3 bp are unique, fail to produce output
#        print(f"choose_matching_seq runtime was {time.time() - t0} seconds")
        return None, nrep

    
    t1 = time.time()

    # efficiently identify rows of msa where the analogous substrings match
    both_msa = np.concatenate((msa,rc_msa),axis=1)
    sub_msas = [both_msa[:,idxs[i]:idxs[i]+l].copy() for i in range(nrep)]
    sub_msa = sub_msas[0]
    for i in range(1,nrep):
        sub_msa -= sub_msas[i]
    match_mask = np.sum(sub_msa,axis=1) == 0
    
    t2 = time.time()
    matching_msa = msa[match_mask]

    # if there are enough hits, choose one at random and convert it back to a string
    N, L = matching_msa.shape
#    print(f"choose_matching_seq found {N} matches from {len(seqs)} seqs for a {l}-bp motif repeated {nrep} times")
    if N > 3:
        sel = matching_msa[np.random.randint(N),:]
        seq = ''.join(["ACTG"[i] for i in sel])
#        print(f"choose_matching_seq runtime was {time.time() - t0} seconds")
        return seq, nrep
    # if there aren't enough hits, failed to find a match
    else:
#        print(f"choose_matching_seq runtime was {time.time() - t0} seconds")
        return None, nrep


def loader_dna_rna(item, params, random_noise=5.0):
    # read PDBs
    pdb_ids = item['CHAINID'].split(':')

    filenameA = params['NA_DIR']+'/torch/'+pdb_ids[0][1:3]+'/'+pdb_ids[0]+'.pt'
    if os.path.exists(filenameA+".v3"):
        filenameA = filenameA+".v3"
    pdbA = torch.load(filenameA)
    pdbB = None
    if (len(pdb_ids)==2):
        filenameB = params['NA_DIR']+'/torch/'+pdb_ids[1][1:3]+'/'+pdb_ids[1]+'.pt'
        if os.path.exists(filenameB+".v3"):
            filenameB = filenameB+".v3"
        pdbB = torch.load(filenameB)

    # RNAs may have an MSA defined, return one if one exists, otherwise, return single-sequence msa
    msaA,insA = parse_fasta_if_exists(pdbA['seq'], params['NA_DIR']+'/torch/'+pdb_ids[0][1:3]+'/'+pdb_ids[0]+'.afa', rmsa_alphabet=True)
    a3m = {'msa':torch.from_numpy(msaA), 'ins':torch.from_numpy(insA)}
    if (len(pdb_ids)==2):
        msaB,insB = parse_fasta_if_exists(pdbB['seq'], params['NA_DIR']+'/torch/'+pdb_ids[1][1:3]+'/'+pdb_ids[1]+'.afa', rmsa_alphabet=True)
        a3mB = {'msa':torch.from_numpy(msaB), 'ins':torch.from_numpy(insB)}
        Ls = [a3m['msa'].shape[1],a3mB['msa'].shape[1]]
        a3m = merge_a3m_hetero(a3m, a3mB, Ls)
    else:
        Ls = [a3m['msa'].shape[1]]

    # get template features -- None
    L = sum(Ls)
    xyz_t = ChemData().INIT_NA_CRDS.reshape(1,1,ChemData().NTOTAL,3).repeat(1,L,1,1) + torch.rand(1,L,1,3)*random_noise
    f1d_t = torch.nn.functional.one_hot(torch.full((1, L), 20).long(), num_classes=ChemData().NAATOKENS-1).float() # all gaps
    mask_t = torch.full((1,L,ChemData().NTOTAL), False)
    conf = torch.zeros((1,L,1)).float() # zero confidence
    f1d_t = torch.cat((f1d_t, conf), -1)

    NMDLS = pdbA['xyz'].shape[0]

    # get MSA features
    msa = a3m['msa'].long()
    ins = a3m['ins'].long()
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params, L_s=Ls)

    xyz = torch.full((NMDLS, L, ChemData().NTOTAL, 3), np.nan).float()
    mask = torch.full((NMDLS, L, ChemData().NTOTAL), False)

    #
    if (len(pdb_ids)==2):
        #fd this can happen in rna/dna hybrids
        if (len(pdbB['xyz'].shape) == 3):
             pdbB['xyz'] = pdbB['xyz'].unsqueeze(0)
             pdbB['mask'] = pdbB['mask'].unsqueeze(0)
        xyz[:,:,:23] = torch.cat((pdbA['xyz'], pdbB['xyz']), dim=1)
        mask[:,:,:23] = torch.cat((pdbA['mask'], pdbB['mask']), dim=1)
    else:
        xyz[:,:,:23] = pdbA['xyz']
        mask[:,:,:23] = pdbA['mask']

    xyz, mask = remap_NA_xyz_tensors(xyz,mask,msa[0])

    # other features
    idx = torch.arange(L)
    if (len(pdb_ids)==2):
        idx[Ls[0]:] += ChemData().CHAIN_GAP
    same_chain = same_chain_2d_from_Ls(Ls)
    bond_feats = bond_feats_from_Ls(Ls).long()

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
        same_chain = same_chain[sel][:,sel]
        bond_feats = bond_feats[sel][:, sel]
    
    ntempl = xyz_t.shape[0]
    xyz_t = torch.stack(
        [center_and_realign_missing(xyz_t[i], mask_t[i], same_chain=same_chain) for i in range(ntempl)]
    )
    xyz_prev, mask_prev = generate_xyz_prev(xyz_t, mask_t, params)
    atom_frames = torch.zeros(0,3,2)
    chirals = torch.zeros(0,5)
    ch_label = torch.zeros((L,)).long()
    ch_label[Ls[0]:] = 1
    dist_matrix = get_bond_distances(bond_feats)

    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa,\
           xyz.float(), mask, idx.long(), \
           xyz_t.float(), f1d_t.float(), mask_t, \
           xyz_prev.float(), mask_prev, \
           same_chain, False, False, atom_frames, bond_feats, dist_matrix, chirals, ch_label, 'C1', "rna",item

def find_residues_to_atomize_covale(lig_partners, prot_partners, covale):
    """
    Updates partner lists to have atomized residues when residues are making
    covalent bonds with small molecules.  Also returns list of atomized
    residues so the other features, MSA, templates etc can be removed.

    Parameters
    ----------
    lig_partners : list of 5-tuples
        Ligands in this assembly. Format is as described in `loader_sm_compl_assembly`.
    prot_partners : list of 5-tuples
        Protein chains in this assembly. Format is as described in `loader_sm_compl_assembly`.
    covale : list
        List of cifutils.Bond objects representing inter-chain bonds in this PDB entry.

    Returns
    -------
    lig_partners : list of 5-tuples
        New list of ligands in this assembly, with additional "ligands" corresponding to
        residues to atomize.
    residues_to_atomize : list of tuples ((ch_letter, res_num, res_name), (ch_letter, xform_index))
    """
    if len(covale)==0:
        return lig_partners, set()

    residues_to_atomize = set()
    for bond in covale:
        # ignore bonds to hydrogens -- these are artifacts of PDB curation
        if bond.a[-1][0]=='H' or bond.b[-1][0]=='H':
            continue

        res_key = None
        i_prot = None
        i_lig = None

        # find protein partner that is bonded to ligand
        for i, (ch_letter, i_xf, n_contacts, min_dist, ptype) in enumerate(prot_partners):
            if bond.a[0] == ch_letter:
                i_prot = i
                res_key = bond.a
                break
            elif bond.b[0] == ch_letter:
                i_prot = i
                res_key = bond.b
                break

        # find ligand partner that is bonded to protein
        for i, (ligand, ch_xfs, n_contacts, min_dist, ptype) in enumerate(lig_partners):
            if any([bond.a[:3] == lig_res or bond.b[:3] == lig_res for lig_res in ligand]):
                i_lig = i
                break

        if i_prot is not None and i_lig is not None:
            lig_partner = lig_partners[i_lig]
            prot_partner = prot_partners[i_prot]

            # append to ligand partner the protein residue that it's bonded to
            lig_partner[0].append(res_key[:3])
            lig_partner[1].append(prot_partner[:2])

            # record this residue to remove from residue representations
            residues_to_atomize.add((res_key[:3], prot_partner[:2]))

    return lig_partners, residues_to_atomize


def featurize_asmb_prot(pdb_id, partners, params, chains, asmb_xfs, modres,
    chid2hash=None, pick_top=True, random_noise=5.0):
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
    partners : list of 5-tuples (partner, transform_index, num_contacts, min_dist, partner_type)
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
        Maps chain ids (<pdbid>_<chain_letter>) to hash strings used to name homology
        template and MSA files. If None, no templates are loaded.
    num_protein_chains : number of protein chains to include in the assembly, if set to None 
                        all neighboring protein chains will be loaded
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
    Ls_prot : list (N_chains,)
        Length of each protein chain
    ch_letters : list (N_chains,)
        Chain letter for each chain
    mod_residues_to_atomize : list
        List of tuples `((chain_letter, residue_num, residue_name),
        (chain_letter, xform_index))` representing chemically modified residues
        that should be atomized.
    """
    # assign number to each unique protein sequence, irrespective of chain letter
    chnum2chlet = map_identical_prot_chains(partners, chains, modres)

    # protein true coords
    xyz_prot, mask_prot, ch_label_prot, seq_prot = [], [], [], []
    xyz_t_prot, f1d_t_prot, mask_t_prot, tplt_ids = [], [], [], []
    ch_letters, Ls_prot = [], []
    for chnum, chlet_set in chnum2chlet.items():
        # every location of this chain
        partners_ch = [p for p in partners if (p[-1]=='polypeptide(L)') and (p[0] in chlet_set)]
        N_mer = len(partners_ch)
        xyz_chxf, mask_chxf, seq_chxf, mod_residues_to_atomize = [], [], [], []
        for p in partners_ch:
            xyz_, mask_, seq_, _, _, residues_to_atomize = cif_prot_to_xyz(chains[p[0]], asmb_xfs[p[1]], modres)
            residues_to_atomize = [(residue, (residue[0], p[1])) for residue in residues_to_atomize]
            xyz_chxf.append(xyz_) # (L, N_atoms, 3)
            mask_chxf.append(mask_)
            seq_chxf.append(seq_)
            mod_residues_to_atomize.extend(residues_to_atomize)
            Ls_prot.append(xyz_.shape[0])
            ch_letters.append(p[0])

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
            xyz_t_ch, f1d_t_ch, mask_t_ch, tplt_ids_ch = \
                blank_template(n_tmpl=1, L=xyz_ch.shape[1], random_noise=random_noise)
        else:
            pdb_hash = chid2hash[pdb_id+'_'+list(chlet_set)[0]] # chlet_set all have same hash
            tplt = torch.load(params['PDB_DIR']+'/torch/hhr/'+pdb_hash[:3]+'/'+pdb_hash+'.pt')
            xyz_t_, f1d_t_, mask_t_, tplt_ids_ = TemplFeaturize(tplt, Ls_prot[-1], params, npick=ntempl, 
                offset=0, pick_top=pick_top, random_noise=random_noise)
            xyz_t_ch = torch.cat([xyz_t_]+[random_rot_trans(xyz_t_) for i in range(N_mer-1)], dim=1) # (ntempl, L*N_mer, natm, 3)
            f1d_t_ch = torch.cat([f1d_t_]*N_mer, dim=1) # (ntempl, L*N_mer, 21)
            mask_t_ch = torch.cat([mask_t_]*N_mer, dim=1) # (ntempl, L*N_mer, natm)
            tplt_ids_ch = np.concatenate([tplt_ids_,], axis=0) # (ntempl) -- don't need to concatenate on the length dimension 

        xyz_t_prot.append(xyz_t_ch)
        f1d_t_prot.append(f1d_t_ch)
        mask_t_prot.append(mask_t_ch)
        tplt_ids.append(tplt_ids_ch)

    # cartesian product over each chain's location permutations
    xyz_prot = cartprodcat(xyz_prot) # (prod_i(N_perm_i), sum_i(L_i*N_mer_i), N_atoms, 3)
    mask_prot = cartprodcat(mask_prot) # (prod_i(N_perm_i), sum_i(L_i*N_mer_i), N_atoms)
    
    xyz_t_prot, f1d_t_prot, mask_t_prot, tplt_ids = \
        merge_hetero_templates(xyz_t_prot, f1d_t_prot, mask_t_prot, tplt_ids, Ls_prot)

    ch_label_prot = torch.cat(ch_label_prot, dim=0)
    seq_prot = torch.cat(seq_prot, dim=0)

    return xyz_prot, mask_prot.bool(), seq_prot, ch_label_prot, xyz_t_prot, f1d_t_prot, \
           mask_t_prot, Ls_prot, ch_letters, mod_residues_to_atomize, tplt_ids

def featurize_single_ligand(ligand, chains, covale, lig_xf_s, asmb_xfs, params):
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

    #fd keep track of the ligands that are added along length dimension (as opposed to conformation)
    unique_lig = []

    for lig_xf in lig_xf_s: # all possible locations for this ligand
        ch2xf = dict(lig_xf)
        xyz_, occ_, msa_, chid_, akeys_ = cif_ligand_to_xyz(lig_atoms, asmb_xfs, ch2xf)
        if (occ_==0).all(): continue # no valid atom positions
        mask_ = (occ_ > 0) # partially occupied atoms are considered valid

        mol_, bond_feats_ = cif_ligand_to_obmol(xyz_, akeys_, lig_atoms, lig_bonds)
        xyz_, mask_ = get_automorphs(mol_, xyz_, mask_)

        # clamp number of atom permutations to save GPU memory
        if xyz_.shape[0] > params['MAXNSYMM']:
            print(f'WARNING: Too many atom permutations ({xyz_.shape[0]}) in {ligand}. '\
                  f'Keeping only {params["MAXNSYMM"]}.')
            xyz_ = xyz_[:params['MAXNSYMM']]
            mask_ = mask_[:params['MAXNSYMM']]

        G = get_nxgraph(mol_)
        frames_ = get_atom_frames(msa_, G, omit_permutation=params['OMIT_PERMUTATE'])
        chirals_ = get_chirals(mol_, xyz_[0])
        # if ligand has too many masked atoms, remove all masked atoms
        # this avoids wasting compute on ligands missing entire chemical fragments
        # while keeping (most) masked atoms that are isolated and integral to a given fragment
        if (~mask_[0]).sum() > params['MAXMASKEDLIGATOMS']:
            xyz_ = xyz_[:,mask_[0]]
            occ_ = occ_[mask_[0]]
            msa_ = msa_[mask_[0]]
            chid_ = chid_[mask_[0]]
            akeys_ = [k for m,k in zip(mask_[0],akeys_) if m]
            bond_feats_ = bond_feats_[mask_[0]][:,mask_[0]]
            G = nx.Graph(bond_feats_.cpu().numpy())
            frames_ = get_atom_frames(msa_, G, omit_permutation=params['OMIT_PERMUTATE'])
            chirals_ = crop_chirals(chirals_, torch.where(mask_[0])[0])
            mask_ = mask_[:,mask_[0]]

        if chirals_.numel()>0:
            chirals_[:,:-1] = chirals_[:,:-1] + sum(Ls_lig)

        if ((occ_<1) & (occ_>0)).any(): 
            # partial occupancy, add to permutation dimension
            #if not ((occ_<1) & (occ_>0)).all():
            #    print('WARNING: Partial occupancy for a subset of atoms in ligand', ligand)
            #    print('         Adding to permutation dimension as alternate coordinates.')
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
                unique_lig = [True]
            else:
                xyz_lig[0] = torch.cat([xyz_lig[0], xyz_],dim=0)
                mask_lig[0] = torch.cat([mask_lig[0], mask_],dim=0)
                unique_lig.append(False)
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
            unique_lig.append(True)

    return xyz_lig, mask_lig, msa_lig, bond_feats_lig, akeys_lig, Ls_lig, frames_lig, \
           chirals_lig, resname_lig, unique_lig

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
    partners : list of 5-tuples (partner, transform_index, num_contacts, min_dist, partner_type)
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

    #fd keep track of the ligands that are added along length dimension (as opposed to conformation)
    uniques = []

    for ligkey, lig_xf_s in lig2xf.items():

        ligand = ast.literal_eval(ligkey)

        xyz_lig, mask_lig, msa_lig, bond_feats_lig, akeys_lig, Ls_lig, frames_lig, \
        chirals_lig, resname_lig, unique_lig = \
            featurize_single_ligand(ligand, chains, covale, lig_xf_s, asmb_xfs, params)

        # residue numbering offset for chirals
        for i in range(len(chirals_lig)):
            if chirals_lig[i].shape[0]>0:
                chirals_lig[i][:,:-1] = chirals_lig[i][:,:-1] + sum(Ls_sm)

        xyz_sm.extend(xyz_lig)
        mask_sm.extend(mask_lig)
        msa_sm.extend(msa_lig)
        bond_feats_sm.extend(bond_feats_lig)
        akeys_sm.extend(akeys_lig)
        Ls_sm.extend(Ls_lig)
        frames.extend(frames_lig)
        chirals.extend(chirals_lig)
        resnames.extend(resname_lig)
        uniques.extend(unique_lig)

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
           ch_label_sm, akeys_sm, resnames, uniques


def featurize_ligand_from_string(ligand_string: str, format: str = "smiles"):
    """featurize_ligand_from_smiles Featurizes a smiles string
    representing a ligand, in a way that can be input into RF2 All Atom.

    Args:
        smiles_string (str): A Smiles String representing a small molecule.

    Returns:
        _type_: Same outputs as _load_sm_from_item, as if the ligand was loaded
        from the RF database.
    """
    generate_conformer = False
    if format == "inchi" or format == "smiles" or format == "smi":
        # We only generate conformers if we are reading from a format
        # that doesn't speicify the coordinates
        generate_conformer = True
    
    mol, msa_sm, _, xyz_sm, mask_sm = parse_mol(filename=ligand_string, filetype=format, string=True, generate_conformer=generate_conformer)
    small_molecule_length = mol.NumAtoms()
    
    chirals = get_chirals(mol, xyz_sm[0])
    bond_feats_sm = get_bond_feats(mol)

    mol_graph = get_nxgraph(mol)
    frames = get_atom_frames(msa_sm, mol_graph, omit_permutation=params['OMIT_PERMUTATE'])
    ch_label_sm = torch.zeros((small_molecule_length), dtype=int)
    lig_names = [ligand_string]

    akeys_sm = []
    return xyz_sm, mask_sm, msa_sm.unsqueeze(0), [bond_feats_sm], frames, chirals, [small_molecule_length], ch_label_sm, akeys_sm, lig_names


def load_ligands_from_partners(
    lig_partners, prot_partners, asmb_xfs, chains, covale, params, mod_residues_to_atomize, 
    num_ligand_chains: Optional[int] = None, 
    check_for_nonpartner_duplicates: Optional[bool] = True
):
    import time

    lig_partners = lig_partners[:params['MAXLIGCHAINS']]
    if num_ligand_chains is not None:
        lig_partners = lig_partners[:min(num_ligand_chains, params['MAXLIGCHAINS'])]

    # update ligand partners to atomize residues that are covalently linked to proteins
    lig_partners, residues_to_atomize = find_residues_to_atomize_covale(lig_partners, prot_partners, covale) 

    # subsample non-standard residues to atomize
    mod_residues_to_atomize = [res for res in mod_residues_to_atomize 
                            if np.random.rand() < params['P_ATOMIZE_MODRES']]

    # update ligand partners and residues_to_atomize with modified residues to be atomized
    lig_partners.extend([([res_tuple], [ch_xf], -1, "nonpoly",) # multi-res ligand format
                        for (res_tuple, ch_xf) in mod_residues_to_atomize])
    residues_to_atomize.update(set(mod_residues_to_atomize))

    # load ligands
    xyz_sm, mask_sm, msa_sm, bond_feats_sm, frames, chirals, Ls_sm, ch_label_sm, akeys_sm, lig_names, uniques = \
        featurize_asmb_ligands(lig_partners, params, chains, asmb_xfs, covale)

    if (check_for_nonpartner_duplicates):
        try:
            xyz_sm, mask_sm = get_extra_identical_copies_from_chains(
                chains, covale, asmb_xfs, xyz_sm, mask_sm, Ls_sm, akeys_sm, lig_partners, exclude_covalent_to_protein=False
            )
        except Exception as e:
            print("Failed to get extra identical copies of the ligands from the cif assembly.")
            print(f"The error was: {e}.")
            pass

    return xyz_sm, mask_sm, msa_sm, bond_feats_sm, frames, chirals, Ls_sm, ch_label_sm, akeys_sm, lig_names, list(residues_to_atomize), uniques

def remove_unsupported_metals(
    lig_partners, xyz_prot, mask_prot, xyz_sm, mask_sm, 
    msa_sm, bond_feats_sm, frames, chirals, Ls_sm, ch_label_sm, akeys_sm, resnames, residues_to_atomize,
    prot_partners, asmb_xfs, chains, covale, params, mod_residues_to_atomize, num_ligand_chains, uniques,
    min_metal_contacts, min_metal_contact_dist
):
    i_start = 0
    lig_partners_new = []
    rebuild = False

    nligands = len(lig_partners)
    assert (len(uniques) >= nligands) #fd this is > since atomized residues may implicitly get added

    i_unique = -1
    for i_lig in range(nligands):
        if (not uniques[i_lig]):
            res = resnames[i_unique]
            if res not in ChemData().METAL_RES_NAMES:
                lig_partners_new.append(lig_partners[i_lig]) # alternate conf of a non-metal, keep
            continue

        i_unique += 1
        res = resnames[i_unique]
        i_stop = i_start+Ls_sm[i_unique]

        if res in ChemData().METAL_RES_NAMES:
            assert Ls_sm[i_unique]==1

            # 1) get ligand contacts
            ds = torch.linalg.norm(xyz_sm[0] - xyz_sm[0,i_start], dim=-1)
            nself = sum( ds[mask_sm[0]]<min_metal_contact_dist )-1 #-1 for self

            # 2) get protein contacts
            ds = torch.linalg.norm(xyz_prot[0,:,1] - xyz_sm[0,i_start], dim=-1)
            # trim to residue contacts (8 is maximal CA/SC atom distance)
            resmask = (ds < (min_metal_contact_dist + 8.0)) * mask_prot[0,:,1]
            ds = torch.linalg.norm(xyz_prot[0,resmask,:] - xyz_sm[0,i_start], dim=-1)
            nprot = sum( ds[mask_prot[0,resmask]]<min_metal_contact_dist )
            if nprot+nself >= min_metal_contacts:
                lig_partners_new.append(lig_partners[i_lig])
            else:
                rebuild = True
        else:
            # nonmetal, keep
            lig_partners_new.append(lig_partners[i_lig])

        i_start = i_stop

    if rebuild:
        if len(lig_partners_new) == 0: # no ligands left after trimming
            return (
                torch.tensor([]), torch.tensor([],dtype=torch.bool), torch.tensor([]), [], torch.tensor([],dtype=torch.long), \
                torch.tensor([]),  [], torch.tensor([],dtype=torch.long), [], [], [], []
            )
        else:
            return load_ligands_from_partners(
                    lig_partners_new, prot_partners, asmb_xfs, chains, covale, params, mod_residues_to_atomize, 
                    num_ligand_chains=num_ligand_chains,
                    check_for_nonpartner_duplicates=False
                )
    else:
        return xyz_sm, mask_sm, msa_sm, bond_feats_sm, frames, chirals, Ls_sm, ch_label_sm, akeys_sm, resnames, residues_to_atomize, uniques

def loader_sm_compl_assembly_single(*args, **kwargs): 
    kwargs['num_protein_chains'] = 1
    return loader_sm_compl_assembly(*args, **kwargs)


def loader_sm_compl_assembly(item, params, chid2hash=None, chid2taxid=None, chid2smpartners=None, task='sm_compl_asmb', 
    num_protein_chains=None, num_ligand_chains=None, pick_top=True, random_noise=5.0, fixbb=False, max_msa_seqs: Optional[int] = None,
    remove_residue=True):
    """Load protein/ligand assembly from pre-parsed CIF files. Outputs can
    represent multiple chains, which are ordered from most to least contacts
    with query ligand.  Protein chains all come before ligand chains, and
    protein chains with identical sequences are grouped contiguously.

    `all_partners` is a list of 5-tuples representing ligands and protein
    chains near the query ligand that should be featurized as part of the
    assembly. The 5-tuple has the form

        (partner, xforms, num_contacts, min_dist, partner_type)
        
    If `partner_type` is "polypeptide", then `partner` is the chain letter and
    `xforms` is an integer index of a coordinate transform in `asmb_xfs`. If
    `partner_type` is "nonpoly", then `partner` is a list of tuples
    `(chain_letter, res_num, res_name)` representing a ligand and `xforms` is a
    list of tuples `(chain_letter, xform_index)` representing transforms.
    `num_contacts` is the number of heavy atoms within 5A of the query ligand.
    `min_dist` is the minimum distance in angstroms between a heavy atom and
    the ligand.  
    """

    pdb_chain = item['CHAINID']
    pdb_id = pdb_chain.split('_')[0]
    # load pre-parsed cif assembly - requires cifutils.py in path for object definitions
    out = \
        pickle.load(gzip.open(params['MOL_DIR']+f'/{pdb_id[1:3]}/{pdb_id}.pkl.gz'))
    if len(out) == 4:
        chains, asmb, covale, modres = out
    elif len(out) == 5:
        chains, asmb, covale, meta, modres = out
    else:
        raise ValueError(f"cif parser returns {len(out)} values")
    # list of proteins and ligands to featurize
    prot_partners = [p for p in item['PARTNERS'] if p[-1]=='polypeptide(L)']
    prot_partners = prot_partners[:params['MAXPROTCHAINS']]
    if num_protein_chains is not None:
        prot_partners = prot_partners[:min(num_protein_chains, params['MAXPROTCHAINS'])]
    
    # get list of coordinate transforms to recreate this bio-assembly
    i_a = str(item['ASSEMBLY'])
    asmb_xfs = asmb[i_a]

    # load protein chains
    xyz_prot, mask_prot, seq_prot, ch_label_prot, xyz_t_prot, f1d_t_prot, \
    mask_t_prot, Ls_prot, ch_letters, mod_residues_to_atomize,tplt_ids = \
        featurize_asmb_prot(pdb_id, prot_partners, params, chains, asmb_xfs, modres,
                            chid2hash, pick_top=pick_top, random_noise=random_noise)

    # keep 1st template and random sample of others for params['MAXTPLT'] total
    if xyz_t_prot.shape[0] > params['MAXTPLT']:
        sel = np.concatenate([[0], np.random.permutation(xyz_t_prot.shape[0]-1)[:params['MAXTPLT']-1]+1])
        xyz_t_prot = xyz_t_prot[sel]
        mask_t_prot = mask_t_prot[sel]
        f1d_t_prot = f1d_t_prot[sel]

    # prune ligands based on exclusion list
    def _prune_lig_partners(lig_items, params):
        lig_partners = [p for p in lig_items if p[-1]=='nonpoly'] # remove polymer
        lig_partners = [
          p for p in lig_partners
          if (p[0][0][2] not in ChemData().METAL_RES_NAMES or np.random.rand() < params['P_METAL'])
        ] # remove metals (dep. on param)
        lig_partners = [ 
          p for p in lig_partners
          if p[0][0][2] not in params['ligands_to_remove']
        ] #fd remove exclusion list
        return lig_partners

    lig_partners = _prune_lig_partners(item['PARTNERS'], params)
    lig_partners = [(item['LIGAND'], item['LIGXF'], -1, -1, 'nonpoly')] + lig_partners
    xyz_sm, mask_sm, msa_sm, bond_feats_sm, frames, chirals, Ls_sm, ch_label_sm, akeys_sm, resnames, residues_to_atomize, uniques = \
        load_ligands_from_partners(
            lig_partners, prot_partners, asmb_xfs, chains, covale, params, mod_residues_to_atomize, 
            num_ligand_chains=num_ligand_chains,
            check_for_nonpartner_duplicates=False
        )

    #fd remove unsupported metals (param dependent)
    #fd this needs all ligand coords loaded, so unfortunately we potentially call load_ligands_from_partners twice
    if params['min_metal_contacts'] > 0:
        lig_partners_trim = lig_partners[:params['MAXLIGCHAINS']]
        xyz_sm, mask_sm, msa_sm, bond_feats_sm, frames, chirals, Ls_sm, ch_label_sm, akeys_sm, resnames, residues_to_atomize, uniques = \
            remove_unsupported_metals(
                lig_partners_trim, xyz_prot, mask_prot, xyz_sm, mask_sm, 
                msa_sm, bond_feats_sm, frames, chirals, Ls_sm, ch_label_sm, akeys_sm, resnames, residues_to_atomize,
                prot_partners, asmb_xfs, chains, covale, params, mod_residues_to_atomize, num_ligand_chains, uniques,
                params['min_metal_contacts'], params['min_metal_contact_dist']
            )

    # combine protein & ligand coordinates
    N_symm_prot = xyz_prot.shape[0]
    N_symm_sm = xyz_sm.shape[0]
    L_total = sum(Ls_prot)+sum(Ls_sm)

    xyz = torch.full((max(N_symm_prot, N_symm_sm), L_total, ChemData().NTOTAL, 3), np.nan).float()
    xyz[:N_symm_prot, :sum(Ls_prot)] = xyz_prot
    if xyz_sm.shape[0] > 0:
        xyz[:N_symm_sm, sum(Ls_prot):, 1, :] = xyz_sm

    mask = torch.full((max(N_symm_prot, N_symm_sm), L_total, ChemData().NTOTAL), False).bool()
    mask[:N_symm_prot, :sum(Ls_prot)] = mask_prot
    if xyz_sm.shape[0] > 0:
        mask[:N_symm_sm, sum(Ls_prot):, 1] = mask_sm

    # combine protein & ligand templates
    N_tmpl = xyz_t_prot.shape[0]
    if chid2smpartners is not None and params["SHOW_SM_TEMPLATES"]:
        assert num_protein_chains == 1, "templating ligands not supported for multiple protein chains (complications in xyz_prev)"
        xyz_t_sm, f1d_t_sm, mask_t_sm = generate_sm_template_feats(tplt_ids, resnames, akeys_sm, Ls_sm,chid2smpartners, params)
    else:
        xyz_t_sm, f1d_t_sm, mask_t_sm, _ = blank_template(N_tmpl, sum(Ls_sm), random_noise)
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
        chain_ids = [pdb_id+'_'+chlet for chlet in ch_letters]
        a3m_prot = load_multi_msa(chain_ids, Ls_prot, chid2hash, chid2taxid, params)
        if msa_sm.shape[0]>0:
            a3m_sm = dict(msa=msa_sm, ins=torch.zeros_like(msa_sm))
            a3m = merge_a3m_hetero(a3m_prot, a3m_sm, [sum(Ls_prot), sum(Ls_sm)])
        else:
            a3m = a3m_prot
        msa, ins = a3m['msa'].long(), a3m['ins'].long()
        
        # ensure that some MSA clusters are fully paired sequences
        # omit query sequence as in MSAFeaturize()
        seed_msa_clus = choose_multimsa_clusters(a3m_prot['is_paired'][1:], params)
    else:
        # no msa hash provided, return query sequence as msa
        msa = torch.cat([seq_prot[None], msa_sm],dim=1)
        ins = torch.zeros_like(msa)
        seed_msa_clus = None
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
                akeys_sm,
                remove_residue=remove_residue
            )
    ntempl = xyz_t.shape[0]
    xyz_t = torch.stack(
        [center_and_realign_missing(xyz_t[i], mask_t[i], same_chain=same_chain) for i in range(ntempl)]
    )

    xyz_prev, mask_prev = generate_xyz_prev(xyz_t, mask_t, params)

    xyz_prev = torch.nan_to_num(xyz_prev)
    xyz = torch.nan_to_num(xyz)
    xyz_t = torch.nan_to_num(xyz_t)

    # keep track of protein positions for reindexing chirals after crop
    L_total = sum(Ls_prot)+sum(Ls_sm)
    is_prot = torch.zeros(L_total) 
    is_prot[:sum(Ls_prot)] = 1

    # N/C-terminus features for MSA features (need to generate before cropping)
    term_info = get_term_feats(Ls_prot+Ls_sm)
    term_info[sum(Ls_prot):, :] = 0 # ligand chains don't get termini features
    
    # crop around query ligand (1st sm chain)
    # always need to run cropping function to remove erroneous ligand partners 
    if sum(Ls_sm)==0:
        sel = get_crop(len(idx), mask[0], msa.device, params['CROP'])
    else:
        if params['RADIAL_CROP']:
            sel = crop_sm_compl_assembly(xyz[0], mask[0], Ls_prot, Ls_sm, params['CROP'])
        else:
            sel = crop_sm_compl_asmb_contig(xyz[0], mask[0], Ls_prot, Ls_sm, bond_feats, params['CROP'], use_partial_ligands=False)
    mask = reassign_symmetry_after_cropping(sel, Ls_prot, ch_label, mask, item)

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
    term_info = term_info[sel]

    # crop small molecule features, assumes all sm chains are after all protein chains
    atom_sel = sel[sel>=sum(Ls_prot)] - sum(Ls_prot) # 0 index all the selected atoms
    frames = frames[atom_sel]
    chirals = crop_chirals(chirals, atom_sel)
        
    # reindex chiral atom positions - assumes all sm chains are after all protein chains
    if chirals.shape[0]>0:
        L1 = is_prot.sum()
        chirals[:, :-1] = chirals[:, :-1] + L1

    dist_matrix = get_bond_distances(bond_feats)
    
    # create MSA features from cropped msa and insertions
    # if len(msa) > params['BLOCKCUT']:
    #     msa, ins = MSABlockDeletion(msa.long(), ins.long())
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = \
        MSAFeaturize(msa.long(), ins.long(), params, p_mask=params["p_msa_mask"], term_info=term_info, fixbb=fixbb, seed_msa_clus=seed_msa_clus)

    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa,\
           xyz.float(), mask, idx.long(), \
           xyz_t.float(), f1d_t.float(), mask_t, \
           xyz_prev.float(), mask_prev, \
           same_chain, False, False, frames, bond_feats, dist_matrix, chirals, ch_label, 'C1', task, item

def loader_atomize_pdb(item, params, homo, n_res_atomize, flank, unclamp=False, 
    pick_top=True, p_homo_cut=0.5, random_noise=5.0):
    """ load pdb with portions represented as atoms instead of residues """
    pdb_chain, pdb_hash = item['CHAINID'], item['HASH']
    pdb = torch.load(params['PDB_DIR']+'/torch/pdb/'+pdb_chain[1:3]+'/'+pdb_chain+'.pt')
    a3m = get_msa(params['PDB_DIR'] + '/a3m/' + pdb_hash[:3] + '/' + pdb_hash + '.a3m.gz', pdb_hash)
    tplt = torch.load(params['PDB_DIR']+'/torch/hhr/'+pdb_hash[:3]+'/'+pdb_hash+'.pt')

    # get msa features
    msa = a3m['msa'].long()
    ins = a3m['ins'].long()
    if len(msa) > params['BLOCKCUT']:
        msa, ins = MSABlockDeletion(msa, ins)
    
    #fd -- do not do this
    #if params.get("NUM_SEQS_SUBSAMPLE", False):
    #    if len(msa) > params["NUM_SEQS_SUBSAMPLE"]:
    #        msa, ins = subsample_MSA(msa, ins, params["NUM_SEQS_SUBSAMPLE"])

    idx = torch.arange(len(pdb['xyz'])) 
    xyz = torch.full((len(idx),ChemData().NTOTAL,3), np.nan).float()
    xyz[:,:14,:] = pdb['xyz']
    mask = torch.full((len(idx), ChemData().NTOTAL), False)
    mask[:,:14] = pdb['mask']
    bond_feats = get_protein_bond_feats(len(idx))
    same_chain = torch.ones(len(idx), len(idx))
    # handle template features
    ntempl = np.random.randint(params['MINTPLT'], params['MAXTPLT']-1)
    ntempl = 0 # RK done to make atomization task harder
    xyz_t_prot, f1d_t_prot, mask_t_prot, _ = TemplFeaturize(tplt, len(pdb['xyz']), params, offset=0, 
        npick=ntempl, pick_top=pick_top, random_noise=random_noise)

    crop_len = params['CROP'] - n_res_atomize*14
    crop_idx = get_crop(len(idx), mask, msa.device, crop_len, unclamp=unclamp)
    msa_prot = msa[:, crop_idx]
    ins_prot = ins[:, crop_idx]
    xyz_prot = xyz[crop_idx]
    mask_prot = mask[crop_idx]
    idx = idx[crop_idx]
    xyz_t_prot = xyz_t_prot[:, crop_idx]
    f1d_t_prot = f1d_t_prot[:, crop_idx]
    mask_t_prot = mask_t_prot[:, crop_idx]
    bond_feats = bond_feats[crop_idx][:, crop_idx]
    same_chain = same_chain[crop_idx][:, crop_idx]
    protein_L, nprotatoms, _ = xyz_prot.shape

    # choose region to atomize
    can_atomize_mask = torch.ones((protein_L,))

    idx_missing_N = torch.where(~mask_prot[1:,0])[0]+1 # residues missing bb N, excluding 1st residue
    idx_missing_C = torch.where(~mask_prot[:-1,2])[0] # residues missing bb C, excluding last residue
    can_atomize_mask[idx_missing_N-1] = 0 # can't atomize residues before a missing N
    can_atomize_mask[idx_missing_C+1] = 0 # can't atomize residues after a missing C

    num_atoms_per_res = ChemData().allatom_mask[msa_prot[0],:14].sum(dim=-1) # how many atoms should each residue have?
    num_atoms_exist = mask_prot.sum(dim=-1) # how many atoms have coords in each residue?
    can_atomize_mask[(num_atoms_per_res != num_atoms_exist)] = 0
    can_atomize_idx = torch.where(can_atomize_mask)[0]
    
    # not enough valid residues to atomize and have space for flanks, treat as monomer example
    if flank + 1 >= can_atomize_idx.shape[0]-(n_res_atomize+flank+1):
        return featurize_single_chain(msa, ins, tplt, pdb, params, random_noise=random_noise) \
            + ("atomize_pdb", item,)
    
    res_idxs_to_atomize = None
    if params.get("ATOMIZE_CLUSTER", False) and (np.random.rand()<0.9): # 10% of time do continuous crop
        res_idxs_to_atomize = get_residue_contacts(xyz_prot[can_atomize_idx], can_atomize_idx, n_res_atomize)

    if res_idxs_to_atomize is None: # this is triggered if triple contact fails or if the task is not triple contact
        i_start = torch.randint(flank+1, can_atomize_idx.shape[0]-(n_res_atomize+flank+1),(1,))
        i_start = can_atomize_idx[i_start] # index of the first residue to be atomized

        for i_end in range(i_start+1, i_start + n_res_atomize):
            if i_end not in can_atomize_idx:
                n_res_atomize = int(i_end-i_start)
                #print(f'WARNING: n_res_atomize set to {n_res_atomize} due to not enough consecutive '\
                #      f'fully-resolved residues to atomize. {item} i_start={i_start}')
                break
        res_idxs_to_atomize = torch.arange(start=int(i_start), end=int(i_start+n_res_atomize))

    seq_atomize_all, ins_atomize_all, xyz_atomize_all, mask_atomize_all, frames_atomize_all, chirals_atomize_all, \
        bond_feats, same_chain = atomize_discontiguous_residues(res_idxs_to_atomize, msa_prot, xyz_prot, mask_prot, bond_feats, same_chain)

    atom_template_motif_idxs = get_atom_template_indices(msa_prot,res_idxs_to_atomize)

    # Generate ground truth structure: account for ligand symmetry
    N_symmetry, sm_L, _ = xyz_atomize_all.shape
    xyz = torch.full((N_symmetry, protein_L+sm_L, ChemData().NTOTAL, 3), np.nan).float()
    mask = torch.full(xyz.shape[:-1], False).bool()
    xyz[:, :protein_L, :nprotatoms, :] = xyz_prot.expand(N_symmetry, protein_L, nprotatoms, 3)
    xyz[:, protein_L:, 1, :] = xyz_atomize_all
    mask[:, :protein_L, :nprotatoms] = mask_prot.expand(N_symmetry, protein_L, nprotatoms)
    mask[:, protein_L:, 1] = mask_atomize_all
    
    # generate template for atoms
    if torch.rand(1) < params["P_ATOMIZE_TEMPLATE"]:
        xyz_t_sm, f1d_t_sm, mask_t_sm = spoof_template(xyz[0, protein_L:], seq_atomize_all, mask[0, protein_L:], atom_template_motif_idxs) 
    else:
        tplt_sm = {"ids":[]}
        xyz_t_sm, f1d_t_sm, mask_t_sm, _ = TemplFeaturize(tplt_sm, xyz_atomize_all.shape[1], params, offset=0, npick=0, pick_top=pick_top)
    ntempl = xyz_t_prot.shape[0]    
    xyz_t = torch.cat((xyz_t_prot, xyz_t_sm.repeat(ntempl,1,1,1)), dim=1)
    f1d_t = torch.cat((f1d_t_prot, f1d_t_sm.repeat(ntempl,1,1)), dim=1)
    mask_t = torch.cat((mask_t_prot, mask_t_sm.repeat(ntempl,1,1)), dim=1)

    Ls = [xyz_prot.shape[0], xyz_atomize_all.shape[1]]
    a3m_prot = {"msa": msa_prot, "ins": ins_prot}
    a3m_sm = {"msa": seq_atomize_all.unsqueeze(0), "ins": ins_atomize_all.unsqueeze(0)}

    a3m = merge_a3m_hetero(a3m_prot, a3m_sm, Ls)
    msa = a3m['msa'].long()
    ins = a3m['ins'].long()

    # handle res_idx
    last_res = idx[-1]
    idx_sm = torch.arange(Ls[1]) + last_res
    idx = torch.cat((idx, idx_sm))

    ch_label = torch.zeros(sum(Ls))
    # remove msa features for atomized portion
    msa, ins, xyz, mask, bond_feats, idx, xyz_t, f1d_t, mask_t, same_chain, ch_label = \
        pop_protein_feats(res_idxs_to_atomize, msa, ins, xyz, mask, bond_feats, idx, xyz_t, f1d_t, mask_t, same_chain, ch_label, Ls)
    
    # N/C-terminus features for MSA features (need to generate before cropping)
    # term_info = get_term_feats(Ls)
    # term_info[xyz_prot.shape[0]:, :] = 0 # ligand chains don't get termini features
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params, 
    #term_info=term_info
    )
    
    ntempl = xyz_t.shape[0]
    xyz_t = torch.stack(
        [center_and_realign_missing(xyz_t[i], mask_t[i], same_chain=same_chain) for i in range(ntempl)]
    )
    xyz_prev, mask_prev = generate_xyz_prev(xyz_t, mask_t, params)

    # xyz_prev = xyz_t[0].clone()
    # # xyz_prev[Ls[0]:] = xyz_prev[i_start] # no templates provided anymore this line won't work
    # mask_prev = mask_t[0].clone()
    xyz = torch.nan_to_num(xyz)

    dist_matrix = get_bond_distances(bond_feats)

    if chirals_atomize_all.shape[0]>0:
        L1 = torch.sum(~is_atom(seq[0]))
        chirals_atomize_all[:, :-1] = chirals_atomize_all[:, :-1] +L1
    
    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa,\
           xyz.float(), mask, idx.long(), \
           xyz_t.float(), f1d_t.float(), mask_t, \
           xyz_prev.float(), mask_prev, \
           same_chain, False, False, frames_atomize_all, bond_feats.long(), dist_matrix, chirals_atomize_all, \
           ch_label, 'C1', "atomize_pdb", item


def loader_atomize_complex(
    item, params, homo, n_res_atomize, flank, unclamp=False,
    pick_top=True, p_homo_cut=0.5, random_noise=5.0
):
    """ load complex with portions represented as atoms instead of residues """
    pdb_pair, pMSA_hash, L_s, taxID = item['CHAINID'], item['HASH'], item['LEN'], item['TAXONOMY']
    msaA_id, msaB_id = pMSA_hash.split('_')

    if len(set(taxID.split(':'))) == 1: # two proteins have same taxID -- use paired MSA
        # read pMSA
        pMSA_fn = params['COMPL_DIR'] + '/pMSA/' + msaA_id[:3] + '/' + msaB_id[:3] + '/' + pMSA_hash + '.a3m.gz'
        a3m = get_msa(pMSA_fn, pMSA_hash)
    else:
        # read MSA for each subunit & merge them
        a3mA_fn = params['PDB_DIR'] + '/a3m/' + msaA_id[:3] + '/' + msaA_id + '.a3m.gz'
        a3mB_fn = params['PDB_DIR'] + '/a3m/' + msaB_id[:3] + '/' + msaB_id + '.a3m.gz'
        a3mA = get_msa(a3mA_fn, msaA_id)
        a3mB = get_msa(a3mB_fn, msaB_id)
        a3m = merge_a3m_hetero(a3mA, a3mB, L_s)

    # get MSA features
    msa = a3m['msa'].long()
    ins = a3m['ins'].long()
    if len(msa) > params['BLOCKCUT']:
        msa, ins = MSABlockDeletion(msa, ins)

    # read template info
    tpltA_fn = params['PDB_DIR'] + '/torch/hhr/' + msaA_id[:3] + '/' + msaA_id + '.pt'
    tpltB_fn = params['PDB_DIR'] + '/torch/hhr/' + msaB_id[:3] + '/' + msaB_id + '.pt'
    tpltA = torch.load(tpltA_fn)
    tpltB = torch.load(tpltB_fn)

    ntemplA = np.random.randint(params['MINTPLT'], params['MAXTPLT']+1)
    ntemplB = np.random.randint(0, params['MAXTPLT']+1-ntemplA)
    xyz_t_A, f1d_t_A, mask_t_A, _ = TemplFeaturize(tpltA, L_s[0], params, offset=0, npick=ntemplA, npick_global=max(1,max(ntemplA, ntemplB)), pick_top=pick_top, random_noise=random_noise)
    xyz_t_B, f1d_t_B, mask_t_B, _ = TemplFeaturize(tpltB, L_s[1], params, offset=0, npick=ntemplB, npick_global=max(1,max(ntemplA, ntemplB)), pick_top=pick_top, random_noise=random_noise)
    xyz_t_prot = torch.cat((xyz_t_A, random_rot_trans(xyz_t_B)), dim=1) # (T, L1+L2, natm, 3)
    f1d_t_prot = torch.cat((f1d_t_A, f1d_t_B), dim=1) # (T, L1+L2, natm, 3)
    mask_t_prot = torch.cat((mask_t_A, mask_t_B), dim=1) # (T, L1+L2, natm, 3)

    # read PDB
    pdbA_id, pdbB_id = pdb_pair.split(':')
    pdbA = torch.load(params['PDB_DIR']+'/torch/pdb/'+pdbA_id[1:3]+'/'+pdbA_id+'.pt')
    pdbB = torch.load(params['PDB_DIR']+'/torch/pdb/'+pdbB_id[1:3]+'/'+pdbB_id+'.pt')

    # read metadata
    pdbid = pdbA_id.split('_')[0]
    meta = torch.load(params['PDB_DIR']+'/torch/pdb/'+pdbid[1:3]+'/'+pdbid+'.pt')

    # get transform
    xformA = meta['asmb_xform%d'%item['ASSM_A']][item['OP_A']]
    xformB = meta['asmb_xform%d'%item['ASSM_B']][item['OP_B']]

    # apply transform
    xyzA = torch.einsum('ij,raj->rai', xformA[:3,:3], pdbA['xyz']) + xformA[:3,3][None,None,:]
    xyzB = torch.einsum('ij,raj->rai', xformB[:3,:3], pdbB['xyz']) + xformB[:3,3][None,None,:]
    xyz = torch.full((sum(L_s), ChemData().NTOTAL, 3), np.nan).float()
    xyz[:,:14] = torch.cat((xyzA, xyzB), dim=0)
    mask = torch.full((sum(L_s), ChemData().NTOTAL), False)
    mask[:,:14] = torch.cat((pdbA['mask'], pdbB['mask']), dim=0)
    xyz = torch.nan_to_num(xyz)

    idx = torch.arange(sum(L_s))
    idx[L_s[0]:] += ChemData().CHAIN_GAP

    same_chain = torch.zeros((sum(L_s), sum(L_s))).long()
    same_chain[:L_s[0], :L_s[0]] = 1
    same_chain[L_s[0]:, L_s[0]:] = 1
    bond_feats = torch.zeros((sum(L_s), sum(L_s))).long()
    bond_feats[:L_s[0], :L_s[0]] = get_protein_bond_feats(L_s[0])
    bond_feats[L_s[0]:, L_s[0]:] = get_protein_bond_feats(sum(L_s[1:]))

    # center templates
    ntempl = xyz_t_prot.shape[0]
    xyz_t_prot = torch.stack(
        [center_and_realign_missing(xyz_t_prot[i], mask_t_prot[i], same_chain=same_chain) for i in range(ntempl)]
    )

    crop_len = params['CROP'] - n_res_atomize*12
    if sum(L_s) > crop_len:
        params_temp = copy.deepcopy(params)
        params_temp['CROP'] = crop_len
        crop_idx = get_spatial_crop(xyz, mask, torch.arange(sum(L_s)), L_s, params_temp, pdb_pair)
    else:
        crop_idx = torch.arange(sum(L_s))

    msa_prot = msa[:, crop_idx]
    ins_prot = ins[:, crop_idx]
    xyz_prot = xyz[crop_idx]
    mask_prot = mask[crop_idx]
    idx = idx[crop_idx]
    xyz_t_prot = xyz_t_prot[:, crop_idx]
    f1d_t_prot = f1d_t_prot[:, crop_idx]
    mask_t_prot = mask_t_prot[:, crop_idx]
    bond_feats = bond_feats[crop_idx][:, crop_idx]
    same_chain = same_chain[crop_idx][:, crop_idx]
    protein_L, nprotatoms, _ = xyz_prot.shape

    # choose region to atomize
    can_atomize_mask = torch.ones((protein_L,))

    idx_missing_N = torch.where(~mask_prot[1:,0])[0]+1 # residues missing bb N, excluding 1st residue
    idx_missing_C = torch.where(~mask_prot[:-1,2])[0] # residues missing bb C, excluding last residue
    can_atomize_mask[idx_missing_N-1] = 0 # can't atomize residues before a missing N
    can_atomize_mask[idx_missing_C+1] = 0 # can't atomize residues after a missing C

    num_atoms_per_res = ChemData().allatom_mask[msa_prot[0],:14].sum(dim=-1) # how many atoms should each residue have?
    num_atoms_exist = mask_prot.sum(dim=-1) # how many atoms have coords in each residue?
    can_atomize_mask[(num_atoms_per_res != num_atoms_exist)] = 0
    can_atomize_idx = torch.where(can_atomize_mask)[0]

    # not enough valid residues to atomize and have space for flanks, treat as complex example
    if flank + 1 >= can_atomize_idx.shape[0]-(n_res_atomize+flank+1):
        print ('error atomizing complex',item, flank)
        chirals = torch.Tensor()
        L_s = [ torch.sum(crop_idx<L_s[0]).numpy(), torch.sum(crop_idx>=L_s[0]).numpy() ]
        seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa_prot, ins_prot, params, L_s=L_s)
        ch_label = torch.zeros(seq[0].shape)
        ch_label[L_s[0]:] = 1
        xyz_prev, mask_prev = generate_xyz_prev(xyz_t_prot, mask_t_prot, params)
        dist_matrix = get_bond_distances(bond_feats)

        return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa,\
           xyz_prot.float(), mask_prot, idx.long(), \
           xyz_t_prot.float(), f1d_t_prot.float(), mask_t_prot, \
           xyz_prev.float(), mask_prev, \
           same_chain, False, False, torch.zeros(seq.shape), bond_feats.long(), dist_matrix, chirals, \
           ch_label, 'C1', "atomize_complex", item

    res_idxs_to_atomize = None
    if params.get("ATOMIZE_CLUSTER", False) and (np.random.rand()<0.9): # 10% of time do continuous crop
        res_idxs_to_atomize = get_residue_contacts(xyz_prot[can_atomize_idx], can_atomize_idx, n_res_atomize)

    if res_idxs_to_atomize is None: # this is triggered if triple contact fails or if the task is not triple contact
        i_start = torch.randint(flank+1, can_atomize_idx.shape[0]-(n_res_atomize+flank+1),(1,))
        i_start = can_atomize_idx[i_start] # index of the first residue to be atomized

        for i_end in range(i_start+1, i_start + n_res_atomize):
            if i_end not in can_atomize_idx:
                n_res_atomize = int(i_end-i_start)
                #print(f'WARNING: n_res_atomize set to {n_res_atomize} due to not enough consecutive '\
                #      f'fully-resolved residues to atomize. {item} i_start={i_start}')
                break
        res_idxs_to_atomize = torch.arange(start=int(i_start), end=int(i_start+n_res_atomize))

    seq_atomize_all, ins_atomize_all, xyz_atomize_all, mask_atomize_all, frames_atomize_all, chirals_atomize_all, \
        bond_feats, same_chain = atomize_discontiguous_residues(res_idxs_to_atomize, msa_prot, xyz_prot, mask_prot, bond_feats, same_chain)

    atom_template_motif_idxs = get_atom_template_indices(msa_prot,res_idxs_to_atomize)

    # Generate ground truth structure: account for ligand symmetry
    N_symmetry, sm_L, _ = xyz_atomize_all.shape
    xyz = torch.full((N_symmetry, protein_L+sm_L, ChemData().NTOTAL, 3), np.nan).float()
    mask = torch.full(xyz.shape[:-1], False).bool()
    xyz[:, :protein_L, :nprotatoms, :] = xyz_prot.expand(N_symmetry, protein_L, nprotatoms, 3)
    xyz[:, protein_L:, 1, :] = xyz_atomize_all
    mask[:, :protein_L, :nprotatoms] = mask_prot.expand(N_symmetry, protein_L, nprotatoms)
    mask[:, protein_L:, 1] = mask_atomize_all

    # generate template for atoms
    if torch.rand(1) < params["P_ATOMIZE_TEMPLATE"]:
        xyz_t_sm, f1d_t_sm, mask_t_sm = spoof_template(xyz[0, protein_L:], seq_atomize_all, mask[0, protein_L:], atom_template_motif_idxs)
    else:
        tplt_sm = {"ids":[]}
        xyz_t_sm, f1d_t_sm, mask_t_sm, _ = TemplFeaturize(tplt_sm, xyz_atomize_all.shape[1], params, offset=0, npick=0, pick_top=pick_top)
    ntempl = xyz_t_prot.shape[0]
    xyz_t = torch.cat((xyz_t_prot, xyz_t_sm.repeat(ntempl,1,1,1)), dim=1)
    f1d_t = torch.cat((f1d_t_prot, f1d_t_sm.repeat(ntempl,1,1)), dim=1)
    mask_t = torch.cat((mask_t_prot, mask_t_sm.repeat(ntempl,1,1)), dim=1)

    Ls = [xyz_prot.shape[0], xyz_atomize_all.shape[1]]
    a3m_prot = {"msa": msa_prot, "ins": ins_prot}
    a3m_sm = {"msa": seq_atomize_all.unsqueeze(0), "ins": ins_atomize_all.unsqueeze(0)}

    a3m = merge_a3m_hetero(a3m_prot, a3m_sm, Ls)
    msa = a3m['msa'].long()
    ins = a3m['ins'].long()

    # handle res_idx
    last_res = idx[-1]
    idx_sm = torch.arange(Ls[1]) + last_res
    idx = torch.cat((idx, idx_sm))

    ch_label = torch.zeros(sum(Ls))
    # remove msa features for atomized portion
    msa, ins, xyz, mask, bond_feats, idx, xyz_t, f1d_t, mask_t, same_chain, ch_label = \
        pop_protein_feats(res_idxs_to_atomize, msa, ins, xyz, mask, bond_feats, idx, xyz_t, f1d_t, mask_t, same_chain, ch_label, Ls)

    # N/C-terminus features for MSA features (need to generate before cropping)
    # term_info = get_term_feats(Ls)
    # term_info[xyz_prot.shape[0]:, :] = 0 # ligand chains don't get termini features
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params,
    #term_info=term_info
    )

    ntempl = xyz_t.shape[0]
    xyz_t = torch.stack(
        [center_and_realign_missing(xyz_t[i], mask_t[i], same_chain=same_chain) for i in range(ntempl)]
    )
    xyz_prev, mask_prev = generate_xyz_prev(xyz_t, mask_t, params)

    # xyz_prev = xyz_t[0].clone()
    # # xyz_prev[Ls[0]:] = xyz_prev[i_start] # no templates provided anymore this line won't work
    # mask_prev = mask_t[0].clone()
    xyz = torch.nan_to_num(xyz)

    dist_matrix = get_bond_distances(bond_feats)

    if chirals_atomize_all.shape[0]>0:
        L1 = torch.sum(~is_atom(seq[0]))
        chirals_atomize_all[:, :-1] = chirals_atomize_all[:, :-1] +L1

    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa,\
           xyz.float(), mask, idx.long(), \
           xyz_t.float(), f1d_t.float(), mask_t, \
           xyz_prev.float(), mask_prev, \
           same_chain, False, False, frames_atomize_all, bond_feats.long(), dist_matrix, chirals_atomize_all, \
           ch_label, 'C1', "atomize_complex", item


def loader_sm(item, params, pick_top=True):
    """Load small molecule with atom tokens. Also, compute frames for atom FAPE loss calc"""
    # Load small molecule
    fname = params['CSD_DIR']+'/torch/'+item['label'][:2]+'/'+item['label']+'.pt'
    data = torch.load(fname)

    mol, msa_sm, ins_sm, xyz_sm, mask_sm = parse_mol(data["mol2"], string=True)
    a3m = {"msa": msa_sm.unsqueeze(0), "ins": ins_sm.unsqueeze(0)}
    G = get_nxgraph(mol)
    frames = get_atom_frames(msa_sm, G, omit_permutation=params['OMIT_PERMUTATE'])

    if xyz_sm.shape[0] > params['MAXNSYMM']: # clip no. of symmetry variants to save GPU memory
        xyz_sm = xyz_sm[:params['MAXNSYMM']]
        mask_sm = mask_sm[:params['MAXNSYMM']]

    chirals = get_chirals(mol, xyz_sm[0])
    N_symmetry, sm_L, _ = xyz_sm.shape

    if sm_L < 2:
        print(f'WARNING [loader_sm]: Sm mol. {item} only has one atom. Skipping.')
        return [torch.tensor([-1])]*20 # flag for bad example

    # Generate ground truth structure: account for ligand symmetry
    xyz = torch.full((N_symmetry, sm_L, ChemData().NTOTAL, 3), np.nan).float()
    xyz[:, :, 1, :] = xyz_sm

    mask = torch.full(xyz.shape[:-1], False).bool()
    mask[:, :, 1] = True # CAs

    msa = a3m['msa'].long()
    ins = a3m['ins'].long()
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params)

    idx = torch.arange(sm_L)
    same_chain = torch.ones((sm_L, sm_L)).long()
    bond_feats = get_bond_feats(mol)
    dist_matrix = get_bond_distances(bond_feats)

    xyz_t, f1d_t, mask_t, _ = TemplFeaturize({"ids":[]}, sm_L, params, offset=0,
        npick=0, pick_top=pick_top)
    ntempl = xyz_t.shape[0]
    xyz_t = torch.stack(
        [center_and_realign_missing(xyz_t[i], mask_t[i], same_chain=same_chain) for i in range(ntempl)]
    )
    xyz_prev, mask_prev = generate_xyz_prev(xyz_t, mask_t, params)

    xyz = torch.nan_to_num(xyz)
    ch_label = torch.zeros(seq[0].shape)
    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa,\
           xyz.float(), mask, idx.long(), \
           xyz_t.float(), f1d_t.float(), mask_t, \
           xyz_prev.float(), mask_prev, \
           same_chain, False, False, frames, bond_feats, dist_matrix, chirals, ch_label, 'C1', "sm", item


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
    item = clus_df.sample(1, random_state=rng).to_dict(orient='records')[0]
    return copy.deepcopy(item) # prevents dataframe from being modified by downstream changes

def sample_item_sm_compl(df, ID, dedup_ligand=True):
    """Sample a protein-ligand training example from sequence cluster `ID` from
    the dataset represented by DataFrame `df`"""
    # get all examples in this cluster
    tmp_df = df[df.CLUSTER==ID]

    # uniformly sample from unique PDB chains
    chid = np.random.choice(tmp_df.CHAINID.drop_duplicates().values)
    tmp_df = tmp_df[tmp_df.CHAINID==chid]

    if dedup_ligand and "LIGAND" in tmp_df:
        # uniform sample from unique ligands
        lignames = list(set([x[0][2] for x in tmp_df['LIGAND']]))
        chosen_lig = np.random.choice(lignames)
        tmp_df = tmp_df[tmp_df['LIGAND'].apply(lambda x: x[0][2]==chosen_lig)]

    item = tmp_df.sample(1).to_dict(orient='records')[0] # choose 1 random row
    return copy.deepcopy(item) # prevents dataframe from being modified by downstream changes


class Dataset(data.Dataset):
    def __init__(
        self, IDs, loader, data_df, params, homo, unclamp_cut=0.9, pick_top=True, 
        p_short_crop=-1.0, p_dslf_crop=-1.0, p_homo_cut=-1.0, n_res_atomize=0, flank=0, seed=None
    ):
        self.IDs = IDs
        self.data_df = data_df
        self.loader = loader
        self.params = params
        self.homo = homo
        self.pick_top = pick_top
        self.unclamp_cut = unclamp_cut
        self.p_homo_cut = p_homo_cut
        self.p_short_crop = p_short_crop
        self.p_dslf_crop = p_dslf_crop
        self.n_res_atomize = n_res_atomize
        self.flank = flank
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        #print (index, ID, self.data_df)
        item = sample_item(self.data_df, ID, self.rng)
        kwargs = dict()
        if self.n_res_atomize > 0:
            kwargs['n_res_atomize'] = self.n_res_atomize
            kwargs['flank'] = self.flank
        else:
            kwargs['p_short_crop'] = self.p_short_crop
            kwargs['p_dslf_crop'] = self.p_dslf_crop

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
        return 5*len(self.IDs)

    def __getitem__(self, index):
        index = index % len(self.IDs)
        ID = self.IDs[index]
        item = sample_item(self.data_df, ID, self.rng)
        try:
            out = self.loader(item,
                          self.params,
                          pick_top = self.pick_top,
                          negative = self.negative,
                          native_NA_frac = self.native_NA_frac
            )
        except Exception as e:
            print('error in DatasetNAComplex',item)
            raise e
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

class DatasetTFComplex(data.Dataset):
    def __init__(self, IDs, loader, data_df, params, negative=False, seed=None):
        self.IDs = IDs
        self.data_df = data_df
        self.loader = loader
        self.params = params
        self.negative = negative
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return 5*len(self.IDs)

    def __getitem__(self, index):
        index = index % len(self.IDs)
        ID = self.IDs[index]
        item = sample_item(self.data_df, ID, self.rng)
        try:
            out = self.loader(item, self.params, negative=self.negative)
        except Exception as e:
            print('error in DatasetTFComplex',item)
            raise e
        return out

class DatasetDNADistil(data.Dataset):
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
        item = sample_item(self.data_df)
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
            raise e
        return out

class DatasetSMComplexAssembly(data.Dataset):
    def __init__(self, IDs, loader, data_df, chid2hash, chid2taxid, params, task, num_protein_chains=None, num_ligand_chains: Optional[int] = None, seed = None, select_farthest_residues: bool = False, load_ligand_from_column: Optional[str] = None, ligand_column_string_format: str = "sdf", is_negative: bool = False, ligand_dictionary: Optional[Dict] = None):
        self.IDs = IDs
        self.data_df = data_df
        self.loader = loader
        self.chid2hash = chid2hash
        self.chid2taxid = chid2taxid
        self.params = params
        self.task = task
        self.num_protein_chains = num_protein_chains
        self.num_ligand_chains = num_ligand_chains
        self.rng = np.random.RandomState(seed)
        self.select_farthest_residues = select_farthest_residues
        self.load_ligand_from_column = load_ligand_from_column
        self.ligand_column_string_format = ligand_column_string_format
        self.is_negative = is_negative
        self.ligand_dictionary = ligand_dictionary
        

    def __len__(self):
        return len(self.IDs)
        
    def __getitem__(self, index):
        ID = self.IDs[index]
        item = sample_item_sm_compl(self.data_df, ID)

        ligand_string_tuple = None
        if self.load_ligand_from_column is not None:
            possible_ligands = item[self.load_ligand_from_column]
            chosen_ligand = np.random.choice(possible_ligands)

            if self.ligand_dictionary is not None and chosen_ligand in self.ligand_dictionary:
                chosen_ligand = self.ligand_dictionary[chosen_ligand]

        try:
            out = self.loader(
                item,
                self.params,
                self.chid2hash,
                self.chid2taxid,
                task=self.task,
                num_protein_chains=self.num_protein_chains,
                num_ligand_chains=self.num_ligand_chains,
            )
        except Exception as e:
            print('error in DatasetSMComplexAssembly',item)
            raise e
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
    def __init__(
        self, ID_dict, dataset_dict, loader_dict, homo, chid2hash, chid2taxid,chid2smpartners, params, 
        native_NA_frac=0.05, p_homo_cut=0.0, p_short_crop=0.0, p_dslf_crop=0.0, unclamp_cut=0.9, 
        ligand_dictionary: Optional[Dict] = None
    ):

        self.ID_dict = ID_dict
        self.dataset_dict = dataset_dict
        self.loader_dict = loader_dict
        self.homo = homo
        self.p_homo_cut = p_homo_cut
        self.p_short_crop = p_short_crop
        self.p_dslf_crop = p_dslf_crop
        self.chid2hash = chid2hash
        self.chid2taxid = chid2taxid
        self.chid2smpartners = chid2smpartners
        self.params = params
        self.unclamp_cut = unclamp_cut
        self.native_NA_frac = native_NA_frac
        self.index_dict = OrderedDict([
            (k, np.arange(len(self.ID_dict[k]))) for k in self.dataset_dict.keys()
        ])
        self.ligand_dictionary = ligand_dictionary

        self.correct_dataset_ordering = ["pdb", "fb", "compl", "neg_compl", "na_compl", "neg_na_compl", "distil_tf","tf","neg_tf","rna","dna", "sm_compl", "metal_compl", "sm_compl_multi", "sm_compl_covale", "sm_compl_asmb", "sm", "atomize_pdb", "atomize_complex"]
        for index, (key, dataset_name) in enumerate(zip(self.index_dict.keys(), self.correct_dataset_ordering)):
            error_message = f"Expected dataset {dataset_name} at index {index}, but you provided dataset {key}. "
            error_message += "See DistilledDataset for the correct dataset names and ordering."
            assert key == dataset_name, error_message

    def __len__(self):
        return sum([len(v) for k,v in self.index_dict.items()])

    def __getitem__(self, index):
        p_unclamp = np.random.rand()

        # try:
        if True:
            # order of datasets here must match key order in self.dataset_dict
            offset = 0
            if index >= offset and index < offset + len(self.index_dict['pdb']):
                task = 'pdb'
                ID = self.ID_dict['pdb'][index-offset]
                item = sample_item(self.dataset_dict['pdb'], ID)
                out = self.loader_dict['pdb'](
                    item, self.params, self.homo, 
                    p_homo_cut=self.p_homo_cut, p_short_crop=self.p_short_crop, p_dslf_crop=self.p_dslf_crop, 
                    unclamp=(p_unclamp > self.unclamp_cut)
                )
            offset += len(self.index_dict['pdb'])

            if index >= offset and index < offset + len(self.index_dict['fb']):
                task = 'fb'
                ID = self.ID_dict['fb'][index-offset]
                item = sample_item(self.dataset_dict['fb'], ID)
                out = self.loader_dict['fb'](
                    item, self.params, p_short_crop=self.p_short_crop, p_dslf_crop=self.p_dslf_crop, unclamp=(p_unclamp > self.unclamp_cut))
            offset += len(self.index_dict['fb'])

            if index >= offset and index < offset + len(self.index_dict['compl']):
                task = 'compl'
                ID = self.ID_dict['compl'][index-offset]
                item = sample_item(self.dataset_dict['compl'], ID)
                out = self.loader_dict['compl'](item, self.params, negative=False)
            offset += len(self.index_dict['compl'])

            if index >= offset and index < offset + len(self.index_dict['neg_compl']):
                task = 'neg_compl'
                ID = self.ID_dict['neg_compl'][index-offset]
                item = sample_item(self.dataset_dict['neg_compl'], ID)
                out = self.loader_dict['neg_compl'](item, self.params, negative=True)
            offset += len(self.index_dict['neg_compl'])

            if index >= offset and index < offset + len(self.index_dict['na_compl']):
                task = 'na_compl'
                ID = self.ID_dict['na_compl'][index-offset]
                item = sample_item(self.dataset_dict['na_compl'], ID)
                out = self.loader_dict['na_compl'](item, self.params, negative=False, native_NA_frac=self.native_NA_frac)
            offset += len(self.index_dict['na_compl'])

            if index >= offset and index < offset + len(self.index_dict['neg_na_compl']):
                task = 'neg_na_compl'
                ID = self.ID_dict['neg_na_compl'][index-offset]
                item = sample_item(self.dataset_dict['neg_na_compl'], ID)
                out = self.loader_dict['neg_na_compl'](item, self.params, negative=True, native_NA_frac=self.native_NA_frac)
            offset += len(self.index_dict['neg_na_compl'])

            if index >= offset and index < offset + len(self.index_dict['distil_tf']):
                task = 'distil_tf'
                ID = self.ID_dict['distil_tf'][index-offset]
                item = sample_item(self.dataset_dict['distil_tf'], ID)
                out = self.loader_dict['distil_tf'](item, self.params)
            offset += len(self.index_dict['distil_tf'])

            if index >= offset and index < offset + len(self.index_dict['tf']):
                task = 'tf'
                ID = self.ID_dict['tf'][index-offset]
                item = sample_item(self.dataset_dict['tf'], ID)
                out = self.loader_dict['tf'](item, self.params, negative=False)
            offset += len(self.index_dict['tf'])

            if index >= offset and index < offset + len(self.index_dict['neg_tf']):
                task = 'neg_tf'
                ID = self.ID_dict['neg_tf'][index-offset]
                item = sample_item(self.dataset_dict['neg_tf'], ID)
                out = self.loader_dict['neg_tf'](item, self.params, negative=True)
            offset += len(self.index_dict['neg_tf'])

            if index >= offset and index < offset + len(self.index_dict['rna']):
                task = 'rna'
                ID = self.ID_dict['rna'][index-offset]
                item = sample_item(self.dataset_dict['rna'], ID)
                out = self.loader_dict['rna'](item, self.params)
            offset += len(self.index_dict['rna'])

            if index >= offset and index < offset + len(self.index_dict['dna']):
                task = 'dna'
                ID = self.ID_dict['dna'][index-offset]
                item = sample_item(self.dataset_dict['dna'], ID)
                out = self.loader_dict['dna'](item, self.params)
            offset += len(self.index_dict['dna'])

            if index >= offset and index < offset + len(self.index_dict['sm_compl']):
                task='sm_compl'
                ID = self.ID_dict['sm_compl'][index-offset]
                item = sample_item_sm_compl(self.dataset_dict['sm_compl'], ID)
                out = self.loader_dict['sm_compl'](item, self.params, self.chid2hash, 
                self.chid2taxid, self.chid2smpartners, task='sm_compl', num_protein_chains=1)
            offset += len(self.index_dict['sm_compl'])

            if index >= offset and index < offset + len(self.index_dict['metal_compl']): 
                task='metal_compl'
                ID = self.ID_dict['metal_compl'][index-offset]
                item = sample_item_sm_compl(self.dataset_dict['metal_compl'], ID)
                out = self.loader_dict['metal_compl'](item, self.params, self.chid2hash, 
                self.chid2taxid, task='metal_compl', num_protein_chains=1)
            offset += len(self.index_dict['metal_compl'])

            if index >= offset and index < offset + len(self.index_dict['sm_compl_multi']):
                task='sm_compl_multi'
                ID = self.ID_dict['sm_compl_multi'][index-offset]
                item = sample_item_sm_compl(self.dataset_dict['sm_compl_multi'], ID)
                out = self.loader_dict['sm_compl_multi'](item, self.params, self.chid2hash, 
                self.chid2taxid, task=task, num_protein_chains=1)
            offset += len(self.index_dict['sm_compl_multi'])

            if index >= offset and index < offset + len(self.index_dict['sm_compl_covale']):
                task='sm_compl_covale'
                ID = self.ID_dict['sm_compl_covale'][index-offset]
                item = sample_item_sm_compl(self.dataset_dict['sm_compl_covale'], ID)
                out = self.loader_dict['sm_compl_covale'](item, self.params, self.chid2hash, 
                self.chid2taxid, task=task)
            offset += len(self.index_dict['sm_compl_covale'])

            if index >= offset and index < offset + len(self.index_dict['sm_compl_asmb']):
                task = 'sm_compl_asmb'
                ID = self.ID_dict['sm_compl_asmb'][index-offset]
                item = sample_item_sm_compl(self.dataset_dict['sm_compl_asmb'], ID)
                out = self.loader_dict['sm_compl_asmb'](item, self.params, self.chid2hash, 
                self.chid2taxid, task=task)
            offset += len(self.index_dict['sm_compl_asmb'])

            if index >= offset and index < offset + len(self.index_dict['sm']):
                task="sm"
                ID = self.ID_dict['sm'][index-offset]
                item = sample_item(self.dataset_dict['sm'], ID)
                out = self.loader_dict['sm'](item, self.params)
            offset += len(self.index_dict['sm'])

            if index >= offset and index < offset + len(self.index_dict['atomize_pdb']):
                task = "atomize_pdb"
                ID = self.ID_dict['atomize_pdb'][index-offset]
                item = sample_item(self.dataset_dict['atomize_pdb'], ID)
                n_res_atomize = np.random.randint(self.params['NRES_ATOMIZE_MIN'], self.params['NRES_ATOMIZE_MAX']+1)
                out = self.loader_dict['atomize_pdb'](item,
                    self.params, self.homo, n_res_atomize, self.params['ATOMIZE_FLANK'], 
                    unclamp=(p_unclamp > self.unclamp_cut))
            offset += len(self.index_dict['atomize_pdb'])

            if index >= offset and index < offset + len(self.index_dict['atomize_complex']):
                task = "atomize_complex"
                ID = self.ID_dict['atomize_complex'][index-offset]
                item = sample_item(self.dataset_dict['atomize_complex'], ID)
                n_res_atomize = np.random.randint(self.params['NRES_ATOMIZE_MIN'], self.params['NRES_ATOMIZE_MAX']+1)
                out = self.loader_dict['atomize_complex'](item,
                    self.params, self.homo, n_res_atomize, self.params['ATOMIZE_FLANK'],
                    unclamp=(p_unclamp > self.unclamp_cut))
            offset += len(self.index_dict['atomize_complex'])

        # except Exception as e:
        #    print('error loading',item, '\n',repr(e), task)
        #    raise e
        return out
