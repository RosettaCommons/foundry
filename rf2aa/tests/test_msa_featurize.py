import torch
import os
from rf2aa.data.data_loader import get_term_feats, cluster_sum, MSAFeaturize
from rf2aa.data.parsers import parse_a3m
from rf2aa.chemical import ChemicalData as ChemData, initialize_chemdata
from rf2aa.util import is_atom, is_protein, is_nucleic
from rf2aa.tests.test_conditions import setup_data
from rf2aa.tensor_util import assert_equal
from rf2aa.trainer_new import seed_all
import numpy as np
import pytest
from functools import partial


def OldMSAFeaturize(msa, ins, params, p_mask=0.15, eps=1e-6, nmer=1, L_s=[], 
    term_info=None, tocpu=False, fixbb=False, seed_msa_clus=None, deterministic=False):
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
    if deterministic:
        seed_all()

    if fixbb:
        p_mask = 0
        msa = msa[:1]
        ins = ins[:1]
    N, L = msa.shape
    
    if term_info is None:
        if len(L_s)==0:
            L_s = [L]
        term_info = get_term_feats(L_s)
    term_info = term_info.to(msa.device)

    # raw MSA profile
    raw_profile = torch.nn.functional.one_hot(msa, num_classes=ChemData().NAATOKENS) # N x L x NAATOKENS
    raw_profile = raw_profile.float().mean(dim=0) # L x NAATOKENS

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

        # add MSA clusters pre-chosen before calling this function
        if seed_msa_clus is not None:
            sample_orig_shape = sample.shape
            sample_seed = seed_msa_clus[i_cycle]
            sample_more = torch.tensor([i for i in sample if i not in sample_seed])
            N_sample_more = len(sample) - len(sample_seed)
            if N_sample_more > 0:
                sample_more = sample_more[torch.randperm(len(sample_more))[:N_sample_more]]
                sample = torch.cat([sample_seed, sample_more])
            else:
                sample = sample_seed[:len(sample)] # take all clusters from pre-chosen ones

        msa_clust = torch.cat((msa[:1,:], msa[1:,:][sample[:Nclust-1]]), dim=0)
        ins_clust = torch.cat((ins[:1,:], ins[1:,:][sample[:Nclust-1]]), dim=0)

        # 15% random masking 
        # - 10%: aa replaced with a uniformly sampled random amino acid
        # - 10%: aa replaced with an amino acid sampled from the MSA profile
        # - 10%: not replaced
        # - 70%: replaced with a special token ("mask")
        seq = msa_clust[0]
        
        random_aa = torch.tensor([[0.05]*20 + [0.0]*(ChemData().NAATOKENS-20)], device=msa.device)
        same_aa = torch.nn.functional.one_hot(msa_clust, num_classes=ChemData().NAATOKENS)
        # explicitly remove probabilities from nucleic acids and atoms
        same_aa[..., ChemData().NPROTAAS:] = 0
        raw_profile[...,ChemData().NPROTAAS:] = 0
        probs = 0.1*random_aa + 0.1*raw_profile + 0.1*same_aa
        #probs = torch.nn.functional.pad(probs, (0, 1), "constant", 0.7)
        
        # explicitly set the probability of masking for nucleic acids and atoms
        probs[...,is_protein(seq),ChemData().MASKINDEX]=0.7
        probs[...,~is_protein(seq), :] = 0 # probably overkill but set all none protein elements to 0
        probs[1:, ~is_protein(seq),20] = 1.0 # want to leave the gaps as gaps
        probs[0,is_nucleic(seq), ChemData().MASKINDEX] = 1.0
        probs[0,is_atom(seq), ChemData().aa2num["ATM"]] = 1.0
        
        sampler = torch.distributions.categorical.Categorical(probs=probs)
        mask_sample = sampler.sample()

        mask_pos = torch.rand(msa_clust.shape, device=msa_clust.device) < p_mask
        #mask_pos[msa_clust>MASKINDEX]=False # no masking on NAs
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
        msa_clust_onehot = torch.nn.functional.one_hot(msa_masked, num_classes=ChemData().NAATOKENS)
        msa_extra_onehot = torch.nn.functional.one_hot(msa_extra, num_classes=ChemData().NAATOKENS)
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
        try:
            msa_extra = torch.cat((msa_extra_onehot[:Nextra], ins_extra[:,:,None], term_info[None].expand(Nextra,-1,-1)), dim=-1)
        except Exception as e:
            print('msa_extra.shape',msa_extra.shape)
            print('ins_extra.shape',ins_extra.shape)

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

datasets = ["pdb", "na_compl", "sm_compl", "sm_compl_covale"]
test_data = setup_data()
test_data = [v for k, v in test_data.items() if k in datasets]

@pytest.mark.parametrize("name,item,loader_params,chem_params,loader,loader_kwargs", test_data)
def test_msa_featurize_regression(name, item, loader_params, chem_params, loader, loader_kwargs):
    ChemData.reset()
    init = partial(initialize_chemdata,chem_params)
    init()

    hash = item["HASH"]
    msa_fn = os.path.join(
        loader_params["PDB_DIR"],
        "a3m",
        hash[:3],
        f"{hash}.a3m.gz"
        )
    msa, ins, taxids = parse_a3m(msa_fn)
    msa, ins = torch.from_numpy(msa).long(), torch.from_numpy(ins).long()
    old_out = OldMSAFeaturize(msa, ins, loader_params, p_mask=0, deterministic=True)
    new_out = MSAFeaturize(msa, ins, loader_params, p_mask=0, deterministic=True)

    for i in range(len(old_out)):
        assert_equal(old_out[i], new_out[i])
