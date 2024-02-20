import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract as einsum
import torch.utils.checkpoint as checkpoint
from rf2aa.util import *
from rf2aa.util_module import Dropout, get_clones, create_custom_forward, rbf, init_lecun_normal, get_res_atom_dist
from rf2aa.model.layers.Attention_module import Attention, TriangleMultiplication, TriangleAttention, FeedForwardLayer
from rf2aa.model.Track_module import PairStr2Pair, PositionalEncoding2D
from rf2aa.chemical import ChemicalData as ChemData

# Module contains classes and functions to generate initial embeddings

class MSA_emb(nn.Module):
    # Get initial seed MSA embedding
    def __init__(self, d_msa=256, d_pair=128, d_state=32, d_init=0,
                 minpos=-32, maxpos=32, maxpos_atom=8, p_drop=0.1, use_same_chain=False, enable_same_chain=False):
        if (d_init==0):
            d_init = 2*ChemData().NAATOKENS+2+2
            
        super(MSA_emb, self).__init__()
        self.emb = nn.Linear(d_init, d_msa) # embedding for general MSA
        self.emb_q = nn.Embedding(ChemData().NAATOKENS, d_msa) # embedding for query sequence -- used for MSA embedding
        self.emb_left = nn.Embedding(ChemData().NAATOKENS, d_pair) # embedding for query sequence -- used for pair embedding
        self.emb_right = nn.Embedding(ChemData().NAATOKENS, d_pair) # embedding for query sequence -- used for pair embedding
        self.emb_state = nn.Embedding(ChemData().NAATOKENS, d_state)
        self.pos = PositionalEncoding2D(d_pair, minpos=minpos, maxpos=maxpos, 
                                        maxpos_atom=maxpos_atom, p_drop=p_drop, use_same_chain=use_same_chain,
                                        enable_same_chain=enable_same_chain)
        self.enable_same_chain = enable_same_chain
        self.reset_parameter()
    
    def reset_parameter(self):
        self.emb = init_lecun_normal(self.emb)
        self.emb_q = init_lecun_normal(self.emb_q)
        self.emb_left = init_lecun_normal(self.emb_left)
        self.emb_right = init_lecun_normal(self.emb_right)
        self.emb_state = init_lecun_normal(self.emb_state)

        nn.init.zeros_(self.emb.bias)


    def _msa_emb(self, msa, seq):
        N = msa.shape[1]
        msa = self.emb(msa) # (B, N, L, d_pair) # MSA embedding
        tmp = self.emb_q(seq).unsqueeze(1) # (B, 1, L, d_pair) -- query embedding
        msa = msa + tmp.expand(-1, N, -1, -1) # adding query embedding to MSA

        return msa

    def _pair_emb(self, seq, idx, bond_feats, dist_matrix, same_chain=None):
        left = self.emb_left(seq)[:,None] # (B, 1, L, d_pair)
        right = self.emb_right(seq)[:,:,None] # (B, L, 1, d_pair)
        pair = left + right # (B, L, L, d_pair)
        pair = pair + self.pos(seq, idx, bond_feats, dist_matrix, same_chain=same_chain) # add relative position

        return pair

    def _state_emb(self, seq):
        return self.emb_state(seq)

    def forward(self, msa, seq, idx, bond_feats, dist_matrix, same_chain=None):
        # Inputs:
        #   - msa: Input MSA (B, N, L, d_init)
        #   - seq: Input Sequence (B, L)
        #   - idx: Residue index
        #   - bond_feats: Bond features (B, L, L)
        # Outputs:
        #   - msa: Initial MSA embedding (B, N, L, d_msa)
        #   - pair: Initial Pair embedding (B, L, L, d_pair)

        if self.enable_same_chain == False:
            same_chain = None

        msa = self._msa_emb(msa, seq)

        # pair embedding 
        pair = self._pair_emb(seq, idx, bond_feats, dist_matrix, same_chain=same_chain)
        # state embedding
        state = self._state_emb(seq)
        return msa, pair, state

class MSA_emb_nostate(MSA_emb):
    def __init__(self, d_msa=256, d_pair=128, d_state=32, d_init=0, minpos=-32, maxpos=32, maxpos_atom=8, p_drop=0.1, use_same_chain=False):
        super().__init__(d_msa, d_pair, d_state, d_init, minpos, maxpos, maxpos_atom, p_drop, use_same_chain)
        if d_init==0:
            d_init = 2*ChemData().NAATOKENS + 2 + 2
        self.emb_state = None # emb state is just the identity

    def forward(self, msa, seq, idx, bond_feats, dist_matrix):
        msa = self._msa_emb(msa, seq)
        pair = self._pair_emb(seq, idx, bond_feats, dist_matrix)
        return msa, pair, None 

class Extra_emb(nn.Module):
    # Get initial seed MSA embedding
    def __init__(self, d_msa=256, d_init=0, p_drop=0.1):
        super(Extra_emb, self).__init__()
        if d_init==0:
            d_init=ChemData().NAATOKENS-1+4
        self.emb = nn.Linear(d_init, d_msa) # embedding for general MSA
        self.emb_q = nn.Embedding(ChemData().NAATOKENS, d_msa) # embedding for query sequence
        #self.drop = nn.Dropout(p_drop)
        

        self.reset_parameter()
    
    def reset_parameter(self):
        self.emb = init_lecun_normal(self.emb)
        nn.init.zeros_(self.emb.bias)

    def forward(self, msa, seq, idx):
        # Inputs:
        #   - msa: Input MSA (B, N, L, d_init)
        #   - seq: Input Sequence (B, L)
        #   - idx: Residue index
        # Outputs:
        #   - msa: Initial MSA embedding (B, N, L, d_msa)
        N = msa.shape[1] # number of sequenes in MSA
        msa = self.emb(msa) # (B, N, L, d_model) # MSA embedding
        seq = self.emb_q(seq).unsqueeze(1) # (B, 1, L, d_model) -- query embedding
        msa = msa + seq.expand(-1, N, -1, -1) # adding query embedding to MSA
        #return self.drop(msa)
        return (msa)

class Bond_emb(nn.Module):
    def __init__(self, d_pair=128, d_init=0):
        super(Bond_emb, self).__init__()

        if d_init==0:
            d_init = ChemData().NBTYPES

        self.emb = nn.Linear(d_init, d_pair)

        self.reset_parameter()
    
    def reset_parameter(self):
        self.emb = init_lecun_normal(self.emb)
        nn.init.zeros_(self.emb.bias)

    def forward(self, bond_feats):
        bond_feats = torch.nn.functional.one_hot(bond_feats, num_classes=ChemData().NBTYPES)
        return self.emb(bond_feats.float())

class TemplatePairStack(nn.Module):
    def __init__(self, n_block=2, d_templ=64, n_head=4, d_hidden=32, d_t1d=22, d_state=32, p_drop=0.25,
                 symmetrize_repeats=False, repeat_length=None, symmsub_k=1, sym_method=None):

        super(TemplatePairStack, self).__init__()
        self.n_block = n_block
        self.proj_t1d = nn.Linear(d_t1d, d_state)

        proc_s = [PairStr2Pair(d_pair=d_templ, 
                               n_head=n_head, 
                               d_hidden=d_hidden, 
                               d_state=d_state, 
                               p_drop=p_drop,
                               symmetrize_repeats=symmetrize_repeats,
                               repeat_length=repeat_length,
                               symmsub_k=symmsub_k,
                               sym_method=sym_method) for i in range(n_block)]

        self.block = nn.ModuleList(proc_s)
        self.norm = nn.LayerNorm(d_templ)
        self.reset_parameter()

    def reset_parameter(self):
        self.proj_t1d = init_lecun_normal(self.proj_t1d)
        nn.init.zeros_(self.proj_t1d.bias)

    def forward(self, templ, rbf_feat, t1d, use_checkpoint=False, p2p_crop=-1):
        B, T, L = templ.shape[:3]
        templ = templ.reshape(B*T, L, L, -1)
        t1d = t1d.reshape(B*T, L, -1)
        state = self.proj_t1d(t1d)

        for i_block in range(self.n_block):
            if use_checkpoint:
                templ = checkpoint.checkpoint(
                    create_custom_forward(self.block[i_block]), 
                    templ, rbf_feat, state, p2p_crop,
                    use_reentrant=True
                )
            else:
                templ = self.block[i_block](templ, rbf_feat, state)
        return self.norm(templ).reshape(B, T, L, L, -1)


def copy_main_2d(pair, Leff, idx):
    """
    Copies the "main unit" of a block in generic 2D representation of shape (...,L,L,h)
    along the main diagonal
    """
    start = idx*Leff
    end   = (idx+1)*Leff

    # grab the main block 
    main  = torch.clone( pair[..., start:end, start:end, :] )

    # copy it around the main diag 
    L = pair.shape[-2]
    assert L%Leff == 0
    N = L//Leff

    for i_block in range(N):
        start = i_block*Leff
        stop  = (i_block+1)*Leff

        pair[...,start:stop, start:stop, :] = main

    return pair 


def copy_main_1d(single, Leff, idx):
    """
    Copies the "main unit" of a block in generic 1D representation of shape (...,L,h)
    to all other (non-main) blocks 
    
    Parameters:
        single (torch.tensor, required): Shape [...,L,h] "1D" tensor
    """
    main_start = idx*Leff
    main_end   = (idx+1)*Leff

    # grab main block 
    main = torch.clone(single[..., main_start:main_end, :])

    # copy it around
    L = single.shape[-2]
    assert L%Leff == 0
    N = L//Leff

    for i_block in range(N):
        start = i_block*Leff
        end   = (i_block+1)*Leff

        single[..., start:end, :] = main

    return single


class Templ_emb(nn.Module):
    # Get template embedding
    # Features are
    #   t2d:
    #   - 61 distogram bins + 6 orientations (67)
    #   - Mask (missing/unaligned) (1)
    #   t1d:
    #   - tiled AA sequence (20 standard aa + gap)
    #   - confidence (1)
    #   
    def __init__(self, d_t1d=0, d_t2d=67+1, d_tor=0, d_pair=128, d_state=32, 
                 n_block=2, d_templ=64,
                 n_head=4, d_hidden=16, p_drop=0.25,
                 symmetrize_repeats=False, repeat_length=None, symmsub_k=1, sym_method='mean', 
                 main_block=None, copy_main_block=None, additional_dt1d=0):
        if d_t1d==0:
            d_t1d=(ChemData().NAATOKENS-1)+1
        if d_tor==0:
            d_tor=3*ChemData().NTOTALDOFS

        self.main_block = main_block
        self.symmetrize_repeats = symmetrize_repeats
        self.copy_main_block = copy_main_block
        self.repeat_length = repeat_length
        d_t1d += additional_dt1d

        super(Templ_emb, self).__init__()
        # process 2D features
        self.emb = nn.Linear(d_t1d*2+d_t2d, d_templ)

        self.templ_stack = TemplatePairStack(n_block=n_block, d_templ=d_templ, n_head=n_head,
                                             d_hidden=d_hidden, d_t1d=d_t1d, d_state=d_state, p_drop=p_drop,
                                             symmetrize_repeats=symmetrize_repeats, repeat_length=repeat_length,
                                             symmsub_k=symmsub_k, sym_method=sym_method)
        
        self.attn = Attention(d_pair, d_templ, n_head, d_hidden, d_pair, p_drop=p_drop)
        
        # process torsion angles
        self.emb_t1d = nn.Linear(d_t1d+d_tor, d_templ)
        self.proj_t1d = nn.Linear(d_templ, d_templ)
        #self.tor_stack = TemplateTorsionStack(n_block=n_block, d_templ=d_templ, n_head=n_head,
        #                                      d_hidden=d_hidden, p_drop=p_drop)
        self.attn_tor = Attention(d_state, d_templ, n_head, d_hidden, d_state, p_drop=p_drop)

        self.reset_parameter()
    
    def reset_parameter(self):
        self.emb = init_lecun_normal(self.emb)
        nn.init.zeros_(self.emb.bias)

        nn.init.kaiming_normal_(self.emb_t1d.weight, nonlinearity='relu')
        nn.init.zeros_(self.emb_t1d.bias)
        
        self.proj_t1d = init_lecun_normal(self.proj_t1d)
        nn.init.zeros_(self.proj_t1d.bias)

    def _get_templ_emb(self, t1d, t2d):
        B, T, L, _ = t1d.shape
        # Prepare 2D template features
        left = t1d.unsqueeze(3).expand(-1,-1,-1,L,-1)
        right = t1d.unsqueeze(2).expand(-1,-1,L,-1,-1)
        #
        templ = torch.cat((t2d, left, right), -1) # (B, T, L, L, 88)
        return self.emb(templ) # Template templures (B, T, L, L, d_templ)

    def _get_templ_rbf(self, xyz_t, mask_t):
        B, T, L = xyz_t.shape[:3]

        # process each template features
        xyz_t = xyz_t.reshape(B*T, L, 3).contiguous()
        mask_t = mask_t.reshape(B*T, L, L)
        assert(xyz_t.is_contiguous())
        rbf_feat = rbf(torch.cdist(xyz_t, xyz_t)) * mask_t[...,None] # (B*T, L, L, d_rbf)
        return rbf_feat

    def forward(self, t1d, t2d, alpha_t, xyz_t, mask_t, pair, state, use_checkpoint=False, p2p_crop=-1):
        # Input
        #   - t1d: 1D template info (B, T, L, 30)
        #   - t2d: 2D template info (B, T, L, L, 44)
        #   - alpha_t: torsion angle info (B, T, L, 30) - DOUBLE-CHECK
        #   - xyz_t: template CA coordinates (B, T, L, 3)
        #   - mask_t: is valid residue pair? (B, T, L, L)
        #   - pair: query pair features (B, L, L, d_pair)
        #   - state: query state features (B, L, d_state)
        B, T, L, _ = t1d.shape

        templ = self._get_templ_emb(t1d, t2d)
        # this looks a lot like a bug but it is not
        # mask_t has already been updated by same_chain in the train_EMA script so pairwise distances between
        # protein chains are ignored
        rbf_feat = self._get_templ_rbf(xyz_t, mask_t) 

        # process each template pair feature
        templ = self.templ_stack(templ, rbf_feat, t1d, use_checkpoint=use_checkpoint, p2p_crop=p2p_crop) # (B, T, L,L, d_templ)

        # DJ - repeat protein symmetrization (2D)
        if self.copy_main_block:
            assert not (self.main_block is None)
            assert self.symmetrize_repeats
            # copy the main repeat unit internally down the pair representation diagonal 
            templ = copy_main_2d(templ, self.repeat_length, self.main_block)

        # Prepare 1D template torsion angle features
        t1d = torch.cat((t1d, alpha_t), dim=-1) # (B, T, L, 30+3*17)
        # process each template features
        t1d = self.proj_t1d(F.relu_(self.emb_t1d(t1d)))

        # DJ - repeat protein symmetrization (1D)
        if self.copy_main_block:
            # already made assertions above 
            # copy main unit down single rep 
            t1d = copy_main_1d(t1d, self.repeat_length, self.main_block)
        
        # mixing query state features to template state features
        state = state.reshape(B*L, 1, -1)
        t1d = t1d.permute(0,2,1,3).reshape(B*L, T, -1)
        if use_checkpoint:
            out = checkpoint.checkpoint(
                create_custom_forward(self.attn_tor), state, t1d, t1d, use_reentrant=True
            )
            out = out.reshape(B, L, -1)
        else:
            out = self.attn_tor(state, t1d, t1d).reshape(B, L, -1)
        state = state.reshape(B, L, -1)
        state = state + out

        # mixing query pair features to template information (Template pointwise attention)
        pair = pair.reshape(B*L*L, 1, -1)
        templ = templ.permute(0, 2, 3, 1, 4).reshape(B*L*L, T, -1)
        if use_checkpoint:
            out = checkpoint.checkpoint(
                create_custom_forward(self.attn), pair, templ, templ, use_reentrant=True
            )
            out = out.reshape(B, L, L, -1)
        else:
            out = self.attn(pair, templ, templ).reshape(B, L, L, -1)
        #
        pair = pair.reshape(B, L, L, -1)
        pair = pair + out

        return pair, state


class Recycling(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, d_state=32, d_rbf=64):
        super(Recycling, self).__init__()
        self.proj_dist = nn.Linear(d_rbf, d_pair)
        self.norm_pair = nn.LayerNorm(d_pair)
        self.norm_msa = nn.LayerNorm(d_msa)
        
        self.reset_parameter()
    
    def reset_parameter(self):
        #self.emb_rbf = init_lecun_normal(self.emb_rbf)
        #nn.init.zeros_(self.emb_rbf.bias)
        self.proj_dist = init_lecun_normal(self.proj_dist)
        nn.init.zeros_(self.proj_dist.bias)

    def forward(self, msa, pair, xyz, state, sctors, mask_recycle=None):
        B, L = msa.shape[:2]
        msa = self.norm_msa(msa)
        pair = self.norm_pair(pair)

        Ca = xyz[:,:,1]
        dist_CA = rbf(
            torch.cdist(Ca, Ca)
        ).reshape(B,L,L,-1)

        if mask_recycle != None:
            dist_CA = mask_recycle[...,None].float()*dist_CA

        pair = pair + self.proj_dist(dist_CA) 

        return msa, pair, state # state is just zeros

class RecyclingAllFeatures(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, d_state=32, d_rbf=64):
        super(RecyclingAllFeatures, self).__init__()
        self.proj_dist = nn.Linear(d_rbf+d_state*2, d_pair)
        self.norm_pair = nn.LayerNorm(d_pair)
        self.proj_sctors = nn.Linear(2*ChemData().NTOTALDOFS, d_msa)
        self.norm_msa = nn.LayerNorm(d_msa)
        self.norm_state = nn.LayerNorm(d_state)

        self.reset_parameter()
    
    def reset_parameter(self):
        self.proj_dist = init_lecun_normal(self.proj_dist)
        nn.init.zeros_(self.proj_dist.bias)
        self.proj_sctors = init_lecun_normal(self.proj_sctors)
        nn.init.zeros_(self.proj_sctors.bias)

    def forward(self, msa, pair, xyz, state, sctors, mask_recycle=None):
        B, L = pair.shape[:2]
        state = self.norm_state(state)

        left = state.unsqueeze(2).expand(-1,-1,L,-1)
        right = state.unsqueeze(1).expand(-1,L,-1,-1)
        
        Ca_or_P = xyz[:,:,1].contiguous()

        dist = rbf(torch.cdist(Ca_or_P, Ca_or_P))
        if mask_recycle != None:
            dist = mask_recycle[...,None].float()*dist
        dist = torch.cat((dist, left, right), dim=-1)
        dist = self.proj_dist(dist)
        pair = dist + self.norm_pair(pair)

        sctors = self.proj_sctors(sctors.reshape(B,-1,2*ChemData().NTOTALDOFS))
        msa = sctors + self.norm_msa(msa)

        return msa, pair, state

recycling_factory = {
    "msa_pair": Recycling,
    "all": RecyclingAllFeatures
}
