import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from functools import partial
import numpy as np

from rf2aa.debug import debug_nans
from rf2aa.model.layers.SE3_network import FullyConnectedSE3, FullyConnectedSE3_noR
from rf2aa.model.layers.structure_bias import structure_bias_factory
from rf2aa.model.layers.Attention_module import BiasedAxialAttention, FeedForwardLayer, MSAColAttention, \
    MSARowAttentionWithBias, TriangleMultiplication, MSAColGlobalAttention, \
    OldMSAColAttention, OldMSAColGlobalAttention, BiasedUntiedAxialAttention, TriangleAttention
from rf2aa.model.layers.outer_product import OuterProductMean # need to code this correctly
from rf2aa.training.checkpoint import create_custom_forward, activation_checkpointing
from rf2aa.util_module import Dropout, init_lecun_normal
from opt_einsum import contract as einsum


# MSA transformer
class AF3_full_block(nn.Module):
    """ 
    AF3_full_block:
       - MSA/Pair updates as in AF2
       - MSA then Pair
    """
    def __init__(self, global_config, block_params, **kwargs):
        super(AF3_full_block, self).__init__()
        d_msa, d_pair = (
            global_config.d_msa_full, 
            global_config.d_pair
        )

        # to do: optionally disable norm bias
        self.norm_pair_bias = nn.LayerNorm(d_pair, bias=False)
        self.drop_row = Dropout(broadcast_dim=1, p_drop=block_params.p_drop_row)
        self.drop_col = Dropout(broadcast_dim=2, p_drop=block_params.p_drop_pair)

        # to do: optionally disable norm bias
        self.msa_row_attn = MSARowAttentionWithBias(
            d_msa=d_msa, 
            d_pair=d_pair, 
            n_head=block_params.n_msa_head, 
            d_hidden=block_params.n_msa_channels,
            nseq_normalization=block_params.norm_msa_row,
            bias=False
        )
        self.msa_col_attn = MSAColGlobalAttention(
            d_msa=d_msa,
            n_head=block_params.n_msa_head,
            d_hidden=block_params.n_msa_channels,
            bias=False
        )
        self.msa_transition = FeedForwardLayer(d_msa, 4, p_drop=block_params.msa_transition_drop)

        # Pair update parameters
        self.outer_product = OuterProductMean(d_msa, d_pair, d_hidden=block_params.outer_product_channels, \
                                              p_drop=block_params.p_drop_outer_product)

        # to do: optionally disable norm bias
        self.tri_mul_outgoing = TriangleMultiplication(
            d_pair, d_hidden=block_params.n_pair_channels, outgoing=True, bias=False)
        self.tri_mul_incoming = TriangleMultiplication(
            d_pair, d_hidden=block_params.n_pair_channels, outgoing=False, bias=False)
        self.tri_attn_start = TriangleAttention(
            d_pair, d_hidden=block_params.n_pair_channels, start_node=True)
        self.tri_attn_end = TriangleAttention(
            d_pair, d_hidden=block_params.n_pair_channels, start_node=False)

        self.pair_transition = FeedForwardLayer(d_pair, 2) # HACK: hardcoded value for transition

    def _unpack_inputs(self, latent_feats):
        pair = latent_feats["pair"]
        msa = latent_feats["msa_full"]

        return msa, pair

    def _pack_outputs(self, msa, pair, latent_feats):
        latent_feats["msa_full"] = msa
        latent_feats["pair"] = pair
        return latent_feats

    def _1d_update(self, msa, pair):
        pair = self.norm_pair_bias(pair)

        msa = msa + self.drop_row(self.msa_row_attn(msa, pair))
        msa = msa + self.msa_col_attn(msa)
        msa = msa + self.msa_transition(msa)

        return msa

    def _2d_update(self, msa, pair):
        msa_bias = self.outer_product(msa)
        pair = pair + msa_bias
        pair = pair + self.drop_row(self.tri_mul_outgoing(pair)) 
        pair = pair + self.drop_row(self.tri_mul_incoming(pair)) 
        pair = pair + self.drop_row(self.tri_attn_start(pair)) 
        pair = pair + self.drop_row(self.tri_attn_end(pair)) 
        pair = pair + self.pair_transition(pair)
        return pair

    def forward(self, latent_feats, use_checkpoint, use_amp):
        msa, pair = self._unpack_inputs(latent_feats)
        if use_checkpoint:
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
                msa  = checkpoint.checkpoint(self._1d_update, msa, pair, use_reentrant=True)
                pair = checkpoint.checkpoint(self._2d_update, msa, pair, use_reentrant=True)
        else:
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
                msa  = self._1d_update(msa, pair)
                pair = self._2d_update(msa, pair)

        latent_feats = self._pack_outputs(msa, pair, latent_feats)
        return latent_feats


# Pair transformer
class AF3_block(nn.Module):
    """ 
    Attempt to replicate AF3 architecture from scratch.
    """
    def __init__(self, global_config, block_params, **kwargs):
        super(AF3_block, self).__init__()
        d_singleseq, d_pair = (
            global_config.d_msa, 
            global_config.d_pair
        )

        # to do: optionally disable norm bias
        self.norm_pair_bias = nn.LayerNorm(d_pair, bias=False)

        self.drop_row = Dropout(broadcast_dim=2, p_drop=block_params.p_drop_row)
        self.drop_col = Dropout(broadcast_dim=1, p_drop=block_params.p_drop_pair)

        # single sequence attn
        self.msa_row_attn = MSARowAttentionWithBias(
            d_msa=d_singleseq, 
            d_pair=d_pair, 
            n_head=block_params.n_msa_head, 
            d_hidden=block_params.n_msa_channels,
            nseq_normalization=block_params.norm_msa_row,
            bias=False
        )

        self.msa_transition = FeedForwardLayer(d_singleseq, 4, p_drop=block_params.msa_transition_drop)
        
        # to do: optionally disable norm bias
        self.tri_mul_outgoing = TriangleMultiplication(
            d_pair, d_hidden=block_params.n_pair_channels, outgoing=True, bias=False)
        self.tri_mul_incoming = TriangleMultiplication(
            d_pair, d_hidden=block_params.n_pair_channels, outgoing=False, bias=False)
        self.tri_attn_start = TriangleAttention(
            d_pair, d_hidden=block_params.n_pair_channels, start_node=True)
        self.tri_attn_end = TriangleAttention(
            d_pair, d_hidden=block_params.n_pair_channels, start_node=False)

        self.pair_transition = FeedForwardLayer(d_pair, 4) # HACK: hardcoded value for transition

        
    def _unpack_inputs(self, latent_feats):
        pair = latent_feats["pair"]
        singleseq = latent_feats["msa"]
        return singleseq, pair

    def _pack_outputs(self, singleseq, pair, latent_feats):
        latent_feats["msa"] = singleseq
        latent_feats["pair"] = pair
        return latent_feats

    def _1d_update(self, msa, pair):
        pair = self.norm_pair_bias(pair)
        msa = msa + self.drop_row(self.msa_row_attn(msa, pair)) # pair biased attn
        msa = msa + self.msa_transition(msa)

        return msa

    def _2d_update(self, msa, pair):
        pair = pair + self.drop_row(self.tri_mul_outgoing(pair)) 
        pair = pair + self.drop_row(self.tri_mul_incoming(pair)) 
        pair = pair + self.drop_row(self.tri_attn_start(pair)) 
        pair = pair + self.drop_col(self.tri_attn_end(pair)) 
        pair = pair + self.pair_transition(pair)
        return pair

    def forward(self, latent_feats, use_checkpoint, use_amp):
        singleseq, pair = self._unpack_inputs(latent_feats)
        drop_layer = 0
        if use_checkpoint:
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
                # 2d then 1d update
                pair = checkpoint.checkpoint(self._2d_update, singleseq, pair, use_reentrant=True)
                singleseq  = checkpoint.checkpoint(self._1d_update, singleseq, pair, use_reentrant=True)
        else:
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
                # 2d then 1d update
                pair = self._2d_update(singleseq, pair)
                singleseq = self._1d_update(singleseq, pair)

        latent_feats = self._pack_outputs(singleseq, pair, latent_feats)
        return latent_feats

class MsaSubsampleEmbedder(nn.Module):
    def __init__(self, num_sequences, dim_raw_msa, c_msa_embed, c_s_inputs):
        super(MsaSubsampleEmbedder, self).__init__()
        self.num_sequences = num_sequences
        self.emb_msa = nn.Linear(dim_raw_msa, c_msa_embed, bias=False)
        self.emb_S_inputs = nn.Linear(c_s_inputs, c_msa_embed, bias=False)
    
    @activation_checkpointing
    def forward(self, 
                msa_SI, # (S, I, 34) (32 tokens + has_deletion + deletion value)
                S_inputs # (L, S_dim)
                ):
        S, I = msa_SI.shape[:2]
        # choose sequences to sample
       # num_samples = torch.min(torch.tensor([self.num_sequences, S]))
        #weights = torch.ones(num_samples.item(), device=msa_SI.device)
        #samples = torch.multinomial(weights, num_samples, replacement=False)
        #msa_SI = torch.index_select(msa_SI, 0, samples)
        ##msa_SI = msa_SI[samples]

        # embed the subsampled MSA
        msa_SI = self.emb_msa(msa_SI)
        msa_SI = msa_SI + self.emb_S_inputs(S_inputs)
        return msa_SI


class MsaPairWeightedAverage(nn.Module):
    """ implements Algorithm 10 from AF3 paper"""
    def __init__(self, c_weighted_average, n_heads, c_msa_embed, c_z):
        super(MsaPairWeightedAverage, self).__init__()
        self.weighted_average_channels = c_weighted_average
        self.n_heads = n_heads
        self.msa_channels = c_msa_embed
        self.pair_channels = c_z
        self.norm_msa = nn.LayerNorm(self.msa_channels)
        self.to_v = nn.Linear(self.msa_channels, self.n_heads*self.weighted_average_channels, bias=False)
        self.norm_pair = nn.LayerNorm(self.pair_channels)
        self.to_bias = nn.Linear(self.pair_channels, self.n_heads, bias=False)
        self.to_gate = nn.Linear(self.msa_channels, self.n_heads, bias=False)
        self.to_out = nn.Linear(self.weighted_average_channels*self.n_heads, self.msa_channels, bias=False)

    @activation_checkpointing
    def forward(self, 
                msa_SI,
                pair_II
                ):
        S, I = msa_SI.shape[:2]
        
        # normalize inputs
        msa_SI = self.norm_msa(msa_SI)

        # construct values, bias and weights
        v_SIH = self.to_v(msa_SI).reshape(S, I, self.n_heads, self.weighted_average_channels)
        bias_IIH = self.to_bias(self.norm_pair(pair_II))
        w_IIH = F.softmax(bias_IIH, dim=-2)
        
        # construct gate
        gate_SIH = torch.sigmoid(self.to_gate(msa_SI))

        # compute weighted average
        weights = torch.einsum( "ijh,sjhc->sihc", w_IIH, v_SIH) 
        
        # apply gate
        o_SIH = gate_SIH[..., None] * weights

        # concatenate heads and project
        msa_update_SI = self.to_out(o_SIH.reshape(S, I, -1))
        return msa_update_SI

class BiasedSequenceAttention(nn.Module):
    def __init__(self, global_params, block_params):
        super(BiasedSequenceAttention, self).__init__()
        self.norm_state = nn.LayerNorm(global_params.d_state, bias=False)
        self.norm_pair = nn.LayerNorm(global_params.d_pair, bias=False)
        #
        self.to_q = nn.Linear(global_params.d_state, block_params.n_channels*block_params.n_heads, bias=False)
        self.to_k = nn.Linear(global_params.d_state, block_params.n_channels*block_params.n_heads, bias=False)
        self.to_v = nn.Linear(global_params.d_state, block_params.n_channels*block_params.n_heads, bias=False)
        self.to_b = nn.Linear(global_params.d_pair, block_params.n_heads, bias=False)
        self.to_g = nn.Linear(global_params.d_state,  block_params.n_channels*block_params.n_heads)
        self.to_out = nn.Linear( block_params.n_channels*block_params.n_heads, global_params.d_state, bias=False)

        self.scaling = 1/np.sqrt(block_params.n_channels)
        self.h = block_params.n_heads
        self.dim = block_params.n_channels
        self.transition = FeedForwardLayer(global_params.d_state, 4, p_drop=block_params.msa_transition_drop)

        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        
        # bias: normal distribution
        self.to_b = init_lecun_normal(self.to_b)

        # gating: zero weights, one biases (mostly open gate at the begining)
        nn.init.zeros_(self.to_g.weight)
        nn.init.ones_(self.to_g.bias)

        # to_out: right before residual connection: zero initialize -- to make it sure residual operation is same to the Identity at the begining
        nn.init.zeros_(self.to_out.weight)

    def forward(self, state, pair): # TODO: make this as tied-attention
        B, L = state.shape[:2]
        #
        state = self.norm_state(state)
        pair = self.norm_pair(pair)

        query = self.to_q(state).reshape(B, L, self.h, self.dim)
        key = self.scaling * self.to_k(state).reshape(B, L, self.h, self.dim)
        value = self.to_v(state).reshape(B, L, self.h, self.dim)
        bias = self.to_b(pair) # (B, L, L, h)
        gate = torch.sigmoid(self.to_g(state))

        attn = einsum('bqhd,bkhd->bqkh', query, key)
        attn = attn + bias
        attn = F.softmax(attn, dim=-2)

        out = einsum('bqkh,bkhd->bqhd', attn, value).reshape(B, L, -1)
        out = gate * out

        out = self.to_out(out)
        out = state + self.transition(out)

        return out

class BiasedSequenceAttention(nn.Module):
    def __init__(self, global_params, block_params):
        super(BiasedSequenceAttention, self).__init__()
        self.norm_state = nn.LayerNorm(global_params.d_state, bias=False)
        self.norm_pair = nn.LayerNorm(global_params.d_pair, bias=False)
        #
        self.to_q = nn.Linear(global_params.d_state, block_params.n_channels*block_params.n_heads, bias=False)
        self.to_k = nn.Linear(global_params.d_state, block_params.n_channels*block_params.n_heads, bias=False)
        self.to_v = nn.Linear(global_params.d_state, block_params.n_channels*block_params.n_heads, bias=False)
        self.to_b = nn.Linear(global_params.d_pair, block_params.n_heads, bias=False)
        self.to_g = nn.Linear(global_params.d_state,  block_params.n_channels*block_params.n_heads)
        self.to_out = nn.Linear( block_params.n_channels*block_params.n_heads, global_params.d_state, bias=False)

        self.scaling = 1/np.sqrt(block_params.n_channels)
        self.h = block_params.n_heads
        self.dim = block_params.n_channels
        self.transition = FeedForwardLayer(global_params.d_state, 4, p_drop=block_params.msa_transition_drop)

        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        
        # bias: normal distribution
        self.to_b = init_lecun_normal(self.to_b)

        # gating: zero weights, one biases (mostly open gate at the begining)
        nn.init.zeros_(self.to_g.weight)
        nn.init.ones_(self.to_g.bias)

        # to_out: right before residual connection: zero initialize -- to make it sure residual operation is same to the Identity at the begining
        nn.init.zeros_(self.to_out.weight)

    def forward(self, state, pair): # TODO: make this as tied-attention
        B, L = state.shape[:2]
        #
        state = self.norm_state(state)
        pair = self.norm_pair(pair)

        query = self.to_q(state).reshape(B, L, self.h, self.dim)
        key = self.scaling * self.to_k(state).reshape(B, L, self.h, self.dim)
        value = self.to_v(state).reshape(B, L, self.h, self.dim)
        bias = self.to_b(pair) # (B, L, L, h)
        gate = torch.sigmoid(self.to_g(state))

        attn = einsum('bqhd,bkhd->bqkh', query, key)
        attn = attn + bias
        attn = F.softmax(attn, dim=-2)

        out = einsum('bqkh,bkhd->bqhd', attn, value).reshape(B, L, -1)
        out = gate * out

        out = self.to_out(out)
        out = state + self.transition(out)

        return out
