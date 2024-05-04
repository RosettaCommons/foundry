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
from rf2aa.training.checkpoint import create_custom_forward
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

class MsaModule(nn.Module):
    def __init__(self,
                n_blocks,
                 subsampled_embedding,
                 outer_product,
                 msa_pair_weighted_averaging,
                 msa_transition,
                 triangle_multiplication_incoming,
                 triangle_multiplication_outgoing,
                 triangle_attention_starting,
                 triangle_attention_ending,
                 pair_transition,
                 ):
        super(MsaModule, self).__init__()
        self.n_blocks = n_blocks
        self.msa_subsampler = MsaSubsampleEmbedder(subsampled_embedding)
        self.outer_product = OuterProductMean(outer_product)
        self.msa_pair_weighted_averaging = MsaPairWeightedAverage(msa_pair_weighted_averaging)
        self.msa_transition = FeedForwardLayer(msa_transition)

        #TODO: check if row and col dropout are right
        self.drop_row = Dropout(broadcast_dim=1, p_drop=0.25)
        self.drop_col = Dropout(broadcast_dim=2, p_drop=0.25)        
        
        self.tri_mult_outgoing = TriangleMultiplication(triangle_multiplication_outgoing)
        self.tri_mult_incoming = TriangleMultiplication(triangle_multiplication_incoming)
        self.tri_attn_start = TriangleAttention(triangle_attention_starting)
        self.tri_attn_end = TriangleAttention(triangle_attention_ending)
        self.pair_transition = FeedForwardLayer(pair_transition)

    def forward(self, 
                f_dict, 
                pair_II,
                S_inputs
                ):
        msa_SI = f_dict["msa_SI"]
        msa_SI = self.msa_subsampler(msa_SI, S_inputs)
        for i in range(self.n_blocks):
            pair_II = pair_II + self.outer_product(msa_SI)
            msa_SI = msa_SI + self.drop_row(self.msa_pair_weighted_averaging(msa_SI, pair_II))
            msa_SI = msa_SI + self.msa_transition(msa_SI)

            pair_II = pair_II + self.drop_row(self.tri_mult_outgoing(pair_II))
            pair_II = pair_II + self.drop_row(self.tri_mult_incoming(pair_II))
            pair_II = pair_II + self.drop_row(self.tri_attn_start(pair_II))
            pair_II = pair_II + self.drop_row(self.tri_attn_end(pair_II))
            pair_II = pair_II + self.pair_transition(pair_II)
        return pair_II

class MsaSubsampleEmbedder(nn.Module):
    def __init__(self, params):
        super(MsaSubsampleEmbedder, self).__init__()
        self.num_sequences = params["num_sequences"]
        self.emb_msa = nn.Linear(params["msa_dim"], params["msa_channels"], bias=False)
        self.emb_S_inputs = nn.Linear(params["S_dim"], params["msa_channels"], bias=False)
    
    def forward(self, 
                msa_SI,
                S_inputs # (B, L, S_dim)
                ):
        B, S, I = msa_SI.shape[:3]
        num_samples = torch.min(torch.tensor([self.num_sequences, S]))
        weights = torch.ones(num_samples.item(), device=msa_SI.device)
        samples = torch.multinomial(weights, num_samples, replacement=False)
        msa_SI = msa_SI[:, samples]
        msa_SI = self.emb_msa(msa_SI)

        msa_SI = msa_SI + self.emb_S_inputs(S_inputs)
        return msa_SI


class MsaPairWeightedAverage(nn.Module):
    """ implements Algorithm 10 from AF3 paper"""
    def __init__(self, params):
        super(MsaPairWeightedAverage, self).__init__()
        self.weighted_average_channels = params["weighted_average_channels"]
        self.n_heads = params["n_heads"]
        self.msa_channels = params["msa_channels"]
        self.pair_channels = params["pair_channels"]
        self.norm_msa = nn.LayerNorm(self.msa_channels)
        self.to_v = nn.Linear(self.msa_channels, self.n_heads*self.weighted_average_channels, bias=False)
        self.norm_pair = nn.LayerNorm(self.pair_channels)
        self.to_bias = nn.Linear(self.msa_channels, self.n_heads, bias=False)
        self.to_gate = nn.Linear(self.msa_channels, self.n_heads, bias=False)
        self.to_out = nn.Linear(self.weighted_average_channels, self.msa_channels, bias=False)

    def forward(self, 
                msa_SI,
                pair_II
                ):
        B, S, I = msa_SI.shape[:3]
        msa_SI = self.norm_msa(msa_SI)
        v_SIH = self.to_v(msa_SI).reshape(B, S, I, self.n_heads, self.d_head)
        bias_IIH = self.to_bias(self.norm_pair(pair_II))
        gate_SIH = torch.sigmoid(self.to_gate(msa_SI))
        w_IIH = F.softmax(bias_IIH, dim=-2)
        weights = torch.einsum( "bijh,bsjhc->bsihc", w_IIH, v_SIH) 
        o_SIH = gate_SIH * weights
        msa_update_SI = self.to_out(o_SIH.reshape(B, S, I, -1))
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
class TemplateEmbedding(nn.Module):
    def __init__(self, params):
        super(TemplateEmbedding, self).__init__()
        self.template_channels = params["template_channels"]
        self.emb_pair = nn.Linear(params["pair_dim"], params["template_channels"], bias=False)
        self.norm_pair_before_pairformer = nn.LayerNorm(params["pair_dim"])
        self.norm_after_pairformer = nn.LayerNorm(params["template_channels"])
        # HACK: need the actual pairformer block
        self.pairformer = AF3_block(params["pair_dim"], params["template_channels"], params["pairformer_channels"], params["n_pairformer_layers"])
        # NOTE: this is not consistent with AF3 paper which outputs this tensor in the template_channel dimension
        self.agg_emb = nn.Linear(params["template_channels"], params["pair_dim"], bias=False)
    def forward(self,
                f_dict,
                pair_II,
                ):
        B, I = pair_II.shape[:2]
        template_frame_mask = f_dict["template_frame_mask"][None, :] * f_dict["template_frame_mask"][:, None]   
        template_pseudo_beta_mask = f_dict["template_pseudo_beta_mask"][None, :] * f_dict["template_pseudo_beta_mask"][:, None]
        template_feats = torch.cat([f_dict["template_distogram"], template_frame_mask, f_dict["template_unit_vector"], template_pseudo_beta_mask])
        template_feats = template_feats * (f_dict["asym_id"][None, :] == f_dict["asym_id"][:, None])
        T = template_feats.shape[1]
        u_II = torch.zeros(B, I, I, self.template_channels, device=pair_II.device)
        for i in range(T):
            v_II = self.emb_pair(self.norm_pair_before_pairformer(pair_II)) + template_feats[:, i]
            v_II = self.pairformer(v_II)
            u_II = u_II + self.norm_after_pairformer(v_II)
        
        u_II = u_II / T

        return self.agg_emb(F.relu(u_II))
    

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
