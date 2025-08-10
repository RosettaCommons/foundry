import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract as einsum

from modelhub.model.layers.Attention_module import (
    FeedForwardLayer,
)
from modelhub.training.checkpoint import activation_checkpointing
from modelhub.util_module import init_lecun_normal


class MsaSubsampleEmbedder(nn.Module):
    def __init__(self, num_sequences, dim_raw_msa, c_msa_embed, c_s_inputs):
        super(MsaSubsampleEmbedder, self).__init__()
        self.num_sequences = num_sequences
        self.emb_msa = nn.Linear(dim_raw_msa, c_msa_embed, bias=False)
        self.emb_S_inputs = nn.Linear(c_s_inputs, c_msa_embed, bias=False)

    @activation_checkpointing
    def forward(
        self,
        msa_SI,  # (S, I, 34) (32 tokens + has_deletion + deletion value)
        S_inputs,  # (L, S_dim)
    ):
        S, I = msa_SI.shape[:2]
        # choose sequences to sample
        # num_samples = torch.min(torch.tensor([self.num_sequences, S]))
        # weights = torch.ones(num_samples.item(), device=msa_SI.device)
        # samples = torch.multinomial(weights, num_samples, replacement=False)
        # msa_SI = torch.index_select(msa_SI, 0, samples)
        ##msa_SI = msa_SI[samples]

        # embed the subsampled MSA
        msa_SI = self.emb_msa(msa_SI)
        msa_SI = msa_SI + self.emb_S_inputs(S_inputs)
        return msa_SI


class MsaPairWeightedAverage(nn.Module):
    """implements Algorithm 10 from AF3 paper"""

    def __init__(
        self,
        c_weighted_average,
        n_heads,
        c_msa_embed,
        c_z,
        separate_gate_for_every_channel,
    ):
        super(MsaPairWeightedAverage, self).__init__()
        self.weighted_average_channels = c_weighted_average
        self.n_heads = n_heads
        self.msa_channels = c_msa_embed
        self.pair_channels = c_z
        self.norm_msa = nn.LayerNorm(self.msa_channels)
        self.to_v = nn.Linear(
            self.msa_channels, self.n_heads * self.weighted_average_channels, bias=False
        )
        self.norm_pair = nn.LayerNorm(self.pair_channels)
        self.to_bias = nn.Linear(self.pair_channels, self.n_heads, bias=False)

        self.separate_gate_for_every_channel = separate_gate_for_every_channel
        if self.separate_gate_for_every_channel:
            self.to_gate = nn.Linear(
                self.msa_channels,
                self.weighted_average_channels * self.n_heads,
                bias=False,
            )
        else:
            self.to_gate = nn.Linear(self.msa_channels, self.n_heads, bias=False)

        self.to_out = nn.Linear(
            self.weighted_average_channels * self.n_heads, self.msa_channels, bias=False
        )

    @activation_checkpointing
    def forward(self, msa_SI, pair_II):
        S, I = msa_SI.shape[:2]

        # normalize inputs
        msa_SI = self.norm_msa(msa_SI)

        # construct values, bias and weights
        v_SIH = self.to_v(msa_SI).reshape(
            S, I, self.n_heads, self.weighted_average_channels
        )
        bias_IIH = self.to_bias(self.norm_pair(pair_II))
        w_IIH = F.softmax(bias_IIH, dim=-2)

        # construct gate
        gate_SIH = torch.sigmoid(self.to_gate(msa_SI))

        # compute weighted average & apply gate
        if self.separate_gate_for_every_channel:
            weights = torch.einsum("ijh,sjhc->sihc", w_IIH, v_SIH).reshape(S, I, -1)
            o_SIH = gate_SIH * weights
        else:
            weights = torch.einsum("ijh,sjhc->sihc", w_IIH, v_SIH)
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
        self.to_q = nn.Linear(
            global_params.d_state,
            block_params.n_channels * block_params.n_heads,
            bias=False,
        )
        self.to_k = nn.Linear(
            global_params.d_state,
            block_params.n_channels * block_params.n_heads,
            bias=False,
        )
        self.to_v = nn.Linear(
            global_params.d_state,
            block_params.n_channels * block_params.n_heads,
            bias=False,
        )
        self.to_b = nn.Linear(global_params.d_pair, block_params.n_heads, bias=False)
        self.to_g = nn.Linear(
            global_params.d_state, block_params.n_channels * block_params.n_heads
        )
        self.to_out = nn.Linear(
            block_params.n_channels * block_params.n_heads,
            global_params.d_state,
            bias=False,
        )

        self.scaling = 1 / np.sqrt(block_params.n_channels)
        self.h = block_params.n_heads
        self.dim = block_params.n_channels
        self.transition = FeedForwardLayer(
            global_params.d_state, 4, p_drop=block_params.msa_transition_drop
        )

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

    def forward(self, state, pair):  # TODO: make this as tied-attention
        B, L = state.shape[:2]
        #
        state = self.norm_state(state)
        pair = self.norm_pair(pair)

        query = self.to_q(state).reshape(B, L, self.h, self.dim)
        key = self.scaling * self.to_k(state).reshape(B, L, self.h, self.dim)
        value = self.to_v(state).reshape(B, L, self.h, self.dim)
        bias = self.to_b(pair)  # (B, L, L, h)
        gate = torch.sigmoid(self.to_g(state))

        attn = einsum("bqhd,bkhd->bqkh", query, key)
        attn = attn + bias
        attn = F.softmax(attn, dim=-2)

        out = einsum("bqkh,bkhd->bqhd", attn, value).reshape(B, L, -1)
        out = gate * out

        out = self.to_out(out)
        out = state + self.transition(out)

        return out
