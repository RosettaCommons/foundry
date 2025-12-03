import torch
import torch.nn as nn
import torch.nn.functional as F

from foundry.training.checkpoint import activation_checkpointing


class MSASubsampleEmbedder(nn.Module):
    def __init__(self, num_sequences, dim_raw_msa, c_msa_embed, c_s_inputs):
        super(MSASubsampleEmbedder, self).__init__()
        self.num_sequences = num_sequences
        self.emb_msa = nn.Linear(dim_raw_msa, c_msa_embed, bias=False)
        self.emb_S_inputs = nn.Linear(c_s_inputs, c_msa_embed, bias=False)

    @activation_checkpointing
    def forward(
        self,
        msa_SI,  # (S, I, 34) (32 tokens + has_deletion + deletion value)
        S_inputs,  # (L, S_dim)
    ):
        # Embed the subsampled MSA
        # (NOTE: We subsample in the data loader to avoid memory issues)
        msa_SI = self.emb_msa(msa_SI)
        msa_SI = msa_SI + self.emb_S_inputs(S_inputs)
        return msa_SI


class MSAPairWeightedAverage(nn.Module):
    """implements Algorithm 10 from AF3 paper"""

    def __init__(
        self,
        c_weighted_average,
        n_heads,
        c_msa_embed,
        c_z,
        separate_gate_for_every_channel,
    ):
        super(MSAPairWeightedAverage, self).__init__()
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
