import logging

import torch
import torch.nn as nn
from rf3.model.layers.af3_diffusion_transformer import (
    AtomAttentionEncoderDiffusion,
    AtomTransformer,
    DiffusionTransformer,
)
from rf3.model.layers.layer_utils import Transition, linearNoBias
from rf3.model.layers.pairformer_layers import (
    MSAModule,
    PairformerBlock,
    RelativePositionEncoding,
    RF3TemplateEmbedder,
)

from foundry.model.layers.blocks import FourierEmbedding
from foundry.training.checkpoint import activation_checkpointing

logger = logging.getLogger(__name__)

"""
Glossary:
    I: # tokens (coarse representation)
    L: # atoms   (fine representation)
    M: # msa
    T: # templates
    D: # diffusion structure batch dim
"""


class AtomAttentionDecoder(nn.Module):
    def __init__(self, c_token, c_atom, c_atompair, atom_transformer):
        super().__init__()
        self.atom_transformer = AtomTransformer(
            c_atom=c_atom, c_atompair=c_atompair, **atom_transformer
        )
        self.linear_1 = linearNoBias(c_token, c_atom)
        self.to_r_update = nn.Sequential(
            nn.LayerNorm((c_atom,)), linearNoBias(c_atom, 3)
        )

    def forward(
        self,
        f,
        Ai,  # [L, C_token]
        Ql_skip,  # [L, C_atom]
        Cl_skip,  # [L, C_atom]
        Plm_skip,  # [L, L, C_atompair]
    ):
        tok_idx = f["atom_to_token_map"]

        @activation_checkpointing
        def atom_decoder(Ai, Ql_skip, Cl_skip, Plm_skip, tok_idx):
            # Broadcast per-token activiations to per-atom activations and add the skip connection
            Ql = self.linear_1(Ai[..., tok_idx, :]) + Ql_skip

            # Cross attention transformer.
            Ql = self.atom_transformer(Ql, Cl_skip, Plm_skip)

            # Map to positions update
            Rl_update = self.to_r_update(Ql)

            return Rl_update

        return atom_decoder(Ai, Ql_skip, Cl_skip, Plm_skip, tok_idx)


class DiffusionModule(nn.Module):
    def __init__(
        self,
        sigma_data,
        c_atom,
        c_atompair,
        c_token,
        c_s,
        c_z,
        f_pred,
        diffusion_conditioning,
        atom_attention_encoder,
        diffusion_transformer,
        atom_attention_decoder,
    ):
        super().__init__()
        self.sigma_data = sigma_data
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        self.c_s = c_s
        self.f_pred = f_pred

        self.diffusion_conditioning = DiffusionConditioning(
            sigma_data=sigma_data, c_s=c_s, c_z=c_z, **diffusion_conditioning
        )
        self.atom_attention_encoder = AtomAttentionEncoderDiffusion(
            c_token=c_token,
            c_s=c_s,
            c_atom=c_atom,
            c_atompair=c_atompair,
            **atom_attention_encoder,
        )
        self.process_s = nn.Sequential(
            nn.LayerNorm((c_s,)),
            linearNoBias(c_s, c_token),
        )
        self.diffusion_transformer = DiffusionTransformer(
            c_token=c_token, c_s=c_s, c_tokenpair=c_z, **diffusion_transformer
        )
        self.layer_norm_1 = nn.LayerNorm(c_token)
        self.atom_attention_decoder = AtomAttentionDecoder(
            c_token=c_token,
            c_atom=c_atom,
            c_atompair=c_atompair,
            **atom_attention_decoder,
        )

    def forward(
        self,
        X_noisy_L,  # [B, L, 3]
        t,  # [B] (0 is ground truth)
        f,  # Dict (Input feature dictionary)
        S_inputs_I,  # [B, I, C_S_input]
        S_trunk_I,  # [B, I, C_S_trunk]
        Z_trunk_II,  # [B, I, I, C_Z]
    ):
        # Conditioning
        S_I, Z_II = self.diffusion_conditioning(
            t, f, S_inputs_I.float(), S_trunk_I.float(), Z_trunk_II.float()
        )

        # Scale positions to dimensionless vectors with approximately unit variance
        if self.f_pred == "edm":
            R_noisy_L = X_noisy_L / torch.sqrt(
                t[..., None, None] ** 2 + self.sigma_data**2
            )
        elif self.f_pred == "unconditioned":
            R_noisy_L = torch.zeros_like(X_noisy_L)
        elif self.f_pred == "noise_pred":
            R_noisy_L = X_noisy_L
        else:
            raise Exception(f"{self.f_pred=} unrecognized")
        # Sequence-local Atom Attention and aggregation to coarse-grained tokens
        A_I, Q_skip_L, C_skip_L, P_skip_LL = self.atom_attention_encoder(
            f, R_noisy_L, S_trunk_I.float(), Z_II
        )
        # Full self-attention on token level

        A_I = A_I + self.process_s(S_I)
        A_I = self.diffusion_transformer(A_I, S_I, Z_II, Beta_II=None)
        A_I = self.layer_norm_1(A_I)

        # Broadcast token activations to atoms and run Sequence-local Atom Attention
        R_update_L = self.atom_attention_decoder(
            f, A_I.float(), Q_skip_L, C_skip_L, P_skip_LL
        )
        # Rescale updates to positions and combine with input positions
        if self.f_pred == "edm":
            X_out_L = (self.sigma_data**2 / (self.sigma_data**2 + t**2))[
                ..., None, None
            ] * X_noisy_L + (self.sigma_data * t / (self.sigma_data**2 + t**2) ** 0.5)[
                ..., None, None
            ] * R_update_L
        elif self.f_pred == "unconditioned":
            X_out_L = R_update_L
        elif self.f_pred == "noise_pred":
            X_out_L = X_noisy_L + R_update_L
        else:
            raise Exception(f"{self.f_pred=} unrecognized")

        return X_out_L


class DiffusionConditioning(nn.Module):
    def __init__(
        self, sigma_data, c_z, c_s, c_s_inputs, c_t_embed, relative_position_encoding
    ):
        super().__init__()
        self.sigma_data = sigma_data
        self.relative_position_encoding = RelativePositionEncoding(
            c_z=c_z, **relative_position_encoding
        )
        self.to_zii = nn.Sequential(
            nn.LayerNorm(
                c_z * 2
            ),  # Operates on concatenated (z_ij_trunk: [..., c_z]), RelativePositionalEncoding: [..., c_z])
            linearNoBias(c_z * 2, c_z),
        )
        self.transition_1 = nn.ModuleList(
            [
                Transition(c=c_z, n=2),
                Transition(c=c_z, n=2),
            ]
        )
        self.to_si = nn.Sequential(
            nn.LayerNorm(c_s + c_s_inputs), linearNoBias(c_s + c_s_inputs, c_s)
        )
        c_t_embed = 256
        self.fourier_embedding = FourierEmbedding(c_t_embed)
        self.process_n = nn.Sequential(
            nn.LayerNorm(c_t_embed), linearNoBias(c_t_embed, c_s)
        )
        self.transition_2 = nn.ModuleList(
            [
                Transition(c=c_s, n=2),
                Transition(c=c_s, n=2),
            ]
        )

    def forward(self, t, f, S_inputs_I, S_trunk_I, Z_trunk_II):
        # Pair conditioning
        Z_II = torch.cat([Z_trunk_II, self.relative_position_encoding(f)], dim=-1)

        @activation_checkpointing
        def _run_conditioning(Z_II, S_trunk_I, S_inputs_I):
            Z_II = self.to_zii(Z_II)
            for b in range(2):
                Z_II = Z_II + self.transition_1[b](Z_II)

            # Single conditioning
            S_I = torch.cat([S_trunk_I, S_inputs_I], dim=-1)
            S_I = self.to_si(S_I)
            N_D = self.fourier_embedding(1 / 4 * torch.log(t / self.sigma_data))
            S_I = self.process_n(N_D).unsqueeze(-2) + S_I
            for b in range(2):
                S_I = S_I + self.transition_2[b](S_I)

            return S_I, Z_II

        return _run_conditioning(Z_II, S_trunk_I, S_inputs_I)


class DistogramHead(nn.Module):
    def __init__(
        self,
        c_z,
        bins,
    ):
        super().__init__()
        self.predictor = nn.Linear(c_z, bins)
        self.reset_parameters()

    def reset_parameters(self):
        # initialize linear layer for final logit prediction
        nn.init.zeros_(self.predictor.weight)
        nn.init.zeros_(self.predictor.bias)

    def forward(
        self,
        Z_II,
    ):
        return self.predictor(
            Z_II + Z_II.transpose(-2, -3)  # symmetrize pair features
        )


class Recycler(nn.Module):
    def __init__(
        self,
        c_s,
        c_z,
        template_embedder,
        msa_module,
        n_pairformer_blocks,
        pairformer_block,
    ):
        super().__init__()
        self.c_z = c_z
        self.process_zh = nn.Sequential(
            nn.LayerNorm(c_z),
            linearNoBias(c_z, c_z),
        )
        self.template_embedder = RF3TemplateEmbedder(c_z=c_z, **template_embedder)
        self.msa_module = MSAModule(**msa_module)
        self.process_sh = nn.Sequential(
            nn.LayerNorm(c_s),
            linearNoBias(c_s, c_s),
        )
        self.pairformer_stack = nn.ModuleList(
            [
                PairformerBlock(c_s=c_s, c_z=c_z, **pairformer_block)
                for _ in range(n_pairformer_blocks)
            ]
        )

    def forward(
        self,
        f,
        S_inputs_I,
        S_init_I,
        Z_init_II,
        S_I,
        Z_II,
    ):
        Z_II = Z_init_II + self.process_zh(Z_II)
        Z_II = Z_II + self.template_embedder(f, Z_II)
        # NOTE: Implementing bugfix from the Protenix Technical report, where residual-connecting the MSA module is redundant
        # Reference: https://github.com/bytedance/Protenix/blob/main/Protenix_Technical_Report.pdf
        Z_II = self.msa_module(f, Z_II, S_inputs_I)
        S_I = S_init_I + self.process_sh(S_I)
        for block in self.pairformer_stack:
            S_I, Z_II = block(S_I, Z_II)
        return S_I, Z_II
