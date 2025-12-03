import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from atomworks.ml.encoding_definitions import AF3SequenceEncoding
from einops import rearrange
from rfd3.model.layers.attention import (
    GatedCrossAttention,
    LocalAttentionPairBias,
)
from rfd3.model.layers.block_utils import (
    build_valid_mask,
    create_attention_indices,
    group_atoms,
    ungroup_atoms,
)
from rfd3.model.layers.layer_utils import (
    AdaLN,
    EmbeddingLayer,
    LinearBiasInit,
    RMSNorm,
    Transition,
    collapse,
    linearNoBias,
)
from rfd3.model.layers.pairformer_layers import PairformerBlock
from torch.nn.functional import one_hot

from foundry import DISABLE_CHECKPOINTING
from foundry.common import exists

logger = logging.getLogger(__name__)


# SwiGLU transition block with adaptive layernorm
class ConditionedTransitionBlock(nn.Module):
    def __init__(self, c_token, c_s, n=2):
        super().__init__()
        self.ada_ln = AdaLN(c_a=c_token, c_s=c_s)
        self.linear_1 = linearNoBias(c_token, c_token * n)
        self.linear_2 = linearNoBias(c_token, c_token * n)
        self.linear_output_project = nn.Sequential(
            LinearBiasInit(c_s, c_token, biasinit=-2.0),
            nn.Sigmoid(),
        )
        self.linear_3 = linearNoBias(c_token * n, c_token)

    def forward(
        self,
        Ai,  # [B, I, C_token]
        Si,  # [B, I, C_token]
    ):
        Ai = self.ada_ln(Ai, Si)
        # BUG: This is not the correct implementation of SwiGLU
        # Bi = torch.sigmoid(self.linear_1(Ai)) * self.linear_2(Ai)
        # FIX: This is the correct implementation of SwiGLU
        Bi = torch.nn.functional.silu(self.linear_1(Ai)) * self.linear_2(Ai)

        # Output projection (from adaLN-Zero)
        return self.linear_output_project(Si) * self.linear_3(Bi)


class PositionPairDistEmbedder(nn.Module):
    def __init__(self, c_atompair, embed_frame=True):
        super().__init__()
        self.embed_frame = embed_frame
        if embed_frame:
            self.process_d = linearNoBias(3, c_atompair)

        self.process_inverse_dist = linearNoBias(1, c_atompair)
        self.process_valid_mask = linearNoBias(1, c_atompair)

    def forward_af3(self, D_LL, V_LL):
        """Forward the same way reference positions are embeded in AF3"""

        P_LL = self.process_d(D_LL) * V_LL

        # Embed pairwise inverse squared distances, and the valid mask
        if self.training:
            P_LL = (
                P_LL
                + self.process_inverse_dist(
                    1 / (1 + torch.linalg.norm(D_LL, dim=-1, keepdim=True) ** 2)
                )
                * V_LL
            )
            P_LL = P_LL + self.process_valid_mask(V_LL.to(P_LL.dtype)) * V_LL
        else:
            P_LL[V_LL[..., 0]] += self.process_inverse_dist(
                1
                / (1 + torch.linalg.norm(D_LL[V_LL[..., 0]], dim=-1, keepdim=True) ** 2)
            )
            P_LL[V_LL[..., 0]] += self.process_valid_mask(
                V_LL[V_LL[..., 0]].to(P_LL.dtype)
            )
        return P_LL

    def forward(self, ref_pos, valid_mask):
        D_LL = ref_pos.unsqueeze(-2) - ref_pos.unsqueeze(-3)
        V_LL = valid_mask

        if self.embed_frame:
            # Embed pairwise distances
            return self.forward_af3(D_LL, V_LL)
        norm = torch.linalg.norm(D_LL, dim=-1, keepdim=True) ** 2
        norm = torch.clamp(norm, min=1e-6)
        inv_dist = 1 / (1 + norm)
        P_LL = self.process_inverse_dist(inv_dist) * V_LL
        P_LL = P_LL + self.process_valid_mask(V_LL.to(P_LL.dtype)) * V_LL
        return P_LL


class OneDFeatureEmbedder(nn.Module):
    """
    Embeds 1D features into a single vector.

    Args:
        features (dict): Dictionary of feature names and their number of channels.
        output_channels (int): Output dimension of the projected embedding.
    """

    def __init__(self, features, output_channels):
        super().__init__()
        self.features = {k: v for k, v in features.items() if exists(v)}
        total_embedding_input_features = sum(self.features.values())
        self.embedders = nn.ModuleDict(
            {
                feature: EmbeddingLayer(
                    n_channels, total_embedding_input_features, output_channels
                )
                for feature, n_channels in self.features.items()
            }
        )

    def forward(self, f, collapse_length):
        return sum(
            tuple(
                self.embedders[feature](collapse(f[feature].float(), collapse_length))
                for feature, n_channels in self.features.items()
                if exists(n_channels)
            )
        )


class SinusoidalDistEmbed(nn.Module):
    """
    Applies sinusoidal embedding to pairwise distances and projects to c_atompair.

    Args:
        c_atompair (int): Output dimension of the projected embedding (must be even).
    """

    def __init__(self, c_atompair, n_freqs=32):
        super().__init__()
        assert c_atompair % 2 == 0, "Output embedding dim must be even"

        self.n_freqs = (
            n_freqs  # Number of sin/cos pairs → total sinusoidal dim = 2 * n_freqs
        )
        self.c_atompair = c_atompair

        self.output_proj = linearNoBias(2 * n_freqs, c_atompair)
        self.process_valid_mask = linearNoBias(1, c_atompair)

    def forward(self, pos, valid_mask):
        """
        Args:
            pos: [L, 3] or [B, L, 3] ground truth atom positions
            valid_mask: [L, L, 1] or [B, L, L, 1] boolean mask
        Returns:
            P_LL: [L, L, c_atompair] or [B, L, L, c_atompair]
        """
        # Compute pairwise distances
        D_LL = pos.unsqueeze(-2) - pos.unsqueeze(-3)  # [L, L, 3] or [B, L, L, 3]
        dist_matrix = torch.linalg.norm(D_LL, dim=-1)  # [L, L] or [B, L, L]

        # Sinusoidal embedding
        half_dim = self.n_freqs
        freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, half_dim, dtype=torch.float32)
            / half_dim
        ).to(dist_matrix.device)  # [n_freqs]

        angles = dist_matrix.unsqueeze(-1) * freq  # [..., D/2]
        sin_embed = torch.sin(angles)
        cos_embed = torch.cos(angles)
        sincos_embed = torch.cat([sin_embed, cos_embed], dim=-1)  # [..., D]

        # Linear projection
        P_LL = self.output_proj(sincos_embed)  # [..., c_atompair]
        P_LL = P_LL * valid_mask

        # Add linear embedding of valid mask
        P_LL = P_LL + self.process_valid_mask(valid_mask.to(P_LL.dtype)) * valid_mask
        return P_LL


class LinearEmbedWithPool(nn.Module):
    def __init__(self, c_token):
        super().__init__()
        self.c_token = c_token
        self.linear = linearNoBias(3, c_token)

    def forward(self, R_L, tok_idx):
        B = R_L.shape[0]
        I = int(tok_idx.max().item()) + 1
        A_I_shape = (
            B,
            I,
            self.c_token,
        )
        Q_L = self.linear(R_L)
        A_I = (
            torch.zeros(A_I_shape, device=R_L.device, dtype=Q_L.dtype)
            .index_reduce(
                -2,
                tok_idx.long(),
                Q_L,
                "mean",
                include_self=False,
            )
            .clone()
        )
        return A_I


class SimpleRecycler(nn.Module):
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
            RMSNorm(c_z),
            linearNoBias(c_z, c_z),
        )
        self.process_sh = nn.Sequential(
            RMSNorm(c_s),
            linearNoBias(c_s, c_s),
        )
        self.pairformer_stack = nn.ModuleList(
            [
                PairformerBlock(c_s=c_s, c_z=c_z, **pairformer_block)
                for _ in range(n_pairformer_blocks)
            ]
        )
        # Templates and msa's removed:
        # self.template_embedder = TemplateEmbedder(c_z=c_z, **template_embedder)
        # self.msa_module = MSAModule(**msa_module)

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

        # Templates and msa's removed:
        # Z_II = Z_II + self.template_embedder(f, Z_II)
        # Z_II = self.msa_module(f, Z_II, S_inputs_I)

        S_I = S_init_I + self.process_sh(S_I)
        for block in self.pairformer_stack:
            S_I, Z_II = block(S_I, Z_II)
        return S_I, Z_II


class RelativePositionEncodingWithIndexRemoval(nn.Module):
    """
    Usual RPE but utilizes `is_motif_atom_3d_unindexed` to ensure within-chain position is spoofed.
    """

    def __init__(self, r_max, s_max, c_z):
        super().__init__()
        self.r_max = r_max
        self.s_max = s_max
        self.c_z = c_z

        self.num_tok_pos_bins = (
            2 * self.r_max + 2
        ) + 1  # original af3 + 1 for unknown index
        self.linear = linearNoBias(
            2 * self.num_tok_pos_bins + (2 * self.s_max + 2) + 1, c_z
        )

    def forward(self, f):
        b_samechain_II = f["asym_id"].unsqueeze(-1) == f["asym_id"].unsqueeze(-2)
        b_same_entity_II = f["entity_id"].unsqueeze(-1) == f["entity_id"].unsqueeze(-2)
        d_residue_II = torch.where(
            b_samechain_II,
            torch.clip(
                f["residue_index"].unsqueeze(-1)
                - f["residue_index"].unsqueeze(-2)
                + self.r_max,
                0,
                2 * self.r_max,
            ),
            2 * self.r_max + 1,
        )
        b_sameresidue_II = f["residue_index"].unsqueeze(-1) == f[
            "residue_index"
        ].unsqueeze(-2)
        tok_distance = (
            f["token_index"].unsqueeze(-1) - f["token_index"].unsqueeze(-2) + self.r_max
        )
        d_token_II = torch.where(
            b_samechain_II * b_sameresidue_II,
            torch.clip(
                tok_distance,
                0,
                2 * self.r_max,
            ),
            2 * self.r_max + 1,
        )

        # Chain distances are kept
        d_chain_II = torch.where(
            # NOTE: Implementing bugfix from the Protenix Technical report, where we use `same_entity` instead of `not same_chain` (as in the AF-3 pseudocode)
            # Reference: https://github.com/bytedance/Protenix/blob/main/Protenix_Technical_Report.pdf
            b_same_entity_II,
            torch.clip(
                f["sym_id"].unsqueeze(-1) - f["sym_id"].unsqueeze(-2) + self.s_max,
                0,
                2 * self.s_max,
            ),
            2 * self.s_max + 1,
        )
        A_relchain_II = one_hot(d_chain_II.long(), 2 * self.s_max + 2)

        #########################################################
        # Cancel out distances from unidexed motifs
        unindexing_pair_mask = f[
            "unindexing_pair_mask"
        ]  # [L, L] representing the parts which shouldnt' talk to one another

        # Special position case
        d_token_II[unindexing_pair_mask] = self.num_tok_pos_bins - 1
        d_residue_II[unindexing_pair_mask] = self.num_tok_pos_bins - 1

        A_relpos_II = one_hot(d_residue_II.long(), self.num_tok_pos_bins)
        A_reltoken_II = one_hot(d_token_II, self.num_tok_pos_bins)
        #########################################################

        return self.linear(
            torch.cat(
                [
                    A_relpos_II,
                    A_reltoken_II,
                    b_same_entity_II.unsqueeze(-1),
                    A_relchain_II,
                ],
                dim=-1,
            ).to(torch.float)
        )


class VirtualPredictor(nn.Module):
    def __init__(self, c_atom):
        super(VirtualPredictor, self).__init__()
        self.process_atom_embeddings = nn.Sequential(
            RMSNorm((c_atom,)), linearNoBias(c_atom, 1)
        )

    def forward(self, Q_L):
        return self.process_atom_embeddings(Q_L)


class SequenceHead(nn.Module):
    def __init__(self, c_token):
        super(SequenceHead, self).__init__()

        # Distogram feature extraction
        self.dist_fc1 = nn.Linear(196, 128)
        self.dist_relu = nn.ReLU()
        self.dist_fc2 = nn.Linear(128, 64)

        # Embedding feature extraction
        self.embed_fc1 = nn.Linear(c_token, 128)
        self.embed_relu = nn.ReLU()
        self.embed_fc2 = nn.Linear(128, 64)

        # Fusion layer
        self.fusion_fc = nn.Linear(128, 32)

        # Sequence encoding
        self.sequence_encoding_ = AF3SequenceEncoding()

    def forward(self, A_I, Q_L, X_L, f):
        B, L, _ = X_L.shape
        max_res_id = f["atom_to_token_map"].max().item() + 1

        # Detach tensors to avoid gradients through main module
        # X_L = X_L.detach()
        # A_I = A_I.detach()
        # Q_L = Q_L.detach()

        # Compute distograms
        residue_distogram = torch.zeros(B, max_res_id, 14, 14, device=X_L.device)
        for i in range(max_res_id):
            residue_mask = f["atom_to_token_map"] == i
            if residue_mask.sum() == 14:
                coords = X_L[:, residue_mask]  # (B, 14, 3)
                residue_distogram[:, i] = torch.cdist(coords, coords)

        # Flatten distogram
        dist_features = residue_distogram.view(B, max_res_id, 196)

        # Pass through separate MLPs
        dist_out = self.dist_fc1(dist_features)
        dist_out = self.dist_relu(dist_out)
        dist_out = self.dist_fc2(dist_out)

        embed_out = self.embed_fc1(A_I)
        embed_out = self.embed_relu(embed_out)
        embed_out = self.embed_fc2(embed_out)

        # Fusion via concatenation
        fused = torch.cat([dist_out, embed_out], dim=-1)
        Seq_I = self.fusion_fc(fused)

        indices = self.decode(Seq_I)

        return Seq_I, indices

    def decode(self, Seq_I):
        indices = Seq_I.argmax(dim=-1)  # [B, L]
        return indices


class LinearSequenceHead(nn.Module):
    def __init__(self, c_token):
        super().__init__()
        n_tok_all = 32
        disallowed_idxs = AF3SequenceEncoding().encode(["UNK", "X", "DX", "<G>"])
        mask = torch.ones(n_tok_all, dtype=torch.bool)
        mask[disallowed_idxs] = False
        self.register_buffer("valid_out_mask", mask)
        self.linear = nn.Linear(c_token, n_tok_all)

    def forward(self, A_I, **_):
        logits = self.linear(A_I)
        indices = self.decode(logits)
        return logits, indices

    def decode(self, logits):
        # logits: [D, L, 28]
        # indices: [D, L] in [0,32-1]
        D, I, _ = logits.shape
        probs = F.softmax(logits, dim=-1)
        probs = probs * self.valid_out_mask[None, None, :].to(probs.device)
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
        indices = probs.argmax(axis=-1)
        return indices


class Upcast(nn.Module):
    def __init__(
        self, c_token, c_atom, method="broadcast", cross_attention_block=None, n_split=6
    ):
        super().__init__()
        self.method = method
        self.n_split = n_split
        if self.method == "broadcast":
            self.project = nn.Sequential(
                RMSNorm((c_token,)), linearNoBias(c_token, c_atom)
            )
        elif self.method == "cross_attention":
            self.gca = GatedCrossAttention(
                c_query=c_atom, c_kv=c_token // self.n_split, **cross_attention_block
            )
        else:
            raise ValueError(f"Unknown upcast method: {self.method}")

    def forward_(self, Q_IA, A_I, valid_mask=None):
        if self.method == "broadcast":
            Q_IA = Q_IA + self.project(A_I)[..., None, :]
        elif self.method == "cross_attention":
            assert exists(A_I) and exists(valid_mask)
            # Split Tokens
            A_I = rearrange(A_I, "b n (s c) -> b n s c", s=self.n_split)
            n_tokens, n_atom_per_tok = Q_IA.shape[1], Q_IA.shape[2]

            # Attention mask: ..., n_atom_per_tok, n_split
            attn_mask = torch.full(
                (n_tokens, 1, n_atom_per_tok), True, device=Q_IA.device
            )
            attn_mask[~valid_mask.view_as(attn_mask)] = False

            attn_mask = torch.ones(
                (n_tokens, n_atom_per_tok, self.n_split), device=A_I.device, dtype=bool
            )
            attn_mask[~valid_mask, :] = False

            Q_IA = Q_IA + self.gca(q=Q_IA, kv=A_I, attn_mask=attn_mask)
        return Q_IA

    def forward(self, Q_L, A_I, tok_idx):
        valid_mask = build_valid_mask(tok_idx)
        Q_IA = ungroup_atoms(Q_L, valid_mask)
        Q_IA = self.forward_(Q_IA, A_I, valid_mask)
        Q_L = group_atoms(Q_IA, valid_mask)
        return Q_L


class Downcast(nn.Module):
    """Downcast modules for when atoms are already reshaped from N_atoms -> (N_tokens, 14)"""

    def __init__(
        self, c_atom, c_token, c_s=None, method="mean", cross_attention_block=None
    ):
        super().__init__()
        self.method = method
        self.c_token = c_token
        self.c_atom = c_atom
        if c_s is not None:
            self.process_s = nn.Sequential(
                RMSNorm((c_s,)),
                linearNoBias(c_s, c_token),
            )
        else:
            self.process_s = None

        if self.method == "mean":
            self.project = linearNoBias(c_atom, c_token)
        elif self.method == "cross_attention":
            self.gca = GatedCrossAttention(
                c_query=c_token,
                c_kv=c_atom,
                **cross_attention_block,
            )
        else:
            raise ValueError(f"Unknown downcast method: {self.method}")

    def forward_(self, Q_IA, A_I, S_I=None, valid_mask=None):
        if self.method == "mean":
            A_I_update = self.project(Q_IA).sum(-2) / valid_mask.sum(-1, keepdim=True)
        elif self.method == "cross_attention":
            assert exists(A_I) and exists(valid_mask)
            # Attention mask: ..., 1, n_atom_per_tok (1 querying token to atoms in token)
            attn_mask = valid_mask[..., None, :]
            A_I_update = self.gca(
                q=A_I[..., None, :], kv=Q_IA, attn_mask=attn_mask
            ).squeeze(-2)

        A_I = A_I + A_I_update if exists(A_I) else A_I_update

        if self.process_s is not None:
            A_I = A_I + self.process_s(S_I)
        return A_I

    def forward(self, Q_L, A_I, S_I=None, tok_idx=None):
        valid_mask = build_valid_mask(tok_idx)
        if Q_L.ndim == 2:
            squeeze = True
            Q_L = Q_L.unsqueeze(0)
        else:
            squeeze = False

        A_I = A_I.unsqueeze(0) if exists(A_I) and A_I.ndim == 2 else A_I
        S_I = S_I.unsqueeze(0) if exists(S_I) and S_I.ndim == 2 else S_I

        Q_IA = ungroup_atoms(Q_L, valid_mask)

        A_I = self.forward_(Q_IA, A_I, S_I, valid_mask=valid_mask)

        if squeeze:
            A_I = A_I.squeeze(0)
        return A_I


######################################################################################
##########################     Local Atom Transformer       ##########################
######################################################################################


class LocalTokenTransformer(nn.Module):
    def __init__(
        self,
        c_token,
        c_tokenpair,
        c_s,
        n_block,
        diffusion_transformer_block,
        n_registers=None,
        n_local_tokens=8,
        n_keys=32,
    ):
        super().__init__()
        self.n_local_tokens = n_local_tokens
        self.n_keys = n_keys
        self.blocks = nn.ModuleList(
            [
                StructureLocalAtomTransformerBlock(
                    c_atom=c_token,
                    c_s=c_s,
                    c_atompair=c_tokenpair,
                    **diffusion_transformer_block,
                )
                for _ in range(n_block)
            ]
        )

    def forward(self, A_I, S_I, Z_II, f, X_L, full=False):
        indices = create_attention_indices(
            X_L=X_L,
            f=f,
            tok_idx=torch.arange(A_I.shape[1], device=A_I.device),
            n_attn_keys=self.n_keys,
            n_attn_seq_neighbours=self.n_local_tokens,
        )

        for i, block in enumerate(self.blocks):
            # Set checkpointing
            block.attention_pair_bias.use_checkpointing = not DISABLE_CHECKPOINTING
            # A_I: [B, L, C_token]
            # S_I: [B, L, C_s]
            # Z_II: [B, L, L, C_tokenpair]
            A_I = block(
                A_I,
                S_I,
                Z_II,
                indices=indices,
                full=full,  # (self.training and torch.is_grad_enabled()),  # Does not accelerate inference, but memory *does* scale better
            )

        return A_I


class LocalAtomTransformer(nn.Module):
    def __init__(self, c_atom, c_s, c_atompair, atom_transformer_block, n_blocks):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                StructureLocalAtomTransformerBlock(
                    c_atom=c_atom,
                    c_s=c_s,
                    c_atompair=c_atompair,
                    **atom_transformer_block,
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(self, Q_L, C_L, P_LL, **kwargs):
        for block in self.blocks:
            Q_L = block(Q_L, C_L, P_LL, **kwargs)
        return Q_L


class StructureLocalAtomTransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        c_atom,
        c_s,
        c_atompair,
        dropout,
        no_residual_connection_between_attention_and_transition,
        **transformer_block,
    ):
        super().__init__()
        assert not no_residual_connection_between_attention_and_transition
        self.c_s = c_s
        self.dropout = nn.Dropout(dropout)
        self.attention_pair_bias = LocalAttentionPairBias(
            c_a=c_atom, c_s=c_s, c_pair=c_atompair, **transformer_block
        )
        if exists(c_s):
            self.transition_block = ConditionedTransitionBlock(c_token=c_atom, c_s=c_s)
        else:
            self.transition_block = Transition(c=c_atom, n=4)

    def forward(
        self,
        Q_L,  # [..., I, C_token]
        C_L,  # [..., I, C_s]
        P_LL,  # [..., I, I, C_tokenpair]
        f=None,
        chunked_pairwise_embedder=None,
        initializer_outputs=None,
        **kwargs,
    ):
        Q_L = Q_L + self.dropout(
            self.attention_pair_bias(
                Q_L,
                C_L,
                P_LL,
                f=f,
                chunked_pairwise_embedder=chunked_pairwise_embedder,
                initializer_outputs=initializer_outputs,
                **kwargs,
            )
        )
        if exists(C_L):
            Q_L = Q_L + self.transition_block(Q_L, C_L)
        else:
            Q_L = Q_L + self.transition_block(Q_L)
        return Q_L


class CompactStreamingDecoder(nn.Module):
    def __init__(
        self,
        c_atom,
        c_atompair,
        c_token,
        c_s,
        c_tokenpair,
        atom_transformer_block,
        upcast,
        downcast,
        n_blocks,
        diffusion_transformer_block=False,
    ):
        super().__init__()
        self.n_blocks = n_blocks

        self.upcast = nn.ModuleList(
            [Upcast(c_atom=c_atom, c_token=c_token, **upcast) for _ in range(n_blocks)]
        )
        self.atom_transformer = nn.ModuleList(
            [
                StructureLocalAtomTransformerBlock(
                    c_atom=c_atom,
                    c_s=c_atom,
                    c_atompair=c_atompair,
                    **atom_transformer_block,
                )
                for _ in range(n_blocks)
            ]
        )
        self.downcast = Downcast(c_atom=c_atom, c_token=c_token, c_s=c_s, **downcast)

    def forward(
        self,
        A_I,
        S_I,
        Z_II,
        Q_L,
        C_L,
        P_LL,
        tok_idx,
        indices,
        f=None,
        chunked_pairwise_embedder=None,
        initializer_outputs=None,
    ):
        for i in range(self.n_blocks):
            Q_L = self.upcast[i](Q_L, A_I, tok_idx=tok_idx)
            Q_L = self.atom_transformer[i](
                Q_L,
                C_L,
                P_LL,
                indices=indices,
                f=f,
                chunked_pairwise_embedder=chunked_pairwise_embedder,
                initializer_outputs=initializer_outputs,
            )

        # Downcast to sequence
        A_I = self.downcast(Q_L.detach(), A_I.detach(), S_I.detach(), tok_idx=tok_idx)

        o = {}
        return A_I, Q_L, o
