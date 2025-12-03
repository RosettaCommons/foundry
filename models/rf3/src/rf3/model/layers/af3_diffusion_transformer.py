import numpy as np
import torch
import torch.nn as nn
from rf3.loss.loss import calc_chiral_grads_flat_impl
from rf3.model.layers.layer_utils import (
    AdaLN,
    LinearBiasInit,
    MultiDimLinear,
    collapse,
    linearNoBias,
)
from rf3.model.layers.mlff import ConformerEmbeddingWeightedAverage

from foundry.training.checkpoint import activation_checkpointing
from foundry.utils.torch import device_of


class AtomAttentionEncoderDiffusion(nn.Module):
    def __init__(
        self,
        c_atom,
        c_atompair,
        c_token,
        c_tokenpair,
        c_s,
        atom_1d_features,
        c_atom_1d_features,
        atom_transformer,
        broadcast_trunk_feats_on_1dim_old,
        use_chiral_features,
        no_grad_on_chiral_center,
        use_inv_dist_squared: bool = False,
        use_atom_level_embedding: bool = False,
        atom_level_embedding_dim: int = 384,
    ):
        super().__init__()
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        self.c_tokenpair = c_tokenpair
        self.c_s = c_s
        self.atom_1d_features = atom_1d_features
        self.broadcast_trunk_feats_on_1dim_old = broadcast_trunk_feats_on_1dim_old
        self.use_chiral_features = use_chiral_features
        self.no_grad_on_chiral_center = no_grad_on_chiral_center
        self.use_atom_level_embedding = use_atom_level_embedding
        self.atom_level_embedding_dim = atom_level_embedding_dim

        self.process_input_features = linearNoBias(c_atom_1d_features, c_atom)

        self.process_d = linearNoBias(3, c_atompair)  # x,y,z

        self.process_inverse_dist = linearNoBias(1, c_atompair)
        self.process_valid_mask = linearNoBias(1, c_atompair)

        self.process_s_trunk = nn.Sequential(
            nn.LayerNorm(c_s), linearNoBias(c_s, c_atom)
        )
        self.process_z = nn.Sequential(
            nn.LayerNorm(c_tokenpair), linearNoBias(c_tokenpair, c_atompair)
        )
        self.process_r = linearNoBias(3, c_atom)
        if self.use_chiral_features:
            self.process_ch = linearNoBias(3, c_atom)

        self.process_single_l = nn.Sequential(
            nn.ReLU(), linearNoBias(c_atom, c_atompair)
        )
        self.process_single_m = nn.Sequential(
            nn.ReLU(), linearNoBias(c_atom, c_atompair)
        )

        self.pair_mlp = nn.Sequential(
            nn.ReLU(),
            linearNoBias(self.c_atompair, c_atompair),
            nn.ReLU(),
            linearNoBias(self.c_atompair, c_atompair),
            nn.ReLU(),
            linearNoBias(self.c_atompair, c_atompair),
        )

        self.process_q = nn.Sequential(
            linearNoBias(c_atom, c_token),
            nn.ReLU(),
        )

        self.atom_transformer = AtomTransformer(
            c_atom=c_atom, c_atompair=c_atompair, **atom_transformer
        )

        self.use_inv_dist_squared = use_inv_dist_squared

        if self.use_atom_level_embedding:
            self.process_atom_level_embedding = ConformerEmbeddingWeightedAverage(
                atom_level_embedding_dim=self.atom_level_embedding_dim,
                c_atompair=c_atompair,
                c_atom=c_atom,
            )

    def reset_parameters(self):
        super().reset_parameters()
        if self.use_chiral_features:
            nn.init.zeros_(self.process_ch.weight)

    def forward(
        self,
        f,  # Dict (Input feature dictionary)
        R_L,  # [D, L, 3]
        S_trunk_I,  # [B, I, C_S_trunk] [...,I,C_S_trunk]
        Z_II,  # [B, I, I, C_Z] [...,I,I,C_Z]
    ):
        assert R_L is not None

        tok_idx = f["atom_to_token_map"]
        L = len(tok_idx)
        I = tok_idx.max() + 1

        f["ref_atom_name_chars"] = f["ref_atom_name_chars"].reshape(L, -1)
        # Create the atom single conditioning: Embed per-atom meta data
        C_L = self.process_input_features(
            torch.cat(
                tuple(
                    collapse(f[feature_name], L)
                    for feature_name in self.atom_1d_features
                ),
                dim=-1,
            )
        )

        if self.use_atom_level_embedding:
            assert "atom_level_embedding" in f
            C_L = C_L + self.process_atom_level_embedding(f["atom_level_embedding"])

        # Embed offsets between atom reference positions
        D_LL = f["ref_pos"].unsqueeze(-2) - f["ref_pos"].unsqueeze(-3)
        V_LL = (
            f["ref_space_uid"].unsqueeze(-1) == f["ref_space_uid"].unsqueeze(-2)
        ).unsqueeze(-1)
        P_LL = self.process_d(D_LL) * V_LL

        @activation_checkpointing
        def embed_atom_feats(R_L, C_L, D_LL, V_LL, P_LL, tok_idx):
            # Embed pairwise inverse squared distances, and the valid mask
            if self.training:
                if self.use_inv_dist_squared:
                    P_LL = (
                        P_LL
                        + self.process_inverse_dist(
                            1 / (1 + torch.sum(D_LL * D_LL, dim=-1, keepdim=True))
                        )
                        * V_LL
                    )
                else:
                    P_LL = (
                        P_LL
                        + self.process_inverse_dist(
                            1 / (1 + torch.linalg.norm(D_LL, dim=-1, keepdim=True))
                        )
                        * V_LL
                    )
                P_LL = P_LL + self.process_valid_mask(V_LL.to(P_LL.dtype)) * V_LL
            else:
                if self.use_inv_dist_squared:
                    P_LL[V_LL[..., 0]] += self.process_inverse_dist(
                        1
                        / (
                            1
                            + torch.sum(
                                D_LL[V_LL[..., 0]] * D_LL[V_LL[..., 0]],
                                dim=-1,
                                keepdim=True,
                            )
                        )
                    )
                else:
                    P_LL[V_LL[..., 0]] += self.process_inverse_dist(
                        1
                        / (
                            1
                            + torch.linalg.norm(
                                D_LL[V_LL[..., 0]], dim=-1, keepdim=True
                            )
                        )
                    )
                P_LL[V_LL[..., 0]] += self.process_valid_mask(
                    V_LL[V_LL[..., 0]].to(P_LL.dtype)
                )

            # Initialise the atom single representation as the single conditioning.
            Q_L = C_L

            # If provided, add trunk embeddings and noisy positions.
            if R_L is not None:
                # Broadcast the single and pair embedding from the trunk.
                S_trunk_embed_L = self.process_s_trunk(S_trunk_I)[..., tok_idx, :]

                C_L = C_L + S_trunk_embed_L
                assert not (C_L == Q_L).all()
                if self.broadcast_trunk_feats_on_1dim_old:
                    P_LL = P_LL + self.process_z(Z_II)[..., tok_idx, tok_idx, :]
                else:
                    P_LL = (
                        P_LL + self.process_z(Z_II)[..., tok_idx, :, :][..., tok_idx, :]
                    )

                # Add the noisy positions.
                Q_L = self.process_r(R_L) + Q_L

                # Add chirality gradients
                if self.use_chiral_features:
                    with torch.autocast(
                        device_type=device_of(self).type, enabled=False
                    ):
                        # Do not pass grads through grad calc
                        R_L = calc_chiral_grads_flat_impl(
                            R_L.detach(),
                            f["chiral_centers"],
                            f["chiral_center_dihedral_angles"],
                            self.no_grad_on_chiral_center,
                        ).nan_to_num()
                    Q_L = self.process_ch(R_L) + Q_L

            # Add the combined single conditioning to the pair representation.
            P_LL = P_LL + (
                self.process_single_l(C_L).unsqueeze(-2)
                + self.process_single_m(C_L).unsqueeze(-3)
            )

            # Run a small MLP on the pair activations
            P_LL = P_LL + self.pair_mlp(P_LL)

            # Cross attention transformer.
            Q_L = self.atom_transformer(Q_L, C_L, P_LL)

            A_I_shape = Q_L.shape[:-2] + (
                I,
                self.c_token,
            )
            # Aggregate per-atom representation to per-token representation
            processed_Q_L = self.process_q(Q_L)  # [L, C_atom] -> [L, C_token]
            # Ensure dtype consistency for index_reduce
            processed_Q_L = processed_Q_L.to(Q_L.dtype)

            A_I = (
                torch.zeros(A_I_shape, device=Q_L.device, dtype=Q_L.dtype)
                .index_reduce(
                    -2,
                    f["atom_to_token_map"].long(),
                    processed_Q_L,
                    "mean",
                    include_self=False,
                )
                .clone()
            )

            return A_I, Q_L, C_L, P_LL

        return embed_atom_feats(R_L, C_L, D_LL, V_LL, P_LL, tok_idx)


class AtomTransformer(nn.Module):
    def __init__(
        self,
        c_atom,
        c_atompair,
        diffusion_transformer,
        n_queries,
        n_keys,
        l_max: int = None,  # HACK: Unused, kept for backwards compatibility with 9/21 checkpoint
    ):
        super().__init__()

        self.diffusion_transformer = DiffusionTransformer(
            c_token=c_atom, c_s=c_atom, c_tokenpair=c_atompair, **diffusion_transformer
        )

    def forward(
        self,
        Ql,  # [B, L, C_atom]
        Cl,  # [B, L, C_atom]
        Plm,  # [B, L, L, C_atompair]
    ):
        Beta_lm = True
        return self.diffusion_transformer(Ql, Cl, Plm, Beta_lm)


class DiffusionTransformer(nn.Module):
    def __init__(self, c_token, c_s, c_tokenpair, n_block, diffusion_transformer_block):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [
                DiffusionTransformerBlock(
                    c_token=c_token,
                    c_s=c_s,
                    c_tokenpair=c_tokenpair,
                    **diffusion_transformer_block,
                )
                for _ in range(n_block)
            ]
        )

    def forward(
        self,
        A_I,  # [..., I, C_token]
        S_I,  # [..., I, C_token]
        Z_II,  # [..., I, I, C_tokenpair]
        Beta_II,  # [I, I]
    ):
        for block in self.blocks:
            A_I = block(A_I, S_I, Z_II, Beta_II)
        return A_I


class DiffusionTransformerBlock(nn.Module):
    def __init__(
        self,
        c_token,
        c_s,
        c_tokenpair,
        n_head,
        no_residual_connection_between_attention_and_transition,
        kq_norm,
    ):
        super().__init__()
        self.attention_pair_bias = AttentionPairBiasDiffusion(
            c_a=c_token, c_s=c_s, c_pair=c_tokenpair, n_head=n_head, kq_norm=kq_norm
        )
        self.conditioned_transition_block = ConditionedTransitionBlock(
            c_token=c_token, c_s=c_s
        )
        self.no_residual_connection_between_attention_and_transition = (
            no_residual_connection_between_attention_and_transition
        )

    @activation_checkpointing
    def forward(
        self,
        A_I,  # [..., I, C_token]
        S_I,  # [..., I, C_s]
        Z_II,  # [..., I, I, C_tokenpair]
        Beta_II,  # [I, I]
    ):
        if self.no_residual_connection_between_attention_and_transition:
            B_I = self.attention_pair_bias(A_I, S_I, Z_II, Beta_II)
            A_I = A_I + B_I + self.conditioned_transition_block(A_I, S_I)
        else:
            A_I = A_I + self.attention_pair_bias(A_I, S_I, Z_II, Beta_II)
            A_I = A_I + self.conditioned_transition_block(A_I, S_I)

        return A_I


class ConditionedTransitionBlock(nn.Module):
    """SwiGLU transition block with adaptive layernorm"""

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


class AttentionPairBiasDiffusion(nn.Module):
    def __init__(self, c_a, c_s, c_pair, n_head, kq_norm):
        super().__init__()
        self.n_head = n_head
        self.c_a = c_a
        self.c_pair = c_pair
        self.c = c_a // n_head

        self.to_q = MultiDimLinear(c_a, (n_head, self.c), bias=False)
        self.to_k = MultiDimLinear(c_a, (n_head, self.c), bias=False)
        self.to_v = MultiDimLinear(c_a, (n_head, self.c), bias=False)
        self.to_b = linearNoBias(c_pair, n_head)
        self.to_g = nn.Sequential(
            MultiDimLinear(c_a, (n_head, self.c), bias=False),
            nn.Sigmoid(),
        )
        self.to_a = linearNoBias(c_a, c_a)
        self.linear_output_project = nn.Sequential(
            LinearBiasInit(c_s, c_a, biasinit=-2.0),
            nn.Sigmoid(),
        )
        self.ln_0 = nn.LayerNorm((c_pair,))
        self.ada_ln_1 = AdaLN(c_a=c_a, c_s=c_s)
        self.use_deepspeed_evo = False
        self.force_bfloat16 = True

        self.kq_norm = kq_norm
        if self.kq_norm:
            self.key_layer_norm = nn.LayerNorm((self.n_head * self.c,))
            self.query_layer_norm = nn.LayerNorm((self.n_head * self.c,))

    @activation_checkpointing
    def forward(
        self,
        A_I,  # [I, C_a]
        S_I,  # [I, C_a] | None
        Z_II,  # [I, I, C_z]
        Beta_II,  # [I, I]
    ):
        # Input projections
        assert S_I is not None
        if S_I is not None:
            A_I = self.ada_ln_1(A_I, S_I)

        if Beta_II is not None:
            # zero out layer norms for the key and query
            return self.atom_attention(A_I, S_I, Z_II)

        if self.use_deepspeed_evo or self.force_bfloat16:
            A_I = A_I.to(torch.bfloat16)
            assert len(A_I.shape) == 3, f"(Diffusion batch, I, C_a) but got {A_I.shape}"

        Q_IH = self.to_q(A_I)  # / np.sqrt(self.c)
        K_IH = self.to_k(A_I)
        V_IH = self.to_v(A_I)
        B_IIH = self.to_b(self.ln_0(Z_II))
        G_IH = self.to_g(A_I)

        if self.kq_norm:
            Q_IH = self.query_layer_norm(
                Q_IH.reshape(-1, self.n_head * self.c)
            ).reshape(Q_IH.shape)
            K_IH = self.key_layer_norm(K_IH.reshape(-1, self.n_head * self.c)).reshape(
                K_IH.shape
            )

        _, L = B_IIH.shape[:2]

        if not self.use_deepspeed_evo or L <= 24:
            # Attention
            Q_IH = Q_IH / np.sqrt(self.c)
            A_IIH = torch.softmax(
                torch.einsum("...ihd,...jhd->...ijh", Q_IH, K_IH) + B_IIH, dim=-2
            )  # softmax over j
            ## G_IH: [B, I, H, C]
            ## A_IIH: [B, I, I, H]
            ## V_IH: [B, I, H, C]
            A_I = torch.einsum("...ijh,...jhc->...ihc", A_IIH, V_IH)
            A_I = G_IH * A_I  # [B, I, H, C]
            A_I = A_I.flatten(start_dim=-2)  # [B, I, Ca]
        else:
            # DS4Sci_EvoformerAttention
            # Q, K, V: [Batch, N_seq, N_res, Head, Dim]
            # res_mask: [Batch, N_seq, 1, 1, N_res]
            # pair_bias: [Batch, 1, Head, N_res, N_res]
            from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention

            Q_IH = Q_IH[:, None]
            K_IH = K_IH[:, None]
            V_IH = V_IH[:, None]
            B_IIH = B_IIH.repeat(Q_IH.shape[0], 1, 1, 1)
            B_IIH = B_IIH[:, None]
            B_IIH = B_IIH.permute(0, 1, 4, 2, 3).to(torch.bfloat16)
            mask = torch.zeros(
                [Q_IH.shape[0], 1, 1, 1, B_IIH.shape[-1]],
                dtype=torch.bfloat16,
                device=B_IIH.device,
            )
            A_I = DS4Sci_EvoformerAttention(Q_IH, K_IH, V_IH, [mask, B_IIH])
            A_I = A_I * G_IH[:, None]
            A_I = A_I.view(A_I.shape[0], A_I.shape[2], -1)

        A_I = self.to_a(A_I)
        # Output projection (from adaLN-Zero)
        if S_I is not None:
            A_I = self.linear_output_project(S_I) * A_I

        return A_I

    def atom_attention(self, A_I, S_I, Z_II, qbatch=32, kbatch=128):
        assert qbatch % 2 == 0
        assert kbatch % 2 == 0

        if len(A_I.shape) == 2:
            A_I = A_I[None]
        Z_II = Z_II[None]
        D, L = A_I.shape[:2]
        Q_IH = self.to_q(A_I)
        K_IH = self.to_k(A_I)
        V_IH = self.to_v(A_I)
        B_IIH = self.to_b(self.ln_0(Z_II))
        G_IH = self.to_g(A_I)

        if self.kq_norm:
            Q_IH = self.query_layer_norm(
                Q_IH.reshape(-1, self.n_head * self.c)
            ).reshape(Q_IH.shape)
            K_IH = self.key_layer_norm(K_IH.reshape(-1, self.n_head * self.c)).reshape(
                K_IH.shape
            )

        nqbatch = (L + qbatch - 1) // qbatch
        Cs = torch.arange(nqbatch, device=A_I.device) * qbatch + qbatch // 2
        patchq = torch.arange(qbatch, device=A_I.device) - qbatch // 2
        patchk = torch.arange(kbatch, device=A_I.device) - kbatch // 2

        indicesQ = Cs[:, None] + patchq[None, :]
        maskQ = (indicesQ < 0) | (indicesQ > L - 1)
        indicesQ = torch.clamp(indicesQ, 0, L - 1)

        indicesK = Cs[:, None] + patchk[None, :]
        maskK = (indicesK < 0) | (indicesK > L - 1)
        indicesK = torch.clamp(indicesK, 0, L - 1)

        query_subset = Q_IH[:, indicesQ]
        key_subset = K_IH[:, indicesK]
        attn = torch.einsum("...ihd,...jhd->...ijh", query_subset, key_subset)
        attn = attn / (self.c**0.5)

        attn += B_IIH[:, indicesQ[:, :, None], indicesK[:, None, :]] - 1e9 * (
            maskQ[None, :, :, None, None] + maskK[None, :, None, :, None]
        )
        attn = torch.softmax(attn, dim=-2)

        value_subset = V_IH[:, indicesK]
        atom_features = torch.einsum("...ijh,...jhc->...ihc", attn, value_subset)
        atom_features = atom_features[:, ~maskQ]
        atom_features = (G_IH * atom_features).view(D, L, -1)
        atom_features = self.to_a(atom_features.view(D, L, -1))

        A_I = self.linear_output_project(S_I) * atom_features

        return A_I
