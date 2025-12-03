import torch
from rf3.data.ground_truth_template import (
    af3_noise_scale_to_noise_level,
)
from rf3.model.layers.af3_diffusion_transformer import AtomTransformer
from rf3.model.layers.attention import TriangleAttention, TriangleMultiplication
from rf3.model.layers.layer_utils import (
    MultiDimLinear,
    Transition,
    collapse,
    create_batch_dimension_if_not_present,
    linearNoBias,
)
from rf3.model.layers.mlff import ConformerEmbeddingWeightedAverage
from rf3.model.layers.outer_product import (
    OuterProductMean_AF3,
)
from rf3.model.RF3_blocks import MSAPairWeightedAverage, MSASubsampleEmbedder
from torch import nn
from torch.nn.functional import one_hot, relu

from foundry.model.layers.blocks import Dropout
from foundry.training.checkpoint import activation_checkpointing


class AtomAttentionEncoderPairformer(nn.Module):
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
        use_inv_dist_squared: bool = False,  # HACK: For 9/21 checkpoint, default to False (as this argument was not present in the checkpoint config)
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

        self.process_input_features = linearNoBias(c_atom_1d_features, c_atom)

        self.process_d = linearNoBias(3, c_atompair)
        self.process_inverse_dist = linearNoBias(1, c_atompair)
        self.process_valid_mask = linearNoBias(1, c_atompair)

        self.use_atom_level_embedding = use_atom_level_embedding

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
                atom_level_embedding_dim=atom_level_embedding_dim,
                c_atompair=c_atompair,
                c_atom=c_atom,
            )

    def forward(
        self,
        f,  # Dict (Input feature dictionary)
        R_L,  # [D, L, 3]
        S_trunk_I,  # [B, I, C_S_trunk] [...,I,C_S_trunk]
        Z_II,  # [B, I, I, C_Z] [...,I,I,C_Z]
    ):
        assert R_L is None
        assert S_trunk_I is None
        assert Z_II is None

        # ... get the number of atoms and tokens
        tok_idx = f["atom_to_token_map"]
        L = len(tok_idx)  # N_atom
        I = tok_idx.max() + 1  # N_token

        # ... flatten the last two dimensions of ref_atom_name_chars
        # (the letter dimension and the one-hot encoding of the unicode character dimension)
        f["ref_atom_name_chars"] = f["ref_atom_name_chars"].reshape(
            L, -1
        )  # [L, 4, 64] -> [L, 256], where L = N_atom

        # Atom single conditioning (C_L): Linearly embed concatenated per-atom features
        # (e.g., ref_pos, ref_charge, ref_mask, ref_element, ref_atom_name_chars)
        C_L = self.process_input_features(
            torch.cat(
                tuple(
                    collapse(f[feature_name], L)
                    for feature_name in self.atom_1d_features
                ),
                dim=-1,
            )
        )  # [L, C_atom]

        if self.use_atom_level_embedding:
            assert "atom_level_embedding" in f
            C_L = C_L + self.process_atom_level_embedding(f["atom_level_embedding"])

        # Now, we have the single conditioning (C_L) for each atom. We will:
        # 1. Use C_L to initialize the pair atom representation
        # 2. Pass C_L as a skip connection to the diffusion module

        # Embed offsets between atom reference positions
        # ref_pos is of shape [L, 3], so ref_pos.unsqueeze(-2) is of shape [L, 1, 3] and ref_pos.unsqueeze(-3) is of shape [1, L, 3]
        # We then take the outer difference between these two tensors to get a tensor of shape [L, L, 3] (via broadcasting both to shape [L, L, 3], and then taking the difference)
        D_LL = f["ref_pos"].unsqueeze(-2) - f["ref_pos"].unsqueeze(
            -3
        )  # [L, 1, 3] - [1, L, 3] -> [L, L, 3]

        # Create a mask indicating if two atoms are on the same chain AND the same residue (e.g., the same ref_space_uid)
        # (We add a singleton dimension to the mask to make it broadcastable with D_LL, which will be useful later)
        V_LL = (
            f["ref_space_uid"].unsqueeze(-1) == f["ref_space_uid"].unsqueeze(-2)
        ).unsqueeze(-1)  # [L, 1] == [1, L] -> [L, L, 1]

        @activation_checkpointing
        def embed_features(C_L, D_LL, V_LL):
            P_LL = self.process_d(D_LL) * V_LL  # [L, L, 3] -> [L, L, C_atompair]

            # Embed pairwise inverse squared distances, and the valid mask
            if self.use_inv_dist_squared:
                P_LL += (
                    self.process_inverse_dist(
                        1 / (1 + torch.sum(D_LL * D_LL, dim=-1, keepdim=True))
                    )
                    * V_LL
                )  # [L, L, 1] -> [L, L, C_atompair]
            else:
                P_LL = (
                    P_LL
                    + self.process_inverse_dist(
                        1 / (1 + torch.linalg.norm(D_LL, dim=-1, keepdim=True))
                    )
                    * V_LL
                )  # [L, L, 1] -> [L, L, C_atompair]

            P_LL = P_LL + self.process_valid_mask(V_LL.to(P_LL.dtype)) * V_LL

            # Initialise the atom single representation as the single conditioning.
            # NOTE: We create a new view on the tensor, so that the original tensor is not modified (unless we perform an in-place operation)
            Q_L = C_L

            # Add the combined single conditioning to the pair representation.
            # (With a residual connection)
            P_LL = P_LL + (
                self.process_single_l(C_L).unsqueeze(-2)
                + self.process_single_m(C_L).unsqueeze(-3)
            )  # [L, 1, C_atompair] + [1, L, C_atompair] -> [L, L, C_atompair]

            # Run a small MLP on the pair activations
            # (With a residual connection)
            P_LL = P_LL + self.pair_mlp(
                P_LL
            )  # [L, L, C_atompair] -> [L, L, C_atompair]

            # Cross attention transformer
            Q_L = self.atom_transformer(Q_L, C_L, P_LL)  # [L, C_atom]

            # ...get the desired shape of the per-token representation, which is [I, C_token]
            A_I_shape = Q_L.shape[:-2] + (
                I,
                self.c_token,
            )

            # Aggregate per-atom representation to per-token representation
            # (Set the per-token representation to be the mean activation of all atoms in the token)
            processed_Q_L = self.process_q(Q_L)  # [L, C_atom] -> [L, C_token]
            # Ensure dtype consistency for index_reduce
            processed_Q_L = processed_Q_L.to(Q_L.dtype)

            A_I = torch.zeros(
                A_I_shape, device=Q_L.device, dtype=Q_L.dtype
            ).index_reduce(
                -2,  # Operate on the second-to-last dimension (the atom dimension)
                f[
                    "atom_to_token_map"
                ].long(),  # [L], mapping from atom index to token index. Must be a torch.int64 or torch.int32 tensor.
                processed_Q_L,  # [L, C_atom] -> [L, C_token]
                "mean",
                include_self=False,  # Do not use the original values in A_I (all zeros) when aggregating
            )  # [L, C_atom] -> [I, C_token]

            return A_I, Q_L, C_L, P_LL

        return embed_features(C_L, D_LL, V_LL)


class AttentionPairBiasPairformerDeepspeed(nn.Module):
    def __init__(self, c_a, c_s, c_pair, n_head):
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
        # self.linear_output_project = nn.Sequential(
        # LinearBiasInit(c_s, c_a, biasinit=-2.),
        # nn.Sigmoid(),
        # )
        self.ln_0 = nn.LayerNorm((c_pair,))
        # self.ada_ln_1 = AdaLN(c_a=c_a, c_s=c_s)
        self.ln_1 = nn.LayerNorm((c_a,))
        self.use_deepspeed_evo = False
        self.force_bfloat16 = True

    def forward(
        self,
        A_I,  # [I, C_a]
        S_I,  # [I, C_a] | None
        Z_II,  # [I, I, C_z]
        Beta_II=None,  # [I, I]
    ):
        # Input projections
        assert S_I is None
        A_I = self.ln_1(A_I)

        if self.use_deepspeed_evo or self.force_bfloat16:
            A_I = A_I.to(torch.bfloat16)

        Q_IH = self.to_q(A_I)  # / np.sqrt(self.c)
        K_IH = self.to_k(A_I)
        V_IH = self.to_v(A_I)
        B_IIH = self.to_b(self.ln_0(Z_II)) + Beta_II[..., None]
        G_IH = self.to_g(A_I)

        B, L = B_IIH.shape[:2]

        if not self.use_deepspeed_evo or L <= 24:
            Q_IH = Q_IH / torch.sqrt(
                torch.tensor(self.c).to(Q_IH.device, torch.bfloat16)
            )
            # Attention
            A_IIH = torch.softmax(
                torch.einsum("...ihd,...jhd->...ijh", Q_IH, K_IH) + B_IIH, dim=-2
            )  # softmax over j
            ## G_IH: [I, H, C]
            ## A_IIH: [I, I, H]
            ## V_IH: [I, H, C]
            A_I = torch.einsum("...ijh,...jhc->...ihc", A_IIH, V_IH)
            A_I = G_IH * A_I  # [B, I, H, C]
            A_I = A_I.flatten(start_dim=-2)  # [B, I, Ca]
        else:
            # DS4Sci_EvoformerAttention
            # Q, K, V: [Batch, N_seq, N_res, Head, Dim]
            # res_mask: [Batch, N_seq, 1, 1, N_res]
            # pair_bias: [Batch, 1, Head, N_res, N_res]
            from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention

            assert Q_IH.shape[0] != 1, "this code assumes your structure is not batched"
            batch = 1
            n_res = Q_IH.shape[0]
            n_head = self.n_head
            c = self.c

            Q_IH = Q_IH[None, None]
            K_IH = K_IH[None, None]
            V_IH = V_IH[None, None]
            B_IIH = B_IIH.repeat(Q_IH.shape[0], 1, 1, 1)
            B_IIH = B_IIH[:, None]
            B_IIH = B_IIH.permute(0, 1, 4, 2, 3).to(torch.bfloat16)
            mask = torch.zeros(
                [Q_IH.shape[0], 1, 1, 1, B_IIH.shape[-1]],
                dtype=torch.bfloat16,
                device=B_IIH.device,
            )

            assert Q_IH.shape == (batch, 1, n_res, n_head, c)
            assert K_IH.shape == (batch, 1, n_res, n_head, c)
            assert V_IH.shape == (batch, 1, n_res, n_head, c)
            assert mask.shape == (batch, 1, 1, 1, n_res)
            assert B_IIH.shape == (batch, 1, n_head, n_res, n_res)

            A_I = DS4Sci_EvoformerAttention(Q_IH, K_IH, V_IH, [mask, B_IIH])

            assert A_I.shape == (batch, 1, n_res, n_head, c)
            A_I = A_I * G_IH[None, None]
            A_I = A_I.view(n_res, -1)

        A_I = self.to_a(A_I)

        return A_I


class PairformerBlock(nn.Module):
    """
    Attempt to replicate AF3 architecture from scratch.
    """

    def __init__(
        self,
        c_s,
        c_z,
        p_drop,
        triangle_multiplication,
        triangle_attention,
        attention_pair_bias,
        n_transition=4,
        **kwargs,  # Catch-all for backwards compatibility
    ):
        super().__init__()

        self.drop_row = Dropout(broadcast_dim=-2, p_drop=p_drop)
        self.drop_col = Dropout(broadcast_dim=-3, p_drop=p_drop)

        self.tri_mul_outgoing = TriangleMultiplication(
            d_pair=c_z,
            d_hidden=triangle_multiplication["d_hidden"],
            direction="outgoing",
            bias=True,
            use_cuequivariance=True,
        )
        self.tri_mul_incoming = TriangleMultiplication(
            d_pair=c_z,
            d_hidden=triangle_multiplication["d_hidden"],
            direction="incoming",
            bias=True,
            use_cuequivariance=True,
        )

        self.tri_attn_start = TriangleAttention(
            c_z,
            **triangle_attention,
            start_node=True,
            use_cuequivariance=True,
        )
        self.tri_attn_end = TriangleAttention(
            c_z,
            **triangle_attention,
            start_node=False,
            use_cuequivariance=True,
        )

        self.z_transition = Transition(c=c_z, n=n_transition)

        if c_s > 0:
            self.s_transition = Transition(c=c_s, n=n_transition)

            self.attention_pair_bias = AttentionPairBiasPairformerDeepspeed(
                c_a=c_s, c_s=0, c_pair=c_z, **attention_pair_bias
            )
        triangle_operations_expected_dim = 4  # B, L, L, C
        self.maybe_make_batched = create_batch_dimension_if_not_present(
            triangle_operations_expected_dim
        )

    @activation_checkpointing
    def forward(self, S_I, Z_II):
        Z_II = Z_II + self.drop_row(
            self.maybe_make_batched(self.tri_mul_outgoing)(Z_II)
        )
        Z_II = Z_II + self.drop_row(
            self.maybe_make_batched(self.tri_mul_incoming)(Z_II)
        )
        Z_II = Z_II + self.drop_row(self.maybe_make_batched(self.tri_attn_start)(Z_II))
        Z_II = Z_II + self.drop_col(self.maybe_make_batched(self.tri_attn_end)(Z_II))
        Z_II = Z_II + self.z_transition(Z_II)
        if S_I is not None:
            S_I = S_I + self.attention_pair_bias(
                S_I, None, Z_II, Beta_II=torch.tensor([0.0], device=Z_II.device)
            )
            S_I = S_I + self.s_transition(S_I)

        return S_I, Z_II


class FeatureInitializer(nn.Module):
    def __init__(
        self,
        c_s,
        c_z,
        c_atom,
        c_atompair,
        c_s_inputs,
        input_feature_embedder,
        relative_position_encoding,
    ):
        super().__init__()
        self.input_feature_embedder = InputFeatureEmbedder(
            c_atom=c_atom, c_atompair=c_atompair, **input_feature_embedder
        )
        self.to_s_init = linearNoBias(c_s_inputs, c_s)
        self.to_z_init_i = linearNoBias(c_s_inputs, c_z)
        self.to_z_init_j = linearNoBias(c_s_inputs, c_z)
        self.relative_position_encoding = RelativePositionEncoding(
            c_z=c_z, **relative_position_encoding
        )
        self.process_token_bonds = linearNoBias(1, c_z)

    def forward(
        self,
        f,
    ):
        S_inputs_I = self.input_feature_embedder(f)
        S_init_I = self.to_s_init(S_inputs_I)
        Z_init_II = self.to_z_init_i(S_inputs_I).unsqueeze(-3) + self.to_z_init_j(
            S_inputs_I
        ).unsqueeze(-2)
        Z_init_II = Z_init_II + self.relative_position_encoding(f)
        Z_init_II = Z_init_II + self.process_token_bonds(
            f["token_bonds"].unsqueeze(-1).to(torch.float)
        )
        return S_inputs_I, S_init_I, Z_init_II


class InputFeatureEmbedder(nn.Module):
    def __init__(self, features, c_atom, c_atompair, atom_attention_encoder):
        super().__init__()
        self.atom_attention_encoder = AtomAttentionEncoderPairformer(
            c_atom=c_atom, c_atompair=c_atompair, c_s=0, **atom_attention_encoder
        )
        self.features = features
        self.features_to_unsqueeze = ["deletion_mean"]

    def forward(
        self,
        f,
    ):
        A_I, _, _, _ = self.atom_attention_encoder(f, None, None, None)
        S_I = torch.cat(
            [A_I.squeeze(0)]
            + [
                f[feature].unsqueeze(-1)
                if feature in self.features_to_unsqueeze
                else f[feature]
                for feature in self.features
            ],
            dim=-1,
        )
        return S_I


class RelativePositionEncoding(nn.Module):
    def __init__(self, r_max, s_max, c_z):
        super().__init__()
        self.r_max = r_max
        self.s_max = s_max
        self.c_z = c_z
        self.linear = linearNoBias(
            2 * (2 * self.r_max + 2) + (2 * self.s_max + 2) + 1, c_z
        )

    def forward(self, f):
        b_samechain_II = f["asym_id"].unsqueeze(-1) == f["asym_id"].unsqueeze(-2)
        b_sameresidue_II = f["residue_index"].unsqueeze(-1) == f[
            "residue_index"
        ].unsqueeze(-2)
        b_same_entity_II = f["entity_id"].unsqueeze(-1) == f["entity_id"].unsqueeze(-2)

        # Handle cyclic chains
        cyclic_asym_ids = f.get("cyclic_asym_ids", [])
        if len(cyclic_asym_ids) > 0:
            offset = f["residue_index"].unsqueeze(-1) - f["residue_index"].unsqueeze(-2)

            for cyclic_asym_id in cyclic_asym_ids:
                len_cyclic_chain = (
                    f["residue_index"][f["asym_id"] == cyclic_asym_id].unique().shape[0]
                )
                cyclic_chain_mask = (f["asym_id"].unsqueeze(-1) == cyclic_asym_id) & (
                    f["asym_id"].unsqueeze(-2) == cyclic_asym_id
                )

                # cyclic offset
                if len_cyclic_chain > 0:
                    offset_plus = offset + len_cyclic_chain
                    offset_minus = offset - len_cyclic_chain
                    abs_offset = offset.abs()
                    abs_offset_plus = offset_plus.abs()
                    abs_offset_minus = offset_minus.abs()

                    choice_plus_or_minus = torch.where(
                        abs_offset_plus <= abs_offset_minus, offset_plus, offset_minus
                    )
                    c_offset = torch.where(
                        (abs_offset <= abs_offset_plus)
                        & (abs_offset <= abs_offset_minus),
                        offset,
                        choice_plus_or_minus,
                    )
                    offset = torch.where(cyclic_chain_mask, c_offset, offset)

            offset = (offset + self.r_max).clamp(0, 2 * self.r_max)
            d_residue_II = torch.where(
                b_samechain_II, offset, (2 * self.r_max + 1) * torch.ones_like(offset)
            )

        else:
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

        A_relpos_II = one_hot(d_residue_II.long(), 2 * self.r_max + 2)
        d_token_II = torch.where(
            b_samechain_II * b_sameresidue_II,
            torch.clip(
                f["token_index"].unsqueeze(-1)
                - f["token_index"].unsqueeze(-2)
                + self.r_max,
                0,
                2 * self.r_max,
            ),
            2 * self.r_max + 1,
        )
        A_reltoken_II = one_hot(d_token_II, 2 * self.r_max + 2)
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


class MSAModule(nn.Module):
    def __init__(
        self,
        n_block,
        c_m,
        p_drop_msa,
        p_drop_pair,
        msa_subsample_embedder,
        outer_product,
        msa_pair_weighted_averaging,
        msa_transition,
        triangle_multiplication_outgoing,
        triangle_multiplication_incoming,
        triangle_attention_starting,
        triangle_attention_ending,
        pair_transition,
    ):
        super().__init__()
        self.n_block = n_block
        self.msa_subsampler = MSASubsampleEmbedder(**msa_subsample_embedder)
        self.outer_product = OuterProductMean_AF3(**outer_product)
        self.msa_pair_weighted_averaging = MSAPairWeightedAverage(
            **msa_pair_weighted_averaging
        )
        self.msa_transition = Transition(**msa_transition)

        self.drop_row_msa = Dropout(broadcast_dim=-2, p_drop=p_drop_msa)
        self.drop_row_pair = Dropout(broadcast_dim=-2, p_drop=p_drop_pair)
        self.drop_col_pair = Dropout(broadcast_dim=-3, p_drop=p_drop_pair)

        self.tri_mult_outgoing = TriangleMultiplication(
            d_pair=triangle_multiplication_outgoing["d_pair"],
            d_hidden=triangle_multiplication_outgoing["d_hidden"],
            direction="outgoing",
            bias=True,
            use_cuequivariance=True,
        )
        self.tri_mult_incoming = TriangleMultiplication(
            d_pair=triangle_multiplication_incoming["d_pair"],
            d_hidden=triangle_multiplication_incoming["d_hidden"],
            direction="incoming",
            bias=True,
            use_cuequivariance=True,
        )
        self.tri_attn_start = TriangleAttention(
            **triangle_attention_starting, start_node=True, use_cuequivariance=True
        )
        self.tri_attn_end = TriangleAttention(
            **triangle_attention_ending, start_node=False, use_cuequivariance=True
        )
        self.pair_transition = Transition(**pair_transition)

        outer_product_expected_dim = 4  # B, S, I, C
        self.maybe_make_batched_outer_product = create_batch_dimension_if_not_present(
            outer_product_expected_dim
        )

        triangle_ops_expected_dim = 4  # B, I, I, C
        self.maybe_make_batched_triangle_ops = create_batch_dimension_if_not_present(
            triangle_ops_expected_dim
        )

    @activation_checkpointing
    def forward(
        self,
        f,
        Z_II,
        S_inputs_I,
    ):
        msa = f["msa"]
        msa_SI = self.msa_subsampler(msa, S_inputs_I)

        for i in range(self.n_block):
            # update MSA features
            Z_II = Z_II + self.maybe_make_batched_outer_product(self.outer_product)(
                msa_SI
            )
            msa_SI = msa_SI + self.drop_row_msa(
                self.msa_pair_weighted_averaging(msa_SI, Z_II)
            )
            msa_SI = msa_SI + self.msa_transition(msa_SI)

            # update pair features
            Z_II = Z_II + self.drop_row_pair(
                self.maybe_make_batched_triangle_ops(self.tri_mult_outgoing)(Z_II)
            )
            Z_II = Z_II + self.drop_row_pair(
                self.maybe_make_batched_triangle_ops(self.tri_mult_incoming)(Z_II)
            )

            Z_II = Z_II + self.drop_row_pair(
                self.maybe_make_batched_triangle_ops(self.tri_attn_start)(Z_II)
            )
            Z_II = Z_II + self.drop_col_pair(
                self.maybe_make_batched_triangle_ops(self.tri_attn_end)(Z_II)
            )
            Z_II = Z_II + self.pair_transition(Z_II)

        return Z_II


class RF3TemplateEmbedder(nn.Module):
    """
    Template track that enables conditioning on noisy ground-truth templates at the token level.
    Supports all chain types.
    """

    def __init__(
        self,
        n_block,
        raw_template_dim,
        c_z,
        c,
        p_drop,
        use_fourier_encoding: bool = False,  # HACK: Unused, kept for backwards compatibility with 9/21 checkpoint
    ):
        super().__init__()
        self.c = c
        self.emb_pair = nn.Linear(c_z, c, bias=False)
        self.norm_pair_before_pairformer = nn.LayerNorm(c_z)
        self.norm_after_pairformer = nn.LayerNorm(c)
        self.emb_templ = nn.Linear(raw_template_dim, c, bias=False)

        # template pairformer does not operate on sequence representation
        self.pairformer = nn.ModuleList(
            [
                PairformerBlock(
                    c_s=0,
                    c_z=c,
                    p_drop=p_drop,
                    triangle_multiplication=dict(d_hidden=c),
                    triangle_attention=dict(d_hidden=c),
                    attention_pair_bias={},
                    n_transition=4,
                )
                for _ in range(n_block)
            ]
        )

        # NOTE: this is not consistent with AF3 paper which outputs this tensor in the template_channel dimension
        # In Algorithm 1, line 9, the outputs of this function are added to the Z_II tensor which has dimensions [B, I, I, C_z]
        # so we make the outputs of this module also has those dimensions
        self.agg_emb = nn.Linear(c, c_z, bias=False)

    def forward(
        self,
        f,
        Z_II,
    ):
        @activation_checkpointing
        def embed_templates_like_rf3(
            has_distogram_condition,  # [I, I]
            distogram_condition_noise_scale,  # [I]
            distogram_condition,  # [I, I, 64], where 64 is the number of distogram bins
        ):
            I = Z_II.shape[0]  # n_tokens

            # Transform noise scale to reasonable range
            joint_noise_scale = (
                distogram_condition_noise_scale[None, :] ** 2
                + distogram_condition_noise_scale[:, None] ** 2
            ).sqrt()
            joint_noise_level = af3_noise_scale_to_noise_level(joint_noise_scale)

            # ---------------------------- #

            # ... concatenate along the channel dimension
            template_feats = torch.cat(
                [
                    distogram_condition,  # [I, I, 64]
                    has_distogram_condition.unsqueeze(-1),  # [I, I, 1]
                    joint_noise_level.unsqueeze(-1),  # [I, I, 1]
                ],
                dim=-1,
            )  # [I, I, 66]

            # ... remove any invalid interactions
            template_feats = template_feats * has_distogram_condition.unsqueeze(
                -1
            )  # [I, I, 66], where 66 = 64 + 1 + 1

            # ... embed template features
            template_channels = self.emb_templ(template_feats)  # [I, I, c]

            # ---------------------------- #

            # ... pass through pairformer
            u_II = torch.zeros(I, I, self.c, device=Z_II.device)
            v_II = (
                self.emb_pair(self.norm_pair_before_pairformer(Z_II))
                + template_channels
            )  # [I, I, c]
            for block in self.pairformer:
                _, v_II = block(None, v_II)
            u_II = u_II + self.norm_after_pairformer(v_II)

            return self.agg_emb(relu(u_II))

        #  Ground-truth template embedding (noisy ground-truth template as input)
        embedded_templates = embed_templates_like_rf3(
            has_distogram_condition=f["has_distogram_condition"],  # [I, I]
            distogram_condition_noise_scale=f["distogram_condition_noise_scale"],  # [I]
            distogram_condition=f[
                "distogram_condition"
            ],  # [I, I, 64], where 64 is the number of distogram bins
        )

        return embedded_templates
