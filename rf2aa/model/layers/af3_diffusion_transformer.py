import torch
import torch.nn as nn
import numpy as np

from rf2aa.training.checkpoint import activation_checkpointing
#from rf2aa.model.layers.pairformer_layers import AttentionPairBias
from rf2aa.model.layers.layer_utils import linearNoBias, LinearBiasInit, AdaLN, MultiDimLinear, collapse


class AtomAttentionEncoderDiffusion(nn.Module):

    def __init__(self, c_atom, c_atompair, c_token, c_tokenpair, c_s, atom_1d_features, c_atom_1d_features, atom_transformer):
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

        self.process_s_trunk = nn.Sequential(
            nn.LayerNorm(c_s),
            linearNoBias(c_s, c_atom)
        )
        self.process_z = nn.Sequential(
            nn.LayerNorm(c_tokenpair),
            linearNoBias(c_tokenpair, c_atompair)
        )
        self.process_r = linearNoBias(3, c_atom)

        self.process_single_l = nn.Sequential(
            nn.ReLU(),
            linearNoBias(c_atom, c_atompair)
        )
        self.process_single_m = nn.Sequential(
            nn.ReLU(),
            linearNoBias(c_atom, c_atompair)
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

        self.atom_transformer = AtomTransformer(c_atom=c_atom, c_atompair=c_atompair, **atom_transformer)
    
    def forward(
            self,
            f, # Dict (Input feature dictionary)
            R_L, # [D, L, 3]
            S_trunk_I, # [B, I, C_S_trunk] [...,I,C_S_trunk]
            Z_II, # [B, I, I, C_Z] [...,I,I,C_Z]
    ):
        assert R_L is not None

        tok_idx = f['atom_to_token_map']
        L = len(tok_idx)
        I = tok_idx.max() + 1

        f["ref_atom_name_chars"] = f["ref_atom_name_chars"].reshape(L, -1)
        # Create the atom single conditioning: Embed per-atom meta data
        C_L = self.process_input_features(torch.cat(tuple(collapse(f[feature_name], L) for feature_name in self.atom_1d_features), dim=-1))

        # Embed offsets between atom reference positions
        D_LL = f['ref_pos'].unsqueeze(-2) - f['ref_pos'].unsqueeze(-3)
        V_LL = (f['ref_space_uid'].unsqueeze(-1) == f['ref_space_uid'].unsqueeze(-2)).unsqueeze(-1)
        P_LL = self.process_d(D_LL) * V_LL

        # Embed pairwise inverse squared distances, and the valid mask
        P_LL = P_LL + self.process_inverse_dist(1/(1+torch.linalg.norm(D_LL, dim=-1, keepdim=True))) * V_LL
        P_LL = P_LL + self.process_valid_mask(V_LL.to(torch.float)) * V_LL

        # Initialise the atom single representation as the single conditioning.
        Q_L = C_L

        # If provided, add trunk embeddings and noisy positions.
        if R_L is not None:
            # Broadcast the single and pair embedding from the trunk.
            S_trunk_embed_L = self.process_s_trunk(S_trunk_I)[..., tok_idx, :]

            C_L = C_L + S_trunk_embed_L
            assert not (C_L == Q_L).all()
            P_LL = P_LL + self.process_z(Z_II)[..., tok_idx, tok_idx, :]

            # Add the noisy positions.
            Q_L = self.process_r(R_L) + Q_L

        # Add the combined single conditioning to the pair representation.
        P_LL = P_LL + (self.process_single_l(C_L).unsqueeze(-2) + self.process_single_m(C_L).unsqueeze(-3))

        # Run a small MLP on the pair activations
        P_LL = P_LL + self.pair_mlp(P_LL)

        # Cross attention transformer.
        Q_L = self.atom_transformer(Q_L, C_L, P_LL)

        A_I_shape = Q_L.shape[:-2] + (I, self.c_token,)
        # Aggregate per-atom representation to per-token representation
        A_I = torch.zeros(A_I_shape, device=Q_L.device).index_reduce(
            -2,
            f['atom_to_token_map'].long(),
            self.process_q(Q_L),
            'mean',
            include_self=False).clone()
        
        return A_I, Q_L, C_L, P_LL



class AtomTransformer(nn.Module):

    def __init__(
            self,
            c_atom,
            c_atompair,
            diffusion_transformer,
            n_queries,
            n_keys,
            l_max,
    ):
        super().__init__()
        self.l_max = l_max
        subset_centers = torch.arange(0, l_max, n_queries) + (n_queries-1) / 2
        
        l = torch.arange(l_max).unsqueeze(-1).unsqueeze(-1)   # [l_max, 1, 1]
        m = torch.arange(l_max).unsqueeze(0).unsqueeze(-1)    # [1, l_max, 1]
        c = subset_centers.unsqueeze(0).unsqueeze(0) # [1, 1, S]

        Beta_lms_binary = (torch.abs(l - c) < n_queries / 2) * (torch.abs(m - c) < n_keys / 2)
        Beta_lm_binary = Beta_lms_binary.sum(dim=-1, dtype=bool)
        Beta_lm = torch.where(Beta_lm_binary, 0, -1e5) # is -1e10 in the paper but getting nans
        self.register_buffer('Beta_lm', Beta_lm)
        self.diffusion_transformer = DiffusionTransformer(c_token=c_atom, c_s=c_atom, c_tokenpair=c_atompair, **diffusion_transformer)

    def forward(
            self,
            Ql,  # [B, L, C_atom]
            Cl,  # [B, L, C_atom]
            Plm, # [B, L, L, C_atompair]
    ):
        L = Ql.shape[-2]
        assert L < self.l_max
        Beta_lm = self.Beta_lm[:L, :L]
        return self.diffusion_transformer(Ql, Cl, Plm, Beta_lm)


class DiffusionTransformer(nn.Module):

    def __init__(self, c_token, c_s, c_tokenpair, n_block, diffusion_transformer_block):
        super().__init__()
        self.blocks = torch.nn.ModuleList([
                DiffusionTransformerBlock(c_token=c_token, c_s=c_s, c_tokenpair=c_tokenpair, **diffusion_transformer_block)
                for _ in range(n_block)
        ])

    def forward(
            self,
            A_I,    # [..., I, C_token]
            S_I,    # [..., I, C_token]
            Z_II,   # [..., I, I, C_tokenpair]
            Beta_II,   # [I, I]
    ):
        for block in self.blocks:
            #NOTE: inconsistent with the original implementation added residual connection
            # old implementation A_I = block(A_I, S_I, Z_II, Beta_II)
            A_I = A_I + block(A_I, S_I, Z_II, Beta_II)
        return A_I


class DiffusionTransformerBlock(nn.Module):
    def __init__(self, c_token, c_s, c_tokenpair, n_head):
        super().__init__()
        self.attention_pair_bias = AttentionPairBiasDiffusionDeepspeed(c_a=c_token, c_s=c_s, c_pair=c_tokenpair, n_head=n_head)
        self.conditioned_transition_block = ConditionedTransitionBlock(c_token=c_token, c_s=c_s)

    @activation_checkpointing
    def forward(
            self,
            A_I,    # [..., I, C_token]
            S_I,    # [..., I, C_s]
            Z_II,   # [..., I, I, C_tokenpair]
            Beta_II,   # [I, I]
    ):
        use_amp = True
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
            B_I = self.attention_pair_bias(A_I, S_I, Z_II, Beta_II)
        A_I = B_I + self.conditioned_transition_block(A_I, S_I)
        return A_I


# SwiGLU transition block with adaptive layernorm
class ConditionedTransitionBlock(nn.Module):
    def __init__(self, c_token, c_s, n=2):
        super().__init__()
        self.ada_ln = AdaLN(c_a=c_token, c_s=c_s)
        self.linear_1 = linearNoBias(c_token, c_token*n)
        self.linear_2 = linearNoBias(c_token, c_token*n)
        self.linear_output_project = nn.Sequential(
            LinearBiasInit(c_s, c_token, biasinit=-2.),
            nn.Sigmoid(),
        )
        self.linear_3 = linearNoBias(c_token*n, c_token)

    def forward(
            self,
            Ai,      # [B, I, C_token]
            Si,      # [B, I, C_token]
    ):
        Ai = self.ada_ln(Ai, Si)
        # BUG: This is not the correct implementation of SwiGLU
        # Bi = torch.sigmoid(self.linear_1(Ai)) * self.linear_2(Ai)
        # FIX: This is the correct implementation of SwiGLU
        Bi = torch.nn.functional.silu(self.linear_1(Ai)) * self.linear_2(Ai)
        
        # Output projection (from adaLN-Zero)
        return self.linear_output_project(Si) * self.linear_3(Bi)

class AttentionPairBiasDiffusion(nn.Module):
    def __init__(self, c_a, c_s, c_pair, n_head):
        super().__init__()
        self.c_a = c_a
        self.c_pair = c_pair
        self.c = c_a // n_head

        self.to_q = MultiDimLinear(c_a, (n_head, self.c))
        self.to_k = MultiDimLinear(c_a, (n_head, self.c), bias=False)
        self.to_v = MultiDimLinear(c_a, (n_head, self.c), bias=False)
        self.to_b = linearNoBias(c_pair, n_head)
        self.to_g = nn.Sequential(
            MultiDimLinear(c_a, (n_head, self.c), bias=False),
            nn.Sigmoid(),
        )
        self.to_a = linearNoBias(c_a, c_a)
        self.linear_output_project = nn.Sequential(
            LinearBiasInit(c_s, c_a, biasinit=-2.),
            nn.Sigmoid(),
        )
        self.ln_0 = nn.LayerNorm((c_pair,))
        self.ada_ln_1 = AdaLN(c_a=c_a, c_s=c_s)
        #self.ln_1 = nn.LayerNorm((c_a,))

    def reset_parameters(self) -> None:
        super().reset_parameters()

    @activation_checkpointing
    def forward(
            self,
            A_I,      # [B, I, C_a]
            S_I,      # [B, I, C_a] 
            Z_II,     # [B, I, I, C_z]
            Beta_II=None, # [I, I]
    ):
        # Input projections
        assert S_I is not None
        if S_I is not None:
            A_I = self.ada_ln_1(A_I, S_I)
        #else:
            #A_I = self.ln_1(A_I)
        
        Q_IH = self.to_q(A_I)
        K_IH = self.to_k(A_I)
        V_IH = self.to_v(A_I)
        B_IIH = self.to_b(self.ln_0(Z_II)) + Beta_II[..., None]
        G_IH = self.to_g(A_I)

        # Attention
        A_IIH = torch.softmax(torch.tensor(self.c).pow(-1/2) * torch.einsum("...ihd,...jhd->...ijh", Q_IH, K_IH) + B_IIH, dim=-2) # softmax over j

        ## G_IH: [B, I, H, C]
        ## A_IIH: [B, I, I, H]
        ## V_IH: [B, I, H, C]
        head_I = torch.einsum("...ijh,...jhc->...ihc", A_IIH, V_IH)
        head_I = G_IH * head_I # [B, I, H, C]
        A_I = head_I.flatten(start_dim=-2) # [B, I, Ca]
        A_I = self.to_a(A_I)

        # Output projection (from adaLN-Zero)
        if S_I is not None:
            A_I = self.linear_output_project(S_I) * A_I
        
        return A_I

class AttentionPairBiasDiffusionDeepspeed(nn.Module):

    def __init__(self, c_a, c_s, c_pair, n_head):
        super().__init__()
        self.n_head = n_head
        self.c_a = c_a
        self.c_pair = c_pair
        self.c = c_a // n_head

        self.to_q = MultiDimLinear(c_a, (n_head, self.c))
        self.to_k = MultiDimLinear(c_a, (n_head, self.c), bias=False)
        self.to_v = MultiDimLinear(c_a, (n_head, self.c), bias=False)
        self.to_b = linearNoBias(c_pair, n_head)
        self.to_g = nn.Sequential(
            MultiDimLinear(c_a, (n_head, self.c), bias=False),
            nn.Sigmoid(),
        )
        self.to_a = linearNoBias(c_a, c_a)
        self.linear_output_project = nn.Sequential(
            LinearBiasInit(c_s, c_a, biasinit=-2.),
            nn.Sigmoid(),
        )
        self.ln_0 = nn.LayerNorm((c_pair,))
        self.ada_ln_1 = AdaLN(c_a=c_a, c_s=c_s)
        #self.ln_1 = nn.LayerNorm((c_a,))
        self.use_deepspeed_evo = True
        self.force_bfloat16 = True

    @activation_checkpointing
    def forward(
            self,
            A_I,      # [I, C_a]
            S_I,      # [I, C_a] | None
            Z_II,     # [I, I, C_z]
            Beta_II, # [I, I]
    ):
        # Input projections
        assert S_I is not None
        if S_I is not None:
            A_I = self.ada_ln_1(A_I, S_I)
 
        
        if self.use_deepspeed_evo or self.force_bfloat16:
            A_I = A_I.to(torch.bfloat16)

        Q_IH = self.to_q(A_I) / np.sqrt(self.c)
        K_IH = self.to_k(A_I)
        V_IH = self.to_v(A_I)
        B_IIH = self.to_b(self.ln_0(Z_II)) + Beta_II[..., None]
        G_IH = self.to_g(A_I)

        B, L = B_IIH.shape[:2]

        if not self.use_deepspeed_evo or L<=24: 
            # Attention
            A_IIH = torch.softmax(torch.einsum("...ihd,...jhd->...ijh", Q_IH, K_IH) + B_IIH, dim=-2) # softmax over j
            ## G_IH: [I, H, C]
            ## A_IIH: [I, I, H]
            ## V_IH: [I, H, C]
            A_I = torch.einsum("...ijh,...jhc->...ihc", A_IIH, V_IH)
            A_I = G_IH * A_I # [B, I, H, C]
            A_I = A_I.flatten(start_dim=-2) # [B, I, Ca]
        else:
            # DS4Sci_EvoformerAttention
            # Q, K, V: [Batch, N_seq, N_res, Head, Dim]
            # res_mask: [Batch, N_seq, 1, 1, N_res]
            # pair_bias: [Batch, 1, Head, N_res, N_res]
            from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention
            print(Q_IH.shape, K_IH.shape, V_IH.shape, B_IIH.shape)
            if len(Q_IH.shape) == 3:
                Q_IH = Q_IH[None]
                K_IH = K_IH[None]
                V_IH = V_IH[None]
                B_IIH = B_IIH[None]
                G_IH = G_IH[None]
            batch = Q_IH.shape[0]
            n_res = Q_IH.shape[1]
            n_head = self.n_head
            c = self.c

            Q_IH = Q_IH[:, None]
            K_IH = K_IH[:,None]
            V_IH = V_IH[:,None]
            B_IIH = B_IIH.repeat(Q_IH.shape[0],1,1,1)
            B_IIH = B_IIH[:,None]
            B_IIH = B_IIH.permute(0,1,4,2,3).to(torch.bfloat16)
            mask = torch.ones([Q_IH.shape[0],1,1,1,B_IIH.shape[-1]], dtype=torch.bfloat16, device=B_IIH.device)
            print(Q_IH.shape, K_IH.shape, V_IH.shape, mask.shape, B_IIH.shape)
            try:
                assert Q_IH.shape == (batch, 1, n_res, n_head, c)
            except:
                import pdb; pdb.set_trace()
            assert K_IH.shape == (batch, 1, n_res, n_head, c)
            assert V_IH.shape == (batch, 1, n_res, n_head, c)
            assert mask.shape == (batch, 1, 1, 1, n_res)
            assert B_IIH.shape == (batch, 1, n_head, n_res, n_res)

            A_I = DS4Sci_EvoformerAttention(Q_IH, K_IH, V_IH, [mask,B_IIH])

            assert A_I.shape == (batch, 1, n_res, n_head, c)
            print(A_I.shape, G_IH.shape)
            A_I = A_I * G_IH[:,None]
            print(A_I.shape)
            A_I = A_I.view(batch, n_res,-1)

        A_I = self.to_a(A_I)
        # Output projection (from adaLN-Zero)
        if S_I is not None:
            A_I = self.linear_output_project(S_I) * A_I

        return A_I