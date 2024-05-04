import torch
import torch.nn as nn
from torch.nn.functional import one_hot, sigmoid
import torch.utils.checkpoint as checkpoint
from functools import partial
import numpy as np
from torch import relu
from icecream import ic

from rf2aa.debug import debug_nans
from rf2aa.model.layers.SE3_network import FullyConnectedSE3, FullyConnectedSE3_noR
from rf2aa.model.layers.structure_bias import structure_bias_factory
from rf2aa.model.layers.Attention_module import BiasedAxialAttention, FeedForwardLayer, MSAColAttention, \
    MSARowAttentionWithBias, TriangleMultiplication, MSAColGlobalAttention, \
    OldMSAColAttention, OldMSAColGlobalAttention, BiasedUntiedAxialAttention, TriangleAttention
from rf2aa.model.layers.outer_product import OuterProductMean # need to code this correctly
from rf2aa.training.checkpoint import create_custom_forward
from rf2aa.util_module import Dropout
#from rf2aa.alignment import weighted_rigid_align

'''
Glossary:
    I: # tokens (coarse representation)
    L: # atoms   (fine representation)
    M: # msa
    T: # templates
'''

linearNoBias = partial(torch.nn.Linear, bias=False)

class AtomAttentionEncoder(nn.Module):

    def __init__(self, c_atom, c_atompair, c_token, atom_1d_features, atom_transformer):
        super().__init__()
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_s = c_token
        self.atom_1d_features = atom_1d_features

        self.pair_mlp = nn.Sequential(
                nn.ReLU(),
                linearNoBias(self.c_atompair, c_atompair),
                nn.ReLU(),
                linearNoBias(self.c_atompair, c_atompair),
                nn.ReLU(),
                linearNoBias(self.c_atompair, c_atompair),
        )

        self.atom_transformer = AtomTransformer(c_atom=c_atom, c_atompair=c_atompair, **atom_transformer)

    def forward(
            self,
            f, # Dict (Input feature dictionary)
            Rl, # [B, L, 3]
            Si_trunk, # [B, I, C_S_trunk]
            Zij, # [B, I, I, C_Z]
            tok_idx, # [L] maps l --> i
    ):
        B, I, _ = Si_trunk.shape

        # Create the atom single conditioning: Embed per-atom meta data
        Cl = self.linear_no_bias_1(torch.concat(f[feature_name] for feature_name in self.atom_1d_features))

        # Embed offsets between atom reference positions
        Dlm = f['ref_pos'].unsqueeze(-1) - f['ref_pos'].unsqueeze(-2)
        Vlm = f['ref_space_uid'].unsqueeze(-1) == f['ref_space_uid'].unsqueeze(-2)
        Plm = self.linear_1(Dlm) * Vlm

        # Embed pairwise inverse squared distances, and the valid mask
        Plm += self.linear_2(1/(1+torch.linalg.norm(Dlm, dim=-1))) * Vlm

        # Initialise the atom single representation as the single conditioning.
        Ql = Cl

        # If provided, add trunk embeddings and noisy positions.
        ## Broadcast the single and pair embedding from the trunk.
        Sl_trunk = Si_trunk[:, self.tok_idx]
        Cl += self.linear_3(self.layer_norm_1(Sl_trunk))

        ## Add the noisy positions.
        Ql += self.linear_4(Rl)

        # Add the combined single conditioning to the pair representation.
        Plm += self.linear_5(relu(Cl)).unsqueeze(-1) + self.linear_6(relu(Cl)).unsqueeze(-2)

        # Run a small MLP on the pair activations
        Plm += self.pair_mlp(Plm)

        # Cross attention transformer.
        Ql = self.atom_transformer(Ql, Cl, Plm)

        # Aggregate per-atom representation to per-token representation
        Ai = torch.zeros((B, I, self.c_token)).reduce(
            1,
            tok_idx,
            relu(self.linear_6(Ql)),
            'mean',
            include_self=False)
        
        return Ai, Ql, Cl, Plm


class AtomAttentionDecoder(nn.Module):

    def __init__(self, c_token, c_atom, c_atompair, atom_transformer):
        super().__init__()
        self.atom_transformer = AtomTransformer(c_atom=c_atom, c_atompair=c_atompair, **atom_transformer)
        self.linear_1 = linearNoBias(c_token, c_atom)
        self.linear_2 = linearNoBias(c_atom, 3)

    def forward(
        self,
        Ai, # [L, C_token]
        Ql_skip, # [L, C_atom]
        Cl_skip, # [L, C_atom]
        Plm_skip, # [L, L, C_atompair]
        tok_idx, # [L] maps l --> i
    ):
        # Broadcast per-token activiations to per-atom activations and add the skip connection
        Ql = self.linear_1(Ai[tok_idx]) + Ql_skip

        # Cross attention transformer.
        Ql = self.atom_transformer(Ql, Cl_skip, Plm_skip)

        # Map to positions update
        Rl_update = self.linear_2(self.layer_norm_2(Ql))

        return Rl_update
    
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
        subset_centers = torch.arange(0, n_queries, l_max) + (n_queries-1 + n_queries / 2)
        
        l = torch.arange(l_max).unsqueeze(-1).unsqueeze(-1)   # [l_max, 1, 1]
        m = torch.arange(l_max).unsqueeze(0).unsqueeze(-1)    # [1, l_max, 1]
        c = subset_centers.unsqueeze(0).unsqueeze(0) # [1, 1, S]

        Beta_lms_binary = (torch.abs(l - m) < n_queries / 2) * (torch.abs(m - c) < n_keys / 2)
        ic(
            Beta_lms_binary.dtype,
        )
        Beta_lm_binary = Beta_lms_binary.prod(dim=-1, dtype=bool)
        ic(
            Beta_lm_binary.dtype,
        )
        self.Beta_lm = torch.where(Beta_lm_binary, 0, -10e10)

        self.diffusion_transformer = DiffusionTransformer(c_token=c_atom, c_tokenpair=c_atompair, **diffusion_transformer)

    def forward(
            self,
            Ql,  # [B, L, C_atom]
            Cl,  # [B, L, C_atom]
            Plm, # [B, L, L, C_atompair]
    ):
        B, L, _ = Ql.shape
        assert L < self.l_max
        Beta_lm = self.Beta_lm[:L, :L]
        return self.diffusion_transformer(Ql, Cl, Plm, Beta_lm)

class DiffusionTransformer(nn.Module):

    def __init__(self, c_token, c_tokenpair, n_block, diffusion_transformer_block):
        super().__init__()
        self.blocks = torch.nn.Sequential(*[
                DiffusionTransformerBlock(c_token=c_token, c_tokenpair=c_tokenpair, **diffusion_transformer_block)
                for _ in range(n_block)
        ])

    def forward(
            self,
            Ai,    # [B, I, C_token]
            Si,    # [B, I, C_token]
            Zij,   # [B, I, I, C_tokenpair]
            Beta_ij,   # [I, I]
    ):
        return self.blocks(Ai, Si, Zij, Beta_ij)

class DiffusionTransformerBlock(nn.Module):
    def __init__(self, c_token, c_tokenpair, n_head):
        super().__init__()
        self.attention_pair_bias = AttentionPairBias(c_a=c_token, c_pair=c_tokenpair, n_head=n_head)
        self.conditioned_transition_block = ConditionedTransitionBlock(c_token=c_token)

    def forward(
            self,
            Ai,    # [B, I, C_token]
            Si,    # [B, I, C_token]
            Zij,   # [B, I, I, C_tokenpair]
            Beta_ij,   # [I, I]
    ):
        Bi = self.attention_pair_bias(Ai, Si, Zij, Beta_ij)
        Ai = Bi + self.conditioned_transition_block(Ai, Si)
        return Ai, Si, Zij, Beta_ij

# class MultiHeadLinear(nn.Linear):
#     def __init__(self, in_features, out_features, h, *args, **kwargs):
#         self.h = h
#         self.out_features = out_features
#         super().__init__(in_features, out_features, *args, **kwargs)
    
#     def forward(self, x):
#         return sel
    

class MultiDimLinear(nn.Linear):
    def __init__(self, in_features, out_shape, **kwargs):
        self.out_shape = out_shape
        out_features = np.prod(out_shape)
        super().__init__(in_features, out_features, **kwargs)

    def forward(self, x):
        out = super().forward(x)
        return out.reshape(x.shape[:-1] + self.out_shape)

class LinearBiasInit(nn.Linear):

    def __init__(self, *args, biasinit, **kwargs):
        assert biasinit == -2. # Sanity check
        self.biasinit = biasinit
        super().__init__(*args, **kwargs)

    def reset_parameters(self) -> None:
        super().reset_parameters()
        self.bias.data.fill_(self.biasinit)

class AttentionPairBias(nn.Module):
    def __init__(self, c_a, c_pair, n_head):
        super().__init__()
        c = c_a // n_head

        ic(c, n_head)
        self.to_q = MultiDimLinear(c, (n_head, c))
        self.to_k = MultiDimLinear(c, (n_head, c), bias=False)
        self.to_v = MultiDimLinear(c, (n_head, c), bias=False)
        self.to_b = linearNoBias(c_pair, n_head)
        self.to_g = nn.Sequential(
            MultiDimLinear(c_a, (n_head, c), bias=False),
            nn.Sigmoid(),
        )
        self.to_a = linearNoBias(c_a, c_a)
        self.linear_output_project = nn.Sequential(
            LinearBiasInit(c_a, c_a, biasinit=-2.),
            nn.Sigmoid(),
        )
    
    def forward(
            self,
            Ai,      # [B, I, C_token]
            Si,      # [B, I, C_token] | None
            Zij,     # [B, I, I, C_tokepair]
            Beta_ij, # [I, I]
    ):
        # Input projections
        if Si is not None:
            Ai = self.ada_ln_1(Ai, Si)
        else:
            Ai = self.ln_1(Ai, Si)
        
        Qih = self.to_q(Ai)
        Kih = self.to_k(Ai)
        Vih = self.to_v(Ai)
        Bijh = self.to_b(Zij) + Beta_ij
        Gih = self.to_g(Ai)

        # Attention
        Aijh = torch.softmax(torch.pow(self.c, -1/2) * torch.einsum("bihd,bjhd->bijh", Qih, Kih) + Bijh, dim=-2) # softmax over j

        ## Gih: [B, I, H, C]
        ## Aijh: [B, I, I, H]
        ## ViH: [B, I, H, C]
        head_i = torch.einsum("bijh,bjhc->bihc", Aijh, Vih)
        head_i = Gih * head_i # [B, I, H, C]
        Ai = torch.concat(head_i, dim=-2) # [B, I, Ca]
        Ai = self.to_a(Ai)

        # Output projection (from adaLN-Zero)
        if Si is not None:
            Ai = self.linear_output_project(Si) * Ai
        
        return Ai

# SwiGLU transition block with adaptive layernorm
class ConditionedTransitionBlock(nn.Module):
    def __init__(self, c_token, n=2):
        super().__init__()
        self.ada_ln = AdaLN(c_token=c_token)
        self.linear_1 = linearNoBias(c_token, c_token*n)
        self.linear_2 = linearNoBias(c_token, c_token*n)
        self.linear_output_project = nn.Sequential(
            LinearBiasInit(c_token, c_token, biasinit=-2.),
            nn.Sigmoid(),
        )

    def forward(
            self,
            Ai,      # [B, I, C_token]
            Si,      # [B, I, C_token]
    ):
        Ai = self.ada_ln(Ai, Si)
        Bi = torch.silu(self.linear_1(Ai)) * self.linear_2(Ai)
        
        # Output projection (from adaLN-Zero)
        return self.linear_output_project(Si) * self.linear_3(Bi)

            
class AdaLN(nn.Module):
    def __init__(self, c_token, n=2):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape=(c_token,), elementwise_affine=False)
        self.ln_learnable_gain = nn.LayerNorm(normalized_shape=(c_token,), bias=False)
        self.linear_1 = nn.Linear(c_token, c_token)
        self.linear_2 = nn.Linear(c_token, c_token)
    
    def forward(
            self,
            Ai,      # [B, I, C_token]
            Si,      # [B, I, C_token]
    ):
        Ai = self.ln(Ai)
        Si = self.ln_learnable_gain(Si)
        return torch.sigmoid(self.linear_1(Si)) * Ai + self.linear_2(Si)
        
class DiffusionModule(nn.Module):
    def __init__(self, sigma_data, c_atom, c_atompair, c_token, c_s, c_z, diffusion_conditioning, atom_attention_encoder, diffusion_transformer, atom_attention_decoder):
        super().__init__()
        self.sigma_data = sigma_data
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token

        self.diffusion_conditioning = DiffusionConditioning(sigma_data=sigma_data, c_s=c_s, c_z=c_z, **diffusion_conditioning)
        self.atom_attention_encoder = AtomAttentionEncoder(c_token=c_token, c_atom=c_atom, c_atompair=c_atompair, **atom_attention_encoder)
        self.diffusion_transformer = DiffusionTransformer(c_token=c_token, c_tokenpair=c_atompair, **diffusion_transformer)
        self.layer_norm_1 = nn.LayerNorm(c_token)
        self.atom_attention_decoder = AtomAttentionDecoder(c_token=c_token, c_atom=c_atom, c_atompair=c_atompair, **atom_attention_decoder)
        
    def forward(self,
                X_noisy_L, # [B, L, 3]
                t, # [B] (0 is ground truth)
                f, # Dict (Input feature dictionary)
                S_input_I, # [B, I, C_S_input]
                S_trunk_I, # [B, I, C_S_trunk]
                Z_trunk_II, # [B, I, I, C_Z]
    ):
        # Conditioning
        S_I, Z_II = self.diffusion_conditioning(t, f, S_input_I, S_trunk_I, Z_trunk_II)

        # Scale positions to dimensionless vectors with approximately unit variance
        R_noisy_L = X_noisy_L / torch.sqrt(t^2 + self.sigma_data)

        # Sequence-local Atom Attention and aggregation to coarse-grained tokens
        A_I, Q_skip_L, C_skip_L, P_skip_LL = self.atom_attention_encoder(f, R_noisy_L, S_trunk_I, Z_II)

        # Full self-attention on token level
        A_I += self.linear_no_bias(self.layer_norm(S_I))
        A_I = self.diffusion_transformer(A_I, S_I, Z_II, Beta_II=0)
        A_I = self.layer_norm_1(A_I)

        # Broadcast token activations to atoms and run Sequence-local Atom Attention
        R_update_L = self.atom_attention_decoder(A_I, Q_skip_L, C_skip_L, P_skip_LL)

        # Rescale updates to positions and combine with input positions
        X_out_L = self.sigma_data**2 / (self.sigma_data**2 + t**2) * X_noisy_L + self.sigma_data * t / (self.sigma_data**2 + t**2) ** 0.5 * R_update_L

        return X_out_L

class DiffusionConditioning(nn.Module):
    def __init__(self, sigma_data, c_z, c_s, c_t_embed):
        super().__init__()
        self.sigma_data = sigma_data
        self.to_zii = nn.Sequential(
            nn.LayerNorm(c_z),
            linearNoBias(c_z, c_z)
        )
        self.transition_1 = nn.ModuleList([
            Transition(c=c_s, n=2),
            Transition(c=c_s, n=2),
        ])
        self.to_si = nn.Sequential(
            nn.LayerNorm(c_s),
            linearNoBias(c_s, c_s)
        )
        c_t_embed = 256
        self.fourier_embedding = FourierEmbedding(c_t_embed)
        self.process_n = nn.Sequential(
            nn.LayerNorm(c_t_embed),
            linearNoBias(c_t_embed, c_s)
        )
        self.transition_2 = nn.ModuleList([
            Transition(c=c_s, n=2),
            Transition(c=c_s, n=2),
        ])
    
    def forward(self,
                t,
                f,
                S_inputs_I,
                S_trunk_I,
                Z_trunk_II):
        # Pair conditioning
        Z_II = torch.concat([Z_trunk_II, self.relative_position_encoding(f)])
        Z_II = self.to_zii(Z_II)
        for b in range(2):
            Z_II += self.transition_1[b](Z_II)
        
        # Single conditioning
        S_I = torch.concat([S_trunk_I, S_inputs_I])
        S_I = self.to_si(S_I)
        N_T = self.fourier_embedding(1/4 * torch.log(t/self.sigma_data))
        S_I += self.process_n(N_T)
        for b in range(2):
            S_I += self.transition_2[b](S_I)
        
        return S_I, Z_II


class Transition(nn.Module):
    def __init__(self, n, c):
        super().__init__()
        self.n = n
        self.layer_norm_1 = nn.LayerNorm(c)
        self.linear_1 = linearNoBias(c, n*c)
        self.linear_2 = linearNoBias(c, n*c)
        self.linear_3 = linearNoBias(n*c, c)
    
    def forward(self,
                X,
                ):
        X = self.layer_norm_1(X)
        A = self.linear_1(X)
        B = self.linear_2(X)
        X = self.linear_3(torch.silu(A) * B)
        return X

pi = torch.acos(torch.zeros(1)).item() * 2
class FourierEmbedding(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.register_buffer('w', torch.zeros((c), dtype=torch.float32))
        self.register_buffer('b', torch.zeros((c), dtype=torch.float32))
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        nn.init.normal_(self.w)
        nn.init.normal_(self.b)
    
    def forward(self,
                t,
                ):
        return torch.cos(2 * pi * (t*self.w + self.b))

# from dataclasses import dataclass
# @dataclass
# class RecyclingInput:


class Model(nn.Module):
    def __init__(self,
                 c_s,
                 c_z,
                 c_atom,
                 c_atompair,
                 feature_initializer,
                 recycler,
                 diffusion_module,
                 **kwargs
                 ):
        super().__init__()
        self.feature_initializer = FeatureInitializer(c_s=c_s, c_z=c_z, c_atom=c_atom, c_atompair=c_atompair, **feature_initializer)
        self.recycler = Recycler(c_s=c_s, c_z=c_z, **recycler)
        self.diffusion_module = DiffusionModule(c_atom=c_atom, c_atompair=c_atompair, c_s=c_s, c_z=c_z, **diffusion_module)
    
    def forward(self,
                f,
                X_noisy_L,
                t,
                n_cycle,
                ):
        super().__init__()
        S_input_I, S_init_I, Z_init_II = self.feature_initializer(f)
        S_I = torch.zeros_like(S_init_I)
        Z_II = torch.zeros_like(Z_init_II)
        for _ in range(n_cycle):
            S_I, Z_II = self.recycler(f, S_input_I, S_init_I, Z_init_II, S_I, Z_II)
        X_pred = self.diffusion_module(
                X_noisy_L,
                t,
                f,
                S_input_I, 
                S_I,
                Z_II,
        )
        return X_pred
    
    def pre_recycle(self,
                    f,
                    X_noisy_L,
                    t):
        S_input_I, S_init_I, Z_init_II = self.feature_initializer(f)
        S_I = torch.zeros_like(S_init_I)
        Z_II = torch.zeros_like(Z_init_II)
        return S_input_I, S_init_I, Z_init_II, S_I, Z_II, f, X_noisy_L, t
    
    def recycle(self,
                S_input_I,
                S_init_I,
                Z_init_II,
                S_I,
                Z_II,
                f,
                X_noisy_L,
                t,
                ):
        S_I, Z_II = self.recycler(
            S_input_I,
            S_init_I,
            Z_init_II,
            S_I,
            Z_II
        )
        return S_input_I, S_init_I, Z_init_II, S_I, Z_II, f, X_noisy_L, t
    
    def post_recycle(self,
                     S_input_I,
                     S_init_I,
                     Z_init_II,
                     S_I,
                     Z_II,
                     f,
                     X_noisy_L,
                     t,
                    ):
        X_pred = self.diffusion_module(
            X_noisy_L,
            t,
            f,
            S_input_I, 
            S_I,
            Z_II,
        )
        return X_pred
        

class Recycler(nn.Module):
    def __init__(self,
                 c_s,
                 c_z,
                 template_embedder,
                 msa_module,
                 n_pairformer_blocks,
                 pairformer_block,
                 ):
        super().__init__()
        self.process_zh = nn.Sequential(
            nn.LayerNorm(c_z),
            linearNoBias(c_z, c_z),
        )
        self.template_embedder = TemplateEmbedder(c_z=c_z, **template_embedder)
        self.msa_module = MSAModule(**msa_module)
        self.process_sh = nn.Sequential(
            nn.LayerNorm(c_s),
            linearNoBias(c_s, c_s),
        )
        self.pairformer_stack = nn.Sequential(*[
            PairformerBlock(c_s=c_s, c_z=c_z, **pairformer_block) for _ in range(n_pairformer_blocks)
        ])

    def forward(self,
                f,
                S_inputs_I,
                S_init_I,
                Z_init_II,
                Sh_I,
                Zh_II,
                ):
        Z_II = Z_init_II + self.process_zh(Zh_II)
        Z_II += self.template_embedder(f, Z_II)
        Z_II += self.msa_module(f['msa'], Z_II, S_inputs_I)
        S_I = S_init_I + self.process_sh(Sh_I)
        S_I, Z_II = self.pairformer_stack(S_I, Z_II)
        return S_I, Z_II

class PairformerBlock(nn.Module):
    """ 
    Attempt to replicate AF3 architecture from scratch.
    """
    def __init__(self,
                 c_s,
                 c_z,
                 p_drop,
                 c,
                 attention_pair_bias,
                 n_transition=4,
    ):
        super().__init__()

        self.drop_row = Dropout(broadcast_dim=2, p_drop=p_drop)
        self.drop_col = Dropout(broadcast_dim=1, p_drop=p_drop)

        self.tri_mul_outgoing = TriangleMultiplication(
            c_z, d_hidden=c, outgoing=True, bias=False)
        self.tri_mul_incoming = TriangleMultiplication(
            c_z, d_hidden=c, outgoing=False, bias=False)
        self.tri_attn_start = TriangleAttention(
            c_z, d_hidden=c, start_node=True)
        self.tri_attn_end = TriangleAttention(
            c_z, d_hidden=c, start_node=False)

        self.z_transition = Transition(c_z, n_transition)
        self.s_transition = Transition(c_s, n_transition)

        self.attention_pair_bias = AttentionPairBias(c_a=c_s, c_pair=c_z, **attention_pair_bias)

    def forward(self,
                S_I,
                Z_II):
        Z_II += self.drop_row(self.tri_mul_outgoing(Z_II))
        Z_II += self.drop_row(self.tri_mul_incoming(Z_II))
        Z_II += self.drop_row(self.tri_attn_start(Z_II))
        Z_II += self.drop_col(self.tri_attn_end(Z_II))
        Z_II += self.z_transition(Z_II)
        S_I += self.attention_pair_bias(S_I, None, Z_II, Beta_II=0)
        S_I += self.s_transition(S_I)
        return S_I, Z_II


class FeatureInitializer(nn.Module):
    def __init__(self,
                 c_s,
                 c_z,
                 c_atom,
                 c_atompair,
                 input_feature_embedder,
                 relative_position_encoding):
        super().__init__()
        self.input_feature_embedder = InputFeatureEmbedder(c_atom=c_atom, c_atompair=c_atompair, c_s=c_s, **input_feature_embedder)
        self.to_s_init = linearNoBias(c_s, c_s)
        self.to_z_init_i = linearNoBias(c_s, c_z)
        self.to_z_init_j = linearNoBias(c_s, c_z)
        self.relative_position_encoding = RelativePositionEncoding(c_z=c_z, **relative_position_encoding)
        self.process_token_bonds = linearNoBias(1, c_z)
    
    def forward(self,
                f,
                ):
        S_inputs_I = self.input_feature_embedder(f)
        S_init_I = self.to_s_init(S_init_I)
        Z_init_II = self.to_z_init_i(S_inputs_I).unsqueeze(-3) + self.to_z_init_j(S_inputs_I).unsqueeze(-2)
        Z_init_II += self.relative_position_encoding(f)
        Z_init_II += self.process_token_bonds(f['token_bonds'])
        return S_inputs_I, S_init_I, Z_init_II


class InputFeatureEmbedder(nn.Module):
    def __init__(self,
                 features, 
                 c_atom,
                 c_atompair,
                 c_s,
                 atom_attention_encoder):
        super().__init__()
        self.atom_attention_encoder = AtomAttentionEncoder(c_atom=c_atom, c_atompair=c_atompair, c_token=c_s, **atom_attention_encoder)
        self.features = features
    
    def forward(self,
                f,
                A_I,
                ):
        S_I, _, _, _ = self.atom_attention_encoder(A_I)
        S_I = torch.concat([A_I] + [f[feature] for feature in self.features])
        return S_I

class RelativePositionEncoding(nn.Module):
    def __init__(self,
                 r_max,
                 s_max,
                 c_z):
        super().__init__()
        self.r_max = r_max
        self.s_max = s_max
        self.c_z = c_z
        self.linear = linearNoBias(2*(2*self.r_max+2) + (2*self.s_max+2) + 1, c_z)
    
    def forward(self,
                f):
        b_samechain_II = f['asym_id'].unsqueeze(-1) == f['asym_id'].unsqueeze(-2)
        b_sameresidue_II = f['residue_index'].unsqueeze(-1) == f['residue_index'].unsqueeze(-2)
        b_same_entity_II = f['entity_id'].unsqueeze(-1) == f['entity_id'].unsqueeze(-2)
        d_residue_II = torch.where(
            b_samechain_II,
            torch.clip(f['residue_index'].unsqueeze(-2) - f['residue_index'].unsqueeze(-1) + self.r_max, 0, 2*self.r_max),
            2 * self.r_max + 1
        )
        A_relpos_II = one_hot(d_residue_II, 2*self.r_max+2)
        d_token_II = torch.where(
            b_samechain_II * b_sameresidue_II,
            torch.clip(f['token_index'].unsqueeze(-2) - f['token_index'].unsqueeze(-1) + self.r_max, 0, 2*self.r_max),
            2 * self.r_max + 1
        )
        A_reltoken_II = one_hot(d_token_II, 2*self.r_max+2)
        d_chain_II = torch.where(
            b_samechain_II,
            torch.clip(f['sym_id'].unsqueeze(-2) - f['sym_id'].unsqueeze(-1) + self.s_max, 0, 2*self.s_max),
            2 * self.s_max + 1
        )
        A_relchain_II = one_hot(d_chain_II, 2*self.s_max+2)
        return self.linear(torch.cat([A_relpos_II, A_reltoken_II, b_same_entity_II.unsqueeze(-1), A_relchain_II], dim=3))

# Mock for testing.
class MSAModule(nn.Module):
    def __init__(self, n_block, c_m):
        super().__init__()
    
    def forward(self,
                f,
                Z_II,
                S_inputs_I,
                ):
        return Z_II

# Mock for testing.
class TemplateEmbedder(nn.Module):
    def __init__(self,
                 n_block,
                 c_z,
                 c):
        super().__init__()
        self.c =c
        self.linear = linearNoBias(c_z, c)

    def forward(self,
                f,
                Z_II,
                ):
        return self.linear(Z_II)

class Loss:
    def __init__(self,
                 sigma_data,
                 ):
        self.sigma_data = sigma_data
    
    def __call__(self,
                 f,
                 X_L, # [B, L, 3]
                 X_gt_L, # [B, L, 3]
                 t, # [B]
                 ):
        w_L = 1 + (
            f['is_dna']*self.alpha_is_dna +
            f['is_rna'] * self.alpha_is_rna + 
            f['is_ligand'] * self.alpha_is_ligand
        )
        # Align ground truth onto predictions.
        X_gt_aligned_L = weighted_rigid_align(X_gt_L, X_L, w_L)
        l_mse = 1/3 * w_L * torch.mean(torch.linalg.norm(X_L, X_gt_aligned_L, dim=-1), dim=-1) # [B]
        
        l_diffusion = (t**2 + self.sigma_data**2) / (t + self.sigma_data)**2 * l_mse

        # TODO: implement auxiliary losses
        l_total = l_diffusion.sum()

        return l_total, {
            'diffusion_loss': l_diffusion
        }
        