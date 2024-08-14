import torch
import torch.nn as nn
from torch.nn.functional import one_hot, sigmoid, silu, relu
import torch.utils.checkpoint as checkpoint
from functools import partial
import numpy as np
from torch import relu
from torch.nn import functional as F
from icecream import ic
from contextlib import ExitStack
import logging

from rf2aa.training.checkpoint import activation_checkpointing
from rf2aa.chemical import ChemicalData as ChemData 
from rf2aa.debug import debug_nans
from rf2aa.model.layers.SE3_network import FullyConnectedSE3, FullyConnectedSE3_noR
from rf2aa.model.layers.structure_bias import structure_bias_factory
from rf2aa.model.layers.Attention_module import BiasedAxialAttention, FeedForwardLayer, MSAColAttention, \
    MSARowAttentionWithBias, TriangleMultiplication, MSAColGlobalAttention, \
    OldMSAColAttention, OldMSAColGlobalAttention, BiasedUntiedAxialAttention, TriangleAttention
from rf2aa.model.layers.outer_product import OuterProductMean_AF3 # need to code this correctly
from rf2aa.model.AF3_blocks import MsaPairWeightedAverage, MsaSubsampleEmbedder
from rf2aa.training.checkpoint import create_custom_forward
from rf2aa.util_module import Dropout
from rf2aa.alignment import weighted_rigid_align
from rf2aa.debug import pretty_describe_dict
from rf2aa.tensor_util import assert_shape, assert_cmp

logger = logging.getLogger(__name__)

'''
Glossary:
    I: # tokens (coarse representation)
    L: # atoms   (fine representation)
    M: # msa
    T: # templates
    D: # diffusion structure batch dim
'''


class ProteinLinear(nn.Linear):
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__(in_features, out_features, **kwargs)
    
    def reset_parameters(self, **kwargs) -> None:
        pass

    def forward(self, x):
        return super().forward(x)
linearNoBias = partial(torch.nn.Linear, bias=False)
def collapse(x, L):
    return x.reshape((L,x.numel()//L))

class AtomAttentionEncoder(nn.Module):

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
        tok_idx = f['tok_idx']
        L = len(tok_idx)
        I = tok_idx.max() + 1

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
            # S_trunk_L = S_trunk_I[..., tok_idx, :]
            # S_trunk_embed_L_slow = self.process_s_trunk(S_trunk_L)
            S_trunk_embed_L_slow = self.process_s_trunk(S_trunk_I[..., tok_idx, :])
            S_trunk_embed_L = self.process_s_trunk(S_trunk_I)[..., tok_idx, :]
            assert_cmp(S_trunk_embed_L_slow, S_trunk_embed_L)

            C_L = C_L + S_trunk_embed_L
            # P_LL = P_LL + self.process_z(Z_II[..., tok_idx, tok_idx, :])
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
            f['tok_idx'],
            self.process_q(Q_L),
            'mean',
            include_self=False).clone()
        
        return A_I, Q_L, C_L, P_LL


class AtomAttentionDecoder(nn.Module):

    def __init__(self, c_token, c_atom, c_atompair, atom_transformer):
        super().__init__()
        self.atom_transformer = AtomTransformer(c_atom=c_atom, c_atompair=c_atompair, **atom_transformer)
        self.linear_1 = linearNoBias(c_token, c_atom)
        self.to_r_update = nn.Sequential(
            nn.LayerNorm((c_atom,)),
            linearNoBias(c_atom, 3)
        )

    def forward(
        self,
        f,
        Ai, # [L, C_token]
        Ql_skip, # [L, C_atom]
        Cl_skip, # [L, C_atom]
        Plm_skip, # [L, L, C_atompair]
    ):
        tok_idx = f['tok_idx']
        # Broadcast per-token activiations to per-atom activations and add the skip connection
        Ql = self.linear_1(Ai[...,tok_idx,:]) + Ql_skip

        # Cross attention transformer.
        Ql = self.atom_transformer(Ql, Cl_skip, Plm_skip)

        # Map to positions update
        Rl_update = self.to_r_update(Ql)

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
        subset_centers = torch.arange(0, l_max, n_queries) + (n_queries-1) / 2
        
        l = torch.arange(l_max).unsqueeze(-1).unsqueeze(-1)   # [l_max, 1, 1]
        m = torch.arange(l_max).unsqueeze(0).unsqueeze(-1)    # [1, l_max, 1]
        c = subset_centers.unsqueeze(0).unsqueeze(0) # [1, 1, S]

        Beta_lms_binary = (torch.abs(l - c) < n_queries / 2) * (torch.abs(m - c) < n_keys / 2)
        Beta_lm_binary = Beta_lms_binary.sum(dim=-1, dtype=bool)
        Beta_lm = torch.where(Beta_lm_binary, 0, -1e10)
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
            A_I = block(A_I, S_I, Z_II, Beta_II)
        return A_I


class DiffusionTransformerBlock(nn.Module):
    def __init__(self, c_token, c_s, c_tokenpair, n_head):
        super().__init__()
        self.attention_pair_bias = AttentionPairBias(c_a=c_token, c_s=c_s, c_pair=c_tokenpair, n_head=n_head)
        self.conditioned_transition_block = ConditionedTransitionBlock(c_token=c_token, c_s=c_s)

    @activation_checkpointing
    def forward(
            self,
            A_I,    # [..., I, C_token]
            S_I,    # [..., I, C_s]
            Z_II,   # [..., I, I, C_tokenpair]
            Beta_II,   # [I, I]
    ):
        B_I = self.attention_pair_bias(A_I, S_I, Z_II, Beta_II)
        A_I = B_I + self.conditioned_transition_block(A_I, S_I)
        return A_I

class MultiDimLinear(nn.Linear):
    def __init__(self, in_features, out_shape, **kwargs):
        self.out_shape = out_shape
        out_features = np.prod(out_shape)
        super().__init__(in_features, out_features, **kwargs)

    def reset_parameters(self, **kwargs) -> None:
        super().reset_parameters()
        nn.init.xavier_uniform_(self.weight)

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
        self.ln_1 = nn.LayerNorm((c_a,))

    def reset_parameters(self) -> None:
        super().reset_parameters()

    def forward(
            self,
            A_I,      # [B, I, C_a]
            S_I,      # [B, I, C_a] | None
            Z_II,     # [B, I, I, C_z]
            Beta_II, # [I, I]
    ):
        # Input projections
        if S_I is not None:
            A_I = self.ada_ln_1(A_I, S_I)
        else:
            A_I = self.ln_1(A_I)
        
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
        Bi = torch.sigmoid(self.linear_1(Ai)) * self.linear_2(Ai)
        
        # Output projection (from adaLN-Zero)
        return self.linear_output_project(Si) * self.linear_3(Bi)

            
class AdaLN(nn.Module):
    def __init__(self, c_a, c_s, n=2):
        super().__init__()
        self.ln_a = nn.LayerNorm(normalized_shape=(c_a,), elementwise_affine=False)
        self.ln_s = nn.LayerNorm(normalized_shape=(c_s,), bias=False)
        self.to_gain = nn.Sequential(
            nn.Linear(c_s, c_a),
            nn.Sigmoid(),
        )
        self.to_bias = linearNoBias(c_s, c_a)
    
    def forward(
            self,
            Ai,      # [B, I, C_a]
            Si,      # [B, I, C_s]
    ):
        '''
        Output:
            [B, I, C_a]
        '''
        Ai = self.ln_a(Ai)
        Si = self.ln_s(Si)
        return  self.to_gain(Si) * Ai + self.to_bias(Si)
        
class DiffusionModule(nn.Module):
    def __init__(self, sigma_data, c_atom, c_atompair, c_token, c_s, c_z, f_pred, diffusion_conditioning, atom_attention_encoder, diffusion_transformer, atom_attention_decoder):
        super().__init__()
        self.sigma_data = sigma_data
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        self.c_s = c_s
        self.f_pred = f_pred

        self.diffusion_conditioning = DiffusionConditioning(sigma_data=sigma_data, c_s=c_s, c_z=c_z, **diffusion_conditioning)
        self.atom_attention_encoder = AtomAttentionEncoder(c_token=c_token, c_s=c_s, c_atom=c_atom, c_atompair=c_atompair, **atom_attention_encoder)
        self.process_s = nn.Sequential(
            nn.LayerNorm((c_s,)),
            linearNoBias(c_s, c_token),

        )
        self.diffusion_transformer = DiffusionTransformer(c_token=c_token, c_s=c_s, c_tokenpair=c_z, **diffusion_transformer)
        self.layer_norm_1 = nn.LayerNorm(c_token)
        self.atom_attention_decoder = AtomAttentionDecoder(c_token=c_token, c_atom=c_atom, c_atompair=c_atompair, **atom_attention_decoder)
        
    def forward(self,
                X_noisy_L, # [B, L, 3]
                t, # [B] (0 is ground truth)
                f, # Dict (Input feature dictionary)
                S_inputs_I, # [B, I, C_S_input]
                S_trunk_I, # [B, I, C_S_trunk]
                Z_trunk_II, # [B, I, I, C_Z]
    ):
        # Conditioning
        S_I, Z_II = self.diffusion_conditioning(t, f, S_inputs_I, S_trunk_I, Z_trunk_II)

        # Scale positions to dimensionless vectors with approximately unit variance
        if self.f_pred == 'edm':
            R_noisy_L = X_noisy_L / torch.sqrt(t[...,None,None]**2 + self.sigma_data**2)
        elif self.f_pred == 'unconditioned':
            R_noisy_L = torch.zeros_like(X_noisy_L)
        elif self.f_pred == 'noise_pred':
            R_noisy_L = X_noisy_L
        else:
            raise Exception(f'{self.f_pred=} unrecognized')

        # Sequence-local Atom Attention and aggregation to coarse-grained tokens
        A_I, Q_skip_L, C_skip_L, P_skip_LL = self.atom_attention_encoder(f, R_noisy_L, S_trunk_I, Z_II)

        # Full self-attention on token level
        A_I = A_I + self.process_s(S_I)
        A_I = self.diffusion_transformer(A_I, S_I, Z_II, Beta_II=torch.tensor(0.0, device=Z_II.device))
        A_I = self.layer_norm_1(A_I)

        # Broadcast token activations to atoms and run Sequence-local Atom Attention
        R_update_L = self.atom_attention_decoder(f, A_I, Q_skip_L, C_skip_L, P_skip_LL)

        # Rescale updates to positions and combine with input positions
        if self.f_pred == 'edm':
            X_out_L = (
                (self.sigma_data**2 / (self.sigma_data**2 + t**2))[...,None,None] * X_noisy_L +
                (self.sigma_data * t / (self.sigma_data**2 + t**2) ** 0.5)[...,None,None] * R_update_L
            )
        elif self.f_pred == 'unconditioned':
            X_out_L = R_update_L
        elif self.f_pred == 'noise_pred':
            X_out_L = X_noisy_L + R_update_L
        else:
            raise Exception(f'{self.f_pred=} unrecognized')

        return X_out_L

class DiffusionConditioning(nn.Module):
    def __init__(self, sigma_data, c_z, c_s, c_s_inputs, c_t_embed, relative_position_encoding):
        super().__init__()
        self.sigma_data = sigma_data
        self.relative_position_encoding = RelativePositionEncoding(c_z=c_z, **relative_position_encoding)
        self.to_zii = nn.Sequential(
            nn.LayerNorm(c_z * 2), # Operates on concatenated (z_ij_trunk: [..., c_z]), RelativePositionalEncoding: [..., c_z])
            linearNoBias(c_z * 2, c_z)
        )
        self.transition_1 = nn.ModuleList([
            Transition(c=c_z, n=2),
            Transition(c=c_z, n=2),
        ])
        self.to_si = nn.Sequential(
            nn.LayerNorm(c_s+c_s_inputs),
            linearNoBias(c_s+c_s_inputs, c_s)
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
        Z_II = torch.cat([Z_trunk_II, self.relative_position_encoding(f)], dim=-1)
        Z_II = self.to_zii(Z_II)
        for b in range(2):
            Z_II = Z_II + self.transition_1[b](Z_II)
        
        # Single conditioning
        S_I = torch.cat([S_trunk_I, S_inputs_I], dim=-1)
        S_I = self.to_si(S_I)
        N_D = self.fourier_embedding(1/4 * torch.log(t/self.sigma_data))
        S_I = self.process_n(N_D).unsqueeze(-2) + S_I
        for b in range(2):
            S_I = S_I + self.transition_2[b](S_I)
        
        return S_I, Z_II


class Transition(nn.Module):
    def __init__(self, n, c):
        super().__init__()
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
        X = self.linear_3(silu(A) * B)
        return X

pi = torch.acos(torch.zeros(1)).item() * 2
class FourierEmbedding(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.register_buffer('w', torch.zeros(c, dtype=torch.float32))
        self.register_buffer('b', torch.zeros(c, dtype=torch.float32))
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        # super().reset_parameters()
        nn.init.normal_(self.w)
        nn.init.normal_(self.b)
    
    def forward(self,
                t, # [D]
                ):
        return torch.cos(2 * pi * (t[:, None]*self.w + self.b))

class Model(nn.Module):
    def __init__(self,
                 c_s,
                 c_z,
                 c_atom,
                 c_atompair,
                 feature_initializer,
                 recycler,
                 diffusion_module,
                 distogram_head,
                 **kwargs
                 ):
        super().__init__()
        self.feature_initializer = FeatureInitializer(c_s=c_s, c_z=c_z, c_atom=c_atom, c_atompair=c_atompair, **feature_initializer)
        self.recycler = Recycler(c_s=c_s, c_z=c_z, **recycler)
        self.diffusion_module = DiffusionModule(c_atom=c_atom, c_atompair=c_atompair, c_s=c_s, c_z=c_z, **diffusion_module)
        self.distogram_head = DistogramHead(c_z=c_z, **distogram_head) 

    def forward(self, input, n_cycle, no_sync):
        '''
        Runs recycling with gradients only on final recycle.

        Assums model has methods:
            pre_recycle: input --> recycling_input
            recycle: recycling_input --> recycling_input
            post_recycle: recycling_input --> output
        '''
        recycling_input = self.pre_recycle(**input)
        for i_cycle in range(n_cycle):
                with ExitStack() as stack:
                    if i_cycle < n_cycle -1:
                        stack.enter_context(torch.no_grad())
                        stack.enter_context(no_sync())
                    recycling_input = self.recycle(**recycling_input)
        return self.post_recycle(**recycling_input)

    
    def pre_recycle(self,
                    f,
                    X_noisy_L,
                    t):
        S_inputs_I, S_init_I, Z_init_II = self.feature_initializer(f)
        S_I = torch.zeros_like(S_init_I)
        Z_II = torch.zeros_like(Z_init_II)
        return dict(
            S_inputs_I=S_inputs_I,
            S_init_I=S_init_I,
            Z_init_II=Z_init_II,
            S_I=S_I,
            Z_II=Z_II,
            f=f,
            X_noisy_L=X_noisy_L,
            t=t
        )
    
    def recycle(self,
                S_inputs_I,
                S_init_I,
                Z_init_II,
                S_I,
                Z_II,
                f,
                X_noisy_L,
                t,
                ):
        S_I, Z_II = self.recycler(
            f=f,
            S_inputs_I=S_inputs_I,
            S_init_I=S_init_I,
            Z_init_II=Z_init_II,
            S_I=S_I,
            Z_II=Z_II,
        )
        return dict(
            S_inputs_I=S_inputs_I,
            S_init_I=S_init_I,
            Z_init_II=Z_init_II,
            S_I=S_I,
            Z_II=Z_II,
            f=f,
            X_noisy_L=X_noisy_L,
            t=t
        )
    
    def post_recycle(self,
                     S_inputs_I,
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
            S_inputs_I, 
            S_I,
            Z_II,
        )
        distogram_pred = self.distogram_head(Z_II)
        return {
            "X_L": X_pred,
            "distogram": distogram_pred,
        }
        
class DistogramHead(nn.Module):
    def __init__(self,
                 c_z,
                 bins,
                 ):
        super().__init__()
        self.predictor = nn.Linear(c_z, bins) 
    
    def reset_parameters(self):
        # initialize linear layer for final logit prediction
        nn.init.zeros_(self.predictor.weight)
        nn.init.zeros_(self.predictor.bias)

    def forward(self,
                Z_II,
                ):
        return self.predictor(
            Z_II+Z_II.transpose(-2,-3) # symmetrize pair features
            )


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
        self.c_z = c_z
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
        self.pairformer_stack = nn.ModuleList([
            PairformerBlock(c_s=c_s, c_z=c_z, **pairformer_block) for _ in range(n_pairformer_blocks)
        ])

    def forward(self,
                f,
                S_inputs_I,
                S_init_I,
                Z_init_II,
                S_I,
                Z_II,
                ):
        Z_II = Z_init_II + self.process_zh(Z_II)
        Z_II = Z_II + self.template_embedder(f, Z_II)
        Z_II = Z_II + self.msa_module(f, Z_II, S_inputs_I)
        S_I = S_init_I + self.process_sh(S_I)
        for block in self.pairformer_stack:
            S_I, Z_II = block(S_I, Z_II)
        return S_I, Z_II
    
def create_batch_dimension_if_not_present(batched_n_dim):
    """
    Decorator for adapting a function which expects batched arguments with ndim `batched_n_dim` also
    accept unbatched arguments.
    """
    def wrap(f):
        def _wrap(arg):
            inserted_batch_dim = False
            if arg.ndim == batched_n_dim - 1:
                arg = arg[None]
                inserted_batch_dim = True
            elif arg.ndim == batched_n_dim:
                pass
            else:
                raise Exception(f'arg must have {batched_n_dim-1} or {batched_n_dim} dimensions, got shape {arg.shape=}')
            o = f(arg)

            if inserted_batch_dim:
                assert o.shape[0] == 1, f'{o.shape=}[0] != 1'
                return o[0]
            return o
        return _wrap
    return wrap

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

        self.drop_row = Dropout(broadcast_dim=-2, p_drop=p_drop)
        self.drop_col = Dropout(broadcast_dim=-3, p_drop=p_drop)

        self.tri_mul_outgoing = TriangleMultiplication(
            c_z, d_hidden=c, outgoing=True, bias=False)
        self.tri_mul_incoming = TriangleMultiplication(
            c_z, d_hidden=c, outgoing=False, bias=False)
        self.tri_attn_start = TriangleAttention(
            c_z, d_hidden=c, start_node=True)
        self.tri_attn_end = TriangleAttention(
            c_z, d_hidden=c, start_node=False)

        self.z_transition = Transition(c=c_z, n=n_transition)
        
        if c_s > 0:
            self.s_transition = Transition(c=c_s, n=n_transition)

            self.attention_pair_bias = AttentionPairBias(c_a=c_s, c_s=0, c_pair=c_z, **attention_pair_bias)
        triangle_operations_expected_dim = 4 # B, L, L, C
        self.maybe_make_batched = create_batch_dimension_if_not_present(triangle_operations_expected_dim)

    @activation_checkpointing
    def forward(self,
                S_I,
                Z_II):
        Z_II = Z_II + self.drop_row(self.maybe_make_batched(self.tri_mul_outgoing)(Z_II))
        Z_II = Z_II + self.drop_row(self.maybe_make_batched(self.tri_mul_incoming)(Z_II))
        Z_II = Z_II + self.drop_row(self.maybe_make_batched(self.tri_attn_start)(Z_II))
        Z_II = Z_II + self.drop_col(self.maybe_make_batched(self.tri_attn_end)(Z_II))
        Z_II = Z_II + self.z_transition(Z_II)
        if S_I is not None:
            S_I = S_I + self.attention_pair_bias(S_I, None, Z_II, Beta_II=torch.tensor([0.], device=Z_II.device))
            S_I = S_I + self.s_transition(S_I)
        return S_I, Z_II

class PairformerBlock_batched(nn.Module):
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

        self.drop_row = Dropout(broadcast_dim=-2, p_drop=p_drop)
        self.drop_col = Dropout(broadcast_dim=-3, p_drop=p_drop)

        self.tri_mul_outgoing = TriangleMultiplication(
            c_z, d_hidden=c, outgoing=True, bias=False)
        self.tri_mul_incoming = TriangleMultiplication(
            c_z, d_hidden=c, outgoing=False, bias=False)
        self.tri_attn_start = TriangleAttention(
            c_z, d_hidden=c, start_node=True)
        self.tri_attn_end = TriangleAttention(
            c_z, d_hidden=c, start_node=False)

        self.z_transition = Transition(c=c_z, n=n_transition)
        
        if c_s > 0:
            self.s_transition = Transition(c=c_s, n=n_transition)
            self.attention_pair_bias = AttentionPairBias(c_a=c_s, c_s=0, c_pair=c_z, **attention_pair_bias)
        triangle_operations_expected_dim = 4 # B, L, L, C
        self.maybe_make_batched = create_batch_dimension_if_not_present(triangle_operations_expected_dim)

    @activation_checkpointing
    def forward(self,
                S_I,
                Z_II):
        if len(Z_II.shape) == 3:
            Z_II = Z_II[None]
        Z_II = Z_II + self.drop_row(self.tri_mul_outgoing(Z_II))
        Z_II = Z_II + self.drop_row(self.tri_mul_incoming(Z_II))
        Z_II = Z_II + self.drop_row(self.tri_attn_start(Z_II))
        Z_II = Z_II + self.drop_col(self.tri_attn_end(Z_II))

        if len(Z_II.shape) == 4:
            Z_II = Z_II[0]

        Z_II = Z_II + self.z_transition(Z_II)
        if S_I is not None:
            S_I = S_I + self.attention_pair_bias(S_I, None, Z_II, Beta_II=torch.tensor([0.], device=Z_II.device))
            S_I = S_I + self.s_transition(S_I)
        return S_I, Z_II


class FeatureInitializer(nn.Module):
    def __init__(self,
                 c_s,
                 c_z,
                 c_atom,
                 c_atompair,
                 c_s_inputs,
                 input_feature_embedder,
                 relative_position_encoding):
        super().__init__()
        self.input_feature_embedder = InputFeatureEmbedder(c_atom=c_atom, c_atompair=c_atompair, **input_feature_embedder)
        self.to_s_init = linearNoBias(c_s_inputs, c_s)
        self.to_z_init_i = linearNoBias(c_s_inputs, c_z)
        self.to_z_init_j = linearNoBias(c_s_inputs, c_z)
        self.relative_position_encoding = RelativePositionEncoding(c_z=c_z, **relative_position_encoding)
        self.process_token_bonds = linearNoBias(1, c_z)
    
    def forward(self,
                f,
                ):
        S_inputs_I = self.input_feature_embedder(f)
        S_init_I = self.to_s_init(S_inputs_I)
        Z_init_II = self.to_z_init_i(S_inputs_I).unsqueeze(-3) + self.to_z_init_j(S_inputs_I).unsqueeze(-2)
        Z_init_II = Z_init_II + self.relative_position_encoding(f)
        Z_init_II = Z_init_II + self.process_token_bonds(f['token_bonds'].unsqueeze(-1).to(torch.float))
        return S_inputs_I, S_init_I, Z_init_II


class InputFeatureEmbedder(nn.Module):
    def __init__(self,
                 features, 
                 c_atom,
                 c_atompair,
                 atom_attention_encoder):
        super().__init__()
        self.atom_attention_encoder = AtomAttentionEncoder(c_atom=c_atom, c_atompair=c_atompair, c_s=0, **atom_attention_encoder)
        self.features = features
        self.features_to_unsqueeze = ['deletion_mean']
    
    def forward(self,
                f,
                ):
        A_I, _, _, _ = self.atom_attention_encoder(
            f, None, None, None
        )
        S_I = torch.cat([A_I] + [f[feature].unsqueeze(-1) if feature in self.features_to_unsqueeze else f[feature] for feature in self.features], dim=-1)
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
        return self.linear(torch.cat([A_relpos_II, A_reltoken_II, b_same_entity_II.unsqueeze(-1), A_relchain_II], dim=-1).to(torch.float))


class MSAModule(nn.Module):
    def __init__(self, n_block, 
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
        self.msa_subsampler = MsaSubsampleEmbedder(**msa_subsample_embedder)
        self.outer_product = OuterProductMean_AF3(**outer_product)
        self.msa_pair_weighted_averaging = MsaPairWeightedAverage(**msa_pair_weighted_averaging)
        self.msa_transition = Transition(**msa_transition)

        self.drop_row_msa = Dropout(broadcast_dim=-2, p_drop=p_drop_msa)
        self.drop_row_pair = Dropout(broadcast_dim=-2, p_drop=p_drop_pair)
        self.drop_col_pair = Dropout(broadcast_dim=-3, p_drop=p_drop_pair)        
        
        self.tri_mult_outgoing = TriangleMultiplication(**triangle_multiplication_outgoing, outgoing=True)
        self.tri_mult_incoming = TriangleMultiplication(**triangle_multiplication_incoming, outgoing=False)
        self.tri_attn_start = TriangleAttention(**triangle_attention_starting, start_node=True)
        self.tri_attn_end = TriangleAttention(**triangle_attention_ending, start_node=False)
        self.pair_transition = Transition(**pair_transition)

        outer_product_expected_dim = 4 # B, S, I, C
        self.maybe_make_batched_outer_product = create_batch_dimension_if_not_present(outer_product_expected_dim)
        
        triangle_ops_expected_dim = 4 # B, I, I, C
        self.maybe_make_batched_triangle_ops = create_batch_dimension_if_not_present(triangle_ops_expected_dim)

    @activation_checkpointing
    def forward(self,
                f,
                Z_II,
                S_inputs_I,
                ):
        msa = torch.cat([f["msa"], f["has_deletion"][...,None], f["deletion_value"][...,None]], dim=-1)
        msa_SI = self.msa_subsampler(msa, S_inputs_I)
        for i in range(self.n_block):
            # update MSA features
            Z_II = Z_II + self.maybe_make_batched_outer_product(self.outer_product)(msa_SI)
            msa_SI = msa_SI + self.drop_row_msa(self.msa_pair_weighted_averaging(msa_SI, Z_II))
            msa_SI = msa_SI + self.msa_transition(msa_SI)

            # update pair features
            Z_II = Z_II + self.drop_row_pair(self.maybe_make_batched_triangle_ops(self.tri_mult_outgoing)(Z_II))
            Z_II = Z_II + self.drop_row_pair(self.maybe_make_batched_triangle_ops(self.tri_mult_incoming)(Z_II))
            Z_II = Z_II + self.drop_row_pair(self.maybe_make_batched_triangle_ops(self.tri_attn_start)(Z_II))
            Z_II = Z_II + self.drop_col_pair(self.maybe_make_batched_triangle_ops(self.tri_attn_end)(Z_II))
            Z_II = Z_II + self.pair_transition(Z_II)
        return Z_II

class TemplateEmbedder(nn.Module):
    def __init__(self,
                 n_block,
                 raw_template_dim,
                 c_z,
                 c
                 ):
        super().__init__()
        self.c =c
        self.emb_pair = nn.Linear(c_z, c, bias=False)
        self.norm_pair_before_pairformer = nn.LayerNorm(c_z)
        self.norm_after_pairformer = nn.LayerNorm(c)
        self.emb_templ = nn.Linear(raw_template_dim, c, bias=False)

        # template pairformer does not operate on sequence representation
        self.pairformer = nn.ModuleList(
            [
                PairformerBlock(c_s=0, c_z=c, p_drop=0.0, c=c, attention_pair_bias={}, n_transition=4)
                for _ in range(n_block)
             ])

        # NOTE: this is not consistent with AF3 paper which outputs this tensor in the template_channel dimension
        # In Algorithm 1, line 9, the outputs of this function are added to the Z_II tensor which has dimensions [B, I, I, C_z]
        # so we make the outputs of this module also has those dimensions 
        self.agg_emb = nn.Linear(c, c_z, bias=False)


    def forward(self,
                f,
                Z_II,
                ):
        I = Z_II.shape[0]
        template_frame_mask = f["template_backbone_frame_mask"][:, None] * f["template_backbone_frame_mask"][:, :, None]   
        template_pseudo_beta_mask = f["template_pseudo_beta_mask"][:, None, :] * f["template_pseudo_beta_mask"][:, :, None]

        template_feats = torch.cat([f["template_distogram"], template_frame_mask[..., None], f["template_unit_vector"], template_pseudo_beta_mask[...,None]], dim=-1)
        template_feats = template_feats * (f["asym_id"][None, :] == f["asym_id"][:, None])[..., None]
        template_feats = torch.cat([template_feats, f["template_restype"][:, None, :, :].repeat(1, I, 1,1)], dim=-1)
        T = template_feats.shape[0]
        u_II = torch.zeros(I, I, self.c, device=Z_II.device)
        for i in range(T):
            v_II = self.emb_pair(self.norm_pair_before_pairformer(Z_II)) + self.emb_templ(template_feats[i])
            for block in self.pairformer:
                _, v_II = block(None, v_II)
            u_II = u_II + self.norm_after_pairformer(v_II)
        u_II = u_II / T

        return self.agg_emb(relu(u_II))

def calc_smoothed_lddt_loss(X_gt_L, X_L, crd_mask_I, seq, tok_idx, is_dna, is_rna):
    """
    compute smoothed lddt loss from AF3 paper
    """
    # compute distances between ground truth atoms
    ground_truth_distances = torch.cdist(X_gt_L,X_gt_L)
    # compute distances between predicted atoms
    predicted_distances = torch.cdist(X_L, X_L)
    # compute LDDT score for each pair of distances
    difference_distances = torch.abs(ground_truth_distances - predicted_distances)
    lddt_matrix = torch.zeros_like(difference_distances)
    lddt_matrix = 0.25 * torch.sigmoid(4.0 - difference_distances) + \
                    0.25 * torch.sigmoid(2.0 - difference_distances) + \
                    0.25 * torch.sigmoid(1.0 - difference_distances) + \
                    0.25 * torch.sigmoid(0.5 - difference_distances) 
    # remove unresolved atoms, atoms within same residue
    is_real_atom = ChemData().heavyatom_mask.to(seq.device)[seq]
    is_resolved_atom_L = crd_mask_I[is_real_atom]
    is_unresolved_distance_LL = is_resolved_atom_L[...,None] & is_resolved_atom_L[None,...]
    in_same_residue_LL = tok_idx[:,None] == tok_idx[None,:]

    is_na_L = is_dna[tok_idx] | is_rna[tok_idx]
    is_close_distance = (ground_truth_distances < 30) * is_na_L + (ground_truth_distances < 10) * ~is_na_L
    mask = is_unresolved_distance_LL & ~in_same_residue_LL & is_close_distance[0]
    lddt = torch.div(lddt_matrix[:, mask].sum(dim=(-1)), mask.sum(dim=(-1,-2)))
    return 1 - lddt

class DiffusionLoss:
    def __init__(self,
                 weight,
                 sigma_data,
                 alpha_dna,
                 alpha_rna,
                 alpha_ligand,
                 edm_lambda, # Use EDM-style loss weighting
                 se3_invariant_loss,
                 ):
        self.sigma_data = sigma_data
        self.alpha_dna = alpha_dna
        self.alpha_rna = alpha_rna
        self.alpha_ligand = alpha_ligand
        self.se3_invariant_loss = se3_invariant_loss
        self.weight = weight
        
        # AF3-style loss weighting
        self.get_lambda = lambda sigma: (sigma**2 + self.sigma_data**2) / (sigma + self.sigma_data)**2
        if edm_lambda:
            # Use EDM-style loss weighting
            self.get_lambda = lambda sigma:  (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data)**2
    
    def __call__(self,
                 f,
                 X_L, # [D, L, 3]
                 X_gt_L, # [D, L, 3]
                 t, # [D]
                 seq, # [I],
                 crd_mask_I, # [I]  # Mask for resolved atoms
                 ):
        D = X_L.shape[0]
        w_L = 1 + (
            f['is_dna']*self.alpha_dna +
            f['is_rna'] * self.alpha_rna + 
            f['is_ligand'] * self.alpha_ligand
        )[f['tok_idx']].to(torch.float)
        
        is_resolved_atom_L = convert_residue_mask_to_allatom_mask(crd_mask_I, seq)
        w_L = w_L * is_resolved_atom_L  
        # Align ground truth onto predictions.
        if self.se3_invariant_loss:
            X_gt_aligned_L = weighted_rigid_align(X_gt_L, X_L, w_L.tile(D, 1))
        else:
            X_gt_aligned_L = X_gt_L
        l_mse = 1/3 * torch.div(torch.sum(w_L * torch.sum((X_L-X_gt_aligned_L)**2, dim=-1), dim=-1), torch.sum(is_resolved_atom_L)) # [D]

        assert l_mse.shape == (D,)
        l_diffusion = self.get_lambda(t) * l_mse
        
        smoothed_lddt_loss = calc_smoothed_lddt_loss(X_gt_L, X_L, crd_mask_I, seq, f['tok_idx'], f['is_dna'], f['is_rna'])
        loss_dict_batched = {
            'diffusion_loss': l_diffusion,
            'smoothed_lddt_loss': smoothed_lddt_loss,
        }

        # TODO: implement auxiliary losses

        loss_dict = {k:v.mean() for k,v in loss_dict_batched.items()}
        l_total = sum(loss_dict.values())
        loss_dict_batched['total_diffusion_loss'] = l_total
        loss_dict_batched = {k: v.detach() for k,v in loss_dict_batched.items()}
        return self.weight*l_total, loss_dict_batched

class DistogramLoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.cce_loss = nn.CrossEntropyLoss(reduction='none')
        self.weight = weight
        self.eps = 1e-4

    def __call__(
        self,
        distogram_pred, # [I, I, 37]
        X_gt_L, # [D, L, 3]     
        crd_mask_I, # [I]
        seq,
        f 
    ):
        # Convert to I, 36
        I = seq.shape[0]
        is_real_atom = ChemData().heavyatom_mask.to(X_gt_L.device)[seq]
        X_gt_I = torch.zeros((seq.shape[0], ChemData().NTOTAL, 3), device=X_gt_L.device)
        X_gt_I[is_real_atom] = X_gt_L[0]
        #    cbeta for all protein residues except glycine
        #    calpha for glycine
        #    c4 for purines 
        #   c2 for pyrimidines
        seq_is_protein = f["is_protein"].to(torch.bool)
        use_cbeta = seq_is_protein & (seq != ChemData().aa2num["GLY"])
        use_calpha = seq_is_protein & (seq == ChemData().aa2num["GLY"])
        use_c4 = (seq == ChemData().aa2num[" DA"]) | (seq == ChemData().aa2num[" DG"]) | (seq == ChemData().aa2num[" RA"]) | (seq == ChemData().aa2num[" RG"])
        use_c2 = (seq == ChemData().aa2num[" DC"]) | (seq == ChemData().aa2num[" DT"]) | (seq == ChemData().aa2num[" RC"]) | (seq == ChemData().aa2num[" RU"])
        idx_to_use = torch.ones_like(seq) 
        idx_to_use[use_cbeta] = 5 # cbeta
        idx_to_use[use_calpha] = 1 # calpha
        idx_to_use[use_c4] = 8 # c4
        idx_to_use[use_c2] = 2 # c2

        dist_node = X_gt_I[torch.arange(seq.shape[0]), idx_to_use]
        crd_mask_I = crd_mask_I[torch.arange(seq.shape[0]), idx_to_use]

        crd_mask_II = crd_mask_I.unsqueeze(-1) * crd_mask_I.unsqueeze(-2)
        dist = torch.cdist(dist_node, dist_node)
        from rf2aa.data.dataloader_adaptor_af3 import discretize_distance_matrix
        distogram_target = discretize_distance_matrix(dist, num_bins=36)
        cce_loss =  self.cce_loss(distogram_pred.reshape(I*I, 37), distogram_target.reshape(I*I)).reshape(I, I)
        cce_loss = torch.sum(cce_loss[crd_mask_II])/(torch.sum(crd_mask_II) + self.eps)
        loss_dict = {"distogram_loss": cce_loss.detach()}
        return self.weight * cce_loss, loss_dict

class Loss(nn.Module):
    def __init__(self,
                 diffusion_loss,
                 distogram_loss,
                 ):
        super().__init__()
        self.diffusion_loss = DiffusionLoss(**diffusion_loss)
        self.distogram_loss = DistogramLoss(**distogram_loss)

    def forward(self,
                network_input,
                network_output,
                loss_input,
                ):
        loss_dict = {}
        diffusion_loss, diffusion_loss_dict = self.diffusion_loss(
                                            network_input["f"], 
                                             network_output["X_L"], 
                                             loss_input["X_gt_L"], 
                                             network_input["t"], 
                                             loss_input["seq"], 
                                             loss_input["crd_mask_I"]
                                             )
        
        distogram_loss, distogram_loss_dict = self.distogram_loss(
                                            network_output["distogram"], 
                                             loss_input["X_gt_L"], 
                                             loss_input["crd_mask_I"],
                                             loss_input["seq"],
                                             network_input["f"]
                                             )
        loss_dict.update(diffusion_loss_dict)
        loss_dict.update(distogram_loss_dict)
        return diffusion_loss + distogram_loss, loss_dict 
                

def convert_residue_mask_to_allatom_mask(crd_mask_I, seq):
    """
    Converts a residue mask to an atom mask. The atom mask is True if any atom in the residue is True.
    """
    is_real_atom = ChemData().heavyatom_mask.to(seq.device)[seq]
    is_resolved_atom_L = crd_mask_I[is_real_atom]
    return is_resolved_atom_L

