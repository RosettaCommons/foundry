import logging

import torch
import torch.nn as nn

from modelhub.common import exists
from modelhub.model.AF3_structure import (
    DiffusionTransformer,
)
from modelhub.model.layers.layer_utils import linearNoBias
from projects.aa_design.model.blocks import (
    AtomTransformerWrapper,
    SequenceHead,
)
from projects.aa_design.model.encoders import (
    DiffusionAtomEncoder,
    DiffusionTokenEncoder,
)

logger = logging.getLogger(__name__)


class BaseDiffusionModule(nn.Module):
    '''
    Diffusion Module Class
    '''
    def __init__(
        self,
        *,
        sigma_data,
        c_atom,
        c_atompair,
        c_token,
        c_s,
        c_z,
        f_pred,
        # Modules
        diffusion_token_encoder,
        diffusion_atom_encoder,

        **kwargs
    ):
        super().__init__()
        self.sigma_data = sigma_data
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        self.c_s = c_s
        self.f_pred = f_pred

        self.diffusion_token_encoder = DiffusionTokenEncoder(
            c_s=c_s, c_z=c_z, c_token=c_token,
            **diffusion_token_encoder
        )
        self.diffusion_atom_encoder = DiffusionAtomEncoder(
            c_s=c_s, c_token=c_token, c_atom=c_atom, c_atompair=c_atompair, c_tokenpair=c_z, 
            **diffusion_atom_encoder
        )
        self.to_r_update = nn.Sequential(
            nn.LayerNorm((c_atom,)), linearNoBias(c_atom, 3)
        )
        self.sequence_head = SequenceHead(c_token=c_token)

        self._post_init_(
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_token=c_token,
            c_s=c_s,
            c_z=c_z,
            **kwargs
        )

    def scale_positions_in(self, X_noisy_L, t):
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
        return R_noisy_L

    def scale_positions_out(self, R_update_L, X_noisy_L, t):
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

    def forward(
        self,
        X_noisy_L,  # [B, L, 3]
        t,  # [B] (0 is ground truth)
        f,  # Dict (Input feature dictionary)
        S_init_I,
        Z_init_II,
    ):
        # Scale positions to dimensionless vectors with approximately unit variance
        R_noisy_L = self.scale_positions_in(X_noisy_L, t)

        #################################
        # Embed tokens and atoms
        # Convetion: init features are batchless and otherwise have a batch dim.
        A_I, S_I, Z_II = self.diffusion_token_encoder(f, R_L=R_noisy_L, t=t, S_init_I=S_init_I, Z_init_II=Z_init_II)

        # Z_trunk_I passed as coarse (batchless) features to atom level encoding
        # Provides token position embedding, residue type and token bond features.
        Q_L, C_L, P_LL = self.diffusion_atom_encoder(f, R_L=R_noisy_L, S_I=S_init_I, Z_II=Z_init_II) 

        #################################
        # U-net or similar architecture
        A_I, Q_L = self.process(A_I, S_I, Z_II, Q_L, C_L, P_LL, f=f)

        #################################
        # Map to positions update
        R_update_L = self.to_r_update(Q_L)

        # Rescale updates to positions and combine with input positions
        X_out_L = self.scale_positions_out(R_update_L, X_noisy_L, t)

        # Map embeddings to sequence
        Seq_I = self.sequence_head(A_I, Q_L, X_out_L, f)

        return {
            'X_L': X_out_L,  # [B, L, 3] denoised positions
            'Seq_I': Seq_I,  # [B, I, 32] sequence predictions
        }

    def _post_init_(self, *,
        c_atom, c_atompair, c_token, c_s, c_z, c_s_inputs, **kwargs
    ):
        raise NotImplementedError

    def process(self, A_I, S_I, Z_II, Q_L, C_L, P_LL, f):
        raise NotImplementedError

class UNetDiffusionModule(BaseDiffusionModule):
    '''
    Diffusion Module Class
    '''
    def _post_init_(self, *,
        c_atom, c_atompair, c_token, c_s, c_z,
        atom_attention_encoder, atom_attention_decoder, diffusion_transformer, **_
    ):
        self.atom_attention_encoder = AtomTransformerWrapper(
            c_atom=c_atom,
            c_atompair=c_atompair,
            atom_transformer=atom_attention_encoder.atom_transformer,
        )
        self.process_q = nn.Sequential(
            linearNoBias(c_atom, c_token),
            nn.ReLU(),
        )

        self.diffusion_transformer = DiffusionTransformer(
            c_token=c_token, c_s=c_s, c_tokenpair=c_z, **diffusion_transformer
        )
        self.layer_norm_1 = nn.LayerNorm(c_token)
        self.process_s = nn.Sequential(
            nn.LayerNorm((c_s,)),
            linearNoBias(c_s, c_token),
        )
        self.linear_1 = linearNoBias(c_token, c_atom)
        self.atom_attention_decoder = AtomTransformerWrapper(
            c_atom=c_atom,
            c_atompair=c_atompair,
            atom_transformer=atom_attention_decoder.atom_transformer,
        )

        self.a_init_to_s  = linearNoBias(c_token, c_s)
        self.a_init_to_z_i = linearNoBias(c_token, c_z)
        self.a_init_to_z_j = linearNoBias(c_token, c_z)

    def process(self, A_I, S_I, Z_II, Q_L, C_L, P_LL, f):
        tok_idx = f["atom_to_token_map"]
        I = tok_idx.max() + 1

        # Sequence-local Atom Attention and aggregation to coarse-grained tokens
        Q_L = self.atom_attention_encoder(Q_L, C_L, P_LL)

        # Aggregate per-atom representation to per-token representation
        A_I_shape = Q_L.shape[:-2] + (I, self.c_token,)
        A_I_update = (torch.zeros(A_I_shape, device=Q_L.device, dtype=Q_L.dtype)
            .index_reduce(
                -2, tok_idx.long(),
                self.process_q(Q_L),
                "mean", include_self=False,
            ).clone()
        )
        if exists(A_I):
            A_I = A_I + A_I_update
        else:
            A_I = A_I_update

        if len(Z_II.shape) != 4:
            Z_II = Z_II[None]

        S_I = S_I   + self.a_init_to_s(A_I_update)
        Z_II = Z_II + self.a_init_to_z_i(A_I_update).unsqueeze(-3) \
                    + self.a_init_to_z_j(A_I_update).unsqueeze(-2)

        # Full self-attention on token level
        A_I = A_I + self.process_s(S_I)
        A_I = self.diffusion_transformer(A_I, S_I, Z_II, Beta_II=None)
        A_I = self.layer_norm_1(A_I)

        # Broadcast per-token activiations to per-atom activations and add the skip connection
        Q_L = self.linear_1(A_I[..., tok_idx, :]) + Q_L

        # Broadcast token activations to atoms and run Sequence-local Atom Attention
        Q_L = self.atom_attention_decoder(Q_L, C_L, P_LL)

        return A_I, Q_L


