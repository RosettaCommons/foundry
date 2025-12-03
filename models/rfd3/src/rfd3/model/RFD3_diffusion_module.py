import functools
import logging
import os
from contextlib import ExitStack

import torch
import torch.nn as nn
from rfd3.model.layers.block_utils import (
    bucketize_scaled_distogram,
    create_attention_indices,
)
from rfd3.model.layers.blocks import (
    CompactStreamingDecoder,
    Downcast,
    LinearEmbedWithPool,
    LinearSequenceHead,
    LocalAtomTransformer,
    LocalTokenTransformer,
)
from rfd3.model.layers.encoders import (
    DiffusionTokenEncoder,
)
from rfd3.model.layers.layer_utils import RMSNorm, linearNoBias

from foundry.model.layers.blocks import (
    FourierEmbedding,
)

logger = logging.getLogger(__name__)


class RFD3DiffusionModule(nn.Module):
    def __init__(
        self,
        *,
        c_atom,
        c_atompair,
        c_token,
        c_s,
        c_z,
        c_t_embed,
        sigma_data,
        f_pred,
        n_attn_seq_neighbours,
        n_attn_keys,
        n_recycle,
        atom_attention_encoder,
        diffusion_token_encoder,
        diffusion_transformer,
        atom_attention_decoder,
        # upcast,
        downcast,
        use_local_token_attention=True,
        **_,
    ):
        super().__init__()
        self.sigma_data = sigma_data
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        self.c_s = c_s
        self.c_z = c_z
        self.f_pred = f_pred
        self.n_attn_seq_neighbours = n_attn_seq_neighbours
        self.n_attn_keys = n_attn_keys
        self.use_local_token_attention = use_local_token_attention

        # Auxiliary
        self.process_r = linearNoBias(3, c_atom)
        self.to_r_update = nn.Sequential(RMSNorm((c_atom,)), linearNoBias(c_atom, 3))
        self.sequence_head = LinearSequenceHead(c_token=c_token)

        self.n_recycle = n_recycle
        self.n_bins = 65
        self.bucketize_fn = functools.partial(
            bucketize_scaled_distogram,
            min_dist=1,
            max_dist=30,
            sigma_data=1,
            n_bins=self.n_bins,
        )

        # Time processing
        self.fourier_embedding = nn.ModuleList(
            [FourierEmbedding(c_t_embed), FourierEmbedding(c_t_embed)]
        )
        self.process_n = nn.ModuleList(
            [
                nn.Sequential(RMSNorm(c_t_embed), linearNoBias(c_t_embed, c_atom)),
                nn.Sequential(RMSNorm(c_t_embed), linearNoBias(c_t_embed, c_s)),
            ]
        )
        self.downcast_c = Downcast(c_atom=c_atom, c_token=c_s, c_s=None, **downcast)
        self.downcast_q = Downcast(c_atom=c_atom, c_token=c_token, c_s=c_s, **downcast)
        self.process_a = LinearEmbedWithPool(c_token)
        self.process_c = nn.Sequential(RMSNorm(c_atom), linearNoBias(c_atom, c_atom))

        # UNet-like architecture for processing across tokens and atoms
        self.encoder = LocalAtomTransformer(
            c_atom=c_atom, c_s=c_atom, c_atompair=c_atompair, **atom_attention_encoder
        )

        self.diffusion_token_encoder = DiffusionTokenEncoder(
            c_s=c_s,
            c_token=c_token,
            c_z=c_z,
            c_atompair=c_atompair,
            **diffusion_token_encoder,
        )

        self.diffusion_transformer = LocalTokenTransformer(
            c_token=c_token,
            c_tokenpair=c_z,
            c_s=c_s,
            **diffusion_transformer,
        )

        self.decoder = CompactStreamingDecoder(
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_token=c_token,
            c_s=c_s,
            c_tokenpair=c_z,
            **atom_attention_decoder,
        )

    def scale_positions_in(self, X_noisy_L, t):
        if t.ndim == 1:
            t = t[..., None, None]  # [B, (n_atoms), (3)]
        elif t.ndim == 2:
            t = t[..., None]  # [B, n_atoms, (3)]

        if self.f_pred == "edm":
            R_noisy_L = X_noisy_L / torch.sqrt(t**2 + self.sigma_data**2)
        elif self.f_pred == "unconditioned":
            R_noisy_L = torch.zeros_like(X_noisy_L)
        elif self.f_pred == "noise_pred":
            R_noisy_L = X_noisy_L
        else:
            raise Exception(f"{self.f_pred=} unrecognized")
        return R_noisy_L

    def scale_positions_out(self, R_update_L, X_noisy_L, t):
        if t.ndim == 1:
            t = t[..., None, None]
        elif t.ndim == 2:
            t = t[..., None]  # [B, n_atoms, (3)]

        if self.f_pred == "edm":
            X_out_L = (self.sigma_data**2 / (self.sigma_data**2 + t**2)) * X_noisy_L + (
                self.sigma_data * t / (self.sigma_data**2 + t**2) ** 0.5
            ) * R_update_L
        elif self.f_pred == "unconditioned":
            X_out_L = R_update_L
        elif self.f_pred == "noise_pred":
            X_out_L = X_noisy_L + R_update_L
        else:
            raise Exception(f"{self.f_pred=} unrecognized")
        return X_out_L

    def process_time_(self, t_L, i):
        C_L = self.process_n[i](
            self.fourier_embedding[i](
                1 / 4 * torch.log(torch.clamp(t_L, min=1e-20) / self.sigma_data)
            )
        )
        # Mask out zero-time features;
        C_L = C_L * (t_L > 0).float()[..., None]  # [B, L, C_atom]
        return C_L

    def forward(
        self,
        X_noisy_L,
        t,
        f,
        # Features from initialization
        Q_L_init,
        C_L,
        P_LL,
        S_I,
        Z_II,
        n_recycle=None,
        # Chunked memory optimization parameters
        chunked_pairwise_embedder=None,
        initializer_outputs=None,
        **kwargs,
    ):
        """
        Diffusion forward pass with recycling.
        Computes denoised positions given encoded features and noisy coordinates.
        """
        # ... Collect inputs
        tok_idx = f["atom_to_token_map"]
        L = len(tok_idx)
        I = tok_idx.max() + 1  # Number of tokens
        f["attn_indices"] = create_attention_indices(
            X_L=X_noisy_L,
            f=f,
            n_attn_keys=self.n_attn_keys,
            n_attn_seq_neighbours=self.n_attn_seq_neighbours,
        )

        # ... Expand t tensors
        t_L = t.unsqueeze(-1).expand(-1, L) * (
            ~f["is_motif_atom_with_fixed_coord"]
        ).float().unsqueeze(0)
        t_I = t.unsqueeze(-1).expand(-1, I) * (
            ~f["is_motif_token_with_fully_fixed_coord"]
        ).float().unsqueeze(0)

        # ... Create scaled positions
        R_L_uniform = self.scale_positions_in(X_noisy_L, t)
        R_noisy_L = self.scale_positions_in(X_noisy_L, t_L)

        # ... Pool initial representation to sequence level
        A_I = self.process_a(R_noisy_L, tok_idx=tok_idx)
        S_I = self.downcast_c(C_L, S_I, tok_idx=tok_idx)

        # ... Add batch-wise features to inputs
        Q_L = Q_L_init.unsqueeze(0) + self.process_r(R_noisy_L)
        C_L = C_L.unsqueeze(0) + self.process_time_(t_L, i=0)
        S_I = S_I.unsqueeze(0) + self.process_time_(t_I, i=1)
        C_L = C_L + self.process_c(C_L)

        # ... Run Local-Atom Self Attention and Pool
        if chunked_pairwise_embedder is not None:
            # Chunked mode: pass chunked embedder and feature dict
            Q_L = self.encoder(
                Q_L,
                C_L,
                P_LL=None,
                indices=f["attn_indices"],
                f=f,  # Pass feature dict for chunked computation
                chunked_pairwise_embedder=chunked_pairwise_embedder,
                initializer_outputs=initializer_outputs,
            )
        else:
            # Standard mode: use full P_LL
            Q_L = self.encoder(Q_L, C_L, P_LL, indices=f["attn_indices"])
        A_I = self.downcast_q(Q_L, A_I=A_I, S_I=S_I, tok_idx=tok_idx)

        # Debug chunked parameters

        # ... Run forward with recycling
        recycled_features = self.forward_with_recycle(
            n_recycle,
            X_noisy_L=X_noisy_L,
            R_L_uniform=R_L_uniform,
            t_L=t_L,
            f=f,
            Q_L=Q_L,
            C_L=C_L,
            P_LL=P_LL,
            A_I=A_I,
            S_I=S_I,
            Z_II=Z_II,
            chunked_pairwise_embedder=chunked_pairwise_embedder,
            initializer_outputs=initializer_outputs,
        )

        # ... Collect outputs
        outputs = {
            "X_L": recycled_features["X_L"],  # [B, L, 3] denoised positions
            "sequence_indices_I": recycled_features["sequence_indices_I"],
            "sequence_logits_I": recycled_features["sequence_logits_I"],
        }
        return outputs

    def forward_with_recycle(
        self,
        n_recycle,
        **kwargs,
    ):
        if not self.training:
            n_recycle = self.n_recycle
        else:
            assert n_recycle is not None

        recycled_features = {}
        for i in range(n_recycle):
            with ExitStack() as stack:
                last = not (i < n_recycle - 1)
                if not last:
                    stack.enter_context(torch.no_grad())

                # Clear the autocast cache if gradients are enabled (workaround for autocast bug)
                # See: https://github.com/pytorch/pytorch/issues/65766
                if torch.is_grad_enabled():
                    torch.clear_autocast_cache()

                # Run forward
                recycled_features = self.process_(
                    D_II_self=recycled_features.get("D_II_self"),
                    X_L_self=recycled_features.get("X_L"),
                    **kwargs,
                )

        return recycled_features

    def process_(
        self,
        D_II_self,
        X_L_self,
        *,
        R_L_uniform,
        X_noisy_L,
        t_L,
        f,
        Q_L,
        C_L,
        P_LL,
        A_I,
        S_I,
        Z_II,
        chunked_pairwise_embedder=None,
        initializer_outputs=None,
        **_,
    ):
        # ... Embed token level features with atom level encodings
        S_I, Z_II = self.diffusion_token_encoder(
            f=f,
            R_L=R_L_uniform,
            D_II_self=D_II_self,
            S_init_I=S_I,
            Z_init_II=Z_II,
            C_L=C_L,
            P_LL=P_LL,
        )

        # ... Diffusion transformer
        A_I = self.diffusion_transformer(
            A_I,
            S_I,
            Z_II,
            f=f,
            X_L=(
                X_noisy_L[..., f["is_ca"], :]
                if X_L_self is None
                else X_L_self[..., f["is_ca"], :]
            ),
            full=not (os.environ.get("RFD3_LOW_MEMORY_MODE", None) == "1"),
        )

        # ... Decoder readout
        # Check if using chunked P_LL mode

        if chunked_pairwise_embedder is not None:
            # Chunked mode: pass embedder and no P_LL
            A_I, Q_L, o = self.decoder(
                A_I,
                S_I,
                Z_II,
                Q_L,
                C_L,
                P_LL=None,  # Not used in chunked mode
                tok_idx=f["atom_to_token_map"],
                indices=f["attn_indices"],
                f=f,  # Pass f for chunked computation
                chunked_pairwise_embedder=chunked_pairwise_embedder,
                initializer_outputs=initializer_outputs,
            )
        else:
            # Original mode: use full P_LL
            A_I, Q_L, o = self.decoder(
                A_I,
                S_I,
                Z_II,
                Q_L,
                C_L,
                P_LL=P_LL,
                tok_idx=f["atom_to_token_map"],
                indices=f["attn_indices"],
            )

        # ... Process outputs to positions update
        R_update_L = self.to_r_update(Q_L)
        X_out_L = self.scale_positions_out(R_update_L, X_noisy_L, t_L)

        sequence_logits_I, sequence_indices_I = self.sequence_head(A_I=A_I)
        D_II_self = self.bucketize_fn(X_out_L[..., f["is_ca"], :].detach())

        return {
            "X_L": X_out_L,
            "D_II_self": D_II_self,
            "sequence_logits_I": sequence_logits_I,
            "sequence_indices_I": sequence_indices_I,
        } | o
