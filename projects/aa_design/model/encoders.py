
import functools
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from modelhub.model.AF3_structure import (
    FourierEmbedding,
)
from modelhub.model.layers.layer_utils import AdaLN, Transition, collapse, linearNoBias
from modelhub.model.layers.pairformer_layers import PairformerBlock
from modelhub.training.checkpoint import activation_checkpointing
from projects.aa_design.model.blocks import RelativePositionEncodingWithIndexRemoval

logger = logging.getLogger(__name__)


def bucketize_scaled_distogram(R_L, min_dist=1, max_dist=30, sigma_data=16, n_bins=65):
    '''
    Bucketizes pairwise distances into bins based on edm scaling
    
    min dist and max dist given as angstroms
    Will use bin ranges based on scaled angstrom distances

    R_L: B, N, 3
    D_LL: B, N, N
    D_LL_binned: B, N, N, n_bins
    '''
    D_LL = R_L.unsqueeze(-2) - R_L.unsqueeze(-3)  # [B, N, N, 3]
    D_LL = torch.linalg.norm(D_LL, dim=-1) # [B, N, N]

    # normalize    
    min_dist, max_dist = min_dist / sigma_data, max_dist / sigma_data

    bins = torch.linspace(
        min_dist, max_dist, n_bins - 1, device=D_LL.device
    )
    bin_idxs = torch.bucketize(D_LL, bins)
    return F.one_hot(bin_idxs, num_classes = len(bins) + 1).float().detach()


class TokenInitializer(nn.Module):
    '''
    Input Feature Embedder but without atom attention and some minor modifications

    Takes care of the relative position encoding
    '''
    def __init__(
        self,
        c_s,
        c_z,
        c_s_init,
        relative_position_encoding,
        c_token_to_embed,
        n_pairformer_blocks,
        pairformer_block
    ):
        super().__init__()

        # Processing of raw inputs
        self.embed_mask_type = linearNoBias(3, c_token_to_embed)
        self.to_s_init = linearNoBias(c_s_init, c_s)
        self.to_z_init_i = linearNoBias(c_s_init, c_z)
        self.to_z_init_j = linearNoBias(c_s_init, c_z)
        self.relative_position_encoding = RelativePositionEncodingWithIndexRemoval(
            c_z=c_z, **relative_position_encoding
        )
        self.process_token_bonds = linearNoBias(1, c_z)

        # Processing of Z_init
        self.process_z_init = nn.Sequential(
            nn.LayerNorm(c_z * 2), 
            linearNoBias(c_z * 2, c_z),
        )
        self.transition_1 = nn.ModuleList([
            Transition(c=c_z, n=2),
            Transition(c=c_z, n=2),
        ])

        # Final pairformer to mix
        self.pairformer_stack = nn.ModuleList(
            [
                PairformerBlock(c_s=c_s, c_z=c_z, **pairformer_block)
                for _ in range(n_pairformer_blocks)
            ]
        )
    
    def forward(self, f, t):
        '''
        Provides initial representation for token representations
        '''
        A_I = f['ref_motif_token_type'].float()  # encodes non-motif, indexed-motif, unindexed-motif

        @activation_checkpointing
        def run_init(A_I, f, t):
            A_I = self.embed_mask_type(A_I)
            S_inputs_I = torch.cat([A_I, f['restype']], dim=-1)

            S_init_I = self.to_s_init(S_inputs_I)
            Z_init_II = self.to_z_init_i(S_inputs_I).unsqueeze(-3) + \
                        self.to_z_init_j(S_inputs_I).unsqueeze(-2)
            Z_init_II = Z_init_II + self.relative_position_encoding(f)
            Z_init_II = Z_init_II + self.process_token_bonds(
                f["token_bonds"].unsqueeze(-1).to(torch.float)
            )


            # Run a small pairformer to provide position encodings to single.
            for block in self.pairformer_stack:
                S_init_I, Z_init_II = block(S_init_I, Z_init_II)

            # Also cat the relative position encoding and mix
            Z_init_II = torch.cat([
                Z_init_II, self.relative_position_encoding(f)
            ], dim=-1)
            Z_init_II = self.process_z_init(Z_init_II)
            for b in range(2):
                Z_init_II = Z_init_II + self.transition_1[b](Z_init_II)

            return S_init_I, Z_init_II
        return run_init(A_I, f, t)


class DiffusionTokenEncoder(nn.Module): # FeatureInitializerShort
    def __init__(self, 
        c_s, c_z, c_token,
        sigma_data, n_bins, c_t_embed, use_distogram
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_token = c_token
        self.sigma_data = sigma_data

        self.bucketize_fn = functools.partial(
            bucketize_scaled_distogram,
            min_dist=1, max_dist=30,
            sigma_data=sigma_data, n_bins=n_bins
        )
        
        # Sequence processing
        self.fourier_embedding = FourierEmbedding(c_t_embed)
        self.process_n = nn.Sequential(
            nn.LayerNorm(c_t_embed), 
            linearNoBias(c_t_embed, c_s)
        )
        self.transition_1 = nn.ModuleList([
            Transition(c=c_s, n=2),
            Transition(c=c_s, n=2),
        ])

        # Post-processing of z
        self.use_distogram=use_distogram
        if self.use_distogram:  
            self.process_distogram = linearNoBias(n_bins, c_z)
            self.process_z = nn.Sequential(
                nn.LayerNorm(c_z * 2),
                linearNoBias(c_z * 2, c_z),
            )
        # self.process_n2 = nn.Sequential(
        #     nn.LayerNorm(c_t_embed),
        #     linearNoBias(c_t_embed, c_z)
        # )
        # self.adaln_z = AdaLN(c_a=c_z, c_s=c_z, n=2)
        else:
            self.process_z = nn.Sequential(
                nn.LayerNorm(c_z),
                linearNoBias(c_z, c_z),
            )
            
        self.transition_2 = nn.ModuleList([
            Transition(c=c_z, n=2),
            Transition(c=c_z, n=2),
        ])

    def forward(self, f, R_L, t, S_init_I, Z_init_II):
        B = R_L.shape[0]
        I = S_init_I.shape[-2]

        if self.use_distogram:
            # BUG: doesn't work for unindexed (removed for now)
            D_LL = self.bucketize_fn( R_L[..., f['is_central'], :])  # [B, I, I, n_bins]
        else:
            D_LL = None

        @activation_checkpointing
        def token_embed(f, D_LL, t, S_init_I, Z_init_II):
            # Time conditioning single
            N_D = self.fourier_embedding(1 / 4 * torch.log(t / self.sigma_data))
            S_I = self.process_n(N_D).unsqueeze(-2) + S_init_I  # Adds batch dim to S_I
            for b in range(2):
                S_I = S_I + self.transition_1[b](S_I)

            Z_II = Z_init_II.unsqueeze(0).expand(
                B, -1, -1, -1
            ) # B, I, I, c_z

            # Noise conditioning pair via bins
            if self.use_distogram:
                Z_II_distogram = self.process_distogram(D_LL)
                Z_II = torch.cat([Z_II, Z_II_distogram], dim=-1)
            
            Z_II = self.process_z(Z_II)
            # Optional: Collect via time-conditioned Ada-LN
            # N_DD = self.process_n2(N_D)[..., None, None, :].expand(
            #     -1, I, I, -1
            # )
            # Z_II = self.adaln_z(Z_II, N_DD)
            for b in range(2):
                Z_II = Z_II + self.transition_2[b](Z_II)

            # For now, provide no skip connetion from A_I
            A_I = None
            return A_I, S_I, Z_II
        return token_embed(f, D_LL, t, S_init_I, Z_init_II)

class DiffusionAtomEncoder(nn.Module):
    '''
    AtomAttentionEncoder without the Attention part
    For delegating processing to other parts of the network
    '''
    def __init__(
        self,
        c_atom,
        c_atompair,
        c_token,
        c_tokenpair,
        c_s,
        atom_1d_features,
        c_atom_1d_features,
        broadcast_trunk_feats_on_1dim_old,
    ):
        super().__init__()
        # Channels
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        self.c_tokenpair = c_tokenpair
        self.c_s = c_s
        
        # 1D Feature tracks
        self.atom_1d_features = atom_1d_features
        self.broadcast_trunk_feats_on_1dim_old = broadcast_trunk_feats_on_1dim_old
        self.process_input_features = linearNoBias(c_atom_1d_features, c_atom)

        # "Trunk" processing
        self.process_s_trunk = nn.Sequential(
            nn.LayerNorm(c_s), linearNoBias(c_s, c_atom)
        )
        self.process_z = nn.Sequential(
            nn.LayerNorm(c_tokenpair), linearNoBias(c_tokenpair, c_atompair)
        )

        # Coordinate processing
        self.process_r = linearNoBias(3, c_atom)

        self.process_single_l = nn.Sequential(
            nn.ReLU(), linearNoBias(c_atom, c_atompair)
        )
        self.process_single_m = nn.Sequential(
            nn.ReLU(), linearNoBias(c_atom, c_atompair)
        )

        self.pair_mlp = nn.Sequential(
            nn.ReLU(),
            linearNoBias(c_atompair, c_atompair),
            nn.ReLU(),
            linearNoBias(c_atompair, c_atompair),
            nn.ReLU(),
            linearNoBias(c_atompair, c_atompair),
        )

        self.ref_pos_embedder = PositionPairDistEmbedder(c_atompair)
        # self.process_gt_coord = linearNoBias(3, c_atom)
        # self.gt_pos_embedder = PositionPairEmbedder(c_atompair)

    def forward(
        self,
        f,  # Dict (Input feature dictionary)
        R_L,  # [D, L, 3]
        S_I,  # [B, I, C_S_trunk] [...,I,C_S_trunk]
        Z_II,  # [B, I, I, C_Z] [...,I,I,C_Z]
    ):
        tok_idx = f["atom_to_token_map"]
        L = len(tok_idx)
        f["ref_atom_name_chars"] = f["ref_atom_name_chars"].reshape(L, -1)

        @activation_checkpointing
        def atom_embed(f, R_L, S_I, Z_II):
            ##############################################################
            # Create the atom single conditioning: Embed per-atom meta data
            C_L = self.process_input_features(
                torch.cat(
                    tuple(
                        collapse(f[feature_name], L)
                        for feature_name in self.atom_1d_features
                    ),dim=-1,
                )
            )  # [L, C_atom]

            # Initialise the atom single representation as the single conditioning.
            Q_L = C_L
            
            # Add unmasked gt coordinates to single
            # Q_L = Q_L.unsqueeze(0) + self.process_gt_coord(f['gt_pos_scaled'])

            ##############################################################
            # Embed offsets between atom reference positions
            valid_mask = (
                f['ref_space_uid'].unsqueeze(-1) == f['ref_space_uid'].unsqueeze(-2)
            ).unsqueeze(-1)
            P_LL = self.ref_pos_embedder(f["ref_pos"], valid_mask)  # (L, L, c_atompair)

            ##############################################################
            # Embed gt coordinates similarly
            # Batch size is not needed here, since distances are embedded
            # is_unmasked = ~f["is_masked_token"][tok_idx]  # (N_atoms, )
            # valid_mask = (
            #     is_unmasked.unsqueeze(-1) * is_unmasked.unsqueeze(-2)
            # ).unsqueeze(-1)  #.unsqueeze(0).expand(R_L.shape[0], -1, -1, -1)
            # P_LL = P_LL + self.gt_pos_embedder(f["gt_pos"][0, ...], valid_mask) # (L, L, c_atompair)

            ##############################################################
            # Broadcast the single and pair embedding from the trunk.
            S_trunk_embed_L = self.process_s_trunk(S_I)[..., tok_idx, :]
            C_L = C_L + S_trunk_embed_L

            if len(Z_II.shape) == 3 and len(P_LL.shape) == 4:
                Z_II = Z_II.unsqueeze(0)
            elif len(Z_II.shape) == 4 and len(P_LL.shape) == 3:
                P_LL = P_LL.unsqueeze(0)
            if self.broadcast_trunk_feats_on_1dim_old:
                P_LL = P_LL + self.process_z(Z_II)[..., tok_idx, tok_idx, :]
            else:
                P_LL = P_LL + self.process_z(Z_II)[..., tok_idx, :, :][..., tok_idx, :]
            ##############################################################
            
            # Add the noisy positions.
            Q_L = self.process_r(R_L) + Q_L

            # Add the combined single conditioning to the pair representation.
            P_LL = P_LL + (
                  self.process_single_l(C_L).unsqueeze(-2)
                + self.process_single_m(C_L).unsqueeze(-3)
            )

            # Run a small MLP on the pair activations
            P_LL = P_LL + self.pair_mlp(P_LL)

            return Q_L, C_L, P_LL
        return atom_embed(f, R_L, S_I, Z_II)


class PositionPairDistEmbedder(nn.Module):

    def __init__(self, c_atompair):
        super().__init__()
        self.process_d = linearNoBias(3, c_atompair)
        self.process_inverse_dist = linearNoBias(1, c_atompair)
        self.process_valid_mask = linearNoBias(1, c_atompair)
    
    def forward(self, ref_pos, valid_mask):
        D_LL = ref_pos.unsqueeze(-2) - ref_pos.unsqueeze(-3)
        V_LL = valid_mask

        P_LL = self.process_d(D_LL) * V_LL

        # Embed pairwise inverse squared distances, and the valid mask
        if self.training:
            P_LL = (
                P_LL
                + self.process_inverse_dist(
                    1 / (1 + torch.linalg.norm(D_LL, dim=-1, keepdim=True))
                )
                * V_LL
            )
            P_LL = P_LL + self.process_valid_mask(V_LL.to(P_LL.dtype)) * V_LL
        else:
            P_LL[V_LL[..., 0]] += self.process_inverse_dist(
                1
                / (1 + torch.linalg.norm(D_LL[V_LL[..., 0]], dim=-1, keepdim=True))
            )
            P_LL[V_LL[..., 0]] += self.process_valid_mask(
                V_LL[V_LL[..., 0]].to(P_LL.dtype)
            )
        return P_LL
