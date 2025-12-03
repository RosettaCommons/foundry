import functools
import logging

import torch
import torch.nn as nn
from rfd3.model.layers.block_utils import (
    bucketize_scaled_distogram,
    pairwise_mean_pool,
)
from rfd3.model.layers.blocks import (
    Downcast,
    LocalAtomTransformer,
    OneDFeatureEmbedder,
    PositionPairDistEmbedder,
    RelativePositionEncodingWithIndexRemoval,
    SinusoidalDistEmbed,
)
from rfd3.model.layers.chunked_pairwise import (
    ChunkedPairwiseEmbedder,
    ChunkedPositionPairDistEmbedder,
    ChunkedSinusoidalDistEmbed,
)
from rfd3.model.layers.layer_utils import (
    RMSNorm,
    Transition,
    linearNoBias,
)
from rfd3.model.layers.pairformer_layers import PairformerBlock

from foundry.common import exists
from foundry.training.checkpoint import activation_checkpointing

logger = logging.getLogger(__name__)


class TokenInitializer(nn.Module):
    """
    Token embedding module for RFD3
    """

    def __init__(
        self,
        c_s,
        c_z,
        c_atom,
        c_atompair,
        relative_position_encoding,
        n_pairformer_blocks,
        pairformer_block,
        downcast,
        token_1d_features,
        atom_1d_features,
        atom_transformer,
        use_chunked_pll=False,  # New parameter for memory optimization
    ):
        super().__init__()

        # Store chunked mode flag
        self.use_chunked_pll = use_chunked_pll

        # Features
        self.atom_1d_embedder_1 = OneDFeatureEmbedder(atom_1d_features, c_s)
        self.atom_1d_embedder_2 = OneDFeatureEmbedder(atom_1d_features, c_atom)
        self.token_1d_embedder = OneDFeatureEmbedder(token_1d_features, c_s)

        self.downcast_atom = Downcast(c_atom=c_s, c_token=c_s, c_s=None, **downcast)
        self.transition_post_token = Transition(c=c_s, n=2)
        self.transition_post_atom = Transition(c=c_s, n=2)
        self.process_s_init = nn.Sequential(
            RMSNorm(c_s),
            linearNoBias(c_s, c_s),
        )

        # Operations to mix into Z_II and S_I
        self.to_z_init_i = linearNoBias(c_s, c_z)
        self.to_z_init_j = linearNoBias(c_s, c_z)
        self.relative_position_encoding = RelativePositionEncodingWithIndexRemoval(
            c_z=c_z, **relative_position_encoding
        )
        self.relative_position_encoding2 = RelativePositionEncodingWithIndexRemoval(
            c_z=c_z, **relative_position_encoding
        )
        self.process_token_bonds = linearNoBias(1, c_z)

        # Processing of Z_init
        self.process_z_init = nn.Sequential(
            RMSNorm(c_z * 2),
            linearNoBias(c_z * 2, c_z),
        )
        self.transition_1 = nn.ModuleList(
            [
                Transition(c=c_z, n=2),
                Transition(c=c_z, n=2),
            ]
        )
        self.ref_pos_embedder_tok = PositionPairDistEmbedder(c_z, embed_frame=False)

        # Pairformer without triangle updates
        self.transformer_stack = nn.ModuleList(
            [
                PairformerBlock(c_s=c_s, c_z=c_z, **pairformer_block)
                for _ in range(n_pairformer_blocks)
            ]
        )

        #############################################################################
        # Token track processing
        self.process_s_trunk = nn.Sequential(RMSNorm(c_s), linearNoBias(c_s, c_atom))
        self.process_single_l = nn.Sequential(
            nn.ReLU(), linearNoBias(c_atom, c_atompair)
        )
        self.process_single_m = nn.Sequential(
            nn.ReLU(), linearNoBias(c_atom, c_atompair)
        )
        self.process_z = nn.Sequential(RMSNorm(c_z), linearNoBias(c_z, c_atompair))

        # ALWAYS create these MLPs - they will be shared between chunked and standard modes
        self.motif_pos_embedder = SinusoidalDistEmbed(c_atompair=c_atompair)
        self.ref_pos_embedder = PositionPairDistEmbedder(c_atompair, embed_frame=False)
        self.pair_mlp = nn.Sequential(
            nn.ReLU(),
            linearNoBias(c_atompair, c_atompair),
            nn.ReLU(),
            linearNoBias(c_atompair, c_atompair),
            nn.ReLU(),
            linearNoBias(c_atompair, c_atompair),
        )

        # Atom pair feature processing
        if self.use_chunked_pll:
            # Initialize chunked embedders and share the trained MLPs!
            self.chunked_pairwise_embedder = ChunkedPairwiseEmbedder(
                c_atompair=c_atompair,
                motif_pos_embedder=ChunkedSinusoidalDistEmbed(c_atompair=c_atompair),
                ref_pos_embedder=ChunkedPositionPairDistEmbedder(
                    c_atompair, embed_frame=False
                ),
                process_single_l=self.process_single_l,  # Share trained parameters!
                process_single_m=self.process_single_m,  # Share trained parameters!
                process_z=self.process_z,  # Share trained parameters!
                pair_mlp=self.pair_mlp,  # Share trained parameters!
            )
        self.process_pll = linearNoBias(c_atompair, c_atompair)
        self.project_pll = linearNoBias(c_atompair, c_z)

        if atom_transformer["n_blocks"] > 0:
            self.atom_transformer = LocalAtomTransformer(
                c_atom=c_atom, c_s=None, c_atompair=c_atompair, **atom_transformer
            )
        else:
            self.atom_transformer = None

        # Post-processing
        # self.process_s_post = nn.Sequential(
        #     RMSNorm(c_s),
        #     linearNoBias(c_s, c_s),
        # )
        # self.process_z_post = nn.Sequential(
        #     RMSNorm(c_z),
        #     linearNoBias(c_z, c_z),
        # )

    def forward(self, f):
        """
        Provides initial representation for atom and token representations
        """
        tok_idx = f["atom_to_token_map"]
        L = len(tok_idx)
        f["ref_atom_name_chars"] = f["ref_atom_name_chars"].reshape(L, -1)
        I = len(f["restype"])

        def init_tokens():
            # Embed token features
            S_I = self.token_1d_embedder(f, I)
            S_I = S_I + self.transition_post_token(S_I)

            # Embed atom features and downcast to token features
            S_I = self.downcast_atom(
                Q_L=self.atom_1d_embedder_1(f, L), A_I=S_I, tok_idx=tok_idx
            )
            S_I = S_I + self.transition_post_atom(S_I)
            S_I = self.process_s_init(S_I)

            # Embed Z_II
            Z_init_II = self.to_z_init_i(S_I).unsqueeze(-3) + self.to_z_init_j(
                S_I
            ).unsqueeze(-2)
            Z_init_II = Z_init_II + self.relative_position_encoding(f)
            Z_init_II = Z_init_II + self.process_token_bonds(
                f["token_bonds"].unsqueeze(-1).float()
            )

            # Embed reference coordinates of ligands
            token_id = f["ref_space_uid"][f["is_ca"]]
            valid_mask = (token_id.unsqueeze(-1) == token_id.unsqueeze(-2)).unsqueeze(
                -1
            )
            Z_init_II = Z_init_II + self.ref_pos_embedder_tok(
                f["ref_pos"][f["is_ca"]], valid_mask
            )

            # Run a small transformer to provide position encodings to single.
            for block in self.transformer_stack:
                S_I, Z_init_II = block(S_I, Z_init_II)

            # Also cat the relative position encoding and mix
            Z_init_II = torch.cat(
                [
                    Z_init_II,
                    self.relative_position_encoding2(f),
                ],
                dim=-1,
            )
            Z_init_II = self.process_z_init(Z_init_II)
            for b in range(2):
                Z_init_II = Z_init_II + self.transition_1[b](Z_init_II)

            return {"S_init_I": S_I, "Z_init_II": Z_init_II}

        @activation_checkpointing
        def init_atoms(S_init_I, Z_init_II):
            Q_L_init = self.atom_1d_embedder_2(f, L)
            C_L = Q_L_init + self.process_s_trunk(S_init_I)[..., tok_idx, :]

            if self.use_chunked_pll:
                # Chunked mode: return embedder for later sparse computation
                return {
                    "Q_L_init": Q_L_init,
                    "C_L": C_L,
                    "chunked_pairwise_embedder": self.chunked_pairwise_embedder,
                    "S_I": S_init_I,
                    "Z_II": Z_init_II,
                }
            else:
                # Original full P_LL computation
                ##################################################################################
                # Embed motif coordinates
                valid_mask = (
                    f["is_motif_atom_with_fixed_coord"].unsqueeze(-1)
                    & f["is_motif_atom_with_fixed_coord"].unsqueeze(-2)
                ).unsqueeze(-1)
                P_LL = self.motif_pos_embedder(
                    f["motif_pos"], valid_mask
                )  # (L, L, c_atompair)

                # Embed ref pos
                atoms_in_same_token = (
                    f["ref_space_uid"].unsqueeze(-1) == f["ref_space_uid"].unsqueeze(-2)
                ).unsqueeze(-1)
                # Only consider ref_pos for atoms given seq (otherwise ref_pos is 0, doesn't make sense to compute)
                atoms_has_seq = (
                    f["is_motif_atom_with_fixed_seq"].unsqueeze(-1)
                    & f["is_motif_atom_with_fixed_seq"].unsqueeze(-2)
                ).unsqueeze(-1)
                valid_mask = atoms_in_same_token & atoms_has_seq
                P_LL = P_LL + self.ref_pos_embedder(f["ref_pos"], valid_mask)

                ##################################################################################

                P_LL = P_LL + (
                    self.process_single_l(C_L).unsqueeze(-2)
                    + self.process_single_m(C_L).unsqueeze(-3)
                )
                P_LL = (
                    P_LL
                    + self.process_z(Z_init_II)[..., tok_idx, :, :][..., tok_idx, :]
                )
                P_LL = P_LL + self.pair_mlp(P_LL)
                P_LL = P_LL.contiguous()

                # Pool P_LL to token level to provide atom-level resolution for token track
                pooled_atom_level_features = pairwise_mean_pool(
                    pairwise_atom_features=self.process_pll(P_LL).unsqueeze(0),
                    atom_to_token_map=tok_idx,
                    I=int(tok_idx.max().item()) + 1,
                    dtype=P_LL.dtype,
                ).squeeze(0)
                Z_init_II = Z_init_II + self.project_pll(pooled_atom_level_features)

                # Mix atom conditioning features via sequence-local attention
                if exists(self.atom_transformer):
                    C_L = self.atom_transformer(
                        C_L.unsqueeze(0), None, P_LL, indices=None, f=f, X_L=None
                    ).squeeze(0)

                return {
                    "Q_L_init": Q_L_init,
                    "C_L": C_L,
                    "P_LL": P_LL,
                    "S_I": S_init_I,
                    "Z_II": Z_init_II,
                }

        tokens = init_tokens()
        return init_atoms(**tokens)


class DiffusionTokenEncoder(nn.Module):
    def __init__(
        self,
        c_s,
        c_z,
        c_token,
        c_atompair,
        sigma_data,
        n_pairformer_blocks,
        pairformer_block,
        use_distogram,
        use_self,
        use_sinusoidal_distogram_embedder=True,
        **_,
    ):
        super().__init__()

        # Sequence processing
        self.transition_1 = nn.ModuleList(
            [
                Transition(c=c_s, n=2),
                Transition(c=c_s, n=2),
            ]
        )

        # Post-processing of z
        self.n_bins_distogram = 65  # n bins for both self distogram and distogram
        n_bins_noise = self.n_bins_distogram
        self.use_self = use_self
        self.use_distogram = use_distogram
        self.use_sinusoidal_distogram_embedder = use_sinusoidal_distogram_embedder
        if self.use_distogram:
            if self.use_sinusoidal_distogram_embedder:
                self.dist_embedder = SinusoidalDistEmbed(c_atompair=c_z)
                n_bins_noise = c_z
            else:
                self.bucketize_fn = functools.partial(
                    bucketize_scaled_distogram,
                    min_dist=1,
                    max_dist=30,
                    sigma_data=sigma_data,
                    n_bins=self.n_bins_distogram,
                )
        cat_c_z = (
            c_z
            + int(self.use_distogram) * n_bins_noise
            + int(self.use_self) * self.n_bins_distogram
        )
        self.process_z = nn.Sequential(
            RMSNorm(cat_c_z),
            linearNoBias(cat_c_z, c_z),
        )

        self.transition_2 = nn.ModuleList(
            [
                Transition(c=c_z, n=2),
                Transition(c=c_z, n=2),
            ]
        )

        # Pairformer without triangle updates
        self.pairformer_stack = nn.ModuleList(
            [
                PairformerBlock(c_s=c_s, c_z=c_z, **pairformer_block)
                for _ in range(n_pairformer_blocks)
            ]
        )

    def forward(self, f, R_L, S_init_I, Z_init_II, C_L, P_LL, **kwargs):
        B = R_L.shape[0]
        """
        Pools atom-level features to token-level features and encodes them into Z_II, S_I and prepares A_I.
        """

        @activation_checkpointing
        def token_embed(S_init_I, Z_init_II):
            S_I = S_init_I
            for b in range(2):
                S_I = S_I + self.transition_1[b](S_I)

            Z_II = Z_init_II.unsqueeze(0).expand(B, -1, -1, -1)  # B, I, I, c_z

            Z_II_list = [Z_II]
            if self.use_distogram:
                # Noise / self conditioning pair
                if self.use_sinusoidal_distogram_embedder:
                    mask = f["is_motif_atom_with_fixed_coord"][f["is_ca"]]
                    mask = (mask[None, :] != mask[:, None]).unsqueeze(
                        -1
                    )  # remove off-diagonals where distances don't make sense across time
                    D_LL = self.dist_embedder(R_L[..., f["is_ca"], :], ~mask)
                else:
                    D_LL = self.bucketize_fn(
                        R_L[..., f["is_ca"], :]
                    )  # [B, L, I, n_bins]
                Z_II_list.append(D_LL)
            if self.use_self:
                D_II_self = kwargs.get("D_II_self")
                if D_II_self is None:
                    D_II_self = torch.zeros(
                        Z_II.shape[:-1] + (self.n_bins_distogram,),
                        device=Z_II.device,
                        dtype=Z_II.dtype,
                    )
                Z_II_list.append(D_II_self)
            Z_II = torch.cat(Z_II_list, dim=-1)

            # Flatten concatenated dims
            Z_II = self.process_z(Z_II)

            for b in range(2):
                Z_II = Z_II + self.transition_2[b](Z_II)

            # Pairformer to mix
            for block in self.pairformer_stack:
                S_I, Z_II = block(S_I, Z_II)

            return S_I, Z_II

        return token_embed(S_init_I, Z_init_II)
