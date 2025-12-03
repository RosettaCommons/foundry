"""
Chunked pairwise embedding implementation for memory-efficient large structure processing.

This module provides memory-optimized versions of pairwise embedders that compute
only the pairs needed for sparse attention, reducing memory usage from O(L²) to O(L×k).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from rfd3.model.layers.layer_utils import RMSNorm, linearNoBias


class ChunkedPositionPairDistEmbedder(nn.Module):
    """
    Memory-efficient version of PositionPairDistEmbedder that computes pairs on-demand.
    """

    def __init__(self, c_atompair, embed_frame=True):
        super().__init__()
        self.c_atompair = c_atompair
        self.embed_frame = embed_frame
        if embed_frame:
            self.process_d = linearNoBias(3, c_atompair)

        self.process_inverse_dist = linearNoBias(1, c_atompair)
        self.process_valid_mask = linearNoBias(1, c_atompair)

    def compute_pairs_chunked(
        self,
        query_pos: torch.Tensor,  # [D, 3]
        key_pos: torch.Tensor,  # [D, k, 3]
        valid_mask: torch.Tensor,  # [D, k, 1]
    ) -> torch.Tensor:
        """
        Compute pairwise embeddings for specific query-key pairs.

        Args:
            query_pos: Query positions [D, 3]
            key_pos: Key positions [D, k, 3]
            valid_mask: Valid pair mask [D, k, 1]

        Returns:
            P_sparse: Pairwise embeddings [D, k, c_atompair]
        """
        D, k = key_pos.shape[:2]

        # Compute pairwise distances: [D, k, 3]
        D_pairs = query_pos.unsqueeze(1) - key_pos  # [D, 1, 3] - [D, k, 3] = [D, k, 3]

        if self.embed_frame:
            # Embed pairwise distances
            P_pairs = self.process_d(D_pairs) * valid_mask  # [D, k, c_atompair]

            # Add inverse distance embedding
            norm_sq = torch.linalg.norm(D_pairs, dim=-1, keepdim=True) ** 2  # [D, k, 1]
            inv_dist = 1 / (1 + norm_sq)
            P_pairs = P_pairs + self.process_inverse_dist(inv_dist) * valid_mask

            # Add valid mask embedding
            P_pairs = (
                P_pairs
                + self.process_valid_mask(valid_mask.to(P_pairs.dtype)) * valid_mask
            )
        else:
            # Simplified version without frame embedding
            norm_sq = torch.linalg.norm(D_pairs, dim=-1, keepdim=True) ** 2
            norm_sq = torch.clamp(norm_sq, min=1e-6)
            inv_dist = 1 / (1 + norm_sq)
            P_pairs = self.process_inverse_dist(inv_dist) * valid_mask
            P_pairs = (
                P_pairs
                + self.process_valid_mask(valid_mask.to(P_pairs.dtype)) * valid_mask
            )

        return P_pairs


class ChunkedSinusoidalDistEmbed(nn.Module):
    """
    Memory-efficient version of SinusoidalDistEmbed.
    """

    def __init__(self, c_atompair, n_freqs=32):
        super().__init__()
        assert c_atompair % 2 == 0, "Output embedding dim must be even"

        self.n_freqs = n_freqs
        self.c_atompair = c_atompair

        self.output_proj = linearNoBias(2 * n_freqs, c_atompair)
        self.process_valid_mask = linearNoBias(1, c_atompair)

    def compute_pairs_chunked(
        self,
        query_pos: torch.Tensor,  # [D, 3]
        key_pos: torch.Tensor,  # [D, k, 3]
        valid_mask: torch.Tensor,  # [D, k, 1]
    ) -> torch.Tensor:
        """
        Compute sinusoidal distance embeddings for specific query-key pairs.
        """
        D, k = key_pos.shape[:2]
        device = query_pos.device

        # Compute pairwise distances
        D_pairs = query_pos.unsqueeze(1) - key_pos  # [D, k, 3]
        dist_matrix = torch.linalg.norm(D_pairs, dim=-1)  # [D, k]

        # Sinusoidal embedding
        half_dim = self.n_freqs
        freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, half_dim, dtype=torch.float32, device=device)
            / half_dim
        )  # [n_freqs]

        angles = dist_matrix.unsqueeze(-1) * freq  # [D, k, n_freqs]
        sin_embed = torch.sin(angles)
        cos_embed = torch.cos(angles)
        sincos_embed = torch.cat([sin_embed, cos_embed], dim=-1)  # [D, k, 2*n_freqs]

        # Linear projection
        P_pairs = self.output_proj(sincos_embed)  # [D, k, c_atompair]
        P_pairs = P_pairs * valid_mask

        # Add linear embedding of valid mask
        P_pairs = (
            P_pairs + self.process_valid_mask(valid_mask.to(P_pairs.dtype)) * valid_mask
        )

        return P_pairs


class ChunkedPairwiseEmbedder(nn.Module):
    """
    Main chunked pairwise embedder that combines all embedding types.
    This replaces the full P_LL computation with sparse computation.
    """

    def __init__(
        self,
        c_atompair: int,
        motif_pos_embedder: Optional[ChunkedPositionPairDistEmbedder] = None,
        ref_pos_embedder: Optional[ChunkedPositionPairDistEmbedder] = None,
        process_single_l: Optional[nn.Module] = None,
        process_single_m: Optional[nn.Module] = None,
        process_z: Optional[nn.Module] = None,
        pair_mlp: Optional[nn.Module] = None,
        **kwargs,
    ):
        super().__init__()
        self.c_atompair = c_atompair
        self.motif_pos_embedder = motif_pos_embedder
        self.ref_pos_embedder = ref_pos_embedder

        # Use shared trained MLPs if provided, otherwise create new ones
        if process_single_l is not None:
            self.process_single_l = process_single_l
        else:
            self.process_single_l = nn.Sequential(
                nn.ReLU(), linearNoBias(128, c_atompair)
            )

        if process_single_m is not None:
            self.process_single_m = process_single_m
        else:
            self.process_single_m = nn.Sequential(
                nn.ReLU(), linearNoBias(128, c_atompair)
            )

        if process_z is not None:
            self.process_z = process_z
        else:
            self.process_z = nn.Sequential(RMSNorm(128), linearNoBias(128, c_atompair))

        if pair_mlp is not None:
            self.pair_mlp = pair_mlp
        else:
            self.pair_mlp = nn.Sequential(
                nn.ReLU(),
                linearNoBias(c_atompair, c_atompair),
                nn.ReLU(),
                linearNoBias(c_atompair, c_atompair),
                nn.ReLU(),
                linearNoBias(c_atompair, c_atompair),
            )

    def forward_chunked(
        self,
        f: dict,
        indices: torch.Tensor,  # [D, L, k] - sparse attention indices
        C_L: torch.Tensor,  # [D, L, c_token] - atom features
        Z_init_II: torch.Tensor,  # [I, I, c_z] - token pair features
        tok_idx: torch.Tensor,  # [L] - atom to token mapping
    ) -> torch.Tensor:
        # Add logging for chunked P_LL computation
        import logging

        logger = logging.getLogger(__name__)
        logger.info(
            f"ChunkedPairwiseEmbedder: Computing sparse P_LL for {indices.shape[1]} atoms with {indices.shape[2]} neighbors each"
        )
        """
        Compute P_LL only for the pairs specified by attention indices.
        
        Args:
            f: Feature dictionary
            indices: Sparse attention indices [D, L, k]
            C_L: Atom-level features [D, L, c_token]
            Z_init_II: Token-level pair features [I, I, c_z]
            tok_idx: Atom to token mapping [L]
            
        Returns:
            P_LL_sparse: Sparse pairwise features [D, L, k, c_atompair]
        """
        D, L, k = indices.shape
        device = indices.device

        # Initialize sparse P_LL
        P_LL_sparse = torch.zeros(
            D, L, k, self.c_atompair, device=device, dtype=C_L.dtype
        )

        # Handle both batched and non-batched C_L
        if C_L.dim() == 2:  # [L, c_token] - add batch dimension
            C_L = C_L.unsqueeze(0)  # [1, L, c_token]
        # Add bounds checking to prevent index errors
        L_max = C_L.shape[1]
        valid_indices = torch.clamp(
            indices, 0, L_max - 1
        )  # Clamp indices to valid range

        # Ensure indices have the right shape for gathering
        if valid_indices.dim() == 2:  # [L, k] - add batch dimension
            valid_indices = valid_indices.unsqueeze(0).expand(
                C_L.shape[0], -1, -1
            )  # [D, L, k]

        # 1. Motif position embedding (if exists)
        if self.motif_pos_embedder is not None and "motif_pos" in f:
            motif_pos = f["motif_pos"]  # [L, 3]
            is_motif = f["is_motif_atom_with_fixed_coord"]  # [L]

            # For each query position
            for l in range(L):
                if is_motif[l]:  # Only compute if query is motif
                    key_indices = valid_indices[:, l, :]  # [D, k] - use clamped indices
                    key_pos = motif_pos[key_indices]  # [D, k, 3]
                    query_pos = motif_pos[l].unsqueeze(0).expand(D, -1)  # [D, 3]

                    # Valid mask: both query and keys must be motif
                    key_is_motif = is_motif[key_indices]  # [D, k]
                    valid_mask = key_is_motif.unsqueeze(-1).float()  # [D, k, 1]

                    if valid_mask.sum() > 0:
                        motif_pairs = self.motif_pos_embedder.compute_pairs_chunked(
                            query_pos, key_pos, valid_mask
                        )
                        P_LL_sparse[:, l, :, :] += motif_pairs

        # 2. Reference position embedding (if exists)
        if self.ref_pos_embedder is not None and "ref_pos" in f:
            ref_pos = f["ref_pos"]  # [L, 3]
            ref_space_uid = f["ref_space_uid"]  # [L]
            is_motif_seq = f["is_motif_atom_with_fixed_seq"]  # [L]

            for l in range(L):
                if is_motif_seq[l]:  # Only compute if query has sequence
                    key_indices = valid_indices[:, l, :]  # [D, k] - use clamped indices
                    key_pos = ref_pos[key_indices]  # [D, k, 3]
                    query_pos = ref_pos[l].unsqueeze(0).expand(D, -1)  # [D, 3]

                    # Valid mask: same token and both have sequence
                    key_space_uid = ref_space_uid[key_indices]  # [D, k]
                    key_is_motif_seq = is_motif_seq[key_indices]  # [D, k]

                    same_token = key_space_uid == ref_space_uid[l]  # [D, k]
                    valid_mask = (
                        (same_token & key_is_motif_seq).unsqueeze(-1).float()
                    )  # [D, k, 1]

                    if valid_mask.sum() > 0:
                        ref_pairs = self.ref_pos_embedder.compute_pairs_chunked(
                            query_pos, key_pos, valid_mask
                        )
                        P_LL_sparse[:, l, :, :] += ref_pairs

        # 3. Single embedding terms (broadcasted)
        # Gather key features for each query
        C_L_keys = torch.gather(
            C_L.unsqueeze(2).expand(-1, -1, k, -1),
            1,
            valid_indices.unsqueeze(-1).expand(-1, -1, -1, C_L.shape[-1]),
        )  # [D, L, k, c_token]
        C_L_queries = C_L.unsqueeze(2).expand(-1, -1, k, -1)  # [D, L, k, c_token]

        # Add single embeddings - match standard implementation structure
        # Standard does: self.process_single_l(C_L).unsqueeze(-2) + self.process_single_m(C_L).unsqueeze(-3)
        # We need to broadcast from [D, L, k, c_atompair] to match this
        single_l = self.process_single_l(C_L_queries)  # [D, L, k, c_atompair]
        single_m = self.process_single_m(C_L_keys)  # [D, L, k, c_atompair]
        P_LL_sparse += single_l + single_m

        # 4. Token pair features Z_init_II
        # Map atoms to tokens and gather token pair features
        # Handle tok_idx dimensions properly
        if tok_idx.dim() == 1:  # [L] - add batch dimension for consistency
            tok_idx_expanded = tok_idx.unsqueeze(0)  # [1, L]
        else:
            tok_idx_expanded = tok_idx

        tok_queries = tok_idx_expanded.unsqueeze(2).expand(-1, -1, k)  # [D, L, k]
        # Use valid_indices for token mapping as well
        tok_keys = torch.gather(
            tok_idx_expanded.unsqueeze(2).expand(-1, -1, k), 1, valid_indices
        )  # [D, L, k]

        # Gather Z_init_II[tok_queries, tok_keys] with safe indexing
        # Z_init_II shape is [I, I, c_z] (3D), not 4D
        # tok_queries shape: [D, L, k] - each value is a token index
        # We want: Z_init_II[tok_queries[d,l,k], tok_keys[d,l,k], :] for all d,l,k

        I_z, I_z2, c_z = Z_init_II.shape

        # CRITICAL: Match standard implementation exactly!
        # Standard does: self.process_z(Z_init_II)[..., tok_idx, :, :][..., tok_idx, :]
        # This means: 1) Process Z_init_II first, 2) Then do double token indexing

        # Step 1: Process Z_init_II to get processed token pair features
        Z_processed = self.process_z(Z_init_II)  # [I, I, c_atompair]

        # Step 2: Do the double indexing like the standard implementation
        # Standard: Z_processed[..., tok_idx, :, :][..., tok_idx, :]
        # This creates Z_processed[tok_idx, :][:, tok_idx] which is [L, L, c_atompair]
        # Then we need to gather the sparse version

        Z_pairs_processed = torch.zeros(
            D, L, k, self.c_atompair, device=device, dtype=Z_processed.dtype
        )

        for d in range(D):
            # For this batch, get the token queries and keys
            tq = tok_queries[d]  # [L, k]
            tk = tok_keys[d]  # [L, k]

            # Ensure indices are within bounds
            tq = torch.clamp(tq, 0, I_z - 1)
            tk = torch.clamp(tk, 0, I_z2 - 1)

            # Apply the double token indexing like standard implementation
            Z_pairs_processed[d] = Z_processed[tq, tk]  # [L, k, c_atompair]

        P_LL_sparse += Z_pairs_processed

        # 5. Final MLP - ADD the result, don't replace (to match standard implementation)
        P_LL_sparse = P_LL_sparse + self.pair_mlp(P_LL_sparse)

        return P_LL_sparse.contiguous()


def create_chunked_embedders(
    c_atompair: int, embed_frame: bool = True
) -> ChunkedPairwiseEmbedder:
    """
    Factory function to create chunked pairwise embedder with standard components.
    """
    motif_pos_embedder = ChunkedPositionPairDistEmbedder(c_atompair, embed_frame)
    ref_pos_embedder = ChunkedPositionPairDistEmbedder(c_atompair, embed_frame)

    return ChunkedPairwiseEmbedder(
        c_atompair=c_atompair,
        motif_pos_embedder=motif_pos_embedder,
        ref_pos_embedder=ref_pos_embedder,
    )
