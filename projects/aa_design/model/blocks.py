
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from opt_einsum import contract as einsum
from torch.nn.functional import one_hot

from modelhub.model.AF3_structure import (
    AtomAttentionDecoder,  # noqa: F401
    AtomAttentionEncoderDiffusion,  # noqa: F401
)
from modelhub.model.layers.af3_diffusion_transformer import (
    AtomTransformer,
    DiffusionTransformer,
)
from modelhub.model.layers.layer_utils import (
    LinearBiasInit,
    linearNoBias,
)
from modelhub.training.checkpoint import activation_checkpointing

logger = logging.getLogger(__name__)


class RelativePositionEncodingWithIndexRemoval(nn.Module):
    '''
    Usual RPE but utilizes `is_motif_atom_without_index` to ensure within-chain position is spoofed.
    '''
    def __init__(self, r_max, s_max, c_z):
        super().__init__()
        self.r_max = r_max
        self.s_max = s_max
        self.c_z = c_z
        
        self.num_tok_pos_bins = (2 * self.r_max + 2) + 1  # original af3 + 1 for unknown index
        self.linear = linearNoBias(
            2 * self.num_tok_pos_bins + (2 * self.s_max + 2) + 1, c_z
        )

    def forward(self, f):
        b_samechain_II = f["asym_id"].unsqueeze(-1) == f["asym_id"].unsqueeze(-2)
        b_same_entity_II = f["entity_id"].unsqueeze(-1) == f["entity_id"].unsqueeze(-2)
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
        b_sameresidue_II = f["residue_index"].unsqueeze(-1) == f[
            "residue_index"
        ].unsqueeze(-2)
        tok_distance = f["token_index"].unsqueeze(-1) \
                - f["token_index"].unsqueeze(-2) \
                + self.r_max
        d_token_II = torch.where(
            b_samechain_II * b_sameresidue_II,
            torch.clip(
                tok_distance,
                0,
                2 * self.r_max,
            ),
            2 * self.r_max + 1,
        )

        #########################################################
        # Cancel out distances from unidexed motifs
        unindexing_pair_mask = f['unindexing_pair_mask']  # [L, L] representing the parts which shouldnt' talk to one another

        # Special position case
        d_token_II[unindexing_pair_mask] = self.num_tok_pos_bins - 1
        d_residue_II[unindexing_pair_mask] = self.num_tok_pos_bins - 1

        A_relpos_II = one_hot(d_residue_II.long(), self.num_tok_pos_bins)
        A_reltoken_II = one_hot(d_token_II, self.num_tok_pos_bins)
        #########################################################

        # Chain distances are kept
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

class AtomTransformerWrapper(nn.Module):

    def __init__(self, c_atom, c_atompair, atom_transformer):
        super().__init__()
        self.atom_transformer = AtomTransformer(
            c_atom=c_atom, c_atompair=c_atompair, **atom_transformer
        )

    def forward(
        self,
        Q_L,  # [L, C_atom]
        C_L,  # [L, C_atom]
        P_LL,  # [L, L, C_atompair]
    ):
        @activation_checkpointing
        def atom_attn(Q_L, C_L, P_LL):
            return self.atom_transformer(Q_L, C_L, P_LL)
        return atom_attn(Q_L, C_L, P_LL)

class GatedCrossAttention(nn.Module):
    def __init__(self,
        c_query,
        c_kv,
        c_model=128,
        n_head=4,
        dropout=0.,
    ):
        super().__init__()
        self.n_head = n_head
        self.scale = 1/math.sqrt(c_model // n_head)
        assert c_model % n_head == 0, "c_model must be divisible by n_heads"

        self.to_q = linearNoBias(c_query, c_model)
        self.to_k = linearNoBias(c_kv, c_model)
        self.to_v = linearNoBias(c_kv, c_model)
        self.to_g = nn.Sequential(
                    LinearBiasInit(c_query, c_model, biasinit=-2.),
                    nn.Sigmoid())
        self.to_out = nn.Sequential(
            nn.Linear(c_model, c_query),
            nn.Dropout(dropout)
        )
        self.reset_parameter()
    
    def reset_parameter(self):
        # query/key/value projection: Xavier uniform
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        nn.init.xavier_uniform_(self.to_g[0].weight)
        nn.init.xavier_uniform_(self.to_out[0].weight)
    
    def forward(self, q, kv, attn_mask):
        '''
        Args:
            q: [B, tok, n_q, c_query]
            kv: [B, tok, n_kv, c_kv]
            attn_mask: [n_q, n_kv]
        Returns:
            attn_out: [B, tok, n_q, c_query]
        '''
        q, k, v, g = self.to_q(q), self.to_k(kv), self.to_v(kv), self.to_g(q)
        
        q, k, v, g = map(lambda t: 
            rearrange(t, 'b t n (h c) -> b h t n c', h=self.n_head),
            (q, k, v, g)) # [B, tok, n, heads, c] ->  [B, heads, tok, n, c]

        invalid_queries=torch.logical_not(torch.any(attn_mask, dim=-1, keepdim=False))  # [n_q,]
        attn_mask = attn_mask[None, None]  # [1, 1, n_q, n_kv]

        attn = einsum('bhtqc,bhtkc->bhtqk', q, k) * self.scale
        attn = attn.masked_fill(~attn_mask, float('-inf'))
        
        # Bugfix: Empty queries need to have a constant value otherwise nans are in the forward graph. I don't 
        # know why this causes instabilities because the invalid queries are masked out later. Oh well!
        attn[:, :, invalid_queries, :] = 0.

        attn = F.softmax(attn, dim=-1)
        attn_out = einsum('bhtqk,bhtkd->bhtqd', attn, v)

        attn_out = attn_out * g
        
        attn_out = rearrange(attn_out, 'b h t n c -> b t n (h c)')
        attn_out = self.to_out(attn_out) # [B, n_tok, n_k, c]
        return attn_out

class AtomTokenCrossAttention(nn.Module):
    def __init__(self,
        c_atom=128,
        c_token=768,
        n_split=6,
        c_model=128,
        n_head=4,
        dropout=0.,
        query_type='token',
        **kwargs
    ):
        super().__init__()
        self.query_type = query_type
        self.n_split = n_split
        assert c_token % n_split == 0, "c_token must be divisible by n_split"
        self.ln_atom = nn.LayerNorm(c_atom)

        if query_type == 'token':
            self.ln_token = nn.LayerNorm(c_token)
            self.gca = GatedCrossAttention(
                c_query=c_token,
                c_kv=c_atom,
                c_model=c_model,
                n_head=n_head,
                dropout=dropout,
                **kwargs
            )
        elif query_type == 'atom':
            self.ln_token = nn.LayerNorm(c_token // n_split, bias=False)
            self.gca = GatedCrossAttention(
                c_query=c_atom,
                c_kv=c_token // self.n_split,
                c_model=c_model,
                n_head=n_head,
                dropout=dropout,
                **kwargs
            )
        else:
            raise ValueError(f"Invalid query type: {query_type}. Choose either 'token' or 'atom'.")

    def forward(self, Q_L, A_I, tok_idx):
        '''
        Q_L: atom-level features  [B, n_atoms, 128]
        A_I: token-level features [B, n_tokens, 768]
        tok_idx: atom_idx_to_tok_idx mapping [n_atoms]
        
        This function is made complex because of the optimization of the tokens acting
        as a batch dimension (since for every token there is a unique set of atoms to attend to).

        returns: attn_out.shape == {'tokens': A_I, 'atoms': Q_L}[self.query_type].shape
        '''
        Q_L_shape_orig = Q_L.shape
        A_I_shape_orig = A_I.shape
        B, n_atoms, c_atom = Q_L_shape_orig
        B, n_tokens, c_token = A_I_shape_orig
        
        Q_L = self.ln_atom(Q_L)

        # Expand atoms to be max num atoms per token in input
        # [B, n_atoms, 128] -> [B, n_tokens, n_atoms_per_tok, 128]
        # Attn mask: [n_tokens, n_atoms_per_tok]  # which atoms were padded
        Q_L, valid_mask = self.ungroup_atoms(Q_L, tok_idx)
        _, _, n_atom_per_tok, _ = Q_L.shape

        # Split the tokens if necessary and prepare the attention mask
        if self.query_type=='token':
            # tokens -> [B, n_tok, 1, 768]
            A_I = A_I[..., None, :]
            A_I = self.ln_token(A_I)
            attn_mask = torch.full((n_tokens, 1, n_atom_per_tok), True, device=Q_L.device)
            attn_mask[~valid_mask.view_as(attn_mask)] = False
            attn_out = self.gca(q=A_I, kv=Q_L, attn_mask=attn_mask)
            attn_out = attn_out.squeeze(-2)

        elif self.query_type=='atom': 
            # split tokens -> [B, n_tok, n_split, 128]
            A_I = rearrange(A_I, 'b n (s c) -> b n s c', s=self.n_split)
            A_I = self.ln_token(A_I)
            attn_mask = torch.full((n_tokens, n_atom_per_tok, self.n_split), True, device=Q_L.device)
            attn_mask[~valid_mask, :] = False
            attn_out = self.gca(q=Q_L, kv=A_I, attn_mask=attn_mask)
            attn_out = attn_out[:, valid_mask, :]

        assert attn_out.shape == {'token': A_I_shape_orig, 'atom': Q_L_shape_orig}[self.query_type], 'Output shape mismatch: '\
            f'{attn_out.shape}, neither {Q_L_shape_orig} nor {A_I_shape_orig}'
        
        return attn_out

    @staticmethod
    def ungroup_atoms(Q_L, tok_idx):
        """
        Reshapes atom-level features to a fixed number of atoms per token and returns the valid mask.
        
        Args:
            Q_L (Tensor): [B, n_atoms, c_atom]
            tok_idx (Tensor): [n_atoms] with token index (assumed nonnegative integers)
        
        Returns:
            Q_L_expanded (Tensor): [B, n_tokens, n_atom_per_tok, c_atom]
            valid_mask (Tensor): [n_tokens, n_atom_per_tok] (True for valid, unpadded atoms)
        """
        B, n_atoms, c_atom = Q_L.shape
        tokens, counts = torch.unique(tok_idx, return_counts=True)
        n_tokens = tokens.numel()
        n_atom_per_tok = int(counts.max().item())

        if n_atom_per_tok != 14:
            logger.info(f"n_atom_per_tok is not 14, it is: {n_atom_per_tok} ")
        
        # if n_atom_per_tok == counts.min().item():
        #     logger.info("n_atom_per_tok is constant: ", n_atom_per_tok)
        #     Q_L_expanded = Q_L.view(B, n_tokens, n_atom_per_tok, c_atom)
        #     valid_mask = torch.full((n_tokens, n_atom_per_tok), True, device=Q_L.device)
        #     return Q_L_expanded, valid_mask

        # Split atoms into groups based on the counts.
        # This returns a tuple of tensors each with shape [B, count, c_atom] for that token.
        Q_L_split = torch.split(Q_L, counts.tolist(), dim=-2)
        
        # Pad each group to the same shape (B, n_atom_per_tok, c_atom) and stack along [:, n_tok, : , :].
        Q_L_padded = [F.pad(Q_L_i, 
            (0, 0,                                  # zero left/right pad for -1 dimension 
            0, n_atom_per_tok - Q_L_i.shape[-2])    # zero left pad, n right pad for -2 dimension
        ) for Q_L_i in Q_L_split]
        Q_L_expanded = torch.stack(Q_L_padded, dim=1)
        
        # Build a valid mask (same for every batch element).
        # For each token, positions [0, count) are valid.
        atom_idxs = torch.arange(n_atom_per_tok, device=Q_L.device).unsqueeze(0).expand(n_tokens, -1)
        valid_mask = atom_idxs < counts[:, None]  # [n_tok, n_atom_per_tok] < [n_tok, 1] (of num atoms)
        
        Q_L_expanded = Q_L_expanded.contiguous()

        return Q_L_expanded, valid_mask

class CrossTalkBlock(nn.Module):
    def __init__(self,
        c_token,
        c_s,
        c_atompair,
        c_atom,
        c_z,
        atom_transformer,
        diffusion_transformer,
    ):
        super().__init__()

        self.process_q = nn.Sequential(
            linearNoBias(c_atom, c_token),
        )

        self.atom_transformer = AtomTransformer(
            c_atom=c_atom, c_atompair=c_atompair, **atom_transformer
        )

        self.c_token = c_token 

        self.process_s = nn.Sequential(
            nn.LayerNorm((c_s,)),
            linearNoBias(c_s, c_token),
        )

        self.a_to_s = linearNoBias(c_token, c_s)
        self.a_init_to_z_i = linearNoBias(c_token, c_z)
        self.a_init_to_z_j = linearNoBias(c_token, c_z)

        self.diffusion_transformer = DiffusionTransformer(
            c_token=c_token, c_s=c_s, c_tokenpair=c_atom, **diffusion_transformer
        )

        self.layer_norm_1 = nn.LayerNorm(c_token)
       
        self.process_a = nn.Sequential(
            linearNoBias(c_token, c_atom),
            nn.ReLU(),
        )

    def forward(self,
            Q_L, 
            C_L, 
            P_LL, 
            A_I, 
            S_I, 
            Z_II, 
            f,
            Beta_II,
        ):
        tok_idx = f["atom_to_token_map"]
        I = tok_idx.max() + 1

        Q_L = self.atom_transformer(Q_L, C_L, P_LL)

        A_I_shape = Q_L.shape[:-2] + (
            I,
            self.c_token,
        )

        # Aggregate per-atom representation to per-token representation
        _A_I = (
            torch.zeros(A_I_shape, device=Q_L.device, dtype=Q_L.dtype)
            .index_reduce(
                -2,
                f["atom_to_token_map"].long(),
                self.process_q(Q_L),
                "mean",
                include_self=False,
            )
            .clone()
        ) 

        # update A_I
        if A_I is not None:
            A_I = A_I + _A_I
        else:
            A_I = _A_I

        # update S_I
        S_I = S_I + self.a_to_s(A_I)

        # update Z_II
        if len(Z_II.shape) != 4:
            Z_II = Z_II[None]
            
        Z_II = Z_II + self.a_init_to_z_i(A_I).unsqueeze(-3) \
                    + self.a_init_to_z_j(A_I).unsqueeze(-2)

        # Full self-attention on token level
        A_I = A_I + self.process_s(S_I)
        A_I = self.diffusion_transformer(A_I, S_I, Z_II, Beta_II=Beta_II)
        A_I = self.layer_norm_1(A_I)

        # Broadcast from A_I to Q_L 
        Q_L = Q_L + self.process_a(A_I[..., tok_idx, :])
        
        return A_I, Q_L, S_I


class SequenceHead(nn.Module):
    def __init__(self, c_token):
        super(SequenceHead, self).__init__()
        
        # Distogram feature extraction
        self.dist_fc1 = nn.Linear(196, 128)
        self.dist_relu = nn.ReLU()
        self.dist_fc2 = nn.Linear(128, 64)

        # Embedding feature extraction
        self.embed_fc1 = nn.Linear(c_token, 128)
        self.embed_relu = nn.ReLU()
        self.embed_fc2 = nn.Linear(128, 64)

        # Fusion layer
        self.fusion_fc = nn.Linear(128, 32)

    def forward(self, A_I, Q_L, X_L, f):
        X_L = X_L.detach()
        A_I = A_I.detach()
        B, L, _ = X_L.shape

        max_res_id = f["atom_to_token_map"].max().item() + 1

        # Compute distograms
        residue_distogram = torch.zeros(B, max_res_id, 14, 14, device=X_L.device)
        for i in range(max_res_id):
            residue_mask = f["atom_to_token_map"] == i
            if residue_mask.sum() == 14:
                coords = X_L[:, residue_mask]  # (B, 14, 3)
                residue_distogram[:, i] = torch.cdist(coords, coords)

        # Flatten distogram
        dist_features = residue_distogram.view(B, max_res_id, 196)

        # Pass through separate MLPs
        dist_out = self.dist_fc1(dist_features)
        dist_out = self.dist_relu(dist_out)
        dist_out = self.dist_fc2(dist_out)

        embed_out = self.embed_fc1(A_I)
        embed_out = self.embed_relu(embed_out)
        embed_out = self.embed_fc2(embed_out)

        # Fusion via concatenation
        fused = torch.cat([dist_out, embed_out], dim=-1)
        Seq_I = self.fusion_fc(fused)

        return Seq_I