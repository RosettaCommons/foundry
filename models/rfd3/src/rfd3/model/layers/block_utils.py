import logging
from typing import Tuple

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int

logger = logging.getLogger(__name__)


def bucketize_scaled_distogram(R_L, min_dist=1, max_dist=30, sigma_data=16, n_bins=65):
    """
    Bucketizes pairwise distances into bins based on edm scaling

    min dist and max dist given as angstroms
    Will use bin ranges based on scaled angstrom distances

    R_L: B, N, 3
    D_LL: B, N, N
    D_LL_binned: B, N, N, n_bins
    """
    D_LL = R_L.unsqueeze(-2) - R_L.unsqueeze(-3)  # [B, N, N, 3]
    D_LL = torch.linalg.norm(D_LL, dim=-1)  # [B, N, N]

    # normalize
    min_dist, max_dist = min_dist / sigma_data, max_dist / sigma_data

    bins = torch.linspace(min_dist, max_dist, n_bins - 1, device=D_LL.device)
    bin_idxs = torch.bucketize(D_LL, bins)
    return F.one_hot(bin_idxs, num_classes=len(bins) + 1).float()


def build_valid_mask(
    tok_idx: torch.Tensor, n_atoms_per_tok_max: int | None = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args
    ----
    tok_idx : (n_atoms,)  non negative integer array
    n_atoms_per_tok_max : if given, pad/truncate up to this size

    Returns
    -------
    valid_mask : (n_tokens, A)  True where an atom exists
    tokens     : (n_tokens,)    the unique token IDs in ascending order
    """
    tokens, counts = torch.unique(tok_idx, return_counts=True)
    A = int(counts.max()) if n_atoms_per_tok_max is None else int(n_atoms_per_tok_max)

    # build [n_tokens, A] mask; broadcasting keeps it vectorised
    atom_idx_grid = torch.arange(A, device=tok_idx.device)[None, :]  # (1, A)
    valid_mask = atom_idx_grid < counts[:, None]  # (n_tok, A)

    return valid_mask


def ungroup_atoms(Q_L, valid_mask):
    """
    Args
    ----
    Q_L        : (B, n_atoms, c)
    valid_mask : (n_tokens, A)          # same object returned by `ungroup_atoms`

    Returns
    -------
    Q_IA       : (B, n_tokens, A, c)    # padded with zeros
    """
    B, n_atoms, c = Q_L.shape
    n_tokens, A = valid_mask.shape
    Q_IA = torch.zeros(B, n_tokens, A, c, dtype=Q_L.dtype, device=Q_L.device)
    mask4d = valid_mask.unsqueeze(0).unsqueeze(-1)  # (1, n_tok, A, 1)
    mask4d = mask4d.expand(B, -1, -1, c)  # (B, n_tok, A, c)
    Q_IA.masked_scatter_(mask4d, Q_L)
    return Q_IA


def group_atoms(Q_IA: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """
    Args
    ----
    Q_IA       : (B, n_tokens, A, c)
    valid_mask : (n_tokens, A)

    Returns
    -------
    Q_L        : (B, n_atoms, c)  flattened real atoms, order preserved
    """
    B, _, _, c = Q_IA.shape
    mask4d = valid_mask.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, c)  # (B,n_tok,A,c)
    Q_L = Q_IA[mask4d].view(B, -1, c)  # restore 2‑D shape
    return Q_L


def group_pair(P_IAA, valid_mask):
    # Valid mask: [L, A]
    # P_IAA: (B, L, A, A, c) or (L, A, A, c)
    if P_IAA.ndim == 5:
        B, _, _, A, c = P_IAA.shape
        mask5d = valid_mask[None, ..., None, None].expand(
            B, -1, -1, A, c
        )  # (B, L, L, A, c)
        P_LA = P_IAA[mask5d].view(B, -1, A, c)  # (B, n_valid, A, c)
    elif P_IAA.ndim == 4:
        _, _, A, c = P_IAA.shape
        mask4d = valid_mask[..., None, None].expand(-1, -1, A, c)  # (L, L, A, c)
        P_LA = P_IAA[mask4d].view(-1, A, c)  # (n_valid, A, c)
    else:
        raise ValueError(
            f"Unexpected input shape {P_IAA.shape}: must be (B, L, A, A, c) or (L, A, A, c)"
        )

    return P_LA


def scatter_add_pair_features(P_LK_tgt, P_LK_indices, P_LA_src, P_LA_indices):
    """
    Adds features from P_LA_C into P_LK_C at positions where P_LA matches P_LK.

    Parameters
    ----------
    P_LK_indices   : (D, L, k) LongTensor
        Key indices | P_LK_indices[d, i, k] = global atom index for which atom i attends to.
    P_LK : (D, L, k, c) FloatTensor
        Key features to scatter add into

    P_LA_indices   : (D, L, a) LongTensor
        Additional feature indices to scatter into P_LK.
    P_LA : (D, L, a, c) FloatTensor
        Features corresponding to P_LA.

    Both index tensors contain indices representing D batch dim,
    L sequence positions and k keys / a additional features.
    This function will scatter indices from P_LA into P_LK based on
    matching indices.

    """
    # Handle case when indices and P_LA don't have batch dimensions
    D, L, k = P_LK_indices.shape
    if P_LA_indices.ndim == 2:
        P_LA_indices = P_LA_indices.unsqueeze(0).expand(D, -1, -1)
    if P_LA_src.ndim == 3:
        P_LA_src = P_LA_src.unsqueeze(0).expand(D, -1, -1)
    assert (
        P_LA_src.shape[-1] == P_LK_tgt.shape[-1]
    ), "Channel dims do not match, got: {} vs {}".format(
        P_LA_src.shape[-1], P_LK_tgt.shape[-1]
    )

    matches = P_LA_indices.unsqueeze(-1) == P_LK_indices.unsqueeze(-2)  # (D, L, a, k)
    if not torch.all(matches.sum(dim=(-1, -2)) >= 1):
        raise ValueError("Found multiple scatter indices for some atoms")
    elif not torch.all(matches.sum(dim=-1) <= 1):
        raise ValueError("Did not find a scatter index for every atom")
    k_indices = matches.long().argmax(dim=-1)  # (D, L, a)
    scatter_indices = k_indices.unsqueeze(-1).expand(
        -1, -1, -1, P_LK_tgt.shape[-1]
    )  # (D, L, a, c)
    P_LK_tgt = P_LK_tgt.scatter_add(dim=2, index=scatter_indices, src=P_LA_src)
    return P_LK_tgt


def _batched_gather(values: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    values : (D, L, C)
    idx    : (D, L, k)
    returns: (D, L, k, C)
    """
    D, L, C = values.shape
    k = idx.shape[-1]

    #   (D, L, 1, C)  → stride-0 along k  → (D, L, k, C)
    src = values.unsqueeze(2).expand(-1, -1, k, -1)
    idx = idx.unsqueeze(-1).expand(-1, -1, -1, C)  # (D, L, k, C)

    return torch.gather(src, 1, idx)  # dim=1 is the L-axis


@torch.no_grad()
def create_attention_indices(
    f, n_attn_keys, n_attn_seq_neighbours, X_L=None, tok_idx=None
):
    """
    Entry-point function for creating attention indices for sequence & structure-local attention

    f: input features of the model
    n_attn_keys: number of (atom) attention keys
    n_attn_seq_neighbours: number of neighbouring sequence tokens (residues) to attend to
    X_L: optional input tensor for atom positions | if None, choose random padding atoms
    """

    tok_idx = f["atom_to_token_map"] if tok_idx is None else tok_idx
    device = X_L.device if X_L is not None else tok_idx.device
    L = len(tok_idx)

    if X_L is None:
        X_L = torch.randn(
            (1, L, 3), device=device, dtype=torch.float
        )  # [L, 3] - random
    D_LL = torch.cdist(X_L, X_L, p=2)  # [D, L, L] - pairwise atom distances

    # Create attention indices using neighbour distances
    base_mask = ~f["unindexing_pair_mask"][
        tok_idx[None, :], tok_idx[:, None]
    ]  # [n_atoms, n_atoms]
    k_actual = min(n_attn_keys, L)

    # For symmetric structures, ensure inter-chain interactions are included
    chain_ids = f["asym_id"][tok_idx] if "asym_id" in f else None
    if (
        chain_ids is not None and len(torch.unique(chain_ids)) > 3
    ):  # Multi-chain structure
        # Reserve 25% of attention keys for inter-chain interactions
        k_inter_chain = max(32, k_actual // 4)  # At least 32 inter-chain keys
        k_intra_chain = k_actual - k_inter_chain

        attn_indices = get_sparse_attention_indices_with_inter_chain(
            tok_idx,
            D_LL,
            n_seq_neighbours=n_attn_seq_neighbours,
            k_intra=k_intra_chain,
            k_inter=k_inter_chain,
            chain_id=chain_ids,
            base_mask=base_mask,
        )
    else:
        # Regular attention for single chain or small structures
        attn_indices = get_sparse_attention_indices(
            tok_idx,
            D_LL,
            n_seq_neighbours=n_attn_seq_neighbours,
            k_max=k_actual,
            chain_id=chain_ids,
            base_mask=base_mask,
        )  # [D, L, k] | indices[b, i, j] = atom index for atom i to j-th attn query

    return attn_indices


@torch.no_grad()
def get_sparse_attention_indices_with_inter_chain(
    tok_idx, D_LL, n_seq_neighbours, k_intra, k_inter, chain_id, base_mask
):
    """
    Create attention indices that guarantee inter-chain interactions for clash avoidance.

    Args:
        tok_idx: atom to token mapping
        D_LL: pairwise distances [D, L, L]
        n_seq_neighbours: number of sequence neighbors
        k_intra: number of intra-chain attention keys
        k_inter: number of inter-chain attention keys
        chain_id: chain IDs for each atom
        base_mask: base mask for valid pairs

    Returns:
        attn_indices: [D, L, k_total] where k_total = k_intra + k_inter
    """
    D, L, _ = D_LL.shape

    # Get regular intra-chain indices (limited to k_intra)
    intra_indices = get_sparse_attention_indices(
        tok_idx, D_LL, n_seq_neighbours, k_intra, chain_id, base_mask
    )  # [D, L, k_intra]

    # Get inter-chain indices for clash avoidance
    inter_indices = torch.zeros(D, L, k_inter, dtype=torch.long, device=D_LL.device)

    for d in range(D):
        for l in range(L):
            query_chain = chain_id[l]

            # Find atoms from different chains
            other_chain_mask = (chain_id != query_chain) & base_mask[l, :]
            other_chain_atoms = torch.where(other_chain_mask)[0]

            if len(other_chain_atoms) > 0:
                # Get distances to other chains
                distances_to_other = D_LL[d, l, other_chain_atoms]

                # Select k_inter closest atoms from other chains
                n_select = min(k_inter, len(other_chain_atoms))
                _, closest_idx = torch.topk(distances_to_other, n_select, largest=False)
                selected_atoms = other_chain_atoms[closest_idx]

                # Fill inter-chain indices
                inter_indices[d, l, :n_select] = selected_atoms
                # Pad with random atoms if needed
                if n_select < k_inter:
                    padding = torch.randint(
                        0, L, (k_inter - n_select,), device=D_LL.device
                    )
                    inter_indices[d, l, n_select:] = padding
            else:
                # No other chains found, fill with random indices
                inter_indices[d, l, :] = torch.randint(
                    0, L, (k_inter,), device=D_LL.device
                )

    # Combine intra and inter chain indices
    combined_indices = torch.cat(
        [intra_indices, inter_indices], dim=-1
    )  # [D, L, k_total]

    return combined_indices


@torch.no_grad()
def build_index_mask(
    tok_idx: torch.Tensor,
    n_sequence_neighbours: int,
    k_max: int,
    chain_id: torch.Tensor | None = None,
    base_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Builds a mask that includes entire tokens from neighboring positions within a
    tokenized sequence, never partially including a token. Limits range to k_max,
    which is interpreted at the token level.

    Parameters:
        tok_idx: (L,) tensor of token indices.
        n_sequence_neighbours: number of tokens to include on either side.
        k_max: max total number of tokens (across both directions).
        chain_id: (L,) chain identifiers for each position (optional).
        base_mask: (L, L) optional pre-mask to AND with.
    """
    device = tok_idx.device
    L = tok_idx.shape[0]
    k_max = min(k_max, L)
    I = int(tok_idx.max()) + 1  # Number of unique tokens
    n_atoms_per_token = torch.zeros(I, device=device).float()
    n_atoms_per_token.scatter_add_(0, tok_idx.long(), torch.ones_like(tok_idx).float())

    # Create index masks for tokens and atoms
    token_indices = torch.arange(I, device=device)
    token_diff = (token_indices[:, None] - token_indices[None, :]).abs()
    atom_indices = torch.arange(L, device=device)
    atom_diff = (atom_indices[:, None] - atom_indices[None, :]).abs()

    # Build token-token mask: [I, I]
    token_mask = token_diff <= n_sequence_neighbours

    # Expand token_mask to full [L, L] mask using broadcast
    # token_to_idx maps each position to a token index [L]
    token_i = tok_idx[:, None]  # (L, 1)
    token_j = tok_idx[None, :]  # (1, L)
    mask = token_mask[token_i, token_j]  # (L, L)
    mask = mask & (atom_diff <= (k_max // 2))

    # Exclude tokens which are partially filled (L, I)
    n_query_per_token = torch.zeros((L, I), device=device).float()
    n_query_per_token.scatter_add_(
        1, tok_idx.long()[None, :].expand(L, -1), mask.float()
    )

    # Find mask for the atoms for which the number of keys
    # match the number of atoms in the token (L, I)
    fully_included = n_query_per_token == n_atoms_per_token[None, :]

    # Contract to (L, L) and count the number of atoms within tokens that
    # fully include other tokens
    n_atoms_fully_included = torch.zeros((I, I), device=device)
    n_atoms_fully_included.index_add_(0, tok_idx.long(), fully_included.float())
    full_token_mask = n_atoms_fully_included == n_atoms_per_token[:, None]

    # Map this back to (L, L) — include token j in row i only if all its atoms are included
    full_token_mask = full_token_mask[token_i, token_j]  # (L, L)
    mask &= full_token_mask

    if chain_id is not None:
        same_chain = chain_id.unsqueeze(-1) == chain_id.unsqueeze(-2)
        mask = mask & same_chain

    if base_mask is not None:
        mask = mask & base_mask

    return mask


def extend_index_mask_with_neighbours(
    mask: torch.Tensor, D_LL: torch.Tensor, k: int
) -> torch.LongTensor:
    """
    Parameters
    ----------
    mask   : (L, L) bool                # pre-selected neighbours (True = keep)
    D_LL   : (B, L, L) float32/float64  # pairwise distances (lower = closer)
    k: int                        # desired neighbours per query token

    Returns
    -------
    neigh_idx : (L, k_neigh) long       # exactly k_neigh indices per row

    NB: Indices of the mask are placed first along k dimension. e.g.
           indices[i, :] = [1, 2, 3, nan, nan] (from pre-built mask)
        -> indices[i, :] = [1, 2, 3, 0, 5]  # where 0, 5 are additional k NN (here k=5)
    NB: If k_neigh = 14 * (2*n_seq_neigh + 1) (from above), then for tokens in the middle there will
        be exactly no D_LL-local neighbours, but for tokens at the edges there will be an increasingly
        large number of neighbours.
    """
    if D_LL.ndim == 2:
        D_LL = D_LL.unsqueeze(0)
    B, L, _ = D_LL.shape
    k = min(k, L)
    assert mask.shape == (L, L) and D_LL.shape == (B, L, L)
    device = D_LL.device
    inf = torch.tensor(float("inf"), dtype=D_LL.dtype, device=device)

    # 1. Selection of sequence neighbours
    all_idx_row = torch.arange(L, device=device).expand(L, L)
    indices = torch.where(mask, all_idx_row, inf)  # sentinel inf if not-forced
    indices = indices.sort(dim=1)[0][:, :k]  # (L, k)

    # 2. Find k-nn excluding forced indices
    D_LL = torch.where(mask, inf, D_LL)
    filler_idx = torch.topk(D_LL, k, dim=-1, largest=False).indices

    # ... Reverse last axis s.t. best matched indices are last
    filler_idx = filler_idx.flip(dims=[-1])

    # 3. Fill indices
    to_fill = indices == inf
    to_fill = to_fill.expand_as(filler_idx)
    indices = indices.expand_as(filler_idx)
    indices = torch.where(to_fill, filler_idx, indices)

    return indices.long()  # (B, L, k)


def get_sparse_attention_indices(
    res_idx, D_LL, n_seq_neighbours, k_max, chain_id=None, base_mask=None
):
    mask = build_index_mask(
        res_idx, n_seq_neighbours, k_max, chain_id=chain_id, base_mask=base_mask
    )
    indices = extend_index_mask_with_neighbours(mask, D_LL, k_max)

    # Sort and assert no duplicates (optional but good practise)
    indices, _ = torch.sort(indices, dim=-1)
    if (indices[..., 1:] == indices[..., :-1]).any():
        raise AssertionError("Tensor has duplicate elements along the last dimension.")

    assert (
        indices.shape[-1] == k_max
    ), f"Expected k_max={k_max} indices, got {indices.shape[-1]} instead."
    # Detach to avoid gradients flowing through indices

    return indices.detach()


@torch.no_grad()
def indices_to_mask(neigh_idx):
    """
    Helper function for converting indices to masks for visualization

    Args:
        neigh_idx: [L, k] or [B, L, k] tensor of indices for attention.
    """
    neigh_idx = neigh_idx.to(dtype=torch.long)

    if neigh_idx.ndim == 2:
        L = neigh_idx.shape[0]
        mask_out = torch.zeros((L, L), dtype=torch.bool, device=neigh_idx.device)
        mask_out.scatter_(1, neigh_idx, torch.ones_like(neigh_idx, dtype=torch.bool))

    elif neigh_idx.ndim == 3:
        B, L, k = neigh_idx.shape
        mask_out = torch.zeros((B, L, L), dtype=torch.bool, device=neigh_idx.device)
        mask_out.scatter_(2, neigh_idx, torch.ones_like(neigh_idx, dtype=torch.bool))

    else:
        raise ValueError(f"Expected ndim 2 or 3, got {neigh_idx.ndim}")

    return mask_out


def create_valid_mask_LA(valid_mask):
    """
    Helper function for X_IAA (token-grouped atom-pair representations).
    valid_mask: [I, A] represents which atoms in the token-grouping are real,
        sum(valid_mask) = L, where L is total number of atoms.

    Returns
    -------
    valid_mask_LA: [L, A] L atoms by A atoms in token grouping.
    indices: [L, A] absolute atom indices of atoms in token grouping.

    E.g. Allows you to have [14, 14] matrices for every token in your protein,
    where atomized tokens (or similar) will have invalid indices outside of [0,0].
    """
    I, A = valid_mask.shape
    L = valid_mask.sum()
    pos = torch.arange(A, device=valid_mask.device)
    rel_pos = pos.unsqueeze(-2) - pos.unsqueeze(-1)  # [A, A]
    rel_pos = rel_pos.unsqueeze(0).expand(I, -1, -1)  # [I, A, A]
    rel_pos_LA = rel_pos[valid_mask[..., None].expand_as(rel_pos)].view(
        L, A
    )  # [I, A, A] -> [L, A]

    indices = torch.arange(L, device=valid_mask.device).unsqueeze(-1).expand(L, A)
    indices = indices + rel_pos_LA

    valid_mask_IAA = valid_mask.unsqueeze(-2).expand(-1, A, -1)
    valid_mask_LA = valid_mask_IAA[
        valid_mask.unsqueeze(-1).expand_as(valid_mask_IAA)
    ].view(L, A)

    indices[~valid_mask_LA] = -1

    return valid_mask_LA, indices


def pairwise_mean_pool(
    pairwise_atom_features: Float[torch.Tensor, "batch n_atoms n_atoms d_hidden"],
    atom_to_token_map: Int[torch.Tensor, "n_atoms"],
    I: int,
    dtype: torch.dtype,
) -> Float[torch.Tensor, "batch n_tokens n_tokens d_hidden"]:
    """Mean pooling of pairwise atom features to pairwise token features.

    Args:
        pairwise_atom_features: Pairwise features between atoms
        atom_to_token_map: Mapping from atoms to tokens
        I: Number of tokens
        dtype: Data type for computations

    Returns:
        Token pairwise features pooled by averaging over atom pairs within tokens
    """
    B, _, _, _ = pairwise_atom_features.shape

    # Create one-hot encoding for atom-to-token mapping
    atom_to_token_onehot = F.one_hot(atom_to_token_map.long(), num_classes=I).to(
        dtype
    )  # (L, I)

    # Use einsum to aggregate features across atom pairs for each token pair
    # For each token pair (i, j), sum over all atom pairs (l1, l2) where l1→i and l2→j
    # Result[b,i,j,d] = sum_l1,l2 ( onehot[l1,i] * onehot[l2,j] * features[b,l1,l2,d] )
    use_memory_efficient_einsum = True
    if use_memory_efficient_einsum:
        # Memory-optimized implementation using two-step einsum:
        # First step: contract on axis 1 (left-side tokens)
        # (L, I)^T = (I, L), (B, L, L, d) → (B, I, L, d)
        temp = torch.einsum(
            "ia,bacd->bicd", atom_to_token_onehot.T, pairwise_atom_features
        )

        # Free the original to save memory if not needed
        del pairwise_atom_features

        # Second step: contract on axis 2 (right-side tokens)
        # (L, I) = (L, I), (B, I, L, d) → (B, I, I, d)
        token_features_sum = torch.einsum("cj,bicd->bijd", atom_to_token_onehot, temp)

        # Optionally free temp
        del temp
    else:
        token_features_sum = torch.einsum(
            "ai,cj,bacd->bijd",
            atom_to_token_onehot,  # (L, I)
            atom_to_token_onehot,  # (L, I)
            pairwise_atom_features,  # (B, L, L, d_hidden)
        )  # (B, I, I, d_hidden)

    # Count the number of atom pairs contributing to each token pair
    # count[i, j] = number of atom pairs (l1, l2) where l1→i and l2→j (same for all batches)
    atom_counts_per_token = atom_to_token_onehot.sum(dim=0)  # (I,)
    token_pair_counts = torch.outer(
        atom_counts_per_token, atom_counts_per_token
    )  # (I, I) (= outer product)

    # Expand to match batch dimension: (I, I) -> (B, I, I)
    token_pair_counts = token_pair_counts.unsqueeze(0).expand(B, -1, -1)

    # Avoid division by zero and compute mean
    token_pair_counts = torch.clamp(token_pair_counts, min=1)
    token_pairwise_features = token_features_sum / token_pair_counts.unsqueeze(-1)

    return token_pairwise_features
