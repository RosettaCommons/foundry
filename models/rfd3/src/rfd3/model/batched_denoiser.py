"""Tensor-only padded inference, reusing RFD3's trained modules and state dict.

Static singles/pairs have singleton D. Dynamic tensors have explicit B,D axes;
no topology extraction, Python tensor branching, or feature-dictionary mutation
occurs here. The legacy forward remains available as a regression reference.
"""

import torch
from rfd3.model.layers.batched import (
    downcast,
    gather_nodes,
    mask_nodes,
    mask_pairs,
    pair_valid,
    pairformer,
    pool_atoms,
    select_neighbors,
    sparse_pairs,
    transformer_block,
    upcast,
)

from foundry.model.layers.attention import use_dense_sdpa


def encode_tokens(module, f, r, s, z, recycled):
    valid = f["token_valid"]
    for block in module.transition_1:
        s = mask_nodes(s + block(s), valid)
    b, d = r.shape[:2]
    z = z.expand(b, d, -1, -1, -1)
    parts = [z]
    if module.use_distogram:
        reps = gather_nodes(r, f["representative_atom"])
        if module.use_sinusoidal_distogram_embedder:
            fixed = gather_nodes(
                f["is_motif_atom_with_fixed_coord"][:, None, :, None],
                f["representative_atom"],
            ).squeeze(-1)
            same_time = fixed.unsqueeze(-1) == fixed.unsqueeze(-2)
            distances = module.dist_embedder(
                reps, (same_time & pair_valid(valid))[..., None]
            )
        else:
            distances = module.bucketize_fn(reps)
        parts.append(mask_pairs(distances, valid))
    if module.use_self:
        if recycled is None:
            recycled = z.new_zeros(*z.shape[:-1], module.n_bins_distogram)
        parts.append(mask_pairs(recycled, valid))
    z = mask_pairs(module.process_z(torch.cat(parts, -1)), valid)
    for block in module.transition_2:
        z = mask_pairs(z + block(z), valid)
    for block in module.pairformer_stack:
        s, z = pairformer(block, s, z, valid)
    return s, z


def denoise(module, X_noisy_L, t, f, Q_L_init, C_L, P_LL, S_I, Z_II, n_recycle=None):
    av, tv = f["atom_valid"], f["token_valid"]
    x = mask_nodes(X_noisy_L, av)
    atom_t = t[..., None] * (av & ~f["is_motif_atom_with_fixed_coord"].bool())[:, None]
    token_t = (
        t[..., None]
        * (tv & ~f["is_motif_token_with_fully_fixed_coord"].bool())[:, None]
    )
    # Existing scale_positions_in/out accept explicitly broadcast 4D times.
    uniform = module.scale_positions_in(x, t[..., None, None])
    r = module.scale_positions_in(x, atom_t[..., None])
    a = pool_atoms(module.process_a.linear(r), f)
    s = downcast(module.downcast_c, C_L, S_I, None, f)
    q = mask_nodes(Q_L_init + module.process_r(r), av)
    c = mask_nodes(C_L + module.process_time_(atom_t, 0), av)
    s = mask_nodes(s + module.process_time_(token_t, 1), tv)
    c = mask_nodes(c + module.process_c(c), av)
    indices, neighbor_valid = select_neighbors(
        x, av, f["atom_sequence_mask"], module.n_attn_keys,
        f.get("atom_forced_neighbors"),
    )
    gathered = isinstance(P_LL, dict)
    if gathered:
        P_LL = sparse_pairs(P_LL, f, indices, neighbor_valid)
    else:
        # Dense static pairs remain allocated. Project only selected neighbors.
        b, d, n, k = indices.shape
        channels = P_LL.shape[-1]
        P_LL = torch.gather(
            P_LL.expand(b, d, n, n, channels),
            3,
            indices[..., None].expand(b, d, n, k, channels),
        )
    for block in module.encoder.blocks:
        q = transformer_block(
            block, q, c, P_LL, indices, neighbor_valid, av, gathered=True
        )
    a = downcast(module.downcast_q, q, a, s, f)
    recycled, previous_x = None, x
    n_recycle = module.n_recycle if n_recycle is None else n_recycle
    if n_recycle < 1:
        raise ValueError("n_recycle must be positive")
    for _ in range(n_recycle):
        # Recycling restarts from the same encoder representations. Only predicted
        # coordinates/distograms are recycled, exactly as in legacy process_.
        si, zi = encode_tokens(
            module.diffusion_token_encoder, f, uniform, s, Z_II, recycled
        )
        ti, tm = select_neighbors(
            gather_nodes(previous_x, f["representative_atom"]),
            tv,
            f["token_sequence_mask"],
            module.diffusion_transformer.n_keys,
            f.get("token_forced_neighbors"),
        )
        ai = a
        for block in module.diffusion_transformer.blocks:
            ai = transformer_block(
                block,
                ai,
                si,
                zi,
                ti,
                tm,
                tv,
                full=not gathered,
                sdpa=use_dense_sdpa(block.attention_pair_bias, ai),
            )
        qi = q
        for up, block in zip(module.decoder.upcast, module.decoder.atom_transformer):
            qi = upcast(up, qi, ai, f)
            qi = transformer_block(
                block, qi, c, P_LL, indices, neighbor_valid, av, gathered=True
            )
        ai = downcast(module.decoder.downcast, qi.detach(), ai.detach(), si.detach(), f)
        previous_x = mask_nodes(
            module.scale_positions_out(module.to_r_update(qi), x, atom_t[..., None]), av
        )
        logits = mask_nodes(module.sequence_head.linear(ai), tv)
        probabilities = logits.softmax(-1) * module.sequence_head.valid_out_mask
        selected = torch.where(tv[:, None], probabilities.argmax(-1), 0)
        recycled = mask_pairs(
            module.bucketize_fn(
                gather_nodes(previous_x.detach(), f["representative_atom"])
            ),
            tv,
        )
    return dict(X_L=previous_x, sequence_logits_I=logits, sequence_indices_I=selected)
