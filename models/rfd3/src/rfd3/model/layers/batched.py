"""Padding-safe tensor primitives. Singles have explicit (B,D,N,C) axes.

Only preprocessing builds topology. These functions contain no data-dependent
Python branches or variable-size boolean indexing and may be captured fullgraph.
"""

import torch
import torch.nn.functional as F
from rfd3.model import inference_acceleration as accel
from rfd3.model.kernel_ops import padded_attention

from foundry.model.layers.attention import dense_pair_attention, use_dense_sdpa


@torch.library.custom_op("rfd3::sort_neighbor_values", mutates_args=())
def _sort_neighbor_values(x: torch.Tensor) -> torch.Tensor:
    """Keep integer sorting out of Inductor's large fan-out fusion kernels."""
    return x.sort(-1).values.contiguous()


@_sort_neighbor_values.register_fake
def _sort_neighbor_values_fake(x):
    return torch.empty_like(x, memory_format=torch.contiguous_format)


def sorted_neighbors(x):
    if torch.compiler.is_compiling() and x.is_cuda:
        return _sort_neighbor_values(x)
    return x.sort(-1).values


def mask_nodes(x, valid):
    return torch.where(valid[:, None, :, None], x, 0)


def pair_valid(valid):
    return valid[:, None, :, None] & valid[:, None, None, :]


def mask_pairs(x, valid):
    return torch.where(pair_valid(valid)[..., None], x, 0)


def gather_nodes(x, indices):
    """(B,D,N,C), (B,...) -> (B,D,...,C), never across examples."""
    b, d, _, c = x.shape
    flat = indices.reshape(b, -1)
    index = flat[:, None, :, None].expand(b, d, flat.shape[-1], c)
    return torch.gather(x, 2, index).reshape(b, d, *indices.shape[1:], c)


def group_atoms(x, f):
    grouped = gather_nodes(mask_nodes(x, f["atom_valid"]), f["atom_slots"])
    return torch.where(f["slot_valid"][:, None, ..., None], grouped, 0)


def ungroup_atoms(x, f):
    return mask_nodes(
        gather_nodes(x.flatten(2, 3), f["atom_slot_inverse"]), f["atom_valid"]
    )


def pool_atoms(x, f):
    grouped = group_atoms(x, f)
    count = f["slot_valid"].sum(-1).clamp(min=1)
    return mask_nodes(grouped.sum(-2) / count[:, None, :, None], f["token_valid"])


def pool_pairs(x, f):
    i = f["token_valid"].shape[-1]
    assignment = F.one_hot(f["atom_to_token_map"], i).to(x.dtype)
    assignment = assignment * f["atom_valid"][..., None]
    x = mask_pairs(x, f["atom_valid"])
    sums = torch.einsum("bli,bdlmc,bmj->bdijc", assignment, x, assignment)
    count = assignment.sum(1)
    denominator = (count[:, :, None] * count[:, None, :]).clamp(min=1)
    return mask_pairs(sums / denominator[:, None, :, :, None], f["token_valid"])


def masked_softmax(logits, valid):
    has_keys = valid.any(-1, keepdim=True)
    safe = torch.where(has_keys, logits.masked_fill(~valid, float("-inf")), 0)
    return torch.where(valid, safe.softmax(-1), 0)


def select_neighbors(x, valid, sequence_mask, k_max, forced_neighbors=None):
    """Legacy forced-sequence + nearest-fill policy, with explicit padded slots."""
    b, d, n, _ = x.shape
    k = min(k_max, n)
    x = mask_nodes(x, valid)
    distances = torch.cdist(x, x)
    allowed = pair_valid(valid)
    forced = sequence_mask[:, None] & allowed
    distances = distances.masked_fill(forced | ~allowed, float("inf"))
    nearest = distances.topk(k, largest=False, dim=-1).indices
    effective_k = valid.sum(-1).clamp(max=k)
    position = torch.arange(k, device=x.device)
    reverse = (effective_k[:, None] - 1 - position).clamp(min=0)
    nearest = nearest.gather(-1, reverse[:, None, None].expand(b, d, n, k))
    if forced_neighbors is None:
        grid = torch.arange(n, device=x.device)
        selected = sorted_neighbors(torch.where(forced, grid, n))[..., :k]
    else:
        selected = forced_neighbors[:, None]
    selected = selected.expand(b, d, n, k)
    selected = torch.where(selected == n, nearest, selected)
    slot_valid = (
        valid[:, None, :, None] & (position < effective_k[:, None])[:, None, None]
    )
    selected = torch.where(slot_valid, selected, n)
    selected = sorted_neighbors(selected)
    neighbor_valid = (selected < n) & valid[:, None, :, None]
    return selected.clamp(max=n - 1), neighbor_valid


def cross_attention(module, q, kv, valid):
    """Grouped attention: (B,D,I,N,C), mask (B,1,I,Q,K)."""
    q, kv = module.ln_q(q), module.ln_kv(kv)
    q, k, v, g = module.to_q(q), module.to_k(kv), module.to_v(kv), module.to_g(q)
    if module.kq_norm:
        q, k = module.q_norm(q), module.k_norm(k)
    h = module.n_head
    q, k, v, g = [x.unflatten(-1, (h, -1)).transpose(-3, -2) for x in (q, k, v, g)]
    logits = (q @ k.transpose(-1, -2)) * module.scale
    p = masked_softmax(logits, valid.unsqueeze(-3))
    out = (p @ v) * g
    out = module.to_out(out.transpose(-3, -2).flatten(-2))
    return torch.where(valid.any(-1, keepdim=True), out, 0)


def downcast(module, q, a, s, f):
    grouped = group_atoms(q, f)
    if module.method == "mean":
        count = f["slot_valid"].sum(-1).clamp(min=1)
        update = module.project(grouped).sum(-2) / count[:, None, :, None]
    else:
        update = cross_attention(
            module.gca, a.unsqueeze(-2), grouped, f["slot_valid"][:, None, :, None, :]
        ).squeeze(-2)
    a = a + update if a is not None else update
    if module.process_s is not None:
        a = a + module.process_s(s)
    return mask_nodes(a, f["token_valid"])


def upcast(module, q, a, f):
    grouped = group_atoms(q, f)
    if module.method == "broadcast":
        grouped = grouped + module.project(a).unsqueeze(-2)
    else:
        split = a.unflatten(-1, (module.n_split, -1))
        valid = f["slot_valid"][:, None, :, :, None].expand(
            -1, 1, -1, -1, module.n_split
        )
        grouped = grouped + cross_attention(module.gca, grouped, split, valid)
    return ungroup_atoms(grouped, f)


def local_attention(
    module,
    q,
    s,
    pairs,
    indices,
    neighbor_valid,
    valid,
    *,
    full=False,
    gathered=False,
    sdpa=False,
):
    q = mask_nodes(q, valid)
    q = module.ada_ln_1(q, s) if s is not None else module.ln_1(q)
    q, k, v, g = module.to_q(q), module.to_k(q), module.to_v(q), module.to_g(q)
    if module.kq_norm:
        q, k = module.ln_q(q), module.ln_k(k)
    b, d, n, c = q.shape
    h, keys = module.n_head, indices.shape[-1]
    bias = module.to_b(pairs).expand(b, d, n, keys if gathered else n, h)
    if full or sdpa:
        allowed = torch.zeros((b, d, n, n), dtype=torch.bool, device=q.device)
        # Invalid slots can share an index with a real key. scatter_add prevents
        # a False padding slot from overwriting that key's validity.
        counts = torch.zeros_like(allowed, dtype=torch.int32).scatter_add(
            -1, indices, neighbor_valid.to(torch.int32)
        )
        allowed = counts > 0
        q, k, v, g = [
            x.unflatten(-1, (h, c // h)).transpose(-3, -2) for x in (q, k, v, g)
        ]
        if sdpa:
            attended = dense_pair_attention(
                q,
                k,
                v,
                bias.movedim(-1, -3),
                scale=(c // h) ** -0.5,
                allowed=allowed.unsqueeze(-3),
            )
        else:
            logits = (q @ k.transpose(-1, -2)) / (c // h) ** 0.5
            logits = logits + bias.movedim(-1, -3)
            p = masked_softmax(logits, allowed.unsqueeze(-3))
            attended = p @ v
        result = (attended * g).transpose(-3, -2).flatten(-2)
    elif (
        accel.enabled(module, q)
        and q.dtype == k.dtype
        and all(x.dtype in (torch.float32, torch.bfloat16) for x in (q, v, bias, g))
    ):
        if not gathered:
            bias = bias.gather(3, indices[..., None].expand(b, d, n, keys, h))
        result = padded_attention(q, k, v, bias, indices, neighbor_valid, g, h)
    else:
        batch = torch.arange(b, device=q.device)[:, None, None, None]
        diffusion = torch.arange(d, device=q.device)[None, :, None, None]
        k = k[batch, diffusion, indices].unflatten(-1, (h, c // h))
        v = v[batch, diffusion, indices].unflatten(-1, (h, c // h))
        q = q.unflatten(-1, (h, c // h))
        if not gathered:
            bias = bias.gather(3, indices[..., None].expand(b, d, n, keys, h))
        logits = (q.unsqueeze(-3) * k).sum(-1) / (c // h) ** 0.5 + bias
        p = masked_softmax(
            logits.transpose(-1, -2), neighbor_valid.unsqueeze(-2)
        ).transpose(-1, -2)
        result = (p[..., None] * v).sum(-3).flatten(-2) * g
    result = module.to_o(result)
    if s is not None:
        result = module.linear_output_project(s) * result
    return mask_nodes(result, valid)


def transformer_block(
    module,
    q,
    s,
    pairs,
    indices,
    neighbor_valid,
    valid,
    full=False,
    gathered=False,
    sdpa=False,
):
    q = mask_nodes(q, valid)
    q = mask_nodes(
        q
        + module.dropout(
            local_attention(
                module.attention_pair_bias,
                q,
                s,
                pairs,
                indices,
                neighbor_valid,
                valid,
                full=full,
                gathered=gathered,
                sdpa=sdpa,
            )
        ),
        valid,
    )
    update = (
        module.transition_block(q, s) if s is not None else module.transition_block(q)
    )
    return mask_nodes(q + update, valid)


def pairformer(module, s, z, valid):
    # Match the existing pairformer's precision policy on each backend.
    with torch.amp.autocast(
        device_type=s.device.type,
        enabled=s.device.type != "mps" and module.attention_pair_bias.force_bfloat16,
        dtype=torch.bfloat16,
    ):
        z = mask_pairs(z + module.z_transition(z), valid)
        attn = module.attention_pair_bias
        normalized = attn.ln_1(s)
        if (attn.use_deepspeed_evo or attn.force_bfloat16) and s.device.type != "mps":
            normalized = normalized.to(torch.bfloat16)
        q, k, v = attn.to_q(normalized), attn.to_k(normalized), attn.to_v(normalized)
        bias = attn.to_b(attn.ln_0(z))
        q = q / torch.sqrt(torch.tensor(attn.c, device=q.device, dtype=q.dtype))
        if use_dense_sdpa(attn, q):
            update = dense_pair_attention(
                q.transpose(-3, -2),
                k.transpose(-3, -2),
                v.transpose(-3, -2),
                bias.movedim(-1, -3),
                scale=1.0,
                allowed=pair_valid(valid).unsqueeze(-3),
            ).transpose(-3, -2)
        else:
            scores = torch.einsum("...ihc,...jhc->...ijh", q, k) + bias
            p = masked_softmax(
                scores.movedim(-1, -3), pair_valid(valid).unsqueeze(-3)
            ).movedim(-3, -1)
            update = torch.einsum("...ijh,...jhc->...ihc", p, v)
        update = update * attn.to_g(normalized)
        s = mask_nodes(s + attn.to_a(update.flatten(-2)), valid)
        s = mask_nodes(s + module.s_transition(s), valid)
    return s, z


def embed_features(module, f, valid):
    b, n = valid.shape
    result = sum(
        layer(f[key].float().reshape(b, n, -1))
        for key, layer in module.embedders.items()
    )
    return mask_nodes(result[:, None], valid)


def token_pairs_to_atoms(z, f):
    b, d, _, _, c = z.shape
    tok = f["atom_to_token_map"]
    batch = torch.arange(b, device=z.device)[:, None, None, None]
    diffusion = torch.arange(d, device=z.device)[None, :, None, None]
    return mask_pairs(
        z[batch, diffusion, tok[:, None, :, None], tok[:, None, None, :]],
        f["atom_valid"],
    )


def initialize(module, f):
    """Batched TokenInitializer using the checkpoint's existing parameters."""
    if module.atom_transformer is not None:
        raise ValueError("Static random atom attention requires the legacy path")
    av, tv = f["atom_valid"], f["token_valid"]
    s = embed_features(module.token_1d_embedder, f, tv)
    s = mask_nodes(s + module.transition_post_token(s), tv)
    s = downcast(
        module.downcast_atom,
        embed_features(module.atom_1d_embedder_1, f, av),
        s,
        None,
        f,
    )
    s = mask_nodes(module.process_s_init(s + module.transition_post_atom(s)), tv)
    z = module.to_z_init_i(s).unsqueeze(-3) + module.to_z_init_j(s).unsqueeze(-2)
    z = z + module.relative_position_encoding(f)[:, None]
    z = z + module.process_token_bonds(f["token_bonds"][:, None, ..., None].float())
    reps = f["representative_atom"]
    ref = gather_nodes(f["ref_pos"][:, None], reps)
    uid = gather_nodes(f["ref_space_uid"][:, None, :, None], reps).squeeze(-1)
    allowed = (uid.unsqueeze(-1) == uid.unsqueeze(-2)) & pair_valid(tv)
    z = mask_pairs(z + module.ref_pos_embedder_tok(ref, allowed[..., None]), tv)
    for block in module.transformer_stack:
        s, z = pairformer(block, s, z, tv)
    z = module.process_z_init(
        torch.cat([z, module.relative_position_encoding2(f)[:, None]], -1)
    )
    for block in module.transition_1:
        z = mask_pairs(z + block(z), tv)
    q = embed_features(module.atom_1d_embedder_2, f, av)
    c = mask_nodes(
        q + gather_nodes(module.process_s_trunk(s), f["atom_to_token_map"]), av
    )
    if module.use_chunked_pll:
        # Per-initialization cache: never mutate the legacy embedder's shared cache.
        cache = dict(
            module=module,
            sl=module.process_single_l(c),
            sm=module.process_single_m(c),
            z=module.process_z(z),
        )
        return dict(Q_L_init=q, C_L=c, P_LL=cache, S_I=s, Z_II=z)
    motif = f["is_motif_atom_with_fixed_coord"].bool() & av
    p = module.motif_pos_embedder(f["motif_pos"][:, None], pair_valid(motif)[..., None])
    uid = f["ref_space_uid"]
    same = (uid[:, :, None] == uid[:, None, :])[:, None]
    has_seq = f["is_motif_atom_with_fixed_seq"].bool() & av
    p = p + module.ref_pos_embedder(
        f["ref_pos"][:, None], (same & pair_valid(has_seq))[..., None]
    )
    p = (
        p
        + module.process_single_l(c).unsqueeze(-2)
        + module.process_single_m(c).unsqueeze(-3)
    )
    p = p + token_pairs_to_atoms(module.process_z(z), f)
    p = mask_pairs(p + module.pair_mlp(p), av)
    z = mask_pairs(z + module.project_pll(pool_pairs(module.process_pll(p), f)), tv)
    return dict(Q_L_init=q, C_L=c, P_LL=p, S_I=s, Z_II=z)


def sparse_pairs(cache, f, indices, valid):
    """Vectorized low-memory embeddings for selected pairs, without an L×L tensor."""
    import math

    module = cache["module"]
    b, d, n, k = indices.shape
    batch = torch.arange(b, device=indices.device)[:, None, None, None]
    diffusion = torch.arange(d, device=indices.device)[None, :, None, None]

    def keys(x):
        return x.expand(b, d, *x.shape[2:])[batch, diffusion, indices]

    motif = f["is_motif_atom_with_fixed_coord"].bool() & f["atom_valid"]
    same_motif = (
        motif[:, None, :, None] & keys(motif[:, None, :, None]).squeeze(-1) & valid
    )
    pos = f["motif_pos"][:, None]
    distance = torch.linalg.vector_norm(pos.unsqueeze(-2) - keys(pos), dim=-1)
    embedder = module.motif_pos_embedder
    frequency = torch.exp(
        -math.log(10000.0)
        * torch.arange(embedder.n_freqs, device=pos.device, dtype=torch.float32)
        / embedder.n_freqs
    )
    angles = distance[..., None] * frequency
    p = embedder.output_proj(torch.cat((angles.sin(), angles.cos()), -1))
    p = torch.where(
        same_motif[..., None],
        p + embedder.process_valid_mask(same_motif[..., None].to(p.dtype)),
        0,
    )
    seq = f["is_motif_atom_with_fixed_seq"].bool() & f["atom_valid"]
    uid = f["ref_space_uid"][:, None, :, None]
    ref_valid = seq[:, None, :, None] & keys(seq[:, None, :, None]).squeeze(-1)
    ref_valid = (
        ref_valid & (uid.squeeze(-1).unsqueeze(-1) == keys(uid).squeeze(-1)) & valid
    )
    pos = f["ref_pos"][:, None]
    dist2 = (
        torch.linalg.vector_norm(pos.unsqueeze(-2) - keys(pos), dim=-1, keepdim=True)
        .square()
        .clamp(min=1e-6)
    )
    embedder = module.ref_pos_embedder
    ref = embedder.process_inverse_dist(1 / (1 + dist2)) + embedder.process_valid_mask(
        ref_valid[..., None].to(p.dtype)
    )
    p = p + torch.where(ref_valid[..., None], ref, 0)
    p = p + cache["sl"].unsqueeze(-2) + keys(cache["sm"])
    tok = f["atom_to_token_map"]
    tok_key = keys(tok[:, None, :, None]).squeeze(-1)
    # Static token pairs are shared over D, but never shared over B.
    z = cache["z"][:, 0]
    p = p + z[batch, tok[:, None, :, None], tok_key]
    return torch.where(valid[..., None], p + module.pair_mlp(p), 0)
