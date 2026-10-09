"""CPU collation/topology preparation, separate from the compiled tensor region."""

import torch
from rfd3.inference.feature_schema import FEATURE_AXES

DEFAULT_COMPILE_ATOM_BUCKETS = (
    256,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
    6144,
    8192,
    12288,
    16384,
    24576,
    32768,
)
DEFAULT_COMPILE_TOKEN_BUCKETS = (128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096)
DEFAULT_COMPILE_SLOT_BUCKETS = (1, 4, 8, 14, 24, 32, 64, 128)


def validate_buckets(values, name):
    """Normalize config/OmegaConf sequences once, outside model execution."""
    values = tuple(values)
    if (
        not values
        or any(type(v) is not int or v < 1 for v in values)
        or any(a >= b for a, b in zip(values, values[1:]))
    ):
        raise ValueError(f"{name} must contain strictly increasing positive integers")
    return values


def padding_capacity(length, multiple=1, buckets=None, *, axis="length"):
    """Choose a ceiling, never round down or truncate to the largest bucket."""
    if buckets is not None:
        for capacity in buckets:
            if length <= capacity:
                return capacity
        raise ValueError(
            f"{axis} {length} exceeds largest compile bucket {buckets[-1]}; increase the configured buckets"
        )
    return ((length + multiple - 1) // multiple) * multiple


def collate_examples(
    examples,
    *,
    atom_capacity=None,
    token_capacity=None,
    batch_capacity=None,
    slot_capacity=None,
):
    if not examples:
        raise ValueError("Cannot collate an empty batch")
    fs = [e["feats"] for e in examples]
    lengths = [(len(f["atom_to_token_map"]), len(f["restype"])) for f in fs]
    if any(l == 0 or i == 0 for l, i in lengths):
        raise ValueError("Empty input structures are not supported")
    b = batch_capacity if batch_capacity is not None else len(examples)
    l = atom_capacity if atom_capacity is not None else max(x[0] for x in lengths)
    i = token_capacity if token_capacity is not None else max(x[1] for x in lengths)
    if b < len(examples) or any(n > l or m > i for n, m in lengths):
        raise ValueError("Padding capacity cannot truncate an example")
    first = examples[0]["coord_atom_lvl_to_be_noised"]
    if first.ndim != 3 or first.shape[0] < 1 or first.shape[-1] != 3:
        raise ValueError("Coordinates must have shape (D,L,3) with D > 0")
    d, device = first.shape[0], first.device
    coords = first.new_zeros(b, d, l, 3)
    keys = set().union(*(f.keys() for f in fs))
    unknown = keys - FEATURE_AXES.keys() - {"sym_transform"}
    if unknown:
        raise ValueError(f"Unregistered RFD3 features: {sorted(unknown)}")
    # Optional features must be made explicit by preprocessing. Missing fields are
    # errors rather than guessing whether zero is a scientifically valid default.
    if any(set(f) != keys for f in fs):
        raise ValueError("Feature sets differ; partition examples before collation")
    f = {}
    for key in sorted(keys - {"sym_transform"}):
        axes, value = FEATURE_AXES[key], fs[0][key]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Feature {key} must be a tensor")
        n = len(axes)
        shape = tuple(l if axis == "L" else i for axis in axes)
        f[key] = value.new_zeros((b, *shape, *value.shape[n:]))
        for row, (src, (nl, ni)) in enumerate(zip(fs, lengths)):
            v = src[key]
            expected = tuple(nl if axis == "L" else ni for axis in axes)
            if (
                v.shape[:n] != expected
                or v.shape[n:] != value.shape[n:]
                or v.dtype != value.dtype
                or v.device != device
            ):
                raise ValueError(f"Incompatible shape/dtype/device for {key}")
            if v.is_floating_point() and not torch.isfinite(v).all():
                raise ValueError(f"Nonfinite input feature: {key}")
            f[key][(row, *(slice(0, size) for size in expected))] = v
    atom_valid = torch.zeros(b, l, dtype=torch.bool, device=device)
    token_valid = torch.zeros(b, i, dtype=torch.bool, device=device)
    reps = torch.zeros(b, i, dtype=torch.long, device=device)
    counts = []
    for row, (ex, (nl, ni)) in enumerate(zip(examples, lengths)):
        x, src = ex["coord_atom_lvl_to_be_noised"], ex["feats"]
        if x.shape != (d, nl, 3) or x.dtype != first.dtype or x.device != device:
            raise ValueError("Examples require a common D, coordinate dtype and device")
        if not torch.isfinite(x).all():
            raise ValueError("Nonfinite input coordinates")
        tok = src["atom_to_token_map"].long()
        if (
            (tok < 0).any()
            or (tok >= ni).any()
            or not torch.equal(tok, src["atom_to_token_map"])
        ):
            raise ValueError("Invalid atom-to-token map")
        count = torch.bincount(tok, minlength=ni)
        if (count == 0).any():
            raise ValueError("Each real token must contain atoms")
        rep = src["is_ca"].bool().nonzero().flatten()
        if len(rep) != ni or not torch.equal(tok[rep], torch.arange(ni, device=device)):
            raise ValueError("Expected one ordered representative per token")
        counts.append(count)
        coords[row, :, :nl] = x
        atom_valid[row, :nl], token_valid[row, :ni] = True, True
        reps[row, :ni] = rep
    a = (
        slot_capacity
        if slot_capacity is not None
        else max(int(c.max()) for c in counts)
    )
    if any(int(c.max()) > a for c in counts):
        raise ValueError("Atom slot capacity cannot truncate a token")
    slots = torch.zeros(b, i, a, dtype=torch.long, device=device)
    slot_valid = torch.zeros_like(slots, dtype=torch.bool)
    inverse = torch.zeros(b, l, dtype=torch.long, device=device)
    for row, (src, (_, ni)) in enumerate(zip(fs, lengths)):
        for token in range(ni):
            indices = (src["atom_to_token_map"] == token).nonzero().flatten()
            slots[row, token, : len(indices)] = indices
            slot_valid[row, token, : len(indices)] = True
            inverse[row, indices] = token * a + torch.arange(
                len(indices), device=device
            )
    f.update(
        atom_valid=atom_valid,
        token_valid=token_valid,
        representative_atom=reps,
        atom_slots=slots,
        slot_valid=slot_valid,
        atom_slot_inverse=inverse,
    )
    f["atom_to_token_map"] = f["atom_to_token_map"].long()
    if "sym_transform" in keys:
        f.update(symmetry_maps(fs, b, l, device))
    return {"f": f, "coords": coords}


def symmetry_maps(features, b, l, device):
    """Convert variable-sized symmetry dictionaries into per-atom tensor maps."""
    source = torch.arange(l, device=device).expand(b, l).clone()
    rotation = torch.eye(3, device=device).expand(b, l, 3, 3).clone()
    translation = torch.zeros(b, l, 3, device=device)
    apply = torch.zeros(b, l, dtype=torch.bool, device=device)
    for row, f in enumerate(features):
        entities, tids, asu = (
            f["sym_entity_id"],
            f["sym_transform_id"],
            f["is_sym_asu"].bool(),
        )
        for entity in entities.unique().tolist():
            if entity == -1:
                continue
            mask = entities == entity
            original = (mask & asu).nonzero().flatten()
            if not len(original):
                continue
            for tid in tids[mask].unique().tolist():
                target = (mask & (tids == tid)).nonzero().flatten()
                if len(target) != len(original) or str(tid) not in f["sym_transform"]:
                    raise ValueError(
                        "Symmetry subunit/ASU sizes or transform IDs are inconsistent"
                    )
                r, t = f["sym_transform"][str(tid)]
                source[row, target] = original
                rotation[row, target] = r
                translation[row, target] = t
                apply[row, target] = True
    return dict(
        symmetry_source=source,
        symmetry_rotation=rotation,
        symmetry_translation=translation,
        symmetry_apply=apply,
    )


def prepare_attention(f, *, atom_keys, atom_neighbors, token_keys, token_neighbors):
    """Freeze legacy sequence-neighbor policy using true, unpadded lengths.

    Geometric neighbors remain diffusion-dependent and are selected in tensor code.
    Multi-chain (>3) legacy selection uses a separate randomized algorithm and must
    use the explicitly reported legacy fallback until that policy is migrated.
    """
    from rfd3.model.layers.block_utils import build_index_mask

    f = dict(f)
    for level, keys, neighbors in [
        ("atom", atom_keys, atom_neighbors),
        ("token", token_keys, token_neighbors),
    ]:
        valid = f[f"{level}_valid"]
        b, n = valid.shape
        seq = torch.zeros(b, n, n, device=valid.device, dtype=torch.bool)
        for row in range(b):
            size = int(valid[row].sum())
            if not size:
                continue
            tok = (
                f["atom_to_token_map"][row, :size]
                if level == "atom"
                else torch.arange(size, device=valid.device)
            )
            chains = f["asym_id"][row, tok]
            if len(chains.unique()) > 3:
                raise ValueError(
                    "More than three chains require legacy neighbor selection"
                )
            base = ~f["unindexing_pair_mask"][row][tok[None, :], tok[:, None]]
            seq[row, :size, :size] = build_index_mask(
                tok, neighbors, min(keys, size), chain_id=chains, base_mask=base
            )
        f[f"{level}_sequence_mask"] = seq
        # This part of neighbor selection depends only on topology. Sorting an
        # N-by-N integer grid in every denoiser/recycle is needlessly expensive.
        forced = seq & valid[:, :, None] & valid[:, None, :]
        grid = torch.arange(n, device=valid.device)
        f[f"{level}_forced_neighbors"] = torch.where(forced, grid, n).sort(-1).values[
            ..., : min(keys, n)
        ].contiguous()
    return f


def unpad_output(output, row, example):
    """Restore the legacy per-example tensor output contract before structure IO."""
    l = len(example["feats"]["atom_to_token_map"])
    i = len(example["feats"]["restype"])
    out = {}
    for key, value in output.items():
        if value is None:
            out[key] = None
        elif key in ("X_L",):
            out[key] = value[row, :, :l]
        elif key in ("sequence_logits_I", "sequence_indices_I"):
            out[key] = value[row, :, :i]
        elif key in ("X_noisy_L_traj", "X_denoised_L_traj"):
            out[key] = [x[row, :, :l] for x in value]
        elif key == "sequence_entropy_traj":
            out[key] = [x[row, :, :i] for x in value]
        elif key == "t_hats":
            out[key] = value
        else:
            raise ValueError(f"Undeclared batched output: {key}")
    return out


def cfg_reference_examples(examples, zero_features):
    """Legacy unindexed-tail removal, using declared axes rather than size guesses."""
    from rfd3.inference.feature_schema import crop_features

    result = []
    for ex in examples:
        f = ex["feats"]
        am, tm = (
            f["is_motif_atom_unindexed"].bool(),
            f["is_motif_token_unindexed"].bool(),
        )
        na = int(am.nonzero()[0]) if am.any() else len(am)
        nt = int(tm.nonzero()[0]) if tm.any() else len(tm)
        if not na or not nt or not am[na:].all() or not tm[nt:].all():
            raise ValueError(
                "CFG requires a nonempty indexed prefix and an unindexed tail"
            )
        result.append(
            {
                **ex,
                "feats": crop_features(f, na, nt, zero_features),
                "coord_atom_lvl_to_be_noised": ex["coord_atom_lvl_to_be_noised"][
                    :, :na
                ],
            }
        )
    return result


def batch_signature(
    example,
    atom_multiple=1,
    token_multiple=1,
    *,
    atom_buckets=None,
    token_buckets=None,
    slot_buckets=None,
    batch_max=False,
):
    """Compatibility/bucket key after transforms; no scientific features guessed."""
    f = example["feats"]
    l, i = len(f["atom_to_token_map"]), len(f["restype"])
    a = int(torch.bincount(f["atom_to_token_map"].long(), minlength=i).max())
    partial = float(f["partial_t"].mean()) if "partial_t" in f else None
    schema = tuple(
        (k, tuple(v.shape[len(FEATURE_AXES[k]) :]), str(v.dtype))
        for k, v in sorted(f.items())
        if k in FEATURE_AXES
    )
    x = example["coord_atom_lvl_to_be_noised"]
    # A multiple of 1 means no length bucketing: each batch pads to its maximum.
    lb = (
        padding_capacity(l, atom_multiple, atom_buckets, axis="atom length")
        if atom_multiple > 1 or atom_buckets is not None
        else None
    )
    ib = (
        padding_capacity(i, token_multiple, token_buckets, axis="token length")
        if token_multiple > 1 or token_buckets is not None
        else None
    )
    a = padding_capacity(a, buckets=slot_buckets, axis="atoms per token")
    if batch_max:
        lb, ib, a = None, None, None
    return (lb, ib, a, x.shape[0], str(x.dtype), partial, schema)


def iter_compatible_batches(
    examples, batch_size, atom_multiple=1, token_multiple=1, **buckets
):
    """Bounded contiguous grouping: at most batch_size transformed examples live."""
    if min(batch_size, atom_multiple, token_multiple) < 1:
        raise ValueError("Batch size and padding multiples must be positive")
    pending, signature = [], None
    for example in examples:
        current = batch_signature(example, atom_multiple, token_multiple, **buckets)
        if pending and current != signature:
            yield pending
            pending = []
        pending.append(example)
        signature = current
        if len(pending) == batch_size:
            yield pending
            pending = []
    if pending:
        yield pending
