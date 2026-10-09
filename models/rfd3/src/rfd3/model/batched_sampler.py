"""Padded EDM rollout; Python scheduling/RNG stays outside the compiled denoiser."""

import hashlib

import torch
from rfd3.model.layers.batched import gather_nodes, mask_nodes


def stable_seed(seed, *parts):
    payload = repr((seed, *parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


class ExampleRandomness:
    """CPU streams keyed by example, sample and event, unaffected by padding.

    True lengths determine how many numbers are generated. This intentionally
    defines a new deterministic seed contract for the opt-in batched path.
    """

    def __init__(self, examples, seed):
        self.ids = [e["example_id"] for e in examples]
        self.lengths = [len(e["feats"]["atom_to_token_map"]) for e in examples]
        self.seed = seed

    def __call__(self, tag, template, atomwise=True):
        result = torch.zeros(
            template.shape, dtype=template.dtype, device="cpu", pin_memory=template.is_cuda
        )
        for b, (name, length) in enumerate(zip(self.ids, self.lengths)):
            for d in range(template.shape[1]):
                generator = torch.Generator().manual_seed(
                    stable_seed(self.seed, name, d, tag)
                )
                shape = (
                    (length, *template.shape[3:]) if atomwise else template.shape[2:]
                )
                values = torch.randn(shape, generator=generator, dtype=template.dtype)
                if atomwise:
                    result[b, d, :length] = values
                else:
                    result[b, d] = values
        # Each call owns its pinned buffer; PyTorch retains the allocation until
        # the asynchronous copy completes. Preserve the CPU RNG/seed contract.
        return result.to(template.device, non_blocking=template.is_cuda)


def masked_mean(x, mask):
    weights = mask[:, None, :, None]
    return torch.where(weights, x, 0).sum(-2, keepdim=True) / weights.sum(
        -2, keepdim=True
    ).clamp(min=1)


def masked_align(target, moving, mask):
    """Independent weighted Kabsch, with translation-only degenerate fallback.

    Empty rows are unchanged. Fewer than three or collinear points determine no
    unique rotation, so these use centroid translation instead of an arbitrary SVD
    rotation. Padding is excluded before reductions.
    """
    center_t, center_m = masked_mean(target, mask), masked_mean(moving, mask)
    weights = mask[:, None, :, None]
    t = torch.where(weights, target - center_t, 0)
    m = torch.where(weights, moving - center_m, 0)
    covariance = m.transpose(-1, -2) @ t
    identity = torch.eye(3, device=target.device, dtype=target.dtype).expand_as(
        covariance
    )
    enough = (mask.sum(-1) >= 3)[:, None, None, None]
    covariance = torch.where(enough, covariance, identity)
    # SVD is evaluated in fp32 under mixed-precision inference.
    with torch.amp.autocast(device_type=target.device.type, enabled=False):
        u, singular, vh = torch.linalg.svd(covariance.float())
        correction = torch.ones_like(singular)
        correction[..., -1] = torch.where(torch.linalg.det(u @ vh) < 0, -1.0, 1.0)
        rotation = (u * correction.unsqueeze(-2)) @ vh
    nondegenerate = (singular[..., 1] > singular[..., 0] * 1e-6)[
        ..., None, None
    ] & enough
    rotation = torch.where(nondegenerate, rotation.to(target.dtype), identity)
    aligned = (moving - center_m) @ rotation + center_t
    return torch.where(mask.any(-1)[:, None, None, None], aligned, moving)


def rotation_from_noise(z):
    """Match the legacy independent uniform Euler-angle distribution, vectorized."""
    angles = 0.5 * (1 + torch.erf(z / 2**0.5)) * (2 * torch.pi)
    x, y, z = angles.unbind(-1)
    cx, cy, cz, sx, sy, sz = x.cos(), y.cos(), z.cos(), x.sin(), y.sin(), z.sin()
    entries = (
        cz * cy,
        cz * sy * sx - sz * cx,
        cz * sy * cx + sz * sx,
        sz * cy,
        sz * sy * sx + cz * cx,
        sz * sy * cx - cz * sx,
        -sy,
        cy * sx,
        cy * cx,
    )
    return torch.stack(entries, -1).unflatten(-1, (3, 3))


def augment(
    x,
    original,
    f,
    rotation,
    translation,
    *,
    center_option="all",
    centering_affects_motif=True,
    reinsert_motif=True,
):
    valid = f["atom_valid"]
    fixed = f["is_motif_atom_with_fixed_coord"].bool() & valid
    x = mask_nodes(x, valid)
    if reinsert_motif:
        aligned = masked_align(x, original, fixed)
        x = torch.where(fixed[:, None, :, None], aligned, x)
    if center_option == "motif":
        chosen = fixed
    elif center_option == "diffuse":
        chosen = valid & ~fixed
    else:
        chosen = valid
    # No motif or no diffuse atoms: use the whole valid structure, never 0/0.
    chosen = torch.where(chosen.any(-1, keepdim=True), chosen, valid)
    center = masked_mean(x, chosen)
    movable = valid if centering_affects_motif else valid & ~fixed
    x = torch.where(movable[:, None, :, None], x - center, x)
    return mask_nodes(x @ rotation.transpose(-1, -2) + translation, valid)


def apply_symmetry(x, f, allow_realignment):
    valid = f["atom_valid"]
    movable = valid & (f["sym_entity_id"] != -1)
    fixed = (
        f["is_motif_atom_with_fixed_coord"].bool() & f["is_sym_asu"].bool() & movable
    )
    held_motif = fixed.any(-1) & (not allow_realignment)
    recenter = (~held_motif)[:, None, None, None] if "partial_t" not in f else False
    center = masked_mean(x, movable)
    centered = torch.where(movable[:, None, :, None] & recenter, x - center, x)
    original = gather_nodes(centered, f["symmetry_source"])
    transformed = torch.einsum(
        "bdlc,blce->bdle", original, f["symmetry_rotation"].to(x.dtype)
    )
    transformed = transformed + f["symmetry_translation"][:, None].to(x.dtype)
    return mask_nodes(
        torch.where(f["symmetry_apply"][:, None, :, None], transformed, centered), valid
    )


def sample(
    sampler,
    diffusion_module,
    coords,
    f,
    initializer,
    *,
    f_ref=None,
    ref_initializer=None,
    noise_source=None,
    capture_trajectories=True,
):
    valid = f["atom_valid"]
    fixed = valid & f["is_motif_atom_with_fixed_coord"].bool()
    diffuse = valid & ~fixed
    b, d = coords.shape[:2]
    symmetric = hasattr(sampler, "sym_step_frac")
    if symmetric and sampler.use_classifier_free_guidance:
        raise ValueError("Symmetry and classifier-free guidance cannot be combined")
    if noise_source is None:

        def noise_source(tag, template, atomwise=True):
            return torch.randn_like(template)

    partial = None
    if "partial_t" in f:
        values = (f["partial_t"] * valid).sum(-1) / valid.sum(-1).clamp(min=1)
        real = values[valid.any(-1)]
        if not torch.all(real == real[0]):
            raise ValueError("Partial diffusion schedules differ; partition this batch")
        partial = real[:1]
    schedule = sampler._construct_inference_noise_schedule(coords.device, partial)
    if len(schedule) < 2:
        raise ValueError("Diffusion schedule requires at least two timesteps")
    # Keep schedule arithmetic on its original device, but resolve Python
    # control flow together before sampling instead of reading CUDA scalars at
    # every step. Comparing on device preserves threshold rounding exactly.
    current_times = schedule[1:]
    gamma_active = current_times > sampler.gamma_min
    cfg_active = torch.ones_like(gamma_active)
    if sampler.cfg_t_max is not None:
        cfg_active = current_times > sampler.cfg_t_max
    symmetry_active = torch.zeros_like(gamma_active)
    if symmetric:
        symmetry_until = schedule[
            min(int(len(schedule) * sampler.sym_step_frac), len(schedule) - 1)
        ]
        symmetry_active = current_times > symmetry_until
    decisions = torch.stack((gamma_active, cfg_active, symmetry_active), -1).cpu().tolist()
    x = mask_nodes(
        coords
        + torch.where(
            diffuse[:, None, :, None], schedule[0] * noise_source("initial", coords), 0
        ),
        valid,
    )
    if sampler.s_jitter_origin > 0 and not symmetric:
        jitter = (
            noise_source("jitter", coords[:, :, :1], atomwise=False)
            * sampler.s_jitter_origin
        )
        x = mask_nodes(x + torch.where(fixed[:, None, :, None], jitter, 0), valid)
    noisy_traj, denoised_traj, entropy_traj, times = [], [], [], []
    threshold = (len(schedule) - 1) * sampler.fraction_of_steps_to_fix_motif

    def augmentation(step, x, **kwargs):
        angles = noise_source(f"rotation_{step}", coords[:, :, 0], atomwise=False)
        translation = noise_source(
            f"translation_{step}", coords[:, :, :1], atomwise=False
        )
        scale = kwargs.pop("s_trans", sampler.s_trans)
        return augment(
            x, coords, f, rotation_from_noise(angles), translation * scale, **kwargs
        )

    for step, (previous, current) in enumerate(zip(schedule, schedule[1:])):
        use_gamma, use_cfg, use_symmetry = decisions[step]
        if sampler.allow_realignment:
            if symmetric:
                x = augmentation(step, x, s_trans=1.0)
            else:
                x = augmentation(
                    step,
                    x,
                    center_option=sampler.center_option,
                    centering_affects_motif=max(step - 1, 0) >= threshold,
                    s_trans=sampler.s_trans if step >= threshold else 0.0,
                )
        gamma = sampler.gamma_0 if use_gamma else 0.0
        t_hat = previous * (1 + gamma)
        noise = (
            sampler.noise_scale
            * (t_hat.square() - previous.square()).sqrt()
            * noise_source(f"step_{step}", coords)
        )
        noisy = mask_nodes(x + torch.where(diffuse[:, None, :, None], noise, 0), valid)
        outs = diffusion_module.forward_batched(
            noisy, t_hat.expand(b, d), f, **initializer, n_recycle=sampler.n_recycle
        )
        denoised = mask_nodes(outs["X_L"], valid)
        if symmetric and use_symmetry:
            denoised = apply_symmetry(denoised, f, sampler.allow_realignment)
        delta = (noisy - denoised) / t_hat
        if (
            sampler.use_classifier_free_guidance
            and sampler.cfg_scale != 1
            and use_cfg
        ):
            if f_ref is None or ref_initializer is None:
                raise ValueError(
                    "CFG requires separately initialized reference features"
                )
            ref_noisy = mask_nodes(noisy, f_ref["atom_valid"])
            ref = diffusion_module.forward_batched(
                ref_noisy,
                t_hat.expand(b, d),
                f_ref,
                **ref_initializer,
                n_recycle=sampler.n_recycle,
            )
            delta_ref = mask_nodes(
                (ref_noisy - ref["X_L"]) / t_hat, f_ref["atom_valid"]
            )
            delta = delta + (sampler.cfg_scale - 1) * (delta - delta_ref)
        x = mask_nodes(noisy + sampler.step_scale * (current - t_hat) * delta, valid)
        if capture_trajectories:
            noisy_traj.append(
                mask_nodes(
                    sampler.sigma_data
                    * noisy
                    / (t_hat.square() + sampler.sigma_data**2).sqrt(),
                    valid,
                )
            )
            denoised_traj.append(denoised)
            times.append(t_hat)
            if outs.get("sequence_logits_I") is not None:
                p = outs["sequence_logits_I"].softmax(-1)
                entropy_traj.append(
                    torch.where(
                        f["token_valid"][:, None], -(p * (p + 1e-10).log()).sum(-1), 0
                    )
                )
    if sampler.allow_realignment:
        final = augmentation(
            "final", x, s_trans=1.0, reinsert_motif=sampler.insert_motif_at_end
        )
        if symmetric:
            final = apply_symmetry(final, f, sampler.allow_realignment)
        final = masked_align(coords, final, fixed)
        x = mask_nodes(torch.where(fixed.any(-1)[:, None, None, None], final, x), valid)
    # Preserve the public CPU entropy-trajectory contract with one transfer,
    # after all denoising steps have been enqueued.
    if entropy_traj:
        entropy_traj = list(torch.stack(entropy_traj).cpu().unbind())
    return dict(
        X_L=x,
        X_noisy_L_traj=noisy_traj,
        X_denoised_L_traj=denoised_traj,
        t_hats=times,
        sequence_entropy_traj=entropy_traj,
        sequence_logits_I=outs.get("sequence_logits_I"),
        sequence_indices_I=outs.get("sequence_indices_I"),
    )
