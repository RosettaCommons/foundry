"""Differentiable RF3 inference used only by RFO sequence optimization.

The public RF3 confidence head detaches its trunk inputs at the top of forward
(standard AF3 architecture: confidence is a post-hoc scorer over frozen trunk
outputs). RFO needs gradients to flow *through* the confidence head back to
the sequence, so ``_grad_through_confidence_head`` wraps the head and turns
``torch.Tensor.detach`` into a no-op only for the duration of its call.
Diffusion coordinates are still held constant: they are detached explicitly
in ``optimization_forward`` before entering the head, so the extra internal
detach is redundant and safely no-ops.
"""

from collections import deque
from contextlib import contextmanager

import torch
from torch.utils.checkpoint import checkpoint


@contextmanager
def _no_op_detach():
    """Temporarily make ``torch.Tensor.detach`` a no-op (identity).

    Scoped to the wrapped block via a context manager; the original is
    restored on exit even if the wrapped code raises.
    """
    orig = torch.Tensor.detach
    torch.Tensor.detach = lambda self: self
    try:
        yield
    finally:
        torch.Tensor.detach = orig


class _GradThroughConfidenceHead(torch.nn.Module):
    """Wrap ``ConfidenceHead`` to let gradients flow to its trunk inputs.

    RF3's ``ConfidenceHead.forward`` runs ``S_trunk_I.detach()``,
    ``Z_trunk_II.detach()``, ``S_inputs_I.detach()`` and ``seq.detach()``
    at the top of forward. That severs autograd from the confidence-based
    losses (``pae_interface_mean`` etc.) back to ``restype`` — RFO's whole
    seq-optimization premise. This wrapper turns ``detach`` into a no-op
    only inside the head's ``forward`` call. ``X_pred_L`` is already
    detached by RFO before entering the head (see below), so its internal
    ``.detach()`` no-ops both ways.
    """

    def __init__(self, head: torch.nn.Module):
        super().__init__()
        self.head = head

    def forward(self, *args, **kwargs):
        with _no_op_detach():
            return self.head(*args, **kwargs)


def optimization_forward(model, inputs, n_cycle, coordinates, skip_diffusion=False):
    if model.training:
        raise ValueError("RFO optimization requires an RF3 model in eval mode.")
    if n_cycle < 1:
        raise ValueError("n_cycle must be at least 1.")
    if not hasattr(model, "confidence_head"):
        raise ValueError("RFO requires an RF3 checkpoint with a confidence head.")

    # Match RF3's feature casting without assuming a CUDA device.
    device_type = inputs["f"]["restype"].device.type
    if torch.is_autocast_enabled(device_type):
        dtype = torch.get_autocast_dtype(device_type)
        for key in ("msa_stack", "profile", "deletion_mean", "restype", "ref_pos"):
            if key in inputs["f"]:
                inputs["f"][key] = inputs["f"][key].to(dtype)

    recycled = deque(
        model.trunk_forward_with_recycling(f=inputs["f"], n_recycles=n_cycle),
        maxlen=1,
    ).pop()
    output = {
        "early_stopped": False,
        "X_L": None,
        "distogram": model.distogram_head(recycled["Z_II"]),
        "S_I": recycled["S_I"],
        "Z_II": recycled["Z_II"],
    }
    if skip_diffusion:
        return output

    with torch.no_grad():
        sampled = model.inference_sampler.sample_diffusion_like_af3(
            f=inputs["f"],
            S_inputs_I=recycled["S_inputs_I"],
            S_trunk_I=recycled["S_I"],
            Z_trunk_II=recycled["Z_II"],
            diffusion_module=model.diffusion_module,
            diffusion_batch_size=inputs["t"].shape[0],
            coord_atom_lvl_to_be_noised=coordinates,
        )
    coords = sampled["X_L"].detach()
    confidence = {}
    conf_head = _GradThroughConfidenceHead(model.confidence_head)
    for sample in coords:
        values = checkpoint(
            conf_head,
            recycled["S_inputs_I"],
            recycled["S_I"],
            recycled["Z_II"],
            sample.unsqueeze(0),
            inputs["seq"],
            inputs["rep_atom_idxs"],
            frame_atom_idxs=inputs["frame_atom_idxs"],
            use_reentrant=False,
        )
        for key, value in values.items():
            confidence.setdefault(key, []).append(value)
    output.update(
        {key: sampled[key] for key in ("X_noisy_L_traj", "X_denoised_L_traj", "t_hats")}
    )
    output["X_pred_rollout_L"] = coords
    for key in ("plddt", "pae", "pde", "exp_resolved"):
        output[key] = torch.cat(confidence[f"{key}_logits"], dim=0)
    return output
