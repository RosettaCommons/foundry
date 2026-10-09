"""Native dense attention shared by inference models; no checkpoint parameters."""

import torch
import torch.nn.functional as F


def validate_attention_backend(backend: str) -> str:
    if backend not in {"auto", "vanilla", "sdpa"}:
        raise ValueError("dense_attention_backend must be auto, vanilla or sdpa")
    return backend


def use_dense_sdpa(module: torch.nn.Module, x: torch.Tensor) -> bool:
    backend = module.dense_attention_backend
    return backend == "sdpa" or (
        backend == "auto" and x.device.type == "mps" and not module.training
    )


def dense_pair_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    *,
    scale: float,
    allowed: torch.Tensor | None = None,
) -> torch.Tensor:
    """SDPA on (...,H,Q,C), with additive (...,H,Q,K) pair bias.

    Collapse leading axes explicitly: Metal SDPA cannot flatten some broadcast
    5D mask strides. Only the projected head bias is expanded, never pair features.
    Empty padded queries return finite zeros, including their backward pass.
    Callers sanitize padded Q/K/V before projecting and mask final projections.
    """
    leading = torch.broadcast_shapes(
        q.shape[:-3],
        k.shape[:-3],
        v.shape[:-3],
        bias.shape[:-3],
        () if allowed is None else allowed.shape[:-3],
    )
    shape = (*leading, *q.shape[-3:])
    mask_shape = (*shape[:-1], k.shape[-2])
    bias = bias.to(q.dtype).expand(mask_shape)
    if allowed is not None:
        has_keys = allowed.any(-1, keepdim=True)
        bias = torch.where(has_keys, bias.masked_fill(~allowed, float("-inf")), 0)
    out = F.scaled_dot_product_attention(
        q.expand(shape).reshape(-1, *shape[-3:]),
        k.expand(*shape[:-3], *k.shape[-3:]).reshape(-1, *k.shape[-3:]),
        v.expand(*shape[:-3], *v.shape[-3:]).reshape(-1, *v.shape[-3:]),
        attn_mask=bias.reshape(-1, *mask_shape[-3:]),
        dropout_p=0.0,
        scale=scale,
    ).reshape(*shape[:-1], v.shape[-1])
    return torch.where(has_keys, out, 0) if allowed is not None else out
