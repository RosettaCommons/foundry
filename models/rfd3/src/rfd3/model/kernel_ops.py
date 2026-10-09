"""Compiler-visible inference operators; CUDA implementations load lazily."""

import torch


@torch.library.custom_op("rfd3::transition", mutates_args=())
def transition(
    x: torch.Tensor,
    weight_a: torch.Tensor,
    weight_b: torch.Tensor,
    weight_out: torch.Tensor,
) -> torch.Tensor:
    from rfd3.model.kernels.transition import transition as implementation

    return implementation(x, weight_a, weight_b, weight_out)


@transition.register_fake
def _transition_fake(x, weight_a, weight_b, weight_out):
    return torch.empty_like(
        x, dtype=torch.bfloat16, memory_format=torch.contiguous_format
    )


@torch.library.custom_op("rfd3::padded_attention", mutates_args=())
def padded_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    indices: torch.Tensor,
    neighbor_valid: torch.Tensor,
    gate: torch.Tensor,
    heads: int,
) -> torch.Tensor:
    from rfd3.model.kernels.padded_attention import attention

    return attention(q, k, v, bias, indices, neighbor_valid, gate, heads)


@padded_attention.register_fake
def _padded_attention_fake(q, k, v, bias, indices, neighbor_valid, gate, heads):
    dtype = torch.promote_types(q.dtype, bias.dtype)
    dtype = torch.promote_types(dtype, v.dtype)
    dtype = torch.promote_types(dtype, gate.dtype)
    return torch.empty_like(v, dtype=dtype, memory_format=torch.contiguous_format)
