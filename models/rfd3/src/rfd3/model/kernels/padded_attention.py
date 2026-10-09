# SPDX-License-Identifier: BSD-3-Clause
# Implements layers/batched.py::local_attention; see FOUNDRY_LICENSE.
"""Sparse slot attention for explicit B,D axes and padding masks.

One program handles one query/head. All neighbor slots are reduced together,
preserving the native sparse path's duplicate-slot semantics and BF16 rounding
at each tensor operation. Empty rows produce zero. Unlike the legacy index-set
kernel, pair biases are already gathered into neighbor slots.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _attention(
    Q,
    K,
    V,
    Bias,
    Idx,
    Valid,
    Gate,
    Out,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    KEYS: tl.constexpr,
    KB: tl.constexpr,
    E: tl.constexpr,
    SCORE_FP32: tl.constexpr,
    WEIGHTED_FP32: tl.constexpr,
):
    row, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    channel = head * (C // H) + tl.arange(0, E)
    slot = tl.arange(0, KB)
    # Flatten only B,D, after contiguous conversion in the wrapper.
    base = batch.to(tl.int64) * N
    offset = (base + row) * KEYS + slot
    index = tl.load(Idx + offset, slot < KEYS, other=0)
    valid = tl.load(Valid + offset, slot < KEYS, other=0)
    valid = valid & (slot < KEYS) & (index >= 0) & (index < N)
    cmask = tl.arange(0, E) < C // H
    q = tl.load(Q + (base + row) * C + channel, cmask, other=0)
    k = tl.load(
        K + (base + index[:, None]) * C + channel[None, :],
        valid[:, None] & cmask[None, :],
        other=0,
    )
    product = (q[None, :].to(tl.float32) * k.to(tl.float32)).to(Q.dtype.element_ty)
    dot = tl.sum(product.to(tl.float32), 1).to(Q.dtype.element_ty)
    scaled = (dot.to(tl.float32) * ((C // H) ** -0.5)).to(Q.dtype.element_ty)
    bias = tl.load(Bias + offset * H + head, valid, other=0)
    # PyTorch promotes q/k scores and bias before softmax.
    score_dtype: tl.constexpr = tl.float32 if SCORE_FP32 else tl.bfloat16
    weighted_dtype: tl.constexpr = tl.float32 if WEIGHTED_FP32 else tl.bfloat16
    score = (scaled.to(tl.float32) + bias.to(tl.float32)).to(score_dtype)
    score = tl.where(valid, score.to(tl.float32), -float("inf"))
    maximum = tl.max(score, 0)
    maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    p = tl.exp(score - maximum)
    denominator = tl.sum(p, 0)
    p = (p / tl.maximum(denominator, 1.0e-30)).to(score_dtype)
    v = tl.load(
        V + (base + index[:, None]) * C + channel[None, :],
        valid[:, None] & cmask[None, :],
        other=0,
    )
    weighted = (p[:, None].to(tl.float32) * v.to(tl.float32)).to(weighted_dtype)
    result = tl.sum(weighted.to(tl.float32), 0).to(weighted_dtype)
    gate = tl.load(Gate + (base + row) * C + channel, cmask, other=0)
    result = result.to(tl.float32) * gate.to(tl.float32)
    tl.store(Out + (base + row) * C + channel, result, cmask)


def attention(q, k, v, bias, indices, neighbor_valid, gate, heads):
    b, d, n, c = q.shape
    keys = indices.shape[-1]
    # Bias and masks can be broadcast or strided independently of the two batch
    # axes. These are O(BDNK), never copies of a dense pair grid.
    q, k, v, bias, indices, neighbor_valid, gate = (
        x.contiguous() for x in (q, k, v, bias, indices, neighbor_valid, gate)
    )
    dtype = torch.promote_types(torch.promote_types(q.dtype, bias.dtype), v.dtype)
    dtype = torch.promote_types(dtype, gate.dtype)
    out = torch.empty(q.shape, device=q.device, dtype=dtype)
    if n and b and d:
        _attention[(n, heads, b * d)](
            q,
            k,
            v,
            bias,
            indices,
            neighbor_valid,
            gate,
            out,
            n,
            c,
            heads,
            keys,
            triton.next_power_of_2(keys),
            triton.next_power_of_2(c // heads),
            q.dtype == torch.float32 or bias.dtype == torch.float32,
            q.dtype == torch.float32
            or bias.dtype == torch.float32
            or v.dtype == torch.float32,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out
