# SPDX-License-Identifier: BSD-3-Clause
# Implements this repository's RFD3 Transition; see FOUNDRY_LICENSE.
"""Two-stage implementation of RFD3's post-normalization transition.

Specification: model/layers/layer_utils.py::Transition.forward at repository
commit 829b3a1806ddee6e2dccebf911a6a9bf841075d5. RMSNorm remains outside this
kernel. Stage one computes two separate input projections and their SwiGLU
activation, tiled over rows and hidden channels. Stage two uses PyTorch's
matrix multiplication for the output projection.

This implementation is written from those RFD3 operations and Triton's public
dot/load/store API. It replaces the previously imported fpf_transition kernel;
it does not retain its device code, interleaved weight packing, launch tables,
normalization/residual variants, or helper routines. The development history
includes prior review of that kernel; this is not a clean-room certification.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _project_swiglu(
    inputs,
    weight_a,
    weight_b,
    hidden_output,
    rows: tl.constexpr,
    channels: tl.constexpr,
    hidden: tl.constexpr,
    tile_rows: tl.constexpr,
    tile_hidden: tl.constexpr,
    tile_reduce: tl.constexpr,
):
    row = tl.program_id(0) * tile_rows + tl.arange(0, tile_rows)
    feature = tl.program_id(1) * tile_hidden + tl.arange(0, tile_hidden)
    reduce = tl.arange(0, tile_reduce)
    sum_a = tl.full((tile_rows, tile_hidden), 0, tl.float32)
    sum_b = tl.full((tile_rows, tile_hidden), 0, tl.float32)

    for offset in range(tl.cdiv(channels, tile_reduce)):
        channel = offset * tile_reduce + reduce
        x = tl.load(
            inputs + row[:, None] * channels + channel[None, :],
            (row[:, None] < rows) & (channel[None, :] < channels),
            other=0,
        )
        weight_offset = feature[None, :] * channels + channel[:, None]
        weight_mask = (feature[None, :] < hidden) & (channel[:, None] < channels)
        a = tl.load(weight_a + weight_offset, weight_mask, other=0)
        b = tl.load(weight_b + weight_offset, weight_mask, other=0)
        sum_a = tl.dot(x, a, sum_a)
        sum_b = tl.dot(x, b, sum_b)

    # Match the bf16 tensors produced by each native Linear, SiLU, and multiply.
    a_bf16 = sum_a.to(tl.bfloat16).to(tl.float32)
    b_bf16 = sum_b.to(tl.bfloat16).to(tl.float32)
    activated = (a_bf16 / (1.0 + libdevice.exp(-a_bf16))).to(tl.bfloat16)
    product = activated.to(tl.float32) * b_bf16
    tl.store(
        hidden_output + row[:, None] * hidden + feature[None, :],
        product,
        (row[:, None] < rows) & (feature[None, :] < hidden),
    )


def transition(x, weight_a, weight_b, weight_out):
    """BF16 post-RMSNorm transition with separate, contiguous projection weights."""
    shape = x.shape
    channels = shape[-1]
    flat = x.to(torch.bfloat16).reshape(-1, channels).contiguous()
    rows, hidden = flat.shape[0], weight_a.shape[0]
    activation = torch.empty((rows, hidden), device=x.device, dtype=torch.bfloat16)
    # Measured locally on the A4000: wider hidden tiles amortize input loads on
    # large pair grids; smaller tiles keep more parallel work for short inputs.
    tile_hidden = 128 if rows >= 8192 else 64
    if rows:
        _project_swiglu[(triton.cdiv(rows, 64), triton.cdiv(hidden, tile_hidden))](
            flat,
            weight_a,
            weight_b,
            activation,
            rows,
            channels,
            hidden,
            tile_rows=64,
            tile_hidden=tile_hidden,
            tile_reduce=32,
            num_warps=4,
            num_stages=2,
        )
    return F.linear(activation, weight_out).reshape(shape)
