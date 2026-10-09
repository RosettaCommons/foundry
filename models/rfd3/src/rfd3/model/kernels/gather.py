# Copyright 2026 Anthropic, PBC. Licensed under Apache-2.0 (see LICENSE).
# Adapted from anthropics/uplifting-biomolecular-modeling at
# f4f62fa6592ae4938d49b1757bea0cfeff9f468e.
# Modified: extracted device kernels; native RFD3 dispatch and A4000 launch settings.
"""Fused index-set attention. Indices must be sorted and in range."""

import triton
import triton.language as tl


@triton.jit
def _gather_attn_fwd(
    Q,
    K,
    V,
    Bias,
    Idx,
    Gate,
    Out,
    sqb,
    sqm,
    skb,
    skn,
    svb,
    svn,
    sbb,
    sbi,
    sbj,
    sbh,
    sib,
    sim,
    sgb,
    sgm,
    sob,
    som,
    LQ,
    KTOT,
    qk_scale,
    HEAD_DIM: tl.constexpr,
    DPAD: tl.constexpr,
    BQ: tl.constexpr,
    KC: tl.constexpr,
    HAS_GATE: tl.constexpr,
    ROUND_QK: tl.constexpr,
):
    # one program = BQ queries of one (batch element, head); grid (query tiles, heads, batch): the query tiles and heads of one batch
    # element are adjacent in launch order, so its K / V rows stay L2-resident while they are gathered.
    pid_q = tl.program_id(0)
    h = tl.program_id(1)
    h64 = h.to(tl.int64)
    b64 = tl.program_id(2).to(tl.int64)
    rows = pid_q * BQ + tl.arange(0, BQ)
    row_ok = rows < LQ
    rows64 = rows.to(tl.int64)
    offs_e = tl.arange(0, DPAD)
    e_ok = offs_e < HEAD_DIM
    col0 = h * HEAD_DIM

    q = tl.load(
        Q + b64 * sqb + rows64[:, None] * sqm + col0 + offs_e[None, :],
        mask=row_ok[:, None] & e_ok[None, :],
        other=0.0,
    )
    if ROUND_QK:
        q = q.to(V.dtype.element_ty)
    q = q.to(tl.float32)

    NEG: tl.constexpr = -1.0e30
    m_i = tl.zeros([BQ], dtype=tl.float32) + NEG
    l_i = tl.zeros([BQ], dtype=tl.float32)
    acc = tl.zeros([BQ, DPAD], dtype=tl.float32)

    idx_row = Idx + b64 * sib + rows64[:, None] * sim
    k_base = K + b64 * skb + col0
    v_base = V + b64 * svb + col0
    bias_row = (
        Bias + b64 * sbb + rows64[:, None] * sbi + h64 * sbh
    )  # int64: sbh may be LQ*LK (any bias strides)

    for kc in range(0, KTOT, KC):
        jj = kc + tl.arange(0, KC)
        j_ok = jj < KTOT
        valid = row_ok[:, None] & j_ok[None, :]
        idx = tl.load(idx_row + jj[None, :], mask=valid, other=-1)  # [BQ, KC] int32
        prev = tl.load(
            idx_row + (jj[None, :] - 1), mask=valid & (jj[None, :] >= 1), other=-1
        )
        valid = (
            valid & (idx >= 0) & (idx != prev)
        )  # -1 padding and adjacent duplicates: masked
        idx64 = tl.where(valid, idx, 0).to(tl.int64)
        kg = tl.load(
            k_base + idx64[:, :, None] * skn + offs_e[None, None, :],
            mask=valid[:, :, None] & e_ok[None, None, :],
            other=0.0,
        )
        if ROUND_QK:
            kg = kg.to(V.dtype.element_ty)
        s = tl.sum(q[:, None, :] * kg.to(tl.float32), 2) * qk_scale  # [BQ, KC] fp32
        bg = tl.load(bias_row + idx64 * sbj, mask=valid, other=0.0).to(tl.float32)
        s = tl.where(valid, s + bg, NEG)
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp(s - m_new[:, None])
        p = tl.where(valid, p, 0.0)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        vg = tl.load(
            v_base + idx64[:, :, None] * svn + offs_e[None, None, :],
            mask=valid[:, :, None] & e_ok[None, None, :],
            other=0.0,
        )
        pr = p.to(V.dtype.element_ty).to(tl.float32)
        acc = acc * alpha[:, None] + tl.sum(pr[:, :, None] * vg.to(tl.float32), 1)
        m_i = m_new

    out = acc / l_i[:, None]
    if HAS_GATE:
        g = tl.load(
            Gate + b64 * sgb + rows64[:, None] * sgm + col0 + offs_e[None, :],
            mask=row_ok[:, None] & e_ok[None, :],
            other=0.0,
        )
        out = out * g.to(tl.float32)
    tl.store(
        Out + b64 * sob + rows64[:, None] * som + col0 + offs_e[None, :],
        out.to(Out.dtype.element_ty),
        mask=row_ok[:, None] & e_ok[None, :],
    )
