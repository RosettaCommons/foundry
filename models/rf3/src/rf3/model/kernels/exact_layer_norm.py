# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Anthropic, PBC. See NOTICE and LICENSE.
"""ATen-exact LayerNorm forward, adapted from uplifting-biomolecular-modeling.

The CUDA arithmetic is unchanged. This adapter omits upstream's residual fusion,
prebuilt binaries, disk cache and broad provider registry. Kernels compile once
per device/dtype/width/affine in each process using NVRTC shipped with CUDA Torch.
Launch buffers belong to the calling thread, and modules are cached per device.
"""

from __future__ import annotations

import ctypes
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from cuda.bindings import driver, nvrtc

MAX_C = 1024
BLOCK_THREADS = 256
BLOCKS_PER_SM = 8
_T = {"float32": "float", "bfloat16": "exactln::bf16_t"}
_LOCAL = threading.local()


@lru_cache(maxsize=1)
def _source() -> bytes:
    return Path(__file__).with_name("exactln_fwd.cu").read_bytes()


def _device_facts():
    p = torch.cuda.get_device_properties(torch.cuda.current_device())
    return f"{p.major}{p.minor}", p.multi_processor_count


@lru_cache(maxsize=None)
def _compile(tin, tpar, tout, C, affine, rpw, device):
    # The caller holds torch's device guard. Device is part of the cache key;
    # CUmodules cannot be reused across devices/primary CUDA contexts.
    cc, _ = _device_facts()
    expr = (
        f"exactln::exactln_fwd<{_T[tin]}, {_T[tpar]}, {_T[tout]}, {C}, {affine}, {rpw}>"
    )
    cubin, lowered = _nvrtc(expr, _opts(cc))
    torch.cuda.current_stream()
    _ensure_ctx()
    module = _check(driver.cuModuleLoadData(cubin), "cuModuleLoadData")[0]
    func = _check(driver.cuModuleGetFunction(module, lowered), "cuModuleGetFunction")[0]
    return func, module


class Unsupported(Exception):
    """A call the replica does not serve, by reason word (the caller takes torch's own layer_norm for it)."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _check(res, what: str):
    """cuda.bindings calls return (err, *values); raise by name on error, return the values."""
    err = res[0]
    if int(err) != 0:
        raise RuntimeError(f"exactln: {what} failed: {err}")
    return res[1:] if len(res) > 1 else ()


def _opts(cc: Optional[str]):
    """The NVRTC options of every exactln compile (arch first; None = without the arch option, for the bundle index)."""
    # no --use_fast_math, no --ftz: the fp32 operations are explicit intrinsics; rsqrtf keeps its non-FTZ expansion (as ATen's build)
    tail = [
        b"--std=c++17",
        b"-default-device",
        b"--ftz=false",
        b"--prec-div=true",
        b"--fmad=true",
    ]
    return ([f"--gpu-architecture=sm_{cc}".encode()] + tail) if cc else tail


def _nvrtc(expr: str, opts) -> Tuple[bytes, bytes]:
    """One NVRTC compile of the carried source for the name expression `expr`: (cubin, lowered name)."""
    prog = _check(
        nvrtc.nvrtcCreateProgram(_source(), b"exactln_fwd.cu", 0, [], []),
        "nvrtcCreateProgram",
    )[0]
    try:
        _check(
            nvrtc.nvrtcAddNameExpression(prog, expr.encode()), "nvrtcAddNameExpression"
        )
        err = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)[0]
        if int(err) != 0:
            n = _check(nvrtc.nvrtcGetProgramLogSize(prog), "nvrtcGetProgramLogSize")[0]
            log = b" " * n
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError(
                f"exactln: NVRTC compile of {expr} failed:\n{log.decode(errors='replace')[:4000]}"
            )
        lowered = _check(
            nvrtc.nvrtcGetLoweredName(prog, expr.encode()), "nvrtcGetLoweredName"
        )[0]
        lowered = bytes(lowered) if not isinstance(lowered, bytes) else lowered
        n = _check(nvrtc.nvrtcGetCUBINSize(prog), "nvrtcGetCUBINSize")[0]
        cubin = b" " * n
        _check((nvrtc.nvrtcGetCUBIN(prog, cubin)[0],), "nvrtcGetCUBIN")
    finally:
        nvrtc.nvrtcDestroyProgram(prog)
    return bytes(cubin), bytes(lowered)


def _ensure_ctx() -> None:
    """Make torch's primary context current on this thread for the driver-API calls (it is after any torch CUDA op on the thread)."""
    ctx = _check(driver.cuCtxGetCurrent(), "cuCtxGetCurrent")[0]
    if int(ctx) == 0:
        dev = _check(driver.cuDeviceGet(torch.cuda.current_device()), "cuDeviceGet")[0]
        pctx = _check(driver.cuDevicePrimaryCtxRetain(dev), "cuDevicePrimaryCtxRetain")[
            0
        ]
        _check((driver.cuCtxSetCurrent(pctx)[0],), "cuCtxSetCurrent")


def _rows_per_warp(C: int) -> int:
    return 2 if C <= 64 else 1


def _collapse_leading(sizes, strides) -> Optional[Tuple[int, int, int, int]]:
    """Leading dims (all but the last) -> (rows, inner, stride_outer, stride_inner) when they form at most two stride groups, else None.
    Size-1 dims are dropped; adjacent dims merge when the outer stride equals inner stride * inner size."""
    dims = [(int(s), int(st)) for s, st in zip(sizes, strides) if int(s) != 1]
    rows = 1
    for s, _ in dims:
        rows *= s
    if not dims:
        return 1, 1, 0, 0
    groups = [list(dims[-1])]  # innermost leading dim first
    for s, st in reversed(dims[:-1]):
        gs, gst = groups[-1]
        if st == gst * gs:
            groups[-1][0] = gs * s  # merge into the current group
        else:
            groups.append([s, st])
    if len(groups) == 1:
        return rows, rows, 0, groups[0][1]
    if len(groups) == 2:
        (inner, st_in), (_outer, st_out) = groups[0], groups[1]
        return rows, inner, st_out, st_in
    return None


def plan(
    x: torch.Tensor,
    normalized_shape,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    widen: bool = False,
) -> Dict[str, Any]:
    """The launch plan for a call, or Unsupported(reason) — the reasons are the census words of the caller.  ``widen``: the autocast form of a
    call on a bf16 x — parameters fp32, arithmetic on float(x), fp32 output (ATen runs its float kernel on autocast's fp32 copy of x; the replica
    reads the bf16 storage and widens in the load, which is that copy's value)."""
    if not x.is_cuda:
        raise Unsupported("not_cuda")
    ns = tuple(
        int(v)
        for v in (
            normalized_shape
            if isinstance(normalized_shape, (tuple, list, torch.Size))
            else (normalized_shape,)
        )
    )
    if len(ns) != 1 or x.dim() < 1 or int(x.shape[-1]) != ns[0]:
        raise Unsupported(
            "normalized_shape"
        )  # ATen serves multi-dim normalized shapes with the same kernel over N = prod; not needed here
    C = ns[0]
    if C % 4 != 0 or C < 4 or C > MAX_C:
        raise Unsupported(f"width:{C}")
    if x.dtype == torch.float32:
        tin, esize = "float32", 4
    elif x.dtype == torch.bfloat16:
        tin, esize = "bfloat16", 2
    else:
        raise Unsupported(f"dtype:{x.dtype}".replace("torch.", ""))
    if widen and tin != "bfloat16":
        raise Unsupported("widen_form")
    pdtype, psize = (torch.float32, 4) if widen else (x.dtype, esize)
    tpar = "float32" if widen else tin
    tout = "float32" if widen else tin
    for name, t in (("weight", weight), ("bias", bias)):
        if t is not None:
            if (
                not t.is_cuda
                or t.device != x.device
                or t.dtype != pdtype
                or tuple(t.shape) != (C,)
            ):
                raise Unsupported(
                    f"{name}_form"
                )  # F.layer_norm itself raises on a dtype mismatch; never reached from nn.LayerNorm
            if not t.is_contiguous():
                raise Unsupported(
                    f"{name}_noncontig"
                )  # ATen: expect_contiguous copies it; rare — not served
            if t.data_ptr() % (4 * psize) != 0:
                raise Unsupported(
                    "aten_rowwise_path"
                )  # ATen's can_vectorize(gamma/beta) fails -> RowwiseMoments arithmetic
    affine = (1 if weight is not None else 0) | (2 if bias is not None else 0)
    if x.numel() == 0:
        raise Unsupported("empty")
    if x.stride(-1) != 1 and x.shape[-1] != 1:
        copy = True  # ATen copies; so do we (below)
        lay = None
    else:
        lay = _collapse_leading(x.shape[:-1], x.stride()[:-1])
        copy = lay is None
    align = 4 * esize
    if not copy:
        assert lay is not None
        rows, inner, st_out, st_in = lay
        aligned = (
            (x.data_ptr() % align == 0)
            and (st_in * esize) % align == 0
            and (st_out * esize) % align == 0
        )
        if not aligned:
            if x.is_contiguous() and not widen:
                raise Unsupported(
                    "aten_rowwise_path"
                )  # contiguous but misaligned: ATen does NOT copy and takes the non-vectorized kernels
            copy = True  # non-contiguous (or autocast's fresh fp32 copy): ATen works on a fresh aligned buffer; so do we
    if copy:
        rows = 1
        for s in x.shape[:-1]:
            rows *= int(s)
        inner, st_out, st_in = rows, 0, C
    if rows >= 2**31 - 2**20:
        raise Unsupported("rows_int32")
    return {
        "C": C,
        "tin": tin,
        "tpar": tpar,
        "tout": tout,
        "affine": affine,
        "rows": int(rows),
        "inner": int(inner),
        "stride_outer": int(st_out),
        "stride_inner": int(st_in),
        "copy": bool(copy),
        "rpw": _rows_per_warp(C),
        "widen": bool(widen),
    }


class _Launcher:
    """cuLaunchKernel with a prebuilt void** parameter block: per call only the argument VALUES are poked (pointers / ints), no tuple parsing.
    One instance per cached call signature; the argument layout is fixed by `spec` = sequence of ctypes types."""

    __slots__ = ("func", "store", "params", "addr", "n")

    def __init__(self, func, spec):
        self.func = func
        self.store = [t() for t in spec]
        self.n = len(spec)
        arr = (ctypes.c_void_p * self.n)()
        for i, c in enumerate(self.store):
            arr[i] = ctypes.addressof(c)
        self.params = arr
        self.addr = ctypes.addressof(arr)

    def launch(self, values, blocks: int, stream_handle) -> None:
        st = self.store
        for i in range(self.n):
            st[i].value = values[i]
        _check(
            (
                driver.cuLaunchKernel(
                    self.func,
                    blocks,
                    1,
                    1,
                    BLOCK_THREADS,
                    1,
                    1,
                    0,
                    stream_handle,
                    self.addr,
                    0,
                )[0],
            ),
            "cuLaunchKernel",
        )


def _grid(rows: int, rpw: int):
    _, sms = _device_facts()
    warps_per_block = BLOCK_THREADS // 32
    warps = (rows + rpw - 1) // rpw
    blocks = max(
        1, min((warps + warps_per_block - 1) // warps_per_block, sms * BLOCKS_PER_SM)
    )
    return blocks, blocks * warps_per_block * rpw


def call_signature(
    x: torch.Tensor, normalized_shape, weight, bias, widen: bool
) -> tuple:
    """Everything plan() depends on, as a hashable key (shape, strides, dtypes, 16-byte alignment classes, parameter identity)."""
    return (
        tuple(x.shape),
        tuple(x.stride()),
        x.dtype,
        x.data_ptr() & 15,
        x.device.index,
        None if weight is None else (weight.data_ptr(), weight.dtype),
        None if bias is None else (bias.data_ptr(), bias.dtype),
        tuple(normalized_shape)
        if isinstance(normalized_shape, (tuple, list, torch.Size))
        else (int(normalized_shape),),
        bool(widen),
    )


def _layer_norm(
    x: torch.Tensor,
    normalized_shape,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    widen: bool = False,
) -> torch.Tensor:
    """Exact LayerNorm; ``widen`` serves BF16 autocast with FP32 output."""
    if not hasattr(_LOCAL, "plans"):
        _LOCAL.plans = {}
    _PLANS = _LOCAL.plans
    sig = call_signature(x, normalized_shape, weight, bias, widen)
    ent = _PLANS.get(sig)
    if ent is None:
        p = plan(x, normalized_shape, weight, bias, widen=widen)
        tout_ = p["tout"]
        func_, _m = _compile(
            p["tin"], p["tpar"], tout_, p["C"], p["affine"], p["rpw"], x.get_device()
        )
        blocks_, row_step_ = _grid(p["rows"], p["rpw"])
        ent = (
            p,
            _Launcher(func_, _LN_SPEC),
            blocks_,
            row_step_,
            {"float32": torch.float32, "bfloat16": torch.bfloat16}[tout_],
        )
        if len(_PLANS) < 4096:
            _PLANS[sig] = ent
    p, launcher, blocks, row_step, ydt = ent
    rows = p["rows"]
    if p["copy"]:
        x = x.clone(memory_format=torch.contiguous_format)
    y = torch.empty(x.shape, dtype=ydt, device=x.device)
    g = weight if weight is not None else x
    b = bias if bias is not None else x
    stream = torch.cuda.current_stream(x.device)
    with torch.cuda.device(x.device):
        launcher.launch(
            (
                x.data_ptr(),
                g.data_ptr(),
                b.data_ptr(),
                y.data_ptr(),
                int(rows),
                int(p["inner"]),
                int(p["stride_outer"]),
                int(p["stride_inner"]),
                float(eps),
                int(row_step),
            ),
            blocks,
            stream.cuda_stream,
        )
    return y


_LN_SPEC = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_longlong,
    ctypes.c_longlong,
    ctypes.c_double,
    ctypes.c_int,
)


def layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5, widen=False):
    with torch.cuda.device(x.device):
        return _layer_norm(x, normalized_shape, weight, bias, eps, widen=widen)
