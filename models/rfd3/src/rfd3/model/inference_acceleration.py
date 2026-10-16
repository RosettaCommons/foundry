"""Rollout caches and native dispatch for optional RFD3 inference kernels.

The rollout-local cache follows the invariant-projection optimization in
anthropics/uplifting-biomolecular-modeling (f4f62fa). It keeps input references
alive, separates CFG inputs, and is discarded even if sampling raises.
"""

from contextlib import contextmanager
from contextvars import ContextVar

import torch
from torch.utils._pytree import tree_map

_cache = ContextVar("rfd3_rollout_cache", default=None)


@contextmanager
def rollout_cache(enabled=True, constants=(), *, atom_layout=True):
    token = _cache.set(
        {"constants": tuple(constants), "atom_layout": atom_layout}
        if enabled and not torch.is_grad_enabled()
        else None
    )
    try:
        yield
    finally:
        _cache.reset(token)


def is_constant(x):
    cache = _cache.get()
    return cache is not None and any(x is t for t in cache["constants"])


def indexing_enabled():
    cache = _cache.get()
    return cache is not None and cache["atom_layout"] and not torch.is_grad_enabled()


def cached_projection(module, *inputs, **kwargs):
    cache = _cache.get()
    if cache is None or torch.is_grad_enabled():
        return module(*inputs, **kwargs)
    # Only used with rollout constants, never coordinate-dependent pair features.
    key = (
        id(module),
        *(id(x) for x in inputs),
        tuple((name, id(value)) for name, value in sorted(kwargs.items())),
        torch.is_autocast_enabled("cuda"),
        torch.get_autocast_dtype("cuda"),
    )
    if key not in cache:
        cache[key] = (inputs, kwargs, module(*inputs, **kwargs))
    return cache[key][2]


def enabled(module, x):
    return (
        getattr(module, "inference_kernel_backend", "torch") == "triton"
        and not module.training
        and not torch.is_grad_enabled()
        and x.is_cuda
    )


def graph_enabled(module, x):
    return (
        getattr(module, "inference_cuda_graph", False)
        and not module.training
        and not torch.is_grad_enabled()
        and x.is_cuda
        and _cache.get() is not None
    )


def graph_call(module, fn, inputs, *, full, constant_inputs=()):
    """Replay a tensor-only stack, retaining graphs only for this rollout.

    Neighbor selection happens before this boundary. Variable inputs, including
    conditioning and neighbor indices, are refreshed on every call. Immutable
    rollout constants and constant_inputs are retained, with their identities in
    the graph key. Output ownership belongs to the caller. Shape-specific graphs
    separate different CFG reference shapes.
    Capture errors propagate rather than silently reporting an accelerated run.
    """
    cache = _cache.get()
    # Rollout constants are immutable and retained by the context. In particular,
    # do not clone/copy the large P_LL or lose its cached bias projections.
    constant = tuple(
        is_constant(x) or i in constant_inputs for i, x in enumerate(inputs)
    )
    signature = tuple(
        (tuple(x.shape), x.dtype, x.device, id(x) if fixed else None)
        for x, fixed in zip(inputs, constant)
    )
    key = (
        "graph",
        id(module),
        full,
        signature,
        torch.is_autocast_enabled("cuda"),
        torch.get_autocast_dtype("cuda"),
    )
    if key not in cache:
        static = tuple(x if fixed else x.clone() for x, fixed in zip(inputs, constant))
        stream = torch.cuda.Stream(device=inputs[0].device)
        stream.wait_stream(torch.cuda.current_stream())
        # Autocast's global weight cache must not escape graph-private storage.
        with (
            torch.cuda.stream(stream),
            torch.autocast(
                "cuda",
                enabled=torch.is_autocast_enabled("cuda"),
                dtype=torch.get_autocast_dtype("cuda"),
                cache_enabled=False,
            ),
        ):
            for _ in range(2):
                fn(*static, full=full)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with (
            torch.cuda.graph(graph, stream=stream),
            torch.autocast(
                "cuda",
                enabled=torch.is_autocast_enabled("cuda"),
                dtype=torch.get_autocast_dtype("cuda"),
                cache_enabled=False,
            ),
        ):
            output = fn(*static, full=full)
        cache[key] = (static, graph, output)
    static, graph, output = cache[key]
    for dst, src, fixed in zip(static, inputs, constant):
        if not fixed:
            dst.copy_(src)
    graph.replay()
    return tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, output)


def gather_attention(q, k, v, bias, indices, heads, gate):
    """Index-set attention, with the dense path's bf16 Q/K rounding.

    The index producer sorts once per denoiser call. Duplicate indices count once,
    matching dense masked attention. No dense masked bias or gathered K/V is built.
    """
    import triton
    from rfd3.model.kernels.gather import _gather_attn_fwd

    batch, length, channels = q.shape
    dh = channels // heads
    # Denoiser prepares int32 once; standalone attention / initializer calls
    # supply the model's int64 indices and must also handle multi-chain order.
    if indices.dtype != torch.int32:
        indices = indices.sort(dim=-1).values.to(torch.int32)
    bias = bias.unsqueeze(0) if bias.ndim == 3 else bias
    out = torch.empty_like(v)
    g = gate if gate is not None else out
    _gather_attn_fwd[(triton.cdiv(length, 16), heads, batch)](
        q,
        k,
        v,
        bias,
        indices,
        g,
        out,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        0 if bias.shape[0] == 1 else bias.stride(0),
        bias.stride(1),
        bias.stride(2),
        bias.stride(3),
        0 if indices.shape[0] == 1 else indices.stride(0),
        indices.stride(1),
        g.stride(0),
        g.stride(1),
        out.stride(0),
        out.stride(1),
        length,
        indices.shape[-1],
        dh**-0.5,
        HEAD_DIM=dh,
        DPAD=max(16, triton.next_power_of_2(dh)),
        BQ=16,
        KC=8,
        HAS_GATE=gate is not None,
        ROUND_QK=True,
        num_warps=4,
        num_stages=2,
    )
    return out


def fused_transition(module, x):
    """Run the native RFD3 two-stage transition on supported inference inputs.

    Separate BF16 weights are retained for one rollout, without changing model
    parameters or checkpoint keys. RMSNorm is performed by the caller.
    """
    if (
        x.shape[-1] not in (128, 256)
        or not torch.is_autocast_enabled("cuda")
        or torch.get_autocast_dtype("cuda") != torch.bfloat16
    ):
        return None
    from rfd3.model.kernels.transition import transition

    if torch.compiler.is_compiling():
        from rfd3.model.kernel_ops import transition as compiled_transition

        weights = tuple(
            layer.weight.to(torch.bfloat16).contiguous()
            for layer in (module.linear_1, module.linear_2, module.linear_3)
        )
        return compiled_transition(x, *weights)

    cache = _cache.get()
    key = ("transition_weights", id(module))
    if cache is None or key not in cache:
        weights = tuple(
            layer.weight.to(torch.bfloat16).contiguous()
            for layer in (module.linear_1, module.linear_2, module.linear_3)
        )
        if cache is not None:
            cache[key] = weights
    else:
        weights = cache[key]
    return transition(x, *weights)
