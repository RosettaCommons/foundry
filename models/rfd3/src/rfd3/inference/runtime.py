"""Inference backend policy and persistent compiler artifact location."""

import importlib.util
import os
from functools import wraps
from pathlib import Path

import torch


def dynamic_diffusion_batch(compiled):
    """Keep length buckets static while allowing CUDA diffusion batches D > 1.

    Marking happens outside Dynamo. Singleton dimensions retain their separate
    specialization, as required by PyTorch's shape semantics.
    """

    @wraps(compiled)
    def forward(x, t, *args, **kwargs):
        if x.is_cuda and x.shape[1] > 1:
            torch._dynamo.mark_dynamic(x, 1)
            torch._dynamo.mark_dynamic(t, 1)
        return compiled(x, t, *args, **kwargs)

    return forward


def select_kernel_backend(
    *, device, compile_model, inference_batch_size, low_memory_mode
):
    """Select measured inference policies, keeping attention and MLPs separable.

    Explicit triton also supports padded compilation, but the matched A4000
    small-length measurements did not show an advantage over native compilation.
    """
    if (
        low_memory_mode
        or os.environ.get("RFD3_LOW_MEMORY_MODE") == "1"
        or device.type != "cuda"
        or importlib.util.find_spec("triton") is None
    ):
        return "torch"
    capability = torch.cuda.get_device_capability(device)
    if compile_model:
        # RTX PRO 6000 (SM12x), B=4: native attention plus the independent
        # transition is faster than native-only execution in our comparison.
        # Keep unmeasured hardware/smaller input batches on the native policy.
        if capability[0] == 12 and inference_batch_size >= 4:
            return "triton-transition"
        return "torch"
    # The independent BF16 transition requires Ampere or newer.
    return "triton" if capability[0] >= 8 else "torch"


def configure_compile_cache(directory):
    """Keep Inductor artifacts across weight loads; explicit env settings win.

    This caches generated code, not model weights or live CUDA graphs. PyTorch
    owns the cache keys and invalidation for shapes, compiler versions and GPU.
    None keeps the framework's default cache location and configuration.
    """
    if directory is None:
        return {}
    from torch._inductor.runtime.cache_dir_utils import default_cache_dir

    environment = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    # Importing the compiler itself can populate this variable with its temp
    # default, before the engine exists. Do not mistake that for a user override.
    if environment == default_cache_dir():
        environment = None
    path = Path(environment or directory).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(path)
    # Include the explicitly dispatched Triton kernels as well as Inductor's
    # generated kernels. A framework-specific user override still takes priority.
    os.environ.setdefault("TRITON_CACHE_DIR", str(path / "triton"))
    return {
        "fx_graph_cache": os.environ.get("TORCHINDUCTOR_FX_GRAPH_CACHE", "1") == "1"
    }
