"""General convenience utilities for PyTorch."""

__all__ = [
    "map_to",
    "assert_no_nans",
    "assert_shape",
    "assert_same_shape",
    "scatter_mean",
]

import time
import warnings
from contextlib import contextmanager

import numpy as np
import torch
from beartype.typing import Any, Iterator, Sequence
from toolz import valmap
from torch import Tensor
from torch._prims_common import DeviceLikeType
from torch.types import _dtype

from mpnn._core import should_check_nans
from mpnn._core.common import at_least_one_exists, do_nothing


def map_to(
    x: Any,
    *,
    device: DeviceLikeType | None = None,
    dtype: _dtype | None = None,
    non_blocking: bool = False,
    **to_kwargs: Any,
) -> Any:
    """
    Recursively applies the `.to()` method to all tensors in a nested structure.

    This function handles nested structures such as dictionaries and lists, applying the `.to()` method
    to any PyTorch tensors while leaving other types unchanged.

    NOTE: If you are instantiating a new tensor, you should use the `device` and `dtype` arguments
    instead of calling `map_to()` on the tensor.
    (https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#create-tensors-directly-on-the-target-device)


    Args:
        - x (Any): The input structure, which can be a tensor, dictionary, list, or any other type.
        - device (DeviceLikeType | None): The target device to move tensors to (e.g., 'cpu', 'cuda').
        - dtype (_dtype | None): The target dtype to cast tensors to.
        - non_blocking (bool): Whether to use non-blocking transfers when possible.
        - **to_kwargs: Additional keyword arguments to pass to the `.to()` method.

    Returns:
        - Any: The input structure with all contained tensors processed by the `.to()` method.

    Example:
        >>> data = {"tensor": torch.tensor([1, 2, 3]), "list": [torch.tensor([4, 5]), "string"]}
        >>> map_to(data, device="cuda", dtype=torch.float32)
        {'tensor': tensor([1., 2., 3.], device='cuda:0', dtype=torch.float32),
         'list': [tensor([4., 5.], device='cuda:0', dtype=torch.float32), 'string']}
    """
    torch._assert(
        at_least_one_exists(device, dtype),
        "Must provide at least one of `device` or `dtype`",
    )

    if isinstance(x, dict):
        return valmap(
            lambda v: map_to(
                v, device=device, dtype=dtype, non_blocking=non_blocking, **to_kwargs
            ),
            x,
        )
    elif isinstance(x, (list, tuple)):
        return type(x)(
            map(
                lambda v: map_to(
                    v,
                    device=device,
                    dtype=dtype,
                    non_blocking=non_blocking,
                    **to_kwargs,
                ),
                x,
            )
        )
    elif isinstance(x, Tensor):
        return x.to(device=device, dtype=dtype, non_blocking=non_blocking, **to_kwargs)
    else:
        return x


def _assert_no_nans(x: Any, *, msg: str = "", fail_if_not_tensor: bool = False) -> None:
    """Recursively checks for NaN values in tensor-like objects.

    Args:
        - x (Any): Input to check for NaNs. Can be a tensor, dict, list, tuple, or other type.
        - msg (str): Prefix for error messages.
        - fail_if_not_tensor (bool): If True, raises error for non-tensor types.
    """
    if isinstance(x, Tensor):
        torch._assert(
            not torch.isnan(x).any(),
            ": ".join(filter(bool, [msg, "Tensor contains NaNs!"])),
        )
    elif isinstance(x, np.ndarray):
        torch._assert(
            not np.isnan(x).any(),
            ": ".join(filter(bool, [msg, "Numpy array contains NaNs!"])),
        )
    elif isinstance(x, float):
        torch._assert(
            not np.isnan(x),
            ": ".join(filter(bool, [msg, "float is NaN!"])),
        )
    elif isinstance(x, dict):
        for k, v in x.items():
            _assert_no_nans(
                v,
                msg=".".join(filter(bool, [msg, k])),
                fail_if_not_tensor=fail_if_not_tensor,
            )
    elif isinstance(x, (list, tuple)):
        for idx, v in enumerate(x):
            _assert_no_nans(
                v,
                msg=".".join(filter(bool, [msg, str(idx)])),
                fail_if_not_tensor=fail_if_not_tensor,
            )
    elif fail_if_not_tensor:
        raise ValueError(f"Unsupported type: {type(x)}")


assert_no_nans = _assert_no_nans if should_check_nans else do_nothing


@contextmanager
def _suppress_tracer_warnings() -> Iterator[None]:
    """
    Context manager to temporarily suppress known warnings in torch.jit.trace().
    Note: Cannot use catch_warnings because of https://bugs.python.org/issue29672

    References:
        - https://github.com/NVlabs/edm2/blob/main/torch_utils/misc.py
    """
    tracer_warning_filter = ("ignore", None, torch.jit.TracerWarning, None, 0)
    # warnings.filters is typed as an immutable Sequence in typeshed but is a mutable
    # list at runtime, so insert/remove are valid.
    warnings.filters.insert(0, tracer_warning_filter)  # type: ignore[attr-defined]
    yield
    warnings.filters.remove(tracer_warning_filter)  # type: ignore[attr-defined]


def assert_shape(tensor: Tensor, ref_shape: Sequence[int | None]) -> None:
    """
    Assert that the shape of a tensor matches the given list of integers.
    None indicates that the size of a dimension is allowed to vary.
    Performs symbolic assertion when used in torch.jit.trace().

    Args:
        - tensor (Tensor): The tensor to check the shape of.
        - ref_shape (Sequence[int | None]): The expected shape of the tensor.

    References:
        - https://github.com/NVlabs/edm2/blob/main/torch_utils/misc.py
    """

    if tensor.ndim != len(ref_shape):
        raise AssertionError(
            f"Wrong number of dimensions: got {tensor.ndim}, expected {len(ref_shape)}"
        )

    for idx, (size, ref_size) in enumerate(zip(tensor.shape, ref_shape)):
        if tensor.ndim != len(ref_shape):
            raise AssertionError(
                f"Wrong number of dimensions: got {tensor.ndim}, expected {len(ref_shape)}"
            )

        for idx, (size, ref_size) in enumerate(zip(tensor.shape, ref_shape)):
            if ref_size is None:
                pass
            elif isinstance(ref_size, torch.Tensor):
                with (
                    _suppress_tracer_warnings()
                ):  # as_tensor results are registered as constants
                    torch._assert(
                        torch.equal(torch.as_tensor(size), ref_size),
                        f"Wrong size for dimension {idx}",
                    )
            elif isinstance(size, torch.Tensor):
                with (
                    _suppress_tracer_warnings()
                ):  # as_tensor results are registered as constants
                    torch._assert(
                        torch.equal(size, torch.as_tensor(ref_size)),
                        f"Wrong size for dimension {idx}: expected {ref_size}",
                    )
            elif size != ref_size:
                raise AssertionError(
                    f"Wrong size for dimension {idx}: got {size}, expected {ref_size}"
                )


def assert_same_shape(tensor: Tensor, ref_tensor: Tensor) -> None:
    """Assert that two tensors have the same shape."""
    assert_shape(tensor, ref_tensor.shape)


class _SegmentMean(torch.autograd.Function):
    """Fixed-order mean with a differentiable backward (including double backward)."""

    @staticmethod
    def forward(ctx: Any, source: Tensor, lengths: Tensor) -> Tensor:
        ctx.save_for_backward(lengths)
        ctx.source_size = source.shape[0]
        return torch.segment_reduce(source, "mean", lengths=lengths, axis=0)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None]:
        (lengths,) = ctx.saved_tensors
        shape = (-1,) + (1,) * (grad_output.ndim - 1)
        grad = grad_output / lengths.clamp_min(1).view(shape)
        return grad.repeat_interleave(lengths, dim=0, output_size=ctx.source_size), None


def scatter_mean(zeros: Tensor, dim: int, index: Tensor, source: Tensor) -> Tensor:
    """Average source entries by index, excluding the destination from each mean.

    Unreferenced destinations retain their original values; neither input is mutated.
    CPU/CUDA use a stable ordering and segmented reduction, so repeated calls with
    identical inputs are deterministic without changing global PyTorch settings.
    Half and BF16 inputs accumulate in FP32 and are cast back after division.
    MPS uses a scatter-add fallback because segment_reduce is unavailable there.

    Args:
        zeros: Destination tensor, shape (..., I, C); need not contain zeros.
        dim: Dimension to scatter along.
        index: 1D int32/int64 tensor of length N mapping source to destination.
        source: Same dtype/device/shape as zeros except size N along dim.

    Returns:
        Tensor with the same shape and dtype as zeros.
    """
    if source.ndim != zeros.ndim or not -source.ndim <= dim < source.ndim:
        raise ValueError(
            "source and destination must have matching ranks and a valid dim"
        )
    dim %= source.ndim
    if index.ndim != 1 or index.dtype not in (torch.int32, torch.int64):
        raise ValueError("index must be a one-dimensional int32 or int64 tensor")
    if source.dtype != zeros.dtype or not source.is_floating_point():
        raise ValueError(
            "source and destination must have the same floating-point dtype"
        )
    if source.device != zeros.device or index.device != source.device:
        raise ValueError("source, destination, and index must be on the same device")
    if source.shape[dim] != index.numel() or any(
        source.shape[d] != zeros.shape[d] for d in range(source.ndim) if d != dim
    ):
        raise ValueError("source shape must match destination except along dim")

    index = index.long()
    # Integer counts avoid low-precision count saturation. bincount also rejects
    # negative indices; an oversized count vector detects out-of-range indices.
    lengths = torch.bincount(
        index.cpu() if index.device.type == "mps" else index,
        minlength=zeros.shape[dim],
    ).to(index.device)
    if lengths.numel() != zeros.shape[dim]:
        raise IndexError("scatter_mean index exceeds destination size")
    if source.numel() == 0:
        # Keep both autograd edges, including a zero-sized destination.
        return zeros + source.sum() * 0
    src = source.movedim(dim, 0)
    dst = zeros.movedim(dim, 0)
    shape = (-1,) + (1,) * (src.ndim - 1)
    if source.device.type == "mps":
        if src.dtype in (torch.float16, torch.bfloat16):
            src = src.float()
        expanded = index.view((-1,) + (1,) * (src.ndim - 1)).expand_as(src)
        total = torch.zeros_like(dst, dtype=src.dtype).scatter_add(0, expanded, src)
        means = total / lengths.clamp_min(1).view(shape)
    else:
        # Stable sorting preserves the source order within each group. Sorting
        # and materialization scale with the source, not source x destination.
        order = torch.argsort(index, stable=True)
        ordered = src.index_select(0, order)
        if ordered.dtype in (torch.float16, torch.bfloat16):
            ordered = ordered.float()
        means = _SegmentMean.apply(ordered, lengths)
    return torch.where(lengths.view(shape) > 0, means.to(zeros.dtype), dst).movedim(
        0, dim
    )


def device_of(obj: Any) -> torch.device:
    """Get the device of a PyTorch object, e.g. a `nn.Module` or a `Tensor`."""
    if hasattr(obj, "device"):
        return obj.device
    elif hasattr(obj, "parameters"):
        return next(obj.parameters()).device
    else:
        raise ValueError(f"Unsupported type: {type(obj)}")


class Timer:
    """
    A simple timer class for measuring elapsed time.

    This class provides functionality to start, stop, reset, and measure elapsed time.
    It can optionally use CUDA or MPS synchronization barriers for more accurate timing
    when working with GPU operations.

    Attributes:
        name_ (str): The name of the timer.
        elapsed_ (float): The total elapsed time.
        started_ (bool): Flag indicating if the timer is currently running.
        start_time (float): The start time of the current timing session.
        use_barrier (bool): Whether to use CUDA or MPS synchronization barriers.

    Args:
        name (str): The name of the timer.
        use_barrier (bool, optional): Whether to use synchronization barriers. Defaults to True.
    """

    def __init__(self, name: str, use_barrier: bool = True):
        self.name_ = name
        self.elapsed_ = 0.0
        self.started_ = False
        self.start_time = time.time()
        self.use_barrier = use_barrier

    def start(self) -> None:
        """Start the timer."""
        assert not self.started_, f"timer {self.name_} has already been started"
        if self.use_barrier and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif self.use_barrier and torch.backends.mps.is_available():
            torch.mps.synchronize()
        self.start_time = time.time()
        self.started_ = True

    def stop(self) -> None:
        """Stop the timer."""
        assert self.started_, f"timer {self.name_} is not started"
        if self.use_barrier and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif self.use_barrier and torch.backends.mps.is_available():
            torch.mps.synchronize()
        self.elapsed_ += time.time() - self.start_time
        self.started_ = False

    def reset(self) -> None:
        """Reset timer."""
        self.elapsed_ = 0.0
        self.started_ = False

    def elapsed(self, reset: bool = True) -> float:
        """Calculate the elapsed time."""
        started_ = self.started_
        # If the timing in progress, end it first.
        if self.started_:
            self.stop()
        # Get the elapsed time.
        elapsed_ = self.elapsed_
        # Reset the elapsed time
        if reset:
            self.reset()
        # If timing was in progress, set it back.
        if started_:
            self.start()
        return elapsed_


class Timers:
    """
    A collection of named Timer objects.

    This class manages multiple Timer instances, allowing for easy creation,
    starting, stopping, resetting, and querying of elapsed times for multiple timers.

    Attributes:
        timers (dict): A dictionary of Timer objects, keyed by their names.
    """

    def __init__(self) -> None:
        self.timers: dict[str, Timer] = {}

    def __call__(self, name: str, use_barrier: bool = True) -> Timer:
        """Get or create a Timer object."""
        if name not in self.timers:
            self.timers[name] = Timer(name, use_barrier=use_barrier)
        return self.timers[name]

    def start(self, *names: str) -> None:
        """Start the specified timers."""
        for name in names:
            self(name).start()

    def stop(self, *names: str) -> None:
        """Stop the specified timers."""
        for name in names:
            self.timers[name].stop()

    def reset(self, *names: str) -> None:
        """Reset the specified timers."""
        for name in names:
            self.timers[name].reset()

    def elapsed(self, *names: str, reset: bool = True) -> dict[str, float]:
        """Get the elapsed time for the specified timers."""
        return {name: self.timers[name].elapsed(reset=reset) for name in names}
