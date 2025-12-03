"""General convenience utilities for PyTorch."""

__all__ = ["map_to", "assert_no_nans", "assert_shape", "assert_same_shape"]

import time
import warnings
from contextlib import contextmanager

import numpy as np
import torch
from beartype.typing import Any, Sequence
from toolz import valmap
from torch import Tensor
from torch._prims_common import DeviceLikeType
from torch.types import _dtype

from foundry import should_check_nans
from foundry.common import at_least_one_exists, do_nothing


def map_to(
    x: Any,
    *,
    device: DeviceLikeType | None = None,
    dtype: _dtype | None = None,
    non_blocking: bool = False,
    **to_kwargs,
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
def _suppress_tracer_warnings():
    """
    Context manager to temporarily suppress known warnings in torch.jit.trace().
    Note: Cannot use catch_warnings because of https://bugs.python.org/issue29672

    References:
        - https://github.com/NVlabs/edm2/blob/main/torch_utils/misc.py
    """
    tracer_warning_filter = ("ignore", None, torch.jit.TracerWarning, None, 0)
    warnings.filters.insert(0, tracer_warning_filter)
    yield
    warnings.filters.remove(tracer_warning_filter)


def assert_shape(tensor: Tensor, ref_shape: Sequence[int | None]):
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

    def __init__(self, name, use_barrier: bool = True):
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

    def __init__(self):
        self.timers = {}

    def __call__(self, name, use_barrier: bool = True) -> Timer:
        """Get or create a Timer object."""
        if name not in self.timers:
            self.timers[name] = Timer(name, use_barrier=use_barrier)
        return self.timers[name]

    def start(self, *names) -> None:
        """Start the specified timers."""
        for name in names:
            self(name).start()

    def stop(self, *names) -> None:
        """Stop the specified timers."""
        for name in names:
            self.timers[name].stop()

    def reset(self, *names) -> None:
        """Reset the specified timers."""
        for name in names:
            self.timers[name].reset()

    def elapsed(self, *names, reset: bool = True) -> dict[str, float]:
        """Get the elapsed time for the specified timers."""
        return {name: self.timers[name].elapsed(reset=reset) for name in names}
