"""
Utilities for managing the random number generators in the current process.
Inspired by: https://github.com/Lightning-AI/pytorch-lightning/blob/709a2a9d3b79b0a436eb2d271fbeecf8a7ba1352/src/lightning/fabric/utilities/seed.py
"""

from __future__ import annotations

import random
from contextlib import contextmanager
from random import getstate as python_get_rng_state
from random import setstate as python_set_rng_state
from typing import Any, Generator

import numpy as np
import torch
from assertpy import assert_that

_MAX_SEED_VALUE = np.iinfo(np.uint32).max
_MIN_SEED_VALUE = np.iinfo(np.uint32).min


def _collect_rng_states(include_cuda: bool = True) -> dict[str, Any]:
    r"""Collect the global random state of `torch`, `torch.cuda`, `numpy` and Python in current process.

    Args:
        include_cuda (bool): Whether to include the state of the CUDA RNG. If cuda is not available, the state of
            the CUDA RNG is not included. Defaults to True.
    """
    states = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": python_get_rng_state(),
    }
    if include_cuda:
        states["torch.cuda"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return states


def _set_rng_states(rng_state_dict: dict[str, Any]) -> None:
    r"""Set the global random state of `torch`, `torch.cuda`, `numpy` and Python in the current
    process."""
    if "torch" in rng_state_dict:
        torch.set_rng_state(rng_state_dict["torch"])
    if "torch.cuda" in rng_state_dict:
        torch.cuda.set_rng_state_all(rng_state_dict["torch.cuda"])
    if "numpy" in rng_state_dict:
        np.random.set_state(rng_state_dict["numpy"])
    if "python" in rng_state_dict:
        version, state, gauss = rng_state_dict["python"]
        python_set_rng_state((version, tuple(state), gauss))


@contextmanager
def rng_state(
    rng_state_dict: dict[str, Any] | None = None, include_cuda: bool = True
) -> Generator[dict[str, Any], None, None]:
    """A context manager that resets the global random state on exit to what it was before entering.
    Within the context manager, the RNG states are set to the provided rng state in the dictionary.



    It supports isolating the states for PyTorch, Numpy, and Python built-in random number generators.

    Args:
        rng_state_dict (dict[str, Any] | None): A dictionary of RNG states to set. It can have the following keys:
            - "torch": The state of the PyTorch RNG.
            - "torch.cuda": The state of the PyTorch CUDA RNG.
            - "numpy": The state of the Numpy RNG.
            - "python": The state of the Python built-in RNG.
            If no rng_state_dict is provided, the RNG states are set to the current state of the RNGs. If the
            rng_state_dict only contains a subset of the RNG states, the other RNG states are set to the current
            state of the RNGs.
        include_cuda (bool): Whether to allow this function to also control the `torch.cuda` random number generator.
            Set this to ``False`` when using the function in a forked process where CUDA re-initialization is
            prohibited. Defaults to True.
    """
    # Collect previous states
    prev_states = _collect_rng_states(include_cuda)

    # Set desired new states within the context if provided
    if rng_state_dict is not None:
        _set_rng_states(rng_state_dict)
    yield rng_state_dict

    # Restore previous states
    _set_rng_states(prev_states)


def create_rng_state_from_seeds(
    np_seed: int | None = None, torch_seed: int | None = None, py_seed: int | None = None
) -> dict[str, Any]:
    """Create a dictionary of RNG states from the provided seeds. If no seed is provided, the current state of the RNGs is used.

    Args:
        np_seed (int | None): The seed for the Numpy RNG.
        torch_seed (int | None): The seed for the PyTorch RNG.
        py_seed (int | None): The seed for the Python built-in RNG.
    """
    with rng_state(None):
        # Set seeds in context manager to reset RNG states after creating the rng_state_dict
        if np_seed is not None:
            assert_that(np_seed).is_instance_of(int).is_between(_MIN_SEED_VALUE, _MAX_SEED_VALUE)
            np.random.seed(np_seed)
        if torch_seed is not None:
            torch.manual_seed(torch_seed)
        if py_seed is not None:
            random.seed(py_seed)
        return _collect_rng_states()


def _serialize_tensor_to_str(tensor: torch.Tensor) -> str:
    """Serialize a PyTorch tensor to a string."""
    # Serialize the tensor to a string that can be re-created
    tensor_list = tensor.tolist()
    tensor_str = repr(tensor_list)
    dtype_str = repr(tensor.dtype)
    device_str = "torch." + repr(tensor.device)
    return f"torch.tensor({tensor_str}, dtype={dtype_str}, device={device_str})"


def _serialize_array_to_str(array: np.ndarray) -> str:
    """Serialize a Numpy array to a string."""
    # Serialize the array to a string that can be re-created
    array_str = repr(array.tolist())
    dtype_str = "np." + repr(array.dtype)
    return f"np.array({array_str}, dtype={dtype_str})"


def serialize_rng_state_dict(rng_state_dict: dict[str, Any]) -> str:
    """Convert the RNG state dictionary to a string."""
    # Convert the torch tensor to a string that can be re-created

    # Serialize python state
    py_state = f"{rng_state_dict['python']}"

    # Serialize numpy state
    np_state = [
        (_serialize_array_to_str(val) if isinstance(val, np.ndarray) else repr(val)) for val in rng_state_dict["numpy"]
    ]
    np_state = "(" + ", ".join(np_state) + ")"

    # Serialize torch states
    torch_state = rng_state_dict["torch"]
    torch_state = _serialize_tensor_to_str(torch_state)
    torch_cuda_state = rng_state_dict["torch.cuda"]
    for i, val in enumerate(torch_cuda_state):
        torch_cuda_state[i] = _serialize_tensor_to_str(val) if isinstance(val, torch.Tensor) else val
    torch_cuda_state = "[" + ", ".join(torch_cuda_state) + "]"

    # Escape single quotes and surround with single quotes
    rng_state_serialized = {
        "python": py_state,
        "numpy": np_state,
        "torch": torch_state,
        "torch.cuda": torch_cuda_state,
    }
    return "{" + ",\n".join([f"'{key}': {val}" for key, val in rng_state_serialized.items()]) + "}"


if __name__ == "__main__":
    # Outside the context manager
    print("NumPy:", np.random.random(5))
    print("PyTorch:", torch.rand(5))
    print("Python random:", [random.random() for _ in range(5)])

    # Inside the context manager with fixed seeds
    with rng_state(create_rng_state_from_seeds(np_seed=42, torch_seed=42, py_seed=42)) as rng_state_dict:
        my_state = serialize_rng_state_dict(rng_state_dict)
        print("\nWithin context manager:")
        print("NumPy:", np.random.random(5))
        print("PyTorch:", torch.rand(5))
        print("Python random:", [random.random() for _ in range(5)])

    # Back to the original state outside the context manager
    print("\nBack outside the context manager:")
    print("NumPy:", np.random.random(5))
    print("PyTorch:", torch.rand(5))
    print("Python random:", [random.random() for _ in range(5)])

    # Inside the context manager with fixed seeds
    with rng_state(eval(my_state)):
        print("\nWithin context manager:")
        print("NumPy:", np.random.random(5))
        print("PyTorch:", torch.rand(5))
        print("Python random:", [random.random() for _ in range(5)])
