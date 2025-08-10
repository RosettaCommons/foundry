from functools import wraps

import torch
from beartype.typing import Any, Callable, Iterable
from toolz import merge_with


def run_once(fn: Callable) -> Callable:
    """Decorator to ensure a function is only executed once per process.

    Args:
        fn (Callable): The function to decorate.

    Returns:
        Callable: A wrapped function that only executes once.
    """

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if getattr(wrapper, "_has_run", False):
            return
        wrapper._has_run = True
        return fn(*args, **kwargs)

    return wrapper


def do_nothing(*args: Any, **kwargs: Any) -> None:
    """Does nothing, just returns None"""
    pass


def exists(obj: Any) -> bool:
    """True iff object is not None"""
    return obj is not None


def default(obj: Any, default: Any) -> Any:
    """Return obj if it exists, otherwise return default"""
    return obj if exists(obj) else default


def exactly_one_exists(*args: object) -> bool:
    """True iff exactly one of the arguments exists"""
    return sum(exists(arg) for arg in args) == 1


def at_least_one_exists(*args: object) -> bool:
    """True iff at least one of the arguments exists"""
    return any(exists(arg) for arg in args)


def concat_dicts(*dicts: dict) -> dict:
    """
    Concatenate a list of dicts with the same keys into a single dict.

    Example:
        >>> d1 = {"a": 1, "b": 2}
        >>> d2 = {"a": 3, "b": 4}
        >>> concat_dicts(d1, d2)
        {'a': [1, 3], 'b': [2, 4]}
    """
    return merge_with(list, *dicts)


def listmap(fn: Callable, lst: Iterable[Any]) -> list:
    """
    Apply a function to each element of a single list.

    Args:
        - fn (Callable): Function to apply to each element
        - lst (list): Input list

    Returns:
        - list: Result of applying fn to each element

    Example:
        >>> listmap(lambda x: x + 1, [1, 2, 3])
        [2, 3, 4]
    """
    return [fn(x) for x in lst]


def listmap_with_idx(fn: Callable[[int, Any], Any], lst: Iterable[Any]) -> list:
    """Maps a function over a list while providing both index and value to the function.

    A convenience wrapper around listmap that allows the mapping function to access both the index and value
    of each element in the input list.

    Args:
        - fn (Callable[[int, Any], Any]): Function that takes two arguments (index, value) and returns a transformed value.
        - lst (list): Input list to map over.

    Returns:
        - list: New list containing the results of applying fn to each (index, value) pair.

    Example:
        >>> def add_index(i, x):
        ...     return f"{i}_{x}"
        >>> listmap_with_idx(add_index, ["a", "b", "c"])
        ['0_a', '1_b', '2_c']
    """
    return [fn(idx, x) for idx, x in enumerate(lst)]


def ensure_dtype(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert tensor to target dtype if it's not already that dtype."""
    return tensor if tensor.dtype == dtype else tensor.to(dtype)
