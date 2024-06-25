"""
Convenience utils for common validation checks in transforms.

All checks take a `data` dictionary as input and raise an error if the check fails.
"""

from __future__ import annotations

from typing import Any


def check_contains_keys(data: dict[str, Any], keys: list[str]):
    """Check if a key is in a dictionary."""
    for key in keys:
        if key not in data:
            raise KeyError(f"Key `{key}` not in data")


def check_does_not_contain_keys(data: dict[str, Any], keys: list[str]):
    """Check if a key is not in a dictionary."""
    for key in keys:
        if key in data:
            raise KeyError(f"Key `{key}` already exists in data")


def check_is_instance(data: dict[str, Any], key: str, expected_type: type):
    """Check if the value of a key in a dictionary is of a certain type."""
    if not isinstance(data[key], expected_type):
        raise ValueError(f"Key `{key}` in data is not of type `{expected_type}`")


def check_is_shape(data: dict[str, Any], key: str, expected_shape: tuple[int, ...]):
    """Check if the value of a key in a dictionary has a certain shape."""
    if data[key].shape != expected_shape:
        raise ValueError(f"Key `{key}` in data has shape {data[key].shape} but expected shape {expected_shape}")


def check_nonzero_length(data: dict[str, Any], key: str):
    """Check if the length of the value of a key in a dictionary is nonzero."""
    if len(data[key]) == 0:
        raise ValueError(f"Key {key} in data has length 0")


def check_atom_array_annotation(data: dict[str, Any], required: list[str]):
    """Check if `atom_array` key has the annotations specified in `required`."""
    annotations = set(data["atom_array"].get_annotation_categories())
    if not set(required).issubset(annotations):
        missing = set(required) - annotations
        raise ValueError(f"Key `atom_array` is missing the following annotations: {missing}")
