"""Shared testing utilities for foundry."""

from foundry.testing.fixtures import get_test_data_dir, gpu
from foundry.testing.pytest_hooks import configure_pytest

__all__ = ["configure_pytest", "get_test_data_dir", "gpu"]
