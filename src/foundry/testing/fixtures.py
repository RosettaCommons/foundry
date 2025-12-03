"""Shared pytest fixtures for foundry tests."""

from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="session")
def gpu():
    """Fixture to check GPU availability for tests that require CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")
    return True


def get_test_data_dir(conftest_file: str) -> Path:
    """Get TEST_DATA_DIR path relative to conftest file."""
    return Path(conftest_file).resolve().parent / "data"
