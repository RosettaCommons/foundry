"""Shared pytest fixtures for foundry tests."""

from pathlib import Path

import pytest
import rootutils
import torch
from dotenv import load_dotenv


@pytest.fixture(scope="session")
def gpu() -> bool:
    """Fixture to check GPU availability for tests that require CUDA or MPS."""
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        pytest.skip("GPU not available")
    return True


def get_test_data_dir(conftest_file: str) -> Path:
    """Get TEST_DATA_DIR path relative to conftest file."""
    return Path(conftest_file).resolve().parent / "data"


def configure_pytest(config: pytest.Config, conftest_file: str) -> None:
    """Configure pytest for foundry tests.
    Sets up project root and environment variables.
    """
    # Setup the project root
    rootutils.setup_root(conftest_file, indicator=".project-root", pythonpath=True)

    # Setup environment variables
    load_dotenv(override=True)
