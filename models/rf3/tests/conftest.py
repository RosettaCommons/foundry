from pathlib import Path

import pytest
import rootutils
import torch
from dotenv import load_dotenv

TEST_DATA_DIR = Path(__file__).resolve().parent / "data"


def pytest_configure(config):
    # Set PROJECT_ROOT
    project_root = rootutils.setup_root(
        __file__, indicator=".project-root", pythonpath=True
    )

    # Construct path to .env file at project root
    dotenv_path = project_root / ".env"

    # Check if the .env file exists
    if not dotenv_path.exists():
        raise pytest.UsageError(
            f"ERROR: Required .env file not found at {dotenv_path}. "
            f"Please create this file with the necessary environment variables."
        )

    # Load the environment variables
    load_dotenv(dotenv_path)


@pytest.fixture(scope="session")
def gpu():
    """Fixture to check GPU availability for tests that require CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")
    return True
