from pathlib import Path

import pytest
import rootutils
import torch
from dotenv import load_dotenv

TEST_DATA_DIR = Path(__file__).resolve().parent / "data"


def pytest_configure(config):
    import sys

    # Set PROJECT_ROOT
    project_root = rootutils.setup_root(
        __file__, indicator=".project-root", pythonpath=True
    )

    # Add models/rf3/src to path so RF3 modules can be imported
    rf3_src = project_root / "models" / "rf3" / "src"
    if rf3_src.exists() and str(rf3_src) not in sys.path:
        sys.path.insert(0, str(rf3_src))

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
