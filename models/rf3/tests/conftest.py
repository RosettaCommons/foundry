from pathlib import Path

import pytest
import rootutils
import torch

from modelhub.utils.env import load_ipd_dotenv

TEST_DATA_DIR = Path(__file__).resolve().parent / "data"


def pytest_configure(config):
    # Setup the project root
    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

    # Setup environment variables
    load_ipd_dotenv(override=True)


@pytest.fixture(scope="session")
def gpu():
    """Fixture to check GPU availability for tests that require CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")
    return True
