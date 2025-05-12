import os

import pytest
from dotenv import load_dotenv


def pytest_configure(config):
    # Get the directory where conftest.py is located
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct path to .env file in the parent directory
    dotenv_path = os.path.join(current_dir, "..", ".env")

    # Check if the .env file exists
    if not os.path.exists(dotenv_path):
        raise pytest.UsageError(
            f"ERROR: Required .env file not found at {dotenv_path}. "
            f"Please create this file with the necessary environment variables."
        )

    # Load the environment variables
    load_dotenv(dotenv_path)
