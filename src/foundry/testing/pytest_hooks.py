"""Shared pytest configuration hooks for foundry tests."""

import rootutils
from dotenv import load_dotenv


def configure_pytest(config, conftest_file: str) -> None:
    """Configure pytest for foundry tests.
    Sets up project root and environment variables.
    """
    # Setup the project root
    rootutils.setup_root(conftest_file, indicator=".project-root", pythonpath=True)

    # Setup environment variables
    load_dotenv(override=True)
