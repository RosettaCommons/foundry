"""Test fixtures and utilities for foundry tests."""

from foundry.testing import configure_pytest, get_test_data_dir, gpu  # noqa: F401

TEST_DATA_DIR = get_test_data_dir(__file__)


def pytest_configure(config):
    """Configure pytest for foundry tests."""
    configure_pytest(config, __file__)
