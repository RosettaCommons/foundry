"""Test fixtures and utilities for MPNN tests."""

from modelhub.testing import configure_pytest, get_test_data_dir, gpu  # noqa: F401

TEST_DATA_DIR = get_test_data_dir(__file__)


def pytest_configure(config):
    """Configure pytest for MPNN tests."""
    configure_pytest(config, __file__)
