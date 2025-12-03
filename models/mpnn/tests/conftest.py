"""Test fixtures and utilities for MPNN tests."""

import pytest
from mpnn.loss.nll_loss import LabelSmoothedNLLLoss

from foundry.testing import configure_pytest, get_test_data_dir, gpu  # noqa: F401

TEST_DATA_DIR = get_test_data_dir(__file__)


def pytest_configure(config):
    """Configure pytest for MPNN tests."""
    configure_pytest(config, __file__)


@pytest.fixture
def loss_fn():
    return LabelSmoothedNLLLoss()
