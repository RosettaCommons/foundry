"""Test fixtures and utilities for MPNN tests."""

import pytest
from mpnn.loss.nll_loss import LabelSmoothedNLLLoss

from foundry.testing import configure_pytest, get_test_data_dir, gpu  # noqa: F401

TEST_DATA_DIR = get_test_data_dir(__file__)

# Most of the pre-existing mpnn suite loads structures via atomworks `cached_parse`, which
# needs a DIGS PDB mirror (`/projects/...`) absent in the generic gate — those files fail at
# collection/setup and are run locally on the cluster, not in CI. Keep those out of the gate;
# the CPU-portable files (`test_samplers`, `test_feature_collator`,
# `test_polymer_ligand_interface`, `test_inference_utils`) and any fresh fixture-backed CPU
# tests added here are collected normally. Drop a file from this list once it is CPU-portable.
collect_ignore = [
    "test_inference_engine.py",
    "test_integration.py",
    "test_loss.py",
    "test_metrics.py",
    "test_model.py",
    "test_pipeline.py",
    "test_utils.py",
]


def pytest_configure(config):
    """Configure pytest for MPNN tests."""
    configure_pytest(config, __file__)


@pytest.fixture
def loss_fn():
    return LabelSmoothedNLLLoss()
