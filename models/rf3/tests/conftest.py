"""Test fixtures and utilities for RF3 tests."""

from foundry.testing import configure_pytest, get_test_data_dir, gpu  # noqa: F401

TEST_DATA_DIR = get_test_data_dir(__file__)

# The pre-existing per-model suite is cluster-coupled (needs IPD `/projects/ml/...` data,
# GPU, or checkpoints) and is run locally on the cluster, not in the generic-environment
# gate — it fails at collection/runtime without that data. Keep it out of the top-level
# `pytest` run; the fresh fixture-backed CPU tests in this directory are collected normally.
# New CPU tests need no change here; add a cluster-only file to this list when one is added.
collect_ignore = [
    "test_chiral_metrics.py",
    "test_inference_regression.py",
    "test_write_confidence.py",
]


def pytest_configure(config):
    """Configure pytest for RF3 tests."""
    configure_pytest(config, __file__)
