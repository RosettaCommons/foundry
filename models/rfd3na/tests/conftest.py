"""Test configuration for rfd3na tests."""

from foundry.testing import configure_pytest

# The pre-existing rfd3na suite is cluster-coupled: every file below builds pipelines via
# `rfd3na.testing.testing_utils` / `build_pipelines(...)` at module import (needing IPD
# `/projects/ml/...` data, configs, or the `transforms/regression_test_data`), so it fails
# at collection in the generic gate and is run locally on the cluster. Keep it out of CI;
# the fresh fixture-backed CPU tests in this directory (`test_rfd3na_*`) are collected
# normally. Drop a file from this list once it is made CPU-portable.
collect_ignore = [
    "test_aa_design.py",
    "test_conditioning.py",
    "test_glycines.py",
    "test_legacy_pipeline_equivalence.py",
    "test_metrics.py",
    "test_partial_diffusion.py",
    "test_selections.py",
    "test_subgraph_sampling.py",
    "test_symmetry.py",
    "test_tokenization.py",
    "test_unindexing.py",
    "transforms",
]


def pytest_configure(config):
    """Configure pytest for rfd3na tests."""
    configure_pytest(config, __file__)
