import sys

import rootutils

# The pre-existing per-model suite is cluster-coupled (needs IPD `/projects/ml/...` data,
# GPU, or checkpoints) and is run locally on the cluster, not in the generic-environment
# gate — most files fail at collection without that data. Keep it out of the top-level
# `pytest` run; the fresh fixture-backed CPU tests in this directory are collected normally.
# New CPU tests need no change here; add a cluster-only file to this list when one is added.
collect_ignore = [
    "test_aa_design.py",
    "test_bond_preservation_cases.py",
    "test_conditioning.py",
    "test_glycines.py",
    "test_legacy_pipeline_equivalence.py",
    "test_legacy_ptm_bonds.py",
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
    root = rootutils.setup_root(
        __file__, indicator=".project-root", pythonpath=True, dotenv=True
    )

    paths_to_add = [
        root / "src",
        root / "models" / "rfd3" / "tests",
    ]

    for path in paths_to_add:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))

    # Add markers
    config.addinivalue_line("markers", "fast: mark test as fast (run quickly)")
    config.addinivalue_line("markers", "slow: mark test as slow (run slowly)")
