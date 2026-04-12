import sys

import rootutils


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
