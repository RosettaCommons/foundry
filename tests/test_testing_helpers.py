"""Unit tests for the pure helper in foundry.testing.fixtures.

The ``gpu`` fixture and ``configure_pytest`` hook are environment/side-effecting
glue (GPU detection, project-root setup, dotenv loading) and are not tested.
``get_test_data_dir`` has a small but real contract: the test ``data`` directory
sits next to the conftest file that calls it.
"""

import pytest

from foundry.testing.fixtures import get_test_data_dir


def test_get_test_data_dir_is_data_next_to_conftest(tmp_path):
    conftest = tmp_path / "conftest.py"
    assert get_test_data_dir(str(conftest)) == tmp_path.resolve() / "data"


def test_get_test_data_dir_tracks_the_files_directory(tmp_path):
    nested = tmp_path / "sub" / "conftest.py"
    assert get_test_data_dir(str(nested)) == (tmp_path / "sub").resolve() / "data"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
