"""Repository test setup, inherited by each model suite."""

from tests.helpers import configure_pytest, gpu  # noqa: F401


def pytest_configure(config):
    configure_pytest(config, __file__)
