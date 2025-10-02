"""Environment loading utilities."""

import os
from pathlib import Path

from dotenv import load_dotenv


def load_ipd_dotenv(override: bool = True) -> None:
    """Load environment variables, prioritizing IPD-specific config.

    First checks for ``.ipd/.env`` in the project root. If found, loads it.
    Otherwise falls back to standard ``.env`` loading.

    Note:
      Requires ``PROJECT_ROOT`` environment variable to be set (via ``rootutils.setup_root()``).

    Args:
      override: Override existing environment variables. Defaults to ``True``.

    Raises:
      RuntimeError: If ``PROJECT_ROOT`` is not set.
    """
    project_root = os.environ.get("PROJECT_ROOT")
    if not project_root:
        raise RuntimeError(
            "PROJECT_ROOT environment variable not set. "
            "Call rootutils.setup_root() before load_ipd_dotenv()."
        )

    ipd_env = Path(project_root) / ".ipd" / ".env"
    if ipd_env.exists():
        load_dotenv(dotenv_path=ipd_env, override=override)
    else:
        load_dotenv(override=override)
