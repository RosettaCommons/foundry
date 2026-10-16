"""Management of checkpoints"""

import os
from pathlib import Path
from typing import Iterable, List

import dotenv
from mpnn.checkpoints import REGISTERED_CHECKPOINTS as MPNN_CHECKPOINTS
from mpnn.checkpoints import RegisteredCheckpoint as MPNNCheckpoint
from rf3.checkpoints import REGISTERED_CHECKPOINTS as RF3_CHECKPOINTS
from rf3.checkpoints import RegisteredCheckpoint as RF3Checkpoint
from rfd3.checkpoints import REGISTERED_CHECKPOINTS as RFD3_CHECKPOINTS
from rfd3.checkpoints import RegisteredCheckpoint as RFD3Checkpoint
from rfd3na.checkpoints import REGISTERED_CHECKPOINTS as RFD3NA_CHECKPOINTS
from rfd3na.checkpoints import RegisteredCheckpoint as RFD3NACheckpoint

DEFAULT_CHECKPOINT_DIR = Path.home() / ".foundry" / "checkpoints"


def _normalize_paths(paths: Iterable[Path]) -> list[Path]:
    """Return absolute, deduplicated paths in order."""
    seen = set()
    normalized: List[Path] = []
    for path in paths:
        resolved = path.expanduser().absolute()
        if resolved not in seen:
            normalized.append(resolved)
            seen.add(resolved)
    return normalized


def get_default_checkpoint_dirs() -> list[Path]:
    """Return checkpoint search paths.

    Search FOUNDRY_CHECKPOINT_DIRS first, then ~/.foundry/checkpoints.
    """
    env_dirs = os.environ.get("FOUNDRY_CHECKPOINT_DIRS", "")

    # For backward compatibility, also check FOUNDRY_CHECKPOINTS_DIR
    if not env_dirs:
        env_dirs = os.environ.get("FOUNDRY_CHECKPOINTS_DIR", "")

    extra_dirs: list[Path] = []
    if env_dirs:
        extra_dirs = [Path(p.strip()) for p in env_dirs.split(":") if p.strip()]
    return _normalize_paths([*extra_dirs, DEFAULT_CHECKPOINT_DIR])


def get_default_checkpoint_dir() -> Path:
    """Backward-compatible helper returning the primary checkpoint directory."""
    return get_default_checkpoint_dirs()[0]


def append_checkpoint_to_env(checkpoint_dirs: list[Path]) -> bool:
    dotenv_path = dotenv.find_dotenv()
    if dotenv_path:
        checkpoint_dirs = _normalize_paths(checkpoint_dirs)
        dotenv.set_key(
            dotenv_path=dotenv_path,
            key_to_set="FOUNDRY_CHECKPOINT_DIRS",
            value_to_set=":".join(str(path) for path in checkpoint_dirs),
            export=False,
        )
        return True
    else:
        return False


REGISTERED_CHECKPOINTS: dict[
    str, RF3Checkpoint | RFD3Checkpoint | RFD3NACheckpoint | MPNNCheckpoint
] = {
    **RFD3NA_CHECKPOINTS,
    **RFD3_CHECKPOINTS,
    **RF3_CHECKPOINTS,
    **MPNN_CHECKPOINTS,
}
