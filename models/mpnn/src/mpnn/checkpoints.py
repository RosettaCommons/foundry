"""Management of checkpoints"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

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


@dataclass
class RegisteredCheckpoint:
    url: str
    filename: str
    description: str
    sha256: None = None  # Optional: add checksum for verification

    def get_default_path(self) -> Path:
        checkpoint_dirs = get_default_checkpoint_dirs()
        for checkpoint_dir in checkpoint_dirs:
            candidate = checkpoint_dir / self.filename
            if candidate.exists():
                return candidate
        return checkpoint_dirs[0] / self.filename


REGISTERED_CHECKPOINTS = {
    "proteinmpnn": RegisteredCheckpoint(
        url="https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_020.pt",
        filename="proteinmpnn_v_48_020.pt",
        description="ProteinMPNN checkpoint",
    ),
    "ligandmpnn": RegisteredCheckpoint(
        url="https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_010_25.pt",
        filename="ligandmpnn_v_32_010_25.pt",
        description="LigandMPNN checkpoint",
    ),
    "solublempnn": RegisteredCheckpoint(
        url="https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_020.pt",
        filename="solublempnn_v_48_020.pt",
        description="SolubleMPNN checkpoint",
    ),
}
