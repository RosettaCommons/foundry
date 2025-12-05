"""Management of checkpoints"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import dotenv

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

    Always starts with the default ~/.foundry/checkpoints directory and then
    appends any additional directories from the colon-separated
    FOUNDRY_CHECKPOINT_DIRS environment variable.
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


@dataclass
class RegisteredCheckpoint:
    url: str
    filename: str
    description: str
    sha256: None = None  # Optional: add checksum for verification

    def get_default_path(self):
        checkpoint_dirs = get_default_checkpoint_dirs()
        for checkpoint_dir in checkpoint_dirs:
            candidate = checkpoint_dir / self.filename
            if candidate.exists():
                return candidate
        return checkpoint_dirs[0] / self.filename


REGISTERED_CHECKPOINTS = {
    "rfd3": RegisteredCheckpoint(
        url="https://files.ipd.uw.edu/pub/rfd3/rfd3_foundry_2025_12_01_remapped.ckpt",
        filename="rfd3_latest.ckpt",
        description="RFdiffusion3 checkpoint",
    ),
    "rf3": RegisteredCheckpoint(
        url="https://files.ipd.uw.edu/pub/rf3/rf3_foundry_01_24_latest_remapped.ckpt",
        filename="rf3_foundry_01_24_latest_remapped.ckpt",
        description="latest RF3 checkpoint trained with data until 1/2024 (expect best performance)",
    ),
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
    # Other models
    "rf3_preprint_921": RegisteredCheckpoint(
        url="https://files.ipd.uw.edu/pub/rf3/rf3_foundry_09_21_preprint_remapped.ckpt",
        filename="rf3_foundry_09_21_preprint_remapped.ckpt",
        description="RF3 preprint checkpoint trained with data until 9/2021",
    ),
    "rf3_preprint_124": RegisteredCheckpoint(
        url="https://files.ipd.uw.edu/pub/rf3/rf3_foundry_01_24_preprint_remapped.ckpt",
        filename="rf3_foundry_01_24_preprint_remapped.ckpt",
        description="RF3 preprint checkpoint trained with data until 1/2024",
    ),
    "solublempnn": RegisteredCheckpoint(
        url="https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_020.pt",
        filename="solublempnn_v_48_020.pt",
        description="SolubleMPNN checkpoint",
    ),
}
