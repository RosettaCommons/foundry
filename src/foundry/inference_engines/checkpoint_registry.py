"""Management of checkpoints"""

import os
from dataclasses import dataclass
from pathlib import Path


def get_default_checkpoint_dir() -> Path:
    """Get the default checkpoint directory.

    Priority:
    1. FOUNDRY_CHECKPOINTS_DIR environment variable
    2. ~/.foundry/checkpoints
    """
    if "FOUNDRY_CHECKPOINTS_DIR" in os.environ and os.environ.get(
        "FOUNDRY_CHECKPOINTS_DIR"
    ):
        return Path(os.environ["FOUNDRY_CHECKPOINTS_DIR"]).absolute()
    return Path.home() / ".foundry" / "checkpoints"


@dataclass
class RegisteredCheckpoint:
    url: str
    filename: str
    description: str
    sha256: None = None  # Optional: add checksum for verification

    def get_default_path(self):
        return get_default_checkpoint_dir() / self.filename


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
