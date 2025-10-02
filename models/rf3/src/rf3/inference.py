#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../../scripts/shebang/modelhub_exec.sh" "$0" "$@"'

import os
import tempfile
from pathlib import Path

import hydra
import rootutils
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig

from modelhub.utils.logging import suppress_warnings

load_dotenv(override=True)

# Setup root dir and environment variables (more info: https://github.com/ashleve/rootutils)
# NOTE: Sets the `PROJECT_ROOT` environment variable to the root directory of the project (where `.project-root` is located)
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Find the RF3 configs directory relative to this file
# This file is at: models/rf3/src/rf3/inference.py
# Configs are at: models/rf3/configs/
rf3_package_dir = Path(__file__).parent.parent.parent  # Go up to models/rf3/
_config_path = str(rf3_package_dir / "configs")


@hydra.main(
    config_path=_config_path,
    config_name="inference",
    version_base="1.3",
)
def run_inference(cfg: DictConfig) -> None:
    """Execute the specified inference pipeline"""

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        inference_engine = instantiate(
            cfg, temp_dir=temp_dir, _convert_="partial", _recursive_=False
        )
        inference_engine.trainer.fabric.launch()
        with suppress_warnings(is_inference=True):
            inference_engine.eval()


if __name__ == "__main__":
    run_inference()
