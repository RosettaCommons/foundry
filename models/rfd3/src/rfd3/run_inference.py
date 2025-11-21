#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../../../../.ipd/shebang/rf3_exec.sh" "$0" "$@"'

import os

import hydra
import rootutils
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from modelhub.utils.logging import suppress_warnings

# Setup root dir and environment variables (more info: https://github.com/ashleve/rootutils)
# NOTE: Sets the `PROJECT_ROOT` environment variable to the root directory of the project (where `.project-root` is located)
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

load_dotenv(override=True)

# If the user has set `PROJECT_PATH`, use it to build the config path; otherwise, fall back to `PROJECT_ROOT`
_config_path = os.path.join(os.environ["PROJECT_ROOT"], "models/rfd3/configs")


@hydra.main(
    config_path=_config_path,
    config_name="inference",
    version_base="1.3",
)
def run_inference(cfg: DictConfig) -> None:
    """Execute the specified inference pipeline"""

    run_params_set = {"inputs", "n_batches", "out_dir"}
    run_params = {k: v for k, v in cfg.items() if k in run_params_set}

    # Create __init__ args by filtering for all configs not in run_params
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    init_cfg_dict = {k: v for k, v in cfg_dict.items() if k not in run_params_set}
    init_cfg = OmegaConf.create(init_cfg_dict)
    inference_engine = instantiate(init_cfg, _convert_="partial", _recursive_=False)

    # # Run inference
    with suppress_warnings(is_inference=True):
        inference_engine.run(**run_params)


if __name__ == "__main__":
    run_inference()
