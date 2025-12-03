#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../../../../.ipd/shebang/rf3_exec.sh" "$0" "$@"'

import os

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine

load_dotenv(override=True)

# For pip-installed package, configs should be relative to this file
# Adjust this path based on where configs are bundled in the package
_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "configs")


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
    init_cfg_dict = {
        k: v for k, v in cfg_dict.items() if k not in run_params_set | {"_target_"}
    }

    # Run
    engine = RFD3InferenceEngine(**RFD3InferenceConfig(**init_cfg_dict))
    engine.run(**run_params)


if __name__ == "__main__":
    run_inference()
