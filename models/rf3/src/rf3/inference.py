#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../../../../.ipd/shebang/rf3_exec.sh" "$0" "$@"'

import os

import hydra
import rootutils
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from foundry.utils.logging import suppress_warnings

# Setup root dir and environment variables (more info: https://github.com/ashleve/rootutils)
# NOTE: Sets the `PROJECT_ROOT` environment variable to the root directory of the project (where `.project-root` is located)
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

load_dotenv(override=True)

_config_path = os.path.join(os.environ["PROJECT_ROOT"], "models/rf3/configs")


@hydra.main(
    config_path=_config_path,
    config_name="inference",
    version_base="1.3",
)
def run_inference(cfg: DictConfig) -> None:
    """Execute RF3 inference pipeline."""

    # Extract run() parameters from config
    # Preserve string inputs, convert other sequence-like inputs to a Python list (None -> [])
    inputs_param = cfg.inputs if isinstance(cfg.inputs, str) else list(cfg.inputs or [])

    run_params = {
        "inputs": inputs_param,
        "out_dir": str(cfg.out_dir) if cfg.get("out_dir") else None,
        "dump_predictions": cfg.get("dump_predictions", True),
        "dump_trajectories": cfg.get("dump_trajectories", False),
        "one_model_per_file": cfg.get("one_model_per_file", False),
        "annotate_b_factor_with_plddt": cfg.get("annotate_b_factor_with_plddt", False),
        "sharding_pattern": cfg.get("sharding_pattern", None),
        "skip_existing": cfg.get("skip_existing", False),
        "template_selection": cfg.get("template_selection", None),
        "ground_truth_conformer_selection": cfg.get(
            "ground_truth_conformer_selection", None
        ),
        "cyclic_chains": cfg.get("cyclic_chains", []),
    }

    # Create init config with only __init__ params
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    run_param_keys = set(run_params.keys())
    init_cfg_dict = {k: v for k, v in cfg_dict.items() if k not in run_param_keys}
    init_cfg = OmegaConf.create(init_cfg_dict)

    # Instantiate engine (only __init__ params)
    inference_engine = instantiate(init_cfg, _convert_="partial", _recursive_=False)

    # Run inference
    with suppress_warnings(is_inference=True):
        inference_engine.run(**run_params)


if __name__ == "__main__":
    run_inference()
