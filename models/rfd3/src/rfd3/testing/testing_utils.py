import copy
import getpass
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import hydra
import numpy as np
import rootutils
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from lightning.fabric import seed_everything
from omegaconf import DictConfig

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../..")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../src")

import atomworks
from atomworks import parse
from atomworks.io.parser import STANDARD_PARSER_ARGS
from atomworks.io.utils.io_utils import to_cif_file
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from rfd3.inference.input_parsing import (
    DesignInputSpecification,
    create_atom_array_from_design_specification,
)
from rfd3.transforms.pipelines import (
    build_atom14_base_pipeline,
)

from foundry.hydra.resolvers import register_resolvers

register_resolvers()

print("Atomworks version:", atomworks.__version__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Same as train.py
rootutils.setup_root(__file__ + "/../..", indicator=".project-root", pythonpath=True)
_config_path = "pkg://configs"

print("_config_path:", _config_path)
print(os.getcwd())

# Set globals
DEBUG_MODE_USERS = ["jbutch"]
DEBUG_MODE = getpass.getuser() in DEBUG_MODE_USERS
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

BASE_HYDRA_OVERRIDES = ["hydra.job.num=0", "hydra.runtime.output_dir=/dev/null"]

PATH_TO_REPO_ROOT = os.path.dirname(os.path.abspath(__file__)) + "/../../.."


def load_inference_cfg():
    GlobalHydra.instance().clear()  # Clear existing context
    with patch.object(sys, "argv", ["compile_cfg"]):
        initialize(config_path=_config_path, version_base="1.3", job_name="engine_cfg")
        cfg = compose(
            config_name="inference",
            overrides=["ckpt_path=null", "out_dir=null", "inputs=null"]
            + BASE_HYDRA_OVERRIDES,
            return_hydra_config=True,
        )
        HydraConfig.instance().set_config(cfg)
    return cfg


def load_train_or_val_cfg(name="pretrain", is_val_cfg=False):
    GlobalHydra.instance().clear()  # Clear existing context
    with patch.object(sys, "argv", ["compile_cfg"]):
        initialize(config_path=_config_path, version_base="1.3", job_name="trainer_cfg")
        cfg = compose(
            config_name="validate" if is_val_cfg else "train",
            overrides=["experiment={}".format(name)] + BASE_HYDRA_OVERRIDES,
            return_hydra_config=True,
        )
        HydraConfig.instance().set_config(cfg)
    return cfg


TEST_CFG_INFERENCE = load_inference_cfg()
TEST_CFG_TRAIN = load_train_or_val_cfg()


##########################################################################################
# Additional utilities
##########################################################################################

DIRS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../tests"),
    os.path.join(os.path.dirname(os.path.abspath(__file__))),
    TEST_CFG_TRAIN.paths.data.design_benchmark_data_dir,
]


def load_test_json():
    test_files = ["demo.json", "demo_extended.json", "tests.json"]
    test_files += [
        "mcsa_41.json",
        "rfd_unindexed.json",
        "sym_tests.json",
        "brk_regression.json",
    ]
    test_json_data = {}
    for dir in DIRS:
        test_data_dir = Path(dir, "test_data")

        for test_file in test_files:
            test_json = test_data_dir / test_file
            if not test_json.exists():
                print(f"Test json file {test_json} does not exist.")
                continue

            print(f"Loading test json file {test_json}...")
            with open(test_json, "r") as f:
                test_json_data_ = json.load(f)

            print("Found test json data:", test_json_data.keys())
            assert test_json_data_, "Test json data is empty."

            nonexistent_keys = []
            for k, d in test_json_data_.items():
                if "input" in d:
                    file = test_data_dir / d["input"]
                    test_json_data_[k]["input"] = file
                    if not file.exists():
                        nonexistent_keys.append(k)

            if nonexistent_keys:
                test_json_data_ = {
                    k: v
                    for k, v in test_json_data_.items()
                    if k not in nonexistent_keys
                }
                print(f"Test input files {nonexistent_keys} do not exist for test.")
                print("Skipping the examples.")

            test_json_data = test_json_data | test_json_data_
    return test_json_data


TEST_JSON_DATA = load_test_json()
assert TEST_JSON_DATA, "No test json data loaded!"


def filter_inference_args(args):
    return {k: v for k, v in args.items() if k in valid_keys_}


def instantiate_example(args, is_inference=True):
    """Saves and reloads atom array for pipeline forwarding"""
    args = copy.deepcopy(args)

    if is_inference:
        input = DesignInputSpecification.safe_init(**args).to_pipeline_input(
            example_id=args.get("example_id", "example")
        )
    else:
        file = args.get("input")
        if file is None:
            # args = filter_inference_args(args)
            atom_array, spec = create_atom_array_from_design_specification(**args)
            outdir = tempfile.TemporaryDirectory()
            outdir = Path(outdir.name)
            tmpfile = outdir / "tmp.cif"
            to_cif_file(
                atom_array,
                tmpfile,
                id="example",
                extra_fields="all",
            )
            file = tmpfile
        result_dict = parse(
            filename=file,
            build_assembly=("1",),
            **STANDARD_PARSER_ARGS,
        )
        input = {
            "atom_array": result_dict["assemblies"]["1"][0],  # First model
            "chain_info": result_dict["chain_info"],
            "ligand_info": result_dict["ligand_info"],
            "metadata": result_dict["metadata"],
        }
    return input


# Pre-build pipeline | in debug mode, exclude random augmentation so that the noise scales are more clear


def build_pipelines(
    cfg_name="pretrain",
    train_dataset_nested_keys=["pdb", "sub_datasets", "interface"],
    val_dataset_nested_keys=["unconditional"],
    composed_config: DictConfig | None = None,
    standardize_crop_size: bool = True,
    **transform_kwargs,
):
    pipes = {}
    for is_validation in [True, False]:
        if composed_config is None:
            config = load_train_or_val_cfg(name=cfg_name, is_val_cfg=is_validation)
        else:
            config = composed_config
        dataset_config = config["datasets"]["val" if is_validation else "train"]
        for key in (
            val_dataset_nested_keys if is_validation else train_dataset_nested_keys
        ):
            dataset_config = dataset_config[key]
        transforms_config = dataset_config["dataset"]["transform"]

        OmegaConf.set_struct(transforms_config, False)
        transforms_config.return_atom_array = True
        transforms_config.seed = seed
        transforms_config.pop("_target_")

        if standardize_crop_size:
            transforms_config.crop_size = 9999
            transforms_config.max_atoms_in_crop = 99999

        for k, v in transform_kwargs.items():
            transforms_config[k] = v

        pipes |= {
            is_validation: build_atom14_base_pipeline(
                **(transforms_config),
            )
        }

    if DEBUG_MODE:
        from rfd3.testing.debug_utils import to_debug_pipe

        for is_inference in [True, False]:
            pipes[is_inference] = to_debug_pipe(pipes[is_inference])
    return pipes


def get_train_dataloader(cfg_name="pretrain", seed=42):
    print("This function is deprecated!")
    cfg = load_train_or_val_cfg(name=cfg_name, is_val_cfg=False)
    from foundry.utils.datasets import (
        assemble_distributed_loader,
        recursively_instantiate_datasets_and_samplers,
    )

    seed_everything(seed, workers=True, verbose=True)

    # Number of examples per epoch (accross all GPUs)
    # (We must sample this many indices from our sampler)
    n_examples_per_epoch = cfg.trainer.n_examples_per_epoch

    # HACK: I'm getting OOMS using the full parquets, so overwrite with these
    # NOTE: If using this test, check that your desired use-case is represented in these mini-datasets!
    try:
        PATH_TO_DATADIR_ROOT = "/projects/ml/datahub/dfs/af3_splits/2024_12_16/"
        if "interface" in cfg.datasets.train.pdb.sub_datasets:
            path = PATH_TO_DATADIR_ROOT + "/interfaces_df_train.parquet"
            assert os.path.exists(path), f"Path {path} does not exist!"
            cfg.datasets.train.pdb.sub_datasets.interface.dataset.dataset.data = path
        if "pn_unit" in cfg.datasets.train.pdb.sub_datasets:
            path = PATH_TO_DATADIR_ROOT + "/pn_units_df_train.parquet"
            assert os.path.exists(path), f"Path {path} does not exist!"
            cfg.datasets.train.pdb.sub_datasets.pn_unit.dataset.dataset.data = path
    except Exception as e:
        print("Could not overwrite dataset paths for train dataloader:", e)

    # ... build the train dataset
    # assert (
    #     "train" in cfg.datasets and cfg.datasets.train
    # ), "No 'train' dataloader configuration provided! If only performing validation, use `validate.py` instead."
    dataset_and_sampler = recursively_instantiate_datasets_and_samplers(
        cfg.datasets.train
    )

    train_dataset = dataset_and_sampler["dataset"]
    train_sampler = dataset_and_sampler["sampler"]

    train_loader = assemble_distributed_loader(
        dataset=train_dataset,
        sampler=train_sampler,
        rank=0,
        world_size=1,
        n_examples_per_epoch=n_examples_per_epoch,
        loader_cfg=cfg.dataloader["train"],
    )

    return train_loader


def assert_tensor_or_array_equal(actual, expected, error_msg: str):
    """Compare tensors/arrays with dtype-aware tolerances."""
    if torch.is_tensor(actual):
        if actual.dtype == torch.bool or actual.dtype in (torch.int32, torch.int64):
            assert torch.equal(actual, expected), error_msg
        else:
            assert torch.allclose(
                actual, expected, atol=1e-5, rtol=1e-5, equal_nan=True
            ), error_msg
    elif isinstance(actual, np.ndarray):
        if actual.dtype.kind in {"U", "S"}:
            assert np.array_equal(actual, expected), error_msg
        elif actual.dtype == bool or np.issubdtype(actual.dtype, np.integer):
            assert np.array_equal(actual, expected), error_msg
        else:
            assert np.allclose(
                actual, expected, atol=1e-5, rtol=1e-5, equal_nan=True
            ), error_msg
    else:
        assert actual == expected, error_msg


def _assert_tensor_or_array_equal(actual, expected, error_msg: str):
    """Backwards-compatible alias for legacy imports."""
    return assert_tensor_or_array_equal(actual, expected, error_msg)


def build_pipes_constant():
    cfgs = {
        True: copy.deepcopy(
            TEST_CFG_TRAIN.datasets.val.unconditional.dataset.transform
        ),
        False: copy.deepcopy(
            TEST_CFG_TRAIN.datasets.train.pdb.sub_datasets.pn_unit.dataset.transform
        ),
    }
    # Standardize crop size and atom array return
    pipes = {}
    for is_inference, cfg in cfgs.items():
        OmegaConf.set_struct(cfg, False)
        cfg.seed = seed
        cfg.return_atom_array = True
        cfg.crop_size = 9999
        cfg.max_atoms_in_crop = 99999
        pipes[is_inference] = hydra.utils.instantiate(cfg)
        # pipes[is_inference] = build_atom14_base_pipeline(
        #     **cfg,
        # )
    if DEBUG_MODE:
        from rfd3.testing.debug_utils import to_debug_pipe

        for is_inference in [True, False]:
            pipes[is_inference] = to_debug_pipe(pipes[is_inference])
    return pipes


PIPES = build_pipes_constant()
