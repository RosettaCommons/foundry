# JBs debugging file, please create your own and go crazy!
import logging
import os
import sys
import time

import hydra
import ipdb  # noqa: F401
import numpy as np
import rootutils
import torch
import tree
from atomworks.ml.utils.token import (
    get_token_starts,
)
from rfd3.testing.testing_utils import (
    TEST_CFG_TRAIN,
    TEST_JSON_DATA,
    build_pipelines,
    instantiate_example,
    load_train_or_val_cfg,
)

from foundry.utils.ddp import set_accelerator_based_on_availability

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Same as train.py
rootutils.setup_root(__file__ + "/../..", indicator=".project-root", pythonpath=True)
_config_path = os.path.join(
    os.environ.get("PROJECT_PATH", os.environ.get("PROJECT_ROOT", "../..")), "configs"
)
print(f"Config path: {_config_path}")
print(f"Project root: {os.environ.get('PROJECT_ROOT', '../..')}")


is_inference = True
args = TEST_JSON_DATA["1qys-1-refactored"]
input = instantiate_example(args, is_inference=is_inference)


TEST_CFG_TRAIN = (
    load_train_or_val_cfg(name=sys.argv[1].split("=")[-1])
    if len(sys.argv) > 1
    else TEST_CFG_TRAIN
)


def forward(example, trainer, model, is_inference=is_inference):
    network_input = trainer._assemble_network_inputs(example)

    # Forward pass
    device = "cuda:0"

    def _inmap(path, x):
        if hasattr(x, "cpu") and path != ("f", "msa_stack"):
            return x.to(device)
        else:
            return x

    network_input = tree.map_structure_with_path(_inmap, network_input)
    model.eval() if is_inference else model.train()
    if not is_inference:
        network_output = model.forward(
            input=network_input,
            n_cycle=1,
            coord_atom_lvl_to_be_noised=example["coord_atom_lvl_to_be_noised"].to(
                device
            ),
        )
    else:
        with torch.no_grad():
            network_output = model.forward(
                input=network_input,
                n_cycle=1,
                coord_atom_lvl_to_be_noised=example["coord_atom_lvl_to_be_noised"].to(
                    device
                ),
            )
    return network_output


def prep_forward(cfg):
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        loggers=None,
        callbacks=None,
        _convert_="partial",
        _recursive_=False,
    )
    set_accelerator_based_on_availability(cfg)
    trainer.initialize_or_update_trainer_state({"train_cfg": cfg})
    cfg.trainer.devices_per_node = 1
    cfg.trainer.num_nodes = 1
    try:
        trainer.fabric.launch()
    except Exception as e:
        print(f"Error: {e}")
        print("Switching port")
        os.environ["MASTER_PORT"] = str(1024 + np.random.randint(64512))
        trainer.fabric.launch()
    trainer.construct_model()
    model = trainer.state["model"]
    return model, trainer


def test_conditional_forward():
    unindexed_cfg = load_train_or_val_cfg("test-unindexed")
    unindexed_cfg.datasets.global_transform_args.train_conditions.island.frequency = (
        1e10
    )
    unindexed_cfg.datasets.global_transform_args.train_conditions.island.p_unindex_motif_tokens = 1.0
    unindexed_pipes = build_pipelines(composed_config=unindexed_cfg)

    t0 = time.time()
    example = unindexed_pipes[is_inference](input)
    example["example_id"] = "debug_example"
    print(f"Time taken to process example: {time.time() - t0}")

    aa = example["atom_array"]
    t_aa = aa[get_token_starts(aa)]  # noqa: F841

    from rfd3.testing.debug_utils import pipe_out_to_file

    pipe_out_to_file(example, save=True)

    print("Preparing model")
    model, trainer = prep_forward(TEST_CFG_TRAIN)
    if is_inference:
        model.eval()
        trainer.state["model"].eval()
    network_output = forward(example, trainer, model, is_inference=is_inference)  # noqa: F841


if __name__ == "__main__":
    test_conditional_forward()
    print("Finished main")
