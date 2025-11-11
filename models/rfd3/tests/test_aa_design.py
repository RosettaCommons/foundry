#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../../../scripts/shebang/modelhub_exec.sh" "$0" "$@"'
# Usage:
# >>> apptainer exec scripts/shebang/test_modelhub.sif pytest ./rfd3/tests/test_aa_design.py
# To combine both make format and testing together you can use:
# apptainer exec ./scripts/shebang/test_modelhub.sif make format || true && apptainer exec ./scripts/shebang/test_modelhub.sif pytest ./rfd3/tests/test_aa_design.py

import contextlib
import io
import os
import sys

import pytest
from rfd3.testing.testing_utils import (
    PIPES,
    TEST_CFG_INFERENCE,
    TEST_CFG_TRAIN,
    TEST_JSON_DATA,
    build_pipelines,
    instantiate_example,
)

from modelhub.utils.logging import print_config_tree

PATH_TO_REPO_ROOT = os.path.dirname(os.path.abspath(__file__)) + "/../../.."
PATH_TO_SRC = os.path.dirname(os.path.abspath(__file__)) + "/../../../src"
sys.path.append(PATH_TO_REPO_ROOT)
sys.path.append(PATH_TO_SRC)

smoke_test = list(TEST_JSON_DATA.keys())


def test_inference():
    # Silence outputs:

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_config_tree(TEST_CFG_INFERENCE, resolve=True)
        print_config_tree(TEST_CFG_TRAIN, resolve=True)


@pytest.mark.fast
@pytest.mark.parametrize("example_name", smoke_test)
def test_inference_pipeline(example_name):
    is_inference = True
    args = TEST_JSON_DATA[example_name]
    input = instantiate_example(args, is_inference=is_inference)
    PIPES[is_inference](input)
    print("Finished example for inference:", example_name)


@pytest.mark.fast
@pytest.mark.parametrize("example_name", ["af2-122", "1p5d-0"])
def test_training_pipeline(example_name):
    is_inference = False
    args = TEST_JSON_DATA[example_name]
    input = instantiate_example(args, is_inference=is_inference)
    example = PIPES[is_inference](input)
    print(
        "Finished example for training. Sampled mask:",
        example.get("training_condition_name"),
    )

if __name__ == "__main__":
    # pytest.main(sys.argv)
    pytest.main(["-v", __file__ + "::test_train_pipeline[rfd3-NA-full]"])
