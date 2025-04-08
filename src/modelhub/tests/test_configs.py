import os

import pytest
import torch
from hydra import compose, initialize

from modelhub.tests.test_conditions import config_pickle_path, configs


@pytest.mark.parametrize("config_name", configs)
def test_config(config_name):
    # load config
    with initialize(config_path="../config/train"):
        config = compose(config_name)

    config_pickle = config_pickle_path(config_name)
    if os.path.exists(config_pickle):
        config_regression = torch.load(config_pickle, map_location="cpu")
    else:
        torch.save(config, config_pickle)
        config_regression = config
    assert config == config_regression, (
        f"config: {config} != config_regression: {config_regression}"
    )
