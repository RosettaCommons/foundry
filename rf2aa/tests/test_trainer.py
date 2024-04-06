import os
import pytest
import torch
from unittest.mock import patch
from hydra import initialize, compose

from rf2aa.tests.test_conditions import configs
from rf2aa.trainer_new import trainer_factory

@pytest.mark.parametrize("config", configs)
def test_load_checkpoint_from_scratch(config):
    with initialize(version_base=None, config_path="../config/train"):
        cfg = compose(config_name=config)
 
    trainer = trainer_factory[cfg.experiment.trainer](config=cfg)
    expected_file_path = f"{trainer.output_dir}/{trainer.config.experiment.name}_last.pt"
    with patch.object(os.path, "exists", return_value=False) as mock_exists:
        loaded_checkpoint = trainer.load_checkpoint(rank=0)
        assert loaded_checkpoint == False
    mock_exists.assert_called_once_with(expected_file_path)

@pytest.mark.parametrize("config", configs)
def test_load_checkpoint_from_last(config):
    cfg_overrides = ["eval_params.checkpoint_path=null"]
    with initialize(version_base=None, config_path="../config/train"):
        cfg = compose(config_name=config, overrides=cfg_overrides)
    print(cfg.eval_params.checkpoint_path)
    trainer = trainer_factory[cfg.experiment.trainer](config=cfg)
    expected_file_path = f"{trainer.output_dir}/{trainer.config.experiment.name}_last.pt"
    with patch.object(os.path, "exists", return_value=True) as mock_exists:
        with patch.object(torch, "load", return_value={}) as mock_load:
            loaded_checkpoint = trainer.load_checkpoint(rank=0)
            assert loaded_checkpoint == True
    mock_exists.assert_called_once_with(expected_file_path)
    mock_load.assert_called_once()

@pytest.mark.parametrize("config", configs)
def test_load_checkpoint_from_eval_params(config):
    cfg_overrides = ["eval_params.checkpoint_path=/my/eval/path"]
    with initialize(version_base=None, config_path="../config/train"):
        cfg = compose(config_name=config, overrides=cfg_overrides)

    trainer = trainer_factory[cfg.experiment.trainer](config=cfg)
    expected_file_path = "/my/eval/path"
    with patch.object(os.path, "exists", return_value=True) as mock_exists:
        with patch.object(torch, "load", return_value={}) as mock_load:
            loaded_checkpoint = trainer.load_checkpoint(rank=0)
            assert loaded_checkpoint == True
    mock_exists.assert_called_once_with(expected_file_path)
    mock_load.assert_called_once()

