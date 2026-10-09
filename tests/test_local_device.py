"""An explicit local device must win even when accelerator hardware is present."""

import pytest
import torch
from omegaconf import OmegaConf

from foundry.utils.ddp import set_accelerator_based_on_availability


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_explicit_local_device(monkeypatch, device):
    monkeypatch.setenv("FOUNDRY_DEVICE", device)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    cfg = OmegaConf.create(
        {
            "trainer": {
                "accelerator": "gpu",
                "precision": "bf16-mixed",
                "devices_per_node": 8,
                "num_nodes": 2,
            }
        }
    )
    result = set_accelerator_based_on_availability(cfg)
    assert result.trainer.accelerator == device
    assert result.trainer.precision == "32-true"
    assert result.trainer.devices_per_node == result.trainer.num_nodes == 1


def test_mps_request_does_not_silently_fall_back(monkeypatch):
    monkeypatch.setenv("FOUNDRY_DEVICE", "mps")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    cfg = OmegaConf.create(
        {
            "trainer": {
                "accelerator": "auto",
                "devices_per_node": 1,
                "num_nodes": 1,
            }
        }
    )
    with pytest.raises(RuntimeError, match="MPS is unavailable"):
        set_accelerator_based_on_availability(cfg)
