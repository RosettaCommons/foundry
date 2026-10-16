import hydra
import torch
import torch.nn as nn
from beartype.typing import Any, cast
from omegaconf import DictConfig


class Loss(nn.Module):
    def __init__(self, **losses: Any) -> None:
        super().__init__()
        self.to_compute = []
        for loss_name, loss in losses.items():
            loss_fn = hydra.utils.instantiate(loss)
            self.to_compute.append(loss_fn)
            assert not isinstance(
                loss_fn, DictConfig
            ), f"Loss {loss_name} was instantiated as a DictConfig. Is _target_ present?."

    def forward(
        self,
        network_input: dict[str, Any],
        network_output: dict[str, Any],
        loss_input: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        loss_dict: dict[str, Any] = {}
        # Start the accumulator as the int 0 (not a 0-d tensor): the first `+=`
        # then adopts the device/dtype of the child losses via scalar promotion.
        # A `torch.zeros(())` here would sit on the CPU and break GPU training on
        # a device mismatch. After the (always non-empty) loop `loss` is a Tensor.
        loss = 0
        for loss_fn in self.to_compute:
            loss_, loss_dict_ = loss_fn(network_input, network_output, loss_input)
            loss += loss_
            loss_dict.update(loss_dict_)
        total_loss = cast(torch.Tensor, loss)
        loss_dict["total_loss"] = total_loss.detach()
        return total_loss, loss_dict
