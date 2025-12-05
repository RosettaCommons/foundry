import hydra
import torch.nn as nn
from omegaconf import DictConfig


class Loss(nn.Module):
    def __init__(self, **losses):
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
        network_input,
        network_output,
        loss_input,
    ):
        loss_dict = {}
        loss = 0
        for loss_fn in self.to_compute:
            loss_, loss_dict_ = loss_fn(network_input, network_output, loss_input)
            loss += loss_
            loss_dict.update(loss_dict_)
        loss_dict["total_loss"] = loss.detach()
        return loss, loss_dict
