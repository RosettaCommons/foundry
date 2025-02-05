import torch
import torch.nn as nn

from rf2aa.loss.af3_losses import distogram_loss
from rf2aa.metrics.metrics_base import Metric


class DistogramLoss(Metric):
    def __init__(self):
        super().__init__()
        self.cce_loss = nn.CrossEntropyLoss(reduction="none")

    def __call__(self, network_input, network_output, loss_input):
        pred_distogram = network_output["distogram"]
        X_rep_atoms_I = loss_input["X_rep_atoms_I"]
        crd_mask_rep_atoms_I = loss_input["crd_mask_rep_atoms_I"]
        loss = distogram_loss(
            pred_distogram, X_rep_atoms_I, crd_mask_rep_atoms_I, self.cce_loss
        )
        return {"distogram_loss": loss.detach().item()}


class SaveDistograms(Metric):
    def __call__(self, network_input, network_output, loss_input):
        pred_distogram = network_output["distogram"]
        example_id = loss_input["example_id"]
        torch.save(pred_distogram, f"distograms/{example_id}.pt")
        return {"distogram_saved": True}
