import torch
import torch.nn as nn
from typing import Any

from rf2aa.metrics.metrics_base import Metric


class PAE(Metric):
    def __call__(self, rf_output, loss_calc_items) -> float:
        pae = self.pae_unbin(rf_output["pae"])
        return torch.mean(pae)
    
    @staticmethod
    def pae_unbin(logits_pae, bin_step=0.5):
        nbin = logits_pae.shape[1]
        bins = torch.linspace(bin_step*0.5, bin_step*nbin-bin_step*0.5, nbin,
                                dtype=logits_pae.dtype, device=logits_pae.device)
        logits_pae = torch.nn.Softmax(dim=1)(logits_pae)
        return torch.sum(bins[None,:,None,None]*logits_pae, dim=1)

class PLDDT(Metric):
    def __call__(self, rf_output, loss_calc_items) -> float:
        plddt = self.lddt_unbin(rf_output["plddt"])
        return torch.mean(plddt)
    
    @staticmethod
    def lddt_unbin(pred_lddt):
        # calculate lddt prediction loss
        nbin = pred_lddt.shape[1]
        bin_step = 1.0 / nbin
        lddt_bins = torch.linspace(bin_step, 1.0, nbin, dtype=pred_lddt.dtype, device=pred_lddt.device)

        pred_lddt = nn.Softmax(dim=1)(pred_lddt)
        return torch.sum(lddt_bins[None,:,None]*pred_lddt, dim=1)
