import sys, os, json
import time
import numpy as np
import torch
import torch.nn as nn

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0,script_dir+'/models/fold_and_dock3/')
import util

def get_c6d_dict(logits, grad=True, args=None):
    if grad:
        logits = [logit.float() for logit in logits]
    else:
        logits = [logit.float().detach() for logit in logits] 
    probs = [nn.functional.softmax(l, dim=1) for l in logits]
    dict_pred = {}
    dict_pred['p_dist'] = probs[0].permute([0,2,3,1])
    dict_pred['p_omega'] = probs[1].permute([0,2,3,1])
    dict_pred['p_theta'] = probs[2].permute([0,2,3,1]) 
    dict_pred['p_phi'] = probs[3].permute([0,2,3,1])
    return dict_pred, probs

def calc_entropy_loss(out):
    dict_pred, probs = get_c6d_dict(out['logit_s'], grad=True)
    probs = [dict_pred[key] for key in ['p_dist','p_omega','p_theta','p_phi']]
    
    # exclude last bin, then renormalize
    probs = [prob[...,:-1]/prob[...,:-1].sum(-1,keepdim=True) for prob in probs]

    L_mask = probs[0].shape[1]
    loss_mask = 1-torch.eye(L_mask)[None].to(probs[0].device).float() # (B, L, L)

    def calc_entropy(p, mask, eps=1e-6):
        S_ij = -(p * torch.log(p + eps)).sum(axis=-1)
        S_ave = torch.sum(mask * S_ij, axis=(1,2)) / (torch.sum(mask, axis=(1,2)) + eps)
        return S_ave

    entropy_s = [calc_entropy(prob, loss_mask) for prob in probs]

    loss = torch.stack(entropy_s, dim=0).mean(dim=0)

    return loss


def pae_unbin(logits_pae, bin_step=0.5):
    nbin = logits_pae.shape[1]
    bins = torch.linspace(bin_step*0.5, bin_step*nbin-bin_step*0.5, nbin, dtype=logits_pae.dtype, device=logits_pae.device)
    logits_pae = torch.nn.Softmax(dim=1)(logits_pae)
    return torch.sum(bins[None,:,None,None]*logits_pae, dim=1)

def calc_pae_loss(out, inter=False):
    pae = pae_unbin(out['logit_pae'])
    if inter:
        sm_mask = util.is_atom(out['msa'])[0,0]
        inter_mask = sm_mask[None]*(~sm_mask[:,None]) + (~sm_mask[None])*sm_mask[:,None]
        pae = pae_unbin(out['logit_pae'])
        return (pae*inter_mask).sum(dim=[-1,-2]) / inter_mask.sum(dim=[-1,-2])
    else:
        return pae.mean(dim=[-1,-2])
    return pae


