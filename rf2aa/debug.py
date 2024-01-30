import torch


def debug_nans(latent_feats):
    for k, v in latent_feats.items():
        if torch.is_tensor(v):
            print(k)
            print(torch.sum(v.isnan()))

def debug_unused_params(model):
    for name, param in model.named_parameters():
        if param.grad is None:
            print(name)

def debug_used_params(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(name)
