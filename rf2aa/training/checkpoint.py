import torch
from torch.utils.checkpoint import checkpoint

# for gradient checkpointing
def create_custom_forward(module, **kwargs):
    def custom_forward(*inputs):
        return module(*inputs, **kwargs)
    return custom_forward


def activation_checkpointing(function):
    def wrapper(*args):
        if torch.is_grad_enabled():
            return checkpoint(function, *args, use_reentrant=False)
        return function(*args)
    return wrapper
