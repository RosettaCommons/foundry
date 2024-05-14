import torch
import tree
import json
from icecream import ic


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

def debug_device(rf_inputs):
    for name, tensor in rf_inputs.items():
        if torch.is_tensor(tensor):
            if not tensor.is_cuda():
                print(name)
                print(tensor.device)

def debug_grads(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: {param.grad.norm().item()}")

def debug_nan_params(model):
    for name, param in model.named_parameters():
        print(f"{name}: {torch.sum(param.isnan())}")

def pretty_describe_dict(d):
    mapped = describe_dict(d)
    mapped = tree.map_structure(str, mapped)
    return json.dumps(mapped, indent=4)

def describe_dict(d):
    return tree.map_structure(describe, d)

def describe(t: torch.Tensor):
    out = [f'type:{type(t)}']
    if hasattr(t, 'shape'):
        out.append(f'shape:{str(t.shape)}')
    if hasattr(t, 'dtype'):
        out.append(f'dtype:{str(t.dtype)}')
    if hasattr(t, 'device'):
        out.append(f'device:{str(t.device)}')
    return ' '.join(out)

def safe_shape(t: torch.Tensor):
    if hasattr(t, 'shape'):
        return t.shape
    return None

def log_in_out(f):
    def wrapped(*args, **kwargs):
        o = f(*args, **kwargs)
        ic(
            args, kwargs,
            o
        )
        return o
    return wrapped