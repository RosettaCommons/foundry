from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import silu

from foundry.training.checkpoint import activation_checkpointing

linearNoBias = partial(torch.nn.Linear, bias=False)


def collapse(x, L):
    return x.reshape((L, x.numel() // L))


class MultiDimLinear(nn.Linear):
    def __init__(self, in_features, out_shape, **kwargs):
        self.out_shape = out_shape
        out_features = np.prod(out_shape)
        super().__init__(in_features, out_features, **kwargs)
        self.reset_parameters()

    def reset_parameters(self, **kwargs) -> None:
        super().reset_parameters()
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        out = super().forward(x)
        return out.reshape(x.shape[:-1] + self.out_shape)


class LinearBiasInit(nn.Linear):
    def __init__(self, *args, biasinit, **kwargs):
        assert biasinit == -2.0  # Sanity check
        self.biasinit = biasinit
        super().__init__(*args, **kwargs)

    def reset_parameters(self) -> None:
        super().reset_parameters()
        self.bias.data.fill_(self.biasinit)


class Transition(nn.Module):
    def __init__(self, n, c):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(c)
        self.linear_1 = linearNoBias(c, n * c)
        self.linear_2 = linearNoBias(c, n * c)
        self.linear_3 = linearNoBias(n * c, c)

    @activation_checkpointing
    def forward(
        self,
        X,
    ):
        X = self.layer_norm_1(X)
        A = self.linear_1(X)
        B = self.linear_2(X)
        X = self.linear_3(silu(A) * B)
        return X


class AdaLN(nn.Module):
    def __init__(self, c_a, c_s, n=2):
        super().__init__()
        self.ln_a = nn.LayerNorm(normalized_shape=(c_a,), elementwise_affine=False)
        self.ln_s = nn.LayerNorm(normalized_shape=(c_s,), bias=False)
        self.to_gain = nn.Sequential(
            nn.Linear(c_s, c_a),
            nn.Sigmoid(),
        )
        self.to_bias = linearNoBias(c_s, c_a)

    def forward(
        self,
        Ai,  # [B, I, C_a]
        Si,  # [B, I, C_s]
    ):
        """
        Output:
            [B, I, C_a]
        """
        Ai = self.ln_a(Ai)
        Si = self.ln_s(Si)
        return self.to_gain(Si) * Ai + self.to_bias(Si)


def create_batch_dimension_if_not_present(batched_n_dim):
    """
    Decorator for adapting a function which expects batched arguments with ndim `batched_n_dim` also
    accept unbatched arguments.
    """

    def wrap(f):
        def _wrap(arg):
            inserted_batch_dim = False
            if arg.ndim == batched_n_dim - 1:
                arg = arg[None]
                inserted_batch_dim = True
            elif arg.ndim == batched_n_dim:
                pass
            else:
                raise Exception(
                    f"arg must have {batched_n_dim - 1} or {batched_n_dim} dimensions, got shape {arg.shape=}"
                )
            o = f(arg)

            if inserted_batch_dim:
                assert o.shape[0] == 1, f"{o.shape=}[0] != 1"
                return o[0]
            return o

        return _wrap

    return wrap


def unpack_args_for_checkpointing(arg_names):
    def wrap(f):
        def _wrap(*args):
            f = args[0]
            return f(**dict(zip(arg_names, args)))

        return _wrap

    return wrap
