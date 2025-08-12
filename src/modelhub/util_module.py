import numpy as np
import torch
import torch.nn as nn


def init_lecun_normal(module, scale=1.0):
    def truncated_normal(uniform, mu=0.0, sigma=1.0, a=-2, b=2):
        normal = torch.distributions.normal.Normal(0, 1)

        alpha = (a - mu) / sigma
        beta = (b - mu) / sigma

        alpha_normal_cdf = normal.cdf(torch.tensor(alpha))
        p = (
            alpha_normal_cdf
            + (normal.cdf(torch.tensor(beta)) - alpha_normal_cdf) * uniform
        )

        v = torch.clamp(2 * p - 1, -1 + 1e-8, 1 - 1e-8)
        x = mu + sigma * np.sqrt(2) * torch.erfinv(v)
        x = torch.clamp(x, a, b)

        return x

    def sample_truncated_normal(shape, scale=1.0):
        stddev = np.sqrt(scale / shape[-1]) / 0.87962566103423978  # shape[-1] = fan_in
        return stddev * truncated_normal(torch.rand(shape))

    module.weight = torch.nn.Parameter((sample_truncated_normal(module.weight.shape)))
    return module


# for gradient checkpointing
def create_custom_forward(module, **kwargs):
    def custom_forward(*inputs):
        return module(*inputs, **kwargs)

    return custom_forward


class Dropout(nn.Module):
    # Dropout entire row or column
    def __init__(self, broadcast_dim=None, p_drop=0.15):
        super(Dropout, self).__init__()
        # give ones with probability of 1-p_drop / zeros with p_drop
        self.sampler = torch.distributions.bernoulli.Bernoulli(
            torch.tensor([1 - p_drop])
        )
        self.broadcast_dim = broadcast_dim
        self.p_drop = p_drop

    def forward(self, x):
        if not self.training:  # no drophead during evaluation mode
            return x
        shape = list(x.shape)
        if self.broadcast_dim is not None:
            shape[self.broadcast_dim] = 1
        mask = self.sampler.sample(shape).to(x.device).view(shape)

        x = mask * x / (1.0 - self.p_drop)
        return x


def rbf(D, D_min=0.0, D_count=64, D_sigma=0.5):
    # Distance radial basis function
    D_max = D_min + (D_count - 1) * D_sigma
    D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)
    D_mu = D_mu[None, :]
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
    return RBF
