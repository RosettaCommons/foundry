import torch
import torch.nn as nn
from rf3.util_module import init_lecun_normal

from foundry.training.checkpoint import activation_checkpointing


class OuterProductMean(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, d_hidden=16, p_drop=0.15):
        super(OuterProductMean, self).__init__()
        self.norm = nn.LayerNorm(d_msa)
        self.proj_left = nn.Linear(d_msa, d_hidden)
        self.proj_right = nn.Linear(d_msa, d_hidden)
        self.proj_out = nn.Linear(d_hidden * d_hidden, d_pair)

        self.reset_parameter()

    def reset_parameter(self):
        # normal initialization
        self.proj_left = init_lecun_normal(self.proj_left)
        self.proj_right = init_lecun_normal(self.proj_right)
        nn.init.zeros_(self.proj_left.bias)
        nn.init.zeros_(self.proj_right.bias)

        # zero initialize output
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, msa):
        B, N, L = msa.shape[:3]
        msa = self.norm(msa)
        left = self.proj_left(msa)
        right = self.proj_right(msa)
        right = right / float(N)
        out = torch.einsum("bsli,bsmj->blmij", left, right).reshape(B, L, L, -1)
        out = self.proj_out(out)

        return out


class OuterProductMean_AF3(nn.Module):
    def __init__(self, c_msa_embed, c_outer_product, c_out):
        super(OuterProductMean_AF3, self).__init__()
        self.norm = nn.LayerNorm(c_msa_embed)
        self.proj_left = nn.Linear(c_msa_embed, c_outer_product)
        self.proj_right = nn.Linear(c_msa_embed, c_outer_product)
        self.proj_out = nn.Linear(c_outer_product * c_outer_product, c_out)

    @activation_checkpointing
    def forward(self, msa):
        B, N, L = msa.shape[:3]
        msa = self.norm(msa)
        left = self.proj_left(msa)
        right = self.proj_right(msa)
        right = right / float(N)
        out = torch.einsum("bsli,bsmj->blmij", left, right).reshape(B, L, L, -1)
        out = self.proj_out(out)

        return out
