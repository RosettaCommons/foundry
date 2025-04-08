from typing import Final

import torch
from torch import nn

PI: Final[float] = torch.acos(torch.zeros(1)).item() * 2


class FourierEmbedding(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.register_buffer("w", torch.zeros(c, dtype=torch.float32))
        self.register_buffer("b", torch.zeros(c, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # super().reset_parameters()
        nn.init.normal_(self.w)
        nn.init.normal_(self.b)

    def forward(
        self,
        t,  # [D]
    ):
        return torch.cos(2 * PI * (t[:, None] * self.w + self.b))
