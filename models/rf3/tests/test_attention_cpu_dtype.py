"""RF3's attention blocks must support CPU inference without autocast."""

import pytest
import torch
from rf3.model.layers.af3_diffusion_transformer import AttentionPairBiasDiffusion
from rf3.model.layers.pairformer_layers import AttentionPairBiasPairformerDeepspeed


@pytest.mark.parametrize("diffusion", [False, True])
def test_float32_cpu_attention(diffusion):
    torch.manual_seed(42)
    if diffusion:
        model = AttentionPairBiasDiffusion(16, 16, 8, 4, False)
        conditioning = torch.randn(1, 4, 16)
        bias = None
    else:
        model = AttentionPairBiasPairformerDeepspeed(16, 16, 8, 4)
        conditioning = None
        bias = torch.zeros(1, 4, 4)
    with torch.no_grad():
        output = model(
            torch.randn(1, 4, 16), conditioning, torch.randn(1, 4, 4, 8), bias
        )
    assert output.shape == (1, 4, 16)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()
