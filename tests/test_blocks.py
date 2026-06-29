"""Unit tests for foundry.model.layers.blocks.

``FourierEmbedding`` and ``Dropout`` are small CPU nn.Modules. Their
deterministic, reachable behaviours are pinned here: the Fourier features are
cosines (so bounded to [-1, 1]) of the right shape, and ``Dropout`` is an
identity in eval mode, scales surviving entries by ``1/(1-p)``, and (with a
``broadcast_dim``) applies one mask value across that whole dimension — i.e.
drops entire rows/columns rather than individual entries.
"""

import torch

from foundry.model.layers.blocks import Dropout, FourierEmbedding


def test_fourier_embedding_shape_and_cosine_range():
    embed = FourierEmbedding(c=8)
    t = torch.arange(5, dtype=torch.float32)

    out = embed(t)

    assert out.shape == (5, 8)
    assert torch.all(out <= 1.0) and torch.all(out >= -1.0)


def test_dropout_is_identity_in_eval_mode():
    dropout = Dropout(p_drop=0.5)
    dropout.eval()
    x = torch.randn(4, 6)

    assert dropout(x) is x


def test_dropout_scales_survivors_by_keep_probability():
    # p_drop=0 -> Bernoulli(1.0) always keeps, so output equals input exactly.
    dropout = Dropout(p_drop=0.0)
    dropout.train()
    x = torch.randn(3, 5)

    assert torch.allclose(dropout(x), x)


def test_dropout_broadcasts_one_mask_value_across_the_dimension():
    torch.manual_seed(0)
    dropout = Dropout(broadcast_dim=1, p_drop=0.5)
    dropout.train()
    x = torch.ones(2, 3, 4)

    out = dropout(x)

    # Each surviving entry is scaled by 1/(1-0.5) = 2; dropped entries are 0.
    assert torch.all((out == 0.0) | (out == 2.0))
    # broadcast_dim=1 means the same mask is applied across that dim: every
    # slice along dim 1 is identical.
    assert torch.equal(out[:, 0, :], out[:, 1, :])
    assert torch.equal(out[:, 1, :], out[:, 2, :])


if __name__ == "__main__":
    import pytest

    pytest.main(["-v", __file__])
