import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from opt_einsum import contract as einsum
from rf3.util_module import init_lecun_normal

from foundry import SHOULD_USE_CUEQUIVARIANCE
from foundry.training.checkpoint import activation_checkpointing

if SHOULD_USE_CUEQUIVARIANCE:
    import cuequivariance_torch as cuet


class TriangleAttention(nn.Module):
    """Implementation of Triangle Attention from AlphaFold3.

    Routes to either cuEquivariance or vanilla implementation based on configuration.

    Args:
        pair: Pair representation tensor of shape (B, L, L, d_pair)

    Returns:
        Updated pair representation tensor of shape (B, L, L, d_pair)

    """

    def __init__(
        self,
        d_pair,
        n_head=4,
        d_hidden=32,
        p_drop=0.1,  # noqa: E402
        start_node=True,
        use_cuequivariance=True,
    ):
        super(TriangleAttention, self).__init__()

        self.norm = nn.LayerNorm(d_pair)

        self.to_q = nn.Linear(d_pair, n_head * d_hidden, bias=False)
        self.to_k = nn.Linear(d_pair, n_head * d_hidden, bias=False)
        self.to_v = nn.Linear(d_pair, n_head * d_hidden, bias=False)
        self.to_b = nn.Linear(d_pair, n_head, bias=False)
        self.to_g = nn.Linear(d_pair, n_head * d_hidden)
        self.to_out = nn.Linear(n_head * d_hidden, d_pair)

        self.scaling = 1 / math.sqrt(d_hidden)

        self.h = n_head
        self.dim = d_hidden
        self.start_node = start_node

        self.use_cuequivariance = use_cuequivariance

        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        # bias: normal distribution
        self.to_b = init_lecun_normal(self.to_b)

        # gating: zero weights, one biases (mostly open gate at the begining)
        nn.init.zeros_(self.to_g.weight)
        nn.init.ones_(self.to_g.bias)

        # to_out: right before residual connection: zero initialize -- to make it sure residual operation is same to the Identity at the begining
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    @activation_checkpointing
    def forward(self, pair):
        """Forward pass of triangle attention."""
        pair = self.norm(pair)
        bias = self.to_b(pair)  # (B, L, L, h)

        if not self.start_node:
            pair = rearrange(pair, "b i j d -> b j i d")

        # Route to appropriate implementation
        if self.use_cuequivariance and SHOULD_USE_CUEQUIVARIANCE:
            out = self._forward_cuequivariance(pair, bias)
        else:
            out = self._forward_vanilla(pair, bias)

        if not self.start_node:
            out = rearrange(out, "b i j d -> b j i d")

        # output projection
        out = self.to_out(out)
        return out

    def _forward_cuequivariance(self, pair, bias):
        """cuEquivariance triangle attention implementation."""
        # Handle autocast conversion
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
            pair = pair.to(dtype=dtype)
            bias = bias.to(dtype=dtype)

        assert (
            pair.dtype == torch.bfloat16 and bias.dtype == torch.bfloat16
        ), f"cuEquivariance requires bfloat16 inputs (got pair={pair.dtype}, bias={bias.dtype})"

        # Gate computation
        gate = torch.sigmoid(self.to_g(pair))  # (B, L, L, h*dim)

        # Project and reshape to cuEquivariance format: (B, L, H, L, D)
        query = rearrange(self.to_q(pair), "b i j (h d) -> b i h j d", h=self.h)
        key = rearrange(self.to_k(pair), "b i k (h d) -> b i h k d", h=self.h)
        value = rearrange(self.to_v(pair), "b i k (h d) -> b i h k d", h=self.h)

        # Bias: (B, L, L, H) -> (B, 1, H, L, L)
        bias_cueq = rearrange(bias, "b i j h -> b 1 h i j")

        # Call cuEquivariance triangle attention
        out_cueq = cuet.triangle_attention(
            query, key, value, bias=bias_cueq, scale=self.scaling
        )

        # Reshape back: (B, L, H, L, D) -> (B, L, L, H*D)
        out = rearrange(out_cueq, "b i h j d -> b i j (h d)")
        out = gate * out  # gated attention
        return out

    def _forward_vanilla(self, pair, bias):
        """Vanilla PyTorch triangle attention implementation."""
        B, L = pair.shape[:2]

        # Gate computation
        gate = torch.sigmoid(self.to_g(pair))  # (B, L, L, h*dim)

        # Project and reshape to vanilla format: (B, L, L, H, D)
        query = self.to_q(pair).reshape(B, L, L, self.h, -1)
        key = self.to_k(pair).reshape(B, L, L, self.h, -1)
        value = self.to_v(pair).reshape(B, L, L, self.h, -1)

        query = query * self.scaling

        attn = einsum("bijhd,bikhd->bijkh", query, key)
        attn = attn + bias.unsqueeze(1).expand(-1, L, -1, -1, -1)  # (bijkh)
        attn = F.softmax(attn, dim=-2)

        out = einsum("bijkh,bikhd->bijhd", attn, value).reshape(B, L, L, -1)
        out = gate * out  # gated attention

        return out


class TriangleMultiplication(nn.Module):
    """Implementation of Triangle Multiplicative Update from AlphaFold3.

    Routes to either cuEquivariance or naive implementation based on configuration.

    Args:
        d_pair: Pair representation dimension (must equal d_hidden for cuEquivariance)
        d_hidden: Hidden dimension (must equal d_pair for cuEquivariance)
        direction: "outgoing" or "incoming" triangle multiplication direction
        bias: Whether to use bias in normalization layers
        use_cuequivariance: Whether to use cuEquivariance fused kernel when available
    """

    def __init__(
        self,
        d_pair,
        d_hidden=None,
        direction="outgoing",
        bias=True,
        use_cuequivariance=True,
    ):
        super(TriangleMultiplication, self).__init__()

        # Set d_hidden to d_pair if not specified
        if d_hidden is None:
            d_hidden = d_pair

        self.d_pair = d_pair
        self.d_hidden = d_hidden

        # Validate direction parameter
        if direction not in ["outgoing", "incoming"]:
            raise ValueError(
                f"direction must be 'outgoing' or 'incoming', got '{direction}'"
            )
        self.direction = direction

        self.use_cuequivariance = use_cuequivariance

        if self.use_cuequivariance:
            # cuEquivariance kernel requires d_pair == d_hidden...
            assert (
                d_pair == d_hidden
            ), "cuEquivariance triangle multiplication requires d_pair == d_hidden"
            # ... and d_pair must be a multiple of 32
            assert (
                d_pair % 32 == 0
            ), "cuEquivariance triangle multiplication requires d_pair to be a multiple of 32"

        # Input normalization (optional bias)
        self.norm_in = nn.LayerNorm(d_pair, bias=bias)

        # Input projections: combine left and right projections (2*d_hidden, d_pair) (no bias)
        self.p_in = nn.Linear(d_pair, 2 * d_hidden, bias=False)

        # Input gating: combine left and right gates (2*d_hidden, d_pair) (no bias)
        self.g_in = nn.Linear(d_pair, 2 * d_hidden, bias=False)

        # Output normalization (optional bias)
        self.norm_out = nn.LayerNorm(d_hidden, bias=bias)

        # Output projection (no bias)
        self.p_out = nn.Linear(d_hidden, d_pair, bias=False)

        # Output gating (no bias)
        self.g_out = nn.Linear(d_pair, d_pair, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        # Input projections: lecun normal distribution for regular linear weights
        self.p_in = init_lecun_normal(self.p_in)

        # We use default PyTorch initialization for the other parameters, as in AF-3 they do not specify their
        # weight initialization schemes. Without bias, e.g., the gate initialization from AF-2 is not correct.

    def forward(
        self, pair: Float[torch.Tensor, "B N N D"]
    ) -> Float[torch.Tensor, "B N N D"]:
        """Forward pass of triangle multiplication."""
        # Route to appropriate implementation
        if self.use_cuequivariance and SHOULD_USE_CUEQUIVARIANCE:
            return self._forward_cuequivariance(pair)
        else:
            return self._forward_vanilla(pair)

    def _forward_vanilla(
        self, pair: Float[torch.Tensor, "B N N D"]
    ) -> Float[torch.Tensor, "B N N D"]:
        """Vanilla PyTorch triangle multiplication implementation."""
        B, L = pair.shape[:2]

        # Input normalization
        pair_norm = self.norm_in(pair)

        # Input projections: get combined output and split
        p_combined = self.p_in(pair_norm)  # (B, L, L, 2*d_hidden)
        left = p_combined[..., : self.d_hidden]  # (B, L, L, d_hidden)
        right = p_combined[..., self.d_hidden :]  # (B, L, L, d_hidden)

        # Input gating: get combined output and split
        g_combined = self.g_in(pair_norm)  # (B, L, L, 2*d_hidden)
        left_gate = torch.sigmoid(g_combined[..., : self.d_hidden])
        right_gate = torch.sigmoid(g_combined[..., self.d_hidden :])

        # Apply gating
        left = left_gate * left
        right = right_gate * right

        # Triangle multiplication based on direction
        if self.direction == "outgoing":
            out = torch.einsum("bikd,bjkd->bijd", left, right / float(L))
        else:  # incoming
            out = torch.einsum("bkid,bkjd->bijd", left, right / float(L))

        # Output normalization
        out = self.norm_out(out)

        # Output projection
        out = self.p_out(out)

        # Output gating
        gate = torch.sigmoid(self.g_out(pair_norm))
        out = gate * out

        return out

    def _forward_cuequivariance(
        self, pair: Float[torch.Tensor, "B N N D"]
    ) -> Float[torch.Tensor, "B N N D"]:
        """cuEquivariance triangle multiplication implementation."""
        # Handle autocast conversion
        # (Use bfloat16 for optimal performance)
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
            pair = pair.to(dtype=dtype)

        assert (
            pair.dtype == torch.bfloat16
        ), "cuEquivariance requires bfloat16 inputs for optimal performance"

        output = cuet.triangle_multiplicative_update(
            x=pair,
            direction=self.direction,
            mask=None,
            norm_in_weight=self.norm_in.weight,
            norm_in_bias=self.norm_in.bias,
            p_in_weight=self.p_in.weight,  # (2*d_hidden, d_pair)
            g_in_weight=self.g_in.weight,  # (2*d_hidden, d_pair)
            norm_out_weight=self.norm_out.weight,
            norm_out_bias=self.norm_out.bias,
            p_out_weight=self.p_out.weight,  # (d_pair, d_pair) since d_hidden == d_pair
            g_out_weight=self.g_out.weight,  # (d_pair, d_pair)
            eps=1e-5,
        )

        return output
