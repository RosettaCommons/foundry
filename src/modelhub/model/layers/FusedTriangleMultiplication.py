import torch
import torch.nn as nn
from jaxtyping import Float

from modelhub import SHOULD_USE_CUEQUIVARIANCE
from modelhub.util_module import init_lecun_normal

if SHOULD_USE_CUEQUIVARIANCE:
    import cuequivariance_torch as cuet


class FusedTriangleMultiplication(nn.Module):
    """
    Triangle Multiplicative Update with cuEquivariance-compatible parameter structure.

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
        super(FusedTriangleMultiplication, self).__init__()

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
        """Parameter initialization"""

        # Input projections: lecun normal distribution for regular linear weights
        self.p_in = init_lecun_normal(self.p_in)

        # We use default PyTorch initialization for the other parameters, as in AF-3 they do not specify their
        # weight initialization schemes. Without bias, e.g., the gate initialization from AF-2 is not correct.

    def forward(
        self, pair: Float[torch.Tensor, "B N N D"]
    ) -> Float[torch.Tensor, "B N N D"]:
        """Forward pass with automatic fallback from cuEquivariance to naive implementation."""

        if self.use_cuequivariance and torch.cuda.is_available():
            # Cast to bfloat16 for optimal performance
            # TODO: Trace back why we aren't already using bfloat16
            if pair.dtype != torch.bfloat16:
                pair = pair.to(torch.bfloat16)
            try:
                return self._fused_forward(pair)
            except Exception as e:
                print(
                    f"cuEquivariance failed ({e}), falling back to naive implementation"
                )
                return self._naive_forward(pair)

        return self._naive_forward(pair)

    def _naive_forward(
        self, pair: Float[torch.Tensor, "B N N D"]
    ) -> Float[torch.Tensor, "B N N D"]:
        """Naive PyTorch implementation"""
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

    def _fused_forward(
        self, pair: Float[torch.Tensor, "B N N D"]
    ) -> Float[torch.Tensor, "B N N D"]:
        """Fused cuEquivariance implementation."""

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
