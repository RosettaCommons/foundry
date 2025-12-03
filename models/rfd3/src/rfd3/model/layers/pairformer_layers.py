import torch
from rfd3.model.layers.layer_utils import (
    MultiDimLinear,
    RMSNorm,
    Transition,
    linearNoBias,
)
from torch import nn

from foundry.training.checkpoint import activation_checkpointing
from foundry.utils.torch import device_of


class AttentionPairBiasPairformerDeepspeed(nn.Module):
    def __init__(self, c_a, c_s, c_pair, n_head, kq_norm=False):
        super().__init__()
        self.n_head = n_head
        self.c_a = c_a
        self.c_pair = c_pair
        self.c = c_a // n_head

        self.to_q = MultiDimLinear(c_a, (n_head, self.c))
        self.to_k = MultiDimLinear(c_a, (n_head, self.c), bias=False, norm=kq_norm)
        self.to_v = MultiDimLinear(c_a, (n_head, self.c), bias=False, norm=kq_norm)
        self.to_b = linearNoBias(c_pair, n_head)
        self.to_g = nn.Sequential(
            MultiDimLinear(c_a, (n_head, self.c), bias=False),
            nn.Sigmoid(),
        )
        self.to_a = linearNoBias(c_a, c_a)
        # self.linear_output_project = nn.Sequential(
        # LinearBiasInit(c_s, c_a, biasinit=-2.),
        # nn.Sigmoid(),
        # )
        self.ln_0 = RMSNorm((c_pair,))
        # self.ada_ln_1 = AdaLN(c_a=c_a, c_s=c_s)
        self.ln_1 = RMSNorm((c_a,))
        self.use_deepspeed_evo = False
        self.force_bfloat16 = True

    def forward(
        self,
        A_I,  # [I, C_a]
        S_I,  # [I, C_a] | None
        Z_II,  # [I, I, C_z]
        Beta_II=None,  # [I, I]
    ):
        # Input projections
        assert S_I is None
        A_I = self.ln_1(A_I)

        if self.use_deepspeed_evo or self.force_bfloat16:
            A_I = A_I.to(torch.bfloat16)

        Q_IH = self.to_q(A_I)  # / np.sqrt(self.c)
        K_IH = self.to_k(A_I)
        V_IH = self.to_v(A_I)
        B_IIH = self.to_b(self.ln_0(Z_II)) + Beta_II[..., None]
        G_IH = self.to_g(A_I)

        B, L = B_IIH.shape[:2]

        if not self.use_deepspeed_evo or L <= 24:
            Q_IH = Q_IH / torch.sqrt(
                torch.tensor(self.c).to(Q_IH.device, torch.bfloat16)
            )
            # Attention
            A_IIH = torch.softmax(
                torch.einsum("...ihd,...jhd->...ijh", Q_IH, K_IH) + B_IIH, dim=-2
            )  # softmax over j
            ## G_IH: [I, H, C]
            ## A_IIH: [I, I, H]
            ## V_IH: [I, H, C]
            A_I = torch.einsum("...ijh,...jhc->...ihc", A_IIH, V_IH)
            A_I = G_IH * A_I  # [B, I, H, C]
            A_I = A_I.flatten(start_dim=-2)  # [B, I, Ca]
        else:
            raise NotImplementedError

        A_I = self.to_a(A_I)

        return A_I


class PairformerBlock(nn.Module):
    """
    Attempt to replicate AF3 architecture from scratch.
    """

    def __init__(
        self,
        c_s,
        c_z,
        attention_pair_bias,
        p_drop=0.1,
        triangle_multiplication=None,
        triangle_attention=None,
        n_transition=4,
        use_deepspeed_evo=True,
        use_triangle_mult=False,
        use_triangle_attn=False,
    ):
        super().__init__()

        # self.drop_row = Dropout(broadcast_dim=-2, p_drop=p_drop)
        # self.drop_col = Dropout(broadcast_dim=-3, p_drop=p_drop)

        self.z_transition = Transition(c=c_z, n=n_transition)

        if c_s > 0:
            self.s_transition = Transition(c=c_s, n=n_transition)

            self.attention_pair_bias = AttentionPairBiasPairformerDeepspeed(
                c_a=c_s, c_s=0, c_pair=c_z, **attention_pair_bias
            )

    @activation_checkpointing
    def forward(self, S_I, Z_II):
        with torch.amp.autocast(
            device_type=device_of(self).type, enabled=True, dtype=torch.bfloat16
        ):
            Z_II = Z_II + self.z_transition(Z_II)
            if S_I is not None:
                S_I = S_I + self.attention_pair_bias(
                    S_I, None, Z_II, Beta_II=torch.tensor([0.0], device=Z_II.device)
                )
                S_I = S_I + self.s_transition(S_I)
        return S_I, Z_II
