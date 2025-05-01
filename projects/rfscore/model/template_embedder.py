import torch
from torch import nn
from torch.nn.functional import relu

from modelhub.model.layers.pairformer_layers import PairformerBlock
from modelhub.training.checkpoint import activation_checkpointing
from modelhub.utils.torch_utils import device_of
from projects.rfscore.model.embeddings import FourierEmbedding
from projects.rfscore.transforms.ground_truth_template import (
    af3_noise_scale_to_noise_level,
)


class RFScoreTemplateEmbedder(nn.Module):
    def __init__(
        self,
        n_block,
        raw_template_dim,
        c_z,
        c,
        p_drop,
        use_fourier_encoding=False,
    ):
        super().__init__()
        self.c = c
        self.emb_pair = nn.Linear(c_z, c, bias=False)
        self.norm_pair_before_pairformer = nn.LayerNorm(c_z)
        self.norm_after_pairformer = nn.LayerNorm(c)
        self.emb_templ = nn.Linear(raw_template_dim, c, bias=False)
        if use_fourier_encoding:
            self.emb_noise_level = FourierEmbedding(c)
        self.use_fourier_encoding = use_fourier_encoding

        # template pairformer does not operate on sequence representation
        self.pairformer = nn.ModuleList(
            [
                PairformerBlock(
                    c_s=0,
                    c_z=c,
                    p_drop=p_drop,
                    triangle_multiplication=dict(d_hidden=c),
                    triangle_attention=dict(d_hidden=c),
                    attention_pair_bias={},
                    n_transition=4,
                )
                for _ in range(n_block)
            ]
        )

        # NOTE: this is not consistent with AF3 paper which outputs this tensor in the template_channel dimension
        # In Algorithm 1, line 9, the outputs of this function are added to the Z_II tensor which has dimensions [B, I, I, C_z]
        # so we make the outputs of this module also has those dimensions
        self.agg_emb = nn.Linear(c, c_z, bias=False)

    def forward(
        self,
        f,
        Z_II,
    ):
        @activation_checkpointing
        def embed_templates_like_rfscore(
            has_distogram_condition,  # [I, I]
            distogram_condition_noise_scale,  # [I]
            distogram_condition,  # [I, I, 64], where 64 is the number of distogram bins
        ):
            with torch.amp.autocast(
                device_type=device_of(self).type, enabled=True, dtype=torch.bfloat16
            ):
                I = Z_II.shape[0]  # n_tokens

                # Transform noise scale to reasonable range
                joint_noise_scale = (
                    distogram_condition_noise_scale[None, :] ** 2
                    + distogram_condition_noise_scale[:, None] ** 2
                ).sqrt()
                joint_noise_level = af3_noise_scale_to_noise_level(joint_noise_scale)

                # ---------------------------- #

                if not self.use_fourier_encoding:
                    # OPTION 1: CONCATENATED ENCODING
                    # ... concatenate along the channel dimension
                    template_feats = torch.cat(
                        [
                            distogram_condition,  # [I, I, 64]
                            has_distogram_condition.unsqueeze(-1),  # [I, I, 1]
                            joint_noise_level.unsqueeze(-1),  # [I, I, 1]
                        ],
                        dim=-1,
                    )  # [I, I, 66]

                    # ... remove any invalid interactions
                    template_feats = template_feats * has_distogram_condition.unsqueeze(
                        -1
                    )  # [I, I, 66], where 66 = 64 + 1 + 1

                    # ... embed template features
                    template_channels = self.emb_templ(template_feats)  # [I, I, c]
                else:
                    # OPTION 2: FOURIER ENCODING
                    # ... embed noise scale
                    noise_level_emb = self.emb_noise_level(
                        joint_noise_level.view(-1)
                    ).view(I, I, -1)  # [I, I, c]
                    noise_level_emb = (
                        noise_level_emb * has_distogram_condition.unsqueeze(-1)
                    )

                    # ... embed distogram condition
                    template_channels = self.emb_templ(
                        distogram_condition * has_distogram_condition.unsqueeze(-1)
                    )  # [I, I, c]

                    # ... combine embeddings
                    template_channels = template_channels + noise_level_emb  # [I, I, c]

                # ---------------------------- #

                # ... pass through pairformer
                u_II = torch.zeros(I, I, self.c, device=Z_II.device)
                v_II = (
                    self.emb_pair(self.norm_pair_before_pairformer(Z_II))
                    + template_channels
                )  # [I, I, c]
                for block in self.pairformer:
                    _, v_II = block(None, v_II)
                u_II = u_II + self.norm_after_pairformer(v_II)

            return self.agg_emb(relu(u_II))

        # rfscore template embedding (noisy ground-truth template as input)
        embedded_templates = embed_templates_like_rfscore(
            has_distogram_condition=f["has_distogram_condition"],  # [I, I]
            distogram_condition_noise_scale=f["distogram_condition_noise_scale"],  # [I]
            distogram_condition=f[
                "distogram_condition"
            ],  # [I, I, 64], where 64 is the number of distogram bins
        )

        return embedded_templates
