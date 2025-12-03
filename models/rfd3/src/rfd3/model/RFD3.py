import os

import hydra
import torch
from omegaconf import DictConfig
from rfd3.model.cfg_utils import (
    strip_f,
)
from rfd3.model.inference_sampler import ConditionalDiffusionSampler
from rfd3.model.layers.encoders import TokenInitializer
from torch import nn

from foundry.utils.ddp import RankedLogger

ranked_logger = RankedLogger(__name__, rank_zero_only=True)


class RFD3(nn.Module):
    """
    Simplified model for generation
    This module level serves to wrap the diffusion module of AF3
    to be roughly equivalent to the AF3 model w/o trunk processing.

    Allows the same sampler to be used
    """

    def __init__(
        self,
        *,
        # Channel dimensions ('global' features)
        c_s: int,
        c_z: int,
        c_atom: int,
        c_atompair: int,
        # Arguments for modules that will be instantiated
        token_initializer: DictConfig | dict,
        diffusion_module: DictConfig | dict,
        inference_sampler: DictConfig | dict,
        **_,
    ):
        super().__init__()
        # Check for chunked P_LL mode via environment variable
        use_chunked_pll = os.environ.get("RFD3_LOW_MEMORY_MODE", None) == "1"
        ranked_logger.info(f"RFD3 initialized with chunked_pll={use_chunked_pll}")

        # Simple constant-feature initializer
        self.token_initializer = TokenInitializer(
            c_s=c_s,
            c_z=c_z,
            c_atom=c_atom,
            c_atompair=c_atompair,
            use_chunked_pll=use_chunked_pll,
            **token_initializer,
        )

        # Diffusion module instantiated to allow for config scripting
        self.diffusion_module = hydra.utils.instantiate(
            diffusion_module, c_atom=c_atom, c_atompair=c_atompair, c_s=c_s, c_z=c_z
        )

        self.use_classifier_free_guidance = (
            inference_sampler["use_classifier_free_guidance"]
            and inference_sampler["cfg_scale"] != 1.0
        )
        self.cfg_features = inference_sampler.pop("cfg_features", [])

        # ... initialize the inference sampler, which performs a full diffusion rollout during inference
        self.inference_sampler = ConditionalDiffusionSampler(**inference_sampler)

    def forward(
        self,
        input: dict,
        coord_atom_lvl_to_be_noised: torch.Tensor = None,
        n_cycle=None,
        **_,
    ) -> dict:
        initializer_outputs = self.token_initializer(input["f"])

        if self.training:
            # Single denoising step
            return self.diffusion_module(
                X_noisy_L=input["X_noisy_L"],
                t=input["t"],
                f=input["f"],
                n_recycle=n_cycle,
                **initializer_outputs,
            )  # [D, L, 3]
        else:
            if self.use_classifier_free_guidance:
                f_ref = strip_f(input["f"], self.cfg_features)
                ref_initializer_outputs = self.token_initializer(f_ref)
            else:
                f_ref = None
                ref_initializer_outputs = None

            return self.inference_sampler.sample_diffusion_like_af3(
                f=input["f"],
                f_ref=f_ref,  # for cfg
                diffusion_module=self.diffusion_module,
                diffusion_batch_size=coord_atom_lvl_to_be_noised.shape[0],
                coord_atom_lvl_to_be_noised=coord_atom_lvl_to_be_noised,
                # Forwarded as **kwargs:
                initializer_outputs=initializer_outputs,
                ref_initializer_outputs=ref_initializer_outputs,  # for cfg
            )
