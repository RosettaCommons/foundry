import os
from typing import Any

import hydra
import torch
from omegaconf import DictConfig
from rfd3.model import inference_acceleration as accel
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
        **_: Any,
    ):
        super().__init__()
        # Check for chunked P_LL mode via environment variable
        use_chunked_pll = os.environ.get("RFD3_LOW_MEMORY_MODE", None) == "1"
        ranked_logger.info(f"RFD3 initialized with chunked_pll={use_chunked_pll}")

        # Simple constant-feature initializer.
        # `**token_initializer` / `**inference_sampler` below: omegaconf's DictConfig
        # supports the mapping protocol at runtime but its stubs don't satisfy
        # SupportsKeysAndGetItem, so mypy rejects `**(DictConfig | dict)`. Hydra sub-configs.
        self.token_initializer = TokenInitializer(  # type: ignore[arg-type]
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
        self.inference_sampler = ConditionalDiffusionSampler(**inference_sampler)  # type: ignore[arg-type]

    def forward(
        self,
        input: dict,
        coord_atom_lvl_to_be_noised: torch.Tensor | None = None,
        n_cycle: int | None = None,
        **_: Any,
    ) -> dict:
        if input["f"]["atom_to_token_map"].ndim == 2:
            if self.training or coord_atom_lvl_to_be_noised is None:
                raise ValueError("Padded batching requires inference coordinates")
            from rfd3.inference.batching import prepare_attention
            from rfd3.model.batched_sampler import sample
            from rfd3.model.layers.batched import initialize

            sampler = self.inference_sampler.sampler
            dm = self.diffusion_module

            def prepare(f: dict) -> dict:
                return prepare_attention(
                    f,
                    atom_keys=dm.n_attn_keys,
                    atom_neighbors=dm.n_attn_seq_neighbours,
                    token_keys=dm.diffusion_transformer.n_keys,
                    token_neighbors=dm.diffusion_transformer.n_local_tokens,
                )

            f = prepare(input["f"])
            init = initialize(self.token_initializer, f)
            ref, ref_init = None, None
            if self.use_classifier_free_guidance:
                if input.get("f_ref") is None:
                    raise ValueError(
                        "Batched CFG requires schema-cropped reference features"
                    )
                ref = prepare(input["f_ref"])
                ref_init = initialize(self.token_initializer, ref)
            # Eager padded transitions can reuse their BF16 weights for this
            # rollout. Compiled transitions pass weights directly into the graph
            # and do not consult this Python cache.
            with accel.rollout_cache(accel.enabled(dm, coord_atom_lvl_to_be_noised)):
                return sample(
                    sampler,
                    dm,
                    coord_atom_lvl_to_be_noised,
                    f,
                    init,
                    f_ref=ref,
                    ref_initializer=ref_init,
                    noise_source=input.get("noise_source"),
                    capture_trajectories=input.get("capture_trajectories", False),
                )
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
            # Inference always provides the coordinates to be noised.
            assert coord_atom_lvl_to_be_noised is not None
            if self.use_classifier_free_guidance:
                f_ref = strip_f(input["f"], self.cfg_features)
                ref_initializer_outputs = self.token_initializer(f_ref)
            else:
                f_ref = None
                ref_initializer_outputs = None

            constants = [initializer_outputs.get("P_LL")]
            if ref_initializer_outputs is not None:
                constants.append(ref_initializer_outputs.get("P_LL"))
            with accel.rollout_cache(
                accel.enabled(self.diffusion_module, coord_atom_lvl_to_be_noised)
                or getattr(self.diffusion_module, "inference_cuda_graph", False),
                constants=constants,
            ):
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
