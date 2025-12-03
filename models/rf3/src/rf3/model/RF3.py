from collections import deque
from contextlib import ExitStack

import torch
import torch.utils.checkpoint as checkpoint
from beartype.typing import Any, Generator, Protocol
from omegaconf import DictConfig
from rf3.diffusion_samplers.inference_sampler import (
    SampleDiffusion,
    SamplePartialDiffusion,
)
from rf3.model.layers.pairformer_layers import (
    FeatureInitializer,
)
from rf3.model.RF3_structure import DiffusionModule, DistogramHead, Recycler
from torch import nn

from foundry.training.checkpoint import create_custom_forward

"""
Shape Annotation Glossary:
    I: # tokens (coarse representation)
    L: # atoms   (fine representation)
    M: # msa
    T: # templates
    D: # diffusion structure batch dim

    C_s: # Token-level single reprentation channel dimension
    C_z: # Token-level pair reprentation channel dimension
    C_atom: # Atom-level single reprentation channel dimension
    C_atompair: # Atom-level pair reprentation channel dimension

Tensor Name Glossary:
    S: Token-level single representation (I, C_s)
    Z: Token-level pair representation (I, I, C_z)
    Q: Atom-level single representation (L, C_atom)
    P: Atom-level pair representation (L, L, C_atompair)
"""


class ShouldEarlyStopFn(Protocol):
    def __call__(
        self, confidence_outputs: dict[str, Any], first_recycle_outputs: dict[str, Any]
    ) -> tuple[bool, dict[str, Any]]:
        """Duck-typed function Protocol for early stopping based on confidence outputs.

        Returns:
            tuple: A pair containing:
                - should_stop (bool): Whether to stop early.
                - additional_data (dict): Metadata for the user, if any
        """
        ...


class RF3(nn.Module):
    """RF3 Network module.

    We adhere to the PyTorch Lightning Style Guide; see (1).

    References:
        (1) PyTorch Lightning Style Guide: https://lightning.ai/docs/pytorch/latest/starter/style_guide.html
    """

    def __init__(
        self,
        *,
        # Arguments for modules that will be instantiated
        feature_initializer: DictConfig | dict,
        recycler: DictConfig | dict,
        diffusion_module: DictConfig | dict,
        distogram_head: DictConfig | dict,
        inference_sampler: DictConfig | dict,
        # Channel dimensions
        c_s: int,  # AF-3: 384,
        c_z: int,  # AF-3: 128,
        c_atom: int,  # AF-3: 128,
        c_atompair: int,  # AF-3: 16,
        c_s_inputs: int,  # AF-3: 449,
    ):
        """Initializes the AF3 model.

        Args:
            feature_initializer: Arguments for FeatureInitializer
            recycler: Arguments for Recycler
            diffusion_module: Arguments for DiffusionModule
            distogram_head: Arguments for DistogramHead
            inference_sampler: Arguments for the SampleDiffusion class, used for inference (contains no trainable parameters)
            c_s: Token-level single reprentation channel dimension
            c_z: Token-level pair reprentation channel dimension
            c_atom: Atom-level single reprentation channel dimension
            c_atompair: Atom-level pair reprentation channel dimension
            c_s_inputs: Output dimension of the InputFeatureEmbedder
        """
        super().__init__()

        # ... initialize the FeatureInitializer, which creates the initial token-level representations and conditioning
        self.feature_initializer = FeatureInitializer(
            c_s=c_s,
            c_z=c_z,
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_s_inputs=c_s_inputs,
            **feature_initializer,
        )

        # ... initialize the Recycler, which runs the trunk repeatedly with shared weights
        self.recycler = Recycler(c_s=c_s, c_z=c_z, **recycler)
        self.diffusion_module = DiffusionModule(
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_s=c_s,
            c_z=c_z,
            **diffusion_module,
        )
        self.distogram_head = DistogramHead(c_z=c_z, **distogram_head)

        # ... initialize the inference sampler, which performs a full diffusion rollout during inference
        self.inference_sampler = (
            SampleDiffusion(**inference_sampler)
            if not inference_sampler.get("partial_t", False)
            else SamplePartialDiffusion(**inference_sampler)
        )

    def forward(
        self,
        input: dict,
        n_cycle: int,
        coord_atom_lvl_to_be_noised: torch.Tensor = None,
    ) -> dict:
        """Complete forward pass of the model.

        Runs recycling with gradients only on final recycle.

        Args:
            input (dict): Dictionary of model inputs
            n_cycle (int): Number of recycling cycles for the trunk
            coord_atom_lvl_to_be_noised (torch.Tensor): Atom-level coordinates to be noised further. Optional;
                only used during inference for partial denoising.

        Returns:
            dict: Dictionary of model outputs, including:
                - X_L: Predicted atomic coordinates [D, L, 3]
                - distogram: Predicted distogram [I, I, C], where C is the number of bins in the distogram
                - If not training, additional lists are returned, each of length T:
                    * X_noisy_L_traj: List of noisy atomic coordinates at each timestep [D, L, 3]
                    * X_denoised_L_traj: List of denoised atomic coordinates at each timestep [D, L, 3]
                    * t_hats: List of tensor scalars representing the noise schedule at each timestep
        """
        # Cast features to lower precision if autocast is enabled
        if torch.is_autocast_enabled():
            autocast_dtype = torch.get_autocast_dtype("cuda")
            for x in [
                "msa_stack",
                "profile",
                "template_distogram",
                "template_restype",
                "template_unit_vector",
            ]:
                if x in input["f"]:
                    input["f"][x] = input["f"][x].to(autocast_dtype)

        # ... recycling
        # Gives dictionary of outputs S_inputs_I, S_init_I, Z_init_II, S_I, Z_II
        # (We use `deque` with maxlen=1 to ensure that we only keep the last output in memory)
        try:
            recycling_outputs = deque(
                self.trunk_forward_with_recycling(f=input["f"], n_recycles=n_cycle),
                maxlen=1,
            ).pop()
        except IndexError:
            # Handle the case where the generator is empty
            raise RuntimeError("Recycling generator produced no outputs")

        # Predict the distogram from the pair representation
        distogram_pred = self.distogram_head(recycling_outputs["Z_II"])

        # ... post-recycling (diffusion module)
        if self.training:
            # Single denoising step
            X_pred = self.diffusion_module(
                X_noisy_L=input["X_noisy_L"],
                t=input["t"],
                f=input["f"],
                S_inputs_I=recycling_outputs["S_inputs_I"],
                S_trunk_I=recycling_outputs["S_I"],
                Z_trunk_II=recycling_outputs["Z_II"],
            )  # [D, L, 3]
            return dict(
                X_L=X_pred,
                distogram=distogram_pred,
            )
        else:
            # Full diffusion rollout (no gradients, or will OOM)
            sample_diffusion_outs = self.inference_sampler.sample_diffusion_like_af3(
                f=input["f"],
                S_inputs_I=recycling_outputs["S_inputs_I"],
                S_trunk_I=recycling_outputs["S_I"],
                Z_trunk_II=recycling_outputs["Z_II"],
                diffusion_module=self.diffusion_module,
                diffusion_batch_size=input["t"].shape[0],
                coord_atom_lvl_to_be_noised=coord_atom_lvl_to_be_noised,
            )
            return dict(
                X_L=sample_diffusion_outs["X_L"],
                distogram=distogram_pred,
                # For reporting, inference (validation or testing) only
                X_noisy_L_traj=sample_diffusion_outs["X_noisy_L_traj"],
                X_denoised_L_traj=sample_diffusion_outs["X_denoised_L_traj"],
                t_hats=sample_diffusion_outs["t_hats"],
            )

    def trunk_forward_with_recycling(
        self, f: dict, n_recycles: int
    ) -> Generator[dict[str, torch.Tensor]]:
        """Forward pass of the AF-3 trunk.

        (e.g., the recycling process, including the MSAModule, PairfomerStack, etc.).

        Notes:
            - We run with gradients ONLY on the final recycle
            - All recycles use shared weights (ResNet-style)
            - We yield results after reach recycle to support use cases such as e.g., early stopping during inference

        Args:
            f: Feature dictionary
            n_recycles: Number of recycles to run

        Returns:
            dict: Recycling outputs, with keys:
                - S_inputs_I: Token-level single representation input, prior to AtomAttention [I, c_s_inputs]
                - S_init_I: Token-level single representation initialization [I, c_s], after AtomAttention but before recycling stack
                - Z_init_II: Token-level pair representation initialization [I, I, c_z], after AtomAttention but before recycling stack
                - S_I: Token-level single representation [I, c_s], after recycling stack
                - Z_II: Token-level pair representation [I, I, c_z], after recycling stack
        """
        # ... initialize the recycling process (feature initialization)
        # Gives S_inputs_I, S_init_I, Z_init_II, S_I, Z_II
        initialized_features = self.pre_recycle(f)

        # ... collect the recycling inputs, which will be updated in place
        recycling_inputs = {**initialized_features, "f": f}

        for i_cycle in range(n_recycles):
            with ExitStack() as stack:
                # For the first n_recycles - 1 cycles (all but the last recycle), we run without gradients
                if i_cycle < n_recycles - 1:
                    stack.enter_context(torch.no_grad())

                # Clear the autocast cache if gradients are enabled (workaround for autocast bug)
                # See: https://github.com/pytorch/pytorch/issues/65766
                if torch.is_grad_enabled():
                    torch.clear_autocast_cache()

                # Select the MSA for the current recycle (we sample an i.i.d. MSA for each recycle)
                recycling_inputs["f"]["msa"] = f["msa_stack"][i_cycle]

                # Run the model trunk (MSAModule, PairformerStack, etc.)
                # We alter the S_I and Z_II in place such that the next iteration uses the updated values
                recycling_inputs = self.recycle(**recycling_inputs)

                # Yield after each recycle
                yield {
                    "S_inputs_I": recycling_inputs["S_inputs_I"],
                    "S_init_I": recycling_inputs["S_init_I"],
                    "Z_init_II": recycling_inputs["Z_init_II"],
                    "S_I": recycling_inputs["S_I"],
                    "Z_II": recycling_inputs["Z_II"],
                }

    def pre_recycle(self, f: dict) -> dict:
        """Prepare feature inputs for recycling.

        Includes:
            - Feature initialization (S_inputs_I, S_init_I, Z_init_II)
            - Initializing S_I and Z_II to zeros

        Returns:
            dict: Dictionary of recycling inputs, including:
                - S_inputs_I: Token-level single representation input (prior to AtomAttention) [I, c_s_inputs]
                - S_init_I: Token-level single representation initialization [I, c_s] (after round of AtomAttention)
                - Z_init_II: Token-level pair representation initialization [I, I, c_z] (after round of AtomAttention)
                - S_I: Token-level single representation [I, c_s], initialized to zeros
                - Z_II: Token-level pair representation [I, I, c_z], initialized to zeros
        """
        S_inputs_I, S_init_I, Z_init_II = self.feature_initializer(f)
        S_I = torch.zeros_like(S_init_I)
        Z_II = torch.zeros_like(Z_init_II)

        return dict(
            S_inputs_I=S_inputs_I,
            S_init_I=S_init_I,
            Z_init_II=Z_init_II,
            S_I=S_I,
            Z_II=Z_II,
        )

    def recycle(
        self,
        # TODO: Jax typing
        S_inputs_I,
        S_init_I,
        Z_init_II,
        S_I,
        Z_II,
        f,
    ):
        S_I, Z_II = self.recycler(
            f=f,
            S_inputs_I=S_inputs_I,
            S_init_I=S_init_I,
            Z_init_II=Z_init_II,
            S_I=S_I,
            Z_II=Z_II,
        )
        return dict(
            S_inputs_I=S_inputs_I,
            S_init_I=S_init_I,
            Z_init_II=Z_init_II,
            S_I=S_I,
            Z_II=Z_II,
            f=f,
        )


class RF3WithConfidence(RF3):
    """Model for training and inference with confidence metric computation"""

    def __init__(
        self,
        confidence_head: DictConfig | dict,
        mini_rollout_sampler: DictConfig | dict,
        **kwargs,
    ):
        """
        Args:
            (... all arguments from the AF3 class)
            confidence_head: Hydra configuration for the confidence head architecture
            mini_rollout_sampler: Hydra configuration for the mini-rollout sampler (e.g., SampleDiffusion with 20 rather than
                200 timesteps. Note that the `inference_sampler` argument in the AF3 class will still be used for full
                rollouts during inference)
        """
        # (Lazy import)
        from rf3.model.layers.af3_auxiliary_heads import ConfidenceHead  # noqa

        super().__init__(**kwargs)

        self.confidence_head = ConfidenceHead(**confidence_head)
        self.mini_rollout_sampler = SampleDiffusion(**mini_rollout_sampler)

    def forward(
        self,
        input: dict,
        n_cycle: int,
        coord_atom_lvl_to_be_noised: torch.Tensor | None = None,
        should_early_stop_fn: ShouldEarlyStopFn | None = None,
    ) -> dict:
        """Complete forward pass of the model with confidence head.

        Notes:
            - Performs a mini-rollout without gradients during training (e.g., 20 timesteps) and a full rollout (e.g., 200 timesteps) during inference
            - Runs the trunk forward without gradients to conserve memory (which departs from the AF-3 implementation)
            - Runs the forward pass (with gradients) for the confidence model

        Args:
            input (dict): Dictionary of model inputs. In addition to the standard AF-3 model inputs, we expect:
                - rep_atom_idxs: TBD
                - frame_atom_idxs: TBD
            n_cycle (int): Number of recycling cycles for the trunk
            coord_atom_lvl_to_be_noised (torch.Tensor): Atom-level coordinates to be noised further. Optional;
                only used during inference for partial denoising.
            should_early_stop_fn(Callable): Function that takes the confidence and trunk outputs after the first recycle and returns a boolean
                indicating whether to stop early and a dictionary with additional information (e.g., value and threshold).
                If None, no early stopping is performed. Optional; only used during inference.

        Returns:
            dict: Dictionary of model outputs, including:
                - X_L: Predicted atomic coordinates [D, L, 3] (from the mini rollout during training or full rollout during inference)
                - plddt: TBD
                - pae: TBD
                - pde: TBD
                - exp_resolved: TBD
        """
        # Cast features to lower precision if autocast is enabled
        if torch.is_autocast_enabled():
            autocast_dtype = torch.get_autocast_dtype("cuda")
            for x in [
                "msa_stack",
                "profile",
                "template_distogram",
                "template_restype",
                "template_unit_vector",
            ]:
                if x in input["f"]:
                    input["f"][x] = input["f"][x].to(autocast_dtype)

        diffusion_batch_size = input["t"].shape[0]
        with torch.no_grad():
            # ... recycling
            recycling_output_generator = self.trunk_forward_with_recycling(
                f=input["f"], n_recycles=n_cycle
            )
            if should_early_stop_fn:
                assert (
                    not self.training
                ), "Early stopping is not supported during training!"
                # ... get the recycling outputs after the first recycle
                first_recycle_outputs = next(recycling_output_generator)

                # ... compute confidence metrics (without structure)
                confidence_outputs = checkpoint.checkpoint(
                    create_custom_forward(
                        self.confidence_head, frame_atom_idxs=input["frame_atom_idxs"]
                    ),
                    first_recycle_outputs["S_inputs_I"],
                    first_recycle_outputs["S_I"],
                    first_recycle_outputs["Z_II"],
                    None,  # Omit structure
                    input["seq"],
                    input["rep_atom_idxs"],
                    use_reentrant=False,
                )

                should_early_stop, early_stop_data = should_early_stop_fn(
                    confidence_outputs=confidence_outputs,
                    first_recycle_outputs=first_recycle_outputs,
                )
                if should_early_stop:
                    result = {"early_stopped": True}
                    return result | early_stop_data

            # (We use `deque` with maxlen=1 to ensure that we only keep the last output in memory)
            try:
                recycling_outputs = deque(recycling_output_generator, maxlen=1).pop()
            except IndexError:
                # Handle the case where the generator is empty
                raise RuntimeError("Recycling generator produced no outputs")

            # Predict the distogram from the pair representation
            # (NOTE: Not necessary for confidence head training, but helpful for reporting)
            distogram_pred = self.distogram_head(recycling_outputs["Z_II"])

            # ... post-recycling (diffusion module)
            if self.training:
                # Mini-rollout (no gradients still)
                sample_diffusion_outs = (
                    self.mini_rollout_sampler.sample_diffusion_like_af3(
                        f=input["f"],
                        S_inputs_I=recycling_outputs["S_inputs_I"],
                        S_trunk_I=recycling_outputs["S_I"],
                        Z_trunk_II=recycling_outputs["Z_II"],
                        diffusion_module=self.diffusion_module,
                        diffusion_batch_size=diffusion_batch_size,
                        coord_atom_lvl_to_be_noised=coord_atom_lvl_to_be_noised,
                    )
                )
            else:
                # Full diffusion rollout (no gradients still)
                sample_diffusion_outs = (
                    self.inference_sampler.sample_diffusion_like_af3(
                        f=input["f"],
                        S_inputs_I=recycling_outputs["S_inputs_I"],
                        S_trunk_I=recycling_outputs["S_I"],
                        Z_trunk_II=recycling_outputs["Z_II"],
                        diffusion_module=self.diffusion_module,
                        diffusion_batch_size=diffusion_batch_size,
                        coord_atom_lvl_to_be_noised=coord_atom_lvl_to_be_noised,
                    )
                )

        # ... run non-batched confidence head
        # TODO: Write a version of the confidence head that splits into batches based on memory available
        # (Currently, we OOM with the full batch size, so we loop, which is slow)
        D = sample_diffusion_outs["X_L"].shape[0]
        confidence_stack = {}
        for i in range(D):
            confidence = checkpoint.checkpoint(
                create_custom_forward(
                    self.confidence_head, frame_atom_idxs=input["frame_atom_idxs"]
                ),
                recycling_outputs["S_inputs_I"],
                recycling_outputs["S_I"],
                recycling_outputs["Z_II"],
                sample_diffusion_outs["X_L"][i].unsqueeze(0),
                input["seq"],
                input["rep_atom_idxs"],
                use_reentrant=False,
            )

            for k, v in confidence.items():
                if k in confidence_stack:
                    confidence_stack[k] = torch.cat((confidence_stack[k], v), dim=0)
                else:
                    confidence_stack[k] = v
        confidence = confidence_stack

        # ... run batched confidence head
        # fd too much memory use at training time...
        # confidence = checkpoint.checkpoint(
        #    create_custom_forward(
        #        self.confidence_head, frame_atom_idxs=input["frame_atom_idxs"]
        #    ),
        #    recycling_outputs["S_inputs_I"],
        #    recycling_outputs["S_I"],
        #    recycling_outputs["Z_II"],
        #    sample_diffusion_outs["X_L"],
        #    input["seq"],
        #    input["rep_atom_idxs"],
        #    use_reentrant=False,
        # )

        # TODO: Return outputs in a more structured way (e.g., a dataclass)
        return dict(
            early_stopped=False,
            # We return X_L from diffusion sampling as X_pred_rollout_L to support future joint training with the confidence head (where we would have both X_L and X_pred_rollout_L)
            X_L=None,
            distogram=distogram_pred,
            # For reporting, inference (validation or testing) only
            X_noisy_L_traj=sample_diffusion_outs["X_noisy_L_traj"],
            X_denoised_L_traj=sample_diffusion_outs["X_denoised_L_traj"],
            t_hats=sample_diffusion_outs["t_hats"],
            # Confidence outputs
            X_pred_rollout_L=sample_diffusion_outs["X_L"],
            plddt=confidence["plddt_logits"],
            pae=confidence["pae_logits"],
            pde=confidence["pde_logits"],
            exp_resolved=confidence["exp_resolved_logits"],
        )
