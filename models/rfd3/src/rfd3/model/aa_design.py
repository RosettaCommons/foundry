import inspect
import os

import hydra
import torch
from beartype.typing import Any
from jaxtyping import Float
from omegaconf import DictConfig
from rfd3.inference.symmetry.symmetry_utils import (
    apply_symmetry_to_xyz_atomwise,
)
from rfd3.model.cfg_utils import (
    strip_f,
    strip_X,
)
from rfd3.model.encoders import TokenInitializer
from torch import nn

from modelhub import SWAP_LAYER_NORM_FOR_RMS_NORM
from modelhub.alignment import weighted_rigid_align
from modelhub.common import exists
from modelhub.data.rotation_augmentation import (
    rot_vec_mul,
    uniform_random_rotation,
)
from modelhub.diffusion_samplers.inference_sampler import SampleDiffusion
from modelhub.utils.ddp import RankedLogger

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

        # Register whether the model uses RMSNorms or LayerNorms
        self.register_buffer(
            "use_rmsnorm",
            torch.tensor(SWAP_LAYER_NORM_FOR_RMS_NORM, dtype=torch.bool),
        )

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
        # Assert that the correct swap is used
        if bool(self.use_rmsnorm.item()) != SWAP_LAYER_NORM_FOR_RMS_NORM:
            raise ValueError(
                "Loaded checkpoint with use RMSNorm {} but environment variable set expects {}".format(
                    self.use_rmsnorm.item(),
                    SWAP_LAYER_NORM_FOR_RMS_NORM,
                )
                + " Set environment variable SWAP_LAYER_NORM_FOR_RMS_NORM to {}".format(
                    {True: "1", False: "0"}[self.use_rmsnorm.item()]
                )
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
                diffusion_batch_size=input["t"].shape[0],
                coord_atom_lvl_to_be_noised=coord_atom_lvl_to_be_noised,
                # Forwarded as **kwargs:
                initializer_outputs=initializer_outputs,
                ref_initializer_outputs=ref_initializer_outputs,  # for cfg
            )


def centre_random_augment_around_motif(
    X_L: torch.Tensor,  # (D, L, 3) noisy diffused coordinates
    coord_atom_lvl_to_be_noised: torch.Tensor,  # (D, L, 3) original coordinates
    is_motif_atom_with_fixed_coord: torch.Tensor,  # (D, L) indices in original coordinates to be kept constant
    s_trans: float = 1.0,
    center_option: str = "all",
    centering_affects_motif: bool = True,
    reinsert_motif=True,
):
    D, L, _ = X_L.shape

    if reinsert_motif and torch.any(is_motif_atom_with_fixed_coord):
        # ... Align original coordinates to the prediction
        coords_with_gt_aligned = weighted_rigid_align(
            X_L[..., is_motif_atom_with_fixed_coord, :],
            coord_atom_lvl_to_be_noised[..., is_motif_atom_with_fixed_coord, :],
        )

        # ... Insert original coordinates into X_L
        X_L[..., is_motif_atom_with_fixed_coord, :] = coords_with_gt_aligned

    # ... Centering
    if torch.any(is_motif_atom_with_fixed_coord):
        if center_option == "motif":
            center = torch.mean(
                X_L[..., is_motif_atom_with_fixed_coord, :], dim=-2, keepdim=True
            )  # (D, 1, 3) - COM of motif atoms
        elif center_option == "diffuse":
            center = torch.mean(
                X_L[..., ~is_motif_atom_with_fixed_coord, :], dim=-2, keepdim=True
            )  # (D, 1, 3) - COM of diffused atoms

        else:
            center = torch.mean(X_L, dim=-2, keepdim=True)
    else:
        center = torch.mean(X_L, dim=-2, keepdim=True)

    # ... Center
    if centering_affects_motif:
        X_L = X_L - center
    else:
        X_L[..., ~is_motif_atom_with_fixed_coord, :] = (
            X_L[..., ~is_motif_atom_with_fixed_coord, :] - center
        )

    # ... Random augmentation
    R = uniform_random_rotation((D,)).to(X_L.device)
    noise = (
        torch.normal(mean=0, std=1, size=(D, 1, 3), device=X_L.device) * s_trans
    )  # (D, 1, 3)
    X_L = rot_vec_mul(R[:, None], X_L) + noise

    return X_L, R


class SampleDiffusionWithMotif(SampleDiffusion):
    def __init__(
        self,
        center_option: str = "all",
        move_noise_to_reset_com: bool = False,  # Reset the COM of the diffuse region after the re-noising operation in each diffusion step
        s_trans: float = 1.0,  # Translational noise scale for augmentation during inference
        s_jitter_origin: float = 0.0,  # Random translation of motif at the start of diffusion
        fraction_of_steps_to_fix_motif: float = 0.0,  # Fraction of steps to let the model not move the motif. e.g. if we have 10 steps, set this value to 0.2 will make model not move motif for the first 2 steps.
        skip_few_diffusion_steps: bool = False,  # Choose to skip some diffusion steps based on the noise scheme
        inference_noise_scaling_factor: float = 1.0,
        # Additional argumnets
        gamma_min2: float = 0.0,
        allow_realignment: bool = False,
        insert_motif_at_end: bool = True,
        use_classifier_free_guidance: bool = False,
        cfg_scale: float = 2.0,
        use_frame_guidance: bool = False,  # Use frame guidance to align the virtual atoms to the central atom
        fg_scale: float = 1.5,
        zero_drift_noise: bool = False,
        cfg_t_max: float
        | None = None,  # If not None, use classifier-free guidance only for t < cfg_t_max
        **kwargs,
    ):
        self.gamma_min2 = gamma_min2
        self.allow_realignment = allow_realignment
        self.insert_motif_at_end = insert_motif_at_end
        self.use_classifier_free_guidance = use_classifier_free_guidance
        self.cfg_scale = cfg_scale
        self.cfg_t_max = cfg_t_max

        self.center_option = center_option
        self.fraction_of_steps_to_fix_motif = fraction_of_steps_to_fix_motif
        self.move_noise_to_reset_com = move_noise_to_reset_com
        self.s_trans = s_trans
        self.skip_few_diffusion_steps = skip_few_diffusion_steps
        self.s_jitter_origin = s_jitter_origin
        self.inference_noise_scaling_factor = inference_noise_scaling_factor
        self.zero_drift_noise = zero_drift_noise

        self.use_frame_guidance = use_frame_guidance
        self.fg_scale = fg_scale

        super().__init__(**kwargs)

    # TODO: Make this a properly-parametrized function in terms of instance variables provided in the configs
    # For now, it's just hard-coded for early testing
    def modify_noise_schedule(self, noise_schedule: torch.Tensor) -> torch.Tensor:
        """
        Modify the noise schedule to skip more steps at high noise and fewer at low noise.
        """
        mask = torch.ones_like(noise_schedule, dtype=bool)
        mask_len = len(mask)
        mask[: mask_len // 4] = torch.arange(mask_len // 4) % 5 == 0
        mask[mask_len // 4 : mask_len // 2] = (
            torch.arange(mask_len // 4, mask_len // 2) % 3 == 0
        )
        mask[mask_len // 2 : -mask_len // 4] = (
            torch.arange(mask_len // 2, mask_len - mask_len // 4) % 2 == 0
        )
        return noise_schedule[mask]

    def _get_initial_structure(
        self,
        c0: torch.Tensor,
        D: int,
        L: int,
        coord_atom_lvl_to_be_noised: torch.Tensor,
        is_motif_atom_with_fixed_coord,
    ) -> torch.Tensor:
        noise = c0 * torch.normal(mean=0.0, std=1.0, size=(D, L, 3), device=c0.device)
        noise[..., is_motif_atom_with_fixed_coord, :] = 0  # Zero out noise going in
        X_L = noise + coord_atom_lvl_to_be_noised
        return X_L

    def _move_noise_to_reset_com(self, X_noisy_L, is_motif_atom_with_fixed_coord):
        """
        Reset the COM of the diffuse region after the re-noising operation in each diffusion step.
        """
        if self.center_option == "motif":
            print(
                "Warning: Moving the noise is not relevant when centering on the motif! Will be ignored."
            )
        elif self.center_option == "diffuse":
            displacement_vec = torch.mean(
                X_noisy_L[..., ~is_motif_atom_with_fixed_coord, :],
                dim=-2,
                keepdim=True,
            )  # (D, 1, 3) - COM of noisy diffused atoms

            X_noisy_L[..., ~is_motif_atom_with_fixed_coord, :] = (
                X_noisy_L[..., ~is_motif_atom_with_fixed_coord, :] - displacement_vec
            )
        else:
            n_diffused = (~is_motif_atom_with_fixed_coord).sum()
            displacement_vec = (
                torch.sum(
                    X_noisy_L,
                    dim=-2,
                    keepdim=True,
                )
                / n_diffused
            )

            X_noisy_L[..., ~is_motif_atom_with_fixed_coord, :] = (
                X_noisy_L[..., ~is_motif_atom_with_fixed_coord, :] - displacement_vec
            )

        return X_noisy_L

    def _skip_few_diffusion_steps(self, noise_schedule: torch.Tensor) -> torch.Tensor:
        """
        Modify the noise schedule to skip more steps at high noise and fewer at low noise.
        i.e. When the noise is high (first few diffusion steps), skip more steps;
             When the noise is lower, skip fewer steps;
             When the noise is low, keep all the steps.
        """
        mask = torch.ones_like(noise_schedule, dtype=bool)
        mask_len = len(mask)
        mask[: mask_len // 4] = torch.arange(mask_len // 4) % 5 == 0
        mask[mask_len // 4 : mask_len // 2] = (
            torch.arange(mask_len // 4, mask_len // 2) % 3 == 0
        )
        mask[mask_len // 2 : -mask_len // 4] = (
            torch.arange(mask_len // 2, mask_len - mask_len // 4) % 2 == 0
        )
        return noise_schedule[mask]

    def sample_diffusion_like_af3(
        self,
        *,
        f: dict[str, Any],
        diffusion_module: torch.nn.Module,
        diffusion_batch_size: int,
        coord_atom_lvl_to_be_noised: Float[torch.Tensor, "D L 3"],
        initializer_outputs,
        ref_initializer_outputs: dict[str, Any] | None,
        f_ref: dict[str, Any] | None,
    ) -> dict[str, Any]:
        # Motif setup to recenter the motif at every step
        is_motif_atom_with_fixed_coord = f["is_motif_atom_with_fixed_coord"]

        # Book-keeping
        noise_schedule = self._construct_inference_noise_schedule(
            device=coord_atom_lvl_to_be_noised.device
        )

        # Choose to adjust the noise schedule
        if self.skip_few_diffusion_steps:
            noise_schedule = self._skip_few_diffusion_steps(noise_schedule)

        if "partial_t" in f:
            # For now, partial t is a global parameter
            partial_t = f["partial_t"].mean()
            ranked_logger.info("Using partial diffusion with t={}".format(partial_t))

            # Debug the noise schedule filtering
            original_schedule_len = len(noise_schedule)
            original_max = noise_schedule.max().item()
            original_min = noise_schedule.min().item()

            noise_schedule = noise_schedule[noise_schedule <= partial_t]

            new_schedule_len = len(noise_schedule)
            if new_schedule_len > 0:
                new_max = noise_schedule.max().item()
                new_min = noise_schedule.min().item()
                ranked_logger.info(
                    f"Noise schedule: {original_schedule_len} → {new_schedule_len} steps"
                )
                ranked_logger.info(
                    f"Original range: [{original_min:.3f}, {original_max:.3f}]"
                )
                ranked_logger.info(f"Filtered range: [{new_min:.3f}, {new_max:.3f}]")
            else:
                ranked_logger.warning(
                    f"No noise schedule steps found with t <= {partial_t}!"
                )
                ranked_logger.info(
                    f"Original schedule range: [{original_min:.3f}, {original_max:.3f}]"
                )
                # Fallback to smallest available step
                noise_schedule_original = self._construct_inference_noise_schedule(
                    device=coord_atom_lvl_to_be_noised.device
                )
                noise_schedule = noise_schedule_original[-1:]  # Just use the final step
                ranked_logger.info(
                    f"Using fallback: final step with t={noise_schedule[0].item():.6f}"
                )

        L = f["ref_element"].shape[0]
        D = diffusion_batch_size

        noise_schedule = noise_schedule * self.inference_noise_scaling_factor

        X_L = self._get_initial_structure(
            c0=noise_schedule[0],
            D=D,
            L=L,
            coord_atom_lvl_to_be_noised=coord_atom_lvl_to_be_noised.clone(),
            is_motif_atom_with_fixed_coord=is_motif_atom_with_fixed_coord,
        )  # (D, L, 3)

        if self.s_jitter_origin > 0.0:
            X_L[:, is_motif_atom_with_fixed_coord, :] += torch.normal(
                mean=0.0,
                std=self.s_jitter_origin,
                size=(D, 1, 3),
                device=X_L.device,
            )

        X_noisy_L_traj = []
        X_denoised_L_traj = []
        sequence_entropy_traj = []
        t_hats = []

        threshold_step = (len(noise_schedule) - 1) * self.fraction_of_steps_to_fix_motif

        for step_num, (c_t_minus_1, c_t) in enumerate(
            zip(noise_schedule, noise_schedule[1:])
        ):
            # Assert no grads on X_L
            assert not torch.is_grad_enabled(), "Computation graph should not be active"
            assert not X_L.requires_grad, "X_L should not require gradients"

            # Apply a random rotation and translation to the structure
            if self.allow_realignment:
                X_L, _ = centre_random_augment_around_motif(
                    X_L,
                    coord_atom_lvl_to_be_noised,
                    is_motif_atom_with_fixed_coord,
                    center_option=self.center_option,
                    # If centering_affects_motif is True, the model's predictions from (step_num-1) might affect the motif
                    centering_affects_motif=(max(step_num - 1, 0)) >= threshold_step,
                    # If keeping the motif position wrt the origin fixed, we can't do translational augmentation
                    # We want to keep this position fixed in the interval where the model is not allowed to change it
                    s_trans=self.s_trans if step_num >= threshold_step else 0.0,
                )

            # Update gamma & step scale
            gamma = self.gamma_0 if c_t > self.gamma_min else 0
            step_scale = self.step_scale if c_t > self.gamma_min2 else 3.0

            # Compute the value of t_hat
            t_hat = c_t_minus_1 * (gamma + 1)

            # Noise the coordinates with scaled Gaussian noise
            epsilon_L = (
                self.noise_scale
                * torch.sqrt(torch.square(t_hat) - torch.square(c_t_minus_1))
                * torch.normal(mean=0.0, std=1.0, size=X_L.shape, device=X_L.device)
            )
            if self.zero_drift_noise:
                epsilon_L = epsilon_L - torch.mean(epsilon_L, dim=-2, keepdim=True)
            epsilon_L[..., is_motif_atom_with_fixed_coord, :] = (
                0  # No noise injection for fixed atoms
            )
            X_noisy_L = X_L + epsilon_L

            # Adjustg the center of mass
            if self.move_noise_to_reset_com:
                X_noisy_L = self._move_noise_to_reset_com(
                    X_noisy_L, is_motif_atom_with_fixed_coord
                )

            # Denoise the coordinates
            # Handle chunked mode vs standard mode
            if "chunked_pairwise_embedder" in initializer_outputs:
                # Chunked mode: explicitly provide P_LL=None
                chunked_embedder = initializer_outputs[
                    "chunked_pairwise_embedder"
                ]  # Don't pop, just get
                other_outputs = {
                    k: v
                    for k, v in initializer_outputs.items()
                    if k != "chunked_pairwise_embedder"
                }
                outs = diffusion_module(
                    X_noisy_L=X_noisy_L,
                    t=t_hat.tile(D),
                    f=f,
                    P_LL=None,  # Not used in chunked mode
                    chunked_pairwise_embedder=chunked_embedder,
                    initializer_outputs=other_outputs,
                    **other_outputs,
                )
            else:
                # Standard mode: P_LL is included in initializer_outputs
                outs = diffusion_module(
                    X_noisy_L=X_noisy_L,
                    t=t_hat.tile(D),
                    f=f,
                    **initializer_outputs,
                )

            X_denoised_L = outs["X_L"] if "X_L" in outs else outs

            # Compute the delta between the noisy and denoised coordinates, scaled by t_hat
            delta_L = (
                X_noisy_L - X_denoised_L
            ) / t_hat  # gradient of x wrt. t at x_t_hat
            d_t = c_t - t_hat

            if self.use_classifier_free_guidance and (
                self.cfg_t_max is None or c_t > self.cfg_t_max
            ):
                X_noisy_L_stripped = strip_X(X_noisy_L, f_ref)

                # unconditional forward pass
                outs_ref = diffusion_module(
                    X_noisy_L=X_noisy_L_stripped,  # modify X
                    t=t_hat.tile(D),
                    f=f_ref,  # modified f
                    **ref_initializer_outputs,
                )

                X_denoised_L_stripped = outs_ref["X_L"]

                delta_L_ref = (
                    X_noisy_L_stripped - X_denoised_L_stripped
                ) / t_hat  # gradient of x wrt. t at x_t_hat

                # pad delta_L_ref with zeros to match delta_L (for the unindexed atoms)
                if delta_L_ref.shape[1] < delta_L.shape[1]:
                    delta_L_ref = torch.cat(
                        [
                            delta_L_ref,
                            torch.zeros_like(delta_L[:, delta_L_ref.shape[1] :, :]),
                        ],
                        dim=1,
                    )

                # apply CFG
                delta_L = delta_L + (self.cfg_scale - 1) * (delta_L - delta_L_ref)

            if self.use_frame_guidance:
                X_L_ref_frame = outs.get("X_L_ref_frame")
                delta_L_ref = (X_noisy_L - X_L_ref_frame) / t_hat
                delta_L = delta_L + (self.fg_scale - 1) * (delta_L - delta_L_ref)

            if exists(outs.get("sequence_logits_I")):
                # Compute confidence
                p = torch.softmax(
                    outs["sequence_logits_I"], dim=-1
                ).cpu()  # shape (D, L, 32)
                seq_entropy = -torch.sum(
                    p * torch.log(p + 1e-10), dim=-1
                )  # shape (D, L,)
                sequence_entropy_traj.append(seq_entropy)

            # Update the coordinates, scaled by the step size
            X_L = X_noisy_L + step_scale * d_t * delta_L

            # Append the results to the trajectory (for visualization of the diffusion process)
            X_noisy_L_scaled = (
                self.sigma_data * X_noisy_L / torch.sqrt(t_hat**2 + self.sigma_data**2)
            )  # Save noisy traj as scaled inputs
            X_noisy_L_traj.append(X_noisy_L_scaled)
            X_denoised_L_traj.append(X_denoised_L)
            t_hats.append(t_hat)

        if torch.any(is_motif_atom_with_fixed_coord) and self.allow_realignment:
            # Insert the gt motif at the end
            X_L, _ = centre_random_augment_around_motif(
                X_L,
                coord_atom_lvl_to_be_noised,
                is_motif_atom_with_fixed_coord,
                reinsert_motif=self.insert_motif_at_end,
            )

            # Align prediction to original motif
            X_L = weighted_rigid_align(
                coord_atom_lvl_to_be_noised,
                X_L,
                X_exists_L=is_motif_atom_with_fixed_coord,
            )

        return dict(
            X_L=X_L,  # (D, L, 3)
            X_noisy_L_traj=X_noisy_L_traj,  # list[Tensor[D, L, 3]]
            X_denoised_L_traj=X_denoised_L_traj,  # list[Tensor[D, L, 3]]
            t_hats=t_hats,  # list[Tensor[D]], where D is shared across all diffusion batches
            sequence_logits_I=outs.get("sequence_logits_I"),  # (D, I, 32)
            sequence_indices_I=outs.get("sequence_indices_I"),  # (D, I, 32)
            sequence_entropy_traj=sequence_entropy_traj,  # list[Tensor[D, I]]
        )


class SampleDiffusionWithSymmetry(SampleDiffusionWithMotif):
    """
    This class is a wrapper around the SampleDiffusionWithMotif class.
    It is used to sample diffusion with symmetry.
    """

    def __init__(self, sym_step_frac: float = 0.9, **kwargs):
        assert (
            kwargs.get("gamma_0") > 0.5
        ), "gamma_0 must be greater than 0.5 for symmetry sampling"
        self.sym_step_frac = sym_step_frac
        super().__init__(**kwargs)

    def apply_symmetry_to_X_L(self, X_L, f):
        # check that we are doing symmetric inference

        assert "sym_transform" in f.keys(), "Symmetry transform not found in f"

        # update symmetric frames to correct for change in global frame
        symmetry_feats = {k: v for k, v in f.items() if "sym" in k}

        # apply symmetry frame shift to X_L
        X_L = apply_symmetry_to_xyz_atomwise(
            X_L, symmetry_feats, partial_diffusion=("partial_t" in f)
        )

        return X_L

    def sample_diffusion_like_af3(
        self,
        *,
        f: dict[str, Any],
        diffusion_module: torch.nn.Module,
        diffusion_batch_size: int,
        coord_atom_lvl_to_be_noised: Float[torch.Tensor, "D L 3"],
        initializer_outputs,
        ref_initializer_outputs: dict[str, Any] | None,
        f_ref: dict[str, Any] | None,
        **_,
    ) -> dict[str, Any]:
        # Motif setup to recenter the motif at every step
        is_motif_atom_with_fixed_coord = f["is_motif_atom_with_fixed_coord"]
        # Book-keeping
        noise_schedule = self._construct_inference_noise_schedule(
            device=coord_atom_lvl_to_be_noised.device
        )

        # Handle partial_t for symmetry sampler (same as regular sampler)
        if "partial_t" in f:
            # For now, partial t is a global parameter
            partial_t = f["partial_t"].mean()
            ranked_logger.info(
                "Symmetry sampler: Using partial diffusion with t={}".format(partial_t)
            )

            # Debug the noise schedule filtering
            original_schedule_len = len(noise_schedule)
            original_max = noise_schedule.max().item()
            original_min = noise_schedule.min().item()

            noise_schedule = noise_schedule[noise_schedule <= partial_t]

            new_schedule_len = len(noise_schedule)
            if new_schedule_len > 0:
                new_max = noise_schedule.max().item()
                new_min = noise_schedule.min().item()
                ranked_logger.info(
                    f"Symmetry noise schedule: {original_schedule_len} → {new_schedule_len} steps"
                )
                ranked_logger.info(
                    f"Symmetry original range: [{original_min:.3f}, {original_max:.3f}]"
                )
                ranked_logger.info(
                    f"Symmetry filtered range: [{new_min:.3f}, {new_max:.3f}]"
                )
            else:
                ranked_logger.warning(
                    f"Symmetry sampler: No noise schedule steps found with t <= {partial_t}!"
                )
                ranked_logger.info(
                    f"Symmetry original schedule range: [{original_min:.3f}, {original_max:.3f}]"
                )
                # Fallback to smallest available step
                noise_schedule_original = self._construct_inference_noise_schedule(
                    device=coord_atom_lvl_to_be_noised.device
                )
                noise_schedule = noise_schedule_original[-1:]  # Just use the final step
                ranked_logger.info(
                    f"Symmetry using fallback: final step with t={noise_schedule[0].item():.6f}"
                )

        L = f["ref_element"].shape[0]
        D = diffusion_batch_size
        X_L = self._get_initial_structure(
            c0=noise_schedule[0],
            D=D,
            L=L,
            coord_atom_lvl_to_be_noised=coord_atom_lvl_to_be_noised.clone(),
            is_motif_atom_with_fixed_coord=is_motif_atom_with_fixed_coord,
        )  # (D, L, 3)

        X_noisy_L_traj = []
        X_denoised_L_traj = []
        sequence_entropy_traj = []
        t_hats = []

        # symmetrize X_L until the step gamma = gamma_min_sym
        gamma_min_sym_idx = min(
            int(len(noise_schedule) * self.sym_step_frac), len(noise_schedule) - 1
        )
        gamma_min_sym = noise_schedule[gamma_min_sym_idx]

        ranked_logger.info(f"gamma_min_sym: {gamma_min_sym}")
        ranked_logger.info(f"gamma_min: {self.gamma_min}")
        for step_num, (c_t_minus_1, c_t) in enumerate(
            zip(noise_schedule, noise_schedule[1:])
        ):
            # Assert no grads on X_L
            assert not torch.is_grad_enabled(), "Computation graph should not be active"
            assert not X_L.requires_grad, "X_L should not require gradients"

            # Apply a random rotation and translation to the structure
            if self.allow_realignment:
                X_L, R = centre_random_augment_around_motif(
                    X_L,
                    coord_atom_lvl_to_be_noised,
                    is_motif_atom_with_fixed_coord,
                )

            # Update gamma & step scale
            gamma = self.gamma_0 if c_t > self.gamma_min else 0
            step_scale = self.step_scale if c_t > self.gamma_min2 else 1.05

            # Compute the value of t_hat
            t_hat = c_t_minus_1 * (gamma + 1)

            # Noise the coordinates with scaled Gaussian noise
            epsilon_L = (
                self.noise_scale
                * torch.sqrt(torch.square(t_hat) - torch.square(c_t_minus_1))
                * torch.normal(mean=0.0, std=1.0, size=X_L.shape, device=X_L.device)
            )
            epsilon_L[..., is_motif_atom_with_fixed_coord, :] = (
                0  # No noise injection for fixed atoms
            )

            # NOTE: no symmetry applied to the noisy structure
            X_noisy_L = X_L + epsilon_L

            # Denoise the coordinates
            # Handle chunked mode vs standard mode (same as default sampler)
            if "chunked_pairwise_embedder" in initializer_outputs:
                # Chunked mode: explicitly provide P_LL=None
                chunked_embedder = initializer_outputs[
                    "chunked_pairwise_embedder"
                ]  # Don't pop, just get
                other_outputs = {
                    k: v
                    for k, v in initializer_outputs.items()
                    if k != "chunked_pairwise_embedder"
                }
                outs = diffusion_module(
                    X_noisy_L=X_noisy_L,
                    t=t_hat.tile(D),
                    f=f,
                    P_LL=None,  # Not used in chunked mode
                    chunked_pairwise_embedder=chunked_embedder,
                    initializer_outputs=other_outputs,
                    **other_outputs,
                )
            else:
                # Standard mode: P_LL is included in initializer_outputs
                outs = diffusion_module(
                    X_noisy_L=X_noisy_L,
                    t=t_hat.tile(D),
                    f=f,
                    **initializer_outputs,
                )
            # apply symmetry to X_denoised_L
            if "X_L" in outs and c_t > gamma_min_sym:
                # outs["original_X_L"] = outs["X_L"].clone()
                outs["X_L"] = self.apply_symmetry_to_X_L(outs["X_L"], f)

            X_denoised_L = outs["X_L"] if "X_L" in outs else outs

            # Compute the delta between the noisy and denoised coordinates, scaled by t_hat
            delta_L = (
                X_noisy_L - X_denoised_L
            ) / t_hat  # gradient of x wrt. t at x_t_hat
            d_t = c_t - t_hat

            # NOTE: no classifier-free guidance for symmetry

            if exists(outs.get("sequence_logits_I")):
                # Compute confidence
                p = torch.softmax(
                    outs["sequence_logits_I"], dim=-1
                ).cpu()  # shape (D, L, 32)
                seq_entropy = -torch.sum(
                    p * torch.log(p + 1e-10), dim=-1
                )  # shape (D, L,)
                sequence_entropy_traj.append(seq_entropy)

            # Update the coordinates, scaled by the step size
            # delta_L should be symmetric
            X_L = X_noisy_L + step_scale * d_t * delta_L

            # Append the results to the trajectory (for visualization of the diffusion process)
            X_noisy_L_scaled = (
                self.sigma_data * X_noisy_L / torch.sqrt(t_hat**2 + self.sigma_data**2)
            )  # Save noisy traj as scaled inputs
            X_noisy_L_traj.append(X_noisy_L_scaled)
            X_denoised_L_traj.append(X_denoised_L)
            t_hats.append(t_hat)

        if torch.any(is_motif_atom_with_fixed_coord) and self.allow_realignment:
            # Insert the gt motif at the end
            X_L, R = centre_random_augment_around_motif(
                X_L,
                coord_atom_lvl_to_be_noised,
                is_motif_atom_with_fixed_coord,
                reinsert_motif=self.insert_motif_at_end,
            )

            # apply symmetry frame shift to X_L
            X_L = self.apply_symmetry_to_X_L(X_L, f)

            # Align prediction to original motif
            X_L = weighted_rigid_align(
                coord_atom_lvl_to_be_noised,
                X_L,
                X_exists_L=is_motif_atom_with_fixed_coord,
            )

        return dict(
            X_L=X_L,  # (D, L, 3)
            X_noisy_L_traj=X_noisy_L_traj,  # list[Tensor[D, L, 3]]
            X_denoised_L_traj=X_denoised_L_traj,  # list[Tensor[D, L, 3]]
            t_hats=t_hats,  # list[Tensor[D]], where D is shared across all diffusion batches
            sequence_logits_I=outs.get("sequence_logits_I"),  # (D, I, 32)
            sequence_indices_I=outs.get("sequence_indices_I"),  # (D, I, 32)
            sequence_entropy_traj=sequence_entropy_traj,  # list[Tensor[D, I]]
        )


class ConditionalDiffusionSampler:
    """
    Conditional diffusion sampler, chooses at construction time which sampler to use,
    then forwards `sample_diffusion_like_af3` to the chosen sampler.
    If you write a new sampler, you best add it to the registry below
    and inference_sampler.kind in inference_engine config.
    """

    _registry = {
        "default": SampleDiffusionWithMotif,
        "symmetry": SampleDiffusionWithSymmetry,
    }

    def __init__(self, kind="default", **kwargs):
        ranked_logger.info(
            f"Initializing ConditionalDiffusionSampler with kind: {kind}"
        )
        try:
            SamplerCls = self._registry[kind]
            # remove kwargs that the sampler cannot take
            init_args = self.get_class_init_args(SamplerCls)
            kwargs = {k: v for k, v in kwargs.items() if k in init_args}
        except KeyError:
            raise ValueError(
                f"Invalid sampler kind: {kind}, must be one of {list(self._registry.keys())}"
            )
        self.sampler = SamplerCls(**kwargs)

    def sample_diffusion_like_af3(self, **kwargs):
        return self.sampler.sample_diffusion_like_af3(**kwargs)

    def get_class_init_args(self, cls):
        arg_names = []
        if hasattr(cls, "__init__") and callable(cls.__init__):
            for p_cls in cls.__mro__:
                if "__init__" in p_cls.__dict__ and p_cls is not object:
                    signature = inspect.signature(p_cls.__init__)
                    arg_names.extend(
                        [param.name for param in signature.parameters.values()]
                    )
        return arg_names
