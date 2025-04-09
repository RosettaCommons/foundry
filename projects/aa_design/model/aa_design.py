import hydra
import torch
from beartype.typing import Any
from jaxtyping import Float
from omegaconf import DictConfig
from torch import nn

from modelhub.alignment import weighted_rigid_align
from modelhub.data.rotation_augmentation import (
    rot_vec_mul,
    uniform_random_rotation,
)
from modelhub.diffusion_samplers.inference_sampler import SampleDiffusion
from modelhub.utils.ddp import RankedLogger
from projects.aa_design.model.encoders import TokenInitializer

ranked_logger = RankedLogger(__name__, rank_zero_only=True)

# Placeholder for future modifications
class AF3Design(nn.Module):
    '''
    Simplified model for generation
    This module level serves to wrap the diffusion module of AF3
    to be roughly equivalent to the AF3 model w/o trunk processing.

    Allows the same sampler to be used
    '''
    def __init__(self,
                 *,
                 # Channel dimensions ('global' features)
                 c_s: int,  
                 c_z: int,
                 c_atom: int,
                 c_atompair: int,
                 c_s_init: int,
                 # Arguments for modules that will be instantiated
                 token_initializer: DictConfig | dict,
                 diffusion_module: DictConfig | dict,
                 inference_sampler: DictConfig | dict,
                 **_):

        super().__init__()

        # Simple constant-feature initializer
        self.token_initializer = TokenInitializer(
            c_s=c_s,
            c_z=c_z,
            c_s_init=c_s_init,
            **token_initializer
        )

        # Diffusion module instantiated to allow for config scripting
        self.diffusion_module = hydra.utils.instantiate(diffusion_module,
            c_atom=c_atom, c_atompair=c_atompair, c_s=c_s, c_z=c_z
        )

        # ... initialize the inference sampler, which performs a full diffusion rollout during inference
        self.inference_sampler = SampleDiffusionWithMotif(**inference_sampler)
        
    def forward(
        self,
        input: dict,
        coord_atom_lvl_to_be_noised: torch.Tensor = None,
        **_
    ) -> dict:
        S_init_I, Z_init_II = self.token_initializer(input['f'], input['t'])
    
        if self.training:
            # Single denoising step
            return self.diffusion_module(
                X_noisy_L=input["X_noisy_L"],
                t=input["t"],
                f=input["f"],
                # Additional trunk outputs:
                S_init_I=S_init_I, 
                Z_init_II=Z_init_II,
            )  # [D, L, 3]
        else:
            return self.inference_sampler.sample_diffusion_like_af3(
                f=input["f"],
                diffusion_module=self.diffusion_module,
                diffusion_batch_size=input["t"].shape[0],
                coord_atom_lvl_to_be_noised=coord_atom_lvl_to_be_noised,
                # Forwarded as **kwargs:
                S_init_I=S_init_I, Z_init_II=Z_init_II,
            )

def centre_random_augment_around_motif(
    X_L: torch.Tensor,  # (D, L, 3) noisy diffused coordinates
    coord_atom_lvl_to_be_noised: torch.Tensor,  # (D, L, 3) original coordinates
    is_motif_atom_with_fixed_pos: torch.Tensor,  # (D, L) indices in original coordinates to be kept constant
    s_trans: float = 1.0,
):
    D, L, _ = X_L.shape

    if torch.any(is_motif_atom_with_fixed_pos):
        # ... Align original coordinates to the prediction
        coords_with_gt_aligned = weighted_rigid_align(
            X_L[..., is_motif_atom_with_fixed_pos, :], 
            coord_atom_lvl_to_be_noised[..., is_motif_atom_with_fixed_pos, :],
        )
        
        # ... Insert original coordinates into X_L
        X_L[..., is_motif_atom_with_fixed_pos, :] = coords_with_gt_aligned
    
    # ... Centering
    center = torch.mean(X_L, dim=-2, keepdim=True)
    X_L = X_L - center

    # ... Random augmentation
    R = uniform_random_rotation((D,)).to(X_L.device)
    noise = torch.normal(mean=0, std=1, size=(D, 1, 3), device=X_L.device) * s_trans  # (D, 1, 3)
    X_L = rot_vec_mul(R[:, None], X_L) + noise

    return X_L

class SampleDiffusionWithMotif(SampleDiffusion):

    def _get_initial_structure(
        self,
        c0: torch.Tensor,
        D: int,
        L: int,
        coord_atom_lvl_to_be_noised: torch.Tensor,
        is_motif_atom_with_fixed_pos
    ) -> torch.Tensor:
        noise = c0 * torch.normal(mean=0.0, std=1.0, size=(D, L, 3), device=c0.device)
        noise[..., is_motif_atom_with_fixed_pos, :] = 0  # Zero out noise going in
        X_L = noise + coord_atom_lvl_to_be_noised
        return X_L

    def sample_diffusion_like_af3(
        self,
        *,
        f: dict[str, Any],
        diffusion_module: torch.nn.Module,
        diffusion_batch_size: int,
        coord_atom_lvl_to_be_noised: Float[torch.Tensor, "D L 3"],
        **trunk_outputs,
    ) -> dict[str, Any]:
        # Motif setup to recenter the motif at every step
        is_motif_atom_with_fixed_pos = f['is_motif_atom_with_fixed_pos'] 

        # Book-keeping
        noise_schedule = self._construct_inference_noise_schedule(
            device=coord_atom_lvl_to_be_noised.device
        )
        L = f["ref_element"].shape[0]
        D = diffusion_batch_size
        X_L = self._get_initial_structure(
            c0=noise_schedule[0],
            D=D,
            L=L,
            coord_atom_lvl_to_be_noised=coord_atom_lvl_to_be_noised.clone(),
            is_motif_atom_with_fixed_pos=is_motif_atom_with_fixed_pos
        )  # (D, L, 3)

        X_noisy_L_traj = []
        X_denoised_L_traj = []
        t_hats = []

        for c_t_minus_1, c_t in zip(noise_schedule, noise_schedule[1:]):
            # Apply a random rotation and translation to the structure
            X_L = centre_random_augment_around_motif(
                X_L, coord_atom_lvl_to_be_noised, is_motif_atom_with_fixed_pos,
            )

            # Update gamma
            gamma = self.gamma_0 if c_t > self.gamma_min else 0

            # Compute the value of t_hat
            t_hat = c_t_minus_1 * (gamma + 1)

            # Noise the coordinates with scaled Gaussian noise
            epsilon_L = (
                self.noise_scale
                * torch.sqrt(torch.square(t_hat) - torch.square(c_t_minus_1))
                * torch.normal(mean=0.0, std=1.0, size=X_L.shape, device=X_L.device)
            )
            epsilon_L[..., is_motif_atom_with_fixed_pos, :] = 0  # No noise injection for fixed atoms
            X_noisy_L = X_L + epsilon_L

            # Denoise the coordinates
            outs = diffusion_module(
                X_noisy_L=X_noisy_L,
                t=t_hat.tile(D),
                f=f,
                **trunk_outputs,
            )
            X_denoised_L = outs["X_L"] if isinstance(outs, dict) and "X_L" in outs else outs

            # Compute the delta between the noisy and denoised coordinates, scaled by t_hat
            delta_L = (X_noisy_L - X_denoised_L) / t_hat
            d_t = c_t - t_hat

            # Update the coordinates, scaled by the step size
            X_L = X_noisy_L + self.step_scale * d_t * delta_L

            # Append the results to the trajectory (for visualization of the diffusion process)
            X_noisy_L_traj.append(X_noisy_L)
            X_denoised_L_traj.append(X_denoised_L)
            t_hats.append(t_hat)

        if torch.any(is_motif_atom_with_fixed_pos):
            # Insert the gt motif at the end
            X_L = centre_random_augment_around_motif(
                X_L, coord_atom_lvl_to_be_noised, is_motif_atom_with_fixed_pos,
            )

            # Align prediction to original motif
            X_L = weighted_rigid_align(
                coord_atom_lvl_to_be_noised, X_L, 
                X_exists_L=is_motif_atom_with_fixed_pos
            )


        return dict(
            X_L=X_L,  # (D, L, 3)
            X_noisy_L_traj=X_noisy_L_traj,  # list[Tensor[D, L, 3]]
            X_denoised_L_traj=X_denoised_L_traj,  # list[Tensor[D, L, 3]]
            t_hats=t_hats,  # list[Tensor[D]], where D is shared across all diffusion batches
            Seq_I=outs["Seq_I"] # (D, I, 32)
        )
