import torch
from beartype.typing import Any, Literal
from jaxtyping import Float

from foundry.utils.ddp import RankedLogger
from foundry.utils.rotation_augmentation import centre_random_augmentation

ranked_logger = RankedLogger(__name__, rank_zero_only=True)


class SampleDiffusion:
    """Algorithm 18"""

    def __init__(
        self,
        *,
        # Hyperparameters
        num_timesteps: int,  # AF-3: 200
        min_t: int,  # AF-3: 0
        max_t: int,  # AF-3: 1
        sigma_data: int,  # AF-3: 16
        s_min: float,  # AF-3: 4e-4
        s_max: int,  # AF-3: 160
        p: int,  # AF-3: 7
        gamma_0: float,  # AF-3: 0.8
        gamma_min: float,  # AF-3: 1.0,
        noise_scale: float,  # AF-3: 1.003,
        step_scale: float,  # AF-3: 1.5,
        solver: Literal["af3"],
    ):
        """Initialize the diffusion sampler, to perform a complete diffusion roll-out with the given recycling outputs.

        We do not use default values for the parameters to make the Hydra configuration the single source of truth and avoid silent failures.

        Args:
            num_timesteps (int): The number of timesteps for which the noise schedule is constructed. Default is 200, per AF3.
            min_t (float): The minimum value of t in the schedule. Default is 0, per AF3.
            max_t (float): The maximum value of t in the schedule. Default is 1, per AF3.
            sigma_data (int): A constant determined by the variance of the data. Default is 16, as defined in the AlphaFold 3 Supplement (Algorithm 20, Diffusion Module).
            s_min (float): The minimum value of the noise schedule. Default is 4e-4, per AF3.
            s_max (float): The maximum value of the noise schedule. Default is 160, per AF3.
            p (int): A constant that determines the shape of the noise schedule. Default is 7, per AF3.
            gamma_0 (float): The value of gamma when t > gamma_min. Default is 0.8, per AF3.
            solver (str): The solver to use for the diffusion process. Default is "af3".

            TODO: Continue documentation of the remaining parameters.
        """
        self.num_timesteps = num_timesteps
        self.min_t = min_t
        self.max_t = max_t
        self.sigma_data = sigma_data
        self.s_min = s_min
        self.s_max = s_max
        self.p = p
        self.gamma_0 = gamma_0
        self.gamma_min = gamma_min
        self.noise_scale = noise_scale
        self.step_scale = step_scale
        self.solver = solver

    def _construct_inference_noise_schedule(self, device: torch.device) -> torch.Tensor:
        """Constructs a noise schedule for use during inference.

        The inference noise schedule is defined in the AF-3 supplement as:

            t_hat = sigma_data * (s_max**(1/p) + t * (s_min**(1/p) - s_max**(1/p)))**p

        Returns:
            torch.Tensor: A tensor representing the noise schedule `t_hat`.

        Reference:
            AlphaFold 3 Supplement, Section 3.7.1.
        """
        # Create a linearly spaced tensor of timesteps between min_t and max_t
        t = torch.linspace(self.min_t, self.max_t, self.num_timesteps, device=device)

        # Construct the noise schedule, using the formula provided in the reference
        t_hat = (
            self.sigma_data
            * (
                (self.s_max) ** (1 / self.p)
                + t * (self.s_min ** (1 / self.p) - self.s_max ** (1 / self.p))
            )
            ** self.p
        )

        return t_hat

    def _get_initial_structure(
        self,
        c0: torch.Tensor,
        D: int,
        L: int,
        coord_atom_lvl_to_be_noised: torch.Tensor,
    ) -> torch.Tensor:
        """Sample initial point cloud from a normal distribution.

        Args:
            c0 (torch.Tensor): A scalar tensor that will be used to scale the initial point cloud. Effectively, the same as
                directly changing the standard deviation of the normal distribution. Derived from noise_schedule[0].
            D (int): The number of structures to sample.
            L (int): The number of atoms in the structure.
            coord_atom_lvl_to_be_noised (torch.Tensor): The atom-level coordinates to be noised (either completely or partially)
        """
        noise = c0 * torch.normal(mean=0.0, std=1.0, size=(D, L, 3), device=c0.device)
        X_L = noise + coord_atom_lvl_to_be_noised

        return X_L

    def sample_diffusion_like_af3(
        self,
        *,
        S_inputs_I: Float[torch.Tensor, "I c_s_inputs"],
        S_trunk_I: Float[torch.Tensor, "I c_s"],
        Z_trunk_II: Float[torch.Tensor, "I I c_z"],
        f: dict[str, Any],
        diffusion_module: torch.nn.Module,
        diffusion_batch_size: int,
        coord_atom_lvl_to_be_noised: Float[torch.Tensor, "D L 3"],
    ) -> dict[str, Any]:
        """Perform a complete diffusion roll-out with the given recycling outputs.

        Args:
            diffusion_module (torch.nn.Module): The diffusion module to use for denoising. If using EMA and performing validation or inference,
                this model should be the EMA model.
        """
        # Construct the noise schedule t_hat for inference on the appropriate device
        noise_schedule = self._construct_inference_noise_schedule(
            device=S_inputs_I.device
        )

        # Infer number of atoms from any atom-level feature
        L = f["ref_element"].shape[0]
        D = diffusion_batch_size

        # Initial X_L is drawn from a normal distribution with a mean vector of 0 and a
        # covariance matrix equal to the 3x3 identity matrix, scaled by the noise schedule
        X_L = self._get_initial_structure(
            c0=noise_schedule[0],
            D=D,
            L=L,
            coord_atom_lvl_to_be_noised=coord_atom_lvl_to_be_noised,
        )  # (D, L, 3)

        X_noisy_L_traj = []
        X_denoised_L_traj = []
        t_hats = []

        for c_t_minus_1, c_t in zip(noise_schedule, noise_schedule[1:]):
            # (All predicted atoms exist)
            X_exists_L = torch.ones((D, L)).bool()  # (D, L)

            # Apply a random rotation and translation to the structure
            # TODO: Make s_trans a hyperparameter
            s_trans = 1.0
            X_L = centre_random_augmentation(X_L, X_exists_L, s_trans)

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
            X_noisy_L = X_L + epsilon_L

            # Denoise the coordinates
            X_denoised_L = diffusion_module(
                X_noisy_L=X_noisy_L,
                t=t_hat.tile(D),
                f=f,
                S_inputs_I=S_inputs_I,
                S_trunk_I=S_trunk_I,
                Z_trunk_II=Z_trunk_II,
            )

            # Compute the delta between the noisy and denoised coordinates, scaled by t_hat
            delta_L = (X_noisy_L - X_denoised_L) / t_hat
            d_t = c_t - t_hat

            # Update the coordinates, scaled by the step size
            X_L = X_noisy_L + self.step_scale * d_t * delta_L

            X_noisy_L_scaled = (
                X_noisy_L
                / (torch.sqrt(t_hat[..., None, None] ** 2 + self.sigma_data**2))
            ) * self.sigma_data
            # Append the results to the trajectory (for visualization of the diffusion process)
            X_noisy_L_traj.append(X_noisy_L_scaled)
            X_denoised_L_traj.append(X_denoised_L)
            t_hats.append(t_hat)

        return dict(
            X_L=X_L,  # (D, L, 3)
            X_noisy_L_traj=X_noisy_L_traj,  # list[Tensor[D, L, 3]]
            X_denoised_L_traj=X_denoised_L_traj,  # list[Tensor[D, L, 3]]
            t_hats=t_hats,  # list[Tensor[D]], where D is shared across all diffusion batches
        )


class SamplePartialDiffusion(SampleDiffusion):
    def __init__(self, partial_t: int, **kwargs):
        super().__init__(**kwargs)
        self.partial_t = partial_t

    def _construct_inference_noise_schedule(self, device: torch.device) -> torch.Tensor:
        """Constructs a noise schedule for use during inference with partial t."""
        t_hat_full = super()._construct_inference_noise_schedule(device)

        assert (
            self.partial_t < self.num_timesteps
        ), f"Partial t ({self.partial_t}) must be less than num_timesteps ({self.num_timesteps})"
        ranked_logger.info(
            f"Using partial t index: {self.partial_t} [e.g., {t_hat_full[self.partial_t]:.4}], or {self.partial_t / (self.num_timesteps):.2%}, by index (100% is data, 0% is noise)"
        )

        return t_hat_full[self.partial_t :]
