import math

import torch
from atomworks.ml.utils.rng import create_rng_state_from_seeds, rng_state


def get_recycle_schedule(
    max_cycle: int,
    n_epochs: int,
    n_train: int,
    world_size: int,
    seed: int = 42,
) -> torch.Tensor:
    """Generate a schedule for recycling iterations over multiple epochs.

    Used to ensure that each GPU has the same number of recycles within a given batch.

    Args:
        max_cycle (int): Maximum number of recycling iterations (n_recycle).
        n_epochs (int): Number of training epochs.
        n_train (int): The total number of training examples per epoch (across all GPUs).
        world_size (int): The number of distributed training processes.
        seed (int, optional): The seed for random number generation. Defaults to 42.

    Returns:
        torch.Tensor: A tensor containing the recycling schedule for each epoch,
            with dimensions `(n_epochs, n_train // world_size)`.

    References:
        AF-2 Supplement, Algorithm 31
    """
    # We use a context manager to avoid modifying the global RNG state
    with rng_state(create_rng_state_from_seeds(torch_seed=seed)):
        # ...generate a recycling schedule for each epoch
        recycle_schedule = []
        for i in range(n_epochs):
            schedule = torch.randint(
                1, max_cycle + 1, (math.ceil(n_train / world_size),)
            )
            recycle_schedule.append(schedule)

    return torch.stack(recycle_schedule, dim=0)
