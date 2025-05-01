"""Utils for loading weights from checkpoints."""

import re
from dataclasses import dataclass, field
from enum import StrEnum, auto
from os import PathLike

import torch
from beartype.typing import Pattern
from torch import nn

from modelhub.utils.ddp import RankedLogger

ranked_logger = RankedLogger(__name__, rank_zero_only=True)


class WeightLoadingError(Exception):
    """Exception raised when there's an error loading weights."""

    pass


class WeightLoadingPolicy(StrEnum):
    """Policy for handling weights when loading checkpoints."""

    # Always keep default initialization, regardless of whether the parameter is in the checkpoint or shapes match
    REINIT = auto()

    # Always zero-initialize, regardless of whether the parameter is in the checkpoint or shapes match
    ZERO_INIT = auto()

    # Copy from checkpoint only when shapes match exactly, otherwise error
    COPY = auto()

    # Copy from checkpoint if tensors are the same rank, padding with zeros if shapes don't match exectly
    COPY_AND_ZERO_PAD = auto()


@dataclass
class WeightLoadingConfig:
    """Configuration for handling weights when loading a checkpoint."""

    # Default policy to apply when no pre-defined rule matches
    default_policy: WeightLoadingPolicy | str = WeightLoadingPolicy.COPY

    # Fallback policy to apply when the primary policy cannot be applied
    # (e.g., when COPY fails due to shape mismatch or parameter not found)
    fallback_policy: WeightLoadingPolicy | str = WeightLoadingPolicy.REINIT

    # Dictionary mapping parameter names or patterns to policies
    param_policies: dict[str, WeightLoadingPolicy | str] = field(default_factory=dict)

    # Compiled regex patterns (populated internally)
    _compiled_patterns: dict[Pattern, WeightLoadingPolicy] = field(
        default_factory=dict, repr=False
    )

    def __post_init__(self):
        """Compile regex patterns after initialization."""
        # If any policies are provided as strings, convert them to WeightLoadingPolicy
        if isinstance(self.default_policy, str):
            self.default_policy = WeightLoadingPolicy(self.default_policy)
        if isinstance(self.fallback_policy, str):
            self.fallback_policy = WeightLoadingPolicy(self.fallback_policy)
        for key, value in self.param_policies.items():
            if isinstance(value, str):
                self.param_policies[key] = WeightLoadingPolicy(value)

        # Compile patterns
        if self.param_policies:
            for pattern, policy in list(self.param_policies.items()):
                if any(c in pattern for c in ["*", "?", "[", "]"]):
                    # Convert glob-style pattern to regex
                    regex = (
                        pattern.replace(".", r"\.")
                        .replace("*", ".*")
                        .replace("?", ".")
                        .replace("[", "[")
                        .replace("]", "]")
                    )
                    self._compiled_patterns[re.compile(f"^{regex}$")] = policy

    def get_policy(self, param_name: str) -> WeightLoadingPolicy:
        """Get the policy for a specific parameter name."""
        # First check exact matches
        if self.param_policies and param_name in self.param_policies:
            return self.param_policies[param_name]

        # Then check pattern matches
        for pattern, policy in self._compiled_patterns.items():
            if pattern.match(param_name):
                return policy

        return self.default_policy


def load_weights_with_policies(
    model: nn.Module,
    ckpt: dict[str, torch.Tensor],
    config: WeightLoadingConfig = None,
) -> dict:
    """Load checkpoint weights into model according to the specified configuration.

    Allows for partial loading of weights and zero-initialization of mismatched and arbitrary parameters.

    Args:
        model: The model to load weights INTO. By default, all model weights are re-initialized; we overwrite
            with the checkpoint weights where appropriate
        ckpt: Dictionary mapping parameter names to tensors (loaded from checkpoint on disk)
        config: Configuration for handling weight loading. If None, uses default config

    Returns:
        nn.Module: The model with loaded weights
    """
    if config is None:
        # (Initialize default config if not provided)
        config = WeightLoadingConfig()

    current_state = model.state_dict()
    updated_state = {}  # We will update this with the new weights

    def _apply_policy(
        name: str,
        current_param: torch.Tensor,
        checkpoint_param: torch.Tensor | None,
        policy: WeightLoadingPolicy,
    ) -> torch.Tensor:
        """Apply a weight loading policy and return the resulting tensor.

        Raises WeightLoadingError for any policy application failures.
        """
        if policy == WeightLoadingPolicy.REINIT:
            # Keep original initialization
            return current_param

        elif policy == WeightLoadingPolicy.ZERO_INIT:
            # Zero-initialize
            return torch.zeros_like(current_param)

        elif policy == WeightLoadingPolicy.COPY:
            # Must have checkpoint param and shapes must match
            if checkpoint_param is None:
                raise WeightLoadingError(f"Parameter '{name}' not found in checkpoint")
            if current_param.shape != checkpoint_param.shape:
                raise WeightLoadingError(
                    f"Shape mismatch for '{name}': model {current_param.shape} vs checkpoint {checkpoint_param.shape}"
                )
            return checkpoint_param

        elif policy == WeightLoadingPolicy.COPY_AND_ZERO_PAD:
            # Must have checkpoint param and same number of dimensions
            if checkpoint_param is None:
                raise WeightLoadingError(f"Parameter '{name}' not found in checkpoint")
            if len(current_param.shape) != len(checkpoint_param.shape):
                raise WeightLoadingError(
                    f"Different dimensions for '{name}': model {len(current_param.shape)}D vs checkpoint {len(checkpoint_param.shape)}D"
                )

            # Copy where shapes match, zero-init the rest
            new_param = torch.zeros_like(current_param)
            slices = tuple(
                slice(0, min(d_ckpt, d_current))
                for d_ckpt, d_current in zip(
                    checkpoint_param.shape, current_param.shape
                )
            )
            new_param[slices] = checkpoint_param[slices]
            return new_param

    # ... loop through all named parameters in the model
    for name, current_param in current_state.items():
        # Get the policy for this parameter
        policy = config.get_policy(name)

        # Get the corresponding parameter from the checkpoint
        checkpoint_param = ckpt.get(name, None)

        try:
            # Try to apply the primary policy
            result = _apply_policy(name, current_param, checkpoint_param, policy)
            updated_state[name] = result
        except WeightLoadingError as e:
            # Primary policy failed, try fallback
            ranked_logger.warning(
                f"Failed to apply policy: '{policy}' to '{name}': {str(e)}. Falling back to policy: '{config.fallback_policy}'."
            )
            result = _apply_policy(
                name, current_param, checkpoint_param, config.fallback_policy
            )
            updated_state[name] = result

    return updated_state


@dataclass
class CheckpointConfig:
    """Configuration for loading checkpoints.

    TODO: Implement reset_scheduler and reset_ema
    """

    path: PathLike
    reset_optimizer: bool = False
    weight_loading_config: WeightLoadingConfig | None = None
