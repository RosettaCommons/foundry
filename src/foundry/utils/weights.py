"""Utils for loading weights from checkpoints."""

import re
from dataclasses import dataclass, field
from enum import StrEnum, auto
from os import PathLike

import torch
from beartype.typing import Pattern
from torch import nn

from foundry.utils.ddp import RankedLogger

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


class _PatternPolicyMixin:
    """Mixin for handling glob-to-regex pattern compilation and matching for parameter policies.

    Patterns can use the following wildcards:
        - * matches any sequence of characters
        - ? matches any single character
        - [abc] matches any character in the brackets
        - . matches a literal dot

    Examples:
        - "model.*.weight" matches any weight parameter in the model
        - "model.encoder?.weight" matches encoder1.weight, encoder2.weight, etc.
        - "model.encoder[12].weight" matches encoder1.weight and encoder2.weight
        - "model.encoder.*.bias" matches any bias parameter in encoder submodules
    """

    _compiled_patterns: dict[Pattern, any]

    @staticmethod
    def _glob_to_regex(pattern: str) -> str:
        # Convert glob pattern to regex string
        return (
            pattern.replace(".", r"\.")
            .replace("*", ".*")
            .replace("?", ".")
            .replace("[", "[")
            .replace("]", "]")
        )

    def _compile_patterns(self, policy_dict: dict[str, any]) -> dict[Pattern, any]:
        compiled = {}
        for pattern, value in list(policy_dict.items()):
            if any(c in pattern for c in ["*", "?", "[", "]"]):
                regex = self._glob_to_regex(pattern)
                compiled[re.compile(f"^{regex}$")] = value
        return compiled

    def _get_policy_by_pattern(
        self, param_name: str, policy_dict: dict[str, any], default: any
    ) -> any:
        # Exact match first
        if policy_dict and param_name in policy_dict:
            return policy_dict[param_name]
        # Pattern match
        for pattern, value in self._compiled_patterns.items():
            if pattern.match(param_name):
                return value
        return default


@dataclass
class WeightLoadingConfig(_PatternPolicyMixin):
    """Configuration for handling weights when loading a checkpoint."""

    default_policy: WeightLoadingPolicy | str = WeightLoadingPolicy.COPY
    fallback_policy: WeightLoadingPolicy | str = WeightLoadingPolicy.REINIT
    param_policies: dict[str, WeightLoadingPolicy | str] = field(default_factory=dict)
    _compiled_patterns: dict[Pattern, WeightLoadingPolicy] = field(
        default_factory=dict, repr=False
    )

    def __post_init__(self):
        if isinstance(self.default_policy, str):
            self.default_policy = WeightLoadingPolicy(self.default_policy)
        if isinstance(self.fallback_policy, str):
            self.fallback_policy = WeightLoadingPolicy(self.fallback_policy)
        for key, value in self.param_policies.items():
            if isinstance(value, str):
                self.param_policies[key] = WeightLoadingPolicy(value)
        self._compiled_patterns = self._compile_patterns(self.param_policies)

    def get_policy(self, param_name: str) -> WeightLoadingPolicy:
        policy = self._get_policy_by_pattern(
            param_name, self.param_policies, self.default_policy
        )
        assert isinstance(policy, WeightLoadingPolicy)
        return policy


@dataclass
class ParameterFreezingConfig(_PatternPolicyMixin):
    """Configuration for freezing model parameters after loading weights.

    Allows specifying which parameters to freeze (set requires_grad=False) by exact name or pattern.
    Patterns use glob-style wildcards (*, ?).

    Attributes:
        param_policies: Dict mapping parameter names or patterns to True (freeze) or False (do not freeze).
        freeze_by_default: Whether to freeze parameters not matched by any pattern (default: False).
    """

    param_policies: dict[str, bool] = field(default_factory=dict)
    freeze_by_default: bool = False
    _compiled_patterns: dict[Pattern, bool] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._compiled_patterns = self._compile_patterns(self.param_policies)

    def is_frozen(self, param_name: str) -> bool:
        """Get whether a parameter is frozen according to the config."""
        is_frozen = self._get_policy_by_pattern(
            param_name, self.param_policies, self.freeze_by_default
        )
        assert isinstance(is_frozen, bool)
        return is_frozen


def freeze_parameters_with_config(
    model: nn.Module, config: ParameterFreezingConfig, verbose: bool = True
) -> None:
    """Freeze (set requires_grad=False) or unfreeze parameters according to config.

    Args:
        model: The model whose parameters to freeze/unfreeze.
        config: ParameterFreezingConfig specifying which parameters to freeze.
        verbose: Whether to log which parameters have non-default policies applied.
    """
    for name, param in model.named_parameters():
        is_frozen = config.is_frozen(name)
        param.requires_grad = not is_frozen

        if is_frozen != config.freeze_by_default and verbose:
            ranked_logger.info(f"Non-default freezing applied to {name}: {is_frozen}")


def load_weights_with_policies(
    model: nn.Module,
    ckpt: dict[str, torch.Tensor],
    config: WeightLoadingConfig | None = None,
) -> dict:
    """Load checkpoint weights into model according to the specified configuration.

    Allows for partial loading of weights and zero-initialization of mismatched and arbitrary parameters.

    Args:
        model: The model to load weights INTO. By default, all model weights are re-initialized; we overwrite
            with the checkpoint weights where appropriate
        ckpt: Dictionary mapping parameter names to tensors (loaded from checkpoint on disk)
        config: Configuration for handling weight loading. If None, uses default config
    Returns:
        dict: The updated state_dict (not loaded into model yet)
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
    parameter_freezing_config: ParameterFreezingConfig | None = None
