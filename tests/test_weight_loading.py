import pytest
import torch
import torch.nn as nn

# Import your code here
from foundry.utils.weights import (
    ParameterFreezingConfig,
    WeightLoadingConfig,
    WeightLoadingPolicy,
    freeze_parameters_with_config,
    load_weights_with_policies,
)


def test_custom_config():
    """Test that a custom config has the expected values."""
    config = WeightLoadingConfig(
        default_policy="zero_init",
        fallback_policy=WeightLoadingPolicy.COPY_AND_ZERO_PAD,
        param_policies={"layer1.weight": "reinit"},
    )
    assert config.default_policy == WeightLoadingPolicy.ZERO_INIT
    assert config.fallback_policy == WeightLoadingPolicy.COPY_AND_ZERO_PAD
    assert config.param_policies == {"layer1.weight": WeightLoadingPolicy.REINIT}


def test_pattern_match_policy():
    """Test that pattern matching works."""
    config = WeightLoadingConfig(
        param_policies={
            "layer1.*": WeightLoadingPolicy.REINIT,
            "*.bias": WeightLoadingPolicy.ZERO_INIT,
        }
    )
    assert config.get_policy("layer1.weight") == WeightLoadingPolicy.REINIT
    assert (
        config.get_policy("layer1.bias") == WeightLoadingPolicy.REINIT
    )  # More specific match
    assert config.get_policy("layer2.bias") == WeightLoadingPolicy.ZERO_INIT
    assert config.get_policy("layer2.weight") == WeightLoadingPolicy.COPY  # Default


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
    # Initialize with non-zero values
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name:
                nn.init.normal_(param, mean=0.0, std=1.0)
            else:
                nn.init.constant_(param, 0.5)
    return model


def test_basic_policies(simple_model):
    """Test basic policies with matching and mismatched shapes."""
    # Create checkpoints
    matching_ckpt = {k: v.clone() + 1.0 for k, v in simple_model.state_dict().items()}
    mismatched_ckpt = {
        "0.weight": torch.randn(15, 10),  # Smaller first dimension
        "0.bias": torch.randn(15),  # Smaller size
        "2.weight": torch.randn(5, 20),  # Matches
        "2.bias": torch.randn(5),  # Matches
    }

    # Test 1: COPY policy with matching shapes
    config1 = WeightLoadingConfig(default_policy=WeightLoadingPolicy.COPY)
    updated_state1 = load_weights_with_policies(simple_model, matching_ckpt, config1)

    # Verify all parameters were copied from checkpoint
    for name in simple_model.state_dict():
        assert torch.allclose(updated_state1[name], matching_ckpt[name])

    # Test 2: ZERO_INIT policy
    config2 = WeightLoadingConfig(default_policy=WeightLoadingPolicy.ZERO_INIT)
    updated_state2 = load_weights_with_policies(simple_model, matching_ckpt, config2)

    # Verify all parameters were zero-initialized
    for name in simple_model.state_dict():
        assert torch.allclose(
            updated_state2[name], torch.zeros_like(updated_state2[name])
        )

    # Test 3: COPY_AND_ZERO_PAD with mismatched shapes
    config3 = WeightLoadingConfig(default_policy=WeightLoadingPolicy.COPY_AND_ZERO_PAD)
    updated_state3 = load_weights_with_policies(simple_model, mismatched_ckpt, config3)

    # Verify padding for mismatched parameters
    assert torch.allclose(
        updated_state3["0.weight"][:15, :], mismatched_ckpt["0.weight"]
    )
    assert torch.allclose(
        updated_state3["0.weight"][15:, :],
        torch.zeros_like(updated_state3["0.weight"][15:, :]),
    )
    assert torch.allclose(updated_state3["0.bias"][:15], mismatched_ckpt["0.bias"])
    assert torch.allclose(
        updated_state3["0.bias"][15:], torch.zeros_like(updated_state3["0.bias"][15:])
    )

    # Verify direct copying for matched parameters
    assert torch.allclose(updated_state3["2.weight"], mismatched_ckpt["2.weight"])
    assert torch.allclose(updated_state3["2.bias"], mismatched_ckpt["2.bias"])


def test_mixed_policies_and_fallbacks(simple_model):
    """Test mixed policies and fallback behavior."""
    # Create a checkpoint with mismatches and missing parameters
    checkpoint = {
        "0.weight": torch.randn(15, 10),  # Mismatched shape
        # "0.bias" is missing
        "2.weight": torch.randn(5, 20, 1),  # Different dimensions (3D vs 2D)
        "2.bias": torch.randn(5),  # Matches
    }

    # Create config with mixed policies
    config = WeightLoadingConfig(
        default_policy=WeightLoadingPolicy.COPY,
        fallback_policy=WeightLoadingPolicy.ZERO_INIT,
        param_policies={
            "0.weight": WeightLoadingPolicy.COPY_AND_ZERO_PAD,
            "0.bias": WeightLoadingPolicy.REINIT,
        },
    )

    updated_state = load_weights_with_policies(simple_model, checkpoint, config)

    # Check padding for 0.weight
    assert torch.allclose(updated_state["0.weight"][:15, :], checkpoint["0.weight"])
    assert torch.allclose(
        updated_state["0.weight"][15:, :],
        torch.zeros_like(updated_state["0.weight"][15:, :]),
    )

    # Check reinit for 0.bias (missing in checkpoint but policy is REINIT)
    assert torch.allclose(updated_state["0.bias"], simple_model.state_dict()["0.bias"])

    # Check fallback to ZERO_INIT for 2.weight (dimension mismatch)
    assert torch.allclose(
        updated_state["2.weight"],
        torch.zeros_like(simple_model.state_dict()["2.weight"]),
    )

    # Check direct copy for 2.bias
    assert torch.allclose(updated_state["2.bias"], checkpoint["2.bias"])


def test_freeze_parameters_by_name_and_pattern(simple_model):
    """Test freezing parameters by exact name and pattern."""
    # Get parameter names
    param_names = list(simple_model.state_dict().keys())
    # Freeze only the first parameter by exact name
    config1 = ParameterFreezingConfig(param_policies={param_names[0]: True})
    freeze_parameters_with_config(simple_model, config1)
    for name, param in simple_model.named_parameters():
        if name == param_names[0]:
            assert not param.requires_grad  # frozen
        else:
            assert param.requires_grad  # not frozen

    # Freeze all bias parameters using pattern
    config2 = ParameterFreezingConfig(param_policies={"*.bias": True})
    freeze_parameters_with_config(simple_model, config2)
    for name, param in simple_model.named_parameters():
        if name.endswith("bias"):
            assert not param.requires_grad
        else:
            assert param.requires_grad

    # Freeze all parameters by default
    config3 = ParameterFreezingConfig(freeze_by_default=True)
    freeze_parameters_with_config(simple_model, config3)
    for _, param in simple_model.named_parameters():
        assert not param.requires_grad

    # Unfreeze all parameters by default
    config4 = ParameterFreezingConfig(freeze_by_default=False)
    freeze_parameters_with_config(simple_model, config4)
    for _, param in simple_model.named_parameters():
        assert param.requires_grad


def test_load_weights_with_freezing(simple_model):
    """Test that load_weights_with_policies can freeze parameters after loading."""
    # Create a checkpoint with matching shapes
    ckpt = {k: v.clone() + 1.0 for k, v in simple_model.state_dict().items()}
    # Freeze all weights
    freezing_config = ParameterFreezingConfig(freeze_by_default=True)
    config = WeightLoadingConfig(default_policy=WeightLoadingPolicy.COPY)
    _ = load_weights_with_policies(simple_model, ckpt, config)
    freeze_parameters_with_config(simple_model, freezing_config)
    for _, param in simple_model.named_parameters():
        assert not param.requires_grad

    # Freeze only biases using pattern
    freezing_config2 = ParameterFreezingConfig(param_policies={"*.bias": True})
    _ = load_weights_with_policies(simple_model, ckpt, config)
    freeze_parameters_with_config(simple_model, freezing_config2)
    for name, param in simple_model.named_parameters():
        if name.endswith("bias"):
            assert not param.requires_grad
        else:
            assert param.requires_grad


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
