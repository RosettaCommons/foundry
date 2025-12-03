"""
Unit tests for the FeatureCollator class.

This module tests the core functionality of the FeatureCollator including
tensor padding, scalar feature consistency, and error handling for unsupported
types.
"""

import pytest
import torch
from mpnn.collate.feature_collator import FeatureCollator


class TestFeatureCollator:
    """Unit test suite for the FeatureCollator class."""

    def test_empty_batch_error(self):
        """Test that empty batch raises the correct ValueError."""
        collator = FeatureCollator(default_padding={"feature": 0})
        # Should raise ValueError for empty input
        with pytest.raises(ValueError, match="Cannot collate empty batch"):
            collator([])

    def test_tensor_padding_behavior(self):
        """Test padding behavior: batch two examples of different lengths and verify padding."""
        pad_val = -1.0
        collator = FeatureCollator(default_padding={"sequences": pad_val})

        # Create examples with different sequence lengths
        example1 = {
            "input_features": {
                "sequences": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            },
            "atom_array": "atom1",
        }
        example2 = {
            "input_features": {"sequences": torch.tensor([[7.0, 8.0, 9.0]])},
            "atom_array": "atom2",
        }
        pipeline_outputs = [example1, example2]
        result = collator(pipeline_outputs)

        # Check output shape and padding
        expected_shape = (2, 2, 3)  # (batch_size, max_len, feature_dim)
        assert result["input_features"]["sequences"].shape == expected_shape

        # Check that original values are preserved
        assert torch.equal(
            result["input_features"]["sequences"][0],
            example1["input_features"]["sequences"],
        )
        assert torch.equal(
            result["input_features"]["sequences"][1, :1],
            example2["input_features"]["sequences"],
        )

        # Check that padding values are correct - the shorter sequence should be padded
        assert torch.all(result["input_features"]["sequences"][1, 1:] == pad_val)

    def test_scalar_feature_consistency_valid(self):
        """Test that identical scalar values across examples are handled correctly."""
        collator = FeatureCollator(
            default_padding={"batch_size": 32, "learning_rate": 0.001}
        )
        # All examples have the same scalar values
        example1 = {
            "input_features": {"batch_size": 32, "learning_rate": 0.001},
            "atom_array": 1,
        }
        example2 = {
            "input_features": {"batch_size": 32, "learning_rate": 0.001},
            "atom_array": 2,
        }
        pipeline_outputs = [example1, example2]
        result = collator(pipeline_outputs)

        # Should preserve scalar features.
        assert result["input_features"]["batch_size"] == 32
        assert result["input_features"]["learning_rate"] == 0.001

    def test_scalar_feature_consistency_error(self):
        """Test that differing scalar values trigger the correct ValueError."""
        collator = FeatureCollator(default_padding={"inconsistent_scalar": 0})

        # Examples with different scalar values
        example1 = {"input_features": {"inconsistent_scalar": 5}, "atom_array": 1}
        example2 = {"input_features": {"inconsistent_scalar": 10}, "atom_array": 2}
        pipeline_outputs = [example1, example2]

        # Should raise ValueError for inconsistent scalars
        with pytest.raises(
            ValueError, match="Feature 'inconsistent_scalar' differs across examples"
        ):
            collator(pipeline_outputs)

    def test_disallowed_mixed_types_error(self):
        """Test that mixed feature types raise the correct error."""
        collator = FeatureCollator(default_padding={"mixed_feature": 0})
        # Mix tensor and scalar
        example1 = {
            "input_features": {"mixed_feature": torch.tensor([1, 2, 3])},
            "atom_array": 1,
        }
        example2 = {"input_features": {"mixed_feature": 42}, "atom_array": 2}
        pipeline_outputs = [example1, example2]

        # Should raise ValueError for mixed types
        with pytest.raises(
            ValueError, match=r"mismatched types|differs across examples"
        ):
            collator(pipeline_outputs)

    def test_disallowed_unsupported_types_error(self):
        """Test that unsupported feature types raise the correct error."""
        collator = FeatureCollator(default_padding={"unsupported_feature": []})
        # Use unsupported types (lists)
        example1 = {
            "input_features": {"unsupported_feature": [1, 2, 3]},
            "atom_array": 1,
        }
        example2 = {
            "input_features": {"unsupported_feature": [4, 5, 6]},
            "atom_array": 2,
        }
        pipeline_outputs = [example1, example2]

        # Should raise ValueError for unsupported types
        with pytest.raises(
            ValueError, match=r"mismatched types|differs across examples"
        ):
            collator(pipeline_outputs)

    def test_missing_features_raise_error(self):
        """Test that missing features raise the correct ValueError."""
        collator = FeatureCollator(default_padding={"required_feature": 0})
        # One example missing the required feature
        example1 = {
            "input_features": {"required_feature": torch.tensor([1, 2, 3])},
            "atom_array": 1,
        }
        example2 = {"input_features": {}, "atom_array": 2}
        pipeline_outputs = [example1, example2]
        # Should raise ValueError for missing feature
        with pytest.raises(
            ValueError,
            match="Feature 'required_feature' not found in example 1 input_features",
        ):
            collator(pipeline_outputs)

    def test_device_and_dtype_preservation(self):
        """Test that tensor device and dtype are preserved in padded outputs."""
        collator = FeatureCollator(default_padding={"feature": 0.0})
        # Tensors with the same shape and specific dtype
        example1 = {
            "input_features": {
                "feature": torch.tensor([[1.0, 2.0]], dtype=torch.float64)
            },
            "atom_array": 1,
        }
        example2 = {
            "input_features": {
                "feature": torch.tensor([[3.0, 4.0]], dtype=torch.float64)
            },
            "atom_array": 2,
        }
        pipeline_outputs = [example1, example2]
        result = collator(pipeline_outputs)

        # Should preserve dtype and device
        assert result["input_features"]["feature"].dtype == torch.float64
        assert (
            result["input_features"]["feature"].device
            == example1["input_features"]["feature"].device
        )
