"""
Collation utilities for PyTorch data loading.

This module provides collation functions for batching examples with
variable-length features, including padding and scalar feature
consistency checks.
"""

import copy
from typing import Any, Dict, List

import torch
from atomworks.constants import UNKNOWN_AA
from mpnn.transforms.feature_aggregation.token_encodings import MPNN_TOKEN_ENCODING

MPNN_DEFAULT_PADDING = {
    # Tensor features that require padding
    "X": 0.0,
    "X_m": False,
    "S": MPNN_TOKEN_ENCODING.token_to_idx[UNKNOWN_AA],
    "R_idx": -100,
    "chain_labels": -1,
    "residue_mask": False,
    "Y": 0.0,
    "Y_m": 0,
    "Y_t": 0,
    "designed_residue_mask": False,
    "symmetry_equivalence_group": -1,
    "symmetry_weight": 0,
    "bias": 0,
    "pair_bias": 0,
    "temperature": 1,
}


class FeatureCollator:
    """
    Generic PyTorch collation class for handling variable-length features with
    padding.

    This collator processes batches of examples where each example
    contains features that may have different shapes (requiring padding) or
    scalar values (requiring consistency checks across the batch).
    """

    def __init__(self, default_padding: Dict[str, Any] = None):
        """
        Initialize the FeatureCollator.

        Args:
            default_padding (Dict[str, Any], optional): Maps each feature key
                to a scalar or tensor to use for padding/missing values. Each
                value should be appropriate for the expected feature type
                (e.g., 0 for missing integers, 0.0 for missing floats, or a
                tensor for missing tensor features). If None, uses
                MPNN-specific defaults.
        """
        if default_padding is None:
            default_padding = MPNN_DEFAULT_PADDING
        self.default_padding = default_padding

    def __call__(self, pipeline_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of pipeline outputs into network inputs.

        Args:
            pipeline_outputs (List[Dict[str, Any]]): List of examples, each
                example is a dict with keys "input_features" (feature dict)
                and "atom_array" (raw atom array).

        Returns:
            Dict[str, Any]: dict with keys 'input_features' (collated features)
                and 'atom_array' (list of atom_arrays from each example).

        Raises:
            ValueError: If pipeline_outputs is empty, if feature values differ
                across examples, or if feature types are unsupported or
                mismatched.
        """
        # Raise error if batch is empty.
        if not pipeline_outputs:
            raise ValueError("Cannot collate empty batch")

        # Only collate keys present in the first example's input_features.
        keys_to_collate = pipeline_outputs[0]["input_features"].keys()
        input_features = {}
        for key in keys_to_collate:
            # Assert that each key exists in every example's input_features.
            for i, example in enumerate(pipeline_outputs):
                if key not in example["input_features"]:
                    raise ValueError(
                        f"Feature '{key}' not found in " + f"example {i} input_features"
                    )

            # Extract values for this feature from all examples.
            values = [example["input_features"][key] for example in pipeline_outputs]
            tensor_mask = [isinstance(v, torch.Tensor) for v in values]
            if all(tensor_mask):
                # Validate that all tensors have the same number of dimensions.
                ndims = [v.ndim for v in values]
                if len(set(ndims)) != 1:
                    raise ValueError(
                        f"Tensors for feature '{key}' have "
                        + f"mismatched dimensions: {ndims}"
                    )

                ndim = ndims[0]

                # Handle tensor collation and padding.
                if ndim >= 1:
                    # Check if lengths along dimension 0 vary.
                    shapes_dim0 = [v.shape[0] for v in values]
                    if len(set(shapes_dim0)) > 1:
                        # Padding required - lengths vary along dim 0.
                        if key not in self.default_padding:
                            raise ValueError(
                                "No default padding value for feature "
                                + f"'{key}' required for tensor padding."
                            )
                        pad_val = self.default_padding[key]
                        max_L = max(shapes_dim0)
                        B = len(values)

                        # Verify that all other dimensions are identical.
                        other_shapes = [v.shape[1:] for v in values]
                        if len(set(other_shapes)) != 1:
                            raise ValueError(
                                f"Tensors for feature '{key}' "
                                + "have mismatched shapes beyond dimension 0:"
                                + f" {[v.shape for v in values]}"
                            )

                        # Create padded tensor.
                        shape = (B, max_L, *values[0].shape[1:])
                        padded = torch.full(
                            shape,
                            pad_val,
                            dtype=values[0].dtype,
                            device=values[0].device,
                        )

                        # Copy actual values into padded tensor.
                        for i, v in enumerate(values):
                            padded[i, : v.shape[0]] = v
                        input_features[key] = padded
                    else:
                        input_features[key] = torch.stack(values, dim=0)
                else:
                    input_features[key] = torch.stack(values, dim=0)
            else:
                # Check that all values are the same type.
                first_type = type(values[0])
                if not all(isinstance(v, first_type) for v in values):
                    raise ValueError(
                        f"Feature '{key}' has mismatched "
                        + f"types: {[type(v).__name__ for v in values]}"
                    )

                # Deep comparison for all values.
                first_val = values[0]
                if all(self._deep_equal(first_val, v) for v in values):
                    input_features[key] = copy.deepcopy(first_val)
                else:
                    raise ValueError(
                        f"Feature '{key}' differs across examples: {values}"
                    )
        atom_arrays = [example["atom_array"] for example in pipeline_outputs]

        network_inputs = {"input_features": input_features, "atom_array": atom_arrays}

        return network_inputs

    def _deep_equal(self, a, b):
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            return torch.equal(a, b)
        if isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                return False
            return all(self._deep_equal(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            return len(a) == len(b) and all(
                self._deep_equal(x, y) for x, y in zip(a, b)
            )
        return a == b


class TokenBudgetAwareFeatureCollator(FeatureCollator):
    """
    Feature collator that enforces token budget constraints before collation.

    This collator sorts pipeline outputs by sequence length and removes the
    largest examples first if the batch would exceed the token budget when
    padded to the maximum length.

    Args:
        max_tokens_with_padding: Maximum number of tokens allowed per batch,
            including padding. The constraint is
            max(batch_lengths) * len(batch) <= max_tokens.
        default_padding: Default padding values for features.
    """

    def __init__(
        self, max_tokens_with_padding: int, default_padding: Dict[str, Any] = None
    ):
        super().__init__(default_padding)
        self.max_tokens_with_padding = max_tokens_with_padding

        if max_tokens_with_padding <= 0:
            raise ValueError("max_tokens_with_padding must be greater than 0")

    def __call__(self, pipeline_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate pipeline outputs while respecting token budget constraints.

        Sorts examples by length and removes largest examples first if needed
        to stay within the token budget.

        Args:
            pipeline_outputs: List of pipeline output dictionaries.

        Returns:
            Dict containing collated features and atom arrays.

        Raises:
            ValueError: If pipeline_outputs is empty.
        """
        if not pipeline_outputs:
            raise ValueError("Cannot collate empty batch")

        # Extract lengths.
        examples_with_L = []
        for example in pipeline_outputs:
            L = example["input_features"]["S"].shape[0]
            examples_with_L.append((L, example))

        # Sort by length (ascending).
        examples_with_L.sort(key=lambda x: x[0])

        # Apply token budget constraint by removing largest examples first.
        filtered_examples = []
        max_length = 0
        for L, example in examples_with_L:
            new_batch_size = len(filtered_examples) + 1
            potential_max_length = max(L, max_length)

            if potential_max_length * new_batch_size > self.max_tokens_with_padding:
                # Adding this example would violate constraint, stop here.
                break

            filtered_examples.append((L, example))
            max_length = potential_max_length

        # Extract just the examples (without lengths) for collation.
        filtered_pipeline_outputs = [example for _, example in filtered_examples]

        # If no examples remain after filtering, raise an error
        if not filtered_pipeline_outputs:
            raise ValueError(
                "No examples remain after applying token budget constraint. "
                "All examples exceed max_tokens_with_padding="
                f"{self.max_tokens_with_padding}"
            )

        # Call parent collation on filtered examples
        return super().__call__(filtered_pipeline_outputs)
