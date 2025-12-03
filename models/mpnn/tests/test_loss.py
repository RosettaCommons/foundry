"""
Loss function tests for MPNN models.

This module contains tests specifically focused on loss function behavior
and validation for both ProteinMPNN and LigandMPNN models.
"""

import pytest
from atomworks.ml.utils.testing import cached_parse
from mpnn.pipelines.mpnn import build_mpnn_transform_pipeline
from test_utils import (
    PDB_IDS,
    assert_gradient_properties,
    assert_loss_output_structure,
    compute_loss_and_backward,
    create_feature_collator,
    prepare_features,
    select_model,
)


class TestLoss:
    """Test suite for MPNN loss functions."""

    @pytest.mark.parametrize("pdb_id", PDB_IDS)
    @pytest.mark.parametrize("model_type", ["protein_mpnn", "ligand_mpnn"])
    @pytest.mark.parametrize("is_inference", [False, True])
    def test_loss_function(self, pdb_id, model_type, is_inference, loss_fn):
        """Test that the loss function works correctly for both protein and ligand models."""
        # Load structure and apply pipeline.
        data = cached_parse(pdb_id)
        pipeline = build_mpnn_transform_pipeline(
            model_type=model_type, is_inference=is_inference
        )
        pipeline_output = pipeline(data)

        # Override repeat_sample_num for testing
        prepare_features(pipeline_output["input_features"], repeat_sample_num=2)

        # Collator is used to batch the data.
        collator = create_feature_collator()
        network_input = collator([pipeline_output])

        # Select model
        model = select_model(model_type)

        # Forward pass
        network_output = model(network_input)

        # Compute loss and check its properties and output structure
        loss, loss_output = compute_loss_and_backward(
            model, network_input, network_output, loss_fn
        )

        # Test gradient properties
        assert_gradient_properties(model)

        assert_loss_output_structure(loss, loss_output, network_input)
