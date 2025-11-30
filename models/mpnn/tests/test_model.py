"""
Unit tests for MPNN models.

This module contains unit tests specifically focused on testing the core
functionality of ProteinMPNN and LigandMPNN models, including forward/backward
passes, different configurations, and model initialization.
"""

import pytest
import torch
from atomworks.ml.utils.testing import cached_parse
from mpnn.pipelines.mpnn import build_mpnn_transform_pipeline
from test_utils import (
    PDB_IDS,
    assert_comprehensive_shapes,
    assert_gradient_properties,
    assert_value_checks,
    compute_loss_and_backward,
    create_feature_collator,
    create_symmetry_equivalence_group,
    prepare_features,
    select_model,
)


class TestMPNNModel:
    """Comprehensive test suite for MPNN models."""

    @pytest.mark.parametrize("pdb_id", PDB_IDS)
    @pytest.mark.parametrize("is_inference", [False, True])
    @pytest.mark.parametrize("model_type", ["protein_mpnn", "ligand_mpnn"])
    def test_model_forward_backward_comprehensive(
        self, pdb_id, is_inference, model_type, loss_fn
    ):
        """Test model forward and backward passes with comprehensive shape checking."""
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

        # Test gradient computation for both training and inference modes
        # Gradients should work regardless of the mode for debugging and analysis
        loss, loss_output = compute_loss_and_backward(
            model, network_input, network_output, loss_fn
        )

        # Test gradient properties
        assert_gradient_properties(model)

        # Comprehensive shape assertions
        assert_comprehensive_shapes(
            network_output, network_input["input_features"], model_type
        )
        assert_value_checks(network_output, network_input["input_features"], model_type)

    @pytest.mark.parametrize("pdb_id", PDB_IDS)
    @pytest.mark.parametrize("repeat_sample_num", [1, 3, 5])
    @pytest.mark.parametrize("model_type", ["protein_mpnn", "ligand_mpnn"])
    @pytest.mark.parametrize("is_inference", [False, True])
    def test_repeat_sample_num_shapes(
        self, pdb_id, repeat_sample_num, model_type, is_inference
    ):
        """Test that repeat_sample_num correctly affects batch dimensions."""
        # Load structure and apply pipeline.
        data = cached_parse(pdb_id)
        pipeline = build_mpnn_transform_pipeline(
            model_type=model_type, is_inference=is_inference
        )
        pipeline_output = pipeline(data)

        # Prepare additional features on the fly.
        prepare_features(
            pipeline_output["input_features"], repeat_sample_num=repeat_sample_num
        )

        # Collator is used to batch the data.
        collator = create_feature_collator()
        network_input = collator([pipeline_output])

        # Select model
        model = select_model(model_type)

        # Forward pass
        network_output = model(network_input)

        # Comprehensive shape and value assertions
        assert_comprehensive_shapes(
            network_output, network_input["input_features"], model_type
        )
        assert_value_checks(network_output, network_input["input_features"], model_type)

    @pytest.mark.parametrize("pdb_id", PDB_IDS)
    @pytest.mark.parametrize("model_type", ["protein_mpnn", "ligand_mpnn"])
    @pytest.mark.parametrize("is_inference", [False, True])
    def test_no_repeat_sample_num(self, pdb_id, model_type, is_inference):
        """Test that the model works correctly when repeat_sample_num is None."""
        # Load structure and apply pipeline.
        data = cached_parse(pdb_id)
        pipeline = build_mpnn_transform_pipeline(
            model_type=model_type, is_inference=is_inference
        )
        pipeline_output = pipeline(data)

        # Prepare additional features on the fly.
        prepare_features(pipeline_output["input_features"], repeat_sample_num=None)

        # Collator is used to batch the data.
        collator = create_feature_collator()
        network_input = collator([pipeline_output])

        # Select model
        model = select_model(model_type)

        # Forward pass
        network_output = model(network_input)

        # Comprehensive shape and value assertions
        assert_comprehensive_shapes(
            network_output, network_input["input_features"], model_type
        )
        assert_value_checks(network_output, network_input["input_features"], model_type)

    @pytest.mark.parametrize("pdb_id", PDB_IDS)
    @pytest.mark.parametrize("model_type", ["protein_mpnn", "ligand_mpnn"])
    @pytest.mark.parametrize("is_inference", [False, True])
    def test_features_to_return_specified(
        self, pdb_id, model_type, is_inference, loss_fn
    ):
        """Test model with specific features_to_return."""
        # Load structure and apply pipeline.
        data = cached_parse(pdb_id)
        pipeline = build_mpnn_transform_pipeline(
            model_type=model_type, is_inference=is_inference
        )
        pipeline_output = pipeline(data)

        # Prepare additional features on the fly.
        prepare_features(pipeline_output["input_features"])
        pipeline_output["input_features"]["features_to_return"] = {
            "input_features": ["X", "S", "residue_mask", "mask_for_loss"],
            "graph_features": ["E_idx", "E"],
            "encoder_features": ["h_V", "h_E"],
            "decoder_features": ["logits", "S_sampled", "log_probs"],
        }

        # Collator is used to batch the data.
        collator = create_feature_collator()
        network_input = collator([pipeline_output])

        # Select model
        model = select_model(model_type)

        # Forward pass
        network_output = model(network_input)

        # Test gradient computation
        loss, loss_output = compute_loss_and_backward(
            model, network_input, network_output, loss_fn
        )

        # Test gradient properties
        assert_gradient_properties(model)

        # Assert only specified features are returned
        for key, expected_keys in pipeline_output["input_features"][
            "features_to_return"
        ].items():
            assert key in network_output
            for expected_key in expected_keys:
                assert expected_key in network_output[key]

        # Run value checks on the returned features
        assert_value_checks(network_output, network_input["input_features"], model_type)

    @pytest.mark.parametrize("pdb_id", PDB_IDS)
    @pytest.mark.parametrize("model_type", ["protein_mpnn", "ligand_mpnn"])
    @pytest.mark.parametrize("is_inference", [False, True])
    def test_symmetry_configurations(self, pdb_id, model_type, is_inference, loss_fn):
        """Test model with different symmetry configurations."""
        # Load structure and apply pipeline.
        data = cached_parse(pdb_id)
        pipeline = build_mpnn_transform_pipeline(
            model_type=model_type, is_inference=is_inference
        )
        pipeline_output = pipeline(data)

        # Override repeat_sample_num for testing
        prepare_features(pipeline_output["input_features"], repeat_sample_num=2)

        # Set symmetry configurations after pipeline.
        L = (
            pipeline_output["input_features"]["residue_mask"].shape[0]
            if pipeline_output["input_features"]["residue_mask"].ndim == 1
            else pipeline_output["input_features"]["residue_mask"].shape[1]
        )
        pipeline_output["input_features"]["symmetry_equivalence_group"] = (
            create_symmetry_equivalence_group(L)
        )
        pipeline_output["input_features"]["symmetry_weight"] = (
            torch.ones_like(
                pipeline_output["input_features"]["residue_mask"], dtype=torch.float
            )
            * 1.0
        )

        # Collator is used to batch the data.
        collator = create_feature_collator()
        network_input = collator([pipeline_output])

        # Select model
        model = select_model(model_type)

        # Forward pass
        network_output = model(network_input)

        # Test gradient computation
        loss, loss_output = compute_loss_and_backward(
            model, network_input, network_output, loss_fn
        )

        # Test gradient properties
        assert_gradient_properties(model)
        # Comprehensive shape and value assertions
        assert_comprehensive_shapes(
            network_output, network_input["input_features"], model_type
        )
        assert_value_checks(network_output, network_input["input_features"], model_type)

    @pytest.mark.parametrize("pdb_id", PDB_IDS)
    @pytest.mark.parametrize("model_type", ["protein_mpnn", "ligand_mpnn"])
    @pytest.mark.parametrize("is_inference", [False, True])
    def test_bias_configurations(self, pdb_id, model_type, is_inference, loss_fn):
        """Test model with different bias configurations."""
        # Load structure and apply pipeline.
        data = cached_parse(pdb_id)
        pipeline = build_mpnn_transform_pipeline(
            model_type=model_type, is_inference=is_inference
        )
        pipeline_output = pipeline(data)

        # Override repeat_sample_num for testing
        prepare_features(pipeline_output["input_features"], repeat_sample_num=2)

        # Set bias configurations after pipeline.
        L = pipeline_output["input_features"]["residue_mask"].shape[0]
        pipeline_output["input_features"]["bias"] = (
            torch.randn((L, 21), dtype=torch.float) * 0.5
        )
        pipeline_output["input_features"]["pair_bias"] = (
            torch.randn((L, 21, L, 21), dtype=torch.float) * 0.1
        )

        # Collator is used to batch the data.
        collator = create_feature_collator()
        network_input = collator([pipeline_output])

        # Select model
        model = select_model(model_type)

        # Forward pass
        network_output = model(network_input)

        # Test gradient computation
        loss, loss_output = compute_loss_and_backward(
            model, network_input, network_output, loss_fn
        )

        # Test gradient properties
        assert_gradient_properties(model)

        # Comprehensive shape and value assertions
        assert_comprehensive_shapes(
            network_output, network_input["input_features"], model_type
        )
        assert_value_checks(network_output, network_input["input_features"], model_type)

    @pytest.mark.parametrize("pdb_id", PDB_IDS)
    @pytest.mark.parametrize("model_type", ["protein_mpnn", "ligand_mpnn"])
    @pytest.mark.parametrize("is_inference", [False, True])
    def test_designed_residue_mask_variations(
        self, pdb_id, model_type, is_inference, loss_fn
    ):
        """Test model with different designed_residue_mask configurations."""
        # Load structure and apply pipeline.
        data = cached_parse(pdb_id)
        pipeline = build_mpnn_transform_pipeline(
            model_type=model_type, is_inference=is_inference
        )
        pipeline_output = pipeline(data)

        # Override repeat_sample_num for testing
        prepare_features(pipeline_output["input_features"], repeat_sample_num=2)

        # Set designed residue mask after pipeline.
        pipeline_output["input_features"]["designed_residue_mask"] = torch.ones_like(
            pipeline_output["input_features"]["residue_mask"], dtype=torch.bool
        )

        # Collator is used to batch the data.
        collator = create_feature_collator()
        network_input = collator([pipeline_output])

        # Select model
        model = select_model(model_type)

        # Forward pass
        network_output = model(network_input)

        # Test gradient computation
        loss, loss_output = compute_loss_and_backward(
            model, network_input, network_output, loss_fn
        )

        # Test gradient properties
        assert_gradient_properties(model)

        # Comprehensive shape and value assertions
        assert_comprehensive_shapes(
            network_output, network_input["input_features"], model_type
        )
        assert_value_checks(network_output, network_input["input_features"], model_type)

    @pytest.mark.parametrize("model_type", ["protein_mpnn", "ligand_mpnn"])
    def test_model_initialization(self, model_type):
        """Test that models can be initialized and weights are properly set."""
        model = select_model(model_type)
        for name, param in model.named_parameters():
            if param.dim() > 1:
                assert not torch.isnan(param).any(), f"NaN in {name}"
                assert not torch.isinf(param).any(), f"Inf in {name}"
