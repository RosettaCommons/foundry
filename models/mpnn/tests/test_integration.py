"""
Integration tests for the MPNN model components.

This module contains tests that verify the integration between different
components of the MPNN system including feature collation and end-to-end
model functionality.
"""

import pytest
import torch
from atomworks.ml.utils.testing import cached_parse
from mpnn.metrics.nll import NLL, InterfaceNLL
from mpnn.metrics.sequence_recovery import InterfaceSequenceRecovery, SequenceRecovery
from mpnn.pipelines.mpnn import build_mpnn_transform_pipeline
from mpnn.trainers.mpnn import MPNNTrainer
from omegaconf import DictConfig
from test_utils import (
    PDB_IDS,
    assert_all_metrics_comprehensive,
    assert_comprehensive_shapes,
    assert_gradient_properties,
    assert_loss_output_structure,
    assert_value_checks,
    combine_kwargs_to_compute,
    compute_loss_and_backward,
    create_feature_collator,
    select_model,
)


# Create a simple dataset from our batch
class SingleBatchDataset(torch.utils.data.Dataset):
    def __init__(self, batch):
        self.batch = batch

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.batch


class TestIntegration:
    """Test integration between FeatureCollator, MPNN models, and metrics."""

    @pytest.mark.parametrize("model_type", ["protein_mpnn", "ligand_mpnn"])
    @pytest.mark.parametrize("is_inference", [False, True])
    def test_collator(self, model_type, is_inference, loss_fn):
        """Test FeatureCollator integration with both ProteinMPNN and LigandMPNN, including comprehensive metric testing."""
        examples = []

        # Load and process each example
        for pdb_id in PDB_IDS:
            # Load structure and apply pipeline.
            data = cached_parse(pdb_id)
            pipeline = build_mpnn_transform_pipeline(
                model_type=model_type, is_inference=is_inference
            )
            pipeline_output = pipeline(data)
            examples.append(pipeline_output)

        # Collate all examples
        collator = create_feature_collator()
        network_input = collator(examples)

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

        # Test model output shapes and values
        assert_comprehensive_shapes(
            network_output, network_input["input_features"], model_type
        )
        assert_value_checks(network_output, network_input["input_features"], model_type)
        assert_loss_output_structure(loss, loss_output, network_input)

        # Test all metrics comprehensively
        # Create all metrics with full return options
        seq_recovery = SequenceRecovery(
            return_per_example_metrics=True, return_per_residue_metrics=True
        )
        interface_seq_recovery = InterfaceSequenceRecovery(
            interface_distance_threshold=5.0,
            return_per_example_metrics=True,
            return_per_residue_metrics=True,
        )
        nll = NLL(return_per_example_metrics=True, return_per_residue_metrics=True)
        interface_nll = InterfaceNLL(
            interface_distance_threshold=5.0,
            return_per_example_metrics=True,
            return_per_residue_metrics=True,
        )

        # Compute all metrics
        seq_metrics = seq_recovery.compute(
            **combine_kwargs_to_compute(seq_recovery, network_input, network_output)
        )
        interface_seq_metrics = interface_seq_recovery.compute(
            **combine_kwargs_to_compute(
                interface_seq_recovery, network_input, network_output
            )
        )
        nll_metrics = nll.compute(
            **combine_kwargs_to_compute(nll, network_input, network_output)
        )
        interface_nll_metrics = interface_nll.compute(
            **combine_kwargs_to_compute(interface_nll, network_input, network_output)
        )

        # Use comprehensive testing function to validate all metrics
        assert_all_metrics_comprehensive(
            seq_metrics,
            nll_metrics,
            interface_seq_metrics,
            interface_nll_metrics,
            network_input,
            return_per_example=True,
            return_per_residue=True,
        )

    @pytest.mark.parametrize("model_type", ["protein_mpnn", "ligand_mpnn"])
    @pytest.mark.parametrize("is_inference", [False, True])
    def test_trainer(self, model_type, is_inference):
        """Test MPNNTrainer using only fit method while extracting comprehensive test information."""
        # Create trainer with minimal configuration for testing
        trainer = MPNNTrainer(
            model_type=model_type,
            accelerator="cpu",
            devices_per_node=1,
            max_epochs=1,
            limit_train_batches=1,
            limit_val_batches=1,
            precision="32",
        )

        # Create minimal train_cfg for optimizer and scheduler construction
        train_cfg = DictConfig(
            {
                "model": {
                    "optimizer": {
                        "_target_": "torch.optim.Adam",
                        "lr": 0.001,
                        "betas": [0.9, 0.999],
                        "eps": 1e-8,
                        "weight_decay": 0.0,
                    },
                    "lr_scheduler": {
                        "_target_": "torch.optim.lr_scheduler.StepLR",
                        "step_size": 1000,
                        "gamma": 0.95,
                    },
                }
            }
        )

        # Initialize trainer state with train_cfg
        trainer.initialize_or_update_trainer_state({"train_cfg": train_cfg})

        # Launch Fabric (this sets up the distributed environment)
        trainer.fabric.launch()

        # Construct model
        trainer.construct_model()

        # Construct optimizer and scheduler
        trainer.construct_optimizer()
        trainer.construct_scheduler()

        # Create test data
        examples = []
        for pdb_id in PDB_IDS:
            data = cached_parse(pdb_id)
            pipeline = build_mpnn_transform_pipeline(
                model_type=model_type, is_inference=is_inference
            )
            pipeline_output = pipeline(data)
            examples.append(pipeline_output)

        # Create dataloader
        collator = create_feature_collator()
        network_input = collator(examples)

        # Create DataLoaders
        train_dataset = SingleBatchDataset(network_input)
        val_dataset = SingleBatchDataset(network_input)

        def collate_fn(batch):
            # Get the batch data and ensure it's on the correct device
            batch_data = batch[0]  # Get the single batch
            return batch_data

        # Create separate DataLoader instances for training and validation
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
        )
        val_loaders = {
            "test_val": torch.utils.data.DataLoader(
                val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
            )
        }

        # Store original methods to restore later
        original_training_step = trainer.training_step
        original_validation_step = trainer.validation_step

        # Variables to capture information during fit
        captured_training_info = {}
        captured_validation_info = {}

        def capture_training_step(batch, batch_idx, is_accumulating):
            """Capture training step information."""
            result = original_training_step(batch, batch_idx, is_accumulating)

            # Capture training information
            if hasattr(trainer, "_current_train_return"):
                captured_training_info["loss"] = trainer._current_train_return.get(
                    "total_loss"
                )
                captured_training_info["loss_dict"] = trainer._current_train_return.get(
                    "loss_dict"
                )
                captured_training_info["batch"] = batch

            # Test gradient properties while gradients are still available
            model = trainer.state["model"]
            assert_gradient_properties(model)

            return result

        def capture_validation_step(batch, batch_idx, compute_metrics=True):
            """Capture validation step information."""
            result = original_validation_step(batch, batch_idx, compute_metrics)

            # Capture validation information
            captured_validation_info["metrics_output"] = result.get("metrics_output")
            captured_validation_info["network_output"] = result.get("network_output")
            captured_validation_info["batch"] = batch

            return result

        # Override methods to capture information
        trainer.training_step = capture_training_step
        trainer.validation_step = capture_validation_step

        # Run the full training using fit method
        trainer.fit(train_loader=train_loader, val_loaders=val_loaders)

        # Extract and test all the information we captured
        model = trainer.state["model"]

        # Test training information
        assert (
            "loss" in captured_training_info
        ), "Training loss should be captured during fit"
        assert (
            "loss_dict" in captured_training_info
        ), "Training loss dict should be captured during fit"
        assert (
            "batch" in captured_training_info
        ), "Training batch should be captured during fit"

        # Test loss output structure
        loss = captured_training_info["loss"]
        loss_output = captured_training_info["loss_dict"]
        batch = captured_training_info["batch"]
        assert_loss_output_structure(loss, loss_output, batch)

        # Test validation information
        assert (
            "metrics_output" in captured_validation_info
        ), "Validation metrics should be captured during fit"
        assert (
            "network_output" in captured_validation_info
        ), "Validation network output should be captured during fit"
        assert (
            "batch" in captured_validation_info
        ), "Validation batch should be captured during fit"

        # Test that metrics were computed
        metrics_output = captured_validation_info["metrics_output"]
        assert metrics_output is not None, "Metrics output should not be None"

        # Test network output structure
        network_output = captured_validation_info["network_output"]
        val_batch = captured_validation_info["batch"]
        assert_comprehensive_shapes(
            network_output, val_batch["input_features"], model_type
        )
        assert_value_checks(network_output, val_batch["input_features"], model_type)

        # Verify model is in training mode after fit
        assert model.training, "Model should be in training mode after fit method"
