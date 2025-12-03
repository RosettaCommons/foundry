"""
Metric tests for MPNN models.

This module contains tests specifically focused on testing the interface metric
classes including InterfaceSequenceRecovery and InterfaceNLL.
"""

import pytest
from atomworks.ml.utils.testing import cached_parse
from mpnn.metrics.nll import NLL, InterfaceNLL
from mpnn.metrics.sequence_recovery import InterfaceSequenceRecovery, SequenceRecovery
from mpnn.pipelines.mpnn import build_mpnn_transform_pipeline
from test_utils import (
    PDB_IDS,
    assert_all_metrics_comprehensive,
    combine_kwargs_to_compute,
    create_feature_collator,
    prepare_features,
    select_model,
)


class TestMetrics:
    """Test suite for MPNN metric functions."""

    @pytest.mark.parametrize("pdb_id", PDB_IDS)
    @pytest.mark.parametrize("model_type", ["protein_mpnn", "ligand_mpnn"])
    @pytest.mark.parametrize("is_inference", [False, True])
    def test_metrics_comprehensive(self, pdb_id, model_type, is_inference):
        """Test that the metrics work correctly for both protein and ligand models."""
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
