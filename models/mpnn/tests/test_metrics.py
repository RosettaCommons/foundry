"""
Metric tests for MPNN models.

This module contains tests specifically focused on testing the interface metric
classes including InterfaceSequenceRecovery and InterfaceNLL.
"""

import pytest
import torch
from atomworks.ml.utils.testing import cached_parse
from mpnn.metrics.nll import NLL, InterfaceNLL, SampledInterfaceNLL, SampledNLL
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

    def test_sampled_confidence_metrics_read_sampled_logits(self):
        """SampledNLL/SampledInterfaceNLL must score the *sampled* sequence
        using the raw model logits (not the native sequence or the
        temperature-scaled log_probs)."""
        for metric in (SampledNLL(), SampledInterfaceNLL()):
            mapping = metric.kwargs_to_compute_args
            assert mapping["S"] == ("network_output", "decoder_features", "S_sampled")
            assert mapping["logits"] == (
                "network_output",
                "decoder_features",
                "logits",
            )
            # The parent's native-sequence log_probs input must not leak through.
            assert "log_probs" not in mapping
        # The interface variant additionally needs the atom array for masking.
        assert SampledInterfaceNLL().kwargs_to_compute_args["atom_array"] == (
            "network_input",
            "atom_array",
        )

    def test_sampled_nll_equals_log_softmax_of_logits_on_sampled_sequence(self):
        """SampledNLL.compute must equal the hand-computed NLL of the sampled
        sequence under log_softmax(logits), and must ignore the native
        sequence."""
        batch, length, vocab = 1, 4, 21
        torch.manual_seed(0)
        logits = torch.randn(batch, length, vocab)
        sampled = torch.tensor([[1, 5, 5, 10]])
        native = torch.tensor([[0, 0, 0, 0]])  # deliberately != sampled
        mask = torch.tensor([[True, True, True, False]])

        network_output = {
            "decoder_features": {"logits": logits, "S_sampled": sampled},
            "input_features": {"mask_for_loss": mask, "S": native},
        }

        metric = SampledNLL(
            return_per_example_metrics=True, return_per_residue_metrics=True
        )
        out = metric.compute_from_kwargs(network_output=network_output)

        # Expected: mean over the (3) masked-in positions of -log_softmax(logits).
        log_probs = torch.log_softmax(logits, dim=-1)
        per_res = -log_probs[0, torch.arange(length), sampled[0]]
        expected_nll = per_res[:3].sum() / 3.0

        assert torch.allclose(out["nll_per_example"][0], expected_nll, atol=1e-6)
        # Per-residue NLL is zeroed at the masked-out position.
        assert torch.allclose(
            out["nll_per_residue"][0],
            torch.tensor([per_res[0], per_res[1], per_res[2], 0.0]),
            atol=1e-6,
        )
        # Must NOT coincide with the NLL of the (different) native sequence.
        native_nll = (-log_probs[0, torch.arange(length), native[0]])[:3].sum() / 3.0
        assert not torch.allclose(out["nll_per_example"][0], native_nll, atol=1e-4)

    def test_sampled_interface_nll_restricts_to_interface_and_uses_sampled(
        self, monkeypatch
    ):
        """SampledInterfaceNLL must restrict the NLL to the interface mask and
        score the sampled sequence under log_softmax(logits). The interface-mask
        derivation itself is inherited from InterfaceNLL (covered by the
        integration test); here the structure-derived mask is injected so the
        new sampled+logits+masking contract can be checked numerically."""
        batch, length, vocab = 1, 4, 21
        torch.manual_seed(1)
        logits = torch.randn(batch, length, vocab)
        sampled = torch.tensor([[2, 7, 3, 9]])
        native = torch.tensor([[0, 0, 0, 0]])  # deliberately != sampled
        mask_for_loss = torch.ones(batch, length, dtype=torch.bool)
        # Pretend only positions 1 and 2 are at the polymer-ligand interface.
        interface_mask = torch.tensor([[False, True, True, False]])

        metric = SampledInterfaceNLL(
            return_per_example_metrics=True, return_per_residue_metrics=True
        )
        # Bypass the structure-derived interface mask with a known one.
        monkeypatch.setattr(
            metric, "get_per_residue_mask", lambda mask_for_loss, **kw: interface_mask
        )

        network_output = {
            "decoder_features": {"logits": logits, "S_sampled": sampled},
            "input_features": {"mask_for_loss": mask_for_loss, "S": native},
        }
        network_input = {"atom_array": None}  # unused; mask is injected

        out = metric.compute_from_kwargs(
            network_input=network_input, network_output=network_output
        )

        log_probs = torch.log_softmax(logits, dim=-1)
        per_res = -log_probs[0, torch.arange(length), sampled[0]]
        # Only the two interface positions contribute.
        expected = (per_res[1] + per_res[2]) / 2.0
        assert torch.allclose(out["interface_nll_per_example"][0], expected, atol=1e-6)
        # Non-interface positions are excluded from the per-residue NLL.
        assert out["interface_nll_per_residue"][0, 0] == 0.0
        assert out["interface_nll_per_residue"][0, 3] == 0.0
        # Must NOT coincide with the native sequence over the same mask.
        native_res = -log_probs[0, torch.arange(length), native[0]]
        native_expected = (native_res[1] + native_res[2]) / 2.0
        assert not torch.allclose(
            out["interface_nll_per_example"][0], native_expected, atol=1e-4
        )
