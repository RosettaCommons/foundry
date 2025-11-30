"""
Shared test utilities for MPNN tests.

This module contains common fixtures, helper functions, and shared logic
used across multiple test files in the MPNN test suite.
"""

import pytest
import torch
from mpnn.collate.feature_collator import FeatureCollator
from mpnn.loss.nll_loss import LabelSmoothedNLLLoss
from mpnn.model.mpnn import LigandMPNN, ProteinMPNN

# Shared test constants
PDB_IDS = ["6eb6", "7tdx", "3en2", "2xni"]


# Shared fixtures
@pytest.fixture
def loss_fn():
    return LabelSmoothedNLLLoss()


def protein_model():
    model = ProteinMPNN()
    ProteinMPNN.init_weights(model)
    return model


def ligand_model():
    model = LigandMPNN()
    LigandMPNN.init_weights(model)
    return model


def select_model(model_type):
    if model_type == "protein_mpnn":
        return protein_model()
    elif model_type == "ligand_mpnn":
        return ligand_model()
    else:
        raise ValueError(f"Invalid model type: {model_type}")


# Shared utility functions
def create_symmetry_equivalence_group(L, num_groups=50):
    """Create symmetry equivalence groups for testing."""
    group_size = max(1, L // num_groups)
    symmetry_group = torch.arange(L, dtype=torch.int64)
    for i in range(L):
        symmetry_group[i] = i // group_size

    return symmetry_group


def prepare_features(input_features, repeat_sample_num=None):
    """Prepare features for model input with all required settings."""
    if repeat_sample_num is not None:
        input_features["repeat_sample_num"] = repeat_sample_num


def get_expected_dimensions(model_type):
    """Get expected dimensions for different model types."""
    dims = {
        "vocab_size": 21,
        "hidden_dim": 128,
        "num_atoms": 37,
        "num_backbone_atoms": 4,
        "num_virtual_atoms": 1,
        "num_rep_atoms": 1,
        "num_edge_output_features": 128,
        "num_node_output_features": 128,
    }

    if model_type == "protein_mpnn":
        dims["K"] = 48  # num_neighbors for ProteinMPNN
    elif model_type == "ligand_mpnn":
        dims["K"] = 32  # num_neighbors for LigandMPNN
        dims["M"] = 25  # num_context_atoms for LigandMPNN
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    return dims


def create_feature_collator():
    """Create a FeatureCollator for MPNN models, including ligand features."""
    return FeatureCollator()


def assert_value_checks(network_output, input_features, model_type):
    """Comprehensive value testing for all model outputs."""
    dims = get_expected_dimensions(model_type)
    vocab_size = dims["vocab_size"]

    # ===== VALUE CHECKS =====
    # Check that sequence predictions are in valid range (only if present)
    decoder_features = network_output["decoder_features"]
    if "S_sampled" in decoder_features:
        assert torch.all(decoder_features["S_sampled"] >= 0) and torch.all(
            decoder_features["S_sampled"] < vocab_size
        )
    if "S_argmax" in decoder_features:
        assert torch.all(decoder_features["S_argmax"] >= 0) and torch.all(
            decoder_features["S_argmax"] < vocab_size
        )

    # Check that probabilities sum to 1 (only if present)
    if "probs" in decoder_features:
        probs_sum = decoder_features["probs"].sum(dim=-1)
        assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-6)

    # Check for NaN/Inf values in encoder and decoder features
    for feature_group in ["encoder_features", "decoder_features"]:
        if feature_group in network_output:
            for key, tensor in network_output[feature_group].items():
                if isinstance(tensor, torch.Tensor):
                    assert not torch.isnan(
                        tensor
                    ).any(), f"NaN in {feature_group}.{key}"
                    assert not torch.isinf(
                        tensor
                    ).any(), f"Inf in {feature_group}.{key}"


def assert_comprehensive_shapes(network_output, input_features, model_type):
    """Comprehensive shape testing for all model outputs."""
    dims = get_expected_dimensions(model_type)

    # Determine batch sizes
    if (
        "X" in input_features
        and hasattr(input_features["X"], "shape")
        and input_features["X"].shape[0] > 1
    ):
        original_B = input_features["X"].shape[0]
        repeated_B = original_B
    else:
        original_B = 1  # We always start with batch size 1
        repeat_num = (
            input_features.get("repeat_sample_num", 1)
            if input_features.get("repeat_sample_num") is not None
            else 1
        )
        repeated_B = repeat_num
    L = network_output["input_features"]["X"].shape[1]

    # Expected dimensions
    vocab_size = dims["vocab_size"]
    hidden_dim = dims["hidden_dim"]
    K = dims["K"]
    num_atoms = dims["num_atoms"]
    num_backbone_atoms = dims["num_backbone_atoms"]

    # Use expected dimensions from the model configuration
    num_virtual_atoms = dims["num_virtual_atoms"]
    num_rep_atoms = dims["num_rep_atoms"]

    # ===== INPUT FEATURES SHAPES =====
    input_features = network_output["input_features"]

    # Features that do NOT get repeated (maintain original batch size)
    non_repeated_input_features = [
        "X",
        "X_pre_noise",
        "X_backbone",
        "X_m_backbone",
        "X_virtual_atoms",
        "X_m_virtual_atoms",
        "X_rep_atoms",
        "X_m_rep_atoms",
        "chain_labels",
        "R_idx",
    ]

    for key in non_repeated_input_features:
        if key in input_features and input_features[key] is not None:
            assert (
                input_features[key].shape[0] == original_B
            ), f"{key} should have batch size {original_B}, got {input_features[key].shape[0]}"

    # Features that DO get repeated (based on repeat_along_batch implementation)
    repeated_input_features = [
        "S",
        "residue_mask",
        "designed_residue_mask",
        "temperature",
        "bias",
        "pair_bias",
        "symmetry_equivalence_group",
        "symmetry_weight",
        "mask_for_loss",
    ]

    for key in repeated_input_features:
        if key in input_features and input_features[key] is not None:
            assert (
                input_features[key].shape[0] == repeated_B
            ), f"{key} should have batch size {repeated_B}, got {input_features[key].shape[0]}"

    # Specific shape assertions for input features
    assert (
        input_features["X"].shape == (original_B, L, num_atoms, 3)
    ), f"X shape: expected ({original_B}, {L}, {num_atoms}, 3), got {input_features['X'].shape}"
    assert input_features["X_pre_noise"].shape == (original_B, L, num_atoms, 3)
    assert input_features["X_backbone"].shape == (original_B, L, num_backbone_atoms, 3)
    assert input_features["X_m_backbone"].shape == (original_B, L, num_backbone_atoms)
    assert input_features["X_virtual_atoms"].shape == (
        original_B,
        L,
        num_virtual_atoms,
        3,
    )
    assert input_features["X_m_virtual_atoms"].shape == (
        original_B,
        L,
        num_virtual_atoms,
    )
    assert input_features["X_rep_atoms"].shape == (original_B, L, num_rep_atoms, 3)
    assert input_features["X_m_rep_atoms"].shape == (original_B, L, num_rep_atoms)
    assert input_features["S"].shape == (repeated_B, L)
    assert input_features["residue_mask"].shape == (repeated_B, L)
    assert input_features["chain_labels"].shape == (original_B, L)
    assert input_features["R_idx"].shape == (original_B, L)
    assert input_features["mask_for_loss"].shape == (repeated_B, L)

    # ===== GRAPH FEATURES SHAPES =====
    graph_features = network_output["graph_features"]

    # Graph features have mixed batch sizes
    assert (
        graph_features["E_idx"].shape == (repeated_B, L, K)
    ), f"E_idx shape: expected ({repeated_B}, {L}, {K}), got {graph_features['E_idx'].shape}"
    assert (
        graph_features["E"].shape == (original_B, L, K, hidden_dim)
    ), f"E shape: expected ({original_B}, {L}, {K}, {hidden_dim}), got {graph_features['E'].shape}"

    # ===== ENCODER FEATURES SHAPES =====
    encoder_features = network_output["encoder_features"]

    assert (
        encoder_features["h_V"].shape == (repeated_B, L, hidden_dim)
    ), f"Encoder h_V shape: expected ({repeated_B}, {L}, {hidden_dim}), got {encoder_features['h_V'].shape}"
    assert (
        encoder_features["h_E"].shape == (repeated_B, L, K, hidden_dim)
    ), f"Encoder h_E shape: expected ({repeated_B}, {L}, {K}, {hidden_dim}), got {encoder_features['h_E'].shape}"

    # ===== DECODER FEATURES SHAPES =====
    decoder_features = network_output["decoder_features"]

    # All decoder features should have repeated batch size
    decoder_tensor_features = [
        "causal_mask",
        "anti_causal_mask",
        "decoding_order",
        "decode_last_mask",
        "h_V",
        "logits",
        "log_probs",
        "probs",
        "probs_sample",
        "S_sampled",
        "S_argmax",
    ]

    for key in decoder_tensor_features:
        if key in decoder_features and decoder_features[key] is not None:
            assert (
                decoder_features[key].shape[0] == repeated_B
            ), f"Decoder {key} should have batch size {repeated_B}, got {decoder_features[key].shape[0]}"

    # Specific decoder shape assertions
    assert decoder_features["causal_mask"].shape == (repeated_B, L, K, 1)
    assert decoder_features["anti_causal_mask"].shape == (repeated_B, L, K, 1)
    assert decoder_features["decoding_order"].shape == (repeated_B, L)
    assert decoder_features["decode_last_mask"].shape == (repeated_B, L)
    assert decoder_features["h_V"].shape == (repeated_B, L, hidden_dim)
    assert decoder_features["logits"].shape == (repeated_B, L, vocab_size)
    assert decoder_features["log_probs"].shape == (repeated_B, L, vocab_size)
    assert decoder_features["probs"].shape == (repeated_B, L, vocab_size)
    assert decoder_features["probs_sample"].shape == (repeated_B, L, vocab_size)
    assert decoder_features["S_sampled"].shape == (repeated_B, L)
    assert decoder_features["S_argmax"].shape == (repeated_B, L)

    # ===== LIGAND-SPECIFIC FEATURES =====
    if model_type == "ligand_mpnn":
        N = input_features["Y"].shape[1]  # Number of ligand atoms

        if input_features["atomize_side_chains"]:
            # If we atomize side chains, we need to extract the number of
            # ligand atoms, as the number of side chains atomized is stochastic.
            M = min(dims["M"], input_features["ligand_subgraph_Y_t"].shape[2])
        else:
            # If we have less than M ligand atoms, then M will be the number of
            # ligand atoms.
            M = min(dims["M"], N)

        # Additional input features for LigandMPNN
        ligand_input_keys = [
            "hide_side_chain_mask",
            "Y",
            "Y_pre_noise",
            "ligand_subgraph_Y",
            "ligand_subgraph_Y_m",
            "ligand_subgraph_Y_t",
        ]
        for key in ligand_input_keys:
            assert key in input_features, f"Missing ligand input feature: {key}"

        # Shape assertions for ligand input features
        assert input_features["Y"].shape == (original_B, N, 3)
        assert input_features["Y_pre_noise"].shape == (original_B, N, 3)

        assert input_features["ligand_subgraph_Y"].shape == (original_B, L, M, 3)
        assert input_features["ligand_subgraph_Y_m"].shape == (original_B, L, M)
        assert input_features["ligand_subgraph_Y_t"].shape == (original_B, L, M)
        assert input_features["hide_side_chain_mask"].shape == (original_B, L)

        # Additional graph features for LigandMPNN
        ligand_graph_keys = [
            "E_protein_to_ligand",
            "ligand_subgraph_nodes",
            "ligand_subgraph_edges",
        ]
        for key in ligand_graph_keys:
            assert key in graph_features, f"Missing ligand graph feature: {key}"

        # Shape assertions for ligand graph features
        assert graph_features["E_protein_to_ligand"].shape == (
            original_B,
            L,
            M,
            hidden_dim,
        )
        assert graph_features["ligand_subgraph_nodes"].shape == (
            original_B,
            L,
            M,
            hidden_dim,
        )
        assert graph_features["ligand_subgraph_edges"].shape == (
            original_B,
            L,
            M,
            M,
            hidden_dim,
        )


def assert_loss_output_structure(loss, loss_output, network_input):
    """Check loss properties and output structure for both model types."""
    # Check loss values.
    assert loss.dim() == 0, "Loss should be a scalar"
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert not torch.isinf(loss), "Loss should not be Inf"
    assert loss.detach().item() > 0, "Loss should be positive"

    # Check loss output structure
    assert "label_smoothed_nll_loss_per_residue" in loss_output
    assert "label_smoothed_nll_loss_agg" in loss_output

    # Check loss output shapes.
    B, L = network_input["input_features"]["residue_mask"].shape
    assert loss_output["label_smoothed_nll_loss_per_residue"].shape == (B, L)
    assert loss_output["label_smoothed_nll_loss_agg"].dim() == 0


def assert_gradient_properties(model):
    """
    Test gradient properties for model parameters.

    Args:
        model: The model to test
    """
    # Check gradient properties - all parameters should have gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            # All trainable parameters should have gradients
            assert (
                param.grad is not None
            ), f"Parameter {name} requires gradients but has None gradient"

            # Check gradient shape matches parameter shape
            assert (
                param.grad.shape == param.shape
            ), f"Gradient shape mismatch for {name}: expected {param.shape}, got {param.grad.shape}"

            # Check for NaN/Inf in gradients
            assert not torch.isnan(
                param.grad
            ).any(), f"NaN found in gradients for parameter {name}"
            assert not torch.isinf(
                param.grad
            ).any(), f"Inf found in gradients for parameter {name}"

            # Check gradient magnitude is reasonable (not too large)
            grad_norm = param.grad.norm()
            assert grad_norm < 1e6, f"Gradient norm too large for {name}: {grad_norm}"


def compute_loss_and_backward(model, network_input, network_output, loss_fn):
    """
    Compute loss and perform backward pass.

    Args:
        model: The model to compute gradients for
        network_input: Model input
        network_output: Model output from forward pass
        loss_fn: Loss function to use for gradient computation

    Returns:
        tuple: (loss, loss_output)
    """
    # Compute loss
    loss, loss_output = loss_fn(network_input, network_output, dict())

    # Clear any existing gradients
    model.zero_grad()

    # Backward pass - compute gradients
    loss.backward()

    return loss, loss_output


def assert_sequence_recovery_metric_shapes(
    metrics,
    network_input,
    is_interface=False,
    return_per_example=False,
    return_per_residue=False,
):
    """
    Generic function to test sequence recovery metric shapes.

    Args:
        metrics (dict): Computed metrics dictionary
        network_input (dict): Network input for shape reference
        is_interface (bool): Whether these are interface metrics (affects key prefix)
        return_per_example (bool): Whether per-example metrics should be present
        return_per_residue (bool): Whether per-residue metrics should be present
    """
    prefix = "interface_" if is_interface else ""
    B, L = network_input["input_features"]["S"].shape

    # Check required keys and their shapes
    required_keys = [
        f"{prefix}mean_sequence_recovery_sampled",
        f"{prefix}mean_sequence_recovery_argmax",
    ]

    for key in required_keys:
        assert key in metrics, f"Missing required key: {key}"
        assert isinstance(
            metrics[key], float
        ), f"{key} should be a float scalar, got {type(metrics[key])}"

    if return_per_example:
        per_example_keys = [
            f"{prefix}sequence_recovery_per_example_sampled",
            f"{prefix}sequence_recovery_per_example_argmax",
            f"{prefix}correct_per_example_sampled",
            f"{prefix}correct_per_example_argmax",
            f"{prefix}total_valid_per_example",
            f"{prefix}valid_examples_mask",
        ]

        for key in per_example_keys:
            assert key in metrics, f"Missing per-example key: {key}"
            assert metrics[key].shape == (
                B,
            ), f"{key} should have shape ({B},), got {metrics[key].shape}"

    if return_per_residue:
        per_residue_keys = [
            f"{prefix}correct_predictions_per_residue_sampled",
            f"{prefix}correct_predictions_per_residue_argmax",
            f"{prefix}per_residue_mask",
        ]

        for key in per_residue_keys:
            assert key in metrics, f"Missing per-residue key: {key}"
            assert metrics[key].shape == (
                B,
                L,
            ), f"{key} should have shape ({B}, {L}), got {metrics[key].shape}"


def assert_nll_metric_shapes(
    metrics,
    network_input,
    is_interface=False,
    return_per_example=False,
    return_per_residue=False,
):
    """
    Generic function to test NLL metric shapes.

    Args:
        metrics (dict): Computed metrics dictionary
        network_input (dict): Network input for shape reference
        is_interface (bool): Whether these are interface metrics (affects key prefix)
        return_per_example (bool): Whether per-example metrics should be present
        return_per_residue (bool): Whether per-residue metrics should be present
    """
    prefix = "interface_" if is_interface else ""
    B, L = network_input["input_features"]["S"].shape

    # Check required keys and their shapes
    required_keys = [f"{prefix}mean_nll", f"{prefix}mean_perplexity"]

    for key in required_keys:
        assert key in metrics, f"Missing required key: {key}"
        assert isinstance(
            metrics[key], float
        ), f"{key} should be a float scalar, got {type(metrics[key])}"

    if return_per_example:
        per_example_keys = [
            f"{prefix}nll_per_example",
            f"{prefix}perplexity_per_example",
            f"{prefix}total_valid_per_example",
            f"{prefix}valid_examples_mask",
        ]

        for key in per_example_keys:
            assert key in metrics, f"Missing per-example key: {key}"
            assert metrics[key].shape == (
                B,
            ), f"{key} should have shape ({B},), got {metrics[key].shape}"

    if return_per_residue:
        per_residue_keys = [f"{prefix}nll_per_residue", f"{prefix}per_residue_mask"]

        for key in per_residue_keys:
            assert key in metrics, f"Missing per-residue key: {key}"
            assert metrics[key].shape == (
                B,
                L,
            ), f"{key} should have shape ({B}, {L}), got {metrics[key].shape}"


def assert_metric_values(metrics, is_interface=False):
    """
    Generic function to test metric values are reasonable (no NaN/Inf, proper ranges).

    Args:
        metrics (dict): Computed metrics dictionary
        is_interface (bool): Whether these are interface metrics (affects key prefix)
    """
    prefix = "interface_" if is_interface else ""
    # Masked per-example checks
    valid_mask = metrics.get(f"{prefix}valid_examples_mask", None)
    if valid_mask is not None:
        # Scalar checks
        if torch.any(valid_mask):
            seq_recovery_keys = [
                f"{prefix}mean_sequence_recovery_sampled",
                f"{prefix}mean_sequence_recovery_argmax",
            ]

            for key in seq_recovery_keys:
                if key in metrics:
                    if isinstance(metrics[key], torch.Tensor):
                        assert not torch.isnan(metrics[key]), f"NaN found in {key}"
                        assert not torch.isinf(metrics[key]), f"Inf found in {key}"
                    assert (
                        0 <= metrics[key] <= 1
                    ), f"{key} should be in [0, 1] range, got {metrics[key]}"

            # Check NLL and perplexity values are positive
            nll_keys = [f"{prefix}mean_nll"]
            perplexity_keys = [f"{prefix}mean_perplexity"]

            for key in nll_keys:
                if key in metrics:
                    if isinstance(metrics[key], torch.Tensor):
                        assert not torch.isnan(metrics[key]), f"NaN found in {key}"
                        assert not torch.isinf(metrics[key]), f"Inf found in {key}"
                    assert (
                        metrics[key] >= 0
                    ), f"{key} should be >= 0, got {metrics[key]}"

            for key in perplexity_keys:
                if key in metrics:
                    if isinstance(metrics[key], torch.Tensor):
                        assert not torch.isnan(metrics[key]), f"NaN found in {key}"
                        assert not torch.isinf(metrics[key]), f"Inf found in {key}"
                    assert (
                        metrics[key] >= 1.0
                    ), f"{key} should be >= 1.0, got {metrics[key]}"

        # Sequence recovery per-example
        for key in [
            f"{prefix}sequence_recovery_per_example_sampled",
            f"{prefix}sequence_recovery_per_example_argmax",
        ]:
            if key in metrics:
                vals = metrics[key][valid_mask]
                assert not torch.isnan(
                    vals
                ).any(), f"NaN found in {key} for valid examples"
                assert not torch.isinf(
                    vals
                ).any(), f"Inf found in {key} for valid examples"
                assert torch.all(
                    (0 <= vals) & (vals <= 1)
                ), f"{key} values should be in [0, 1] for valid examples, got {vals}"
        # NLL per-example
        for key in [f"{prefix}nll_per_example"]:
            if key in metrics:
                vals = metrics[key][valid_mask]
                assert not torch.isnan(
                    vals
                ).any(), f"NaN found in {key} for valid examples"
                assert not torch.isinf(
                    vals
                ).any(), f"Inf found in {key} for valid examples"
                assert torch.all(
                    vals >= 0
                ), f"{key} values should be >= 0 for valid examples, got {vals}"
        # Perplexity per-example
        for key in [f"{prefix}perplexity_per_example"]:
            if key in metrics:
                vals = metrics[key][valid_mask]
                assert not torch.isnan(
                    vals
                ).any(), f"NaN found in {key} for valid examples"
                assert not torch.isinf(
                    vals
                ).any(), f"Inf found in {key} for valid examples"
                assert torch.all(
                    vals >= 1.0
                ), f"{key} values should be >= 1.0 for valid examples, got {vals}"
        # Correct per-example
        for key in [
            f"{prefix}correct_per_example_sampled",
            f"{prefix}correct_per_example_argmax",
        ]:
            if key in metrics:
                vals = metrics[key][valid_mask]
                assert not torch.isnan(
                    vals
                ).any(), f"NaN found in {key} for valid examples"
                assert not torch.isinf(
                    vals
                ).any(), f"Inf found in {key} for valid examples"
                assert torch.all(
                    vals >= 0
                ), f"{key} values should be >= 0 for valid examples, got {vals}"
        # Total valid per-example
        for key in [f"{prefix}total_valid_per_example"]:
            if key in metrics:
                vals = metrics[key][valid_mask]
                assert not torch.isnan(
                    vals
                ).any(), f"NaN found in {key} for valid examples"
                assert not torch.isinf(
                    vals
                ).any(), f"Inf found in {key} for valid examples"
                assert torch.all(
                    vals > 0
                ), f"{key} values should be > 0 for valid examples, got {vals}"

        # Per-residue checks (masked by both valid_examples_mask and per_residue_mask)
        per_residue_keys = [
            (
                f"{prefix}correct_predictions_per_residue_sampled",
                f"{prefix}per_residue_mask",
            ),
            (
                f"{prefix}correct_predictions_per_residue_argmax",
                f"{prefix}per_residue_mask",
            ),
            (f"{prefix}nll_per_residue", f"{prefix}per_residue_mask"),
        ]
        for value_key, mask_key in per_residue_keys:
            if value_key in metrics and mask_key in metrics:
                values = metrics[value_key]
                mask = metrics[mask_key]
                # Only check values where both valid_examples_mask and per_residue_mask are True
                # valid_mask: [B], mask: [B, L], values: [B, L]
                # Broadcast valid_mask to [B, 1]
                combined_mask = valid_mask[:, None] & mask.bool()
                selected = values[combined_mask]
                if "correct_predictions" in value_key:
                    # Should be 0 or 1
                    assert not torch.isnan(
                        selected
                    ).any(), f"NaN found in {value_key} for valid residues"
                    assert not torch.isinf(
                        selected
                    ).any(), f"Inf found in {value_key} for valid residues"
                    assert torch.all(
                        (0 <= selected) & (selected <= 1)
                    ), f"{value_key} values should be in [0, 1] for valid residues, got {selected}"
                elif "nll_per_residue" in value_key:
                    # Should be >= 0
                    assert not torch.isnan(
                        selected
                    ).any(), f"NaN found in {value_key} for valid residues"
                    assert not torch.isinf(
                        selected
                    ).any(), f"Inf found in {value_key} for valid residues"
                    assert torch.all(
                        selected >= 0
                    ), f"{value_key} values should be >= 0 for valid residues, got {selected}"


def assert_all_metrics_comprehensive(
    seq_metrics,
    nll_metrics,
    interface_seq_metrics,
    interface_nll_metrics,
    network_input,
    return_per_example=False,
    return_per_residue=False,
):
    """
    Comprehensive testing function for all metrics (sequence recovery, NLL, and their interface versions).

    Args:
        seq_metrics (dict): Non-interface sequence recovery metrics
        nll_metrics (dict): Non-interface NLL metrics
        interface_seq_metrics (dict): Interface sequence recovery metrics
        interface_nll_metrics (dict): Interface NLL metrics
        network_input (dict): Network input for shape reference
        return_per_example (bool): Whether per-example metrics should be present
        return_per_residue (bool): Whether per-residue metrics should be present
    """
    # Test shapes
    assert_sequence_recovery_metric_shapes(
        seq_metrics,
        network_input,
        is_interface=False,
        return_per_example=return_per_example,
        return_per_residue=return_per_residue,
    )
    assert_nll_metric_shapes(
        nll_metrics,
        network_input,
        is_interface=False,
        return_per_example=return_per_example,
        return_per_residue=return_per_residue,
    )
    assert_sequence_recovery_metric_shapes(
        interface_seq_metrics,
        network_input,
        is_interface=True,
        return_per_example=return_per_example,
        return_per_residue=return_per_residue,
    )
    assert_nll_metric_shapes(
        interface_nll_metrics,
        network_input,
        is_interface=True,
        return_per_example=return_per_example,
        return_per_residue=return_per_residue,
    )

    # Test values
    assert_metric_values(seq_metrics, is_interface=False)
    assert_metric_values(nll_metrics, is_interface=False)
    assert_metric_values(interface_seq_metrics, is_interface=True)
    assert_metric_values(interface_nll_metrics, is_interface=True)

    # Check that interface masks have fewer or equal valid residues
    if return_per_example:
        interface_total = interface_seq_metrics[
            "interface_total_valid_per_example"
        ].sum()
        full_total = seq_metrics["total_valid_per_example"].sum()
        assert (
            interface_total <= full_total
        ), f"Interface total ({interface_total}) should be <= full total ({full_total})"


def combine_kwargs_to_compute(metric, network_input, network_output):
    """
    Utility to combine network_input and network_output into kwargs for metric.compute.
    Args:
        metric: Metric instance with kwargs_to_compute_args
        network_input: dict with 'input_features' and 'atom_array'
        network_output: dict with 'decoder_features' and 'input_features'
    Returns:
        dict of arguments for metric.compute
    """
    network_data = {"network_input": network_input, "network_output": network_output}

    args = {}
    for arg, path in metric.kwargs_to_compute_args.items():
        val = network_data
        for key in path:
            val = val.get(key)
        args[arg] = val

    return args
