"""
Tests for the MPNN transform pipeline.

This module contains tests that verify the MPNN pipeline functionality
including data processing, shape validation, and feature extraction.
"""

import numpy as np
import pytest
import torch
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.ml.transforms.base import TransformPipelineError
from atomworks.ml.utils.testing import cached_parse
from mpnn.pipelines.mpnn import build_mpnn_transform_pipeline


class TestMPNNPipeline:
    """Test the MPNN transform pipeline."""

    # ------------------------------
    # AddUserSettings: end-to-end tests
    # ------------------------------
    @pytest.mark.parametrize("model_type", ["protein_mpnn", "ligand_mpnn"])
    def test_user_settings_defaults_training(self, model_type: str):
        """Defaults when nothing is provided (training)."""
        data = cached_parse("3en2")
        pipeline = build_mpnn_transform_pipeline(
            model_type=model_type, is_inference=False
        )
        result = pipeline(data)
        input_features = result["input_features"]

        assert input_features["structure_noise"] == 0.1
        assert input_features["decode_type"] == "teacher_forcing"
        assert input_features["causality_pattern"] == "auto_regressive"
        assert input_features["initialize_sequence_embedding_with_ground_truth"] is True
        # ligand True, protein False
        expected_atomize = model_type == "ligand_mpnn"
        assert input_features["atomize_side_chains"] is expected_atomize
        assert input_features["repeat_sample_num"] is None
        assert input_features["features_to_return"] is None

        # Array-wide settings should be None in training by default
        assert input_features.get("designed_residue_mask", None) is None
        assert input_features.get("temperature", None) is None
        assert input_features.get("symmetry_equivalence_group", None) is None
        assert input_features.get("symmetry_weight", None) is None
        assert input_features.get("bias", None) is None
        assert input_features.get("pair_bias", None) is None

    @pytest.mark.parametrize("model_type", ["protein_mpnn", "ligand_mpnn"])
    def test_user_settings_defaults_inference(self, model_type: str):
        """Defaults when nothing is provided (inference)."""
        data = cached_parse("3en2")
        pipeline = build_mpnn_transform_pipeline(
            model_type=model_type, is_inference=True
        )
        result = pipeline(data)
        input_features = result["input_features"]

        def to_numpy(x):
            if x is None:
                return None
            return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

        assert input_features["structure_noise"] == 0.0
        assert input_features["decode_type"] == "auto_regressive"
        assert input_features["causality_pattern"] == "auto_regressive"
        assert (
            input_features["initialize_sequence_embedding_with_ground_truth"] is False
        )
        assert input_features["atomize_side_chains"] is False
        assert input_features["repeat_sample_num"] is None
        assert input_features["features_to_return"] is None

        # temperature should be 0.1 for each non-atomized token
        temperature = to_numpy(input_features.get("temperature", None))
        assert temperature is not None
        assert temperature.dtype == np.float32
        assert np.allclose(temperature, 0.1)
        # Others remain None
        assert input_features.get("designed_residue_mask", None) is None
        assert input_features.get("symmetry_equivalence_group", None) is None
        assert input_features.get("symmetry_weight", None) is None
        assert input_features.get("bias", None) is None
        assert input_features.get("pair_bias", None) is None

    @pytest.mark.parametrize("model_type", ["protein_mpnn", "ligand_mpnn"])
    def test_user_settings_provided_everything(self, model_type: str):
        """Provide scalars and atom_array annotations before pipeline and verify outputs."""
        data = cached_parse("3en2")

        # Set scalar overrides
        data["structure_noise"] = 0.25
        data["decode_type"] = "auto_regressive"
        data["causality_pattern"] = "conditional"
        data["initialize_sequence_embedding_with_ground_truth"] = False
        data["atomize_side_chains"] = False
        data["repeat_sample_num"] = 2
        data["features_to_return"] = {
            "input_features": ["mask_for_loss"],
            "decoder_features": ["log_probs", "S_sampled", "S_argmax"],
        }

        # Annotate atom_array with per-atom values (constant within tokens)
        atom_array = data["atom_array"]
        n_atoms = (
            atom_array.array_length()
            if hasattr(atom_array, "array_length")
            else len(atom_array)
        )

        # 1D annotations
        atom_array.set_annotation(
            "mpnn_designed_residue_mask", np.full(n_atoms, True, dtype=np.bool_)
        )
        atom_array.set_annotation(
            "mpnn_temperature", np.full(n_atoms, 0.5, dtype=np.float32)
        )
        atom_array.set_annotation(
            "mpnn_symmetry_equivalence_group", np.ones(n_atoms, dtype=np.int32)
        )
        atom_array.set_annotation(
            "mpnn_symmetry_weight", np.ones(n_atoms, dtype=np.float32)
        )

        # Per-atom 21-dim bias; constant vector across atoms
        vocab = 21
        bias_vec = np.linspace(-0.2, 0.2, vocab).astype(np.float32)
        atom_array.set_annotation(
            "mpnn_bias", np.repeat(bias_vec[None, :], n_atoms, axis=0)
        )

        # Optional 2D pair bias for a couple of atom pairs (may be dropped by preprocessing)
        if n_atoms >= 4:
            # Promote to AtomArrayPlus to enable 2D annotations
            data["atom_array"] = as_atom_array_plus(atom_array)
            atom_array = data["atom_array"]
            pairs = np.array([[0, 30], [70, 90]], dtype=np.int32)
            pair_vals = np.stack(
                [
                    np.eye(vocab, dtype=np.float32),
                    np.full((vocab, vocab), 0.01, dtype=np.float32),
                ],
                axis=0,
            )
            atom_array.set_annotation_2d("mpnn_pair_bias", pairs, pair_vals)

        pipeline = build_mpnn_transform_pipeline(
            model_type=model_type, is_inference=False
        )
        result = pipeline(data)
        input_features = result["input_features"]

        def to_numpy(x):
            if x is None:
                return None
            return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

        # Scalars
        assert input_features["structure_noise"] == 0.25
        assert input_features["decode_type"] == "auto_regressive"
        assert input_features["causality_pattern"] == "conditional"
        assert (
            input_features["initialize_sequence_embedding_with_ground_truth"] is False
        )
        assert input_features["atomize_side_chains"] is False
        assert input_features["repeat_sample_num"] == 2
        assert input_features["features_to_return"] == {
            "input_features": ["mask_for_loss"],
            "decoder_features": ["log_probs", "S_sampled", "S_argmax"],
        }

        # Array-wide mapped values
        designed_residue_mask = input_features.get("designed_residue_mask", None)
        assert designed_residue_mask is not None and designed_residue_mask.dtype in (
            torch.bool,
            np.bool_,
            bool,
        )
        temperature = to_numpy(input_features.get("temperature", None))
        assert temperature is not None and np.allclose(temperature, 0.5)
        sym_group = to_numpy(input_features.get("symmetry_equivalence_group", None))
        assert sym_group is not None and np.all(sym_group == 1)
        sym_weight = to_numpy(input_features.get("symmetry_weight", None))
        assert sym_weight is not None and np.allclose(sym_weight, 1.0)
        bias = to_numpy(input_features.get("bias", None))
        assert (
            bias is not None
            and bias.shape[-1] == vocab
            and np.allclose(bias, bias_vec[None, :])
        )
        # pair_bias may be None depending on preprocessing, otherwise assert shape and non-zero content
        pair_bias = to_numpy(input_features.get("pair_bias", None))
        if pair_bias is not None:
            L = bias.shape[0]
            assert (
                pair_bias.shape[0] == L
                and pair_bias.shape[2] == L
                and pair_bias.shape[1] == vocab
                and pair_bias.shape[3] == vocab
            )
            identity = np.eye(vocab, dtype=np.float32)
            const = np.full((vocab, vocab), 0.01, dtype=np.float32)
            found_identity = False
            found_const = False
            for i in range(L):
                for j in range(L):
                    block = pair_bias[i, :, j, :]
                    if not found_identity and np.allclose(block, identity, atol=1e-6):
                        found_identity = True
                    if not found_const and np.allclose(block, const, atol=1e-6):
                        found_const = True
                    if found_identity and found_const:
                        break
                if found_identity and found_const:
                    break
            assert found_identity and found_const

    @pytest.mark.parametrize("model_type", ["protein_mpnn", "ligand_mpnn"])
    def test_user_settings_type_checks_raise(self, model_type: str):
        """Wrong scalar and features_to_return types should raise TypeError."""
        data = cached_parse("3en2")
        # Inject wrong types
        data["structure_noise"] = "bad"
        data["decode_type"] = 123
        data["causality_pattern"] = 5
        data["initialize_sequence_embedding_with_ground_truth"] = "nope"
        data["atomize_side_chains"] = "false"
        data["repeat_sample_num"] = 1.5
        data["features_to_return"] = {1: "not-a-list"}

        pipeline = build_mpnn_transform_pipeline(
            model_type=model_type, is_inference=False
        )
        with pytest.raises(TransformPipelineError):
            _ = pipeline(data)

    @pytest.mark.parametrize("model_type", ["protein_mpnn", "ligand_mpnn"])
    def test_user_settings_token_level_consistency_violation(self, model_type: str):
        """Per-token inconsistency in an annotation should raise ValueError."""
        data = cached_parse("3en2")
        atom_array = data["atom_array"]
        n_atoms = (
            atom_array.array_length()
            if hasattr(atom_array, "array_length")
            else len(atom_array)
        )
        # Alternate values across atoms to guarantee mismatch within tokens
        atom_array.set_annotation(
            "mpnn_temperature", (np.arange(n_atoms) % 2).astype(np.float32)
        )

        pipeline = build_mpnn_transform_pipeline(
            model_type=model_type, is_inference=False
        )
        with pytest.raises(TransformPipelineError):
            _ = pipeline(data)

    @pytest.mark.parametrize("model_type", ["protein_mpnn", "ligand_mpnn"])
    def test_user_settings_pair_bias_token_pair_uniqueness_violation(
        self, model_type: str
    ):
        """Duplicate token-level pairs in mpnn_pair_bias should raise ValueError."""
        data = cached_parse("3en2")
        atom_array = data["atom_array"]
        n_atoms = (
            atom_array.array_length()
            if hasattr(atom_array, "array_length")
            else len(atom_array)
        )
        if n_atoms >= 2:
            # Promote to AtomArrayPlus to enable 2D annotations
            data["atom_array"] = as_atom_array_plus(atom_array)
            atom_array = data["atom_array"]
            pairs = np.array(
                [[0, 1], [0, 1]], dtype=np.int32
            )  # duplicate token pair on purpose
            vocab = 21
            pair_vals = np.stack(
                [
                    np.eye(vocab, dtype=np.float32),
                    np.full((vocab, vocab), 0.02, dtype=np.float32),
                ],
                axis=0,
            )
            atom_array.set_annotation_2d("mpnn_pair_bias", pairs, pair_vals)

            pipeline = build_mpnn_transform_pipeline(
                model_type=model_type, is_inference=False
            )
            with pytest.raises(TransformPipelineError):
                _ = pipeline(data)

    @pytest.mark.parametrize("pdb_id", ["3en2", "2xni"])
    @pytest.mark.parametrize("model_type", ["protein_mpnn", "ligand_mpnn"])
    def test_mpnn_pipeline_basic(self, pdb_id: str, model_type: str):
        """Test that the MPNN pipeline runs successfully on test PDB structures."""

        # Load structure using cached_parse
        data = cached_parse(pdb_id)

        # Create pipeline
        pipeline = build_mpnn_transform_pipeline(model_type=model_type)

        # Apply pipeline
        result = pipeline(data)

        # Check that we have the expected keys
        assert "input_features" in result
        assert "atom_array" in result

        input_features = result["input_features"]

        # Check non-atomized token features
        expected_non_atomized_keys = [
            "X",
            "X_m",
            "S",
            "R_idx",
            "chain_labels",
            "residue_mask",
        ]
        for key in expected_non_atomized_keys:
            assert key in input_features, f"Missing key {key} in input_features"

        if model_type == "ligand_mpnn":
            # Check atomized token features
            expected_atomized_keys = ["Y", "Y_t", "Y_m"]
            for key in expected_atomized_keys:
                assert key in input_features, f"Missing key {key} in input_features"

    @pytest.mark.parametrize("pdb_id", ["3en2", "2xni"])
    @pytest.mark.parametrize("model_type", ["protein_mpnn", "ligand_mpnn"])
    def test_mpnn_pipeline_shapes_and_dtypes(self, pdb_id: str, model_type: str):
        """Test that the MPNN pipeline produces correct shapes and dtypes."""

        # Load structure using cached_parse
        data = cached_parse(pdb_id)

        # Create pipeline
        pipeline = build_mpnn_transform_pipeline(model_type=model_type)

        # Apply pipeline
        result = pipeline(data)
        input_features = result["input_features"]

        # Get dimensions
        L = input_features["X"].shape[0]  # Number of non-atomized tokens

        # Test non-atomized token features
        # X: (L, 37, 3) coordinates, float32
        assert input_features["X"].shape == (
            L,
            37,
            3,
        ), f"X shape mismatch: expected ({L}, 37, 3), got {input_features['X'].shape}"
        assert (
            input_features["X"].dtype == torch.float32
        ), f"X dtype mismatch: expected torch.float32, got {input_features['X'].dtype}"

        # X_m: (L, 37) mask, int32
        assert input_features["X_m"].shape == (
            L,
            37,
        ), f"X_m shape mismatch: expected ({L}, 37), got {input_features['X_m'].shape}"
        assert (
            input_features["X_m"].dtype == torch.bool
        ), f"X_m dtype mismatch: expected torch.bool, got {input_features['X_m'].dtype}"

        # S: (L,) sequence, int64
        assert input_features["S"].shape == (
            L,
        ), f"S shape mismatch: expected ({L},), got {input_features['S'].shape}"
        assert (
            input_features["S"].dtype == torch.int64
        ), f"S dtype mismatch: expected torch.int64, got {input_features['S'].dtype}"

        # R_idx: (L,) residue indices, int32
        assert input_features["R_idx"].shape == (
            L,
        ), f"R_idx shape mismatch: expected ({L},), got {input_features['R_idx'].shape}"
        assert (
            input_features["R_idx"].dtype == torch.int32
        ), f"R_idx dtype mismatch: expected torch.int32, got {input_features['R_idx'].dtype}"

        # chain_labels: (L,) chain labels, int64
        assert (
            input_features["chain_labels"].shape == (L,)
        ), f"chain_labels shape mismatch: expected ({L},), got {input_features['chain_labels'].shape}"
        assert (
            input_features["chain_labels"].dtype == torch.int64
        ), f"chain_labels dtype mismatch: expected torch.int64, got {input_features['chain_labels'].dtype}"

        # residue_mask: (L,) all 1's, float32
        assert (
            input_features["residue_mask"].shape == (L,)
        ), f"residue_mask shape mismatch: expected ({L},), got {input_features['residue_mask'].shape}"
        assert (
            input_features["residue_mask"].dtype == torch.bool
        ), f"residue_mask dtype mismatch: expected torch.bool, got {input_features['residue_mask'].dtype}"
        assert torch.all(
            input_features["residue_mask"] == 1.0
        ), "residue_mask should be all 1's"

        # Test atomized token features
        if model_type == "ligand_mpnn":
            n_atomized = input_features["Y"].shape[0]  # Number of atomized tokens
            # Y: (n_atomized, 3) coordinates, float32
            assert (
                input_features["Y"].shape
                == (
                    n_atomized,
                    3,
                )
            ), f"Y shape mismatch: expected ({n_atomized}, 3), got {input_features['Y'].shape}"
            assert (
                input_features["Y"].dtype == torch.float32
            ), f"Y dtype mismatch: expected torch.float32, got {input_features['Y'].dtype}"

            # Y_t: (n_atomized,) atomic numbers, int32
            assert (
                input_features["Y_t"].shape == (n_atomized,)
            ), f"Y_t shape mismatch: expected ({n_atomized},), got {input_features['Y_t'].shape}"
            assert (
                input_features["Y_t"].dtype == torch.int32
            ), f"Y_t dtype mismatch: expected torch.int32, got {input_features['Y_t'].dtype}"

            # Y_m: (n_atomized,) mask, int32
            assert (
                input_features["Y_m"].shape == (n_atomized,)
            ), f"Y_m shape mismatch: expected ({n_atomized},), got {input_features['Y_m'].shape}"
            assert (
                input_features["Y_m"].dtype == torch.bool
            ), f"Y_m dtype mismatch: expected torch.bool, got {input_features['Y_m'].dtype}"
            assert torch.all(input_features["Y_m"] == 1), "Y_m should be all 1's"

    @pytest.mark.parametrize("model_type", ["protein_mpnn", "ligand_mpnn"])
    def test_mpnn_pipeline_sequence_encoding_range(self, model_type):
        """Test that sequence encoding is in the valid range [0, 20]."""

        # Load structure using cached_parse
        data = cached_parse("3en2")

        # Create pipeline
        pipeline = build_mpnn_transform_pipeline(model_type=model_type)

        # Apply pipeline
        result = pipeline(data)
        input_features = result["input_features"]

        # Check that S is in valid range [0, 20] (21 possible values: 20 AA + UNK)
        S = input_features["S"]
        assert torch.all(S >= 0) and torch.all(
            S <= 20
        ), f"S values out of range [0, 20]: min={S.min()}, max={S.max()}"

    @pytest.mark.parametrize("model_type", ["ligand_mpnn"])
    def test_mpnn_pipeline_atomic_numbers_valid(self, model_type):
        """Test that atomic numbers are reasonable values."""

        # Load structure using cached_parse
        data = cached_parse("3en2")

        # Create pipeline
        pipeline = build_mpnn_transform_pipeline(model_type=model_type)

        # Apply pipeline
        result = pipeline(data)
        input_features = result["input_features"]

        # Check atomic numbers are reasonable (most common atoms in proteins: 1-16, plus some others)
        Y_t = input_features["Y_t"]
        if len(Y_t) > 0:
            assert torch.all(Y_t > 0) and torch.all(
                Y_t < 100
            ), f"Y_t values unreasonable: min={Y_t.min()}, max={Y_t.max()}"

    @pytest.mark.parametrize("model_type", ["protein_mpnn", "ligand_mpnn"])
    def test_mpnn_pipeline_occupancy_filtering(self, model_type):
        """Test that occupancy filtering correctly removes atoms below thresholds."""

        # Load structure
        data = cached_parse("3en2")

        # Define occupancy thresholds
        occupancy_threshold_backbone = 0.8
        occupancy_threshold_sidechain = 0.5

        # Create pipeline with specific thresholds
        pipeline = build_mpnn_transform_pipeline(
            model_type=model_type,
            occupancy_threshold_backbone=occupancy_threshold_backbone,
            occupancy_threshold_sidechain=occupancy_threshold_sidechain,
        )

        # Apply pipeline
        result = pipeline(data)
        atom_array = result["atom_array"]

        # Get non-atomized tokens only (proteins/standard residues)
        non_atomized_mask = ~atom_array.atomize
        non_atomized_array = atom_array[non_atomized_mask]

        if len(non_atomized_array) > 0:
            # Assert no backbone atoms below backbone occupancy threshold
            backbone_atoms = ["N", "CA", "C", "O"]
            backbone_mask = np.isin(non_atomized_array.atom_name, backbone_atoms)
            backbone_occupancies = non_atomized_array.occupancy[backbone_mask]

            assert np.all(backbone_occupancies > occupancy_threshold_backbone), (
                f"Found backbone atoms with occupancy <= {occupancy_threshold_backbone}: "
                f"min={backbone_occupancies.min():.3f}"
            )
