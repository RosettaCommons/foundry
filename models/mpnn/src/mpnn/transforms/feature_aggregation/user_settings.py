"""
This module contains the FeaturizeUserSettings transform that sets
mode-specific and common user features required by MPNN models.
"""

from typing import Any

import numpy as np
from atomworks.io.utils.atom_array_plus import AtomArrayPlus
from atomworks.io.utils.selection import get_annotation
from atomworks.ml.transforms._checks import (
    check_atom_array_annotation,
)
from atomworks.ml.transforms.base import Transform
from atomworks.ml.utils.token import (
    get_token_starts,
    spread_token_wise,
)


class FeaturizeUserSettings(Transform):
    """
    Transform for featurizing user settings to MPNN model inputs.
    """

    def __init__(
        self,
        is_inference: bool = False,
        minimal_return: bool = False,
        train_structure_noise_default: float = 0.1,
    ):
        """
        Initialize the FeaturizeUserSettings transform.

        Args:
            is_inference (bool): Whether this is inference mode. Defaults to
                False (training mode).
            minimal_return (bool): Whether to return minimal intermediate
                features. Defaults to False.
            train_structure_noise_default (float): Default standard
                deviation of Gaussian noise to add to atomic coordinates during
                training for data augmentation. Defaults to 0.1.
        """
        self.is_inference = is_inference
        self.minimal_return = minimal_return
        self.train_structure_noise_default = train_structure_noise_default

    def check_input(self, data: dict[str, Any]) -> None:
        """Check that atomize annotation exists in the data."""
        check_atom_array_annotation(data, ["atomize"])

        # Check that the scalar user settings have the correct types.
        if data.get("structure_noise", None) is not None:
            if not isinstance(data["structure_noise"], (float, int)):
                raise TypeError("structure_noise must be a float or int")

        if data.get("decode_type", None) is not None:
            if not isinstance(data["decode_type"], str):
                raise TypeError("decode_type must be a string")

        if data.get("causality_pattern", None) is not None:
            if not isinstance(data["causality_pattern"], str):
                raise TypeError("causality_pattern must be a string")

        if (
            data.get("initialize_sequence_embedding_with_ground_truth", None)
            is not None
        ):
            if not isinstance(
                data["initialize_sequence_embedding_with_ground_truth"], bool
            ):
                raise TypeError(
                    "initialize_sequence_embedding_with_ground_truth must be a bool"
                )

        if data.get("atomize_side_chains", None) is not None:
            if not isinstance(data["atomize_side_chains"], bool):
                raise TypeError("atomize_side_chains must be a bool")

        if data.get("repeat_sample_num", None) is not None:
            if not isinstance(data["repeat_sample_num"], int):
                raise TypeError("repeat_sample_num must be an int")

        if data.get("features_to_return", None) is not None:
            if not isinstance(data["features_to_return"], dict):
                raise TypeError("features_to_return must be a dict")
            for key, value in data["features_to_return"].items():
                if not isinstance(key, str):
                    raise TypeError("features_to_return keys must be strings")
                if not isinstance(value, list):
                    raise TypeError("features_to_return values must be lists")

        # Check that the array-wide user settings are consistent across all
        # atoms in each token.
        atom_array = data["atom_array"]
        token_starts = get_token_starts(atom_array)
        token_level_array = atom_array[token_starts]
        keys_to_check = [
            "mpnn_designed_residue_mask",
            "mpnn_temperature",
            "mpnn_symmetry_equivalence_group",
            "mpnn_symmetry_weight",
            "mpnn_bias",
        ]
        for key in keys_to_check:
            atom_values = get_annotation(atom_array, key)
            if atom_values is not None:
                token_values = get_annotation(token_level_array, key)
                if not np.all(
                    atom_values == spread_token_wise(atom_array, token_values)
                ):
                    raise ValueError(
                        f"All atoms in each token must have the same value for {key}"
                    )

        # Check pair keys such that token-level pairs are unique.
        pair_keys_to_check = [
            "mpnn_pair_bias",
        ]
        token_idx = spread_token_wise(atom_array, np.arange(len(token_level_array)))
        # Only validate 2D annotations if atom_array supports them
        if isinstance(atom_array, AtomArrayPlus):
            annotations_2d = atom_array.get_annotation_2d_categories()
            for key in pair_keys_to_check:
                if key in annotations_2d:
                    annotation = atom_array.get_annotation_2d(key)
                    pairs = annotation.pairs
                    seen_token_pairs = set()
                    for i, j in pairs:
                        token_pair = (token_idx[i], token_idx[j])
                        if token_pair in seen_token_pairs:
                            raise ValueError(
                                f"Token-level pairs must be unique for {key}"
                                " i.e. token pairs should be represented using "
                                "only one atom pair across the tokens."
                            )
                        seen_token_pairs.add(token_pair)

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply user settings to the input features."""
        # +-------- Scalar User Settings --------- +
        # structure_noise (float): the standard deviation of the Gaussian
        # noise to add to the input coordinates, in Angstroms.
        structure_noise = data.get("structure_noise", None)
        if structure_noise is None:
            structure_noise = (
                0.0 if self.is_inference else self.train_structure_noise_default
            )

        # decode_type (str): the type of decoding to use.
        # - "teacher_forcing": Use teacher forcing for the
        #     decoder, where the decoder attends to the ground
        #     truth sequence S for all previously decoded
        #     residues.
        # - "auto_regressive": Use auto-regressive decoding,
        #     where the decoder attends to the sequence and
        #     decoder representation of residues that have
        #     already been decoded (using the predicted sequence).
        decode_type = data.get("decode_type", None)
        if decode_type is None:
            decode_type = "auto_regressive" if self.is_inference else "teacher_forcing"

        # causality_pattern (str): The pattern of causality to use for the
        # decoder. For all causality patterns, the decoding order is randomized.
        # - "auto_regressive": Use an auto-regressive causality
        #     pattern, where residues can attend to the sequence
        #     and decoder representation of residues that have
        #     already been decoded (NOTE: as mentioned above,
        #     this will be randomized).
        # - "unconditional": Residues cannot attend to the
        #     sequence or decoder representation of any other
        #     residues.
        # - "conditional": Residues can attend to the sequence
        #     and decoder representation of all other residues.
        # - "conditional_minus_self": Residues can attend to the
        #     sequence and decoder representation of all other
        #     residues, except for themselves (as destination
        #     nodes).
        causality_pattern = data.get("causality_pattern", None)
        if causality_pattern is None:
            causality_pattern = "auto_regressive"

        # initialize_sequence_embedding_with_ground_truth (bool):
        # - True: Initialize the sequence embedding with the ground truth
        # sequence S.
        #     - If doing auto-regressive decoding, also
        #         initialize S_sampled with the ground truth
        #         sequence S, which should only affect the
        #         application of pair bias.
        # - False: Initialize the sequence embedding with zeros.
        #     - If doing auto-regressive decoding, initialize
        #         S_sampled with unknown residues.
        initialize_sequence_embedding_with_ground_truth = data.get(
            "initialize_sequence_embedding_with_ground_truth", None
        )
        if initialize_sequence_embedding_with_ground_truth is None:
            initialize_sequence_embedding_with_ground_truth = (
                False if self.is_inference else True
            )

        # atomize_side_chains (bool): Whether to atomize side chains of fixed
        # residues.
        atomize_side_chains = data.get("atomize_side_chains", None)
        if atomize_side_chains is None:
            if data["model_type"] == "ligand_mpnn":
                atomize_side_chains = False if self.is_inference else True
            else:
                atomize_side_chains = False

        # repeat_sample_num (int, optional): Number of times to
        # repeat the samples along the batch dimension. If None,
        # no repetition is performed. If greater than 1, the
        # samples are repeated along the batch dimension. If
        # greater than 1, B must be 1, since repeating samples
        # along the batch dimension is not supported when more
        # than one sample is provided in the batch.
        # NOTE: default is None, so no conditional needed.
        repeat_sample_num = data.get("repeat_sample_num", None)

        # features_to_return (dict, optional): dictionary
        # determining which features to return from the model. If
        # None, return all features (including modified input
        # features, graph features, encoder features, and decoder
        # features). Otherwise, expects a dictionary with the
        # following key, value pairs:
        # - "input_features": list - the input features to return.
        # - "graph_features": list - the graph features to return.
        # - "encoder_features": list - the encoder features to
        #     return.
        # - "decoder_features": list - the decoder features to
        #     return.
        features_to_return = data.get("features_to_return", None)
        if features_to_return is None:
            if self.minimal_return:
                features_to_return = {
                    "input_features": [
                        "mask_for_loss",
                    ],
                    "decoder_features": ["log_probs", "S_sampled", "S_argmax"],
                }

        # Save the scalar settings.
        data["input_features"].update(
            {
                "structure_noise": structure_noise,
                "decode_type": decode_type,
                "causality_pattern": causality_pattern,
                "initialize_sequence_embedding_with_ground_truth": initialize_sequence_embedding_with_ground_truth,
                "atomize_side_chains": atomize_side_chains,
                "repeat_sample_num": repeat_sample_num,
                "features_to_return": features_to_return,
            }
        )

        # +-------- Array-Wide User Settings --------- +
        # Extract atom array.
        atom_array = data["atom_array"]

        # Subset to non-atomized.
        non_atomized_array = atom_array[~atom_array.atomize]

        # Get token starts for non-atomized tokens.
        non_atomized_token_starts = get_token_starts(non_atomized_array)
        non_atomized_token_level = non_atomized_array[non_atomized_token_starts]

        # Project token indices for non-atomized tokens.
        non_atomized_token_idx = spread_token_wise(
            non_atomized_array, np.arange(len(non_atomized_token_level))
        )

        if get_annotation(non_atomized_array, "mpnn_designed_residue_mask") is not None:
            designed_residue_mask = (
                non_atomized_token_level.mpnn_designed_residue_mask.astype(np.bool_)
            )
        else:
            designed_residue_mask = None

        if get_annotation(non_atomized_array, "mpnn_temperature") is not None:
            temperature = non_atomized_token_level.mpnn_temperature.astype(np.float32)
        else:
            if self.is_inference:
                temperature = 0.1 * np.ones(
                    len(non_atomized_token_level), dtype=np.float32
                )
            else:
                temperature = None

        if (
            get_annotation(non_atomized_array, "mpnn_symmetry_equivalence_group")
            is not None
        ):
            symmetry_equivalence_group = (
                non_atomized_token_level.mpnn_symmetry_equivalence_group.astype(
                    np.int32
                )
            )
        else:
            symmetry_equivalence_group = None

        if get_annotation(non_atomized_array, "mpnn_symmetry_weight") is not None:
            symmetry_weight = non_atomized_token_level.mpnn_symmetry_weight.astype(
                np.float32
            )
        else:
            symmetry_weight = None

        if get_annotation(non_atomized_array, "mpnn_bias") is not None:
            bias = non_atomized_token_level.mpnn_bias.astype(np.float32)
        else:
            bias = None

        if (
            isinstance(non_atomized_array, AtomArrayPlus)
            and "mpnn_pair_bias" in non_atomized_array.get_annotation_2d_categories()
        ):
            pair_bias_sparse = non_atomized_array.get_annotation_2d("mpnn_pair_bias")
            pair_bias = np.zeros(
                (
                    len(non_atomized_token_level),
                    pair_bias_sparse.values.shape[1],
                    len(non_atomized_token_level),
                    pair_bias_sparse.values.shape[2],
                ),
                dtype=np.float32,
            )
            for idx in range(pair_bias_sparse.values.shape[0]):
                i, j, pair_bias_ij = pair_bias_sparse[idx]
                token_idx_i = non_atomized_token_idx[i]
                token_idx_j = non_atomized_token_idx[j]
                pair_bias[token_idx_i, :, token_idx_j, :] = pair_bias_ij

        else:
            pair_bias = None

        # Save the array-wide settings.
        data["input_features"].update(
            {
                "designed_residue_mask": designed_residue_mask,
                "temperature": temperature,
                "symmetry_equivalence_group": symmetry_equivalence_group,
                "symmetry_weight": symmetry_weight,
                "bias": bias,
                "pair_bias": pair_bias,
            }
        )

        return data
