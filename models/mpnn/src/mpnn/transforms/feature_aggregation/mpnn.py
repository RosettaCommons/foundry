from typing import Any

import numpy as np
from atomworks.common import KeyToIntMapper
from atomworks.ml.transforms._checks import (
    check_atom_array_annotation,
)
from atomworks.ml.transforms.base import Transform
from atomworks.ml.transforms.encoding import atom_array_to_encoding
from atomworks.ml.utils.token import get_token_starts
from mpnn.transforms.feature_aggregation.token_encodings import MPNN_TOKEN_ENCODING


class EncodeMPNNNonAtomizedTokens(Transform):
    """Encode non-atomized tokens for MPNN with X, X_m, and S features.

    Creates:
    - X: (L, 37, 3) coordinates for non-atomized tokens
    - X_m: (L, 37) mask for atom existence and occupancy > occupancy_threshold
    - S: (L) sequence encoding

    Args:
        occupancy_threshold (float): Minimum occupancy to consider atom as present. Defaults to 0.5.
    """

    def __init__(self, occupancy_threshold: float = 0.5):
        self.occupancy_threshold = occupancy_threshold
        self.encoding = MPNN_TOKEN_ENCODING

    def check_input(self, data: dict[str, Any]) -> None:
        check_atom_array_annotation(data, ["atomize", "res_name", "occupancy"])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array = data["atom_array"]

        # Check that atom_array is not empty.
        assert len(atom_array) > 0, "atom_array cannot be empty"

        # Get non-atomized tokens only
        non_atomized_mask = ~atom_array.atomize
        non_atomized_array = atom_array[non_atomized_mask]

        assert len(non_atomized_array) > 0, "No non-atomized atoms found"

        if len(non_atomized_array) == 0:
            # No non-atomized tokens, create empty arrays
            data["input_features"].update(
                {
                    "X": np.zeros((0, 37, 3), dtype=np.float32),
                    "X_m": np.zeros((0, 37), dtype=np.bool_),
                    "S": np.zeros((0,), dtype=np.int64),
                }
            )
            return data

        # Encode using the MPNN token encoding
        encoded = atom_array_to_encoding(
            non_atomized_array,
            encoding=self.encoding,
            default_coord=0.0,  # Use 0.0 instead of NaN for MPNN
            occupancy_threshold=self.occupancy_threshold,
        )

        # Create X: coordinates (L, 37, 3)
        X = encoded["xyz"].astype(np.float32)

        # Create X_m: mask for existence and occupancy > threshold (L, 37)
        # encoded["mask"] already considers occupancy, we just need to check if atoms exist in encoding
        X_m = encoded["mask"].astype(np.bool_)

        # Create S: sequence encoding (L,)
        S = encoded["seq"].astype(np.int64)

        data["input_features"].update(
            {
                "X": X,
                "X_m": X_m,
                "S": S,
            }
        )

        # Check that we have at least one non-atomized token.
        L = X.shape[0]
        assert L > 0, "At least one non-atomized token should be present"

        return data


class FeaturizeNonAtomizedTokens(Transform):
    """Add additional features for non-atomized tokens: R_idx, chain_labels, residue_mask."""

    def check_input(self, data: dict[str, Any]) -> None:
        check_atom_array_annotation(
            data, ["atomize", "within_chain_res_idx", "chain_iid"]
        )

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array = data["atom_array"]

        # Subset to non-atomized.
        non_atomized_array = atom_array[~atom_array.atomize]

        # Get token starts for non-atomized tokens
        non_atomized_token_starts = get_token_starts(non_atomized_array)
        non_atomized_token_level = non_atomized_array[non_atomized_token_starts]

        if len(non_atomized_token_level) == 0:
            # No non-atomized tokens
            data["input_features"].update(
                {
                    "R_idx": np.zeros((0,), dtype=np.int32),
                    "chain_labels": np.zeros((0,), dtype=np.int64),
                    "residue_mask": np.zeros((0,), dtype=np.bool_),
                }
            )
            return data

        # R_idx: residue indices within chains (0-indexed)
        R_idx = non_atomized_token_level.within_chain_res_idx.astype(np.int32)

        # chain_labels: convert chain_iid to unique integers
        chain_mapper = KeyToIntMapper()
        chain_labels = np.array(
            [
                chain_mapper(chain_iid)
                for chain_iid in non_atomized_token_level.chain_iid
            ],
            dtype=np.int64,
        )

        # residue_mask: all 1's for non-atomized tokens
        residue_mask = np.ones(len(non_atomized_token_level), dtype=np.bool_)

        data["input_features"].update(
            {
                "R_idx": R_idx,
                "chain_labels": chain_labels,
                "residue_mask": residue_mask,
            }
        )
        return data


class FeaturizeAtomizedTokens(Transform):
    """Add features for atomized tokens: Y, Y_t, Y_m."""

    def check_input(self, data: dict[str, Any]) -> None:
        check_atom_array_annotation(data, ["atomize", "atomic_number"])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array = data["atom_array"]

        # Get atomized tokens only
        atomized_mask = atom_array.atomize
        atomized_array = atom_array[atomized_mask]

        if len(atomized_array) == 0:
            # No atomized tokens
            data["input_features"].update(
                {
                    "Y": np.zeros((0, 3), dtype=np.float32),
                    "Y_t": np.zeros((0,), dtype=np.int32),
                    "Y_m": np.zeros((0,), dtype=np.bool_),
                }
            )
            return data

        # Y: coordinates of atomized tokens (n_atomized, 3)
        Y = atomized_array.coord.astype(np.float32)

        # Y_t: atomic numbers of atomized tokens (n_atomized,)
        Y_t = atomized_array.atomic_number.astype(np.int32)

        # Y_m: mask for atomized tokens (all 1's since they exist) (n_atomized,)
        Y_m = np.ones(len(atomized_array), dtype=np.bool_)

        data["input_features"].update(
            {
                "Y": Y,
                "Y_t": Y_t,
                "Y_m": Y_m,
            }
        )
        return data
