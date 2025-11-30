import torch
import torch.nn as nn


class PositionalEncodings(nn.Module):
    def __init__(self, num_positional_embeddings, max_relative_feature=32):
        """
        Positional encodings for the MPNN model.

        Args:
            num_positional_embeddings (int): The dimension of the embeddings for
                the positional encodings.
            max_relative_feature (int): The maximum relative feature offset.
                Default is 32, which means the positional encodings will handle
                offsets in the range [-32, 32]. This is used to determine the
                size of the one-hot encoding for the positional offsets.
        """
        super(PositionalEncodings, self).__init__()

        # Store the number of embeddings and the maximum relative feature.
        self.num_positional_embeddings = num_positional_embeddings
        self.max_relative_feature = max_relative_feature

        # We reserve enough space for the -max_relative_feature,...,0,...,
        # max_relative_feature, plus an additional input for residue pairs not
        self.num_positional_features = 2 * max_relative_feature + 1 + 1

        # Initialize the linear layer that will map the one-hot encoding of the
        # positional offsets to the embeddings.
        self.embed_positional_features = nn.Linear(
            self.num_positional_features, num_positional_embeddings
        )

    def forward(self, positional_offset, same_chain_mask):
        """
        Forward pass of the positional encodings.

        Args:
            positional_offset (torch.Tensor): [B, L, K] - pairwise differences
                between the indices of residues, gathered for the K nearest
                neighbors.
            same_chain_mask (torch.Tensor): [B, L, K] - a mask indicating
                whether the residues are on the same chain (1) or not (0).
        Returns:
            positional_offset_embeddings (torch.Tensor): [B, L, K,
                self.num_positional_embeddings] - the embeddings for the
                positional offsets, where each offset shifted and clipped to
                the range [0, 2 * max_relative_feature], with a special value
                of (2 * max_relative_feature + 1) for residues not on the same
                chain. The embeddings are obtained by passing the one-hot
                encoding of the chain-aware clipped positional offsets through
                a linear layer.
        """
        # Check that the same chain mask has a boolean dtype.
        if same_chain_mask.dtype != torch.bool:
            raise ValueError("The same_chain_mask must be of boolean dtype.")

        # shifted_positional_offset [B, L, K] - the positional offset shifted
        # by the maximum relative feature.
        shifted_positional_offset = positional_offset + self.max_relative_feature

        # clipped_positional_offset [B, L, K] - the shifted positional offset
        # clipped to the range [0, 2 * max_relative_feature]. Combining the
        # shifting and clipping, this captures original positional offsets in
        # the range [-max_relative_feature, max_relative_feature], shifting
        # them to the range [0, 2 * max_relative_feature], clipping any values
        # outside this range to the nearest valid value. The shifting to non-
        # negative values is necessary for the one-hot encoding.
        clipped_positional_offset = torch.clip(
            shifted_positional_offset, 0, 2 * self.max_relative_feature
        )

        # clipped_positional_offset_chain_aware [B, L, K] - the clipped
        # positional offset, where the values for residues on the same chain
        # are preserved, and the values for residues not on the same chain are
        # set to a special value (2 * max_relative_feature + 1). This is
        # done to ensure that the positional embeddings are chain-aware.
        clipped_positional_offset_chain_aware = (
            clipped_positional_offset * same_chain_mask
            + (~same_chain_mask) * (2 * self.max_relative_feature + 1)
        )

        # clipped_positional_offset_chain_aware_onehot [B, L, K,
        # self.num_positional_features] - the chained-aware clipped positional
        # offset converted to a one-hot encoding.
        clipped_positional_offset_chain_aware_onehot = torch.nn.functional.one_hot(
            clipped_positional_offset_chain_aware,
            num_classes=self.num_positional_features,
        ).float()

        # positional_offset_embeddings [B, L, K, self.num_positional_embeddings]
        # - the embeddings for the positional offsets, obtained by passing the
        # one-hot encoding through a linear layer.
        positional_offset_embeddings = self.embed_positional_features(
            clipped_positional_offset_chain_aware_onehot
        )

        return positional_offset_embeddings
