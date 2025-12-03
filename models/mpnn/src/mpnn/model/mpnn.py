import torch
import torch.nn as nn
from atomworks.constants import UNKNOWN_AA
from mpnn.model.layers.graph_embeddings import (
    ProteinFeatures,
    ProteinFeaturesLigand,
    ProteinFeaturesMembrane,
    ProteinFeaturesPSSM,
)
from mpnn.model.layers.message_passing import (
    DecLayer,
    EncLayer,
    cat_neighbors_nodes,
    gather_nodes,
)
from mpnn.utils.probability import sample_bernoulli_rv


class ProteinMPNN(nn.Module):
    """
    Class for default ProteinMPNN.
    """

    HAS_NODE_FEATURES = False

    @staticmethod
    def init_weights(module):
        """
        Initialize the weights of the module.

        Args:
            module (nn.Module): The module to initialize.
        Side Effects:
            Initializes the weights of the module using Xavier uniform
            initialization for parameters with a dimension greater than 1.
        """
        # Initialize the weights of the module using Xavier uniform, skipping
        # any parameters with a dimension of 1 or less (for example, biases).
        for parameter in module.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def __init__(
        self,
        num_node_features=128,
        num_edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        num_neighbors=48,
        dropout_rate=0.1,
        num_positional_embeddings=16,
        min_rbf_mean=2.0,
        max_rbf_mean=22.0,
        num_rbf=16,
        graph_featurization_module=None,
    ):
        """
        Setup the ProteinMPNN model.

        Args:
            num_node_features (int): Number of node features.
            num_edge_features (int): Number of edge features.
            hidden_dim (int): Hidden dimension size.
            num_encoder_layers (int): Number of encoder layers.
            num_decoder_layers (int): Number of decoder layers.
            num_neighbors (int): Number of neighbors for each polymer residue.
            dropout_rate (float): Dropout rate.
            num_positional_embeddings (int): Number of positional embeddings.
            min_rbf_mean (float): Minimum radial basis function mean.
            max_rbf_mean (float): Maximum radial basis function mean.
            num_rbf (int): Number of radial basis functions.
            graph_featurization_module (nn.Module, optional): Custom graph
                featurization module. If None, the default ProteinFeatures
                module is used.
        """
        super(ProteinMPNN, self).__init__()

        # Internal dimensions
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.hidden_dim = hidden_dim

        # Number of layers in the encoder and decoder
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers

        # Dropout rate
        self.dropout_rate = dropout_rate

        # Module for featurizing the graph.
        if graph_featurization_module is not None:
            self.graph_featurization_module = graph_featurization_module
        else:
            self.graph_featurization_module = ProteinFeatures(
                num_edge_output_features=num_edge_features,
                num_node_output_features=num_node_features,
                num_positional_embeddings=num_positional_embeddings,
                min_rbf_mean=min_rbf_mean,
                max_rbf_mean=max_rbf_mean,
                num_rbf=num_rbf,
                num_neighbors=num_neighbors,
            )

        # Provide a shorter reference to the graph featurization token-to-idx
        # mapping.
        self.token_to_idx = self.graph_featurization_module.TOKEN_ENCODING.token_to_idx

        # Size of the vocabulary
        self.vocab_size = self.graph_featurization_module.TOKEN_ENCODING.n_tokens

        # Unknown residue token indices, from the TOKEN_ENCODING.
        self.unknown_token_indices = list(
            map(
                lambda token: self.token_to_idx[token],
                self.graph_featurization_module.TOKEN_ENCODING.unknown_tokens,
            )
        )

        # Linear layer for the edge features.
        self.W_e = nn.Linear(num_edge_features, hidden_dim, bias=True)

        # Linear layer for the sequence features.
        self.W_s = nn.Embedding(self.vocab_size, hidden_dim)

        if self.HAS_NODE_FEATURES:
            # Linear layer for the node features.
            self.W_v = nn.Linear(num_node_features, hidden_dim, bias=True)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Encoder layers
        self.encoder_layers = nn.ModuleList(
            [
                EncLayer(hidden_dim, hidden_dim * 3, dropout=dropout_rate)
                for _ in range(num_encoder_layers)
            ]
        )

        # Decoder layers
        self.decoder_layers = nn.ModuleList(
            [
                DecLayer(hidden_dim, hidden_dim * 4, dropout=dropout_rate)
                for _ in range(num_decoder_layers)
            ]
        )

        # Linear layer for the output
        self.W_out = nn.Linear(hidden_dim, self.vocab_size, bias=True)

    def construct_known_residue_mask(self, S):
        """
        Construct a mask for the known residues based on the sequence S.

        Args:
            S (torch.Tensor): [B, L] - the sequence of residues.
        Returns:
            known_residue_mask (torch.Tensor): [B, L] - mask for known residues,
                where True is a residue with one of the canonical residue types,
                and False is a residue with an unknown residue type.
        """
        # Create a mask for known residues.
        known_residue_mask = torch.isin(
            S,
            torch.tensor(self.unknown_token_indices, device=S.device, dtype=S.dtype),
            invert=True,
        )

        return known_residue_mask

    def sample_and_construct_masks(self, input_features):
        """
        Sample and construct masks for the input features.

        Args:
            input_features (dict): Input features containing the residue mask.
                - residue_mask (torch.Tensor): [B, L] - mask for the residues.
                - S (torch.Tensor): [B, L] - sequence of residues.
                - designed_residue_mask (torch.Tensor): [B, L] - mask for the
                    designed residues.
        Side Effects:
            input_features["residue_mask"] (torch.Tensor): [B, L] - mask for the
                residues, where True is a residue that is valid and False is a
                residue that is invalid.
            input_features["known_residue_mask"] (torch.Tensor): [B, L] - mask
                for known residues, where True is a residue with one of the
                canonical residue types, and False is a residue with an unknown
                residue type.
            input_features["designed_residue_mask"] (torch.Tensor): [B, L] -
                mask for designed residues, where True is a residue that is
                designed, and False is a residue that is not designed.
            input_features["mask_for_loss"] (torch.Tensor): [B, L] - mask for
                loss, where True is a residue that is included in the loss
                calculation, and False is a residue that is not included in the
                loss calculation.
        """
        # Check that the input features contain the necessary keys.
        if "residue_mask" not in input_features:
            raise ValueError("Input features must contain 'residue_mask' key.")
        if "S" not in input_features:
            raise ValueError("Input features must contain 'S' key.")
        if "designed_residue_mask" not in input_features:
            raise ValueError("Input features must contain 'designed_residue_mask' key.")

        # Ensure that the residue_mask is a boolean.
        input_features["residue_mask"] = input_features["residue_mask"].bool()

        # Mask is true for canonical residues, false for unknown residues.
        input_features["known_residue_mask"] = self.construct_known_residue_mask(
            input_features["S"]
        )

        # Mask for residues that are designed. If the designed_residue_mask
        # is None, then we assume that all valid residues are designed.
        if input_features["designed_residue_mask"] is None:
            input_features["designed_residue_mask"] = input_features[
                "residue_mask"
            ].clone()
        else:
            input_features["designed_residue_mask"] = input_features[
                "designed_residue_mask"
            ].bool()

        # Chech that the designed_residue_mask is a subset of valid residues.
        if not torch.all(
            input_features["residue_mask"][input_features["designed_residue_mask"]]
        ):
            raise ValueError("Designed residues must all be valid residues.")

        # Mask for loss.
        input_features["mask_for_loss"] = (
            input_features["residue_mask"]
            & input_features["known_residue_mask"]
            & input_features["designed_residue_mask"]
        )

    def graph_featurization(self, input_features):
        """
        Apply the graph featurization to the input features.

        Args:
            input_features (dict): Input features to be featurized.
        Returns:
            graph_features (dict): Featurized graph features (contains both node
                and edge features).
        """
        graph_features = self.graph_featurization_module(input_features)

        return graph_features

    def encode(self, input_features, graph_features):
        """
        Encode the protein features with message passing.

        # NOTE: K = self.num_neighbors
        Args:
            input_features (dict): Input features containing the residue mask.
                - residue_mask (torch.Tensor): [B, L] - mask for the residues.
            graph_features (dict): Graph features containing the featurized
                node and edge inputs.
                - E (torch.Tensor): [B, L, K, self.num_edge_features] - edge
                    features.
                - E_idx (torch.Tensor): [B, L, K] - edge indices.
                - V (torch.Tensor, optional): [B, L, self.num_node_features] -
                    node features (if HAS_NODE_FEATURES is True).
        Returns:
            encoder_features (dict): Encoded features containing the encoded
                protein node and protein edge features.
                - h_V (torch.Tensor): [B, L, self.hidden_dim] - the protein node
                    features after encoding message passing.
                - h_E (torch.Tensor): [B, L, K, self.hidden_dim] - the protein
                    edge features after encoding message passing.
        """
        # Check that the input features contains the necessary keys.
        if "residue_mask" not in input_features:
            raise ValueError("Input features must contain 'residue_mask' key.")

        # Check that the graph features contains the necessary keys.
        if "E" not in graph_features:
            raise ValueError("Graph features must contain 'E' key.")
        if "E_idx" not in graph_features:
            raise ValueError("Graph features must contain 'E_idx' key.")

        B, L, _, _ = graph_features["E"].shape

        # Embed the node features.
        # h_V [B, L, self.num_node_features] - the embedding of the node
        # features.
        if self.HAS_NODE_FEATURES:
            if "V" not in graph_features:
                raise ValueError("Graph features must contain 'V' key.")
            h_V = self.W_v(graph_features["V"])
        else:
            h_V = torch.zeros(
                (B, L, self.num_node_features), device=graph_features["E"].device
            )

        # Embed the edge features.
        # h_E [B, L, K, self.edge_features] - the embedding of the edge
        # features.
        h_E = self.W_e(graph_features["E"])

        # Gather the per-residue mask of the nearest neighbors.
        # mask_E [B, L, K] - the mask for the edges, gathered at the
        # neighbor indices.
        mask_E = gather_nodes(
            input_features["residue_mask"].unsqueeze(-1), graph_features["E_idx"]
        ).squeeze(-1)
        mask_E = input_features["residue_mask"].unsqueeze(-1) * mask_E

        # Perform the message passing in the encoder.
        for layer in self.encoder_layers:
            # h_V [B, L, self.hidden_dim] - the updated node features.
            # h_E [B, L, K, self.hidden_dim] - the updated edge features.
            h_V, h_E = torch.utils.checkpoint.checkpoint(
                layer,
                h_V,
                h_E,
                graph_features["E_idx"],
                mask_V=input_features["residue_mask"],
                mask_E=mask_E,
                use_reentrant=False,
            )

        # Create the encoder features dictionary.
        encoder_features = {
            "h_V": h_V,
            "h_E": h_E,
        }

        return encoder_features

    def setup_causality_masks(self, input_features, graph_features, decoding_eps=1e-4):
        """
        Setup the causality masks for the decoder. This can involve sampling
        the decoding order.

        Args:
            input_features (dict): Input features containing the residue mask.
                - residue_mask (torch.Tensor): [B, L] - mask for the residues.
                - designed_residue_mask (torch.Tensor): [B, L] - mask for the
                    designed residues.
                - symmetry_equivalence_group (torch.Tensor, optional): [B, L] -
                    an integer for every residue, indicating the symmetry group
                    that it belongs to. If None, the residues are not grouped by
                    symmetry. For example, if residue i and j should be decoded
                    symmetrically, then symmetry_equivalence_group[i] ==
                    symmetry_equivalence_group[j]. Must be torch.int64 to allow
                    for use as an index. These values should range from 0 to
                    the maximum number of symmetry groups - 1 for each example.
                - causality_pattern (str): The pattern of causality to use for
                    the decoder.
            graph_features (dict): Graph features containing the featurized
                node and edge inputs.
                - E_idx (torch.Tensor): [B, L, K] - edge indices.
            decoding_eps (float): Small epsilon value added to the
                decode_last_mask to prevent the case where every randomly
                sampled number is multiplied by zero, which would result
                in an incorrect decoding order.
        Returns:
            decoder_features (dict): Decoding features containing the decoding
                order and masks for the decoder.
                - causal_mask (torch.Tensor): [B, L, K, 1] - the causal mask for
                    the decoder.
                - anti_causal_mask (torch.Tensor): [B, L, K, 1] - the
                    anti-causal mask for the decoder.
                - decoding_order (torch.Tensor): [B, L] - the order in which the
                    residues should be decoded.
                - decode_last_mask (torch.Tensor): [B, L] - mask for residues
                    that should be decoded last, where False is a residue that
                    should be decoded first (invalid or fixed), and True is a
                    residue that should not be decoded first (designed
                    residues).
        """
        # Check that the input features contains the necessary keys.
        if "residue_mask" not in input_features:
            raise ValueError("Input features must contain 'residue_mask' key.")
        if "designed_residue_mask" not in input_features:
            raise ValueError("Input features must contain 'designed_residue_mask' key.")
        if "symmetry_equivalence_group" not in input_features:
            raise ValueError(
                "Input features must contain 'symmetry_equivalence_group' key."
            )
        if "causality_pattern" not in input_features:
            raise ValueError("Input features must contain 'causality_pattern' key.")

        # Check that the encoder features contains the necessary keys.
        if "E_idx" not in graph_features:
            raise ValueError("Graph features must contain 'E_idx' key.")

        B, L = input_features["residue_mask"].shape

        # decode_last_mask [B, L] - mask for residues that should be
        # decoded last, where False is a residue that should be decoded first
        # (invalid or fixed), and True is a residue that should not be decoded
        # first (designed residues).
        decode_last_mask = (
            input_features["residue_mask"] & input_features["designed_residue_mask"]
        ).bool()

        # Compute the noise for the decoding order.
        if input_features["symmetry_equivalence_group"] is None:
            # noise [B, L] - the noise for each residue, sampled from a normal
            # distribution. This is used to randomly sample the decoding order.
            noise = torch.randn((B, L), device=input_features["residue_mask"].device)
        else:
            # Assume that all symmetry groups are non-negative.
            assert input_features["symmetry_equivalence_group"].min() >= 0

            # Compute the maximum number of symmetry groups.
            G = int(input_features["symmetry_equivalence_group"].max().item()) + 1

            # noise_per_group [B, G] - the noise for each
            # symmetry group, sampled from a normal distribution.
            noise_per_group = torch.randn(
                (B, G), device=input_features["residue_mask"].device
            )

            # batch_idx [B, 1] - the batch indices for each example.
            batch_idx = torch.arange(
                B, device=input_features["residue_mask"].device
            ).unsqueeze(-1)

            # noise [B, L] - the noise for each residue, sampled from a normal
            # distribution, where the noise is the same for residues in the same
            # symmetry group.
            noise = noise_per_group[
                batch_idx, input_features["symmetry_equivalence_group"]
            ]

        # decoding_order [B, L] - the order in which the residues should be
        # decoded. Specifically, decoding_order[b, i] = j specifies that the
        # jth residue should be decoded ith. Sampled for every example.
        # Numbers will be smaller where decode_last_mask is False (0), and
        # larger where decode_last_mask is True (1), leading to the appropriate
        # index ordering after the argsort.
        decoding_order = torch.argsort(
            (decode_last_mask.float() + decoding_eps) * torch.abs(noise), dim=-1
        )

        # permutation_matrix_reverse [B, L, L] - the reverse permutation
        # matrix (the transpose/inverse of the permutation matrix) computed from
        # the decoding order; such that permutation_matrix_reverse[i, j] = 1 if
        # the ith entry in the original will be sent to the jth position (the
        # ith row/column in the original all by all causal mask will be sent
        # to the jth row/column in the permuted all by all causal mask).
        permutation_matrix_reverse = torch.nn.functional.one_hot(
            decoding_order, num_classes=L
        ).float()

        # Create the all by all causal mask for the decoder.
        # causal_mask_all_by_all [L, L] - the all by all causal mask for the
        # decoder, constructed based on the specified causality pattern.
        if input_features["causality_pattern"] == "auto_regressive":
            # left_to_right_causal_mask [L, L] - the causal mask for the
            # left-to-right attention (lower triangular with zeros on the
            # diagonal). Residue at position i can "see" residues at positions
            # j < i, but not at positions j >= i.
            left_to_right_causal_mask = 1 - torch.triu(
                torch.ones(L, L, device=input_features["residue_mask"].device)
            )

            causal_mask_all_by_all = left_to_right_causal_mask
        elif input_features["causality_pattern"] == "unconditional":
            # zeros_causal_mask [L, L] - the causal mask for the decoder,
            # where all entries are zeros. Residue at position i cannot see
            # any other residues, including itself.
            zeros_causal_mask = torch.zeros(
                (L, L), device=input_features["residue_mask"].device
            )

            causal_mask_all_by_all = zeros_causal_mask
        elif input_features["causality_pattern"] == "conditional":
            # ones_causal_mask [L, L] - the causal mask for the decoder,
            # where all entries are ones. Residue at position i can see all
            # other residues, including itself.
            ones_causal_mask = torch.ones(
                (L, L), device=input_features["residue_mask"].device
            )

            causal_mask_all_by_all = ones_causal_mask
        elif input_features["causality_pattern"] == "conditional_minus_self":
            # I [L, L] - the identity matrix, repeated along the batch.
            I = torch.eye(L, device=input_features["residue_mask"].device)

            # ones_minus_self_causal_mask [L, L] - the causal mask for the
            # decoder, where all entries are ones except for the diagonal
            # entries, which are zeros. Residue at position i can see all other
            # residues, but not itself.
            ones_minus_self_causal_mask = (
                torch.ones((L, L), device=input_features["residue_mask"].device) - I
            )

            causal_mask_all_by_all = ones_minus_self_causal_mask
        else:
            raise ValueError(
                "Unknown causality pattern: " + f"{input_features['causality_pattern']}"
            )

        # permuted_causal_mask_all_by_all [B, L, L] - the causal mask for the
        # decoder, permuted according to the decoding order.
        permuted_causal_mask_all_by_all = torch.einsum(
            "ij, biq, bjp->bqp",
            causal_mask_all_by_all,
            permutation_matrix_reverse,
            permutation_matrix_reverse,
        )

        # If the symmetry equivalence group is not None, then we need to
        # mask out residues that belong to the same symmetry group.
        if input_features["symmetry_equivalence_group"] is not None:
            # same_symmetry_group [B, L, L] - a mask for the residues that
            # belong to the same symmetry group, where True is a residue pair
            # that belongs to the same symmetry group, and False is a residue
            # pair that does not belong to the same symmetry group.
            same_symmetry_group = (
                input_features["symmetry_equivalence_group"][:, :, None]
                == input_features["symmetry_equivalence_group"][:, None, :]
            )

            permuted_causal_mask_all_by_all[same_symmetry_group] = 0.0

        # causal_mask_nearest_neighbors [B, L, K, 1] - the causal mask for
        # the decoder, gathered at the neighbor indices. This limits the
        # attention to the nearest neighbors.
        causal_mask_nearest_neighbors = torch.gather(
            permuted_causal_mask_all_by_all, 2, graph_features["E_idx"]
        ).unsqueeze(-1)

        # causal_mask [B, L, K, 1] - the final causal mask for the decoder;
        # masked version of causal_mask_nearest_neighbors.
        causal_mask = (
            causal_mask_nearest_neighbors
            * input_features["residue_mask"].view([B, L, 1, 1]).float()
        )

        # anti_causal_mask [B, L, K, 1] - the anti-causal mask for the decoder.
        anti_causal_mask = (1.0 - causal_mask_nearest_neighbors) * input_features[
            "residue_mask"
        ].view([B, L, 1, 1]).float()

        # Add the masks to the decoder features.
        decoder_features = {
            "causal_mask": causal_mask,
            "anti_causal_mask": anti_causal_mask,
            "decoding_order": decoding_order,
            "decode_last_mask": decode_last_mask,
        }

        return decoder_features

    def repeat_along_batch(self, input_features, graph_features, encoder_features):
        """
        Given the input features, graph features, and encoder features,
        repeat the samples along the batch dimension. This is useful during
        inference, to prevent re-running the encoder for every sample (since
        the encoder is deterministic and sequence-agnostic).

        NOTE: if `repeat_sample_num` is not None and greater than 1, then
        B must be 1, since repeating samples along the batch dimension is not
        supported when more than one sample is provided in the batch.

        Args:
            input_features (dict): Input features containing the residue mask
                and sequence.
                - S (torch.Tensor): [B, L] - sequence of residues.
                - residue_mask (torch.Tensor): [B, L] - mask for the residues.
                - temperature (torch.Tensor, optional): [B, L] - the per-residue
                    temperature to use for sampling. If None, the code will
                    implicitly use a temperature of 1.0.
                - bias (torch.Tensor, optional): [B, L, 21] - the per-residue
                    bias to use for sampling. If None, the code will implicitly
                    use a bias of 0.0 for all residues.
                - pair_bias (torch.Tensor, optional): [B, L, 21, L, 21] - the
                    per-residue pair bias to use for sampling. If None, the code
                    will implicitly use a pair bias of 0.0 for all residue
                    pairs.
                - symmetry_equivalence_group (torch.Tensor, optional): [B, L] -
                    an integer for every residue, indicating the symmetry group
                    that it belongs to. If None, the residues are not grouped by
                    symmetry. For example, if residue i and j should be decoded
                    symmetrically, then symmetry_equivalence_group[i] ==
                    symmetry_equivalence_group[j]. Must be torch.int64 to allow
                    for use as an index. These values should range from 0 to
                    the maximum number of symmetry groups - 1 for each example.
                - symmetry_weight (torch.Tensor, optional): [B, L] - the weight
                    for the symmetry equivalence group. If None, the code will
                    implicitly use a weight of 1.0 for all residues.
                - repeat_sample_num (int, optional): Number of times to repeat
                    the samples along the batch dimension. If None, no
                    repetition is performed. If greater than 1, the samples
                    are repeated along the batch dimension. If greater than 1,
                    B must be 1, since repeating samples along the batch
                    dimension is not supported when more than one sample is
                    provided in the batch.
            graph_features (dict): Graph features containing the featurized
                node and edge inputs.
                - E_idx (torch.Tensor): [B, L, K] - edge indices.
            encoder_features (dict): Encoder features containing the encoded
                protein node and protein edge features.
                - h_V (torch.Tensor): [B, L, H] - the protein node features
                    after encoding message passing.
                - h_E (torch.Tensor): [B, L, K, H] - the protein edge features
                    after encoding message passing.
        Side Effects:
            input_features["S"] (torch.Tensor): [repeat_sample_num, L] - the
                sequence of residues, repeated along the batch dimension.
            input_features["residue_mask"] (torch.Tensor):
                [repeat_sample_num, L] - the mask for the residues, repeated
                along the batch dimension.
            input_features["mask_for_loss"] (torch.Tensor):
                [repeat_sample_num, L] - the mask for the loss, repeated
                along the batch dimension.
            input_features["designed_residue_mask"] (torch.Tensor):
                [repeat_sample_num, L] - the mask for designed residues,
                repeated along the batch dimension.
            input_features["temperature"] (torch.Tensor, optional):
                [repeat_sample_num, L] - the per-residue temperature to use for
                sampling, repeated along the batch dimension. If None, the code
                will implicitly use a temperature of 1.0.
            input_features["bias"] (torch.Tensor, optional):
                [repeat_sample_num, L, 21] - the per-residue bias to use for
                sampling, repeated along the batch dimension. If None, the code
                will implicitly use a bias of 0.0 for all residues.
            input_features["pair_bias"] (torch.Tensor, optional):
                [repeat_sample_num, L, 21, L, 21] - the per-residue pair bias
                to use for sampling, repeated along the batch dimension. If
                None, the code will implicitly use a pair bias of 0.0 for all
                residue pairs.
            input_features["symmetry_equivalence_group"] (torch.Tensor,
                optional): [repeat_sample_num, L] - the symmetry equivalence
                group for each residue, repeated along the batch dimension.
            input_features["symmetry_weight"] (torch.Tensor, optional):
                [repeat_sample_num, L] - the symmetry weight for each residue,
                repeated along the batch dimension. If None, the code will
                implicitly use a weight of 1.0 for all residues.
            encoder_features["h_V"] (torch.Tensor): [repeat_sample_num, L, H] -
                the protein node features after encoding message passing,
                repeated along the batch dimension.
            encoder_features["h_E"] (torch.Tensor):
                [repeat_sample_num, L, K, H] - the protein edge features
                after encoding message passing, repeated along the batch
                dimension.
            graph_features["E_idx"] (torch.Tensor): [repeat_sample_num, L, K] -
                the edge indices, repeated along the batch dimension.
        """
        # Check that the input features contains the necessary keys.
        if "S" not in input_features:
            raise ValueError("Input features must contain 'S' key.")
        if "residue_mask" not in input_features:
            raise ValueError("Input features must contain 'residue_mask' key.")
        if "mask_for_loss" not in input_features:
            raise ValueError("Input features must contain 'mask_for_loss' key.")
        if "temperature" not in input_features:
            raise ValueError("Input features must contain 'temperature' key.")
        if "bias" not in input_features:
            raise ValueError("Input features must contain 'bias' key.")
        if "pair_bias" not in input_features:
            raise ValueError("Input features must contain 'pair_bias' key.")
        if "symmetry_equivalence_group" not in input_features:
            raise ValueError(
                "Input features must contain 'symmetry_equivalence_group' key."
            )
        if "symmetry_weight" not in input_features:
            raise ValueError("Input features must contain 'symmetry_weight' key.")
        if "repeat_sample_num" not in input_features:
            raise ValueError("Input features must contain 'repeat_sample_num' key.")

        # Check that the graph features contains the necessary keys.
        if "E_idx" not in graph_features:
            raise ValueError("Graph features must contain 'E_idx' key.")

        # Check that the encoder features contains the necessary keys.
        if "h_V" not in encoder_features:
            raise ValueError("Encoder features must contain 'h_V' key.")
        if "h_E" not in encoder_features:
            raise ValueError("Encoder features must contain 'h_E' key.")

        # Repeating a sample along the batch dimension is not supported
        # when more than one sample is provided in the batch.
        if (
            input_features["repeat_sample_num"] is not None
            and input_features["repeat_sample_num"] > 1
            and input_features["S"].shape[0] > 1
        ):
            raise ValueError(
                "Cannot repeat samples when more than one sample "
                + "is provided in the batch."
            )

        # Repeat the samples along the batch dimension if necessary.
        if (
            input_features["repeat_sample_num"] is not None
            and input_features["repeat_sample_num"] > 1
        ):
            # S [repeat_sample_num, L]
            input_features["S"] = input_features["S"][0].repeat(
                input_features["repeat_sample_num"], 1
            )
            # residue_mask [repeat_sample_num, L]
            input_features["residue_mask"] = input_features["residue_mask"][0].repeat(
                input_features["repeat_sample_num"], 1
            )
            # mask_for_loss [repeat_sample_num, L]
            input_features["mask_for_loss"] = input_features["mask_for_loss"][0].repeat(
                input_features["repeat_sample_num"], 1
            )
            # designed_residue_mask [repeat_sample_num, L]
            input_features["designed_residue_mask"] = input_features[
                "designed_residue_mask"
            ][0].repeat(input_features["repeat_sample_num"], 1)
            if input_features["temperature"] is not None:
                # temperature [repeat_sample_num, L]
                input_features["temperature"] = input_features["temperature"][0].repeat(
                    input_features["repeat_sample_num"], 1
                )
            if input_features["bias"] is not None:
                # bias [repeat_sample_num, L, 21]
                input_features["bias"] = input_features["bias"][0].repeat(
                    input_features["repeat_sample_num"], 1, 1
                )
            if input_features["pair_bias"] is not None:
                # pair_bias [repeat_sample_num, L, 21, L, 21]
                input_features["pair_bias"] = input_features["pair_bias"][0].repeat(
                    input_features["repeat_sample_num"], 1, 1, 1, 1
                )
            if input_features["symmetry_equivalence_group"] is not None:
                # symmetry_equivalence_group [repeat_sample_num, L]
                input_features["symmetry_equivalence_group"] = input_features[
                    "symmetry_equivalence_group"
                ][0].repeat(input_features["repeat_sample_num"], 1)
            if input_features["symmetry_weight"] is not None:
                # symmetry_weight [repeat_sample_num, L]
                input_features["symmetry_weight"] = input_features["symmetry_weight"][
                    0
                ].repeat(input_features["repeat_sample_num"], 1)

            # h_V [repeat_sample_num, L, H]
            encoder_features["h_V"] = encoder_features["h_V"][0].repeat(
                input_features["repeat_sample_num"], 1, 1
            )
            # h_E [repeat_sample_num, L, K, H]
            encoder_features["h_E"] = encoder_features["h_E"][0].repeat(
                input_features["repeat_sample_num"], 1, 1, 1
            )

            # E_idx [repeat_sample_num, L, K]
            graph_features["E_idx"] = graph_features["E_idx"][0].repeat(
                input_features["repeat_sample_num"], 1, 1
            )

    def decode_setup(
        self, input_features, graph_features, encoder_features, decoder_features
    ):
        """
        Given the input features, graph features, encoder features, and initial
        decoder features, set up the decoder for the autoregressive decoding.

        Args:
            input_features (dict): Input features containing the residue mask
                and sequence.
                - S (torch.Tensor): [B, L] - sequence of residues.
                - residue_mask (torch.Tensor): [B, L] - mask for the residues.
                - initialize_sequence_embedding_with_ground_truth (bool):
                    If True, initialize the sequence embedding with the ground
                    truth sequence S. Else, initialize the sequence
                    embedding with zeros.
            graph_features (dict): Graph features containing the featurized
                node and edge inputs.
                - E_idx (torch.Tensor): [B, L, K] - edge indices.
            encoder_features (dict): Encoder features containing the encoded
                protein node and protein edge features.
                - h_V (torch.Tensor): [B, L, H] - the protein node features
                    after encoding message passing.
                - h_E (torch.Tensor): [B, L, K, H] - the protein edge features
                    after encoding message passing.
            decoder_features (dict): Initial decoder features containing the
                anti-causal mask for the decoder.
                - anti_causal_mask (torch.Tensor): [B, L, K, 1] - the
                    anti-causal mask for the decoder.
        Returns:
            h_EXV_encoder_anti_causal (torch.Tensor): [B, L, K, 3H] - the
                encoder embeddings masked with the anti-causal mask.
            mask_E (torch.Tensor): [B, L, K] - the mask for the edges, gathered
                at the neighbor indices.
            h_S (torch.Tensor): [B, L, H] - the sequence embeddings for the
                decoder.
        """
        # Check that the input features contains the necessary keys.
        if "S" not in input_features:
            raise ValueError("Input features must contain 'S' key.")
        if "residue_mask" not in input_features:
            raise ValueError("Input features must contain 'residue_mask' key.")
        if "initialize_sequence_embedding_with_ground_truth" not in input_features:
            raise ValueError(
                "Input features must contain"
                + "'initialize_sequence_embedding_with_ground_truth' key."
            )

        # Check that the graph features contains the necessary keys.
        if "E_idx" not in graph_features:
            raise ValueError("Graph features must contain 'E_idx' key.")

        # Check that the encoder features contains the necessary keys.
        if "h_V" not in encoder_features:
            raise ValueError("Encoder features must contain 'h_V' key.")
        if "h_E" not in encoder_features:
            raise ValueError("Encoder features must contain 'h_E' key.")

        # Check that the decoder features contains the necessary keys.
        if "anti_causal_mask" not in decoder_features:
            raise ValueError("Decoder features must contain 'anti_causal_mask' key.")

        # Build encoder embeddings.
        # h_EX_encoder [B, L, K, 2H] - h_E_ij cat (0 vector); the edge features
        # concatenated with the zero vector, since there is no sequence
        # information in the encoder.
        h_EX_encoder = cat_neighbors_nodes(
            torch.zeros_like(encoder_features["h_V"]),
            encoder_features["h_E"],
            graph_features["E_idx"],
        )

        # h_EXV_encoder [B, L, K, 3H] - h_E_ij cat (0 vector) cat h_V_j; the
        # edge features concatenated with the zero vector and the destination
        # node features from the encoder.
        h_EXV_encoder = cat_neighbors_nodes(
            encoder_features["h_V"], h_EX_encoder, graph_features["E_idx"]
        )

        # h_EXV_encoder_anti_causal [B, L, K, 3H] - the encoder embeddings,
        # masked with the anti-causal mask.
        h_EXV_encoder_anti_causal = h_EXV_encoder * decoder_features["anti_causal_mask"]

        # Gather the per-residue mask of the nearest neighbors.
        # mask_E [B, L, K] - the mask for the edges, gathered at the
        # neighbor indices.
        mask_E = gather_nodes(
            input_features["residue_mask"].unsqueeze(-1), graph_features["E_idx"]
        ).squeeze(-1)
        mask_E = input_features["residue_mask"].unsqueeze(-1) * mask_E

        # Build sequence embedding for the decoder.
        # h_S [B, L, H] - the sequence embeddings for the decoder, obtained by
        # embedding the ground truth sequence S.
        if input_features["initialize_sequence_embedding_with_ground_truth"]:
            h_S = self.W_s(input_features["S"])
        else:
            h_S = torch.zeros_like(encoder_features["h_V"])

        return h_EXV_encoder_anti_causal, mask_E, h_S

    def logits_to_sample(self, logits, bias, pair_bias, S_for_pair_bias, temperature):
        """
        Convert the logits to log probabilities, probabilities, sampled
        probabilities, predicted sequence, and argmax sequence.

        Args:
            logits (torch.Tensor): [B, L, self.vocab_size] - the logits for the
                sequence.
            bias (torch.Tensor, optional): [B, L, self.vocab_size] - the
                bias for the sequence. If None, the code will implicitly use a
                bias of 0.0 for all residues.
            pair_bias (torch.Tensor, optional): [B, L, self.vocab_size, L',
                self.vocab_size] - the pair bias for the sequence. Note,
                L is the length for the logits, and L' is the length for the
                S_for_pair_bias. In some cases, L' may be different from L (
                for example, when the logits are only computed for a subset of
                residues). If None, the code will implicitly use a pair bias
                of 0.0 for all residue pairs.
            S_for_pair_bias (torch.Tensor, optional): [B, L'] - the sequence for
                the pair bias. This is used to compute the total pair bias for
                every position. Allowed to be None if pair_bias is None.
            temperature (torch.Tensor, optional): [B, L] - the per-residue
                temperature to use for sampling. If None, the code will
                implicitly use a temperature of 1.0 for all residues.
        Returns:
            sample_dict (dict): A dictionary containing the following keys:
                - log_probs (torch.Tensor): [B, L, self.vocab_size] - the log
                    probabilities for the sequence.
                - probs (torch.Tensor): [B, L, self.vocab_size] - the
                    probabilities for the sequence.
                - probs_sample (torch.Tensor): [B, L, self.vocab_size] -
                    the probabilities for the sequence, with the unknown
                    residues zeroed out and the other residues normalized.
                - S_sampled (torch.Tensor): [B, L] - the predicted sequence,
                    sampled from the probabilities (unknown residues are not
                    sampled).
                - S_argmax (torch.Tensor): [B, L] - the predicted sequence,
                    obtained by taking the argmax of the probabilities (unknown
                    residues are not selected).
        """
        B, L, vocab_size = logits.shape

        if pair_bias is not None:
            # pair_bias_total [B, L, self.vocab_size] - the total pair bias to
            # add to the sequence logits, computed for every residue by
            # indexing the pair bias with the sequence (S_for_pair_bias) and
            # summing over the second sequence dimension (L').
            pair_bias_total = torch.gather(
                pair_bias,
                -1,
                S_for_pair_bias[:, None, None, :, None].expand(
                    -1, -1, self.vocab_size, -1, -1
                ),
            ).sum(dim=(-2, -1))
        else:
            pair_bias_total = None

        # modified_logits [B, L, self.vocab_size] - the logits for the
        # sequence, modified by temperature, bias, and total pair bias.
        modified_logits = (
            logits
            + (0.0 if bias is None else bias)
            + (0.0 if pair_bias_total is None else pair_bias_total)
        )
        modified_logits = modified_logits / (
            1.0 if temperature is None else temperature.unsqueeze(-1)
        )

        # log_probs [B, L, self.vocab_size] - the log probabilities for the
        # sequence.
        log_probs = torch.nn.functional.log_softmax(modified_logits, dim=-1)

        # probs [B, L, self.vocab_size] - the probabilities for the sequence.
        probs = torch.nn.functional.softmax(modified_logits, dim=-1)

        # probs_sample [B, L, self.vocab_size] - the probabilities for the
        # sequence, with the unknown residues zeroed out and the other residues
        # normalized.
        probs_sample = probs.clone()
        probs_sample[:, :, self.unknown_token_indices] = 0.0
        probs_sample = probs_sample / torch.sum(probs_sample, dim=-1, keepdim=True)

        # probs_sample_flat [B * L, self.vocab_size] - the flattened
        # probabilities for the sequence.
        probs_sample_flat = probs_sample.view(B * L, vocab_size)

        # S_sampled [B, L] - the predicted sequence, sampled from the
        # probabilities (unknown residues are not sampled).
        S_sampled = torch.multinomial(probs_sample_flat, 1).squeeze(-1).view(B, L)

        # S_argmax [B, L] - the predicted sequence, obtained by taking the
        # argmax of the probabilities (unknown residues are not selected).
        S_argmax = torch.argmax(probs_sample, dim=-1)

        sample_dict = {
            "log_probs": log_probs,
            "probs": probs,
            "probs_sample": probs_sample,
            "S_sampled": S_sampled,
            "S_argmax": S_argmax,
        }

        return sample_dict

    def decode_teacher_forcing(
        self, input_features, graph_features, encoder_features, decoder_features
    ):
        """
        Given the input features, graph features, encoder features, and
        decoder features, perform the decoding with teacher forcing.

        Although h_S is computed from the ground truth sequence S, the causal
        mask will ensure that the decoder only attends to the sequence of
        previously decoded residues. Using the ground truth for all previous
        residues is called teacher forcing, and it is a common technique in
        language modeling tasks.

        Args:
            input_features (dict): Input features containing the residue mask
                and sequence.
                - S (torch.Tensor): [B, L] - sequence of residues.
                - residue_mask (torch.Tensor): [B, L] - mask for the residues.
                - bias (torch.Tensor, optional): [B, L, self.vocab_size] - the
                    per-residue bias to use for sampling. If None, the code will
                    implicitly use a bias of 0.0 for all residues.
                - pair_bias (torch.Tensor, optional): [B, L, self.vocab_size
                    , L, self.vocab_size] - the per-residue pair bias to use
                    for sampling. If None, the code will implicitly use a pair
                    bias of 0.0 for all residue pairs.
                - temperature (torch.Tensor, optional): [B, L] - the per-residue
                    temperature to use for sampling. If None, the code will
                    implicitly use a temperature of 1.0.
                - initialize_sequence_embedding_with_ground_truth (bool):
                    If True, initialize the sequence embedding with the ground
                    truth sequence S. Else, initialize the sequence
                    embedding with zeros.
                - symmetry_equivalence_group (torch.Tensor, optional): [B, L] -
                    an integer for every residue, indicating the symmetry group
                    that it belongs to. If None, the residues are not grouped by
                    symmetry. For example, if residue i and j should be decoded
                    symmetrically, then symmetry_equivalence_group[i] ==
                    symmetry_equivalence_group[j]. Must be torch.int64 to allow
                    for use as an index. These values should range from 0 to
                    the maximum number of symmetry groups - 1 for each example.
                -symmetry_weight (torch.Tensor, optional): [B, L] - the weights
                    for each residue, to be used when aggregating across its
                    respective symmetry group. If None, the weights are
                    assumed to be 1.0 for all residues.
            graph_features (dict): Graph features containing the featurized
                node and edge inputs.
                - E_idx (torch.Tensor): [B, L, K] - edge indices.
            encoder_features (dict): Encoder features containing the encoded
                protein node and protein edge features.
                - h_V (torch.Tensor): [B, L, H] - the protein node
                    features after encoding message passing.
                - h_E (torch.Tensor): [B, L, K, H] - the
                    protein edge features after encoding message passing.
            decoder_features (dict): Initial decoder features containing the
                causal mask for the decoder.
                - causal_mask (torch.Tensor): [B, L, K, 1] - the
                    causal mask for the decoder.
                - anti_causal_mask (torch.Tensor): [B, L, K, 1] - the
                    anti-causal mask for the decoder.
        Side Effects:
            decoder_features["h_V"] (torch.Tensor): [B, L, H] - the updated
                node features for the decoder.
            decoder_features["logits"] (torch.Tensor): [B, L, self.vocab_size] -
                the sequence logits for the decoder.
            decoder_features["log_probs"] (torch.Tensor): [B, L,
                self.vocab_size] - the log probabilities for the sequence.
            decoder_features["probs"] (torch.Tensor): [B, L, self.vocab_size] -
                the probabilities for the sequence.
            decoder_features["probs_sample"] (torch.Tensor): [B, L,
                self.vocab_size] - the probabilities for the sequence, with the
                unknown residues zeroed out and the other residues normalized.
            decoder_features["S_sampled"] (torch.Tensor): [B, L] - the
                predicted sequence, sampled from the probabilities (unknown
                residues are not sampled).
            decoder_features["S_argmax"] (torch.Tensor): [B, L] - the predicted
                sequence, obtained by taking the argmax of the probabilities
                (unknown residues are not selected).
        """
        # Check that the input features contains the necessary keys.
        if "residue_mask" not in input_features:
            raise ValueError("Input features must contain 'residue_mask' key.")
        if "S" not in input_features:
            raise ValueError("Input features must contain 'S' key.")
        if "bias" not in input_features:
            raise ValueError("Input features must contain 'bias' key.")
        if "pair_bias" not in input_features:
            raise ValueError("Input features must contain 'pair_bias' key.")
        if "temperature" not in input_features:
            raise ValueError("Input features must contain 'temperature' key.")
        if "symmetry_equivalence_group" not in input_features:
            raise ValueError(
                "Input features must contain 'symmetry_equivalence_group' key."
            )
        if "symmetry_weight" not in input_features:
            raise ValueError("Input features must contain 'symmetry_weight' key.")

        # Check that the graph features contains the necessary keys.
        if "E_idx" not in graph_features:
            raise ValueError("Graph features must contain 'E_idx' key.")

        # Check that the encoder features contains the necessary keys.
        if "h_V" not in encoder_features:
            raise ValueError("Encoder features must contain 'h_V' key.")
        if "h_E" not in encoder_features:
            raise ValueError("Encoder features must contain 'h_E' key.")

        # Check that the decoder features contains the necessary keys.
        if "causal_mask" not in decoder_features:
            raise ValueError("Decoder features must contain 'causal_mask' key.")

        # Do the setup for the decoder.
        # h_EXV_encoder_anti_causal [B, L, K, 3H] - the encoder embeddings,
        # masked with the anti-causal mask.
        # mask_E [B, L, K] - the mask for the edges, gathered at the
        # neighbor indices.
        # h_S [B, L, H] - the sequence embeddings for the decoder.
        h_EXV_encoder_anti_causal, mask_E, h_S = self.decode_setup(
            input_features, graph_features, encoder_features, decoder_features
        )

        # h_ES [B, L, K, 2H] - h_E_ij cat h_S_j; the edge features concatenated
        # with the sequence embeddings for the destination nodes.
        h_ES = cat_neighbors_nodes(
            h_S, encoder_features["h_E"], graph_features["E_idx"]
        )

        # Run the decoder layers.
        h_V_decoder = encoder_features["h_V"]
        for layer in self.decoder_layers:
            # h_ESV_decoder [B, L, K, 3H] - h_E_ij cat h_S_j cat h_V_decoder_j;
            # for the decoder embeddings, the edge features are concatenated
            # with the destination node sequence embeddings and node features.
            h_ESV_decoder = cat_neighbors_nodes(
                h_V_decoder, h_ES, graph_features["E_idx"]
            )

            # h_ESV [B, L, K, 3H] - the encoder and decoder embeddings,
            # combined according to the causal and anti-causal masks.
            # Combine the encoder embeddings with the decoder embeddings,
            # using the causal and anti-causal masks. When decoding the residue
            # at position i:
            #    - for residue j, decoded before i:
            #        - h_ESV_ij = h_E_ij cat h_S_j cat h_V_decoder_j
            #            - encoder edge embedding, decoder destination node
            #              sequence embedding, and decoder destination node
            #              embedding.
            #    - for residue j, decoded after i (including i):
            #        - h_ESV_ij = h_E_ij cat (0 vector) cat h_V_j
            #            - encoder edge embedding, zero vector (no sequence
            #              information), and encoder destination node embedding.
            #              This prevents leakage of sequence information.
            #            - NOTE: h_V_j comes from the encoder.
            #    - NOTE: h_E is not updated in the decoder, h_E_ij comes from
            #      the encoder.
            #    - NOTE: within the decoder layer itself, h_V_decoder_i will
            #      be concatenated to h_ESV_ij.
            h_ESV = (
                decoder_features["causal_mask"] * h_ESV_decoder
                + h_EXV_encoder_anti_causal
            )

            # h_V_decoder [B, L, H] - the updated node features for the
            # decoder.
            h_V_decoder = torch.utils.checkpoint.checkpoint(
                layer,
                h_V_decoder,
                h_ESV,
                mask_V=input_features["residue_mask"],
                mask_E=mask_E,
                use_reentrant=False,
            )

        # logits [B, L, self.vocab_size] - project the final node features to
        # get the sequence logits.
        logits = self.W_out(h_V_decoder)

        # Handle symmetry equivalence groups if they are provided, performing
        # a (possibly weighted) sum of the logits across residues that
        # belong to the same symmetry group.
        if input_features["symmetry_equivalence_group"] is not None:
            # Assume that all symmetry groups are non-negative.
            assert input_features["symmetry_equivalence_group"].min() >= 0

            B, L, _ = logits.shape

            # The maximum number of symmetry groups in the batch.
            G = (input_features["symmetry_equivalence_group"].max().item()) + 1

            # symmetry_equivalence_group_one_hot [B, L, G] - one-hot encoding
            # of the symmetry equivalence group for each residue.
            symmetry_equivalence_group_one_hot = torch.nn.functional.one_hot(
                input_features["symmetry_equivalence_group"], num_classes=G
            ).float()

            # scaled_symmetry_equivalence_group_one_hot [B, L, G] - the one-hot
            # encoding of the symmetry equivalence group, scaled by the
            # symmetry weights, if they are provided. If not provided, the
            # symmetry weights are implicitly assumed to be 1.0 for all
            # residues.
            scaled_symmetry_equivalence_group_one_hot = (
                symmetry_equivalence_group_one_hot
                * (
                    1.0
                    if input_features["symmetry_weight"] is None
                    else input_features["symmetry_weight"].unsqueeze(-1)
                )
            )

            # weighted_sum_logits [B, G, self.vocab_size] - the logits for the
            # sequence, summed across the symmetry groups, weighted by the
            # symmetry weights.
            weighted_sum_logits = torch.einsum(
                "blg,blv->bgv", scaled_symmetry_equivalence_group_one_hot, logits
            )

            # logits [B, L, self.vocab_size] - overwrite the original logits
            # with the weighted and summed logits for the residues that belong
            # to the same symmetry group.
            logits = torch.einsum(
                "blg,bgv->blv", symmetry_equivalence_group_one_hot, weighted_sum_logits
            )

        # Compute the log probabilities, probabilities, sampled probabilities,
        # predicted sequence, and argmax sequence.
        sample_dict = self.logits_to_sample(
            logits,
            input_features["bias"],
            input_features["pair_bias"],
            input_features["S"],
            input_features["temperature"],
        )

        # All outputs from logits_to_sample should be the same across
        # symmetry equivalence groups, except for S_sampled (due to the
        # sampling being stochastic). Correct for this by overwriting S_sampled
        # with the first sampled residue in each group.
        if input_features["symmetry_equivalence_group"] is not None:
            S_sampled = sample_dict["S_sampled"]

            B, L = S_sampled.shape

            # Compute the maximum number of symmetry groups in the batch.
            G = (input_features["symmetry_equivalence_group"].max().item()) + 1

            for b in range(B):
                for g in range(G):
                    # group_mask [L] - mask for the residues that belong to
                    # the symmetry equivalence group g for the batch example b.
                    group_mask = input_features["symmetry_equivalence_group"][b] == g

                    # If there are residues in the group, set every S_sampled
                    # in the group to the first S_sampled in the group.
                    if group_mask.any():
                        first = torch.where(group_mask)[0][0]
                        S_sampled[b, group_mask] = S_sampled[b, first]

            sample_dict["S_sampled"] = S_sampled

        # Update the decoder features with the final node features, the computed
        # logits, log probabilities, probabilities, sampled probabilities,
        # predicted sequence, and argmax sequence.
        decoder_features["h_V"] = h_V_decoder
        decoder_features["logits"] = logits
        decoder_features["log_probs"] = sample_dict["log_probs"]
        decoder_features["probs"] = sample_dict["probs"]
        decoder_features["probs_sample"] = sample_dict["probs_sample"]
        decoder_features["S_sampled"] = sample_dict["S_sampled"]
        decoder_features["S_argmax"] = sample_dict["S_argmax"]

    def decode_auto_regressive(
        self, input_features, graph_features, encoder_features, decoder_features
    ):
        """
        Given the input features, graph features, encoder features, and
        decoder features, perform the autoregressive decoding.

        Args:
            input_features (dict): Input features containing the residue mask
                and sequence.
                - S (torch.Tensor): [B, L] - sequence of residues.
                - residue_mask (torch.Tensor): [B, L] - mask for the residues.
                - bias (torch.Tensor, optional): [B, L, self.vocab_size]
                    - the per-residue bias to use for sampling. If None, the
                    code will implicitly use a bias of 0.0 for all residues.
                - pair_bias (torch.Tensor, optional): [B, L, self.vocab_size
                    , L, self.vocab_size] - the per-residue pair bias to use
                    for sampling. If None, the code will implicitly use a pair
                    bias of 0.0 for all residue pairs.
                - temperature (torch.Tensor, optional): [B, L] - the per-residue
                    temperature to use for sampling. If None, the code will
                    implicitly use a temperature of 1.0.
                - initialize_sequence_embedding_with_ground_truth (bool):
                    If True, initialize the sequence embedding with the ground
                    truth sequence S. Else, initialize the sequence
                    embedding with zeros. Also, if True, initialize S_sampled
                    with the ground truth sequence S, which should only affect
                    the application of pair bias (which relies on the predicted
                    sequence). This is useful if we want to perform
                    auto-regressive redesign.
                - symmetry_equivalence_group (torch.Tensor, optional): [B, L] -
                    an integer for every residue, indicating the symmetry group
                    that it belongs to. If None, the residues are not grouped by
                    symmetry. For example, if residue i and j should be decoded
                    symmetrically, then symmetry_equivalence_group[i] ==
                    symmetry_equivalence_group[j]. Must be torch.int64 to allow
                    for use as an index. These values should range from 0 to
                    the maximum number of symmetry groups - 1 for each example.
                    NOTE: bias, pair_bias, and temperature should be the same
                    for all residues in the symmetry equivalence group;
                    otherwise, the intended behavior may not be achieved. The
                    residues within a symmetry group should all have the same
                    validity and design/fixed status.
                -symmetry_weight (torch.Tensor, optional): [B, L] - the weights
                    for each residue, to be used when aggregating across its
                    respective symmetry group. If None, the weights are
                    assumed to be 1.0 for all residues.
            graph_features (dict): Graph features containing the featurized
                node and edge inputs.
                - E_idx (torch.Tensor): [B, L, K] - edge indices.
            encoder_features (dict): Encoder features containing the encoded
                protein node and protein edge features.
                - h_V (torch.Tensor): [B, L, H] - the protein node features
                    after encoding message passing.
                - h_E (torch.Tensor): [B, L, K, H] - the protein edge features
                    after encoding message passing.
            decoder_features (dict): Initial decoder features containing the
                causal mask for the decoder.
                - decoding_order (torch.Tensor): [B, L] - the order in which
                    the residues should be decoded.
                - decode_last_mask (torch.Tensor): [B, L] - the mask for which
                    residues should be decoded last, where False is a residue
                    that should be decoded first (invalid or fixed), and True
                    is a residue that should not be decoded first (designed
                    residues).
                - causal_mask (torch.Tensor): [B, L, K, 1] - the causal mask
                    for the decoder.
                - anti_causal_mask (torch.Tensor): [B, L, K, 1] - the anti-
                    causal mask for the decoder.
        Side Effects:
            decoder_features["h_V"] (torch.Tensor): [B, L, H] - the updated
                node features for the decoder.
            decoder_features["logits"] (torch.Tensor): [B, L, self.vocab_size] -
                the sequence logits for the decoder.
            decoder_features["log_probs"] (torch.Tensor): [B, L,
                self.vocab_size] - the log probabilities for the sequence.
            decoder_features["probs"] (torch.Tensor): [B, L, self.vocab_size] -
                the probabilities for the sequence.
            decoder_features["probs_sample"] (torch.Tensor): [B, L,
                self.vocab_size] - the probabilities for the sequence, with the
                unknown residues zeroed out and the other residues normalized.
            decoder_features["S_sampled"] (torch.Tensor): [B, L] - the
                predicted sequence, sampled from the probabilities (unknown
                residues are not sampled).
            decoder_features["S_argmax"] (torch.Tensor): [B, L] - the predicted
                sequence, obtained by taking the argmax of the probabilities
                (unknown residues are not selected).
        """
        # Check that the input features contains the necessary keys.
        if "residue_mask" not in input_features:
            raise ValueError("Input features must contain 'residue_mask' key.")
        if "S" not in input_features:
            raise ValueError("Input features must contain 'S' key.")
        if "temperature" not in input_features:
            raise ValueError("Input features must contain 'temperature' key.")
        if "bias" not in input_features:
            raise ValueError("Input features must contain 'bias' key.")
        if "pair_bias" not in input_features:
            raise ValueError("Input features must contain 'pair_bias' key.")
        if "symmetry_equivalence_group" not in input_features:
            raise ValueError(
                "Input features must contain 'symmetry_equivalence_group' key."
            )
        if "symmetry_weight" not in input_features:
            raise ValueError("Input features must contain 'symmetry_weight' key.")

        # Check that the graph features contains the necessary keys.
        if "E_idx" not in graph_features:
            raise ValueError("Graph features must contain 'E_idx' key.")

        # Check that the encoder features contains the necessary keys.
        if "h_V" not in encoder_features:
            raise ValueError("Encoder features must contain 'h_V' key.")
        if "h_E" not in encoder_features:
            raise ValueError("Encoder features must contain 'h_E' key.")

        # Check that the decoder features contains the necessary keys.
        if "decoding_order" not in decoder_features:
            raise ValueError("Decoder features must contain 'decoding_order' key.")
        if "decode_last_mask" not in decoder_features:
            raise ValueError("Decoder features must contain 'decode_last_mask' key.")
        if "causal_mask" not in decoder_features:
            raise ValueError("Decoder features must contain 'causal_mask' key.")

        B, L = input_features["residue_mask"].shape

        # Do the setup for the decoder.
        # h_EXV_encoder_anti_causal [B, L, K, 3H] - the encoder embeddings,
        # masked with the anti-causal mask.
        # mask_E [B, L, K] - the mask for the edges, gathered at the
        # neighbor indices.
        # h_S [B, L, H] - the sequence embeddings for the decoder.
        h_EXV_encoder_anti_causal, mask_E, h_S = self.decode_setup(
            input_features, graph_features, encoder_features, decoder_features
        )

        # We can precompute the output dtype depending on automatic mixed
        # precision settings. This works because the W_out layer is a linear
        # layer, which has predictable dtype behavior with AMP.
        device = input_features["residue_mask"].device
        if device.type in ("cuda", "cpu") and torch.is_autocast_enabled(
            device_type=device.type
        ):
            output_dtype = torch.get_autocast_dtype(device_type=device.type)
        else:
            output_dtype = torch.float32

        # logits [B, L, self.vocab_size] - the logits for every residue
        # position and residue type.
        logits = torch.zeros(
            (B, L, self.vocab_size),
            device=input_features["residue_mask"].device,
            dtype=output_dtype,
        )

        # logits_i [B, 1, self.vocab_size] - the logits for the
        # residue at the current decoding index, computed from the
        # decoded node features. Declared here for accumulation use when
        # performing symmetry decoding.
        logits_i = torch.zeros(
            (B, 1, self.vocab_size),
            device=input_features["residue_mask"].device,
            dtype=output_dtype,
        )

        # log_probs [B, L, self.vocab_size] - the log probabilities for every
        # residue position and residue type.
        log_probs = torch.zeros(
            (B, L, self.vocab_size),
            device=input_features["residue_mask"].device,
            dtype=torch.float32,
        )

        # probs [B, L, self.vocab_size] - the probabilities for every residue
        # position and residue type.
        probs = torch.zeros(
            (B, L, self.vocab_size),
            device=input_features["residue_mask"].device,
            dtype=torch.float32,
        )

        # probs_sample [B, L, self.vocab_size] - the probabilities for every
        # residue position and residue type, with the unknown residues zeroed
        # out and the other residues normalized.
        probs_sample = torch.zeros(
            (B, L, self.vocab_size),
            device=input_features["residue_mask"].device,
            dtype=torch.float32,
        )

        # S_sampled [B, L] - the predicted/sampled sequence of residues. If
        # initialize_sequence_embedding_with_ground_truth is True, it is
        # initialized with the ground truth sequence S; this should only
        # affect the application of pair bias (which relies on the predicted
        # sequence), and is useful if we want to perform auto-regressive
        # redesign. Otherwise, this should have no effect, as we overwrite
        # S_sampled with the sampled sequence at every decoding step.
        if input_features["initialize_sequence_embedding_with_ground_truth"]:
            S_sampled = input_features["S"].clone()
        else:
            S_sampled = torch.full(
                (B, L),
                fill_value=self.token_to_idx[UNKNOWN_AA],
                device=input_features["S"].device,
                dtype=input_features["S"].dtype,
            )

        # S_argmax [B, L] - the argmax sequence of residues, initialized with
        # the unknown residue type.
        S_argmax = torch.full(
            (B, L),
            fill_value=self.token_to_idx[UNKNOWN_AA],
            device=input_features["S"].device,
            dtype=input_features["S"].dtype,
        )

        # h_V_decoder_stack - list containing the hidden node embeddings from
        # each decoder layer; populated iteratively during the decoding.
        # h_V_decoder_stack[i] [B, L, H] - the hidden node embeddings,
        # the 0th entry is the initial node embeddings from the encoder, and
        # the i-th entry is the hidden node embeddings after the i-th decoder
        # layer.
        # NOTE: it is necessary to keep the embeddings from all decoder layers,
        # since later decoding positions rely on the intermediate decoder layer
        # embeddings of previously decoded residues.
        h_V_decoder_stack = [encoder_features["h_V"]] + [
            torch.zeros_like(
                encoder_features["h_V"], device=input_features["residue_mask"].device
            )
            for _ in range(len(self.decoder_layers))
        ]

        # batch_idx [B, 1] - the batch indices for the decoder.
        batch_idx = torch.arange(
            B, device=input_features["residue_mask"].device
        ).unsqueeze(-1)

        # Iteratively decode, updating the hidden sequence embeddings.
        for decoding_idx in range(L):
            # i [B, 1] - the indices of the residues to decode in the
            # current iteration, based on the decoding order.
            i = decoder_features["decoding_order"][:, decoding_idx].unsqueeze(-1)

            # decode_last_mask_i [B, 1] - the mask for residues that should be
            # decoded last, where False is a residue that should be decoded
            # first (invalid or fixed), and True is a residue that should not be
            # decoded first (designed residues); at the current decoding
            # index.
            decode_last_mask_i = decoder_features["decode_last_mask"][batch_idx, i]

            # residue_mask_i [B, 1] - the mask for the residue at the current
            # decoding index.
            residue_mask_i = input_features["residue_mask"][batch_idx, i]

            # mask_E_i [B, 1, K] - the mask for the edges at the current
            # decoding index, gathered at the neighbor indices.
            mask_E_i = mask_E[batch_idx, i]

            # S_i [B, 1] - the ground truth sequence for the residue at the
            # current decoding index (for designed positions, undefined).
            S_i = input_features["S"][batch_idx, i]

            # Setup the temperature, bias, and pair bias for the current
            # decoding index.
            if input_features["temperature"] is not None:
                # temperature_i [B, 1] - the temperature for the residue at the
                # current decoding index.
                temperature_i = input_features["temperature"][batch_idx, i]
            else:
                temperature_i = None

            if input_features["bias"] is not None:
                # bias_i [B, 1, self.vocab_size] - the bias for the residue at
                # the current decoding index.
                bias_i = input_features["bias"][batch_idx, i]
            else:
                bias_i = None

            if input_features["pair_bias"] is not None:
                # pair_bias_i [B, 1, self.vocab_size, L, self.vocab_size] - the
                # pair bias for the residue at the current decoding index.
                pair_bias_i = input_features["pair_bias"][batch_idx, i]
            else:
                pair_bias_i = None

            if input_features["symmetry_equivalence_group"] is not None:
                # symmetry_equivalence_group_i [B, 1] - the symmetry
                # equivalence group for the residue at the current decoding
                # index.
                symmetry_equivalence_group_i = input_features[
                    "symmetry_equivalence_group"
                ][batch_idx, i]
            else:
                symmetry_equivalence_group_i = None

            if input_features["symmetry_weight"] is not None:
                # symmetry_weight_i [B, 1] - the symmetry weights for the
                # residue at the current decoding index.
                symmetry_weight_i = input_features["symmetry_weight"][batch_idx, i]
            else:
                symmetry_weight_i = None

            # Gather the graph, encoder, and sequence features for the
            # current decoding index.
            # E_idx_i [B, 1, K] - the edge indices for the residue at the
            # current decoding index.
            E_idx_i = graph_features["E_idx"][batch_idx, i]

            # h_E_i [B, 1, K, H] - the post-encoder edge features for the
            # residue at the current decoding index.
            h_E_i = encoder_features["h_E"][batch_idx, i]

            # h_ES_i [B, 1, K, 2H] - the edge features concatenated with the
            # sequence embeddings for the destination nodes, for the residue at
            # the current decoding index.
            h_ES_i = cat_neighbors_nodes(h_S, h_E_i, E_idx_i)

            # h_EXV_encoder_anti_causal_i [B, 1, K, 3H] - the encoder
            # embeddings, masked with the anti-causal mask, for the residue at
            # the current decoding index.
            h_EXV_encoder_anti_causal_i = h_EXV_encoder_anti_causal[batch_idx, i]

            # causal_mask_i [B, 1, K, 1] - the causal mask for the residue at
            # the current decoding index.
            causal_mask_i = decoder_features["causal_mask"][batch_idx, i]

            # Apply the decoder layers, updating the hidden node embeddings
            # for the current decoding index.
            for layer_idx, layer in enumerate(self.decoder_layers):
                # h_ESV_decoder_i [B, 1, K, 3H] - h_E_ij cat h_S_j cat
                # h_V_decoder_j; for the decoder embeddings, the edge features
                # are concatenated with the destination node sequence embeddings
                # and node features, for the residue at the current decoding
                # index.
                h_ESV_decoder_i = cat_neighbors_nodes(
                    h_V_decoder_stack[layer_idx], h_ES_i, E_idx_i
                )

                # h_ESV_i [B, 1, K, 3H] - the encoder and decoder embeddings,
                # combined according to the causal and anti-causal masks.
                # Combine the encoder embeddings with the decoder embeddings,
                # using the causal and anti-causal masks. When decoding the
                # residue at position i:
                #    - for residue j, decoded before i:
                #        - h_ESV_ij = h_E_ij cat h_S_j cat h_V_decoder_j
                #            - encoder edge embedding, decoder destination node
                #              sequence embedding, and decoder destination node
                #              embedding.
                #    - for residue j, decoded after i (including i):
                #        - h_ESV_ij = h_E_ij cat (0 vector) cat h_V_j
                #            - encoder edge embedding, zero vector (no sequence
                #              information), and encoder destination node
                #              embedding. This prevents leakage of sequence
                #              information.
                #            - NOTE: h_V_j comes from the encoder.
                #    - NOTE: h_E is not updated in the decoder, h_E_ij comes
                #      from the encoder.
                #    - NOTE: within the decoder layer itself, h_V_decoder_i will
                #      be concatenated to h_ESV_ij.
                h_ESV_i = causal_mask_i * h_ESV_decoder_i + h_EXV_encoder_anti_causal_i

                # h_V_decoder_i [B, 1, H] - the updated node features for the
                # decoder, after applying the layer at the current decoding
                # index.
                h_V_decoder_i = torch.utils.checkpoint.checkpoint(
                    layer,
                    h_V_decoder_stack[layer_idx][batch_idx, i],
                    h_ESV_i,
                    mask_V=residue_mask_i,
                    mask_E=mask_E_i,
                    use_reentrant=False,
                )

                # h_V_decoder_stack[layer_idx + 1][batch_idx, i] [B, 1, H] -
                # the updated node features for the decoder, after applying the
                # layer at the current decoding index.
                if not torch.is_grad_enabled():
                    h_V_decoder_stack[layer_idx + 1][batch_idx, i] = h_V_decoder_i
                else:
                    # For gradient tracking, we can't use in-place operations.
                    h_V_decoder_stack[layer_idx + 1] = h_V_decoder_stack[
                        layer_idx + 1
                    ].scatter(
                        1,
                        i.unsqueeze(-1).expand(-1, -1, self.hidden_dim),
                        h_V_decoder_i,
                    )

            if input_features["symmetry_equivalence_group"] is None:
                # logits_i [B, 1, self.vocab_size] - the logits for the
                # residue at the current decoding index, computed from the
                # decoded node features.
                logits_i = self.W_out(h_V_decoder_stack[-1][batch_idx, i])
            else:
                # logits_i [B, 1, self.vocab_size] - the logits for the
                # residue at the current decoding index, computed from the
                # decoded node features, aggregated across symmetry groups,
                # weighted by the symmetry weights.
                logits_i += self.W_out(h_V_decoder_stack[-1][batch_idx, i]) * (
                    1.0
                    if symmetry_weight_i is None
                    else symmetry_weight_i.unsqueeze(-1)
                )

            # Compute the log probabilities, probabilities, sampled
            # probabilities, predicted sequence, and argmax sequence for the
            # current decoding index.
            sample_dict = self.logits_to_sample(
                logits_i, bias_i, pair_bias_i, S_sampled, temperature_i
            )

            log_probs_i = sample_dict["log_probs"]
            probs_i = sample_dict["probs"]
            probs_sample_i = sample_dict["probs_sample"]
            S_sampled_i = sample_dict["S_sampled"]
            S_argmax_i = sample_dict["S_argmax"]

            if input_features["symmetry_equivalence_group"] is None:
                # Save the logits, probabilities, probabilities, sampled
                # probabilities, and log probabilities for the current decoding
                # index. These are saved but not sampled for invalid/fixed
                # residues.
                logits[batch_idx, i] = logits_i
                probs[batch_idx, i] = probs_i
                probs_sample[batch_idx, i] = probs_sample_i
                if not torch.is_grad_enabled():
                    log_probs[batch_idx, i] = log_probs_i
                else:
                    # For gradient tracking, we can't use in-place operations.
                    log_probs = log_probs.scatter(
                        1, i.unsqueeze(-1).expand(-1, -1, self.vocab_size), log_probs_i
                    )

                # Update the predicted sequence and argmax sequence for the
                # current decoding index.
                S_sampled[batch_idx, i] = (S_sampled_i * decode_last_mask_i) + (
                    S_i * (~decode_last_mask_i)
                )
                S_argmax[batch_idx, i] = (S_argmax_i * decode_last_mask_i) + (
                    S_i * (~decode_last_mask_i)
                )

                # h_S_i [B, 1, self.hidden_dim] - the sequence embeddings of the
                # sampled/fixed residue at the current decoding index.
                h_S_i = self.W_s(S_sampled[batch_idx, i])

                # Update the decoder sequence embeddings with the predicted
                # sequence for the current decoding index.
                if not torch.is_grad_enabled():
                    h_S[batch_idx, i] = h_S_i
                else:
                    # For gradient tracking, we can't use in-place operations.
                    h_S = h_S.scatter(
                        1, i.unsqueeze(-1).expand(-1, -1, self.hidden_dim), h_S_i
                    )
            else:
                # symm_group_end_mask [B, 1] - mask for the residues that are
                # at the end of a symmetry group, where True is a residue that
                # is at the end of a symmetry group, and False is a residue that
                # is not at the end of a symmetry group. When we are at the last
                # decoding index, we know that all residues are at the end of a
                # symmetry group.
                if decoding_idx == (L - 1):
                    symm_group_end_mask = torch.ones(
                        (B, 1),
                        device=input_features["residue_mask"].device,
                        dtype=torch.bool,
                    )
                else:
                    # next_i [B, 1] - the indices of the next residues to decode
                    # in the current iteration, based on the decoding order.
                    next_i = decoder_features["decoding_order"][
                        :, decoding_idx + 1
                    ].unsqueeze(-1)

                    # symmetry_equivalence_group_next_i [B, 1] - the symmetry
                    # equivalence group for the residue at the next decoding
                    # index.
                    symmetry_equivalence_group_next_i = input_features[
                        "symmetry_equivalence_group"
                    ][batch_idx, next_i]

                    symm_group_end_mask = (
                        symmetry_equivalence_group_i
                        != symmetry_equivalence_group_next_i
                    )

                # same_symm_group_mask [B, L] - mask for the residues that
                # belong to the same symmetry group as the current residue,
                # where True is a residue that belongs to the same symmetry
                # group, and False is a residue that does not belong to the same
                # symmetry group.
                same_symm_group_mask = (
                    input_features["symmetry_equivalence_group"]
                    == symmetry_equivalence_group_i
                )

                # symm_end_and_same_mask [B, L] - mask that combines the
                # symm_group_end_mask and same_symm_group_mask, where all
                # residues with the same symmetry group as the current residue
                # (if the current residue is at the end of a symmetry group)
                # are True, and all other residues are False.
                symm_end_and_same_mask = symm_group_end_mask & same_symm_group_mask

                # symm_end_and_same_mask_vocab [B, L, self.vocab_size] -
                # symm_end_and_same_mask projected to the vocabulary size.
                symm_end_and_same_mask_vocab = symm_end_and_same_mask.unsqueeze(
                    -1
                ).expand(-1, -1, self.vocab_size)

                # symm_end_and_same_mask_hidden [B, L, self.hidden_dim] -
                # symm_end_and_same_mask projected to the hidden dimension.
                symm_end_and_same_mask_for_hidden = symm_end_and_same_mask.unsqueeze(
                    -1
                ).expand(-1, -1, self.hidden_dim)

                # Save the logits, probabilities, sampled probabilities, and log
                # probabilities for the current decoding index, if the residue
                # is at the end of a symmetry group.
                logits[symm_end_and_same_mask_vocab] = logits_i.expand(-1, L, -1)[
                    symm_end_and_same_mask_vocab
                ]
                probs[symm_end_and_same_mask_vocab] = probs_i.expand(-1, L, -1)[
                    symm_end_and_same_mask_vocab
                ]
                probs_sample[symm_end_and_same_mask_vocab] = probs_sample_i.expand(
                    -1, L, -1
                )[symm_end_and_same_mask_vocab]
                if not torch.is_grad_enabled():
                    log_probs[symm_end_and_same_mask_vocab] = log_probs_i.expand(
                        -1, L, -1
                    )[symm_end_and_same_mask_vocab]
                else:
                    # For gradient tracking, we can't use in-place operations.
                    log_probs = torch.where(
                        symm_end_and_same_mask_vocab,
                        log_probs_i.expand(-1, L, -1),
                        log_probs,
                    )

                # Update the predicted sequence and argmax sequence for the
                # current decoding index, if the residue is at the end of a
                # symmetry group.
                S_sampled[symm_end_and_same_mask] = (
                    (S_sampled_i * decode_last_mask_i) + (S_i * (~decode_last_mask_i))
                ).expand(-1, L)[symm_end_and_same_mask]
                S_argmax[symm_end_and_same_mask] = (
                    (S_argmax_i * decode_last_mask_i) + (S_i * (~decode_last_mask_i))
                ).expand(-1, L)[symm_end_and_same_mask]

                # h_S_i [B, 1, self.hidden_dim] - the sequence embeddings of the
                # sampled/fixed residue at the current decoding index.
                h_S_i = self.W_s(S_sampled[batch_idx, i])

                # Update the decoder sequence embeddings with the predicted
                # sequence for the current decoding index, if the residue is at
                # the end of a symmetry group.
                if not torch.is_grad_enabled():
                    h_S[symm_end_and_same_mask_for_hidden] = h_S_i.expand(-1, L, -1)[
                        symm_end_and_same_mask_for_hidden
                    ]
                else:
                    # For gradient tracking, we can't use in-place operations.
                    h_S = torch.where(
                        symm_end_and_same_mask_for_hidden, h_S_i.expand(-1, L, -1), h_S
                    )

                # Zero out the current position logits (used for accumulation)
                # if a batch example's current residue is at the end of a
                # symmetry group.
                logits_i[symm_group_end_mask] = 0.0

        # Update the decoder features with the final node features, the computed
        # logits, log probabilities, probabilities, sampled probabilities,
        # predicted sequence, and argmax sequence.
        decoder_features["h_V"] = h_V_decoder_stack[-1]
        decoder_features["logits"] = logits
        decoder_features["log_probs"] = log_probs
        decoder_features["probs"] = probs
        decoder_features["probs_sample"] = probs_sample
        decoder_features["S_sampled"] = S_sampled
        decoder_features["S_argmax"] = S_argmax

    def construct_output_dictionary(
        self, input_features, graph_features, encoder_features, decoder_features
    ):
        """
        Constructs the output dictionary based on the requested features.

        Args:
            input_features (dict): Input features containing the requested
                features to return.
                - features_to_return (dict, optional): dictionary determining
                    which features to return from the model. If None, return all
                    features (including modified input features, graph features,
                    encoder features, and decoder features). Otherwise,
                    expects a dictionary with the following key, value pairs:
                    - "input_features": list - the input features to return.
                    - "graph_features": list - the graph features to return.
                    - "encoder_features": list - the encoder features to return.
                    - "decoder_features": list - the decoder features to return.
            graph_features (dict): Graph features containing the featurized
                node and edge inputs.
            encoder_features (dict): Encoder features containing the encoded
                protein node and protein edge features.
            decoder_features (dict): Decoder features containing the post-
                decoder features (including causal masks, logits, probabilities,
                predicted sequence, etc.).
        Returns:
            output_dict (dict): Output dictionary containing the requested
                features based on the input features' "features_to_return" key.
                If "features_to_return" is None, returns all features.
        """
        # Check that the input features contains the necessary keys.
        if "features_to_return" not in input_features:
            raise ValueError("Input features must contain 'features_to_return' key.")

        # Create the output dictionary based on the requested features.
        if input_features["features_to_return"] is None:
            output_dict = {
                "input_features": input_features,
                "graph_features": graph_features,
                "encoder_features": encoder_features,
                "decoder_features": decoder_features,
            }
        else:
            # Filter the output dictionary based on the requested features.
            output_dict = dict()
            output_dict["input_features"] = {
                key: input_features[key]
                for key in input_features["features_to_return"].get(
                    "input_features", []
                )
            }
            output_dict["graph_features"] = {
                key: graph_features[key]
                for key in input_features["features_to_return"].get(
                    "graph_features", []
                )
            }
            output_dict["encoder_features"] = {
                key: encoder_features[key]
                for key in input_features["features_to_return"].get(
                    "encoder_features", []
                )
            }
            output_dict["decoder_features"] = {
                key: decoder_features[key]
                for key in input_features["features_to_return"].get(
                    "decoder_features", []
                )
            }

        return output_dict

    def forward(self, network_input):
        """
        Forward pass of the ProteinMPNN model.

        A NOTE on shapes:
            - B = batch dimension size
            - L = sequence length (number of residues)
            - K = number of neighbors per residue
            - H = hidden dimension size
            - vocab_size = self.vocab_size
            - num_atoms =
                self.graph_featurization_module.TOKEN_ENCODING.n_atoms_per_token
            - num_backbone_atoms = len(
                self.graph_featurization_module.BACKBONE_ATOM_NAMES
            )
            - num_virtual_atoms = len(
                self.graph_featurization_module.DATA_TO_CALCULATE_VIRTUAL_ATOMS
            )
            - num_rep_atoms = len(
                self.graph_featurization_module.REPRESENTATIVE_ATOM_NAMES
            )
            - num_edge_output_features =
                self.graph_featurization_module.num_edge_output_features
            - num_node_output_features =
                self.graph_featurization_module.num_node_output_features

        Args:
            network_input (dict): Dictionary containing the input to the
                network.
                - input_features (dict): dictionary containing input features
                    and all necessary information for the model to run.
                    - X (torch.Tensor): [B, L, num_atoms, 3] - 3D coordinates of
                        polymer atoms.
                    - X_m (torch.Tensor): [B, L, num_atoms] - Mask indicating
                        which polymer atoms are valid.
                    - S (torch.Tensor): [B, L] - Sequence of the polymer
                        residues.
                    - R_idx (torch.Tensor): [B, L] - indices of the residues.
                    - chain_labels (torch.Tensor): [B, L] - chain labels for
                        each residue.
                    - residue_mask (torch.Tensor): [B, L] - Mask indicating
                        which residues are valid.
                    - designed_residue_mask (torch.Tensor): [B, L] - mask for
                        the designed residues.
                    - symmetry_equivalence_group (torch.Tensor, optional):
                        [B, L] - an integer for every residue, indicating the
                        symmetry group that it belongs to. If None, the
                        residues are not grouped by symmetry. For example, if
                        residue i and j should be decoded symmetrically, then
                        symmetry_equivalence_group[i] ==
                        symmetry_equivalence_group[j]. Must be torch.int64 to
                        allow for use as an index. These values should range
                        from 0 to the maximum number of symmetry groups - 1 for
                        each example. NOTE: bias, pair_bias, and temperature
                        should be the same for all residues in the symmetry
                        equivalence group; otherwise, the intended behavior may
                        not be achieved. The residues within a symmetry group
                        should all have the same validity and design/fixed
                        status.
                    - symmetry_weight (torch.Tensor, optional): [B, L] - the
                        weights for each residue, to be used when aggregating
                        across its respective symmetry group. If None, the
                        weights are assumed to be 1.0 for all residues.
                    - bias (torch.Tensor, optional): [B, L, 21] - the
                        per-residue bias to use for sampling. If None, the code
                        will implicitly use a bias of 0.0 for all residues.
                    - pair_bias (torch.Tensor, optional): [B, L, 21, L, 21] -
                        the per-residue pair bias to use for sampling. If None,
                        the code will implicitly use a pair bias of 0.0 for all
                        residue pairs.
                    - temperature (torch.Tensor, optional): [B, L] - the
                        per-residue temperature to use for sampling. If None,
                        the code will implicitly use a temperature of 1.0.
                    - structure_noise (float): Standard deviation of the
                        Gaussian noise to add to the input coordinates, in
                        Angstroms.
                    - decode_type (str): the type of decoding to use.
                        - "teacher_forcing": Use teacher forcing for the
                            decoder, where the decoder attends to the ground
                            truth sequence S for all previously decoded
                            residues.
                        - "auto_regressive": Use auto-regressive decoding,
                            where the decoder attends to the sequence and
                            decoder representation of residues that have
                            already been decoded (using the predicted sequence).
                    - causality_pattern (str): The pattern of causality to use
                        for the decoder. For all causality patterns, the
                        decoding order is randomized.
                        - "auto_regressive": Use an auto-regressive causality
                            pattern, where residues can attend to the sequence
                            and decoder representation of residues that have
                            already been decoded (NOTE: as mentioned above,
                            this will be randomized).
                        - "unconditional": Residues cannot attend to the
                            sequence or decoder representation of any other
                            residues.
                        - "conditional": Residues can attend to the sequence
                            and decoder representation of all other residues.
                        - "conditional_minus_self": Residues can attend to the
                            sequence and decoder representation of all other
                            residues, except for themselves (as destination
                            nodes).
                    - initialize_sequence_embedding_with_ground_truth (bool):
                        - True: Initialize the sequence embedding with the
                            ground truth sequence S.
                            - If doing auto-regressive decoding, also
                                initialize S_sampled with the ground truth
                                sequence S, which should only affect the
                                application of pair bias.
                        - False: Initialize the sequence embedding with zeros.
                            - If doing auto-regressive decoding, initialize
                                S_sampled with unknown residues.
                    - features_to_return (dict, optional): dictionary
                        determining which features to return from the model. If
                        None, return all features (including modified input
                        features, graph features, encoder features, and decoder
                        features). Otherwise, expects a dictionary with the
                        following key, value pairs:
                        - "input_features": list - the input features to return.
                        - "graph_features": list - the graph features to return.
                        - "encoder_features": list - the encoder features to
                            return.
                        - "decoder_features": list - the decoder features to
                            return.
                    - repeat_sample_num (int, optional): Number of times to
                        repeat the samples along the batch dimension. If None,
                        no repetition is performed. If greater than 1, the
                        samples are repeated along the batch dimension. If
                        greater than 1, B must be 1, since repeating samples
                        along the batch dimension is not supported when more
                        than one sample is provided in the batch.
        Side Effects:
            Any changes denoted below to input_features are also mutated on the
            original input features.
        Returns:
            network_output (dict): Output dictionary containing the requested
                features based on the input features' "features_to_return" key.
                - input_features (dict): The input features from above, with
                    the following keys added or modified:
                    - mask_for_loss (torch.Tensor): [B, L] - mask for loss,
                        where True is a residue that is included in the loss
                        calculation, and False is a residue that is not
                        included in the loss calculation.
                    - X (torch.Tensor): [B, L, num_atoms, 3] - 3D coordinates of
                        polymer atoms with added Gaussian noise.
                    - X_pre_noise (torch.Tensor): [B, L, num_atoms, 3] -
                        3D coordinates of polymer atoms before adding Gaussian
                        noise ('X' before noise).
                    - X_backbone (torch.Tensor): [B, L, num_backbone_atoms, 3] -
                        3D coordinates of the backbone atoms for each residue,
                        built from the noisy 'X' coordinates.
                    - X_m_backbone (torch.Tensor): [B, L, num_backbone_atoms] -
                        mask indicating which backbone atoms are valid.
                    - X_virtual_atoms (torch.Tensor):
                        [B, L, num_virtual_atoms, 3] - 3D coordinates of the
                        virtual atoms for each residue, built from the noisy
                        'X' coordinates.
                    - X_m_virtual_atoms (torch.Tensor):
                        [B, L, num_virtual_atoms] - mask indicating which
                        virtual atoms are valid.
                    - X_rep_atoms (torch.Tensor): [B, L, num_rep_atoms, 3] - 3D
                        coordinates of the representative atoms for each
                        residue, built from the noisy 'X' coordinates.
                    - X_m_rep_atoms (torch.Tensor): [B, L, num_rep_atoms] -
                        mask indicating which representative atoms are valid.
                - graph_features (dict): The graph features.
                    - E_idx (torch.Tensor): [B, L, K] - indices of the top K
                        nearest neighbors for each residue.
                    - E (torch.Tensor): [B, L, K, num_edge_output_features] -
                        Edge features for each pair of neighbors.
                - encoder_features (dict): The encoder features.
                    - h_V (torch.Tensor): [B, L, H] - the protein node features
                        after encoding message passing.
                    - h_E (torch.Tensor): [B, L, K, H] - the protein edge
                        features after encoding message passing.
                - decoder_features (dict): The decoder features.
                    - causal_mask (torch.Tensor): [B, L, K, 1] - the causal
                        mask for the decoder.
                    - anti_causal_mask (torch.Tensor): [B, L, K, 1] - the
                        anti-causal mask for the decoder.
                    - decoding_order (torch.Tensor): [B, L] - the order in
                        which the residues should be decoded.
                    - decode_last_mask (torch.Tensor): [B, L] - mask for
                        residues that should be decoded last, where False is a
                        residue that should be decoded first (invalid or
                        fixed), and True is a residue that should not be
                        decoded first (designed residues).
                    - h_V (torch.Tensor): [B, L, H] - the updated node features
                        for the decoder.
                    - logits (torch.Tensor): [B, L, vocab_size] - the logits
                        for the sequence.
                    - log_probs (torch.Tensor): [B, L, vocab_size] - the log
                        probabilities for the sequence.
                    - probs (torch.Tensor): [B, L, vocab_size] - the
                        probabilities for the sequence.
                    - probs_sample (torch.Tensor): [B, L, vocab_size] -
                        the probabilities for the sequence, with the unknown
                        residues zeroed out and the other residues normalized.
                    - S_sampled (torch.Tensor): [B, L] - the predicted
                        sequence, sampled from the probabilities (unknown
                        residues are not sampled).
                    - S_argmax (torch.Tensor): [B, L] - the predicted sequence,
                        obtained by taking the argmax of the probabilities
                        (unknown residues are not selected).
        """
        input_features = network_input["input_features"]

        # Check that the input features contains the necessary keys.
        if "decode_type" not in input_features:
            raise ValueError("Input features must contain 'decode_type' key.")

        # Setup masks (added to the input features).
        self.sample_and_construct_masks(input_features)

        # Graph featurization (also modifies/adds to input_features).
        graph_features = self.graph_featurization(input_features)

        # Run the encoder.
        encoder_features = self.encode(input_features, graph_features)

        # Setup for decoder (repeat features along the batch dimension, modifies
        # input_features, graph_features, and encoder_features).
        self.repeat_along_batch(
            input_features,
            graph_features,
            encoder_features,
        )

        # Set up the causality masks.
        decoder_features = self.setup_causality_masks(input_features, graph_features)

        # Decoder, either teacher forcing or auto-regressive.
        if input_features["decode_type"] == "teacher_forcing":
            self.decode_teacher_forcing(
                input_features, graph_features, encoder_features, decoder_features
            )
        elif input_features["decode_type"] == "auto_regressive":
            self.decode_auto_regressive(
                input_features, graph_features, encoder_features, decoder_features
            )
        else:
            raise ValueError(f"Unknown decode_type: {input_features['decode_type']}.")

        # Create the output dictionary based on the requested features.
        network_output = self.construct_output_dictionary(
            input_features, graph_features, encoder_features, decoder_features
        )

        return network_output


class SolubleMPNN(ProteinMPNN):
    """
    Same as ProteinMPNN, but with different training set and different weights.
    """

    pass


class AntibodyMPNN(ProteinMPNN):
    """
    Same as ProteinMPNN, but with different training set and different weights.
    """

    pass


class MembraneMPNN(ProteinMPNN):
    """
    Class for per-residue and global membrane label version of MPNN.
    """

    HAS_NODE_FEATURES = True

    def __init__(
        self,
        num_node_features=128,
        num_edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        num_neighbors=48,
        dropout_rate=0.1,
        num_positional_embeddings=16,
        min_rbf_mean=2.0,
        max_rbf_mean=22.0,
        num_rbf=16,
        num_membrane_classes=3,
    ):
        """
        Setup the MembraneMPNN model.

        All args are the same as the parents class, except for the following:
        Args:
            num_membrane_classes (int): Number of membrane classes.
        """
        # The only change necessary here is the graph featurization module.
        graph_featurization_module = ProteinFeaturesMembrane(
            num_edge_output_features=num_edge_features,
            num_node_output_features=num_node_features,
            num_positional_embeddings=num_positional_embeddings,
            min_rbf_mean=min_rbf_mean,
            max_rbf_mean=max_rbf_mean,
            num_rbf=num_rbf,
            num_neighbors=num_neighbors,
            num_membrane_classes=num_membrane_classes,
        )

        super(MembraneMPNN, self).__init__(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            hidden_dim=hidden_dim,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            num_neighbors=num_neighbors,
            dropout_rate=dropout_rate,
            num_positional_embeddings=num_positional_embeddings,
            min_rbf_mean=min_rbf_mean,
            max_rbf_mean=max_rbf_mean,
            num_rbf=num_rbf,
            graph_featurization_module=graph_featurization_module,
        )


class PSSMMPNN(ProteinMPNN):
    """
    Class for pssm-aware version of MPNN.
    """

    HAS_NODE_FEATURES = True

    def __init__(
        self,
        num_node_features=128,
        num_edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        num_neighbors=48,
        dropout_rate=0.1,
        num_positional_embeddings=16,
        min_rbf_mean=2.0,
        max_rbf_mean=22.0,
        num_rbf=16,
        num_pssm_features=20,
    ):
        """
        Setup the PSSMMPNN model.

        All args are the same as the parents class, except for the following:
        Args:
            num_pssm_features (int): Number of PSSM features.
        """
        # The only change necessary here is the graph featurization module.
        graph_featurization_module = ProteinFeaturesPSSM(
            num_edge_output_features=num_edge_features,
            num_node_output_features=num_node_features,
            num_positional_embeddings=num_positional_embeddings,
            min_rbf_mean=min_rbf_mean,
            max_rbf_mean=max_rbf_mean,
            num_rbf=num_rbf,
            num_neighbors=num_neighbors,
            num_pssm_features=num_pssm_features,
        )

        super(PSSMMPNN, self).__init__(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            hidden_dim=hidden_dim,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            num_neighbors=num_neighbors,
            dropout_rate=dropout_rate,
            num_positional_embeddings=num_positional_embeddings,
            min_rbf_mean=min_rbf_mean,
            max_rbf_mean=max_rbf_mean,
            num_rbf=num_rbf,
            graph_featurization_module=graph_featurization_module,
        )


class LigandMPNN(ProteinMPNN):
    """
    Class for ligand-aware version of MPNN.
    """

    # Although there are node-like features, they are actually the protein-
    # ligand subgraph features, so we set this to False. Note, this is because
    # none of these features are embedded prior to protein encoding.
    HAS_NODE_FEATURES = False

    def __init__(
        self,
        num_node_features=128,
        num_edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        num_neighbors=32,
        dropout_rate=0.1,
        num_positional_embeddings=16,
        min_rbf_mean=2.0,
        max_rbf_mean=22.0,
        num_rbf=16,
        num_context_atoms=25,
        num_context_encoding_layers=2,
        overall_atomize_side_chain_probability=0.5,
        per_residue_atomize_side_chain_probability=0.02,
    ):
        # Pass the num_context_atoms to the graph featurization module.
        graph_featurization_module = ProteinFeaturesLigand(
            num_edge_output_features=num_edge_features,
            num_node_output_features=num_node_features,
            num_positional_embeddings=num_positional_embeddings,
            min_rbf_mean=min_rbf_mean,
            max_rbf_mean=max_rbf_mean,
            num_rbf=num_rbf,
            num_neighbors=num_neighbors,
            num_context_atoms=num_context_atoms,
        )

        super(LigandMPNN, self).__init__(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            hidden_dim=hidden_dim,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            num_neighbors=num_neighbors,
            dropout_rate=dropout_rate,
            num_positional_embeddings=num_positional_embeddings,
            min_rbf_mean=min_rbf_mean,
            max_rbf_mean=max_rbf_mean,
            num_rbf=num_rbf,
            graph_featurization_module=graph_featurization_module,
        )
        self.overall_atomize_side_chain_probability = (
            overall_atomize_side_chain_probability
        )
        self.per_residue_atomize_side_chain_probability = (
            per_residue_atomize_side_chain_probability
        )

        # Linear layer for embedding the protein-ligand edge features.
        self.W_protein_to_ligand_edges_embed = nn.Linear(
            num_node_features, hidden_dim, bias=True
        )

        # Linear layers for embedding the output of the protein encoder.
        self.W_protein_encoding_embed = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Linear layers for embedding the ligand nodes and edges.
        self.W_ligand_nodes_embed = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_ligand_edges_embed = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Linear layer for the final context embedding.
        self.W_final_context_embed = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Layer norm for the final context embedding.
        self.final_context_norm = nn.LayerNorm(hidden_dim)

        # Save the number of context encoding layers.
        self.num_context_encoding_layers = num_context_encoding_layers

        self.protein_ligand_context_encoder_layers = nn.ModuleList(
            [
                DecLayer(hidden_dim, hidden_dim * 3, dropout=dropout_rate)
                for _ in range(num_context_encoding_layers)
            ]
        )

        self.ligand_context_encoder_layers = nn.ModuleList(
            [
                DecLayer(hidden_dim, hidden_dim * 2, dropout=dropout_rate)
                for _ in range(num_context_encoding_layers)
            ]
        )

    def sample_and_construct_masks(self, input_features):
        """
        Sample and construct masks for the input features.

        Args:
            input_features (dict): Input features containing the residue mask.
                - residue_mask (torch.Tensor): [B, L] - mask for the residues.
                - S (torch.Tensor): [B, L] - sequence of residues.
                - designed_residue_mask (torch.Tensor): [B, L] - mask for the
                    designed residues.
                - atomize_side_chains (bool): Whether to atomize side chains.
        Side Effects:
            input_features["residue_mask"] (torch.Tensor): [B, L] - mask for the
                residues, where True is a residue that is valid and False is a
                residue that is invalid.
            input_features["known_residue_mask"] (torch.Tensor): [B, L] - mask
                for known residues, where True is a residue with one of the
                canonical residue types, and False is a residue with an unknown
                residue type.
            input_features["designed_residue_mask"] (torch.Tensor): [B, L] -
                mask for designed residues, where True is a residue that is
                designed, and False is a residue that is not designed.
            input_features["hide_side_chain_mask"] (torch.Tensor): [B, L] - mask
                for hiding side chains, where True is a residue with hidden side
                chains, and False is a residue with revealed side chains.
            input_features["mask_for_loss"] (torch.Tensor): [B, L] - mask for
                loss, where True is a residue that is included in the loss
                calculation, and False is a residue that is not included in the
                loss calculation.
        """
        # Create the masks for ProteinMPNN.
        super().sample_and_construct_masks(input_features)

        # Check that the input features contain the necessary keys.
        if "atomize_side_chains" not in input_features:
            raise ValueError("Input features must contain 'atomize_side_chains' key.")

        # Create the mask for hiding or revealing side chains.
        # With no side chain atomization, the side chain mask is all ones.
        if input_features["atomize_side_chains"]:
            # If we are training, randomly reveal side chains.
            if self.training:
                # With a probability specified as the overall atomization
                # side chain probability, we reveal some side chains (otherwise,
                # we hide all side chains).
                if (
                    sample_bernoulli_rv(self.overall_atomize_side_chain_probability)
                    == 1
                ):
                    reveal_side_chain_mask = (
                        torch.rand(
                            input_features["S"].shape, device=input_features["S"].device
                        )
                        < self.per_residue_atomize_side_chain_probability
                    )
                    hide_side_chain_mask = ~reveal_side_chain_mask
                else:
                    hide_side_chain_mask = torch.ones(
                        input_features["S"].shape, device=input_features["S"].device
                    ).bool()
            # If we are not training, only the side chains of fixed residues
            # are revealed.
            else:
                hide_side_chain_mask = input_features["designed_residue_mask"].clone()
        else:
            hide_side_chain_mask = torch.ones(
                input_features["S"].shape, device=input_features["S"].device
            ).bool()

        # Save the hide side chain mask in the input features.
        input_features["hide_side_chain_mask"] = hide_side_chain_mask

        # Update the mask for the loss to include the hide side chain mask.
        input_features["mask_for_loss"] = (
            input_features["mask_for_loss"] & input_features["hide_side_chain_mask"]
        )

    def encode(self, input_features, graph_features):
        """
        Encode the protein features with ligand context.

        NOTE: M = self.graph_featurization_module.num_context_atoms, the number
            of ligand atoms in each residue subgraph.

        Args:
            input_features (dict): Input features containing the residue mask.
                - residue_mask (torch.Tensor): [B, L] - mask for the residues.
                - ligand_subgraph_Y_m (torch.Tensor): [B, L, M] - mask for the
                    ligand subgraph nodes.
            graph_features (dict): Graph features containing the featurized
                node and edge inputs.
                - E_protein_to_ligand (torch.Tensor): [B, L, M,
                    self.num_edge_features] - protein-ligand edge features.
                - ligand_subgraph_nodes (torch.Tensor): [B, L, M,
                    self.num_node_features] - ligand subgraph node features.
                - ligand_subgraph_edges (torch.Tensor): [B, L, M, M,
                    self.num_edge_features] - ligand subgraph edge features.
        Returns:
            encoder_features (dict): Encoded features containing the encoded
                protein node and protein edge features.
                - h_V (torch.Tensor): [B, L, self.hidden_dim] - the protein node
                    features after protein encoding and ligand context
                    encoding.
                - h_E (torch.Tensor): [B, L, K, self.hidden_dim] - the protein
                    edge features after protein encoding message passing.
        """
        # Use the parent encode method to get the initial protein encoding.
        encoder_features = super().encode(input_features, graph_features)

        # Check the encoder features.
        if "h_V" not in encoder_features:
            raise ValueError("Encoder features must contain 'h_V' key.")

        # Check that the input features contain the necessary keys.
        if "residue_mask" not in input_features:
            raise ValueError("Input features must contain 'residue_mask' key.")
        if "ligand_subgraph_Y_m" not in input_features:
            raise ValueError("Input features must contain 'ligand_subgraph_Y_m' key.")

        # Check that the graph features contain the necessary keys.
        if "E_protein_to_ligand" not in graph_features:
            raise ValueError("Graph features must contain 'E_protein_to_ligand' key.")
        if "ligand_subgraph_nodes" not in graph_features:
            raise ValueError("Graph features must contain 'ligand_subgraph_nodes' key.")
        if "ligand_subgraph_edges" not in graph_features:
            raise ValueError("Graph features must contain 'ligand_subgraph_edges' key.")

        # Compute the protein-ligand edge feature encoding.
        # h_E_protein_to_ligand [B, L, M, self.hidden_dim] - the embedding of
        # the protein-ligand edge features.
        h_E_protein_to_ligand = self.W_protein_to_ligand_edges_embed(
            graph_features["E_protein_to_ligand"]
        )

        # Construct the starting context features, to aggregate the ligand
        # context; will be updated in the context encoder.
        # h_V_context [B, L, self.hidden_dim] - the embedding of context.
        h_V_context = self.W_protein_encoding_embed(encoder_features["h_V"])

        # Construct the ligand subgraph edge mask.
        # ligand_subgraph_Y_m_edges [B, L, M, M] - the mask for the
        # ligand-ligand subgraph edges.
        ligand_subgraph_Y_m_edges = (
            input_features["ligand_subgraph_Y_m"][:, :, :, None]
            * input_features["ligand_subgraph_Y_m"][:, :, None, :]
        )

        # Embed the ligand nodes.
        # ligand_subgraph_nodes [B, L, M, self.hidden_dim] - the embedding of
        # the ligand nodes in the subgraph.
        h_ligand_subgraph_nodes = self.W_ligand_nodes_embed(
            graph_features["ligand_subgraph_nodes"]
        )

        # Embed the ligand edges.
        # ligand_subgraph_edges [B, L, M, M, self.hidden_dim] - the embedding
        # of the ligand edges in the subgraph.
        h_ligand_subgraph_edges = self.W_ligand_edges_embed(
            graph_features["ligand_subgraph_edges"]
        )

        # Run the context encoder layers for the protein-ligand context.
        for i in range(self.num_context_encoding_layers):
            # Message passing in the ligand subgraph.
            # BUG: to replicate the original LigandMPNN,destination nodes are
            # not concatenated into ligand_subgraph_edges; This breaks message
            # passing in the small molecule graph (no message passing in the
            # ligand subgraph).
            h_ligand_subgraph_nodes = torch.utils.checkpoint.checkpoint(
                self.ligand_context_encoder_layers[i],
                h_ligand_subgraph_nodes,
                h_ligand_subgraph_edges,
                input_features["ligand_subgraph_Y_m"],
                ligand_subgraph_Y_m_edges,
                use_reentrant=False,
            )

            # Concatenate the protein-ligand edge features with the ligand
            # hidden note features (effectively treating the ligand subgraph
            # node features as protein-ligand edge features).
            # h_E_protein_to_ligand_cat [B, L, M, 2 * self.hidden_dim] - the
            # concatenated protein-ligand edge features.
            h_E_protein_to_ligand_cat = torch.cat(
                [h_E_protein_to_ligand, h_ligand_subgraph_nodes], -1
            )

            # h_V_context [B, L, self.hidden_dim] - the updated context node
            # features. Message passing from ligand subgraph to the protein.
            h_V_context = torch.utils.checkpoint.checkpoint(
                self.protein_ligand_context_encoder_layers[i],
                h_V_context,
                h_E_protein_to_ligand_cat,
                input_features["residue_mask"],
                input_features["ligand_subgraph_Y_m"],
                use_reentrant=False,
            )

        # Final context embedding.
        h_V_context = self.W_final_context_embed(h_V_context)

        # Update the protein node features with the context with a residual
        # connection (after apply dropout and layer norm to the context).
        encoder_features["h_V"] = encoder_features["h_V"] + self.final_context_norm(
            self.dropout(h_V_context)
        )

        return encoder_features

    def forward(self, network_input):
        """
        Forward pass for the LigandMPNN model, which uses the same forward
        function and is repeated here for documentation purposes.

        A NOTE on shapes (in addition to ProteinMPNN):
            - N = number of ligand atoms
            - M = self.num_context_atoms (number of ligand atom neighbors)

        Args:
            network_input (dict): Dictionary containing the input to the
                network.
                - input_features (dict): dictionary containing input features
                    and all necessary information for the model to run, in
                    addition to the input features for the ProteinMPNN model.
                    - Y (torch.Tensor): [B, N, 3] - 3D coordinates of the
                        ligand atoms.
                    - Y_m (torch.Tensor): [B, N] - mask indicating which ligand
                        atoms are valid.
                    - Y_t (torch.Tensor): [B, N] - element types of the ligand
                        atoms.
                    - atomize_side_chains (bool): Whether to atomize side
                        chains of fixed residues.
        Side Effects:
            Any changes denoted below to input_features are also mutated on the
            original input features.
        Returns:
            network_output (dict): Output dictionary containing the requested
                features based on the input features' "features_to_return" key,
                in addition to the output features from the ProteinMPNN model.
                - input_features (dict): The input features from above, with
                    the following keys added or modified:
                    - hide_side_chain_mask (torch.Tensor): [B, L] - mask for
                        hiding side chains, where True is a residue with hidden
                        side chains, and False is a residue with revealed side
                        chains.
                    - Y (torch.Tensor): [B, N, 3] - 3D coordinates of the ligand
                        atoms with added Gaussian noise.
                    - Y_pre_noise (torch.Tensor): [B, N, 3] -
                        3D coordinates of the ligand atoms before adding
                        Gaussian noise ('Y' before noise).
                    - ligand_subgraph_Y (torch.Tensor): [B, L, M, 3] - 3D
                        coordinates of nearest ligand/atomized side chain
                        atoms to the virtual atoms for each residue.
                    - ligand_subgraph_Y_m (torch.Tensor): [B, L, M] - mask
                        indicating which nearest ligand/atomized side chain
                        atoms to the virtual atoms are valid.
                    - ligand_subgraph_Y_t (torch.Tensor): [B, L, M] -
                        element types of the nearest ligand/atomized side chain
                        atoms to the virtual atoms for each residue.
                - graph_features (dict): The graph features.
                    - E_protein_to_ligand (torch.Tensor):
                        [B, L, M, num_node_output_features] - protein to
                        ligand subgraph edges; can also be considered node
                        features of the protein residues (although they are not
                        used as such).
                    - ligand_subgraph_nodes (torch.Tensor):
                        [B, L, M, num_node_output_features] - ligand atom type
                        information, embedded as node features.
                    - ligand_subgraph_edges (torch.Tensor):
                        [B, L, M, M, num_edge_output_features] - embedded and
                        normalized radial basis function embedding of the
                        distances between the ligand atoms in each residue
                        subgraph.
        """
        return super().forward(network_input)
