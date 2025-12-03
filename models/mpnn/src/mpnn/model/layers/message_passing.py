import torch
import torch.nn as nn
from mpnn.model.layers.position_wise_feed_forward import PositionWiseFeedForward


# Gather functions borrowed from ProteinMPNN;
# originally from:
# https://github.com/jingraham/neurips19-graph-protein-design/tree/master
def gather_edges(edge_features, neighbor_idx):
    """
    Gather edge features for the neighbors of each node.
    Args:
        edge_features: [B,L,L,H] - edge features
        neighbor_idx: [B,L,K] - neighbor indices
    Returns:
        edge_features_at_neighbors: [B,L,K,H] - neighbor edge features, gathered
            at the neighbor indices.
    """
    _, _, _, H = edge_features.shape

    # neighbor_idx_expand [B,L,K,H] - expand the neighbor indices along the
    # feature dimension.
    neighbor_idx_expand = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, H)

    # edge_features_at_neighbors [B,L,K,H] - gather the edge features at the
    # neighbor indices.
    edge_features_at_neighbors = torch.gather(edge_features, 2, neighbor_idx_expand)

    return edge_features_at_neighbors


def gather_nodes(node_features, neighbor_idx):
    """
    Gather node features for the neighbors of each node.

    NOTE: in most cases, L1 == L2. This is the straightforward case where the
    node features are gathered at the neighbor indices for every node. However,
    L2 can differ from L1, which allows for gathering node features of less
    nodes (useful for gathering features during decoding, where we decode
    one node at a time).

    Args:
        node_features: [B,L1,H] - node features
        neighbor_idx: [B,L2,K] - neighbor indices
    Returns:
        node_features_at_neighbors: [B,L2,K,H] - neighbor node features,
            gathered at the neighbor indices.
    """
    B, L2, K = neighbor_idx.shape
    _, _, H = node_features.shape

    # neighbor_idx_flat [B,L2 * K] - flatten the residue index and neighbor
    # index dimensions; this is done to allow for gathering.
    neighbor_idx_flat = neighbor_idx.reshape((B, -1))

    # neighbor_idx_flat_expand [B,L2 * K,H] - expand the neighbor indices along
    # the feature dimension.
    neighbor_idx_flat_expand = neighbor_idx_flat.unsqueeze(-1).expand(-1, -1, H)

    # node_features_at_neighbors_flat [B,L2 * K,H] - gather the node features
    # at the flattened neighbor indices.
    node_features_at_neighbors_flat = torch.gather(
        node_features, 1, neighbor_idx_flat_expand
    )

    # node_features_at_neighbors [B,L2,K,H] - reshape the gathered node
    # features to the original shape.
    node_features_at_neighbors = node_features_at_neighbors_flat.view(B, L2, K, H)

    return node_features_at_neighbors


def cat_neighbors_nodes(node_features, edge_features_at_neighbors, neighbor_idx):
    """
    Gather node features for the neighbors of each node and concatenate them
    with the edge features.

    NOTE: in most cases, L1 == L2. This is the straightforward case where the
    node features are gathered at the neighbor indices for every node, then
    concatenated with the edge features. However, L2 can differ from L1, which
    allows for gathering node features and concatenating to edge features for
    less nodes (useful for gathering features during decoding, where we decode
    one node at a time).

    Args:
        node_features [B,L1,H1]: node features
        edge_features_at_neighbors [B,L2,K,H2]: edge hidden states
        neighbor_idx [B,L2,K]: neighbor indices
    Returns:
        edge_and_node_features_at_neighbors [B,L2,K,H2+H1]: concatenated node
            and edge features, with the edge features first.
    """
    # node_features_at_neighbors [B,L2,K,H1] - gather the node features at the
    # neighbor indices.
    node_features_at_neighbors = gather_nodes(node_features, neighbor_idx)

    # edge_and_node_features_at_neighbors [B,L2,K,H2+H1] - concatenate the
    # gathered node features with the edge features.
    edge_and_node_features_at_neighbors = torch.cat(
        [edge_features_at_neighbors, node_features_at_neighbors], -1
    )

    return edge_and_node_features_at_neighbors


class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, scale=30):
        super(EncLayer, self).__init__()

        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)

        self.W11 = nn.Linear(num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)

        self.act = torch.nn.GELU()

        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_E=None):
        """
        Encoder message passing step; updates both the node and edge hidden
        states.

        NOTE: num_in = 3H

        Args:
            h_V [B, L, H] - node hidden states
            h_E [B, L, K, H] - edge hidden states
            E_idx [B, L, K] - edge indices
            mask_V [B, L] - node mask
            mask_E [B, L, K] - edge mask
        Returns:
            h_V [B, L, H] - updated node hidden states
            h_E [B, L, K, H] - updated edge hidden states
        """
        # Concatenate h_V_j to h_E_ij
        # (result h_E_ij cat h_V_j)
        # Shape: [B, L, K, H] + [B, L, H] => [B, L, K, 2H]
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)

        # Concatenate h_V_i to h_E_ij cat h_V_j
        # (result h_E_ij cat h_V_j cat h_V_i)
        # Shape (h_EV): [B, L, K, 2H] + [B, L, H] => [B, L, K, 3H]
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)

        # Compute the message.
        # Shape: [B, L, K, 3H] => [B, L, K, H]
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))

        # Apply the edge mask to the message.
        if mask_E is not None:
            h_message = mask_E.unsqueeze(-1) * h_message

        # Scaled sum aggregation.
        # Shape: [B, L, K, H] => [B, L, H]
        dh = torch.sum(h_message, -2) / self.scale

        # Dropout + residual + norm.
        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward.
        dh = self.dense(h_V)

        # Dropout + residual + norm.
        h_V = self.norm2(h_V + self.dropout2(dh))

        # Apply the node mask to the node hidden states.
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        # Concatenate h_V_j to h_E_ij (using the updated node state).
        # result h_E_ij cat h_V_j
        # Shape: [B, L, K, H] + [B, L, H] => [B, L, K, 2H]
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)

        # Concatenate h_V_i to h_E_ij cat h_V_j (using the updated node state).
        # result h_E_ij cat h_V_j cat h_V_i
        # Shape: [B, L, K, 2H] + [B, L, H] => [B, L, K, 3H]
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)

        # Compute an edge update.
        # Shape: [B, L, K, 3H] => [B, L, K, H]
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))

        # Add the edge update to the edge hidden states.
        # Dropout + residual + norm.
        h_E = self.norm3(h_E + self.dropout3(h_message))

        return h_V, h_E


class DecLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, scale=30):
        super(DecLayer, self).__init__()

        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)

        self.act = torch.nn.GELU()

        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_E=None):
        """
        Decoder message passing step; updates only the node hidden states.
        NOTE: this function is used for both the protein decoder and the ligand
        context encoder. As such, this function operates on the "deepest"
        graph in the tensor.

        Below, the shapes for the protein decoder application will be as
        follows:
            ... = empty
            node_num = L
            neighbor_num = K
            num_in = 4H
        For the ligand subgraph encoder, the shapes will be as follows:
            ... = L
            node_num = M
            neighbor_num = M
            num_in = 2H
                BUG: this should be 3H, but the original LigandMPNN does not
                pre-concatenate the destination node features to the edge
                features, which breaks the message passing in the ligand
                subgraphs.
        For the protein-ligand graph encoder, the shapes will be as follows:
            ... = empty
            node_num = L
            neighbor_num = M
            num_in = 3H

        Args:
            h_V [B, ..., node_num, H] - node hidden states
            h_E [B, ..., node_num, neighbor_num, num_in - H] - edge hidden
                states;
                NOTE: for message passing to behave in the decoder, the
                    destination node features (and sequence if applicable) MUST
                    be pre-concatenated to the edge features.
                    So, h_E is actually:
                        - protein decoder: h_E_ij cat h_S_j cat h_V_j
                        - ligand subgraph encoder: h_ligand_subgraph_edges_ij;
                            BUG: this should be h_ligand_subgraph_edges_ij cat
                            h_ligand_subgraph_nodes_j; in its current form
                            (replicating the original LigandMPNN), the
                            destination node features are not concatenated to
                            the edge features, which breaks the message passing
                            in the ligand subgraph.
                        - protein-ligand graph encoder: h_E_protein_to_ligand_ij
                            cat h_ligand_subgraph_nodes_j
            mask_V [B, ..., node_num] - node mask
            mask_E [B, ..., node_num, neighbor_num] - edge mask
        Returns:
            h_V [B, ..., node_num, H] - updated node hidden states
        """

        # Concatenate source node features to edge features, which include the
        # destination node features.
        #     - protein decoder: concatenate h_V_i to h_E_ij cat h_S_j cat h_V_j
        #         result: h_E_ij cat h_S_j cat h_V_j cat h_V_i
        #     - ligand subgraph encoder: concatenate h_ligand_subgraph_nodes_i
        #         to h_ligand_subgraph_edges_ij
        #         result: h_ligand_subgraph_edges_ij cat
        #             h_ligand_subgraph_nodes_i
        #     - protein-ligand graph encoder: concatenate h_V_i to
        #         h_E_protein_to_ligand_ij cat h_ligand_subgraph_nodes_j
        #         result: h_E_protein_to_ligand_ij cat h_ligand_subgraph_nodes_j
        #             cat h_V_i
        # Shape (h_EV): [B, ..., node_num, neighbor_num, num_in - H] +
        # [B, ..., node_num, H] => [B, ..., node_num, neighbor_num, num_in]
        h_V_expand = h_V.unsqueeze(-2).expand(
            *h_V.shape[:-1],  # B, ..., node_num
            h_E.size(-2),  # neighbor_num
            h_V.shape[-1],  # H
        )
        h_EV = torch.cat([h_V_expand, h_E], -1)

        # Compute the message.
        # Shape: [B, ..., node_num, neighbor_num, num_in] =>
        # [B, ..., node_num, neighbor_num, H]
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))

        # Apply the edge mask to the message.
        if mask_E is not None:
            h_message = mask_E.unsqueeze(-1) * h_message

        # Scaled sum aggregation.
        # Shape: [B, ..., node_num, neighbor_num, H] => [B, ..., node_num, H]
        dh = torch.sum(h_message, -2) / self.scale

        # Dropout + residual + norm.
        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)

        # Dropout + residual + norm.
        h_V = self.norm2(h_V + self.dropout2(dh))

        # Apply the node mask to the node hidden states.
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        return h_V
