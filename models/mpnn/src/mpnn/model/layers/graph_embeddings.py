import torch
import torch.nn as nn
from atomworks.constants import ELEMENT_NAME_TO_ATOMIC_NUMBER
from mpnn.model.layers.message_passing import gather_edges, gather_nodes
from mpnn.model.layers.positional_encoding import PositionalEncodings
from mpnn.transforms.feature_aggregation.token_encodings import MPNN_TOKEN_ENCODING


class ProteinFeatures(nn.Module):
    TOKEN_ENCODING = MPNN_TOKEN_ENCODING
    BACKBONE_ATOM_NAMES = ["N", "CA", "C", "O"]

    REPRESENTATIVE_ATOM_NAMES = ["CA"]

    DATA_TO_CALCULATE_VIRTUAL_ATOMS = [
        (
            {"center_atom": "CA", "atom_1": "N", "atom_2": "C"},
            {
                "weight_normal": 0.58273431,
                "weight_bond_1": -0.56802827,
                "weight_bond_2": -0.54067466,
            },
        )
    ]

    def __init__(
        self,
        num_edge_output_features=128,
        num_node_output_features=128,
        num_positional_embeddings=16,
        min_rbf_mean=2.0,
        max_rbf_mean=22.0,
        num_rbf=16,
        num_neighbors=48,
    ):
        """
        Given a protein structure, extract the features for the graph
        representation of the protein.

        Args:
            num_edge_output_features (int): Number of output features for the
                edges.
            num_node_output_features (int): Number of output features for the
                nodes.
            num_positional_embeddings (int): Number of positional embeddings.
            min_rbf_mean (float): Minimum mean for the radial basis functions.
            max_rbf_mean (float): Maximum mean for the radial basis functions.
            num_rbf (int): Number of radial basis functions.
            num_neighbors (int): Number of neighbors to consider for each
                residue.
        """
        super(ProteinFeatures, self).__init__()

        self.num_edge_output_features = num_edge_output_features
        self.num_node_output_features = num_node_output_features

        self.num_neighbors = num_neighbors

        self.min_rbf_mean = min_rbf_mean
        self.max_rbf_mean = max_rbf_mean
        self.num_rbf = num_rbf

        self.num_positional_embeddings = num_positional_embeddings

        self.num_backbone_atoms = len(self.BACKBONE_ATOM_NAMES)
        self.num_virtual_atoms = len(self.DATA_TO_CALCULATE_VIRTUAL_ATOMS)

        self.num_edge_input_features = num_positional_embeddings + num_rbf * (
            (self.num_backbone_atoms + self.num_virtual_atoms) ** 2
        )

        # Layers
        self.positional_embedding = PositionalEncodings(self.num_positional_embeddings)
        self.edge_embedding = nn.Linear(
            self.num_edge_input_features, self.num_edge_output_features, bias=False
        )
        self.edge_norm = nn.LayerNorm(self.num_edge_output_features)

    def construct_X_atoms(self, X, X_m, S, atom_names):
        """
        Given an array of 3D coordinates and the corresponding atom mask, use
        the sequence and the atom names to construct a subset of X and X_m that
        contains only the requested atoms.

        Args:
            X (torch.Tensor): [B, L, self.TOKEN_ENCODING.n_atoms_per_token, 3] -
                3D coordinates of polymer atoms.
            X_m (torch.Tensor): [B, L, self.TOKEN_ENCODING.n_atoms_per_token] -
                Mask indicating which polymer atoms are valid.
            S (torch.Tensor): [B, L] - the sequence of residues.
            atom_names (list): List of atom names to extract from the
                coordinates.
        Returns:
            X_atoms (torch.Tensor): [B, L, len(atom_names), 3] - 3D coordinates
                of the requested atoms for each residue.
            X_m_atoms (torch.Tensor): [B, L, len(atom_names)] - mask indicating
                which requested atoms are valid for each residue.
        """
        B, L, _, _ = X.shape

        # token_and_atom_name_to_atom_idx [self.TOKEN_ENCODING.n_tokens,
        # len(atom_names)] - a tensor that maps each token/atom name pair to the
        # corresponding atom index.
        token_and_atom_name_to_atom_idx = torch.zeros(
            (self.TOKEN_ENCODING.n_tokens, len(atom_names)),
            device=X.device,
            dtype=torch.int64,
        )
        for token_name, token_idx in self.TOKEN_ENCODING.token_to_idx.items():
            for i, atom_name in enumerate(atom_names):
                token_and_atom_name_to_atom_idx[token_idx, i] = (
                    self.TOKEN_ENCODING.atom_to_idx[(token_name, atom_name)]
                )

        # batch_idx [B, 1, 1] - a tensor that contains the batch index.
        batch_idx = torch.arange(B, dtype=torch.int64, device=X.device).view(B, 1, 1)

        # position_idx [1, L, 1] - a tensor that contains the position index.
        position_idx = torch.arange(L, dtype=torch.int64, device=X.device).view(1, L, 1)

        # atom_indices [B, L, len(atom_names)] - a tensor that contains the
        # atom index for each residue for each atom name.
        atom_indices = token_and_atom_name_to_atom_idx[S]

        # X_atoms [B, L, len(atom_names), 3] - 3D coordinates of the atoms for
        # each residue.
        X_atoms = X[batch_idx, position_idx, atom_indices]

        # X_m_atoms [B, L, len(atom_names)] - mask indicating which atoms are
        # valid for each residue.
        X_m_atoms = X_m[batch_idx, position_idx, atom_indices]

        return X_atoms, X_m_atoms

    def construct_X_rep_atoms(self, X, X_m, S):
        """
        Given an array of 3D coordinates, construct a subset of X that
        contains only the representative atom for each residue.

        Args:
            X (torch.Tensor): [B, L, self.TOKEN_ENCODING.n_atoms_per_token, 3] -
                3D coordinates of polymer atoms.
            X_m (torch.Tensor): [B, L, self.TOKEN_ENCODING.n_atoms_per_token] -
                Mask indicating which polymer atoms are valid.
            S (torch.Tensor): [B, L] - the sequence of residues.
        Returns:
            X_rep_atoms (torch.Tensor):
                [B, L, len(self.REPRESENTATIVE_ATOM_NAMES), 3] - 3D coordinates
                of the representative atoms for each residue.
            X_m_rep_atoms (torch.Tensor):
                [B, L, len(self.REPRESENTATIVE_ATOM_NAMES)] - mask indicating
                which representative atoms are valid.
        """
        X_rep_atoms, X_m_rep_atoms = self.construct_X_atoms(
            X, X_m, S, self.REPRESENTATIVE_ATOM_NAMES
        )

        # Check that the representative atoms are disjoint (only one per
        # residue).
        if torch.any(torch.sum(X_m_rep_atoms, dim=-1) > 1):
            raise ValueError("Each residue should have only one representative atom.")

        return X_rep_atoms, X_m_rep_atoms

    def construct_X_backbone(self, X, X_m, S):
        """
        Given an array of 3D coordinates, construct a subset of X that
        contains only the backbone atoms for each residue.

        Args:
            X (torch.Tensor): [B, L, self.TOKEN_ENCODING.n_atoms_per_token, 3] -
                3D coordinates of polymer atoms.
            X_m (torch.Tensor): [B, L, self.TOKEN_ENCODING.n_atoms_per_token] -
                Mask indicating which polymer atoms are valid.
            S (torch.Tensor): [B, L] - the sequence of residues.
        Returns:
            X_backbone (torch.Tensor): [B, L, len(self.BACKBONE_ATOM_NAMES), 3]
                - 3D coordinates of the backbone atoms for each residue.
            X_m_backbone (torch.Tensor): [B, L, len(self.BACKBONE_ATOM_NAMES)] -
                Mask indicating which backbone atoms are valid.
        """
        X_backbone, X_m_backbone = self.construct_X_atoms(
            X, X_m, S, self.BACKBONE_ATOM_NAMES
        )

        return X_backbone, X_m_backbone

    def construct_X_virtual_atom(
        self,
        X_center_atom,
        X_atom_1,
        X_atom_2,
        weight_normal,
        weight_bond_1,
        weight_bond_2,
    ):
        """
        Predict the virtual atom coordinates based on the coordinates of the
        center atom and the two other atoms.

        Args:
            X_center_atom (torch.Tensor): [B, L, 3] - 3D coordinates of the
                center atom.
            X_atom_1 (torch.Tensor): [B, L, 3] - 3D coordinates of the first
                atom.
            X_atom_2 (torch.Tensor): [B, L, 3] - 3D coordinates of the second
                atom.
            weight_normal (float): Weight for the normal vector.
            weight_bond_1 (float): Weight for the first bond vector.
            weight_bond_2 (float): Weight for the second bond vector.
        """
        # Calculate the bond vectors.
        # bond_1 [B, L, 3] - vector from the center atom to the first atom.
        bond_1 = X_atom_1 - X_center_atom

        # bond_2 [B, L, 3] - vector from the center atom to the second atom.
        bond_2 = X_atom_2 - X_center_atom

        # normal [B, L, 3] - normal vector to the plane defined by the two
        # bond vectors.
        normal = torch.cross(bond_1, bond_2, dim=-1)

        # X_virtual_atom [B, L, 3] - the coordinates of the virtual atom.
        X_virtual_atom = (
            weight_normal * normal
            + weight_bond_1 * bond_1
            + weight_bond_2 * bond_2
            + X_center_atom
        )

        return X_virtual_atom

    def construct_X_virtual_atoms(self, X, X_m, S):
        """
        Given an array of 3D coordinates, construct a the virtual atoms.

        Args:
            X (torch.Tensor): [B, L, self.TOKEN_ENCODING.n_atoms_per_token, 3] -
                3D coordinates of polymer atoms.
            X_m (torch.Tensor): [B, L, self.TOKEN_ENCODING.n_atoms_per_token] -
                Mask indicating which polymer atoms are valid.
            S (torch.Tensor): [B, L] - the sequence of residues.
        Returns:
            X_virtual_atoms (torch.Tensor):
                [B, L, len(self.DATA_TO_CALCULATE_VIRTUAL_ATOMS), 3] - 3D
                coordinates of the virtual atoms for each residue.
            X_m_virtual_atoms (torch.Tensor):
                [B, L, len(self.DATA_TO_CALCULATE_VIRTUAL_ATOMS)] - Mask
                indicating which virtual atoms are valid.
        """
        X_virtual_atoms = []
        X_m_virtual_atoms = []
        for virtual_atom_data in self.DATA_TO_CALCULATE_VIRTUAL_ATOMS:
            virtual_atom_info, weights = virtual_atom_data
            center_atom = virtual_atom_info["center_atom"]
            atom_1 = virtual_atom_info["atom_1"]
            atom_2 = virtual_atom_info["atom_2"]

            # Get the coordinates and masks for the center atom and two
            # other atoms. Stack the coordinates and masks of the three atoms
            # to reduce the number of calls to construct_X_atoms.
            atom_names = [center_atom, atom_1, atom_2]
            X_atoms, X_m_atoms = self.construct_X_atoms(X, X_m, S, atom_names)

            # X_center_atom, X_atom_1, X_atom_2 [B, L, 3] - 3D coordinates of
            # the center atom and the two other atoms.
            X_center_atom = X_atoms[:, :, atom_names.index(center_atom), :]
            X_atom_1 = X_atoms[:, :, atom_names.index(atom_1), :]
            X_atom_2 = X_atoms[:, :, atom_names.index(atom_2), :]

            # X_m_center_atom, X_m_atom_1, X_m_atom_2 [B, L] - mask indicating
            # if the center atom and the two other atoms are valid for each
            # residue.
            X_m_center_atom = X_m_atoms[:, :, atom_names.index(center_atom)]
            X_m_atom_1 = X_m_atoms[:, :, atom_names.index(atom_1)]
            X_m_atom_2 = X_m_atoms[:, :, atom_names.index(atom_2)]

            # X_virtual_atom [B, L, 3] - 3D coordinates of the virtual atom
            # constructed from the center atom and the two other atoms.
            X_virtual_atom = self.construct_X_virtual_atom(
                X_center_atom,
                X_atom_1,
                X_atom_2,
                weight_normal=weights["weight_normal"],
                weight_bond_1=weights["weight_bond_1"],
                weight_bond_2=weights["weight_bond_2"],
            )
            X_virtual_atoms.append(X_virtual_atom)

            # X_m_virtual_atom [B, L] - mask indicating if the virtual atom
            # is valid for each residue.
            X_m_virtual_atom = X_m_center_atom * X_m_atom_1 * X_m_atom_2
            X_m_virtual_atoms.append(X_m_virtual_atom)

        # X_virtual_atoms [B, L, len(self.DATA_TO_CALCULATE_VIRTUAL_ATOMS), 3] -
        # coordinates of the virtual atoms for each residue.
        X_virtual_atoms = torch.stack(X_virtual_atoms, dim=2)

        # X_m_virtual_atoms [B, L, len(self.DATA_TO_CALCULATE_VIRTUAL_ATOMS)] -
        # mask indicating which virtual atoms are valid for each residue.
        X_m_virtual_atoms = torch.stack(X_m_virtual_atoms, dim=2)

        # Check that the virtual atoms are disjoint (only one per residue).
        if torch.any(torch.sum(X_m_virtual_atoms, dim=-1) > 1):
            raise ValueError("Each residue should have only one virtual atom.")

        return X_virtual_atoms, X_m_virtual_atoms

    def compute_representative_atom_pairwise_distances(
        self, X_rep_atoms, X_m_rep_atoms, residue_mask, eps=1e-6
    ):
        """
        Given an array of 3D coordinates, compute the pairwise distances
        between all pairs of atoms. The masked distances are set to the
        maximum distance.

        Args:
            X_rep_atoms (torch.Tensor):
                [B, L, len(self.REPRESENTATIVE_ATOM_NAMES), 3] - 3D coordinates
                of the representative atoms for each residue.
            X_m_rep_atoms (torch.Tensor):
                [B, L, len(self.REPRESENTATIVE_ATOM_NAMES)] - mask indicating
                which representative atoms are valid.
            residue_mask (torch.Tensor): [B, L] - mask indicating which residues
                are valid.
            eps (float): Small value used to distances that are
                numerically zero.
        Returns:
            D_rep_neighbors (torch.Tensor): [B, L, K] - Pairwise distances
                between each residue's representative atom, masked by the
                2D mask, for the top K neighbors.
            E_idx (torch.Tensor): [B, L, K] - Indices of the top K neighbors
                for each residue's representative atom.
        """
        _, L, _, _ = X_rep_atoms.shape

        # mask_2D [B, L, L] - 2D mask indicating which pairs of residues
        # are valid.
        mask_2D = (
            torch.unsqueeze(residue_mask, 1) * torch.unsqueeze(residue_mask, 2)
        ).bool()

        # X_rep_atoms_collapsed [B, L, 3] - collapse the representative atom
        # dimension.
        # NOTE: collapsing along this dimension is okay because the
        # self.construct_X_rep_atoms function ensures that there is only one
        # representative atom per residue.
        X_rep_atoms_collapsed = torch.sum(
            X_rep_atoms * X_m_rep_atoms[:, :, :, None], dim=2
        )

        # dX [B, L, L, 3] - pairwise per-coordinate differences between each
        # residue's representative atom.
        dX_rep = torch.unsqueeze(X_rep_atoms_collapsed, 1) - torch.unsqueeze(
            X_rep_atoms_collapsed, 2
        )

        # D_rep [B, L, L] - pairwise distances between each residue's
        # representative atom, masked by the 2D mask.
        D_rep = mask_2D * torch.sqrt(torch.sum(dX_rep**2, 3) + eps)

        # D_rep_max [B, L, L] - a constant value that is the maximum distance
        # between any two representative atoms in each batch entry.
        D_rep_max, _ = torch.max(D_rep, -1, keepdim=True)

        # D_rep_adjust [B, L, L] - the pairwise distances between each residue's
        # representative atom, with masked distances set to the maximum
        # distance.
        D_rep_adjust = D_rep + (~mask_2D) * D_rep_max

        # D_rep_neighbors [B, L, K] - the top K pairwise distances between
        # each residue's representative atom, masked by the 2D mask.
        # E_idx [B, L, K] - the indices of the top K pairwise distances
        # between each residue's representative atom.
        D_rep_neighbors, E_idx = torch.topk(
            D_rep_adjust, min(self.num_neighbors, L), dim=-1, largest=False
        )

        return D_rep_neighbors, E_idx

    def compute_rbf_embedding_from_distances(self, D):
        """
        Given a tensor of pairwise distances, compute the radial basis
        embedding of the distances.

        Args:
            D (torch.Tensor): [B, L, K] - Pairwise distances between each
                residue's representative atom, masked by the 2D mask.
        Returns:
            rbf_embedding (torch.Tensor): [B, L, K, num_rbf] - Radial basis
                function embedding of the pairwise distances.
        """
        # Linear space the means of the radial basis functions.
        # rbf_mus: [1, 1, 1, num_rbf]
        rbf_mus = torch.linspace(
            self.min_rbf_mean, self.max_rbf_mean, self.num_rbf, device=D.device
        )
        rbf_mus = rbf_mus[None, None, None, :]

        # The standard deviation of the radial basis functions.
        rbf_sigma = (self.max_rbf_mean - self.min_rbf_mean) / self.num_rbf

        # Expand the dimensions of D to match the shape of rbf_mus.
        # D_expand: [B, L, K, 1]
        D_expand = torch.unsqueeze(D, -1)

        # Compute the radial basis function embedding.
        # RBF: [B, L, K, num_rbf]
        rbf_embedding = torch.exp(-(((D_expand - rbf_mus) / rbf_sigma) ** 2))

        return rbf_embedding

    def compute_pairwise_residue_rbf_encoding(self, X, E_idx, X_m, eps=1e-6):
        """
        Given an array of 3D coordinates, compute the atom by atom pairwise
        distances between each pair of neighbors. Mask the RBF features using
        the atom mask.

        NOTE: num_atoms = self.num_backbone_atoms + self.num_virtual_atoms

        Args:
            X (torch.Tensor): [B, L, num_atoms, 3] - 3D coordinates of
                polymer atoms.
            E_idx (torch.Tensor): [B, L, K] - Indices of the top K neighbors.
            X_m (torch.Tensor): [B, L, num_atoms] - mask indicating which
                polymer atoms are valid.
            eps (float): Small value added to distances that are zero.
        Returns:
            RBF_all (torch.Tensor): [B, L, K, num_atoms * num_atoms * num_rbf] -
                Radial basis function embedding of the pairwise atomic
                distances for each pair of residue neighbors.
        """
        B = X.shape[0]
        L = X.shape[1]
        K = E_idx.shape[2]
        num_atoms = X.shape[2]

        # X_flat [B, L, num_atoms * 3] - flatten the last two dimensions.
        X_flat = X.reshape(B, L, -1)

        # X_flat_g [B, L, K, num_atoms * 3] - gather the top K neighbors.
        X_flat_g = gather_nodes(X_flat, E_idx)

        # X_g [B, L, K, num_atoms, 3] - reshape the gathered tensor.
        X_g = X_flat_g.reshape(B, L, K, num_atoms, 3)

        # D [B, L, K, num_atoms, num_atoms] - pairwise distances between
        # each residue's atoms.
        D = torch.sqrt(
            torch.sum(
                (X[:, :, None, :, None, :] - X_g[:, :, :, None, :, :]) ** 2, dim=-1
            )
            + eps
        )

        # RBF_all [B, L, K, num_atoms, num_atoms, num_rbf] - radial basis
        # function embedding of the pairwise distances.
        RBF_all = self.compute_rbf_embedding_from_distances(D)

        # If X_m is not all ones, mask the radial basis function embedding
        # with the atom mask.
        if not torch.all(X_m == 1):
            # X_m_gathered [B, L, K, num_atoms] - gather the atom mask of the
            # top K neighbors.
            X_m_gathered = gather_nodes(X_m, E_idx)

            # RBF_all [B, L, K, num_atoms, num_atoms, num_rbf] - mask the
            # radial basis function embedding with the atom mask.
            RBF_all = (
                RBF_all
                * X_m[:, :, None, :, None, None]
                * X_m_gathered[:, :, :, None, :, None]
            )

        # RBF_all [B, L, K, num_atoms * num_atoms * num_rbf] - flatten the
        # last dimensions.
        RBF_all = RBF_all.view(B, L, K, -1)

        return RBF_all

    def compute_pairwise_positional_encoding(self, R_idx, E_idx, chain_labels):
        """
        Given the indices of the residues and the indices of the top K
        neighbors, compute the positional encoding of the top K neighbors
        for each residue.

        Args:
            R_idx (torch.Tensor): [B, L] - indices of the residues.
            E_idx (torch.Tensor): [B, L, K] - indices of the top K neighbors.
            chain_labels (torch.Tensor): [B, L] - chain labels for each residue.
        Returns:
            positional_encoding (torch.Tensor):
                [B, L, K, num_positional_embeddings] - the positional encoding
                of the top K neighbors.
        """
        # positional_offset [B, L, L] - pairwise differences between the
        # indices of the residues.
        positional_offset = (R_idx[:, :, None] - R_idx[:, None, :]).long()

        # positional_offset_g [B, L, K] - gather the positional offset
        # of the top K neighbors.
        positional_offset_g = gather_edges(positional_offset[:, :, :, None], E_idx)[
            :, :, :, 0
        ]

        # same_chain_mask [B, L, L] - mask indicating which residues are in the
        # same chain.
        same_chain_mask = (chain_labels[:, :, None] - chain_labels[:, None, :]) == 0

        # same_chain_mask_g [B, L, K] - gather the same chain mask of the
        # top K neighbors.
        same_chain_mask_g = gather_edges(same_chain_mask[:, :, :, None], E_idx)[
            :, :, :, 0
        ]

        # positional_encoding [B, L, K, num_positional_embeddings] - the
        # positional encoding of the top K neighbors.
        positional_encoding = self.positional_embedding(
            positional_offset_g, same_chain_mask_g
        )

        return positional_encoding

    def featurize_edges(self, input_features):
        """
        Given input features, construct the edge features for the protein.

        Args:
            input_features (dict): Dictionary containing the input features.
                - residue_mask (torch.Tensor): [B, L] - Mask indicating which
                    residues are valid.
                - R_idx (torch.Tensor): [B, L] - Indices of the residues.
                - chain_labels (torch.Tensor): [B, L] - Chain labels for each
                    residue.
                - X_backbone (torch.Tensor):
                    [B, L, len(self.BACKBONE_ATOM_NAMES), 3] - 3D
                    coordinates of the backbone atoms for each residue.
                - X_m_backbone (torch.Tensor):
                    [B, L, len(self.BACKBONE_ATOM_NAMES)] - mask
                    indicating which backbone atoms are valid.
                - X_virtual_atoms (torch.Tensor):
                    [B, L, len(self.DATA_TO_CALCULATE_VIRTUAL_ATOMS), 3] -
                    3D coordinates of the virtual atoms for each residue.
                - X_m_virtual_atoms (torch.Tensor):
                    [B, L, len(self.DATA_TO_CALCULATE_VIRTUAL_ATOMS)] -
                    mask indicating which virtual atoms are valid.
                - X_rep_atoms (torch.Tensor):
                    [B, L, len(self.REPRESENTATIVE_ATOM_NAMES), 3] - 3D
                    coordinates of the representative atoms for each residue.
                - X_m_rep_atoms (torch.Tensor):
                    [B, L, len(self.REPRESENTATIVE_ATOM_NAMES)] - mask
                    indicating which representative atoms are valid.
        Returns:
            edge_features (dict): Dictionary containing the edge features.
                - E_idx (torch.Tensor): [B, L, K] - Indices of the top K
                    neighbors.
                - E (torch.Tensor): [B, L, K, num_edge_output_features] -
                    Edge features for each pair of neighbors.
        """
        # The following features should come from data loading.
        if "residue_mask" not in input_features:
            raise ValueError("Input features must contain 'residue_mask' key.")
        if "R_idx" not in input_features:
            raise ValueError("Input features must contain 'R_idx' key.")
        if "chain_labels" not in input_features:
            raise ValueError("Input features must contain 'chain_labels' key.")

        # The following features should be computed by the forward function.
        if "X_backbone" not in input_features:
            raise ValueError("Input features must contain 'X_backbone' key.")
        if "X_m_backbone" not in input_features:
            raise ValueError("Input features must contain 'X_m_backbone' key.")
        if "X_virtual_atoms" not in input_features:
            raise ValueError("Input features must contain 'X_virtual_atoms' key.")
        if "X_m_virtual_atoms" not in input_features:
            raise ValueError("Input features must contain 'X_m_virtual_atoms' key.")
        if "X_rep_atoms" not in input_features:
            raise ValueError("Input features must contain 'X_rep_atoms' key.")
        if "X_m_rep_atoms" not in input_features:
            raise ValueError("Input features must contain 'X_m_rep_atoms' key.")

        # Compute the pairwise distances between the representative atoms.
        # D_rep_neighbors [B, L, K] - pairwise distances between each residue's
        # representative atom, masked by the 2D mask.
        # E_idx [B, L, K] - indices of the top K neighbors.
        D_rep_neighbors, E_idx = self.compute_representative_atom_pairwise_distances(
            input_features["X_rep_atoms"],
            input_features["X_m_rep_atoms"],
            input_features["residue_mask"],
        )

        # Concatenate the backbone and virtual atom coordinates.
        # X_backbone_with_virtual_atoms [B, L, num_atoms + num_virtual_atoms, 3]
        # - 3D coordinates of the backbone and virtual atoms for each residue.
        # X_m_backbone_with_virtual_atoms [B, L, num_atoms + num_virtual_atoms]
        # - mask indicating which backbone and virtual atoms are valid.
        X_backbone_with_virtual_atoms = torch.cat(
            (input_features["X_backbone"], input_features["X_virtual_atoms"]), dim=-2
        )
        X_m_backbone_with_virtual_atoms = torch.cat(
            (input_features["X_m_backbone"], input_features["X_m_virtual_atoms"]),
            dim=-1,
        )

        # Compute the RBF features for the atomwise distances for each pair of
        # neighbors.
        # RBF_all [B, L, K, num_atoms * num_atoms * num_rbf] - radial basis
        # function embedding of the pairwise atomic distances for each pair of
        # residue neighbors.
        RBF_all = self.compute_pairwise_residue_rbf_encoding(
            X_backbone_with_virtual_atoms, E_idx, X_m_backbone_with_virtual_atoms
        )

        # Compute the positional encoding for the top K neighbors.
        # positional_encoding [B, L, K, num_positional_embeddings] - the
        # positional encoding of the top K neighbors.
        positional_encoding = self.compute_pairwise_positional_encoding(
            input_features["R_idx"], E_idx, input_features["chain_labels"]
        )

        # Concatenate the positional encoding and the RBF features.
        # E [B, L, K, num_positional_embeddings + num_atoms * num_atoms *
        # num_rbf] - the edge features for each pair of neighbors.
        E_raw = torch.cat((positional_encoding, RBF_all), dim=-1)

        # Embed and normalize the edge features.
        # E [B, L, K, num_edge_output_features] - the edge features for each
        # pair of neighbors.
        E = self.edge_embedding(E_raw)
        E = self.edge_norm(E)

        edge_features = {
            "E_idx": E_idx,
            "E": E,
        }

        return edge_features

    def featurize_nodes(self, input_features, edge_features):
        """
        The default ProteinMPNN does not have any node features.

        Args:
            input_features (dict): Dictionary containing the input features.
            edge_features (dict): Dictionary containing the edge features.
        Returns:
            node_features (dict): Dictionary containing the node features.
        """
        node_features = {}
        return node_features

    def noise_structure(self, input_features):
        """
        Given input features containing 3D coordinates of atoms, add Gaussian
        noise to the coordinates.

        Args:
            input_features (dict): Dictionary containing the input features.
                - X (torch.Tensor): [B, L, num_atoms, 3] - 3D coordinates of
                    polymer atoms.
                - structure_noise (float): Standard deviation of the
                    Gaussian noise to add to the input coordinates, in
                    Angstroms.
        Side Effects:
            input_features["X_pre_noise"] (torch.Tensor): [B, L, num_atoms, 3] -
                3D coordinates of polymer atoms before adding noise.
            input_features["X"] (torch.Tensor): [B, L, num_atoms, 3] - 3D
                coordinates of polymer atoms with added Gaussian noise.
        """
        if "X" not in input_features:
            raise ValueError("Input features must contain 'X' key.")
        if "structure_noise" not in input_features:
            raise ValueError("Input features must contain 'structure_noise' key.")

        structure_noise = input_features["structure_noise"]

        # If the noise is non-zero, add Gaussian noise to the input
        # coordinates.
        if structure_noise > 0:
            # Copy the original coordinates before adding noise.
            input_features["X_pre_noise"] = input_features["X"].clone()

            # Add Gaussian noise to the input coordinates.
            input_features["X"] = input_features[
                "X"
            ] + structure_noise * torch.randn_like(input_features["X"])
        else:
            input_features["X_pre_noise"] = input_features["X"].clone()

    def forward(self, input_features):
        """
        Given input features, construct the graph features for the protein.

        Args:
            input_features (dict): Dictionary containing the input features.
                - X (torch.Tensor): [B, L, num_atoms, 3] - 3D coordinates of
                    polymer atoms.
                - X_m (torch.Tensor): [B, L, num_atoms] - Mask indicating
                    which polymer atoms are valid.
        Returns:
            graph_features (dict): Dictionary containing the graph features.
                Union of edge and node features (see the repsective featurize
                functions).
        """
        if "X" not in input_features:
            raise ValueError("Input features must contain 'X' key.")
        if "X_m" not in input_features:
            raise ValueError("Input features must contain 'X_m' key.")
        if "S" not in input_features:
            raise ValueError("Input features must contain 'S' key.")

        # Add Gaussian noise to the input coordinates.
        self.noise_structure(input_features)

        # Get the backbone atoms and mask.
        # X_backbone [B, L, len(self.BACKBONE_ATOM_NAMES), 3] - 3D coordinates
        # of the backbone atoms for each residue.
        # X_m_backbone [B, L, len(self.BACKBONE_ATOM_NAMES)] - mask indicating
        # which backbone atoms are valid.
        X_backbone, X_m_backbone = self.construct_X_backbone(
            input_features["X"], input_features["X_m"], input_features["S"]
        )
        input_features["X_backbone"] = X_backbone
        input_features["X_m_backbone"] = X_m_backbone

        # Get the virtual atoms and mask.
        # X_virtual_atoms [B, L, len(self.DATA_TO_CALCULATE_VIRTUAL_ATOMS), 3] -
        # 3D coordinates of the virtual atoms for each residue.
        # X_m_virtual_atoms [B, L, len(self.DATA_TO_CALCULATE_VIRTUAL_ATOMS)] -
        # mask indicating which virtual atoms are valid.
        X_virtual_atoms, X_m_virtual_atoms = self.construct_X_virtual_atoms(
            input_features["X"], input_features["X_m"], input_features["S"]
        )
        input_features["X_virtual_atoms"] = X_virtual_atoms
        input_features["X_m_virtual_atoms"] = X_m_virtual_atoms

        # Get the representative atoms.
        # X_rep_atoms [B, L, len(self.REPRESENTATIVE_ATOM_NAMES), 3] - 3D
        # coordinates of the representative atoms for each residue.
        # X_m_rep_atoms [B, L, len(self.REPRESENTATIVE_ATOM_NAMES)] - mask
        # indicating which representative atoms are valid.
        X_rep_atoms, X_m_rep_atoms = self.construct_X_rep_atoms(
            input_features["X"], input_features["X_m"], input_features["S"]
        )
        input_features["X_rep_atoms"] = X_rep_atoms
        input_features["X_m_rep_atoms"] = X_m_rep_atoms

        # Featurize the edges.
        edge_features = self.featurize_edges(input_features)

        # Featurize the nodes.
        # Edge features are sometimes needed for node feature calculation;
        # for instance, for gathering nearest neighbor side chain atoms for
        # per-residue ligand subgraphs in LigandMPNN.
        node_features = self.featurize_nodes(input_features, edge_features)

        # Construct the graph features.
        graph_features = {**edge_features, **node_features}

        return graph_features


class ProteinFeaturesMembrane(ProteinFeatures):
    def __init__(self, num_membrane_classes=3, **kwargs):
        """
        Given a protein structure, extract the features for the graph
        representation of the protein. This class is aware of membrane labels.

        All args are the same as the parents class, except for the following:
        Args:
            num_membrane_classes (int): Number of membrane classes.
        """
        super(ProteinFeaturesMembrane, self).__init__(**kwargs)
        self.num_membrane_classes = num_membrane_classes

        self.node_embedding = nn.Linear(
            self.num_classes, self.num_node_output_features, bias=False
        )
        self.node_norm = nn.LayerNorm(self.num_node_output_features)

    def featurize_nodes(self, input_features, edge_features):
        """
        Given input features, construct the node features for the protein.

        Args:
            input_features (dict): Dictionary containing the input features.
                - membrane_per_residue_labels (torch.Tensor): [B, L] - Class
                    labels for each residue.
            edge_features (dict): Dictionary containing the edge features.
        Returns:
            node_features (dict): Dictionary containing the node features.
                - V (torch.Tensor): [B, L, self.num_node_output_features] - Node
                    features for each residue.
        """
        if "membrane_per_residue_labels" not in input_features:
            raise ValueError(
                "Input features must contain 'membrane_per_residue_labels' key."
            )

        # Turn the class labels into one-hot vectors.
        # class_one_hot [B, L, self.num_membrane_classes] - one-hot encoding
        # of the class
        class_one_hot = torch.nn.functional.one_hot(
            input_features["membrane_per_residue_labels"],
            num_classes=self.num_membrane_classes,
        ).float()

        # Embed and normalize the node features.
        # V [B, L, self.num_node_output_features] - the node features for each
        # residue.
        V = self.node_embedding(class_one_hot)
        V = self.node_norm(V)

        node_features = {"V": V}

        return node_features


class ProteinFeaturesPSSM(ProteinFeatures):
    def __init__(self, num_pssm_features=20, **kwargs):
        """
        Given a protein structure, extract the features for the graph
        representation of the protein. This class is aware of PSSM features.

        All args are the same as the parents class, except for the following:
        Args:
            num_pssm_features (int): Number of PSSM features.
        """
        super(ProteinFeaturesPSSM, self).__init__(**kwargs)
        self.num_pssm_features = num_pssm_features

        self.node_embedding = nn.Linear(
            self.num_pssm_features, self.num_node_output_features, bias=False
        )
        self.node_norm = nn.LayerNorm(self.num_node_output_features)

    def featurize_nodes(self, input_features, edge_features):
        """
        Given input features, construct the node features for the protein.

        Args:
            input_features (dict): Dictionary containing the input features.
                - pssm (torch.Tensor): [B, L, self.num_pssm_features] - PSSM
                    features for each residue.
            edge_features (dict): Dictionary containing the edge features.
        Returns:
            node_features (dict): Dictionary containing the node features.
                - V (torch.Tensor): [B, L, self.num_node_output_features] - Node
                    features for each residue.
        """
        if "pssm" not in input_features:
            raise ValueError("Input features must contain 'pssm' key.")

        # Embed and normalize the node features.
        # V [B, L, self.num_node_output_features] - the node features for each
        # residue.
        V = self.node_embedding(input_features["pssm"])
        V = self.node_norm(V)

        node_features = {"V": V}

        return node_features


class ProteinFeaturesLigand(ProteinFeatures):
    # Note, CB is excluded from the side chain atoms due to the use of the
    # virtual CB.
    SIDE_CHAIN_ATOM_NAMES = [
        "CG",
        "CG1",
        "CG2",
        "OG",
        "OG1",
        "SG",
        "CD",
        "CD1",
        "CD2",
        "ND1",
        "ND2",
        "OD1",
        "OD2",
        "SD",
        "CE",
        "CE1",
        "CE2",
        "CE3",
        "NE",
        "NE1",
        "NE2",
        "OE1",
        "OE2",
        "CH2",
        "NH1",
        "NH2",
        "OH",
        "CZ",
        "CZ2",
        "CZ3",
        "NZ",
        "OXT",
    ]

    # Mapping of side chain atom name to element name.
    SIDE_CHAIN_ATOM_NAME_TO_ELEMENT_NAME = {
        "CG": "C",
        "CG1": "C",
        "CG2": "C",
        "OG": "O",
        "OG1": "O",
        "SG": "S",
        "CD": "C",
        "CD1": "C",
        "CD2": "C",
        "ND1": "N",
        "ND2": "N",
        "OD1": "O",
        "OD2": "O",
        "SD": "S",
        "CE": "C",
        "CE1": "C",
        "CE2": "C",
        "CE3": "C",
        "NE": "N",
        "NE1": "N",
        "NE2": "N",
        "OE1": "O",
        "OE2": "O",
        "CH2": "C",
        "NH1": "N",
        "NH2": "N",
        "OH": "O",
        "CZ": "C",
        "CZ2": "C",
        "CZ3": "C",
        "NZ": "N",
        "OXT": "O",
    }

    def __init__(self, num_neighbors=32, num_context_atoms=25, **kwargs):
        """
        Given a protein structure and ligand structure, extract the features for
        the graph representation of the protein and ligand. This class is aware
        of ligand features.

        All args are the same as the parents class, except for the following:
        Args:
            num_neighbors (int): Number of neighbors to consider, default
                changed from 48 to 32.
            num_context_atoms (int): Number of ligand plus side chain atoms to
                consider for each polymer residue.
        """
        super(ProteinFeaturesLigand, self).__init__(
            num_neighbors=num_neighbors, **kwargs
        )
        self.num_context_atoms = num_context_atoms

        # Number of side chain atoms.
        self.num_side_chain_atoms = len(self.SIDE_CHAIN_ATOM_NAMES)

        # Features for atom type (periodic table features):
        # There is a null group, period, and atomic number.
        self.num_periodic_table_groups = 1 + 18
        self.num_periodic_table_periods = 1 + 7
        self.num_atomic_numbers = 1 + 118
        self.num_atom_type_input_features = (
            self.num_periodic_table_groups
            + self.num_periodic_table_periods
            + self.num_atomic_numbers
        )

        # Number of nearest neighbors residue to consider for finding atomized
        # side chain atoms.
        self.num_neighbors_for_atomized_side_chain = 16

        # Max distance for finding nearest ligand atom neighbors.
        self.max_distance_for_ligand_atoms = 10000.0

        # Projection of the atom type features to the embedding space.
        self.num_atom_type_output_features = 64

        # Number of angle features.
        self.num_angle_features = 4

        # Node features (protein-ligand subgraph edge features).
        # 1. RBF features for ligand atom to each backbone atom and virtual atom
        # 2. Atom type features for the ligand atom
        # 3. Angle features for the ligand atom and the backbone atoms
        self.num_node_input_features = (
            (self.num_backbone_atoms + self.num_virtual_atoms) * self.num_rbf
            + self.num_atom_type_output_features
            + self.num_angle_features
        )

        # Layers for the protein-ligand subgraph edge features.
        self.embed_atom_type_features = nn.Linear(
            self.num_atom_type_input_features,
            self.num_atom_type_output_features,
            bias=True,
        )
        self.node_embedding = nn.Linear(
            self.num_node_input_features, self.num_node_output_features, bias=True
        )
        self.node_norm = nn.LayerNorm(self.num_node_output_features)

        # Layers for the ligand subgraphs.
        self.ligand_subgraph_node_embedding = nn.Linear(
            self.num_atom_type_input_features, self.num_node_output_features, bias=False
        )
        self.ligand_subgraph_node_norm = nn.LayerNorm(self.num_node_output_features)
        self.ligand_subgraph_edge_embedding = nn.Linear(
            self.num_rbf, self.num_node_output_features, bias=False
        )
        self.ligand_subgraph_edge_norm = nn.LayerNorm(self.num_node_output_features)

        # Numeric encoding of the atom type (atomic number for the last 32 atoms
        # in the 37 atom representation).
        self.register_buffer(
            "side_chain_atom_types",
            torch.tensor(
                [
                    ELEMENT_NAME_TO_ATOMIC_NUMBER[
                        self.SIDE_CHAIN_ATOM_NAME_TO_ELEMENT_NAME[atom_name]
                    ]
                    for atom_name in self.SIDE_CHAIN_ATOM_NAMES
                ]
            ),
        )

        # Atomic number, period, and group for the periodic table.
        self.register_buffer(
            "periodic_table_groups",
            torch.tensor(
                [
                    0,
                    1,
                    18,
                    1,
                    2,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    1,
                    2,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    1,
                    2,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    1,
                    2,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                ],
                dtype=torch.long,
            ),
        )
        self.register_buffer(
            "periodic_table_periods",
            torch.tensor(
                [
                    0,
                    1,
                    1,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                ],
                dtype=torch.long,
            ),
        )

    def construct_X_side_chain(self, X, X_m, S):
        """
        Given the 3D coordinates of the atoms and the mask, construct the
        side chain atoms and their mask.

        Args:
            X (torch.Tensor): [B, L, self.TOKEN_ENCODING.n_atoms_per_token, 3] -
                3D coordinates of polymer atoms.
            X_m (torch.Tensor): [B, L, self.TOKEN_ENCODING.n_atoms_per_token] -
                Mask indicating which polymer atoms are valid.
            S (torch.Tensor): [B, L] - Sequence of the polymer residues.
        Returns:
            X_side_chain (torch.Tensor):
                [B, L, len(self.SIDE_CHAIN_ATOM_NAMES), 3] -
                3D coordinates of the side chain atoms for each residue.
            X_m_side_chain (torch.Tensor):
                [B, L, len(self.SIDE_CHAIN_ATOM_NAMES)] -
                Mask indicating which side chain atoms are valid.
        """
        X_side_chain, X_m_side_chain = self.construct_X_atoms(
            X, X_m, S, self.SIDE_CHAIN_ATOM_NAMES
        )

        return X_side_chain, X_m_side_chain

    def construct_angle_features(
        self, center_atom, atom_1, atom_2, ligand_subgraph_Y, eps=1e-8
    ):
        """
        Given the 3D coordinates of the center atom, the first atom, the second
        atom, and the ligand atoms, compute the angle features for the ligand
        atom with respect to the center atom and the two atoms.

        NOTE: M = self.num_context_atoms, the number of ligand atoms in each
            residue subgraph.

        Args:
            center_atom (torch.Tensor): [B, L, 3] - 3D coordinates of the center
                atom.
            atom_1 (torch.Tensor): [B, L, 3] - 3D coordinates of the first atom.
            atom_2 (torch.Tensor): [B, L, 3] - 3D coordinates of the second
                atom.
            ligand_subgraph_Y (torch.Tensor): [B, L, M, 3] - 3D coordinates of
                the M closest ligand atoms to each residue.
            eps (float): Small value added to distances that are zero.
        Returns:
            angle_features (torch.Tensor):
                [B, L, M, self.num_angle_features] - Angle features for the
                ligand atom with respect to the center atom and the two atoms.
                    cos_azimuthal_xy_angle [B, L, M] -
                        Cosine of the azimuthal angle in the local x-y plane.
                    sin_azimuthal_xy_angle [B, L, M] -
                        Sine of the azimuthal angle in the local x-y plane.
                    cos_inclination_angle [B, L, M] -
                        Cosine of the inclination angle (polar angle).
                    sin_inclination_angle [B, L, M] -
                        Sine of the inclination angle (polar angle).
        """
        # Compute the bond vectors.
        # bond_1 [B, L, 3] - vector from the center atom to the first atom.
        # bond_2 [B, L, 3] - vector from the center atom to the second atom.
        bond_1 = atom_1 - center_atom
        bond_2 = atom_2 - center_atom

        # Construct an orthonormal basis from the bond vectors.
        # The first vector in the basis, the normalized bond_1 vector.
        # basis_vector_1 [B, L, 3] - normalized bond_1 vector.
        basis_vector_1 = torch.nn.functional.normalize(bond_1, dim=-1)

        # Project bond_2 onto the first vector in the basis.
        # length_bond_2_proj [B, L, 1] - length of the projection of bond_2 onto
        # basis_vector_1.
        length_bond_2_proj = torch.einsum("bli, bli -> bl", basis_vector_1, bond_2)[
            ..., None
        ]

        # bond_2_orthogonal_component [B, L, 3] - component of bond_2 vector
        # orthogonal to basis_vector_1.
        bond_2_orthogonal_component = bond_2 - basis_vector_1 * length_bond_2_proj

        # basis_vector_2 [B, L, 3] - normalized bond_2 orthogonal component.
        basis_vector_2 = torch.nn.functional.normalize(
            bond_2_orthogonal_component, dim=-1
        )

        # basis_vector_3 [B, L, 3] - cross product of the first two basis
        # vectors.
        basis_vector_3 = torch.cross(basis_vector_1, basis_vector_2, dim=-1)

        # By construction, basis_vector_1, basis_vector_2, and basis_vector_3
        # form an orthonormal basis. We stack them together to form a
        # rotation matrix. This rotation matrix can be used to transform from
        # the local coordinate system defined by the bond vectors to the global
        # coordinate system:
        # v_global = R_residue @ v_local + center_atom
        # R_residue [B, L, 3, 3] - rotation matrix.
        R_residue = torch.cat(
            (
                basis_vector_1[:, :, :, None],
                basis_vector_2[:, :, :, None],
                basis_vector_3[:, :, :, None],
            ),
            dim=-1,
        )

        # Compute the local coordinates of the ligand atoms with respect to
        # the center atom.
        # ligand_subgraph_Y_local [B, L, M, 3] - local coordinates of the ligand
        # atoms with respect to the center atom.
        ligand_subgraph_Y_local = torch.einsum(
            "blqp, blyq -> blyp",
            R_residue,
            ligand_subgraph_Y - center_atom[:, :, None, :],
        )

        # Compute the length of the local vectors projected onto the local x-y
        # plane.
        # ligand_subgraph_Y_proj_xy_local_length [B, L, M] - length of the local
        # vectors projected on the local x-y plane.
        ligand_subgraph_Y_proj_xy_local_length = torch.sqrt(
            ligand_subgraph_Y_local[..., 0] ** 2
            + ligand_subgraph_Y_local[..., 1] ** 2
            + eps
        )

        # Compute the cosine and sine of the azimuthal angle.
        # cos_azimuthal_xy_angle [B, L, M] - cosine of the azimuthal angle in
        # the local x-y plane.
        # sin_azimuthal_xy_angle [B, L, M] - sine of the azimuthal angle in the
        # local x-y plane.
        cos_azimuthal_xy_angle = (
            ligand_subgraph_Y_local[..., 0] / ligand_subgraph_Y_proj_xy_local_length
        )
        sin_azimuthal_xy_angle = (
            ligand_subgraph_Y_local[..., 1] / ligand_subgraph_Y_proj_xy_local_length
        )

        # Compute the length of the local vectors.
        # ligand_subgraph_Y_local_length [B, L, M] - length of the local
        # vectors.
        ligand_subgraph_Y_local_length = (
            torch.norm(ligand_subgraph_Y_local, dim=-1) + eps
        )

        # Compute the cosine and sine of the inclination angle (polar angle).
        # cos_inclination_angle [B, L, M] - cosine of the inclination angle
        # (polar angle).
        # sin_inclination_angle [B, L, M] - sine of the inclination angle
        # (polar angle).
        cos_inclination_angle = (
            ligand_subgraph_Y_proj_xy_local_length / ligand_subgraph_Y_local_length
        )
        sin_inclination_angle = (
            ligand_subgraph_Y_local[..., 2] / ligand_subgraph_Y_local_length
        )

        # Concatenate the angle features.
        angle_features = torch.cat(
            (
                cos_azimuthal_xy_angle[..., None],
                sin_azimuthal_xy_angle[..., None],
                cos_inclination_angle[..., None],
                sin_inclination_angle[..., None],
            ),
            dim=-1,
        )

        return angle_features

    def gather_nearest_per_residue_atoms(
        self,
        per_residue_ligand_coords,
        per_residue_ligand_mask,
        per_residue_ligand_types,
        X_virtual_atoms,
        X_m_virtual_atoms,
        residue_mask,
    ):
        """
        Given the 3D coordinates of the ligand atoms, their mask, and the
        virtual atoms, gather the nearest ligand atoms to the virtual atoms for
        each residue.

        NOTE:
            num_ligand_atoms = N
                when called in self.gather_nearest_ligand_atoms.
            num_ligand_atoms = M
                when called in self.combine_ligand_and_side_chain_atoms.

        NOTE: M = self.num_context_atoms, the number of ligand atoms in each
            residue subgraph.

        Args:
            per_residue_ligand_coords (torch.Tensor): [B, L, num_ligand_atoms,
                3] - per residue 3D coordinates of the ligand atoms.
            per_residue_ligand_mask (torch.Tensor): [B, L, num_ligand_atoms] -
                per residue mask indicating which ligand atoms are valid.
            per_residue_ligand_types (torch.Tensor): [B, L, num_ligand_atoms] -
                per residue element types of the ligand atoms (atomic numbers).
            X_virtual_atoms (torch.Tensor): [B, L, num_virtual_atoms, 3] - 3D
                coordinates of the virtual atoms for each residue.
            X_m_virtual_atoms (torch.Tensor): [B, L, num_virtual_atoms] -
                mask indicating which virtual atoms are valid.
            residue_mask (torch.Tensor): [B, L] - mask indicating which residues
                are valid.
        Returns:
            ligand_subgraph_Y (torch.Tensor):
                [B, L, M, 3] - 3D coordinates of the nearest ligand atoms to the
                virtual atoms for each residue.
            ligand_subgraph_Y_m (torch.Tensor):
                [B, L, M] - mask indicating which nearest ligand atoms to the
                virtual atoms are valid.
            ligand_subgraph_Y_t (torch.Tensor):
                [B, L, M] - element types of the nearest ligand atoms to the
                virtual atoms for each residue.
        """
        B, L, num_ligand_atoms, _ = per_residue_ligand_coords.shape

        # X_virtual_atoms_collapsed [B, L, 3] - collapse the virtual atom
        # dimension.
        # NOTE: collapsing along this dimension is okay because the
        # self.construct_X_virtual_atoms function ensures that there is only one
        # virtual atom per residue.
        X_virtual_atoms_collapsed = torch.sum(
            X_virtual_atoms * X_m_virtual_atoms[:, :, :, None], dim=2
        )

        # ligand_to_virtual_atom_distances [B, L, num_ligand_atoms] -
        # distance between the ligand atoms and the virtual atoms.
        ligand_to_virtual_atom_distances = torch.sqrt(
            torch.sum(
                (X_virtual_atoms_collapsed[:, :, None, :] - per_residue_ligand_coords)
                ** 2,
                dim=-1,
            )
        )

        # residue_and_ligand_mask [B, L, num_ligand_atoms] - mask indicating
        # which residue-ligand atom pairs are valid.
        residue_and_ligand_mask = (
            residue_mask[:, :, None] * per_residue_ligand_mask
        ).bool()

        # ligand_to_virtual_atom_distances_adjusted [B, L, num_ligand_atoms] -
        # distances between the virtual atoms and the ligand atoms, with
        # invalid residue-ligand atom pairs adjusted to a maximum distance.
        ligand_to_virtual_atom_distances_adjusted = (
            ligand_to_virtual_atom_distances * residue_and_ligand_mask
            + (~residue_and_ligand_mask) * self.max_distance_for_ligand_atoms
        )

        # E_idx_ligand_subgraph [B, L, M] - indices of the closest ligand atoms
        # to the virtual atoms.
        _, E_idx_ligand_subgraph = torch.topk(
            ligand_to_virtual_atom_distances_adjusted,
            min(self.num_context_atoms, num_ligand_atoms),
            dim=-1,
            largest=False,
        )

        # Gather the ligand atom coordinates, mask, and types based on the
        # indices of the closest ligand atoms to the virtual atoms.
        # ligand_subgraph_Y [B, L, M, 3] - 3D coordinates of the nearest ligand
        # atoms to the virtual atoms for each residue.
        ligand_subgraph_Y = torch.gather(
            per_residue_ligand_coords,
            dim=2,
            index=E_idx_ligand_subgraph[:, :, :, None].expand(-1, -1, -1, 3),
        )

        # ligand_subgraph_Y_m [B, L, M] - mask indicating which nearest ligand
        # atoms to the virtual atoms are valid.
        ligand_subgraph_Y_m = torch.gather(
            per_residue_ligand_mask, dim=2, index=E_idx_ligand_subgraph
        )

        # ligand_subgraph_Y_t [B, L, M] - element types of the nearest ligand
        # atoms to the virtual atoms for each residue.
        ligand_subgraph_Y_t = torch.gather(
            per_residue_ligand_types, dim=2, index=E_idx_ligand_subgraph
        )

        return ligand_subgraph_Y, ligand_subgraph_Y_m, ligand_subgraph_Y_t

    def gather_nearest_ligand_atoms(
        self, Y, Y_m, Y_t, X_virtual_atoms, X_m_virtual_atoms, residue_mask
    ):
        """
        Given the 3D coordinates of the ligand atoms, their mask, and the
        virtual atoms, gather the nearest ligand atoms to the virtual atoms for
        each residue.

        NOTE: M = self.num_context_atoms, the number of ligand atoms in each
            residue subgraph.

        Args:
            Y (torch.Tensor): [B, N, 3] - 3D coordinates of the ligand atoms.
            Y_m (torch.Tensor): [B, N] - Mask indicating which ligand atoms
                are valid.
            Y_t (torch.Tensor): [B, N] - Element types of the ligand atoms
                (atomic numbers).
            X_virtual_atoms (torch.Tensor): [B, L, num_virtual_atoms, 3] - 3D
                coordinates of the virtual atoms for each residue.
            X_m_virtual_atoms (torch.Tensor): [B, L, num_virtual_atoms] -
                Mask indicating which virtual atoms are valid.
            residue_mask (torch.Tensor): [B, L] - Mask indicating which residues
                are valid.
        Returns:
            ligand_subgraph_Y (torch.Tensor):
                [B, L, M, 3] - 3D coordinates of the nearest ligand atoms to the
                virtual atoms for each residue.
            ligand_subgraph_Y_m (torch.Tensor):
                [B, L, M] - Mask indicating which nearest ligand atoms to the
                virtual atoms are valid.
            ligand_subgraph_Y_t (torch.Tensor):
                [B, L, M] - Element types of the nearest ligand atoms to the
                virtual atoms for each residue.
        """
        B, L, _, _ = X_virtual_atoms.shape

        # Gather the nearest ligand atoms to the virtual atoms for each
        # residue.
        # ligand_subgraph_Y [B, L, M, 3] - 3D coordinates of the nearest ligand
        # atoms to the virtual atoms for each residue.
        # ligand_subgraph_Y_m [B, L, M] - mask indicating which nearest ligand
        # atoms to the virtual atoms are valid.
        # ligand_subgraph_Y_t [B, L, M] - element types of the nearest ligand
        # atoms to the virtual atoms for each residue.
        ligand_subgraph_Y, ligand_subgraph_Y_m, ligand_subgraph_Y_t = (
            self.gather_nearest_per_residue_atoms(
                Y[:, None, :, :].expand(-1, L, -1, -1),
                Y_m[:, None, :].expand(-1, L, -1),
                Y_t[:, None, :].expand(-1, L, -1),
                X_virtual_atoms,
                X_m_virtual_atoms,
                residue_mask,
            )
        )

        return ligand_subgraph_Y, ligand_subgraph_Y_m, ligand_subgraph_Y_t

    def gather_nearest_atomized_side_chain_atoms(
        self, X, X_m, S, E_idx, hide_side_chain_mask
    ):
        """
        Given the 3D coordinates of the polymer atoms, their mask, the indices
        of the top K nearest neighbors for each residue, and a mask indicating
        which side chains are hidden, gather the nearest neighbors side chain
        atoms for each residue. This is used to construct the atomized side
        chain atoms for each residue.

        Args:
            X (torch.Tensor): [B, L, num_atoms, 3] - 3D coordinates of the
                polymer atoms.
            X_m (torch.Tensor): [B, L, num_atoms] - Mask indicating which
                polymer atoms are valid.
            S (torch.Tensor): [B, L] - Sequence of the polymer residues.
            E_idx (torch.Tensor): [B, L, K] - Indices of the top K nearest
                neighbors for each residue.
            hide_side_chain_mask (torch.Tensor): [B, L] - Mask indicating which
                residue side chains are hidden and which are revealed. True
                indicates that the side chain is hidden and False indicates
                that the side chain is revealed.
        Returns:
            ligand_subgraph_R (torch.Tensor): [B, L,
                num_neighbors_for_atomized_side_chain * num_side_chain_atoms, 3]
                - 3D coordinates of the nearest neighbors side chain atoms for
                each residue.
            ligand_subgraph_R_m (torch.Tensor): [B, L,
                num_neighbors_for_atomized_side_chain * num_side_chain_atoms] -
                mask indicating which nearest neighbors side chain atoms are
                valid.
            ligand_subgraph_R_t (torch.Tensor): [B, L,
                num_neighbors_for_atomized_side_chain * num_side_chain_atoms] -
                Element types of the nearest neighbors side chain atoms for each
                residue.
        """
        B, L, _, _ = X.shape

        # X_side_chain [B, L, len(self.SIDE_CHAIN_ATOM_NAMES), 3] - 3D
        # coordinates of the side chain atoms for each residue.
        # X_m_side_chain [B, L, len(self.SIDE_CHAIN_ATOM_NAMES)] - mask
        # indicating which side chain atoms are valid.
        # NOTE: the side chain atoms exclude the CB atom, since in other
        # places, we use the virtual CB atom.
        X_side_chain, X_m_side_chain = self.construct_X_side_chain(X, X_m, S)

        # E_idx_sub [B, L, self.num_neighbors_for_atomized_side_chain] -
        # Indices of the nearest neighbors to consider for atomized side chain
        # atoms.
        E_idx_sub = E_idx[:, :, : self.num_neighbors_for_atomized_side_chain]

        # ligand_subgraph_R [B, L, self.num_neighbors_for_atomized_side_chain *
        # self.num_side_chain_atoms, 3] - 3D coordinates of the nearest
        # neighbors side chain atoms for each residue.
        ligand_subgraph_R = gather_nodes(
            X_side_chain.view(B, L, self.num_side_chain_atoms * 3), E_idx_sub
        ).view(
            B,
            L,
            self.num_neighbors_for_atomized_side_chain * self.num_side_chain_atoms,
            3,
        )

        # ligand_subgraph_R_m [B, L, self.num_neighbors_for_atomized_side_chain
        # * self.num_side_chain_atoms] - mask indicating which nearest
        # neighbors side chain atoms are valid.
        ligand_subgraph_R_m = gather_nodes(
            X_m_side_chain & (~(hide_side_chain_mask[:, :, None].bool())), E_idx_sub
        ).view(
            B, L, self.num_neighbors_for_atomized_side_chain * self.num_side_chain_atoms
        )

        # ligand_subgraph_R_t [B, L, self.num_neighbors_for_atomized_side_chain
        # * self.num_side_chain_atoms] - element types of the nearest
        # neighbors side chain atoms for each residue.
        ligand_subgraph_R_t = (
            self.side_chain_atom_types[None, None, None, :]
            .expand(B, L, self.num_neighbors_for_atomized_side_chain, -1)
            .reshape(
                B,
                L,
                self.num_neighbors_for_atomized_side_chain * self.num_side_chain_atoms,
            )
        )

        return ligand_subgraph_R, ligand_subgraph_R_m, ligand_subgraph_R_t

    def combine_ligand_and_atomized_side_chain_atoms(
        self,
        ligand_subgraph_Y,
        ligand_subgraph_Y_m,
        ligand_subgraph_Y_t,
        ligand_subgraph_R,
        ligand_subgraph_R_m,
        ligand_subgraph_R_t,
        X_virtual_atoms,
        X_m_virtual_atoms,
        residue_mask,
    ):
        """
        Given the 3D coordinates of the nearest ligand atoms to the virtual
        atoms, their mask, the element types of the nearest ligand atoms, the
        3D coordinates of the nearest neighbors side chain atoms, their mask,
        and the element types of the nearest neighbors side chain atoms,
        combine the ligand and side chain atoms into a single tensor.

        NOTE: M = self.num_context_atoms, the number of ligand atoms in each
            residue subgraph.

        Args:
            ligand_subgraph_Y (torch.Tensor): [B, L, M, 3] - 3D coordinates of
                the nearest ligand atoms to the virtual atoms for each residue.
            ligand_subgraph_Y_m (torch.Tensor): [B, L, M] - mask indicating
                which nearest ligand atoms to the virtual atoms are valid.
            ligand_subgraph_Y_t (torch.Tensor): [B, L, M] - element types of the
                nearest ligand atoms to the virtual atoms for each residue.
            ligand_subgraph_R (torch.Tensor): [B, L,
                self.num_neighbors_for_atomized_side_chain *
                self.num_side_chain_atoms, 3] - 3D coordinates of the nearest
                neighbors side chain atoms for each residue.
            ligand_subgraph_R_m (torch.Tensor): [B, L,
                self.num_neighbors_for_atomized_side_chain *
                self.num_side_chain_atoms] - mask indicating which nearest
                neighbors side chain atoms are valid.
            ligand_subgraph_R_t (torch.Tensor): [B, L,
                self.num_neighbors_for_atomized_side_chain *
                self.num_side_chain_atoms] - element types of the nearest
                neighbors side chain atoms for each residue.
            X_virtual_atoms (torch.Tensor): [B, L, num_virtual_atoms, 3] - 3D
                coordinates of the virtual atoms for each residue.
            X_m_virtual_atoms (torch.Tensor): [B, L, num_virtual_atoms] - mask
                indicating which virtual atoms are valid.
            residue_mask (torch.Tensor): [B, L] - mask indicating which
                residues are valid.
        Returns:
            ligand_subgraph_Y_and_R (torch.Tensor): [B, L, M, 3] - 3D
                coordinates of the nearest ligand or side chain atoms to the
                virtual atoms for each residue.
            ligand_subgraph_Y_m_and_R_m (torch.Tensor): [B, L, M] - mask
                indicating which nearest ligand or side chain atoms to the
                virtual atoms are valid.
            ligand_subgraph_Y_t_and_R_t (torch.Tensor): [B, L, M] - element
                types of the nearest ligand or side chain atoms to the virtual
                atoms for each residue.
        """
        # Concatenate the ligand and side chain atom coordinates, masks, and
        # types.
        # ligand_subgraph_Y_cat_R [B, L, M +
        # self.num_neighbors_for_atomized_side_chain *
        # self.num_side_chain_atoms, 3] - 3D coordinates of the nearest ligand
        # atoms and side chain atoms for each residue.
        ligand_subgraph_Y_cat_R = torch.cat(
            (ligand_subgraph_Y, ligand_subgraph_R), dim=2
        )

        # ligand_subgraph_Y_m_cat_R_m [B, L, M +
        # self.num_neighbors_for_atomized_side_chain *
        # self.num_side_chain_atoms] - mask indicating which nearest ligand
        # atoms and side chain atoms are valid.
        ligand_subgraph_Y_m_cat_R_m = torch.cat(
            (ligand_subgraph_Y_m, ligand_subgraph_R_m), dim=2
        )

        # ligand_subgraph_Y_t_cat_R_t [B, L, M +
        # self.num_neighbors_for_atomized_side_chain *
        # self.num_side_chain_atoms] - element types of the nearest ligand
        # atoms and side chain atoms for each residue.
        ligand_subgraph_Y_t_cat_R_t = torch.cat(
            (ligand_subgraph_Y_t, ligand_subgraph_R_t), dim=2
        )

        # Gather the nearest atoms to the virtual atoms from the combined
        # ligand and side chain atoms.
        # ligand_subgraph_Y_and_R [B, L, M, 3] - 3D coordinates of the nearest
        # ligand or side chain atoms to the virtual atoms for each residue.
        # ligand_subgraph_Y_m_and_R_m [B, L, M] - mask indicating which nearest
        # ligand or side chain atoms to the virtual atoms are valid.
        # ligand_subgraph_Y_t_and_R_t [B, L, M] - element types of the nearest
        # ligand or side chain atoms to the virtual atoms for each residue.
        (
            ligand_subgraph_Y_and_R,
            ligand_subgraph_Y_m_and_R_m,
            ligand_subgraph_Y_t_and_R_t,
        ) = self.gather_nearest_per_residue_atoms(
            ligand_subgraph_Y_cat_R,
            ligand_subgraph_Y_m_cat_R_m,
            ligand_subgraph_Y_t_cat_R_t,
            X_virtual_atoms,
            X_m_virtual_atoms,
            residue_mask,
        )

        return (
            ligand_subgraph_Y_and_R,
            ligand_subgraph_Y_m_and_R_m,
            ligand_subgraph_Y_t_and_R_t,
        )

    def featurize_ligand_atom_type_information(self, ligand_subgraph_Y_t):
        """
        Given the element types of the ligand atoms, compute the periodic table
        group, period, and atomic number for each ligand atom.

        NOTE: M = self.num_context_atoms, the number of ligand atoms in each
            residue subgraph.

        Args:
            ligand_subgraph_Y_t (torch.Tensor): [B, L, M] - element types of the
                ligand atoms.
        Returns:
            ligand_subgraph_Y_t_concat_one_hot (torch.Tensor):
                [B, L, M, self.num_atomic_numbers +
                self.num_periodic_table_groups +
                self.num_periodic_table_periods] - atomic number,
                periodic group, and periodic period of the ligand atoms, as
                concatenated one-hot encodings.
        """
        # Get the periodic table group and period for the ligand atoms.
        ligand_subgraph_Y_t = ligand_subgraph_Y_t.long()

        # ligand_subgraph_Y_t_g [B, L, M] - periodic group of the ligand atoms.
        # 18 groups and 1 null group.
        ligand_subgraph_Y_t_g = self.periodic_table_groups[ligand_subgraph_Y_t]

        # ligand_subgraph_Y_t_p [B, L, M] - periodic period of the ligand atoms.
        # 7 periods and 1 null period.
        ligand_subgraph_Y_t_p = self.periodic_table_periods[ligand_subgraph_Y_t]

        # Turn the ligand atom types into one-hot encodings.
        # ligand_subgraph_Y_t_g_one_hot [B, L, M,
        # self.num_periodic_table_groups] - periodic group of the ligand atoms.
        ligand_subgraph_Y_t_g_one_hot = torch.nn.functional.one_hot(
            ligand_subgraph_Y_t_g, self.num_periodic_table_groups
        )

        # ligand_subgraph_Y_t_p_one_hot [B, L, M,
        # self.num_periodic_table_periods] - periodic period of the ligand
        # atoms.
        ligand_subgraph_Y_t_p_one_hot = torch.nn.functional.one_hot(
            ligand_subgraph_Y_t_p, self.num_periodic_table_periods
        )

        # ligand_subgraph_Y_t_one_hot [B, L, M, self.num_atomic_numbers] -
        # atomic number of the ligand atoms.
        ligand_subgraph_Y_t_one_hot = torch.nn.functional.one_hot(
            ligand_subgraph_Y_t, self.num_atomic_numbers
        )

        # Concatenate the one-hot encodings.
        # ligand_subgraph_Y_t_concat_one_hot [B, L, M, self.num_atomic_numbers +
        # self.num_periodic_table_groups + self.num_periodic_table_periods]
        # - atomic number, periodic group, and periodic period of the ligand
        # atoms.
        ligand_subgraph_Y_t_concat_one_hot = torch.cat(
            (
                ligand_subgraph_Y_t_one_hot,
                ligand_subgraph_Y_t_g_one_hot,
                ligand_subgraph_Y_t_p_one_hot,
            ),
            dim=-1,
        )

        return ligand_subgraph_Y_t_concat_one_hot

    def featurize_protein_to_ligand_subgraph_edges(
        self,
        ligand_subgraph_Y_t_concat_one_hot,
        X_backbone,
        X_m_backbone,
        X_virtual_atoms,
        X_m_virtual_atoms,
        ligand_subgraph_Y,
        ligand_subgraph_Y_m,
        eps=1e-6,
    ):
        """
        Given the 3D coordinates of the backbone atoms, the virtual atoms,
        the ligand atoms, and the mask indicating which atoms are valid,
        compute the protein to ligand subgraph edges.

        NOTE: M = self.num_context_atoms, the number of ligand atoms in each
            residue subgraph.

        Args:
            ligand_subgraph_Y_t_concat_one_hot (torch.Tensor):
                [B, L, M, self.num_atomic_numbers +
                self.num_periodic_table_groups +
                self.num_periodic_table_periods] - atomic number,
                periodic group, and periodic period of the ligand atoms.
            X_backbone (torch.Tensor): [B, L, self.num_backbone_atoms, 3] - 3D
                coordinates of the backbone atoms for each residue.
            X_m_backbone (torch.Tensor): [B, L, self.num_backbone_atoms] - mask
                indicating which backbone atoms are valid.
            X_virtual_atoms (torch.Tensor): [B, L, self.num_virtual_atoms, 3] -
                3D coordinates of the virtual atoms for each residue.
            X_m_virtual_atoms (torch.Tensor): [B, L, self.num_virtual_atoms] -
                mask indicating which virtual atoms are valid.
            ligand_subgraph_Y (torch.Tensor): [B, L, M, 3] - 3D coordinates of
                the ligand atoms.
            ligand_subgraph_Y_m (torch.Tensor): [B, L, M] - mask indicating
                which ligand atoms are valid.
            eps (float): Small value added to distances that are zero.
        Returns:
            E_protein_to_ligand (torch.Tensor):
                [B, L, M, self.num_node_output_features] -
                protein to ligand subgraph edges; can also be considered node
                features of the protein residues (although they are not used as
                such).
        """
        B, L, M, _ = ligand_subgraph_Y_t_concat_one_hot.shape

        # Embed the ligand atom type information.
        # ligand_subgraph_Y_t_concat_one_hot_embed
        # [B, L, M, self.num_atom_type_output_features] - embedded atomic
        # number, periodic group, and periodic period of the ligand atoms.
        ligand_subgraph_Y_t_concat_one_hot_embed = self.embed_atom_type_features(
            ligand_subgraph_Y_t_concat_one_hot.float()
        )

        # Concatenate the backbone and virtual atom coordinates and masks.
        # X_backbone_and_virtual_atoms [B, L, num_backbone_atoms +
        #     num_virtual_atoms, 3] - 3D coordinates of the backbone and virtual
        #     atoms for each residue.
        # X_m_backbone_and_virtual_atoms [B, L, num_backbone_atoms +
        #     num_virtual_atoms] - mask indicating which backbone and virtual
        #     atoms are valid.
        X_backbone_and_virtual_atoms = torch.cat((X_backbone, X_virtual_atoms), dim=2)
        X_m_backbone_and_virtual_atoms = torch.cat(
            (X_m_backbone, X_m_virtual_atoms), dim=2
        )

        # Compute the distance of each ligand atom in each residue subgraph to
        # to each of the backbone and virtual atoms.
        # D_ligand_to_backbone_or_virtual [B, L, M, self.num_backbone_atoms +
        #     self.num_virtual_atoms] - distances of each ligand atom in each
        #     residue subgraph to each of the backbone and virtual atoms.
        D_ligand_to_backbone_or_virtual = torch.sqrt(
            torch.sum(
                (
                    ligand_subgraph_Y[:, :, :, None, :]
                    - X_backbone_and_virtual_atoms[:, :, None, :, :]
                )
                ** 2,
                dim=-1,
            )
            + eps
        )

        # RBF_ligand_to_backbone_or_virtual [B, L, M, self.num_backbone_atoms +
        # self.num_virtual_atoms, num_rbf] - radial basis function embedding
        # of the distances of each ligand atom in each residue subgraph to
        # each of the backbone and virtual atoms.
        RBF_ligand_to_backbone_or_virtual = self.compute_rbf_embedding_from_distances(
            D_ligand_to_backbone_or_virtual
        )

        # Mask the radial basis function embedding with the ligand atom mask and
        # the backbone and virtual atom mask.
        RBF_ligand_to_backbone_or_virtual = (
            RBF_ligand_to_backbone_or_virtual
            * ligand_subgraph_Y_m[:, :, :, None, None]
            * X_m_backbone_and_virtual_atoms[:, :, None, :, None]
        )

        # Reshape the radial basis function embedding.
        # RBF_ligand_to_backbone_or_virtual [B, L, M,
        #     (self.num_backbone_atoms + self.num_virtual_atoms) * num_rbf] -
        RBF_ligand_to_backbone_or_virtual = RBF_ligand_to_backbone_or_virtual.view(
            B, L, M, (self.num_backbone_atoms + self.num_virtual_atoms) * self.num_rbf
        )

        # Construct the angle features for the ligand atoms with respect to the
        # backbone atoms.
        # angle_features [B, L, M, 4] - angle features for the ligand atoms
        # with respect to the backbone atoms.
        angle_features = self.construct_angle_features(
            X_backbone[:, :, self.BACKBONE_ATOM_NAMES.index("CA"), :],
            X_backbone[:, :, self.BACKBONE_ATOM_NAMES.index("N"), :],
            X_backbone[:, :, self.BACKBONE_ATOM_NAMES.index("C"), :],
            ligand_subgraph_Y,
        )

        # E_protein_to_ligand [B, L, M,
        #     (self.num_backbone_atoms + self.num_virtual_atoms) * num_rbf +
        #     self.num_atom_type_output_features + 4] - concatenated
        # radial basis function embedding of the distances of each ligand atom
        # in each residue subgraph to each of the backbone and virtual atoms,
        # periodic group, periodic period, and atomic number of the ligand
        # atoms, and the angle features for the ligand atoms with respect to
        # the backbone atoms.
        E_protein_to_ligand = torch.cat(
            (
                RBF_ligand_to_backbone_or_virtual,
                ligand_subgraph_Y_t_concat_one_hot_embed,
                angle_features,
            ),
            dim=-1,
        )

        # While these are protein-ligand subgraph edges, they can also be
        # considered node features of the protein residues.
        # E_protein_to_ligand [B, L, M, self.num_node_output_features] - protein
        # to ligand subgraph edges.
        E_protein_to_ligand = self.node_embedding(E_protein_to_ligand)
        E_protein_to_ligand = self.node_norm(E_protein_to_ligand)

        return E_protein_to_ligand

    def featurize_ligand_subgraph_nodes(self, ligand_subgraph_Y_t_concat_one_hot):
        """
        Given the atomic number, periodic group, and periodic period of the
        ligand atoms, compute the ligand subgraph node features.

        NOTE: M = self.num_context_atoms, the number of ligand atoms in each
            residue subgraph.

        Args:
            ligand_subgraph_Y_t_concat_one_hot (torch.Tensor):
                [B, L, M, self.num_atomic_numbers +
                self.num_periodic_table_groups +
                self.num_periodic_table_periods] - atomic number,
                periodic group, and periodic period of the ligand atoms.
        Returns:
            ligand_subgraph_nodes (torch.Tensor): [B, L, M,
                self.num_node_output_features] - ligand atom type information,
                embedded as node features.
        """
        # Embed and normalize the ligand atom type information.
        # ligand_subgraph_nodes [B, L, M, self.num_atom_type_output_features] -
        # embedded atomic number, periodic group, and periodic period of the
        # ligand atoms.
        ligand_subgraph_nodes = self.ligand_subgraph_node_embedding(
            ligand_subgraph_Y_t_concat_one_hot.float()
        )
        ligand_subgraph_nodes = self.ligand_subgraph_node_norm(ligand_subgraph_nodes)

        return ligand_subgraph_nodes

    def featurize_ligand_subgraph_edges(
        self, ligand_subgraph_Y, ligand_subgraph_Y_m, eps=1e-6
    ):
        """
        Given the 3D coordinates of the ligand atoms and the mask indicating
        which atoms are valid, compute the ligand subgraph edges.

        NOTE: M = self.num_context_atoms, the number of ligand atoms in each
            residue subgraph.

        Args:
            ligand_subgraph_Y (torch.Tensor): [B, L, M, 3] - 3D coordinates of
                the ligand atoms.
            ligand_subgraph_Y_m (torch.Tensor): [B, L, M] - mask indicating
                which ligand atoms are valid.
            eps (float): Small value added to distances that are zero.
        Returns:
            ligand_subgraph_edges (torch.Tensor):
                [B, L, M, M, self.num_edge_output_features] - embedded and
                normalized radial basis function embedding of the distances
                between the ligand atoms in each residue subgraph.
        """
        # D_ligand_to_ligand [B, L, M, M] - distances between the ligand atoms
        # in each residue subgraph.
        D_ligand_to_ligand = torch.sqrt(
            torch.sum(
                (
                    ligand_subgraph_Y[:, :, :, None, :]
                    - ligand_subgraph_Y[:, :, None, :, :]
                )
                ** 2,
                dim=-1,
            )
            + eps
        )

        # RBF_ligand_to_ligand [B, L, M, M, num_rbf] - radial basis function
        # embedding of the distances between the ligand atoms in each residue
        # subgraph.
        RBF_ligand_to_ligand = self.compute_rbf_embedding_from_distances(
            D_ligand_to_ligand
        )

        # Mask the radial basis function embedding with the ligand atom mask.
        RBF_ligand_to_ligand = (
            RBF_ligand_to_ligand
            * ligand_subgraph_Y_m[:, :, :, None, None]
            * ligand_subgraph_Y_m[:, :, None, :, None]
        )

        # ligand_subgraph_edges [B, L, M, M, self.num_edge_output_features] -
        # embedded and normalized radial basis function embedding of the
        # distances between the ligand atoms in each residue subgraph.
        ligand_subgraph_edges = self.ligand_subgraph_edge_embedding(
            RBF_ligand_to_ligand
        )
        ligand_subgraph_edges = self.ligand_subgraph_edge_norm(ligand_subgraph_edges)

        return ligand_subgraph_edges

    def featurize_nodes(self, input_features, edge_features):
        """
        Given the input features and edge features, compute the node features
        for the ligand atoms and the protein to ligand subgraph edges.

        NOTE: N = the total number of ligand atoms.
              M = self.num_context_atoms, the number of ligand atoms in each
                    residue subgraph.

        Args:
            input_features (dict): Dictionary containing the input features.
                - X (torch.Tensor): [B, L, num_atoms, 3] - 3D coordinates of the
                    polymer atoms.
                - X_m (torch.Tensor): [B, L, num_atoms] - mask indicating which
                    polymer atoms are valid.
                - hide_side_chain_mask (torch.Tensor): [B, L] - mask
                    indicating which residue side chains are hidden and which
                    are revealed. True indicates that the side chain is hidden
                    and False indicates that the side chain is revealed.
                - Y (torch.Tensor): [B, N, 3] - 3D coordinates of the ligand
                    atoms.
                - Y_m (torch.Tensor): [B, N] - mask indicating which ligand
                    atoms are valid.
                - Y_t (torch.Tensor): [B, N] - element types of the ligand
                    atoms.
                - X_virtual_atoms (torch.Tensor): [B, L, num_virtual_atoms, 3] -
                    3D coordinates of the virtual atoms for each residue.
                - X_m_virtual_atoms (torch.Tensor): [B, L, num_virtual_atoms] -
                    mask indicating which virtual atoms are valid.
                - residue_mask (torch.Tensor): [B, L] - mask indicating which
                    residues are valid.
                - X_backbone (torch.Tensor): [B, L, num_backbone_atoms, 3] -
                    3D coordinates of the backbone atoms for each residue.
                - X_m_backbone (torch.Tensor): [B, L, num_backbone_atoms] -
                    mask indicating which backbone atoms are valid.
                - atomize_side_chains (bool): Whether to atomize the side chains
                    of the residues. If True, the side chains of the residues
                    not specified in the hide side chain mask will be
                    atomized and added as ligand atoms.
            edge_features (dict): Dictionary containing the edge features.
                - E_idx (torch.Tensor): [B, L, K] - indices of the top K
                    nearest neighbors for each residue.
        Returns:
            node_features (dict): Dictionary containing the node features.
                - E_protein_to_ligand (torch.Tensor):
                    [B, L, M, self.num_node_output_features] - protein to
                    ligand subgraph edges; can also be considered node features
                    of the protein residues (although they are not used as
                    such).
                - ligand_subgraph_nodes (torch.Tensor):
                    [B, L, M, self.num_node_output_features] - ligand atom type
                    information, embedded as node features.
                - ligand_subgraph_edges (torch.Tensor):
                    [B, L, M, M, self.num_edge_output_features] - embedded and
                    normalized radial basis function embedding of the distances
                    between the ligand atoms in each residue subgraph.
        """
        # Check that the needed input features are present.
        if "X" not in input_features:
            raise ValueError("Input features must contain 'X' key.")
        if "X_m" not in input_features:
            raise ValueError("Input features must contain 'X_m' key.")
        if "hide_side_chain_mask" not in input_features:
            raise ValueError("Input features must contain 'hide_side_chain_mask' key.")
        if "Y" not in input_features:
            raise ValueError("Input features must contain 'Y' key.")
        if "Y_m" not in input_features:
            raise ValueError("Input features must contain 'Y_m' key.")
        if "Y_t" not in input_features:
            raise ValueError("Input features must contain 'Y_t' key.")
        if "X_virtual_atoms" not in input_features:
            raise ValueError("Input features must contain 'X_virtual_atoms' key.")
        if "X_m_virtual_atoms" not in input_features:
            raise ValueError("Input features must contain 'X_m_virtual_atoms' key.")
        if "residue_mask" not in input_features:
            raise ValueError("Input features must contain 'residue_mask' key.")
        if "X_backbone" not in input_features:
            raise ValueError("Input features must contain 'X_backbone' key.")
        if "X_m_backbone" not in input_features:
            raise ValueError("Input features must contain 'X_m_backbone' key.")
        if "atomize_side_chains" not in input_features:
            raise ValueError("Input features must contain 'atomize_side_chains' key.")

        # Check that the needed edge features are present.
        if "E_idx" not in edge_features:
            raise ValueError("Edge features must contain 'E_idx' key.")

        atomize_side_chains = input_features["atomize_side_chains"]

        # Gather the coordinates, mask and types of the ligand atoms closest
        # to the virtual atoms.
        # ligand_subgraph_Y [B, L, M, 3] - 3D coordinates of the ligand atoms
        # closest to the virtual atoms for each residue.
        # ligand_subgraph_Y_m [B, L, M] - mask indicating which ligand
        # atoms closest to the virtual atoms are valid.
        # ligand_subgraph_Y_t [B, L, M] - element types of the
        # ligand atoms closest to the virtual atoms for each residue.
        ligand_subgraph_Y, ligand_subgraph_Y_m, ligand_subgraph_Y_t = (
            self.gather_nearest_ligand_atoms(
                input_features["Y"],
                input_features["Y_m"],
                input_features["Y_t"],
                input_features["X_virtual_atoms"],
                input_features["X_m_virtual_atoms"],
                input_features["residue_mask"],
            )
        )

        # Add atomized side chain atoms as ligand atoms.
        if atomize_side_chains:
            # Gather the atomized side chain atoms coordinates, mask, and types.
            # ligand_subgraph_R [B, L, num_neighbors_for_atomized_side_chain *
            # num_side_chain_atoms, 3] - 3D coordinates of the nearest neighbors
            # side chain atoms for each residue.
            # ligand_subgraph_R_m [B, L, num_neighbors_for_atomized_side_chain *
            # num_side_chain_atoms] - mask indicating which nearest neighbors
            # side chain atoms are valid.
            # ligand_subgraph_R_t [B, L, num_neighbors_for_atomized_side_chain
            # * num_side_chain_atoms] - element types of the nearest neighbors
            # side chain atoms for each residue.
            ligand_subgraph_R, ligand_subgraph_R_m, ligand_subgraph_R_t = (
                self.gather_nearest_atomized_side_chain_atoms(
                    input_features["X"],
                    input_features["X_m"],
                    input_features["S"],
                    edge_features["E_idx"],
                    input_features["hide_side_chain_mask"],
                )
            )

            # Get the self.num_context_atoms closest ligand or atomized side
            # chain atoms to the virtual atoms; overwriting the original
            # ligand_subgraph_Y, ligand_subgraph_Y_m, and ligand_subgraph_Y_t.
            ligand_subgraph_Y, ligand_subgraph_Y_m, ligand_subgraph_Y_t = (
                self.combine_ligand_and_atomized_side_chain_atoms(
                    ligand_subgraph_Y,
                    ligand_subgraph_Y_m,
                    ligand_subgraph_Y_t,
                    ligand_subgraph_R,
                    ligand_subgraph_R_m,
                    ligand_subgraph_R_t,
                    input_features["X_virtual_atoms"],
                    input_features["X_m_virtual_atoms"],
                    input_features["residue_mask"],
                )
            )

        # Save the ligand subgraph coordinates, mask, and types in the input
        # features.
        input_features["ligand_subgraph_Y"] = ligand_subgraph_Y
        input_features["ligand_subgraph_Y_m"] = ligand_subgraph_Y_m
        input_features["ligand_subgraph_Y_t"] = ligand_subgraph_Y_t

        # Get the concatenated one hot type information for the ligand atoms.
        # ligand_subgraph_Y_t_concat_one_hot [B, L, M, self.num_atomic_numbers +
        # self.num_periodic_table_groups + self.num_periodic_table_periods] -
        # atomic number, periodic group, and periodic period of the ligand
        # atoms.
        ligand_subgraph_Y_t_concat_one_hot = (
            self.featurize_ligand_atom_type_information(
                input_features["ligand_subgraph_Y_t"]
            )
        )

        # Get the protein to ligand subgraph edges.
        # E_protein_to_ligand [B, L, M, self.num_node_output_features] - protein
        # to ligand subgraph edges.
        E_protein_to_ligand = self.featurize_protein_to_ligand_subgraph_edges(
            ligand_subgraph_Y_t_concat_one_hot,
            input_features["X_backbone"],
            input_features["X_m_backbone"],
            input_features["X_virtual_atoms"],
            input_features["X_m_virtual_atoms"],
            input_features["ligand_subgraph_Y"],
            input_features["ligand_subgraph_Y_m"],
        )

        # ligand_subgraph_nodes [B, L, M, self.num_node_output_features] -
        # ligand atom type information, embedded as node features.
        ligand_subgraph_nodes = self.featurize_ligand_subgraph_nodes(
            ligand_subgraph_Y_t_concat_one_hot
        )

        # ligand_subgraph_edges [B, L, M, M, self.num_edge_output_features] -
        # embedded and normalized radial basis function embedding of the
        # distances between the ligand atoms in each residue subgraph.
        ligand_subgraph_edges = self.featurize_ligand_subgraph_edges(
            input_features["ligand_subgraph_Y"], input_features["ligand_subgraph_Y_m"]
        )

        # Gather the node features.
        node_features = {
            "E_protein_to_ligand": E_protein_to_ligand,
            "ligand_subgraph_nodes": ligand_subgraph_nodes,
            "ligand_subgraph_edges": ligand_subgraph_edges,
        }

        return node_features

    def noise_structure(self, input_features):
        """
        Given input features containing 3D coordinates of atoms, add Gaussian
        noise to the coordinates.

        Args:
            input_features (dict): Dictionary containing the input features.
                - X (torch.Tensor): [B, L, num_atoms, 3] - 3D coordinates of
                    polymer atoms.
                - Y (torch.Tensor): [B, N, 3] - 3D coordinates of the ligand
                    atoms.
                - structure_noise (float): Standard deviation of the
                    Gaussian noise to add to the input coordinates, in
                    Angstroms.
        Side Effects:
            input_features["X"] (torch.Tensor): [B, L, num_atoms, 3] - 3D
                coordinates of atoms with added Gaussian noise.
            input_features["Y"] (torch.Tensor): [B, N, 3] - 3D coordinates
                of the ligand atoms with added Gaussian noise.
            input_features["X_pre_noise"] (torch.Tensor): [B, L, num_atoms, 3] -
                3D coordinates of polymer atoms before adding noise.
            input_features["Y_pre_noise"] (torch.Tensor): [B, N, 3] -
                3D coordinates of the ligand atoms before adding noise.
        """
        if "X" not in input_features:
            raise ValueError("Input features must contain 'X' key.")
        if "Y" not in input_features:
            raise ValueError("Input features must contain 'Y' key.")
        if "structure_noise" not in input_features:
            raise ValueError("Input features must contain 'structure_noise' key.")

        structure_noise = input_features["structure_noise"]

        # If the noise is non-zero, add Gaussian noise to the input
        # coordinates.
        if structure_noise > 0:
            # Copy the original coordinates before adding noise.
            input_features["X_pre_noise"] = input_features["X"].clone()
            input_features["Y_pre_noise"] = input_features["Y"].clone()

            # Add Gaussian noise to the input coordinates.
            input_features["X"] = input_features[
                "X"
            ] + structure_noise * torch.randn_like(input_features["X"])
            input_features["Y"] = input_features[
                "Y"
            ] + structure_noise * torch.randn_like(input_features["Y"])
        else:
            input_features["X_pre_noise"] = input_features["X"].clone()
            input_features["Y_pre_noise"] = input_features["Y"].clone()
