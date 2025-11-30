import torch
from mpnn.transforms.feature_aggregation.token_encodings import (
    legacy_token_order,
    token_order,
)


def load_legacy_weights(model: torch.nn.Module, weights_path: str) -> None:
    """
    Load a legacy MPNN checkpoint from 'weights_path' into 'model' (the
    refactored MPNN implementation).

    This performs several transformations:
      - Copies certain non-learned registries (e.g., periodic table info) from
        the new model into the checkpoint state dict (to match the new code).
      - Renames legacy parameter/buffer names into the new module naming scheme.
      - Fixes a 120->119 atom-type embedding weight size mismatch by dropping
        the unused legacy atom type.
      - Reorders pairwise backbone distance embedding weights to match the new
        atom-pair ordering.
      - Reorders token (AA) embeddings/projections weights from the legacy order
        (alphabetical 1-letter) to the new order (alphabetical 3-letter).
    """
    # Load legacy checkpoint state dict.
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    checkpoint_state_dict = checkpoint["model_state_dict"]

    # Values to copy directly from the *current* model into the state dict.
    # These are effectively "configuration" tensors/registries, not learned
    # weights we want to preserve from the legacy model.
    values_to_copy = [
        "graph_featurization_module.side_chain_atom_types",
        "graph_featurization_module.periodic_table_groups",
        "graph_featurization_module.periodic_table_periods",
    ]
    # Copy over some hard-coded registers/values.
    for value_name in values_to_copy:
        # Walk the attribute chain.
        attr_list = value_name.split(".")
        sub_module = model
        while len(attr_list) > 1 and sub_module is not None:
            attr = attr_list.pop(0)
            if hasattr(sub_module, attr):
                sub_module = getattr(sub_module, attr)
            else:
                sub_module = None

        # If the current sub-module exists, and it has the final attribute,
        # copy it into the checkpoint state dict.
        if sub_module is not None:
            if hasattr(sub_module, attr_list[0]):
                checkpoint_state_dict[value_name] = getattr(sub_module, attr_list[0])

    # Mapping of legacy weight names to new weight names.
    # Left side = name in the old checkpoint.
    # Right side = name expected by the new model implementation.
    legacy_weight_to_new_weight = {
        "features.embeddings.linear.weight": "graph_featurization_module.positional_embedding.embed_positional_features.weight",
        "features.embeddings.linear.bias": "graph_featurization_module.positional_embedding.embed_positional_features.bias",
        "features.edge_embedding.weight": "graph_featurization_module.edge_embedding.weight",
        "features.norm_edges.weight": "graph_featurization_module.edge_norm.weight",
        "features.norm_edges.bias": "graph_featurization_module.edge_norm.bias",
        "context_encoder_layers.0.norm1.weight": "protein_ligand_context_encoder_layers.0.norm1.weight",
        "context_encoder_layers.0.norm1.bias": "protein_ligand_context_encoder_layers.0.norm1.bias",
        "context_encoder_layers.0.norm2.weight": "protein_ligand_context_encoder_layers.0.norm2.weight",
        "context_encoder_layers.0.norm2.bias": "protein_ligand_context_encoder_layers.0.norm2.bias",
        "context_encoder_layers.0.W1.weight": "protein_ligand_context_encoder_layers.0.W1.weight",
        "context_encoder_layers.0.W1.bias": "protein_ligand_context_encoder_layers.0.W1.bias",
        "context_encoder_layers.0.W2.weight": "protein_ligand_context_encoder_layers.0.W2.weight",
        "context_encoder_layers.0.W2.bias": "protein_ligand_context_encoder_layers.0.W2.bias",
        "context_encoder_layers.0.W3.weight": "protein_ligand_context_encoder_layers.0.W3.weight",
        "context_encoder_layers.0.W3.bias": "protein_ligand_context_encoder_layers.0.W3.bias",
        "context_encoder_layers.0.dense.W_in.weight": "protein_ligand_context_encoder_layers.0.dense.W_in.weight",
        "context_encoder_layers.0.dense.W_in.bias": "protein_ligand_context_encoder_layers.0.dense.W_in.bias",
        "context_encoder_layers.0.dense.W_out.weight": "protein_ligand_context_encoder_layers.0.dense.W_out.weight",
        "context_encoder_layers.0.dense.W_out.bias": "protein_ligand_context_encoder_layers.0.dense.W_out.bias",
        "context_encoder_layers.1.norm1.weight": "protein_ligand_context_encoder_layers.1.norm1.weight",
        "context_encoder_layers.1.norm1.bias": "protein_ligand_context_encoder_layers.1.norm1.bias",
        "context_encoder_layers.1.norm2.weight": "protein_ligand_context_encoder_layers.1.norm2.weight",
        "context_encoder_layers.1.norm2.bias": "protein_ligand_context_encoder_layers.1.norm2.bias",
        "context_encoder_layers.1.W1.weight": "protein_ligand_context_encoder_layers.1.W1.weight",
        "context_encoder_layers.1.W1.bias": "protein_ligand_context_encoder_layers.1.W1.bias",
        "context_encoder_layers.1.W2.weight": "protein_ligand_context_encoder_layers.1.W2.weight",
        "context_encoder_layers.1.W2.bias": "protein_ligand_context_encoder_layers.1.W2.bias",
        "context_encoder_layers.1.W3.weight": "protein_ligand_context_encoder_layers.1.W3.weight",
        "context_encoder_layers.1.W3.bias": "protein_ligand_context_encoder_layers.1.W3.bias",
        "context_encoder_layers.1.dense.W_in.weight": "protein_ligand_context_encoder_layers.1.dense.W_in.weight",
        "context_encoder_layers.1.dense.W_in.bias": "protein_ligand_context_encoder_layers.1.dense.W_in.bias",
        "context_encoder_layers.1.dense.W_out.weight": "protein_ligand_context_encoder_layers.1.dense.W_out.weight",
        "context_encoder_layers.1.dense.W_out.bias": "protein_ligand_context_encoder_layers.1.dense.W_out.bias",
        "y_context_encoder_layers.0.norm1.weight": "ligand_context_encoder_layers.0.norm1.weight",
        "y_context_encoder_layers.0.norm1.bias": "ligand_context_encoder_layers.0.norm1.bias",
        "y_context_encoder_layers.0.norm2.weight": "ligand_context_encoder_layers.0.norm2.weight",
        "y_context_encoder_layers.0.norm2.bias": "ligand_context_encoder_layers.0.norm2.bias",
        "y_context_encoder_layers.0.W1.weight": "ligand_context_encoder_layers.0.W1.weight",
        "y_context_encoder_layers.0.W1.bias": "ligand_context_encoder_layers.0.W1.bias",
        "y_context_encoder_layers.0.W2.weight": "ligand_context_encoder_layers.0.W2.weight",
        "y_context_encoder_layers.0.W2.bias": "ligand_context_encoder_layers.0.W2.bias",
        "y_context_encoder_layers.0.W3.weight": "ligand_context_encoder_layers.0.W3.weight",
        "y_context_encoder_layers.0.W3.bias": "ligand_context_encoder_layers.0.W3.bias",
        "y_context_encoder_layers.0.dense.W_in.weight": "ligand_context_encoder_layers.0.dense.W_in.weight",
        "y_context_encoder_layers.0.dense.W_in.bias": "ligand_context_encoder_layers.0.dense.W_in.bias",
        "y_context_encoder_layers.0.dense.W_out.weight": "ligand_context_encoder_layers.0.dense.W_out.weight",
        "y_context_encoder_layers.0.dense.W_out.bias": "ligand_context_encoder_layers.0.dense.W_out.bias",
        "y_context_encoder_layers.1.norm1.weight": "ligand_context_encoder_layers.1.norm1.weight",
        "y_context_encoder_layers.1.norm1.bias": "ligand_context_encoder_layers.1.norm1.bias",
        "y_context_encoder_layers.1.norm2.weight": "ligand_context_encoder_layers.1.norm2.weight",
        "y_context_encoder_layers.1.norm2.bias": "ligand_context_encoder_layers.1.norm2.bias",
        "y_context_encoder_layers.1.W1.weight": "ligand_context_encoder_layers.1.W1.weight",
        "y_context_encoder_layers.1.W1.bias": "ligand_context_encoder_layers.1.W1.bias",
        "y_context_encoder_layers.1.W2.weight": "ligand_context_encoder_layers.1.W2.weight",
        "y_context_encoder_layers.1.W2.bias": "ligand_context_encoder_layers.1.W2.bias",
        "y_context_encoder_layers.1.W3.weight": "ligand_context_encoder_layers.1.W3.weight",
        "y_context_encoder_layers.1.W3.bias": "ligand_context_encoder_layers.1.W3.bias",
        "y_context_encoder_layers.1.dense.W_in.weight": "ligand_context_encoder_layers.1.dense.W_in.weight",
        "y_context_encoder_layers.1.dense.W_in.bias": "ligand_context_encoder_layers.1.dense.W_in.bias",
        "y_context_encoder_layers.1.dense.W_out.weight": "ligand_context_encoder_layers.1.dense.W_out.weight",
        "y_context_encoder_layers.1.dense.W_out.bias": "ligand_context_encoder_layers.1.dense.W_out.bias",
        "features.node_project_down.weight": "graph_featurization_module.node_embedding.weight",
        "features.node_project_down.bias": "graph_featurization_module.node_embedding.bias",
        "features.norm_nodes.weight": "graph_featurization_module.node_norm.weight",
        "features.norm_nodes.bias": "graph_featurization_module.node_norm.bias",
        "features.type_linear.weight": "graph_featurization_module.embed_atom_type_features.weight",
        "features.type_linear.bias": "graph_featurization_module.embed_atom_type_features.bias",
        "features.y_nodes.weight": "graph_featurization_module.ligand_subgraph_node_embedding.weight",
        "features.y_edges.weight": "graph_featurization_module.ligand_subgraph_edge_embedding.weight",
        "features.norm_y_edges.weight": "graph_featurization_module.ligand_subgraph_edge_norm.weight",
        "features.norm_y_edges.bias": "graph_featurization_module.ligand_subgraph_edge_norm.bias",
        "features.norm_y_nodes.weight": "graph_featurization_module.ligand_subgraph_node_norm.weight",
        "features.norm_y_nodes.bias": "graph_featurization_module.ligand_subgraph_node_norm.bias",
        "W_v.weight": "W_protein_to_ligand_edges_embed.weight",
        "W_v.bias": "W_protein_to_ligand_edges_embed.bias",
        "W_c.weight": "W_protein_encoding_embed.weight",
        "W_c.bias": "W_protein_encoding_embed.bias",
        "W_nodes_y.weight": "W_ligand_nodes_embed.weight",
        "W_nodes_y.bias": "W_ligand_nodes_embed.bias",
        "W_edges_y.weight": "W_ligand_edges_embed.weight",
        "W_edges_y.bias": "W_ligand_edges_embed.bias",
        "V_C.weight": "W_final_context_embed.weight",
        "V_C_norm.weight": "final_context_norm.weight",
        "V_C_norm.bias": "final_context_norm.bias",
    }
    # Rename the weights in the checkpoint state dict.
    for legacy_weight_name, new_weight_name in legacy_weight_to_new_weight.items():
        if legacy_weight_name in checkpoint_state_dict:
            checkpoint_state_dict[new_weight_name] = checkpoint_state_dict.pop(
                legacy_weight_name
            )

    # Remove unused atom type embedding weight.
    # - Previous LigandMPNN used 120 atom types, but the last one was unused.
    # - The new model uses 119 atom types.
    atom_type_embedding_keys = [
        "graph_featurization_module.embed_atom_type_features.weight",
        "graph_featurization_module.ligand_subgraph_node_embedding.weight",
    ]
    # For each of these keys, drop the unused atom type embedding.
    for key in atom_type_embedding_keys:
        if key in checkpoint_state_dict:
            legacy_weight = checkpoint_state_dict[key]
            num_atomic_numbers = model.graph_featurization_module.num_atomic_numbers
            checkpoint_state_dict[key] = torch.cat(
                [
                    legacy_weight[:, :num_atomic_numbers],
                    legacy_weight[:, num_atomic_numbers + 1 :],
                ],
                dim=1,
            )

    # Permute weights for embedding of pairwise backbone atom distances.
    # - The legacy model used the order specified in 'legacy_order' dict.
    # - The new model uses the order specified in 'new_order' list (the
    # outer product of the atom types in the order N, Ca, C, O, Cb).
    legacy_order = {
        "Ca-Ca": 0,
        "N-N": 1,
        "C-C": 2,
        "O-O": 3,
        "Cb-Cb": 4,
        "Ca-N": 5,
        "Ca-C": 6,
        "Ca-O": 7,
        "Ca-Cb": 8,
        "N-C": 9,
        "N-O": 10,
        "N-Cb": 11,
        "Cb-C": 12,
        "Cb-O": 13,
        "O-C": 14,
        "N-Ca": 15,
        "C-Ca": 16,
        "O-Ca": 17,
        "Cb-Ca": 18,
        "C-N": 19,
        "O-N": 20,
        "Cb-N": 21,
        "C-Cb": 22,
        "O-Cb": 23,
        "C-O": 24,
    }
    new_order = [
        "N-N",
        "N-Ca",
        "N-C",
        "N-O",
        "N-Cb",
        "Ca-N",
        "Ca-Ca",
        "Ca-C",
        "Ca-O",
        "Ca-Cb",
        "C-N",
        "C-Ca",
        "C-C",
        "C-O",
        "C-Cb",
        "O-N",
        "O-Ca",
        "O-C",
        "O-O",
        "O-Cb",
        "Cb-N",
        "Cb-Ca",
        "Cb-C",
        "Cb-O",
        "Cb-Cb",
    ]
    pairwise_backbone_atom_embeddings_keys = [
        "graph_featurization_module.edge_embedding.weight",
    ]
    for key in pairwise_backbone_atom_embeddings_keys:
        if key in checkpoint_state_dict:
            # Grab the legacy weight and shape.
            legacy_weight = checkpoint_state_dict[key]
            out_dim, _ = legacy_weight.shape

            # Grab the necessary dimensions from the model.
            num_positional_embeddings = (
                model.graph_featurization_module.num_positional_embeddings
            )
            num_atoms = (
                model.graph_featurization_module.num_backbone_atoms
                + model.graph_featurization_module.num_virtual_atoms
            )
            num_rbf = model.graph_featurization_module.num_rbf

            # Split positional and RBF embedding weights.
            legacy_weight_positional_embeddings = legacy_weight[
                :, :num_positional_embeddings
            ]
            legacy_weight_rbf_embeddings_flat = legacy_weight[
                :, num_positional_embeddings:
            ]

            # Reshape the weights to separate atom pairs and the rbf dimension.
            legacy_weight_rbf_embeddings_atom_pairs = (
                legacy_weight_rbf_embeddings_flat.view(
                    out_dim, num_atoms * num_atoms, num_rbf
                )
            )

            # Reorder the atom pairs to match the new order.
            new_weight_rbf_embeddings_atom_pairs = (
                legacy_weight_rbf_embeddings_atom_pairs[
                    :, [legacy_order[atom_pair_name] for atom_pair_name in new_order], :
                ]
            )

            # Flatten the reordered weights back to 2D.
            new_weight_rbf_embeddings_flat = (
                new_weight_rbf_embeddings_atom_pairs.reshape(
                    out_dim, num_atoms * num_atoms * num_rbf
                )
            )

            # Concatenate positional + reordered RBF
            checkpoint_state_dict[key] = torch.cat(
                [legacy_weight_positional_embeddings, new_weight_rbf_embeddings_flat],
                dim=1,
            )

    # Permute the token order of amino acids coming out of the model to match
    # the new vocabulary order.
    # - The legacy model used an order specified by alphabetic order of one-
    # letter amino acid codes.
    # - The new model uses an order specified by alphabetic order of three-
    # letter amino acid codes.
    token_embedding_keys = ["W_s.weight"]
    token_projection_keys = ["W_out.weight", "W_out.bias"]
    # For each of these keys, reorder the embeddings/projections.
    for key in token_embedding_keys + token_projection_keys:
        if key in checkpoint_state_dict:
            # Grab the old weight.
            legacy_weight = checkpoint_state_dict[key]

            # Reorder the weight/bias according to the new token order.
            if "weight" in key:
                checkpoint_state_dict[key] = legacy_weight[
                    [legacy_token_order.index(aa) for aa in token_order], :
                ]
            elif "bias" in key:
                checkpoint_state_dict[key] = legacy_weight[
                    [legacy_token_order.index(aa) for aa in token_order]
                ]
            else:
                raise ValueError(f"Unrecognized key for token projection: {key}")

    # Load the modified state dict into the model.
    model.load_state_dict(checkpoint_state_dict, strict=True)
