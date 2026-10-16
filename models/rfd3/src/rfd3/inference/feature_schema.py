"""Semantic feature axes for padded RFD3 inference and CFG cropping.

Axis declarations are intentionally explicit: a channel size may equal the number
of atoms or tokens. All padded values are zero; validity is carried separately.
"""

ATOM_FEATURES = set(
    """atom_to_token_map ref_atom_name_chars ref_pos ref_mask
ref_element ref_charge ref_space_uid ref_pos_is_ground_truth has_zero_occupancy
ref_is_motif_atom_with_fixed_coord ref_is_motif_atom_unindexed motif_pos
ref_atomwise_rasa active_donor active_acceptor is_atom_level_hotspot is_backbone
is_sidechain is_virtual is_central is_ca is_motif_atom_with_fixed_coord
is_motif_atom_unindexed is_motif_atom_with_fixed_seq partial_t
sym_entity_id sym_transform_id is_sym_asu""".split()
)
TOKEN_FEATURES = set(
    """residue_index token_index asym_id entity_id sym_id restype
is_protein is_rna is_dna is_ligand is_atomized terminus_type is_polar
ref_motif_token_type ref_plddt is_non_loopy is_motif_token_unindexed
is_motif_token_with_fully_fixed_coord""".split()
)
TOKEN_PAIR_FEATURES = {"token_bonds", "unindexing_pair_mask"}
FEATURE_AXES = {
    **dict.fromkeys(ATOM_FEATURES, "L"),
    **dict.fromkeys(TOKEN_FEATURES, "I"),
    **dict.fromkeys(TOKEN_PAIR_FEATURES, "II"),
}


def crop_features(f, atoms, tokens, zero_features=()):
    """Crop a *single* example by semantic axes without changing its index values."""
    import torch

    out = {}
    for key, value in f.items():
        if key not in FEATURE_AXES:
            raise ValueError(f"Unregistered RFD3 feature: {key}")
        axes = FEATURE_AXES[key]
        value = value[:atoms] if axes == "L" else value[:tokens]
        if axes == "II":
            value = value[:, :tokens]
        out[key] = torch.zeros_like(value) if key in zero_features else value.clone()
    return out
