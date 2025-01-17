from typing import Any
import torch
import numpy as np
from biotite.structure import AtomArray

from datahub.transforms._checks import (
    check_contains_keys,
    check_is_instance,
)
from datahub.transforms.base import Transform

class AddAF3PocketAlignmentMask(Transform):
    """
    Adds an AF3 style alignment mask that specifies which atoms should be used
    for aligning on the pocket when calculating ligand RMSD metrics.
    For proteins, this is all CAs, while other molecules use every atom.

    Adds:
        - 'alignment_mask_atm_lvl': torch.Tensor of shape [I] (bool)
    """


    def __init__(self):
        pass

    def check_input(self, data):
        check_contains_keys(data, ["atom_array", "feats"])
        check_is_instance(data, "atom_array", AtomArray)

    def forward(self, data):
        # get the masks
        atom_array = data["atom_array"]
        is_protein = data["feats"]["is_protein"]
        is_nucleic = ~data["feats"]["is_protein"] & ~data["feats"]["is_ligand"]

        # now get the mask for atoms that should be used in the alignment (ie CA protein atoms and all atoms for everything else.)
        protein_alignment_atom = is_protein[atom_array.token_id.astype(np.int32)] & (atom_array.atom_name == "CA")
        other_alignment_atom = is_nucleic[atom_array.token_id.astype(np.int32)]
        alignment_mask = protein_alignment_atom | other_alignment_atom
        data["alignment_mask_atm_lvl"] = alignment_mask

        return data