"""RF3's checkpoint-compatible bond features on AtomWorks 3."""

import numpy as np
from atomworks.ml.transforms.bonds import AddAF3TokenBondFeatures
from atomworks.ml.utils.token import apply_segment_wise_2d, get_token_starts


class AddRF3TokenBondFeatures(AddAF3TokenBondFeatures):
    """Preserve the training-only 2.4 Å cutoff used by RF3 on AtomWorks 2.

    AtomWorks 3 moved distance handling to the parser and removed the feature
    cutoff. RF3 must retain its training feature policy without deleting bonds
    from the structure used by conformers, atomization, or downstream metrics.
    Inference keeps bonds of any finite length, including bonds between
    components whose coordinates have not yet been folded together.
    """

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        bonds = atom_array.bonds.as_array()
        distance = np.linalg.norm(
            atom_array.coord[bonds[:, 0]] - atom_array.coord[bonds[:, 1]], axis=-1
        )
        distances = np.full((len(atom_array), len(atom_array)), np.inf)
        distances[bonds[:, 0], bonds[:, 1]] = distance
        distances[bonds[:, 1], bonds[:, 0]] = distance
        starts = get_token_starts(atom_array, add_exclusive_stop=True)
        # Match the checkpoint's minimum bonded distance per token pair,
        # including its handling of non-finite coordinates.
        token_distances = apply_segment_wise_2d(distances, starts, np.min)
        cutoff = np.inf if data.get("is_inference", False) else 2.4
        token_bonds = token_distances < cutoff
        np.fill_diagonal(token_bonds, False)
        residues = ~atom_array.atomize[starts[:-1]]
        token_bonds[np.outer(residues, residues)] = False
        data.setdefault("feats", {})["token_bonds"] = token_bonds
        return data
