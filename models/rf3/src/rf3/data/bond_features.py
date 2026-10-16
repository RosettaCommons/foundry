"""RF3's checkpoint-compatible bond features on AtomWorks 3."""

import numpy as np
from atomworks.ml.transforms.bonds import AddAF3TokenBondFeatures
from atomworks.ml.utils.token import get_token_starts


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
        starts = get_token_starts(atom_array, add_exclusive_stop=True)
        n_tokens = len(starts) - 1
        atom_to_token = np.repeat(np.arange(n_tokens), np.diff(starts))
        left, right = atom_to_token[bonds[:, :2]].T
        # Reduce only bonded distances: no atom-by-atom dense allocation.
        # minimum.at propagates NaN just like the original segment-wise np.min.
        token_distances = np.full((n_tokens, n_tokens), np.inf)
        np.minimum.at(token_distances, (left, right), distance)
        np.minimum.at(token_distances, (right, left), distance)
        cutoff = np.inf if data.get("is_inference", False) else 2.4
        token_bonds = token_distances < cutoff
        np.fill_diagonal(token_bonds, False)
        residues = ~atom_array.atomize[starts[:-1]]
        token_bonds[np.outer(residues, residues)] = False
        data.setdefault("feats", {})["token_bonds"] = token_bonds
        return data
