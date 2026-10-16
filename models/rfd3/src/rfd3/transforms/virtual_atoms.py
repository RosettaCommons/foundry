"""
Virtual-atom transforms for Atom14
"""

import biotite.structure as struc
import numpy as np
from atomworks.io.utils.atom_array_plus import concatenate_any
from atomworks.ml.transforms.base import (
    Transform,
)
from atomworks.ml.utils.token import get_token_starts
from rfd3.constants import (
    ATOM14_ATOM_NAME_TO_ELEMENT,
    ATOM14_ATOM_NAMES,
    VIRTUAL_ATOM_ELEMENT_NAME,
    association_schemes,
    association_schemes_stripped,
    ccd_ordering_atomchar,
)
from rfd3.transforms.conditioning_base import (
    UnindexFlaggedTokens,
)
from rfd3.transforms.util_transforms import (
    assert_single_representative,
    get_af3_token_representative_masks,
)

from foundry.common import exists


def map_to_association_scheme(atom_names: list | str, res_name: str, scheme="atom14"):
    """
    Maps a list of names to the atom14 naming scheme for that particular name (within a specific residue)
    NB this function is a bit more general since it is used to handle tipatoms too.
    """
    if scheme not in association_schemes_stripped:
        raise ValueError(
            f"Scheme {scheme} not found in association_schemes_stripped. Available schemes: {list(association_schemes_stripped.keys())}"
        )
    atom_names = (
        [str(atom_names)] if isinstance(atom_names, (str, np.str_)) else atom_names
    )
    idxs = np.array(
        [
            association_schemes_stripped[scheme][res_name].index(name)
            for name in atom_names
        ]
    )
    return ATOM14_ATOM_NAMES[idxs]


def map_names_to_elements(
    atom_names: list | str, default=VIRTUAL_ATOM_ELEMENT_NAME
) -> np.ndarray:
    """
    Maps ATOM14 atom names to their corresponding elements.
    If a name is not in ATOM14_ATOM_NAMES (e.g. if atom name is VX - virtual atom),
    then it returns the default value
    """
    atom_names = [atom_names] if isinstance(atom_names, str) else atom_names
    elements = [ATOM14_ATOM_NAME_TO_ELEMENT.get(name, default) for name in atom_names]
    return np.array(elements)


def generate_atom_mappings_(scheme="atom14"):
    scheme = association_schemes[scheme]

    atom_mapping = {}
    symmetry_mapping = {}

    for aaa, atom14_names in ccd_ordering_atomchar.items():
        mapping = list(range(14))
        scheme_names = scheme[aaa]

        for ccd_index in range(len(atom14_names)):
            atom14_name = atom14_names[ccd_index]
            if atom14_name is not None:
                assert (
                    atom14_name in scheme_names
                ), f"{atom14_name} not in CCD ordering for {aaa}"
                scheme_index = scheme_names.index(atom14_name)
                scheme_index_in_cur_mapping = mapping.index(scheme_index)
                mapping[ccd_index], mapping[scheme_index_in_cur_mapping] = (
                    mapping[scheme_index_in_cur_mapping],
                    mapping[ccd_index],
                )

        assert set(mapping) == set(range(len(scheme_names)))

        # atom_mapping[aaa] = mapping
        atom_mapping[aaa] = mapping

        ##################################################################
        # Temporarily comment this out
        # if aaa in symmetric_atomchar:
        #     symmetry_mapping[aaa] = []
        #     for group in symmetric_atomchar[aaa]:
        #         indices = [atom14_names.index(name) for name in group]
        #         symmetry_mapping[aaa].append(indices)
        symmetry_mapping = {}
        ##################################################################

    # Test that the mapping is valid
    for aaa in atom_mapping.keys():
        idxs = atom_mapping[aaa]

        assert len(idxs) == len(set(idxs)), f"Duplicate indices in mapping for {aaa}"

        atom_mapping_expected = np.array(scheme[aaa])[idxs]
        atom_mapping_actual = np.array(ccd_ordering_atomchar[aaa])

        assert np.array_equal(
            atom_mapping_expected, atom_mapping_actual
        ), f"Mapping mismatch for {aaa}: {atom_mapping_expected} != {atom_mapping_actual}"

    return atom_mapping, symmetry_mapping


def permute_symmetric_atom_names_(
    atom_names: np.ndarray, res_name: str, association_map: dict, symmetry_map: dict
) -> np.ndarray:
    # NB: Can leak GT sequence if the model receives the canconical ordering of atoms as input
    # With the structure-local atom attention it will not unless N_keys(n_attn_seq_neighbours) > n_atom_attn_queries.
    if res_name in association_map:
        idx_to_swap = association_map[res_name]
        atom_names = atom_names[idx_to_swap]
        if res_name in symmetry_map:
            for group in symmetry_map[res_name]:
                if np.random.rand() < 0.5:  # random swap
                    atom_names[group] = atom_names[group[::-1]]
    return atom_names


#####################################################################################################
# Virtual atom transforms
#####################################################################################################


class PadTokensWithVirtualAtoms(Transform):
    """
    Pads tokens with virtual atoms to ensure all residue tokens have a fixed number of atoms

    Applies padding only to the tokens who do not have sequence
    Applies association schema during training and to tokens with sequence.
    """

    requires_previous_transforms = [UnindexFlaggedTokens]  # type: ignore[list-item]

    def __init__(
        self,
        n_atoms_per_token,
        atom_to_pad_from,
        association_scheme,
    ):
        self.n_atoms_per_token = n_atoms_per_token
        self.atom_to_pad_from = atom_to_pad_from
        self.association_scheme = association_scheme
        if exists(association_scheme):
            self.association_map_, self.symmetry_map_ = generate_atom_mappings_(
                association_scheme
            )

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        starts = get_token_starts(atom_array, add_exclusive_stop=True)
        token_starts = starts[:-1]
        token_level_array = atom_array[token_starts]
        is_motif_atom_with_fixed_seq = token_level_array.is_motif_atom_with_fixed_seq
        is_motif_token_unindexed = token_level_array.is_motif_atom_unindexed

        token_ids = np.unique(atom_array.token_id)
        assert len(token_ids) == len(
            is_motif_atom_with_fixed_seq
        ), "Token ids and token level array have different lengths!"

        # Unindexed tokens are never fully atomized, but may be assigned as atomized to have repr atoms:
        is_residue = (
            token_level_array.is_protein & ~token_level_array.atomize
        ) | is_motif_token_unindexed

        # Unindexed tokens are never padded, and so are treated as residues with fixed sequence.
        is_paddable = is_residue & ~(
            is_motif_atom_with_fixed_seq | is_motif_token_unindexed
        )
        is_non_paddable_residue = is_residue & (
            is_motif_atom_with_fixed_seq | is_motif_token_unindexed
        )

        # First pass: append virtual atoms to every paddable token that is short of
        # `n_atoms_per_token` atoms. Each virtual atom is a copy of the token's
        # representative atom, and all of them are inserted in a single step.
        n_pad = self.n_atoms_per_token - np.diff(starts)
        pad_token_idxs = np.flatnonzero(is_paddable & (n_pad > 0))
        if len(pad_token_idxs) > 0:
            atom_array_padded = self._insert_virtual_atoms(
                atom_array,
                starts,
                pad_token_idxs,
                n_pad[pad_token_idxs],
                is_inference=data["is_inference"],
            )
        else:
            atom_array_padded = atom_array

        # Initialize gt_atom_name annotation if it doesn't exist
        if "gt_atom_name" not in atom_array_padded.get_annotation_categories():
            atom_array_padded.set_annotation(
                "gt_atom_name", np.empty(len(atom_array_padded), dtype="U4")
            )

        # Second pass: process tokens with proper atom name assignment after padding
        # Get updated token starts after padding
        starts_padded = get_token_starts(atom_array_padded, add_exclusive_stop=True)

        for token_id, (start, end) in enumerate(
            zip(starts_padded[:-1], starts_padded[1:])
        ):
            if is_paddable[token_id]:
                # ... Permutation of atom names during training
                if not data["is_inference"] and exists(self.association_scheme):
                    atom_names = permute_symmetric_atom_names_(
                        ATOM14_ATOM_NAMES,
                        atom_array_padded.res_name[start],
                        association_map=self.association_map_,
                        symmetry_map=self.symmetry_map_,
                    )
                else:
                    atom_names = ATOM14_ATOM_NAMES
                atom_array_padded.atom_name[start:end] = atom_names
                atom_array_padded.get_annotation("gt_atom_name")[start:end] = atom_names

            elif is_non_paddable_residue[token_id]:
                # When sequence-constrained, we want to directly map the residue name based on the sequence
                atom_names, res_name = (
                    atom_array_padded.atom_name[start:end],
                    atom_array_padded.res_name[start],
                )
                atom_array_padded.get_annotation("gt_atom_name")[start:end] = atom_names
                atom_names = map_to_association_scheme(
                    atom_names, res_name, scheme=self.association_scheme
                )
                atom_array_padded.atom_name[start:end] = atom_names
            else:
                # ... Add gt_atom_name annotation to other tokens
                atom_names = atom_array_padded.atom_name[start:end]
                atom_array_padded.get_annotation("gt_atom_name")[start:end] = atom_names

            # ... Update atom array
            assert {VIRTUAL_ATOM_ELEMENT_NAME} != set(
                atom_array_padded.element[start:end].tolist()
            ), (
                "Padded atoms should be virtual atoms, but found: "
                f"{set(atom_array_padded.element[start:end].tolist())}"
            )

        data["atom_array"] = atom_array_padded
        return data

    def _insert_virtual_atoms(
        self,
        atom_array: struc.AtomArray,
        starts: np.ndarray,
        pad_token_idxs: np.ndarray,
        n_pad: np.ndarray,
        is_inference: bool,
    ) -> struc.AtomArray:
        """Insert ``n_pad[i]`` virtual atoms at the end of token ``pad_token_idxs[i]``.

        Vectorised equivalent of building one ``struc.array([pad_atom] * n_pad)``
        per token and inserting them with ``insert_atoms``: the virtual atoms,
        their annotations (including dtypes and order) and their positions in the
        returned array are identical.
        """
        token_of_atom = np.repeat(np.arange(len(starts) - 1), np.diff(starts))
        rep_mask = get_af3_token_representative_masks(
            atom_array, central_atom=self.atom_to_pad_from
        )
        # `assert_single_representative` checks the default (CB) representative.
        single_rep_mask = (
            rep_mask
            if self.atom_to_pad_from == "CB"
            else get_af3_token_representative_masks(atom_array, central_atom="CB")
        )
        n_reps = np.bincount(token_of_atom[rep_mask], minlength=len(starts) - 1)
        n_single_reps = np.bincount(
            token_of_atom[single_rep_mask], minlength=len(starts) - 1
        )
        is_invalid = (n_single_reps[pad_token_idxs] != 1) | (
            n_reps[pad_token_idxs] == 0
        )
        if is_invalid.any():
            # Re-run the per-token checks on the first invalid token so that the
            # same exception (and message) is raised as by a per-token loop.
            token_idx = pad_token_idxs[np.argmax(is_invalid)]
            token = atom_array[starts[token_idx] : starts[token_idx + 1]]
            mask = get_af3_token_representative_masks(
                token, central_atom=self.atom_to_pad_from
            )
            assert_single_representative(token)
            token[mask][0]
            raise AssertionError(f"Token {token_idx} has no representative atom")

        # ... The first representative atom of each token is copied as virtual atom
        rep_idxs = np.flatnonzero(rep_mask)
        _, first = np.unique(token_of_atom[rep_idxs], return_index=True)
        rep_idx_per_token = np.full(len(starts) - 1, -1)
        rep_idx_per_token[token_of_atom[rep_idxs[first]]] = rep_idxs[first]
        rep_idxs = rep_idx_per_token[pad_token_idxs]

        # ... Expand to desired number of atoms (mirrors the annotation order and
        # dtypes of the patched `struc.array`, which `atomworks` installs)
        src_idxs = np.repeat(rep_idxs, n_pad)
        n_total = len(src_idxs)
        pad_array = struc.AtomArray(n_total)
        for annotation in sorted(atom_array.get_annotation_categories()):
            if annotation == "element":
                values = np.full(n_total, VIRTUAL_ATOM_ELEMENT_NAME)
            else:
                values = atom_array.get_annotation(annotation)[src_idxs]
            pad_array.set_annotation(annotation, values)
        pad_array.coord = atom_array.coord[src_idxs].astype(np.float32)

        # ... Change occupancy | if the representative atom has occupancy, set to 1.0
        occ = np.where(atom_array.occupancy[rep_idxs] > 0.0, 1.0, 0.0)
        pad_array.occupancy = np.repeat(occ, n_pad)

        # ... Even if the input pad_atoms are all motif, we don't ever want padded atoms to be motif
        pad_array.is_motif_atom = np.zeros(n_total, dtype=bool)

        # ... Pad_atoms should never inherit fixed state
        if is_inference:
            pad_array.is_motif_atom_with_fixed_coord = np.zeros(
                n_total, dtype=atom_array.is_motif_atom_with_fixed_coord.dtype
            )

        # ... Handle multidimensional annotations
        for annotation in atom_array.get_annotation_categories():
            if len(atom_array.get_annotation(annotation).shape) > 1:
                stacked = np.stack(pad_array.get_annotation(annotation)).astype(float)
                pad_array.del_annotation(annotation)
                pad_array.set_annotation(annotation, stacked)

        # ... Insert the virtual atoms of each token before the start of the next token
        insert_positions = np.repeat(starts[pad_token_idxs + 1], n_pad)
        n_atoms = atom_array.array_length()
        order = np.insert(
            np.arange(n_atoms), insert_positions, np.arange(n_atoms, n_atoms + n_total)
        )
        return concatenate_any([atom_array, pad_array])[order]
