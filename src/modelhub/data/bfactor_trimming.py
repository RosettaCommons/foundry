from collections import defaultdict
from typing import Any, Literal

import numpy as np
import toolz
import torch
from biotite.structure import AtomArray
from cifutils.constants import ELEMENT_NAME_TO_ATOMIC_NUMBER
from cifutils.tools.rdkit import atom_array_from_rdkit
from cifutils.utils.selection import get_residue_starts
from datahub.enums import GroundTruthConformerPolicy
from datahub.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_is_instance,
    check_nonzero_length,
)
from datahub.transforms.af3_reference_molecule import (
    KNOWN_CCD_CODES,
    _encode_atom_names_like_af3,
    _get_rdkit_mols_with_conformers,
    _map_reference_conformer_to_residue,
    logger,
)
from datahub.transforms.base import Transform
from datahub.transforms.chirals import get_rf2aa_chiral_features
from datahub.transforms.rdkit_utils import (
    find_automorphisms_with_rdkit,
    get_chiral_centers,
    sample_rdkit_conformer_for_atom_array,
)
from datahub.utils.geometry import masked_center, random_rigid_augmentation


class TrimOnBfactor(Transform):
    """
    This component marks atoms as occ=0 based on bfactor values

    It takes as input 'brange', a list specifying the Mminimum and maximum B factors to
    keep.

    Example:
        brange = [-1.0,70.0] will mark with occ=0 any atom with b>70 or b<-1
    """

    def __init__(
        self,
        brange,
    ):
        self.bmin = brange[0]
        self.bmax = brange[1]

    def check_input(self, data: dict):
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(
            data, ["b_factor", "occupancy"]
        )

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        bfact = atom_array.get_annotation('b_factor')
        mask = (bfact<self.bmin) | (bfact>self.bmax)
        occ = atom_array.get_annotation('occupancy')
        occ[mask] = 0.0
        atom_array.set_annotation('occupancy',occ)

        data["atom_array"] = atom_array

        return data


def _get_reference_conformer_to_residue_mapping(
    atom_names: np.ndarray, conformer: AtomArray
) -> tuple[np.ndarray]:
    """
    Maps atom indices from a reference conformer (as an AtomArray) to the specified residue, dropping all atoms that are not in the residue.
    Args:
        - atom_names (np.ndarray): Array of atom names in the residue to map to
        - conformer (AtomArray): The reference conformer, as an AtomArray (containing the atom_name annotation)
    Returns:
        - ref_map (np.ndarray): Index in atom of reference positions (-1 if masked)
    """

    # ... mark the atoms that are in the residue (keep) and where
    keep = np.zeros(len(conformer), dtype=bool)  # [n_atoms_in_conformer]
    to_within_res_idx = -np.ones(len(conformer), dtype=int)  # [n_atoms_in_conformer]

    for i, atom_name in enumerate(atom_names):
        matching_atom_idx = np.where(conformer.atom_name == atom_name)[0]
        if len(matching_atom_idx) == 0:
            logger.warning(f"Atom {atom_name} not found in conformer.")
            continue
        matching_atom_idx = matching_atom_idx.item()
        keep[matching_atom_idx] = True
        to_within_res_idx[matching_atom_idx] = i

    return to_within_res_idx  # [n_atoms_in_conf]


class AddAF3ChiralFeatures(Transform):
    """
    AddAF3ChiralFeatures adds chiral features.

    This transform adds the following features to the data dictionary under the 'feats' key:
        - chiral_feats: [N_chiral_centers, 5] A listing of chiral centers of the format:
                  tensor([[ 5.,  1.,  2.,  3.,  0.61546...],...])
          Here, the first 4 columns define atom indices of chiral center; the 5th is target dihedral

    Metadata from GetRDKitChiralCenters is needed for this transform.

    Args:
        data (dict[str, Any]): A dictionary containing the input data, including the atom array and chiral centers.

    Returns:
        dict[str, Any]: The updated `data` dictionary with the added chiral features under the `feats` key.
    """

    requires_previous_transforms = ["GetRDKitChiralCenters"]

    def check_input(self, data: dict[str, Any]):
        check_contains_keys(data, ["atom_array", "chiral_centers", "rdkit", "feats"])
        check_is_instance(data, "atom_array", AtomArray)
        check_nonzero_length(data, "atom_array")

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array: AtomArray = data["atom_array"]

        # We're going to use the same logic we do in GetAF3ReferenceMoleculeFeatures
        # ... get residue-level stochiometry
        _res_start_ends = get_residue_starts(atom_array, add_exclusive_stop=True)
        _res_starts, _res_ends = _res_start_ends[:-1], _res_start_ends[1:]

        all_chirals = []
        for res_start, res_end in zip(_res_starts, _res_ends):
            res_name = atom_array.res_name[res_start]

            chirals = data["chiral_centers"][res_name]
            if len(chirals) == 0:
                continue

            # get rdkit->atomarray mapping
            conformer = atom_array_from_rdkit(
                data["rdkit"][res_name],
                conformer_id=0,
                remove_hydrogens=True,
            )
            _ref_to_conf_map = _get_reference_conformer_to_residue_mapping(
                atom_names=atom_array.atom_name[res_start:res_end], conformer=conformer
            )

            # calculate chirals from reference conformer
            chirals = get_rf2aa_chiral_features(chirals, torch.tensor(conformer.coord))

            # remap reference conformer to native index
            chirals[:, :4] = torch.tensor(_ref_to_conf_map)[chirals[:, :4].long()]

            # remove unmasked chirals
            mask = (chirals[:, :4] >= 0).all(dim=1)
            chirals = chirals[mask]

            # add atom offset
            chirals[:, :4] = chirals[:, :4] + res_start

            all_chirals.append(chirals)

        all_chirals = torch.cat(all_chirals, dim=0)
        data["feats"]["chiral_feats"] = all_chirals  # [n_chirals, 5]

        return data


class GetRDKitChiralCenters(Transform):
    """
    Identify chiral centers in the RDKit molecules stored in the `data["rdkit"]` dictionary.
    Returns a dictionary mapping each residue name to a list of chiral centers, e.g:
      data["chiral_centers"] = {
          ...
          "ILE": [
              {'chiral_center_idx': 1, 'bonded_explicit_atom_idxs': [0, 2, 4], 'chirality': 'S'},
              {'chiral_center_idx': 4, 'bonded_explicit_atom_idxs': [1, 5, 6], 'chirality': 'S'}
          ],
          ...
      }
    Each chiral center is a dict with a center atom index, 3 or 4 bonded atom indices, and the
    RDKit-determined chirality.

    Uses RDKit molecules first computed in GetAF3ReferenceMoleculeFeatures.

    Args:
        data (dict[str, Any]): A dictionary containing the input data, including RDKit molecules
            under the `"rdkit"` key.

    Returns:
        dict[str, Any]: The updated `data` dictionary with `chiral_centers` containing chiral
            centers for each molecule.
    """

    requires_previous_transforms = ["GetAF3ReferenceMoleculeFeatures"]

    def check_input(self, data: dict[str, Any]):
        check_contains_keys(data, ["rdkit"])
        check_is_instance(data, "rdkit", dict)
        check_nonzero_length(data, "rdkit")

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        data["chiral_centers"] = {}
        # Get chiral centers for all rdkit mols
        for resname, rdmol in data["rdkit"].items():
            try:
                # Get the chiral centers (returned are the indices of the chiral center atoms
                #  within the `obmol` object)
                data["chiral_centers"][resname] = get_chiral_centers(rdmol)

            except Exception as e:
                logger.warning(
                    f"Failed to find chiral centers for molecule {resname}: {e}"
                )

        return data
