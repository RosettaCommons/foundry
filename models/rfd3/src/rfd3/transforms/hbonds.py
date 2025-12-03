from typing import Any, Literal, Tuple

import biotite.structure as struc
import hydride
import numpy as np
from atomworks.io.transforms.atom_array import remove_hydrogens
from atomworks.io.utils.ccd import atom_array_from_ccd_code
from atomworks.ml.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_is_instance,
)
from atomworks.ml.transforms.base import Transform
from biotite.structure import AtomArray, AtomArrayStack
from rfd3.constants import SELECTION_NONPROTEIN, SELECTION_PROTEIN

from foundry.utils.ddp import RankedLogger

ranked_logger = RankedLogger()

HYDROGEN_LIKE_SYMBOLS = ("H", "H2", "D", "T")


# TODO: Once the cifutils submodule is bumped, we can use the built-in add_hydrogen_atom_positions function
def add_hydrogen_atom_positions(
    atom_array: AtomArray | AtomArrayStack,
) -> AtomArray | AtomArrayStack:
    """Add hydrogens using biotite supported hydride library

    Args:
        atom_array (AtomArray | AtomArrayStack): The atom array containing the chain information.

    Returns:
        AtomArray: The updated atom array with hydrogens added.
    """

    def _get_charge_from_ccd_code(atom):
        try:
            ccd_array = atom_array_from_ccd_code(atom.res_name)
            charge = ccd_array[
                ccd_array.atom_name.tolist().index(atom.atom_name)
            ].charge
        except Exception:
            ## res_name not found in ccd or atom_name not found in ccd_array
            charge = 0
        return charge

    if "charge" not in atom_array.get_annotation_categories():
        charges = np.vectorize(_get_charge_from_ccd_code)(atom_array)
        atom_array.set_annotation("charge", charges)

    # Add as a custom annotation

    array = remove_hydrogens(atom_array)

    fields_to_copy_from_residue_if_present = [
        "auth_seq_id",
        "label_entity_id",
        "is_can_prot",
        "is_can_nucl",
        "is_sm",
        "chain_type",
    ]
    fields_to_copy_from_residue_if_present = list(
        set(fields_to_copy_from_residue_if_present).intersection(
            set(atom_array.get_annotation_categories())
        )
    )

    def _copy_missing_annotations_residue_wise(
        arr_to_copy_from: AtomArray,
        arr_to_update: AtomArray,
        fields_to_copy_from_residue_if_present: list[str],
    ) -> AtomArray:
        """Copy specified annotations residue-wise from one AtomArray to another. Updates annotations in-place."""
        residue_starts = struc.get_residue_starts(arr_to_copy_from)
        residue_starts_atom_array = arr_to_copy_from[residue_starts]
        annot = {
            item: getattr(residue_starts_atom_array, item)
            for item in fields_to_copy_from_residue_if_present
        }
        for field in fields_to_copy_from_residue_if_present:
            updated_field = struc.spread_residue_wise(arr_to_update, annot[field])
            arr_to_update.set_annotation(field, updated_field)
        return arr_to_update

    def _handle_nan_coords(atom_array, noise_level=1e-3):
        coords = atom_array.coord

        # Find NaNs
        nan_mask = np.isnan(coords)

        # Replace NaNs with 0 + small random offset
        coords[nan_mask] = np.random.uniform(
            -noise_level, noise_level, size=nan_mask.sum()
        )

        # Update atom_array in-place
        atom_array.coord = coords
        return atom_array, nan_mask

    if isinstance(array, AtomArrayStack):
        updated_arrays = []
        for old_arr in array:
            if old_arr.bonds is None:
                old_arr.bonds = struc.connect_via_distances(old_arr)

            ## give some values to nan
            old_arr, nan_mask = _handle_nan_coords(old_arr)
            arr, mask = hydride.add_hydrogen(old_arr)
            ## put back nans
            arr.coord[mask, :][nan_mask] = np.nan
            arr = _copy_missing_annotations_residue_wise(
                old_arr, arr, fields_to_copy_from_residue_if_present
            )
            updated_arrays.append(arr)

        ret_array = struc.stack(updated_arrays)

    elif isinstance(array, AtomArray):
        if array.bonds is None:
            array.bonds = struc.connect_via_distances(array)
        ## give some values to nan
        array, nan_mask = _handle_nan_coords(array)
        arr, mask = hydride.add_hydrogen(array)
        ## put back nans
        arr.coord[mask, :][nan_mask] = np.nan
        ret_array = _copy_missing_annotations_residue_wise(
            array, arr, fields_to_copy_from_residue_if_present
        )
    return ret_array


def check_atom_array_has_hydrogen(data: dict[str, Any]):
    """Check if `atom_array` key has bonds."""
    import numpy as np

    if not np.any(data["atom_array"].element == "H"):
        raise ValueError("Key `atom_array` in data has no hydrogens.")


def calculate_hbonds(
    atom_array: AtomArray,
    selection1: np.ndarray = None,
    selection2: np.ndarray = None,
    selection1_type: Literal["acceptor", "donor", "both"] = "both",
    cutoff_dist: float = 3,
    cutoff_angle: float = 120,
    donor_elements: Tuple[str] = ("O", "N", "S", "F"),
    acceptor_elements: Tuple[str] = ("O", "N", "S", "F"),
    periodic: bool = False,
) -> Tuple[np.ndarray, np.ndarray, AtomArray]:
    """
    Calculates Hbonds with biotite.struc.Hbond.
    Assigns donor, acceptor annotation for each heavy atom involved.
    Args:
            atom_array (AtomArray):Expects the atom_array that contains hydrogens.

            selection1 and selection2 (np.ndarray, optional): (Boolean mask for atoms to limit the hydrogen bond search to specific sections of the model.
            The shape must match the shape of the atoms argument. If None is given, the whole atoms stack is used instead. (Default: None))

            selection1_type (Literal, optional): Determines the type of selection1. The type of selection2 is chosen accordingly (‘both’ or the opposite).
                                               (Default: 'both')
            cutoff_dist (float, optional): The maximal distance between the hydrogen and acceptor to be considered a hydrogen bond. (Default: 2.5)
            cutoff_angle (float, optional): The angle cutoff in degree between Donor-H..Acceptor to be considered a hydrogen bond. (Default: 120)
            donor_elements, acceptor_elements (tuple of str): Elements to be considered as possible donors or acceptors. (Default: O, N, S)
            periodic (bool, optional): If true, hydrogen bonds can also be detected in periodic boundary conditions. The box attribute of atoms is required in this case. (Default: False)


    """
    # Remove NaN coordinates
    has_resolved_coordinates = ~np.isnan(atom_array.coord).any(axis=-1)
    nonNaN_array = atom_array[has_resolved_coordinates]

    # update selections if any
    if selection1 is not None:
        selection1 = selection1[has_resolved_coordinates]
    if selection2 is not None:
        selection2 = selection2[has_resolved_coordinates]

    ## index map from nonNaN_array to original
    index_map = {
        counter: i for counter, i in enumerate(has_resolved_coordinates.nonzero()[0])
    }

    if selection1.sum() == 0 or selection2.sum() == 0:
        # no ligand, or ligand is of same type as selection1 (e.g. 6) (peptide)
        triplets = np.array([])
    else:
        # Compute H bonds
        triplets = struc.hbond(  ## assuming AtomArray, not AtomArrayStack (returns an extra masks in that case)
            nonNaN_array,
            selection1=selection1,
            selection2=selection2,
            selection1_type=selection1_type,
            cutoff_dist=cutoff_dist,
            cutoff_angle=cutoff_angle,
            donor_elements=donor_elements,
            acceptor_elements=acceptor_elements,
            periodic=periodic,
        )

    ## map back triplet indices, nonNaN indices to original indices
    flattened = triplets.flatten()
    triplets = np.array([index_map[i] for i in flattened]).reshape(-1, 3)

    ## add back NaNs

    donor_array = np.array([[0.0] * len(atom_array)])
    acceptor_array = np.array([[0.0] * len(atom_array)])

    if len(triplets) > 0:
        donor_array[:, triplets[:, 0]] = 1.0
        acceptor_array[:, triplets[:, 2]] = 1.0

    ## [is_active_donor, is_active_acceptor] per atom
    types = np.vstack((donor_array, acceptor_array)).T

    return triplets, types, atom_array


class CalculateHbonds(Transform):
    """Transform for calculating Hbonds, expects an AtomArray containing hydrogens."""

    def __init__(
        self,
        selection1_type: Literal["acceptor", "donor", "both"] = "both",
        cutoff_dist: float = 3,
        cutoff_angle: float = 120,
        donor_elements: Tuple[str] = ("O", "N", "S", "F"),
        acceptor_elements: Tuple[str] = ("O", "N", "S", "F"),
        periodic: bool = False,
        make2d: bool = False,
    ):
        """
        Initialize the Hbonds transform.

        Args:

            selection1 and selection2 (list[str], optional): Specify a list of ChainTypes as in atomworks.enums. e.g. selectoin1 = ['POLYPEPTIDE(L)'], selection2 = ['NON-POLYMER', 'POLYRIBONUCLEOTIDE']
            Allowed values: {'PEPTIDE NUCLEIC ACID', 'BRANCHED', 'POLYDEOXYRIBONUCLEOTIDE', 'POLYRIBONUCLEOTIDE', 'CYCLIC-PSEUDO-PEPTIDE', 'MACROLIDE', 'POLYDEOXYRIBONUCLEOTIDE/POLYRIBONUCLEOTIDE HYBRID', 'OTHER', 'POLYPEPTIDE(L)', 'NON-POLYMER', 'POLYPEPTIDE(D)', 'WATER'}

            selection1_type (Literal, optional): Determines the type of selection1. The type of selection2 is chosen accordingly (‘both’ or the opposite).
                                               (Default: 'both')
            cutoff_dist (float, optional): The maximal distance between the hydrogen and acceptor to be considered a hydrogen bond. (Default: 2.5)
            cutoff_angle (float, optional): The angle cutoff in degree between Donor-H..Acceptor to be considered a hydrogen bond. (Default: 120)
            donor_elements, acceptor_elements (tuple of str): Elements to be considered as possible donors or acceptors. (Default: O, N, S)
            periodic (bool, optional): If true, hydrogen bonds can also be detected in periodic boundary conditions. The box attribute of atoms is required in this case. (Default: False)
        """
        self.selection1_type = selection1_type
        self.cutoff_dist = cutoff_dist
        self.cutoff_angle = cutoff_angle
        self.donor_elements = donor_elements
        self.acceptor_elements = acceptor_elements
        self.periodic = periodic
        self.make2d = make2d

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["res_name"])

        ## turn off cause H addition debug ongoing
        # check_atom_array_has_hydrogen(data)

    def forward(self, data: dict) -> dict:
        """
        Calculates Hbonds and adds it to the data dictionary under the key `hbonds`.

        Args:
            data: dict
                A dictionary containing the input data atomarray.
                Expects the atom_array in data["atom_array"] contains hydrogens.


        Returns:
            dict: The data dictionary with hbonds added.
            Sets hbond_type = [Donor, Acceptor] annotation to each atom. Donor, Acceptor can be both 0 or 1 (float). size: Lx2 (L: length of AtomArray)
        """

        atom_array: AtomArray = data["atom_array"]

        try:
            atom_array = add_hydrogen_atom_positions(atom_array)

        except Exception as e:
            print(
                f"WARNING: problem adding hydrogens: {e}.\nThis example will get no hydrogen bond annotations."
            )
            atom_array.set_annotation(
                "active_donor", np.zeros(atom_array.array_length(), dtype=bool)
            )
            atom_array.set_annotation(
                "active_acceptor", np.zeros(atom_array.array_length(), dtype=bool)
            )
            data["atom_array"] = atom_array
            return data

        ## These are the only two use-cases we have so far. Can be extended as needed

        if data["sampled_condition_name"] == "ppi":
            selection1_chain_types = ["POLYPEPTIDE(D)", "POLYPEPTIDE(L)"]
            selection2_chain_types = ["POLYPEPTIDE(D)", "POLYPEPTIDE(L)"]
            separate_selections_for_motif_and_diffused = True
        else:
            selection1_chain_types = SELECTION_PROTEIN
            selection2_chain_types = SELECTION_NONPROTEIN
            separate_selections_for_motif_and_diffused = False

        selection1 = np.isin(atom_array.chain_type, selection1_chain_types)
        selection2 = np.isin(atom_array.chain_type, selection2_chain_types)

        # Optionally restrict to Hbonds between motif and diffused regions
        if separate_selections_for_motif_and_diffused:
            selection1 = selection1 & atom_array.is_motif_atom
            selection2 = selection2 & ~atom_array.is_motif_atom
        else:
            # Include fixed motif atoms for hbond calculations
            selection2 |= np.array(atom_array.is_motif_atom, dtype=bool)
            selection1 = ~selection2

        hbonds, hbond_types, atom_array = calculate_hbonds(
            atom_array,
            selection1=selection1,
            selection2=selection2,
            selection1_type=self.selection1_type,
            cutoff_dist=self.cutoff_dist,
            cutoff_angle=self.cutoff_angle,
            donor_elements=self.donor_elements,
            acceptor_elements=self.acceptor_elements,
            periodic=self.periodic,
        )

        # Initialize log_dict if not present
        data.setdefault("log_dict", {})
        log_dict = data["log_dict"]

        # Log hbond statistics
        log_dict["hbond_total_count"] = len(hbonds)
        log_dict["hbond_total_atoms"] = hbond_types.sum()

        # Subsample if hbond_subsample is set and number of atoms is bigger than 3
        final_hbond_types = hbond_types
        final_hbond_types[:, 0] = final_hbond_types[:, 0] * np.array(
            atom_array.is_motif_atom
        )
        final_hbond_types[:, 1] = final_hbond_types[:, 1] * np.array(
            atom_array.is_motif_atom
        )

        if data["conditions"]["hbond_subsample"] and np.sum(hbond_types) > 3:
            # Linear correlation: fewer hbonds = higher fraction
            base_fraction = 0.1  # minimum fraction (when many hbonds)
            max_fraction = 0.9  # maximum fraction (when few hbonds)
            n_hbonds = len(hbonds)
            max_hbonds = 50  # Expected maximum number of hbonds for scaling

            # Linear interpolation: fraction decreases linearly with number of hbonds
            fraction = max_fraction - (max_fraction - base_fraction) * min(
                n_hbonds / max_hbonds, 1.0
            )
            final_hbond_types = subsample_one_hot_np(hbond_types, fraction)

        # Set annotations and log subsample atoms
        atom_array.set_annotation("active_donor", final_hbond_types[:, 0])
        atom_array.set_annotation("active_acceptor", final_hbond_types[:, 1])
        log_dict["hbond_subsample_atoms"] = final_hbond_types.sum()

        # Remove hydrogens after processing
        atom_array = remove_hydrogens(atom_array)
        data["log_dict"] = log_dict
        data["atom_array"] = atom_array
        return data


def subsample_one_hot_np(array, fraction):
    """
    Subsamples a one-hot encoded NumPy array by randomly keeping a given fraction of the 1s.

    Args:
        array (np.ndarray): One-hot array of 0s and 1s.
        fraction (float): Fraction of 1s to keep (0 < fraction <= 1).

    Returns:
        np.ndarray: Subsampled array with same shape.
    """
    if not (0 < fraction <= 1):
        raise ValueError("Fraction must be in the range (0, 1].")

    array = array.copy()  # Don't modify original
    one_indices = np.argwhere(array == 1)
    num_ones = len(one_indices)

    keep_count = int(num_ones * fraction)

    # Shuffle and choose a subset of indices to keep
    np.random.shuffle(one_indices)
    keep_indices = one_indices[:keep_count]

    # Create new zero array
    new_array = np.zeros_like(array)

    # Set selected indices to 1
    for i, j in keep_indices:
        new_array[i, j] = 1

    return new_array
