import logging
from typing import Literal

import biotite.structure as struc
import numpy as np
from atomworks.enums import ChainType
from atomworks.io.transforms.atom_array import remove_hydrogens
from rfd3.constants import (
    ATOM14_ATOM_NAMES,
    SELECTION_NONPROTEIN,
    SELECTION_PROTEIN,
    association_schemes_stripped,
)
from rfd3.transforms.hbonds import (
    add_hydrogen_atom_positions,
    calculate_hbonds,
)

from foundry.metrics.base import Metric
from foundry.utils.ddp import RankedLogger

logging.basicConfig(level=logging.INFO)
global_logger = RankedLogger(__name__, rank_zero_only=False)


def simplified_processing_atom_array(atom_arrays, central_atom="CB", threshold=0.5):
    """
    Allows for sequence extraction from cleaned up virtual atoms. Needed for hbond metrics.
    """
    final_atom_array = []
    for atom_array in atom_arrays:
        cur_atom_array_list = []

        res_ids = atom_array.res_id
        res_start_indices = np.concatenate(
            [[0], np.where(res_ids[1:] != res_ids[:-1])[0] + 1]
        )
        res_end_indices = np.concatenate([res_start_indices[1:], [len(res_ids)]])

        for start, end in zip(res_start_indices, res_end_indices):
            cur_res_atom_array = atom_array[start:end]

            # Check if the current residue is after padding (seq unknown):
            if_seq_known = not any(
                atom_name.startswith("V") for atom_name in cur_res_atom_array.atom_name
            )

            if not if_seq_known:
                # For Glycine: it doesn't have CB, so set the virtual atom as CA.
                # The current way to handle this is to check if predicted CA and CB are too close, because in the case of glycine and we pad virtual atoms based on CB, CB's coords are set as CA.
                # There might be a better way to do this.
                CA_coord = cur_res_atom_array.coord[
                    cur_res_atom_array.atom_name == "CA"
                ]
                CB_coord = cur_res_atom_array.coord[
                    cur_res_atom_array.atom_name == "CB"
                ]
                if np.linalg.norm(CA_coord - CB_coord) < threshold:
                    central_atom = "CA"

                central_mask = cur_res_atom_array.atom_name == central_atom

                # ... Calculate the distance to the central atom
                central_coord = cur_res_atom_array.coord[central_mask][
                    0
                ]  # Should only have one central atom anyway
                dists = np.linalg.norm(
                    cur_res_atom_array.coord - central_coord, axis=-1
                )

                # ... Select virtual atom by the distance. Shouldn't count the central atom itself.
                is_virtual = (dists < threshold) & ~central_mask

                cur_res_atom_array = cur_res_atom_array[~is_virtual]
                cur_pred_res_atom_names = (
                    cur_res_atom_array.atom_name
                )  # e.g. [N, CA, C, O, CB, V6, V2]

                has_restype_assigned = False
                for restype, atom_names in association_schemes_stripped[
                    "atom14"
                ].items():
                    atom_names = np.array(atom_names)
                    if restype in ["UNK", "MSK"]:
                        continue

                    atom_name_idx_in_atom14_scheme = np.array(
                        [
                            np.where(ATOM14_ATOM_NAMES == atom_name)[0][0]
                            for atom_name in cur_pred_res_atom_names
                        ]
                    )  # [0, 1, 2, 3, 4, 11, 7]
                    atom14_scheme_mask = np.zeros_like(ATOM14_ATOM_NAMES, dtype=bool)
                    atom14_scheme_mask[atom_name_idx_in_atom14_scheme] = True
                    if all(
                        x is not None for x in atom_names[atom14_scheme_mask]
                    ) and all(x is None for x in atom_names[~atom14_scheme_mask]):
                        cur_res_atom_array.res_name = np.array(
                            [restype] * len(cur_res_atom_array)
                        )
                        cur_res_atom_array.atom_name = np.asarray(
                            atom_names[atom14_scheme_mask], dtype=str
                        )
                        cur_atom_array_list.append(cur_res_atom_array)
                        has_restype_assigned = True
                        break
            else:
                cur_atom_array_list.append(cur_res_atom_array)
                has_restype_assigned = True

            if not has_restype_assigned:
                cur_res_atom_array.res_name = np.array(
                    ["UNK"] * len(cur_res_atom_array)
                )
                cur_atom_array_list.append(cur_res_atom_array)

        cur_atom_array = struc.concatenate(cur_atom_array_list)
        cur_atom_array.element = struc.infer_elements(cur_atom_array.atom_name)

        final_atom_array.append(cur_atom_array)

    return final_atom_array


# Training comparison
def calculate_hbond_stats(
    input_atom_array_stack,
    output_atom_array_stack,
    selection1,
    selection2,
    selection1_type,
    cutoff_dist,
    cutoff_angle,
    donor_elements,
    acceptor_elements,
    periodic,
):
    """
    Compare the number of hbonds correctly recapitualted in the output atom array.

    Args:
        input_atom_array_stack: Input atom array stack
        output_atom_array_stack: Output atom array stack
        selection1: Selection of atom types allowed to be donors (5,6)
        selection2: Selection of atom types allowed to be acceptors (1,2,3...)
        cutoff_dist: Cutoff distance for hbonds
        cutoff_angle: Cutoff angle for hbonds
    """
    # Used the latest function above, should check if it works correctly
    output_atom_array_stack = simplified_processing_atom_array(output_atom_array_stack)

    assert len(input_atom_array_stack) == len(
        output_atom_array_stack
    ), "Input and output atom arrays must have the same length"

    total_correct_donors_percent = 0.0
    total_correct_acceptors_percent = 0.0
    total_number_hbonds = 0
    num_valid_samples = 0
    for i in range(len(input_atom_array_stack)):
        correct_donors = 0
        correct_acceptors = 0

        input_atom_array = input_atom_array_stack[i]
        output_atom_array = output_atom_array_stack[i]

        if not (
            "active_donor" in input_atom_array.get_annotation_categories()
            or "active_acceptor" in input_atom_array.get_annotation_categories()
        ):
            # print("active donor/acceptor not in annotation")
            continue
        if np.sum(input_atom_array.active_donor == 0) and np.sum(
            input_atom_array.active_acceptor == 0
        ):
            continue

        # Select possible donors and acceptors for the model output
        if selection1 is None or selection2 is None:
            continue

        # Hack: Temporarily use biotite to infer bonds, should be replaced with cifutils?
        output_atom_array.bonds = struc.connect_via_distances(
            output_atom_array, default_bond_type=1
        )

        # Hack: delete coords_to_be_diffused (if exists) to temporarily solve a weird bug in create hydrogens. Anyway it will not be used.
        if "coord_to_be_noised" in input_atom_array.get_annotation_categories():
            input_atom_array.del_annotation("coord_to_be_noised")
        if "coord_to_be_noised" in output_atom_array.get_annotation_categories():
            output_atom_array.del_annotation("coord_to_be_noised")

        output_atom_array = add_hydrogen_atom_positions(output_atom_array)

        cur_selection1 = np.isin(output_atom_array.chain_type, selection1)
        cur_selection2 = (
            np.isin(output_atom_array.chain_type, selection2)
            | get_motif_features(output_atom_array)["is_motif_atom"]
        )

        hbonds, hbond_types, output_atom_array = calculate_hbonds(
            output_atom_array,
            cur_selection1,
            cur_selection2,
            selection1_type=selection1_type,
            cutoff_dist=cutoff_dist,
            cutoff_angle=cutoff_angle,
            donor_elements=donor_elements,
            acceptor_elements=acceptor_elements,
            periodic=periodic,
        )

        output_atom_array.set_annotation("active_donor", hbond_types[:, 0])
        output_atom_array.set_annotation("active_acceptor", hbond_types[:, 1])

        output_atom_array = remove_hydrogens(output_atom_array)

        given_hbond_donors = np.array(input_atom_array.active_donor, dtype=bool)
        given_hbond_acceptors = np.array(input_atom_array.active_acceptor, dtype=bool)
        given_hbond_donors_index = np.where(input_atom_array.active_donor == 1)[0]
        given_hbond_acceptors_index = np.where(input_atom_array.active_acceptor == 1)[0]

        # Ensure the produced hbonds matches input hbond requirements: have the same atom type, residue name, and atom name
        for idx in given_hbond_donors_index:
            if bool(
                output_atom_array[
                    (output_atom_array.chain_id == input_atom_array.chain_id[idx])
                    & (output_atom_array.res_id == input_atom_array.res_id[idx])
                    & (
                        output_atom_array.atom_name
                        == input_atom_array.gt_atom_name[idx]
                    )
                ].active_donor
            ):
                correct_donors += 1

        for idx in given_hbond_acceptors_index:
            if bool(
                output_atom_array[
                    (output_atom_array.chain_id == input_atom_array.chain_id[idx])
                    & (output_atom_array.res_id == input_atom_array.res_id[idx])
                    & (
                        output_atom_array.atom_name
                        == input_atom_array.gt_atom_name[idx]
                    )
                ].active_acceptor
            ):
                correct_acceptors += 1

        correct_hbond_donors_percent = (
            correct_donors / np.sum(given_hbond_donors)
            if np.sum(given_hbond_donors) > 0
            else 1.0
        )
        correct_hbond_acceptors_percent = (
            correct_acceptors / np.sum(given_hbond_acceptors)
            if np.sum(given_hbond_acceptors) > 0
            else 1.0
        )

        total_correct_donors_percent += correct_hbond_donors_percent
        total_correct_acceptors_percent += correct_hbond_acceptors_percent
        total_number_hbonds += len(hbonds)
        num_valid_samples += 1

    if num_valid_samples == 0:
        return 0, 0, 0
    return (
        total_correct_donors_percent / num_valid_samples,
        total_correct_acceptors_percent / num_valid_samples,
        total_number_hbonds / num_valid_samples,
    )


# Inference comparison -> tempportary fix to test out sm_hbonds, should be merged with hbond in transforms down the line
def get_hbond_metrics(atom_array=None):
    if atom_array is None:
        print("WARNING: atom_array is None")
        return None  # Or raise a more descriptive error

    curr_copy = atom_array.copy()
    o = {}
    selection1 = np.array([ChainType.as_enum(item).value for item in SELECTION_PROTEIN])
    selection2 = np.array(
        [ChainType.as_enum(item).value for item in SELECTION_NONPROTEIN]
    )
    # Hack: Temporarily use biotite to infer bonds, should be replaced with cifutils?
    curr_copy.bonds = struc.connect_via_distances(curr_copy, default_bond_type=1)
    # Hack: delete coords_to_be_diffused (if exists) to temporarily solve a weird bug in create hydrogens. Anyway it will not be used.
    if "coord_to_be_noised" in curr_copy.get_annotation_categories():
        curr_copy.del_annotation("coord_to_be_noised")

    try:
        curr_copy = add_hydrogen_atom_positions(curr_copy)
    except Exception as e:
        print("WARNING: problem adding hydrogen", e)

    if selection1 is not None:
        selection1 = np.isin(curr_copy.chain_type, selection1)
    else:
        selection1 = selection1
    if selection2 is not None:
        selection2 = np.isin(curr_copy.chain_type, selection2)
    else:
        selection2 = selection2

    # Always include fixed motif atoms for hbond calculations
    selection2 |= np.array(curr_copy.is_motif_atom, dtype=bool)
    selection1 = ~selection2

    hbonds, hbond_types, curr_copy = calculate_hbonds(
        curr_copy,
        selection1=selection1,
        selection2=selection2,
    )

    o["num_hbonds"] = int(len(hbonds))
    o["num_donors"] = int(np.sum(hbond_types[:, 0]))
    o["num_acceptors"] = int(np.sum(hbond_types[:, 1]))

    return o


class HbondMetrics(Metric):
    def __init__(
        self,
        selection1: list[str] = SELECTION_PROTEIN,
        selection2: list[str] = SELECTION_NONPROTEIN,
        selection1_type: Literal["acceptor", "donor", "both"] = "both",
        cutoff_dist: float = 3.0,
        cutoff_angle: float = 120.0,
        donor_elements: list[str] = ["N", "O", "S", "F"],
        acceptor_elements: list[str] = ["N", "O", "S", "F"],
        periodic: bool = False,
    ):
        super().__init__()

        self.selection1 = np.array(
            [ChainType.as_enum(item).value for item in selection1]
        )
        self.selection2 = np.array(
            [ChainType.as_enum(item).value for item in selection2]
        )

        self.selection1_type = selection1_type
        self.cutoff_dist = cutoff_dist
        self.cutoff_angle = cutoff_angle
        self.donor_elements = donor_elements
        self.acceptor_elements = acceptor_elements
        self.periodic = periodic

    @property
    def kwargs_to_compute_args(self):
        return {
            "ground_truth_atom_array_stack": ("ground_truth_atom_array_stack",),
            "predicted_atom_array_stack": ("predicted_atom_array_stack",),
        }

    def compute(self, *, ground_truth_atom_array_stack, predicted_atom_array_stack):
        try:
            (
                mean_correct_donors_percent,
                mean_correct_acceptors_percent,
                mean_num_hbonds,
            ) = calculate_hbond_stats(
                input_atom_array_stack=ground_truth_atom_array_stack,
                output_atom_array_stack=predicted_atom_array_stack,
                selection1=self.selection1,
                selection2=self.selection2,
                selection1_type=self.selection1_type,
                cutoff_dist=self.cutoff_dist,
                cutoff_angle=self.cutoff_angle,
                donor_elements=self.donor_elements,
                acceptor_elements=self.acceptor_elements,
                periodic=self.periodic,
            )
        except Exception as e:
            global_logger.error(
                f"Error calculating hydrogen bond metrics: {e} | Skipping"
            )
            return {}

        # Aggregate output for batch-level metrics
        o = {
            "mean_correct_donors_percent": float(mean_correct_donors_percent),
            "mean_correct_acceptors_percent": float(mean_correct_acceptors_percent),
            "mean_num_hbonds": float(mean_num_hbonds),
        }
        return o
