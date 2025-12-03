import logging

import biotite.structure as struc
import numpy as np
from rfd3.constants import (
    ATOM14_ATOM_NAMES,
    association_schemes_stripped,
)
from rfd3.transforms.conditioning_base import get_motif_features
from rfd3.transforms.hbonds_hbplus import calculate_hbonds

from foundry.metrics.metric import Metric
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
                # Glycine fallback to CA
                CA_coord = cur_res_atom_array.coord[
                    cur_res_atom_array.atom_name == "CA"
                ]
                CB_coord = cur_res_atom_array.coord[
                    cur_res_atom_array.atom_name == "CB"
                ]

                if np.linalg.norm(CA_coord - CB_coord) < threshold:
                    central_atom = "CA"

                central_mask = cur_res_atom_array.atom_name == central_atom
                central_coord = cur_res_atom_array.coord[central_mask][0]
                dists = np.linalg.norm(
                    cur_res_atom_array.coord - central_coord, axis=-1
                )
                is_virtual = (dists < threshold) & ~central_mask

                cur_res_atom_array = cur_res_atom_array[~is_virtual]
                cur_pred_res_atom_names = cur_res_atom_array.atom_name

                has_restype_assigned = False
                for restype, atom_names in association_schemes_stripped[
                    "atom14"
                ].items():
                    if restype in ["UNK", "MSK"]:
                        continue
                    atom_names = np.array(atom_names)
                    atom_name_idx = np.array(
                        [
                            np.where(ATOM14_ATOM_NAMES == atom_name)[0][0]
                            for atom_name in cur_pred_res_atom_names
                        ]
                    )
                    atom14_mask = np.zeros_like(ATOM14_ATOM_NAMES, dtype=bool)
                    atom14_mask[atom_name_idx] = True

                    if all(x is not None for x in atom_names[atom14_mask]) and all(
                        x is None for x in atom_names[~atom14_mask]
                    ):
                        cur_res_atom_array.res_name = np.array(
                            [restype] * len(cur_res_atom_array)
                        )
                        cur_res_atom_array.atom_name = np.asarray(
                            atom_names[atom14_mask], dtype=str
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


def calculate_hbond_stats(
    input_atom_array_stack,
    output_atom_array_stack,
    cutoff_HA_dist=3,
    cutoff_DA_distance=3.5,
    inference_metrics=False,
):
    output_atom_array_stack = simplified_processing_atom_array(output_atom_array_stack)
    assert len(input_atom_array_stack) == len(output_atom_array_stack)

    total_correct_donors_percent = 0.0
    total_correct_acceptors_percent = 0.0
    total_number_donors_acceptors = 0
    total_number_hbonds = 0
    num_valid_samples = 0

    for input_atom_array, output_atom_array in zip(
        input_atom_array_stack, output_atom_array_stack
    ):
        # Ensure required annotations exist
        for annotation in ["active_donor", "active_acceptor"]:
            if annotation not in input_atom_array.get_annotation_categories():
                input_atom_array.set_annotation(
                    annotation, np.zeros(len(input_atom_array), dtype=bool)
                )

        # Skip samples with no donors or acceptors
        if (
            np.sum(input_atom_array.active_donor) == 0
            and np.sum(input_atom_array.active_acceptor) == 0
        ):
            continue

        # Clean up coordinate annotations
        for atom_array in [input_atom_array, output_atom_array]:
            if "coord_to_be_noised" in atom_array.get_annotation_categories():
                atom_array.del_annotation("coord_to_be_noised")

        # Calculate hydrogen bonds
        output_atom_array, hbonds, motif_diffused_hbond_count = calculate_hbonds(
            output_atom_array,
        )

        # Update hbond annotations for motif atoms only
        hbond_types = np.vstack(
            (output_atom_array.active_donor, output_atom_array.active_acceptor)
        ).T
        motif_mask = np.array(get_motif_features(output_atom_array)["is_motif_atom"])
        hbond_types[:, 0] *= motif_mask
        hbond_types[:, 1] *= motif_mask

        output_atom_array.set_annotation("active_donor", hbond_types[:, 0])
        output_atom_array.set_annotation("active_acceptor", hbond_types[:, 1])

        # Count correct predictions
        correct_donors = _count_correct_hbond_atoms(
            input_atom_array, output_atom_array, "active_donor"
        )
        correct_acceptors = _count_correct_hbond_atoms(
            input_atom_array, output_atom_array, "active_acceptor"
        )

        # Calculate percentages
        given_donors = np.sum(input_atom_array.active_donor)
        given_acceptors = np.sum(input_atom_array.active_acceptor)

        correct_donor_pct = correct_donors / given_donors if given_donors > 0 else 1.0
        correct_acceptor_pct = (
            correct_acceptors / given_acceptors if given_acceptors > 0 else 1.0
        )

        # Accumulate totals
        total_correct_donors_percent += correct_donor_pct
        total_correct_acceptors_percent += correct_acceptor_pct
        total_number_donors_acceptors += np.sum(hbond_types)
        total_number_hbonds += motif_diffused_hbond_count
        num_valid_samples += 1

    if num_valid_samples == 0:
        if inference_metrics:
            return {
                "correct_donor_percent": "",
                "correct_acceptor_percent": "",
                "num_hbonds": "",
                "hbonds": [],
                "total_number_donors_acceptors": "",
                "output_atom_array": None,
            }
        return 0, 0, 0

    avg_donor_pct = total_correct_donors_percent / num_valid_samples
    avg_acceptor_pct = total_correct_acceptors_percent / num_valid_samples
    avg_hbonds = total_number_hbonds / num_valid_samples

    if inference_metrics:
        return {
            "correct_donor_percent": avg_donor_pct,
            "correct_acceptor_percent": avg_acceptor_pct,
            "num_hbonds": avg_hbonds,
            "hbonds": hbonds,
            "total_number_donors_acceptors": total_number_donors_acceptors,
            "output_atom_array": output_atom_array,
        }

    # Return results
    if num_valid_samples == 0:
        return 0, 0, 0

    return avg_donor_pct, avg_acceptor_pct, avg_hbonds


def _count_correct_hbond_atoms(input_atom_array, output_atom_array, annotation_type):
    """Count correctly predicted hydrogen bond atoms."""
    correct_count = 0
    target_indices = np.where(getattr(input_atom_array, annotation_type) == 1)[0]

    for idx in target_indices:
        matching_atoms = output_atom_array[
            (output_atom_array.chain_iid == input_atom_array.chain_iid[idx])
            & (output_atom_array.res_id == input_atom_array.res_id[idx])
            & (output_atom_array.atom_name == input_atom_array.gt_atom_name[idx])
        ]

        if len(matching_atoms) > 0 and bool(getattr(matching_atoms, annotation_type)):
            correct_count += 1

    return correct_count


def get_hbond_metrics(atom_array=None):
    if atom_array is None:
        global_logger.warning("atom_array is None")
        return None

    try:
        output = calculate_hbond_stats(
            [atom_array.copy()], [atom_array.copy()], inference_metrics=True
        )
        hbonds = output["hbonds"]

        o = {
            "donor_atom_names": list(
                set(f"{hb['d_atom']}_{hb['d_resn']}_{hb['d_resi']}" for hb in hbonds)
            ),
            "acceptor_atom_names": list(
                set(f"{hb['a_atom']}_{hb['a_resn']}_{hb['a_resi']}" for hb in hbonds)
            ),
            "hbond_connections": list(
                set(
                    f"{hb['d_atom']}_{hb['d_resn']}_{hb['d_resi']}-{hb['a_atom']}_{hb['a_resn']}_{hb['a_resi']}"
                    for hb in hbonds
                )
            ),
            "correct_donor_percent": float(output["correct_donor_percent"]),
            "correct_acceptor_percent": float(output["correct_acceptor_percent"]),
            "num_hbonds": float(output["num_hbonds"]),
        }
        return o

    except Exception as e:
        global_logger.warning(f"Could not calculate hbond metrics: {e}")
        return {}


class HbondMetrics(Metric):
    def __init__(
        self,
        cutoff_HA_dist: float = 3,
        cutoff_DA_distance: float = 3.5,
    ):
        super().__init__()
        self.cutoff_HA_dist = cutoff_HA_dist
        self.cutoff_DA_distance = cutoff_DA_distance

    @property
    def kwargs_to_compute_args(self):
        return {
            "ground_truth_atom_array_stack": ("ground_truth_atom_array_stack",),
            "predicted_atom_array_stack": ("predicted_atom_array_stack",),
        }

    def compute(self, *, ground_truth_atom_array_stack, predicted_atom_array_stack):
        try:
            d_pct, a_pct, n_hbonds = calculate_hbond_stats(
                input_atom_array_stack=ground_truth_atom_array_stack,
                output_atom_array_stack=predicted_atom_array_stack,
                cutoff_HA_dist=self.cutoff_HA_dist,
                cutoff_DA_distance=self.cutoff_DA_distance,
            )
        except Exception as e:
            global_logger.error(
                f"Error calculating hydrogen bond metrics: {e} | Skipping"
            )
            return {}

        return {
            "mean_correct_donors_percent": float(d_pct),
            "mean_correct_acceptors_percent": float(a_pct),
            "mean_num_hbonds": float(n_hbonds),
        }
