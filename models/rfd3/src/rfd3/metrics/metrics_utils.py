import itertools

import numpy as np
from atomworks.ml.preprocessing.utils.structure_utils import (
    get_atom_mask_from_cell_list,
)
from atomworks.ml.utils.token import spread_token_wise
from biotite.structure import CellList, annotate_sse, gyration_radius
from rfd3.transforms.conditioning_base import get_motif_features


def get_ss_metrics_and_rg(
    atom_array, ss_conditioning: dict[str, np.ndarray] | None = None
):
    """Compute secondary structure metrics and the radius of gyration for a given input file.

    Args:
        atom_array (AtomArray): Input AtomArray
        ss_conditioning (dict[str, np.ndarray] | None): Dictionary mapping the keys "helix", "sheet", "loop" to the
            corresponding conditioning arrays. If None, secondary structure adherence is not computed.

    NOTE: Biotite computes secondary structures using the P-SEA algorithm:
    G. Labesse, N. Colloc'h, J. Pothier, J. Mornon,
    “P-SEA: a new efficient assignment of secondary structure from Ca trace of proteins,”
    Bioinformatics, vol. 13, pp. 291-295, June 1997. doi: 10.1093/bioinformatics/13.3.291
    """
    # Compute secondary structure
    sse_array = annotate_sse(atom_array)
    sse_array_prot_only = sse_array[sse_array != ""]

    # Basic compositional statistics
    pdb_helix_percent = np.mean(sse_array_prot_only == "a")
    pdb_strand_percent = np.mean(sse_array_prot_only == "b")
    pdb_coil_percent = np.mean(sse_array_prot_only == "c")
    pdb_ss_percent = pdb_helix_percent + pdb_strand_percent

    # Number of disjoint helices or sheets
    num_structural_elements = 0
    for k, _ in itertools.groupby(sse_array):
        if k not in ["", "c"]:
            num_structural_elements += 1

    if ss_conditioning is not None:
        ss_adherence_dict = {}
        atom_level_sse_array = spread_token_wise(atom_array, input_data=sse_array)
        for ss_annot, ss_type in zip(["a", "b", "c"], ["helix", "sheet", "loop"]):
            metric_name = f"{ss_type}_conditioning_adherence"
            expected_indices = np.where(ss_conditioning[ss_type])[0]

            if len(expected_indices) > 0:
                ss_adherence = (
                    atom_level_sse_array[expected_indices] == ss_annot
                ).mean()
                ss_adherence_dict[metric_name] = ss_adherence
            else:
                # Would be misleading to give a numerical value if no conditioning of this type was provided
                ss_adherence_dict[metric_name] = np.nan

    # Compute radius of gyration
    radius_of_gyration = gyration_radius(atom_array)

    # Return output metrics
    output_metrics = {
        "non_loop_fraction": pdb_ss_percent,
        "loop_fraction": pdb_coil_percent,
        "helix_fraction": pdb_helix_percent,
        "sheet_fraction": pdb_strand_percent,
        "num_ss_elements": num_structural_elements,
        "radius_of_gyration": radius_of_gyration,
    }

    if ss_conditioning is not None:
        output_metrics.update(ss_adherence_dict)

    return output_metrics


def _flatten_dict(d, parent="", sep="."):
    """
    Recursively flatten a nested dictionary.
    E.g:
        {"a": {"b": 1, "c": 2}} --> {"a.b": 1, "a.c": 2}
    """
    flat = {}
    for k, v in d.items():
        name = f"{parent}{sep}{k}" if parent else k
        if isinstance(v, dict):
            flat.update(_flatten_dict(v, name, sep=sep))
        else:
            flat[name] = v
    return flat


def get_hotspot_contacts(atom_array, hotspot_mask, distance_cutoff=4.5):
    """Get the number of inter-chain contacts between diffused atoms and hotspots within a distance cutoff."""

    cell_list = CellList(atom_array, cell_size=distance_cutoff)
    hotspot_array = atom_array[hotspot_mask]

    # Compute all contacts with hotspots
    full_contacting_atom_mask = get_atom_mask_from_cell_list(
        hotspot_array.coord, cell_list, len(atom_array), distance_cutoff
    )  # (n_hotspots, n_atoms)

    # We only count interchain contacts
    interchain_mask = hotspot_array.pn_unit_iid[:, None] != atom_array.pn_unit_iid[None]
    interchain_contacts_mask = full_contacting_atom_mask & interchain_mask

    # We only count contacts to diffused atoms
    diffused_interchain_contacts_mask = interchain_contacts_mask[
        :, ~get_motif_features(atom_array)["is_motif_atom"]
    ]

    contacted_hotspots_mask = np.any(
        diffused_interchain_contacts_mask, axis=1
    )  # (n_hotspots,)

    return float(contacted_hotspots_mask.mean())
