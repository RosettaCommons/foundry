"""
Utilities for inference input preparation
"""

import logging
import os
from os import PathLike
from typing import Dict

import biotite.structure as struc
import numpy as np
from atomworks import parse
from atomworks.constants import STANDARD_AA, STANDARD_DNA
from atomworks.io.parser import (
    STANDARD_PARSER_ARGS,
)
from atomworks.ml.encoding_definitions import AF3SequenceEncoding
from atomworks.ml.preprocessing.utils.structure_utils import (
    get_atom_mask_from_cell_list,
)
from atomworks.ml.utils.token import (
    get_token_starts,
    spread_token_wise,
)
from rfd3.constants import (
    REQUIRED_CONDITIONING_ANNOTATIONS,
)
from rfd3.transforms.conditioning_base import (
    convert_existing_annotations_to_bool,
    set_default_conditioning_annotations,
)
from rfd3.transforms.conditioning_utils import sample_island_tokens

from foundry.common import exists
from foundry.utils.components import (
    fetch_mask_from_component,
    get_name_mask,
    unravel_components,
)
from foundry.utils.ddp import RankedLogger

logging.basicConfig(level=logging.INFO)
ranked_logger = RankedLogger(__name__, rank_zero_only=True)

sequence_encoding = AF3SequenceEncoding()
_aa_like_res_names = sequence_encoding.all_res_names[sequence_encoding.is_aa_like]


#################################################################################
# Setter functions for annotations
#################################################################################


def set_common_annotations(array, set_src_component_to_res_name=True):
    annots = array.get_annotation_categories()
    if "occupancy" not in annots:
        array.set_annotation("occupancy", np.ones(array.shape[0], dtype=float))
    if "b_factor" not in annots:
        array.set_annotation("b_factor", np.zeros(array.shape[0], dtype=float))
    if "charge" not in annots:
        array.set_annotation("charge", np.zeros(array.shape[0], dtype=float))
    if "src_component" not in annots:
        if set_src_component_to_res_name:
            array.set_annotation(
                "src_component",
                np.full(
                    array.shape[0], array.res_name.copy(), dtype=array.res_name.dtype
                ),
            )
        else:
            array.set_annotation(
                "src_component", np.full(array.shape[0], "", dtype=array.res_name.dtype)
            )
    return array


def set_indices(array, chain, res_id_start, molecule_id, component):
    n = array.shape[0]
    array.chain_id = np.full(n, chain, dtype=array.chain_id.dtype)
    array.res_id = np.full(n, res_id_start + array.res_id - 1, dtype=array.res_id.dtype)
    array.molecule_id = np.full(n, molecule_id, dtype=np.int32)
    array.set_annotation(
        "src_component", np.full(n, component, dtype=array.chain_id.dtype)
    )
    return array


#################################################################################
# Getters
#################################################################################


def extract_ligand_array(
    atom_array_input,
    ligand,
    fixed_atoms={},
    set_defaults=True,
    additional_annotations=None,
):
    if not exists(atom_array_input):
        raise ValueError(
            "No input file/atom array provided. Cannot add requested ligand."
        )

    ligand_arrays = []
    for lig in ligand.split(","):
        for name in unravel_components(
            lig, atom_array=atom_array_input, allow_multiple_matches=True
        ):  # additional nesting to allow multiple indices per ligand
            mask = fetch_mask_from_component(name, atom_array=atom_array_input)
            ligand_array = atom_array_input[mask].copy()

            # ... Set as fully fixed motif
            if set_defaults:
                ligand_array = set_default_conditioning_annotations(
                    ligand_array, motif=True, additional=additional_annotations
                )  # should be pre-set!
            ligand_array = set_common_annotations(ligand_array)

            # ... Unfix all names not specified if specified in motif_atoms
            if lig in fixed_atoms or name in fixed_atoms:
                if (lig in fixed_atoms and name in fixed_atoms) and name != lig:
                    raise ValueError(
                        f"Got both ligand name and its pdb indices in fixed_atoms dictionary: {lig} and {name}. Please only provide one."
                    )
                fixed = fixed_atoms.get(lig, fixed_atoms.get(name, None))
                if fixed:
                    fixed_mask = get_name_mask(ligand_array.atom_name, fixed)
                    ligand_array.is_motif_atom_with_fixed_coord[~fixed_mask] = np.zeros(
                        np.sum(~fixed_mask), dtype=int
                    )
                else:
                    ligand_array.is_motif_atom_with_fixed_coord = np.zeros(
                        ligand_array.shape[0], dtype=int
                    )
            ligand_arrays.append(ligand_array)

    ligand_arrays = struc.concatenate(ligand_arrays)
    return ligand_arrays


def extract_na_array(atom_array_input):
    # TODO : do it more nicely, take into account modifications to NA reses e.g. 5IU
    if (na_mask := np.isin(atom_array_input.res_name, list(STANDARD_DNA))).any():
        na_array = atom_array_input[na_mask]
        # ...replace chain_id A with literally anything else available
        Achain_mask = na_array.chain_id == "A"

        all_nonAchains = np.unique((atom_array_input + na_array).chain_id).tolist()
        all_nonAchains.remove("A")

        if len(all_nonAchains) > 1:
            new_chain = "".join(all_nonAchains)  # join_them_all !! so definitely unique
        elif len(all_nonAchains) == 1:
            new_chain = all_nonAchains[0] + all_nonAchains[0]
        else:
            new_chain = "B"

        na_array.chain_id[Achain_mask] = new_chain
        na_array = set_default_conditioning_annotations(na_array, motif=True)
        return na_array
    else:
        raise ValueError(
            "Could not find any NA tokens in input file, but requested to add all NA"
        )


def _restore_bonds_for_nonstandard_residues(
    atom_array_accum: struc.AtomArray,
    src_atom_array: struc.AtomArray | None,
    source_to_accum_idx: Dict[int, int],
) -> struc.AtomArray:
    """
    Restores and creates bonds for non-standard residues (PTMs, modified AAs, etc.)
    from source structure and between consecutive residues.
    This function:
    1. Preserves inter-residue bonds from the source structure (if available)
    2. Adds backbone C-N bonds between consecutive residues where at least one is non-standard
    Args:
        atom_array_accum: The accumulated atom array to add bonds to
        src_atom_array: The source atom array containing original bond information
        source_to_accum_idx: Mapping from source atom indices to accumulated array indices
    Returns:
        atom_array_accum with bonds added
    """
    # Initialize bonds if needed
    if atom_array_accum.bonds is None:
        atom_array_accum.bonds = struc.BondList(atom_array_accum.array_length())

    # Step 1: Restore inter-residue bonds from the source atom array (only for non-standard residues)
    if (
        src_atom_array is not None
        and hasattr(src_atom_array, "bonds")
        and src_atom_array.bonds is not None
    ):
        original_bonds = src_atom_array.bonds.as_array()
        if len(original_bonds) > 0:
            # Extract bonds where both atoms are in the accumulated array
            bonds_to_add = []
            for bond in original_bonds:
                atom_i, atom_j, bond_type = bond
                # Check if both atoms are in our mapping
                if (
                    int(atom_i) in source_to_accum_idx
                    and int(atom_j) in source_to_accum_idx
                ):
                    # Check if at least one atom is from a non-standard residue
                    src_res_i = src_atom_array[int(atom_i)].res_name
                    src_res_j = src_atom_array[int(atom_j)].res_name

                    # Only preserve if at least one residue is non-standard
                    if src_res_i not in STANDARD_AA or src_res_j not in STANDARD_AA:
                        new_i = source_to_accum_idx[int(atom_i)]
                        new_j = source_to_accum_idx[int(atom_j)]
                        bonds_to_add.append([new_i, new_j, int(bond_type)])

            if bonds_to_add:
                # Add the preserved bonds
                new_bonds = struc.BondList(
                    atom_array_accum.array_length(),
                    np.array(bonds_to_add, dtype=np.int64),
                )
                atom_array_accum.bonds = atom_array_accum.bonds.merge(new_bonds)
                logger.info(
                    f"Preserved {len(bonds_to_add)} inter-residue bonds involving non-standard residues from source structure"
                )

    # Step 2: Add backbone bonds between consecutive residues where at least one is non-standard
    # This handles: PTM-to-diffused, diffused-to-PTM, PTM-to-PTM, ligand-to-protein
    bonds_to_add = []

    # Group by residue
    token_starts = get_token_starts(atom_array_accum, add_exclusive_stop=True)

    for i in range(
        len(token_starts) - 2
    ):  # -2 because we need pairs and token_starts has exclusive stop
        curr_start, curr_end = token_starts[i], token_starts[i + 1]
        next_start, next_end = token_starts[i + 1], token_starts[i + 2]

        curr_residue = atom_array_accum[curr_start:curr_end]
        next_residue = atom_array_accum[next_start:next_end]

        # Check if at least one residue is non-standard (PTMs, modified AAs, etc.)
        curr_is_nonstandard = curr_residue.res_name[0] not in STANDARD_AA
        next_is_nonstandard = next_residue.res_name[0] not in STANDARD_AA

        # Only add bonds if at least one residue is non-standard
        if not (curr_is_nonstandard or next_is_nonstandard):
            continue

        # Check if consecutive in same chain
        if curr_residue.chain_id[0] != next_residue.chain_id[0]:
            continue
        if next_residue.res_id[0] - curr_residue.res_id[0] != 1:
            continue

        # Find C atom in current residue (C-terminus connection point)
        c_mask = curr_residue.atom_name == "C"
        if not np.any(c_mask):
            # If a non-standard residue doesn't have a C atom, it can't connect to next residue
            # This is expected for some atomized residues or ligands at chain termini
            if curr_is_nonstandard and next_is_nonstandard:
                # Both are non-standard but no C in current - might be an atomized region without proper termini
                logger.debug(
                    f"Non-standard residue {curr_residue.res_name[0]} (res_id {curr_residue.res_id[0]}) "
                    f"has no C atom - cannot form backbone bond to next residue"
                )
            continue
        c_idx = curr_start + np.where(c_mask)[0][0]

        # Find N atom in next residue (N-terminus connection point)
        n_mask = next_residue.atom_name == "N"
        if not np.any(n_mask):
            # If a non-standard residue doesn't have an N atom, it can't connect to previous residue
            # This is expected for some atomized residues or ligands at chain termini
            if curr_is_nonstandard and next_is_nonstandard:
                # Both are non-standard but no N in next - might be an atomized region without proper termini
                logger.debug(
                    f"Non-standard residue {next_residue.res_name[0]} (res_id {next_residue.res_id[0]}) "
                    f"has no N atom - cannot form backbone bond from previous residue"
                )
            continue
        n_idx = next_start + np.where(n_mask)[0][0]

        # Check if this bond already exists (from source preservation)
        existing_bonds = atom_array_accum.bonds.as_array()
        bond_exists = False
        if len(existing_bonds) > 0:
            for existing_bond in existing_bonds:
                if (existing_bond[0] == c_idx and existing_bond[1] == n_idx) or (
                    existing_bond[0] == n_idx and existing_bond[1] == c_idx
                ):
                    bond_exists = True
                    break

        if not bond_exists:
            bonds_to_add.append([c_idx, n_idx, struc.BondType.SINGLE])

    if bonds_to_add:
        new_bonds = struc.BondList(
            atom_array_accum.array_length(), np.array(bonds_to_add, dtype=np.int64)
        )
        atom_array_accum.bonds = atom_array_accum.bonds.merge(new_bonds)
        logger.info(
            f"Added {len(bonds_to_add)} backbone bonds involving non-standard residues"
        )

    return atom_array_accum


#################################################################################
# File IO utilities
#################################################################################


def inference_load_(
    file: PathLike, *, assembly_id: str = "1", cif_parser_args: dict | None = None
):
    # Default cif_parser_args to an empty dictionary if not provided
    if cif_parser_args is None:
        cif_parser_args = {}

    # Convenience utilities to default to loading from and saving to cache if a cache_dir is provided, unless explicitly overridden
    if "cache_dir" in cif_parser_args and cif_parser_args["cache_dir"]:
        cif_parser_args.setdefault("load_from_cache", True)
        cif_parser_args.setdefault("save_to_cache", True)

    merged_cif_parser_args = {
        **STANDARD_PARSER_ARGS,
        **{
            "fix_arginines": False,
            "add_missing_atoms": False,
            "remove_ccds": [],
        },
        **cif_parser_args,
    }
    merged_cif_parser_args["hydrogen_policy"] = "remove"

    # Ensure the required annotations can be loaded
    merged_cif_parser_args["extra_fields"] = list(
        set(
            merged_cif_parser_args.get("extra_fields", [])
            + REQUIRED_CONDITIONING_ANNOTATIONS
        )
    )

    # Use the parse function with the merged CIF parser arguments
    result_dict = parse(
        filename=file,
        build_assembly=(assembly_id,),  # Convert list to tuple (make hashable)
        **merged_cif_parser_args,
    )

    atom_array = result_dict["assemblies"][assembly_id][0]
    atom_array = convert_existing_annotations_to_bool(atom_array)

    data = {
        "atom_array": atom_array,  # First model
        "chain_info": result_dict["chain_info"],
        "ligand_info": result_dict["ligand_info"],
        "metadata": result_dict["metadata"],
    }

    return data


def ensure_input_is_abspath(args: dict, path: PathLike | None):
    """
    Ensures the input source is an absolute path if exists, if not it will convert

    args:
        spec: Inference specification for atom array
        path: None or file to which the input is relative to.
    """
    if isinstance(args, str):
        raise ValueError(
            "Expected args to be a dictionary, got a string: {}. If you are using an input JSON ensure it contains dictionaries of arguments".format(
                args
            )
        )
    if "input" not in args or not exists(args["input"]):
        return args
    input = args["input"]
    if not os.path.isabs(input):
        input = os.path.abspath(os.path.join(os.path.dirname(path), input))
        ranked_logger.info(
            f"Input source path is relative, converted to absolute path: {input}"
        )
        args["input"] = input
    return args


#################################################################################
# Custom infer_ori functions
#################################################################################


def infer_ori_from_hotspots(atom_array: struc.AtomArray):
    assert (
        "is_atom_level_hotspot" in atom_array.get_annotation_categories()
    ), "Atom array must contain 'is_atom_level_hotspot' annotation to infer ori from hotspots."
    hotspot_atom_array = atom_array[atom_array.is_atom_level_hotspot.astype(bool)]
    hotspot_com = hotspot_atom_array.coord.mean(axis=0)

    # We can only perform distance computations on atoms with non-NaN coordinates
    nan_coords_mask = np.any(np.isnan(atom_array.coord), axis=1)
    non_nan_atom_array = atom_array[~nan_coords_mask]

    # Perform the distance computation
    # RFD2 used 10 Angstroms instead of 12, but was for residue-level hotspots
    DISTANCE_CUTOFF = 12.0
    cell_list = struc.CellList(non_nan_atom_array, cell_size=DISTANCE_CUTOFF)
    nearby_atoms_mask = get_atom_mask_from_cell_list(
        hotspot_atom_array.coord,
        cell_list,
        len(non_nan_atom_array),
        cutoff=DISTANCE_CUTOFF,
    )  # (n_query, n_cell_list)

    nearby_atoms_mask = np.any(nearby_atoms_mask, axis=0)  # (n_cell_list,)
    nearby_atoms_com = non_nan_atom_array.coord[nearby_atoms_mask].mean(axis=0)

    vector_from_core_to_hotspot = hotspot_com - nearby_atoms_com
    vector_from_core_to_hotspot = vector_from_core_to_hotspot / np.linalg.norm(
        vector_from_core_to_hotspot
    )

    # This is following RFD2. Both this and the distance cutoff should definitely be configs with defaults
    DISTANCE_ABOVE_HOTSPOTS = 10.0
    ori_token = hotspot_com + DISTANCE_ABOVE_HOTSPOTS * vector_from_core_to_hotspot

    return ori_token


def infer_ori_from_com(atom_array):
    xyz = atom_array.coord
    mask = np.isfinite(xyz).all(axis=-1)  # Ensure no NaN coordinates
    com = np.mean(xyz[..., mask, :], axis=0)
    return com


# This can't go in constants.py because that leads to a circular dependency
INFER_ORI_STRATEGIES_TO_FUNCTIONS = {
    "hotspots": infer_ori_from_hotspots,
    "com": infer_ori_from_com,
}
"""
Constant mapping from infer_ori_strategy keys to the corresponding functions. These functions should take an AtomArray
as input and return a three-element list or numpy array of floats.
"""


def set_com(
    atom_array, ori_token: list | None = None, infer_ori_strategy: str | None = None
):
    if exists(ori_token):
        center = np.array([float(x) for x in ori_token], dtype=atom_array.coord.dtype)
        atom_array.coord = atom_array.coord - center
        ranked_logger.info(f"Received ori_token argument. Setting origin as {center}.")
        if infer_ori_strategy is not None:
            ranked_logger.warning(
                f"Specified infer_ori_strategy of '{infer_ori_strategy}' will be ignored because an ori_token was provided."
            )
    elif "ORI" in atom_array.res_name:
        center = atom_array[atom_array.res_name == "ORI"].coord
        if center.shape[0] != 1:
            center = np.random.choice(center, size=1, replace=False)
            ranked_logger.info(f"Found multiple ORI tokens in input. Sampled: {center}")
        center = np.nan_to_num(center.squeeze())
        atom_array.coord = atom_array.coord - center
        ranked_logger.info(
            f"Found ORI token in input. Setting origin as token value ({center})."
        )
        if infer_ori_strategy is not None:
            ranked_logger.warning(
                f"Specified infer_ori_strategy of '{infer_ori_strategy}' will be ignored because an ori_token was provided."
            )
    elif infer_ori_strategy is not None:
        if infer_ori_strategy in INFER_ORI_STRATEGIES_TO_FUNCTIONS:
            center = INFER_ORI_STRATEGIES_TO_FUNCTIONS[infer_ori_strategy](atom_array)
            atom_array.coord = atom_array.coord - center
            ranked_logger.info(
                f"Inferred origin using strategy '{infer_ori_strategy}'. Setting origin as {center}."
            )
    else:
        # No offset
        if np.any(atom_array.is_motif_atom_with_fixed_coord.astype(bool)):
            center = np.nan_to_num(
                np.mean(
                    atom_array.coord[
                        atom_array.is_motif_atom_with_fixed_coord.astype(bool)
                    ],
                    axis=0,
                )
            )
            ranked_logger.info(
                f"No ori_token or infer_ori_strategy provided. Setting origin as COM of fixed motif ({center})."
            )
            atom_array.coord -= center
        else:
            ranked_logger.warning(
                "No ori_token, infer_ori_strategy, or motif provided. Setting [0,0,0] as origin."
            )
            atom_array.coord = np.zeros_like(
                atom_array.coord, dtype=atom_array.coord.dtype
            )
    return atom_array


#################################################################################
# Custom conditioning functions
#################################################################################


def spoof_helical_bundle_ss_conditioning_fn(atom_array: struc.AtomArray):
    # NOTE: This assumes that all diffused residues are protein residues -- should be updated if that changes!
    # Compute islands within the subset that is diffused and has secondary structure types.
    token_level_array = atom_array[get_token_starts(atom_array)]
    is_diffused_atom_token_level = ~(
        token_level_array.is_motif_atom_with_fixed_coord.astype(bool)
    )

    # My reason for sampling from 3-7 is that I don't want to restrict the model too heavily since this is
    # indexed to specific residues, and it will likely extend helices to reasonable lengths once it has started them.
    where_to_show_helices = sample_island_tokens(
        is_diffused_atom_token_level.sum(),
        island_len_min=3,
        island_len_max=7,
        n_islands_min=1,
        n_islands_max=3,
        max_length=None,
    )

    # Convert this to a mask over the entire token-level atom array
    token_level_helix_mask = np.zeros(token_level_array.array_length(), dtype=bool)
    token_level_helix_mask[is_diffused_atom_token_level] = where_to_show_helices

    # I don't want to sample very near the tails, as this gets too restrictive for the model
    for chain_id in np.unique(token_level_array.chain_id):
        chain_mask = token_level_array.chain_id == chain_id
        chain_indices = np.where(chain_mask)[0]
        chain_start, chain_end = chain_indices[0], chain_indices[-1] + 1
        chain_length = chain_mask.sum()

        buffer_length = chain_length // 8
        buffer_mask = chain_mask.copy()
        buffer_mask[chain_start + buffer_length : chain_end - buffer_length] = False

        token_level_helix_mask[buffer_mask] = False

    helix_conditioning = np.zeros(atom_array.array_length())
    helix_condition_mask = spread_token_wise(atom_array, token_level_helix_mask)

    helix_conditioning[helix_condition_mask] = 1
    return helix_conditioning


#################################################################################
# Patching of bad inputs
#################################################################################


def generate_idealized_cb_position(N: np.array, Ca: np.array, C: np.array) -> np.array:
    """
    Generate Cb coordiantes given (N, CA, C) as if the given coordinates were from an idealized Alanine.

    Args:
        - N (np.array): coordinates of (pseudo) N atoms [..., L, 3]
        - Ca (np.array): coordinates of (pseudo) Ca atoms [..., L, 3]
        - C (np.array): coordinates of (pseudo) C atoms [..., L, 3]

    Returns:
        Cb (torch.Tensor): coordinates of (pseudo) Cb atoms [..., L, 3]
            These will be placed at the idealized Cb distance (based on ALA) from Ca, assuming a frame of the following form:
                - x-axis: along the Ca-C bond
                - z-axis: perpendicular to the Ca-N-C plane, right-handed wrt to (Ca-C) & (Ca-N) vectors.
                - y-axis: in the plane of the Ca-N-C bonds, such that the overall frame is right-handed.
    Reference:
        - https://github.com/google-deepmind/alphafold/blob/d95a92aae161240b645fc10e9d030443011d913e/alphafold/common/residue_constants.py#L126-L335
            ALA:
                ['N',  0, (-0.525,  1.363,  0.000)],  # ca-n bond dist: 1.4606142543
                ['CA', 0, ( 0.000,  0.000,  0.000)],
                ['C',  0, ( 1.526,  0.000,  0.000)],  # ca-c bond dist: 1.526
                ['CB', 0, (-0.529, -0.774, -1.205)],  # cb-ca bond dist: 1.5267422834
    """
    if np.linalg.norm(N) == 0 and np.linalg.norm(C) == 0 and np.linalg.norm(Ca) == 0:
        return np.zeros_like(N)

    def _safe_normalize(vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype=float)
        norms = np.linalg.norm(vec, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vec / norms

    normalize = _safe_normalize

    # ... get local frame x-axis
    to_C = C - Ca
    frame_x = normalize(to_C)

    # ... get local frame z-axis
    to_N = N - Ca
    to_out_of_plane = np.cross(frame_x, normalize(to_N), axis=-1)
    frame_z = normalize(to_out_of_plane)

    # ... get local frame y-axis
    frame_y = normalize(np.cross(frame_z, frame_x, axis=-1))

    # ... place virtual Cb at the desired location
    Cb = Ca + (-0.529 * frame_x - 0.774 * frame_y - 1.205 * frame_z)
    return Cb


def create_cb_atoms(array):
    # array of length 4 with N, CA, C, O
    # Returns array with CB placed ideally
    if array.atom_name.tolist() != ["N", "CA", "C", "O"]:
        raise ValueError(
            "Input array must contain exactly 4 atoms: N, CA, C, O. Got : {}".format(
                array.atom_name.tolist()
            )
        )
    cb_atoms = array[array.atom_name == "CA"].copy()
    cb_atoms.atom_name = np.array(["CB"], dtype=cb_atoms.atom_name.dtype)
    cb_pos = generate_idealized_cb_position(
        array.coord[array.atom_name == "N"].squeeze(),
        array.coord[array.atom_name == "CA"].squeeze(),
        array.coord[array.atom_name == "C"].squeeze(),
    )
    cb_atoms.coord = cb_pos[None]
    return cb_atoms


def create_o_atoms(array):
    if array.atom_name.tolist() != ["N", "CA", "C"]:
        raise ValueError(
            "Input array must contain exactly 4 atoms: N, CA, C, O. Got : {}".format(
                array.atom_name.tolist()
            )
        )

    ca_atoms = array[array.atom_name == "CA"].copy()
    ca_atoms.atom_name = np.array(["O"], dtype=ca_atoms.atom_name.dtype)
    ca_atoms.element = np.array(["O"], dtype=ca_atoms.element.dtype)
    ca_atoms.coord = array.coord[array.atom_name == "C"].squeeze()[None]

    return ca_atoms
