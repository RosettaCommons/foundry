import random
import re
from typing import List

import numpy as np
from atomworks.ml.encoding_definitions import AF3SequenceEncoding
from biotite.structure import AtomArray
from rfd3.constants import (
    TIP_BY_RESTYPE,
)

from foundry.common import exists
from foundry.utils.ddp import RankedLogger

global_logger = RankedLogger(__name__, rank_zero_only=False)
sequence_encoding = AF3SequenceEncoding()
_aa_like_res_names = sequence_encoding.all_res_names[sequence_encoding.is_aa_like]


#################################################################################
# Component / contig parsing
#################################################################################


class ComponentValidationError(ValueError):
    """Raised when contig/component inputs cannot be parsed or validated."""

    def __init__(
        self,
        message: str,
        *,
        component: str | None = None,
        details: dict | None = None,
    ):
        self.component = component
        self.details = details or {}
        prefix = f"[component={component}] " if component else ""
        suffix = f" Details: {self.details}" if self.details else ""
        super().__init__(f"{prefix}{message}{suffix}")


class ComponentStr(str):
    """Component identifier, e.g. "A1" for residues, "B12", etc. Previously named `contig_string`"""

    def split_component(v):
        return split_contig(v)


def split_contig(x):
    try:
        chain = str(x[0])
        idx = x[1:]
        idx = int(idx)
        if idx < 0:
            raise ComponentValidationError(
                "Residue index must be a non-negative integer.", component=str(x)
            )
    except Exception as e:
        raise ComponentValidationError(
            f"Invalid contig format: '{x}'. Expected format is 'ChainIDResID' (e.g. 'A20').",
            component=str(x),
        ) from e
    return [chain, idx]


def extract_pn_unit_info(contig):
    """
    Convert substring like A20-21 or A20 to separate terms: A, 20, 21.
    """
    pattern = r"([A-Za-z])(\d+)(?:-(\d+))?"

    match = re.match(pattern, contig)
    if match:
        pn_unit_id = match.group(1)
        start = int(match.group(2))
        end = int(match.group(3)) if match.group(3) else start
        return pn_unit_id, start, end

    raise ComponentValidationError(
        "Invalid contig format. Expected 'ChainIDStart-Stop' or 'ChainIDIdx'.",
        component=contig,
    )


def get_design_pattern_with_constraints(contig, length=None):
    """
    Convert the contig string to separate modules.
    e.g. '1-5,A20-21,1-5,A25-25,1-5,A30-30,/0,1-5' with length = 10-10 may be converted to [2, A20, A21, 2, A25, 3, A30, /0, 3]
    Integers represent number of free residues to put there.

    """
    contig_parts = contig.split(",")

    # Separate fixed segments (e.g., "A1051-1051") and variable ranges (e.g., "0-40")
    variable_ranges = []
    fixed_parts = []
    pos_to_put_motif = []

    for part in contig_parts:
        if any(c.isalpha() for c in part):  # Detect parts containing letters as fixed
            pn_unit_id, pn_unit_start, pn_unit_end = extract_pn_unit_info(part)
            fixed_parts.append([pn_unit_id, pn_unit_start, pn_unit_end])
            pos_to_put_motif.append(1)
        elif part == "/0":
            pos_to_put_motif.append(2)
        else:
            if "-" in part:
                start, end = map(int, part.split("-"))
            else:
                start = end = int(part)
            variable_ranges.append([start, end])
            pos_to_put_motif.append(0)

    # adjust the total length to solely for free residues
    num_motif_residues = sum([i[2] - i[1] + 1 for i in fixed_parts])

    if length is None:
        length_min, length_max = 0, 9999
    else:
        if "-" in length:
            length_min, length_max = map(int, length.split("-"))
        else:
            length_min = length_max = int(length)

    length_min -= num_motif_residues
    length_max -= num_motif_residues

    remaining_length_min = length_min
    remaining_length_max = length_max

    num_free_atoms = []
    for range_limits in variable_ranges:
        min_value = range_limits[0]
        max_value = range_limits[1]

        # Calculate the valid range for the current segment
        valid_min = max(
            min_value,
            remaining_length_min
            - sum(r[1] for r in variable_ranges[len(num_free_atoms) + 1 :]),
        )
        valid_max = min(
            max_value,
            remaining_length_max
            - sum(r[0] for r in variable_ranges[len(num_free_atoms) + 1 :]),
        )

        if valid_min > valid_max and length is not None:
            raise ComponentValidationError(
                "No valid selections possible with the given constraints."
            )

        # Randomly select a value for the current segment
        selected_value = random.randint(valid_min, valid_max)
        num_free_atoms.append(selected_value)

        # Update remaining lengths
        remaining_length_min -= selected_value
        remaining_length_max -= selected_value

    atoms_with_motif = []
    for idx in range(len(pos_to_put_motif)):
        if pos_to_put_motif[idx] == 1:
            motif = fixed_parts.pop(0)
            pn_unit_id, pn_unit_start, pn_unit_end = motif[0], motif[1], motif[2]
            for index in range(pn_unit_start, pn_unit_end + 1):
                atoms_with_motif.append(f"{pn_unit_id}{index}")
        elif pos_to_put_motif[idx] == 0:
            free_atom = num_free_atoms.pop(0)
            atoms_with_motif.append(free_atom)
        elif pos_to_put_motif[idx] == 2:
            atoms_with_motif.append("/0")

    return atoms_with_motif


def get_motif_components_and_breaks(unindexed_contig, index_all=False):
    """
    Convert a contig string into its components and breaks in motif
    This way you can specify in your contigs where the breaks in the motif should be, so that,
        say, residues aren't glued together by the model. Used for parsing unindexed inputs.

    e.g.:
        contig="A14,A15,A16" -> components=[A14, A15, A16] breaks=[True, True,  True]
        contig="A14-15,A16"  -> components=[A14, A15, A16] breaks=[True, False, True]

    args:
        unindexed_contig: Contig string for unindexed tokens, see above for example on how positional
            encodings between contigs can be selectively leaked
        index_all: No breaks are used, allows for full indexing of concatenated tokens
            Can use cleanup if this is the desired way to provide motif tokens.
    """
    components = []
    breaks = []

    contig_parts = unindexed_contig.split(",")
    for part in contig_parts:
        if any(c.isalpha() for c in part):
            # ... Parse possibilities: A11 | A11-12 | A11-11
            pn_unit_id, pn_unit_start, pn_unit_end = extract_pn_unit_info(part)

            if pn_unit_start == pn_unit_end:
                # ... For single residues, append and break
                components.append(f"{pn_unit_id}{pn_unit_start}")
                breaks.append(True)
            else:
                # ... For multiple residues, break and then append without breaks
                for index in range(pn_unit_start, pn_unit_end + 1):
                    components.append(f"{pn_unit_id}{index}")
                    if index == pn_unit_start:
                        breaks.append(True)
                    else:
                        breaks.append(False)
        elif part == "/0":
            components.append(part)
            breaks.append(None)
        else:
            if "-" in part:
                raise ComponentValidationError(
                    "Partial unindexing without fixed length is not supported.",
                    component=part,
                )
            components.append(part)
            breaks.append(None)

    breaks[0] = True  # Decouple unindexed region from global index
    if index_all:
        global_logger.info("Unindexing all residues")
        breaks = [(False if b is not None else None) for b in breaks]
    return components, breaks


#################################################################################
# Mask getters
#################################################################################


def get_name_mask(
    source_names: np.ndarray, query_names: str, source_resname: str | None = None
):
    """
    Args:
        source_names: list of all names to match in current token
        query_string: specifier of names to get:
            "ALL" - All atom names in token are matched
            "BKBN - Only backbone atoms (not CB)
            "TIP" - 2 farthest atoms from the backbone are fixed with any
                additional atoms that automatically constrain geometries
                (e.g. 4 atoms for carboxylates/amides). See `constants.py`.
            Comma-separated string - e.g. "N,CA,C,O,CB" for exact queries
            List of names - e.g. ["N", "CA", "C", "O"] for exact queries
        source_resname: residue name is required when specifying just to grab the names for a "TIP"

    Raises error if not all exact atom names are found and unique

    Returns:
        mask of atoms corresponding to token
    """
    if isinstance(query_names, list):
        names = query_names
    elif isinstance(query_names, str):
        if query_names.upper() == "ALL":
            return np.ones(source_names.shape[0], dtype=bool)
        elif query_names.upper() == "BKBN":
            names = ["N", "CA", "C", "O"]
        elif query_names.upper() == "TIP":
            if not exists(source_resname):
                raise ComponentValidationError(
                    "TIP selection requires a residue name.",
                    component=str(source_resname),
                )
            names = TIP_BY_RESTYPE[source_resname]
            if not exists(names):
                raise ComponentValidationError(
                    "Residue does not define TIP atoms; use ALL, BKBN, or explicit names.",
                    component=str(source_resname),
                )
        elif query_names == "":
            names = []
        else:
            names = query_names.split(",")
    else:
        raise ComponentValidationError(
            "query_names must be a string or list of strings.",
            details={"got_type": str(type(query_names))},
        )

    if any(n == "" for n in names):
        raise ComponentValidationError(
            f"Empty atom name found in selection '{query_names}'.",
            component=str(source_resname),
        )
    mask = np.isin(source_names, names)

    if len(names) == 0:
        return mask

    if not len(set(names)) == len(names):
        raise ComponentValidationError(
            f"Atom names in '{query_names}' must be unique.",
            details={"duplicates": names},
        )
    if not mask.any():
        raise ComponentValidationError(
            f"Could not find requested atoms '{query_names}' in atom array.",
            details={"source_names": np.asarray(source_names).tolist()},
        )
    if mask.sum() != len(names):
        global_logger.warning(
            "Not all atoms found in atom array. Are you expecting multiple residues/ligands with the same names? "
            + "If not, check your input pdb file. "
            + "Atom array requested to contain names {}. Got: {}. Requested {}".format(
                query_names,
                np.asarray(source_names).tolist(),
                np.asarray(names).tolist(),
            )
        )
    if mask.sum() % len(names) != 0:
        # for the case where source_names are originated from multiple residues with the same names
        # (e.g. two ORO ligands in the input pdb: {ligand: "ORO", fixed_atoms: {ORO:"N3,C2,C4,N1"}})
        raise ComponentValidationError(
            "Number of atoms must be a multiple of the requested names.",
            details={
                "query": query_names,
                "source_names": np.asarray(source_names).tolist(),
                "requested": np.asarray(names).tolist(),
            },
        )

    return mask


def fetch_mask_from_idx(contig_str, *, atom_array):
    """
    contig_str: A11
    returns:
        mask of atoms within contig (e.g. residue 11 in chain A)
    """
    chain, res_id = split_contig(contig_str)
    mask = (atom_array.chain_id == chain) & (atom_array.res_id == res_id)
    if not np.any(mask):
        raise ComponentValidationError(
            f"Residue {chain}{res_id} not found in atom array.",
            component=f"{chain}{res_id}",
        )
    return mask


def fetch_mask_from_name(name, *, atom_array):
    """
    name: LIG_NAME
    returns:
        mask of atoms corresponding to non-protein
    """
    mask = atom_array.res_name == name
    if not np.any(mask):
        non_protein_res_names = np.unique(
            atom_array.res_name[~np.isin(atom_array.res_name, _aa_like_res_names)]
        )
        raise ComponentValidationError(
            "Component not found in input atom array.",
            component=name,
            details={"available_non_protein": non_protein_res_names.tolist()},
        )
    return mask


def fetch_mask_from_component(component, *, atom_array):
    """
    Catch-all function for fetching a component by non-protein name or contig
    component: A11 or LIG_NAME
    returns:
        mask of atoms corresponding to component
    """
    try:
        mask = fetch_mask_from_name(component, atom_array=atom_array)
    except ComponentValidationError:
        mask = fetch_mask_from_idx(component, atom_array=atom_array)
    return mask


def unravel_components(
    v: str, *, atom_array: AtomArray = None, allow_multiple_matches: bool = False
) -> List[str]:
    """Safely unravel components from a string input."""
    components = []
    if "," in v or "-" in v:
        components.extend(get_design_pattern_with_constraints(v))
    else:
        # Safely canonicalize to single component
        mask = fetch_mask_from_component(v, atom_array=atom_array)
        if mask.sum() > 0:
            res_ids, chain_ids = atom_array.res_id[mask], atom_array.chain_id[mask]
            # assert unique resids for component
            if len(set(zip(chain_ids, res_ids))) != 1:
                if not allow_multiple_matches:
                    raise ComponentValidationError(
                        f"Component '{v}' maps to multiple residues.",
                        component=v,
                    )
                else:
                    global_logger.warning(
                        f"Component '{v}' maps to multiple residues. If you are using Symmetry this is OK."
                    )
                    components.extend([f"{c}{r}" for c, r in zip(chain_ids, res_ids)])
                    components = list(set(components))  # unique components
                    return components
            res_id, chain_id = res_ids[0], chain_ids[0]

            component = f"{chain_id}{res_id}"
            global_logger.debug(
                "Canonicalized component string: %s -> %s", v, component
            )
            components.append(component)
    return components
