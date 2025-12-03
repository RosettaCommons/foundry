import copy
import functools
import logging
from os import PathLike

import biotite.structure as struc
import numpy as np
from atomworks.constants import STANDARD_AA
from atomworks.io.utils.io_utils import to_cif_file
from atomworks.ml.encoding_definitions import AF3SequenceEncoding
from atomworks.ml.utils.token import (
    get_token_starts,
)
from rfd3.constants import (
    INFERENCE_ANNOTATIONS,
    OPTIONAL_CONDITIONING_VALUES,
    REQUIRED_INFERENCE_ANNOTATIONS,
)
from rfd3.inference.symmetry.symmetry_utils import (
    center_symmetric_src_atom_array,
    make_symmetric_atom_array,
)
from rfd3.transforms.conditioning_base import (
    check_has_required_conditioning_annotations,
    convert_existing_annotations_to_bool,
    get_motif_features,
    set_default_conditioning_annotations,
)
from rfd3.transforms.util_transforms import assign_types_
from rfd3.utils.inference import (
    create_cb_atoms,
    create_o_atoms,
    extract_ligand_array,
    inference_load_,
    set_com,
    set_common_annotations,
    set_indices,
)

from foundry.common import exists
from foundry.utils.components import (
    fetch_mask_from_component,
    fetch_mask_from_idx,
    get_design_pattern_with_constraints,
    get_motif_components_and_breaks,
    get_name_mask,
    split_contig,
)
from foundry.utils.ddp import RankedLogger

logging.basicConfig(level=logging.INFO)
ranked_logger = RankedLogger(__name__, rank_zero_only=True)

sequence_encoding = AF3SequenceEncoding()
_aa_like_res_names = sequence_encoding.all_res_names[sequence_encoding.is_aa_like]


def assert_non_intersecting_contigs(indexed_components, unindexed_components):
    assert not any(
        [
            (
                (unindexed_component in indexed_components)
                and unindexed_component[0].isalpha()
            )
            for unindexed_component in unindexed_components
        ]
    ), "Unindexed residues must not be part of the indexing contig. got: {} and {}".format(
        unindexed_components, indexed_components
    )


def set_atom_level_argument(atom_array, args, name: str):
    default_value = OPTIONAL_CONDITIONING_VALUES.get(name, np.nan)
    atom_values = np.full(atom_array.array_length(), default_value)
    atom_idxs = np.arange(atom_array.array_length())

    if args is not None:
        for component_name, d in args.items():
            component_mask = fetch_mask_from_component(
                component_name, atom_array=atom_array
            )
            for names, value in d.items():
                mask = component_mask & np.isin(
                    atom_array.atom_name, np.array(names.split(","))
                )
                assert mask.sum() == len(
                    names.split(",")
                ), f"Not all atoms in {names} found in {atom_array.atom_name}"

                atom_idxs_selected = atom_idxs[mask]
                atom_values[atom_idxs_selected] = value

    atom_array.set_annotation(name, atom_values)
    return atom_array


def fetch_motif_residue_(
    src_chain,
    src_resid,
    *,
    components,
    src_atom_array,
    redesign_motif_sidechains,
    unindexed_components,
    unfixed_sequence_components,
    fixed_atoms,
    unfix_all,
    flexible_backbone,
    unfix_residues,
):
    """
    Given source chain and resid, returns the residue if present in the source atom array

    NB: For glycines, we extend the array with a CB position so as to not leak whether
    the original residue is a glycine if sequence is masked during inference.
    """

    assert (
        src_atom_array is not None
    ), "Motif provided in contigs, but no input provided. input={} contig={}".format(
        input, components
    )

    # ... Fetch residue in the input atom array
    mask = fetch_mask_from_idx(f"{src_chain}{src_resid}", atom_array=src_atom_array)
    subarray = src_atom_array[mask]
    res_name = subarray.res_name[0]

    # Check if we have a redesign_motif_sidechains contig
    if isinstance(redesign_motif_sidechains, list):
        # If we have a list, check if the residue is in the list
        if f"{src_chain}{src_resid}" in redesign_motif_sidechains:
            redesign_motif_sidechains = True
        else:
            redesign_motif_sidechains = False

    # Assign base properties
    subarray = set_default_conditioning_annotations(
        subarray, motif=True, unindexed=False, dtype=int
    )  # all values init to True (fix all)

    # Assign is motif atom and sequence
    if exists(atoms := fixed_atoms.get(f"{src_chain}{src_resid}")):
        atom_mask = get_name_mask(subarray.atom_name, atoms, res_name)
        subarray.set_annotation("is_motif_atom", atom_mask)
        # subarray.set_annotation("is_motif_atom_with_fixed_coord", atom_mask)  # BUGFIX: uncomment

    elif redesign_motif_sidechains and res_name in STANDARD_AA:
        n_atoms = subarray.shape[0]
        diffuse_oxygen = False
        if n_atoms < 3:
            raise ValueError(
                f"Not enough data for {src_chain}{src_resid} in input atom array."
            )
        if n_atoms == 3:
            # Handle cases with N, CA, C only;
            subarray = subarray + create_o_atoms(subarray.copy())
            diffuse_oxygen = True  # flag oxygen for generation

        # Subset to the first 4 atoms (N, CA, C, O) only
        subarray = subarray[np.isin(subarray.atom_name, ["N", "CA", "C", "O"])]

        # exactly N, CA, C, O but no CB. Place CB onto idealized position and conver to ALA
        # Sequence name ALA ensures the padded atoms to be diffused from the fixed backbone
        # are placed on the CB so as to not leak the identity of the residue.
        subarray = subarray + create_cb_atoms(subarray.copy())

        # Sequence name must be set to ALA such that the central atom is correctly CB
        subarray.res_name = np.full_like(
            subarray.res_name, "ALA", dtype=subarray.res_name.dtype
        )
        subarray.set_annotation(
            "is_motif_atom",
            (
                np.arange(subarray.shape[0], dtype=int) < (4 - int(diffuse_oxygen))
            ).astype(int),
        )
        subarray.set_annotation(
            "is_motif_atom_with_fixed_seq", np.zeros(subarray.shape[0], dtype=int)
        )
    if unfix_all or f"{src_chain}{src_resid}" in unfix_residues:
        subarray.set_annotation(
            "is_motif_atom_with_fixed_coord", np.zeros(subarray.shape[0], dtype=int)
        )
    else:
        subarray.set_annotation(
            "is_motif_atom_with_fixed_coord", subarray.is_motif_atom.copy()
        )
    if flexible_backbone:
        backbone_atoms = ["N", "CA", "C", "O"]
        is_flexible_motif_atom = np.isin(subarray.atom_name, backbone_atoms)
        subarray.set_annotation(
            "is_flexible_motif_atom",
            is_flexible_motif_atom,
        )
    else:
        subarray.set_annotation(
            "is_flexible_motif_atom", np.zeros(subarray.shape[0], dtype=bool)
        )
    to_unindex = f"{src_chain}{src_resid}" in unindexed_components
    if to_unindex:
        subarray.set_annotation(
            "is_motif_atom_unindexed", subarray.is_motif_atom.copy()
        )
        # Subset to desired motif atoms
        subarray = subarray[subarray.is_motif_atom.astype(bool)]

    # ... Relax sequence constraint if provided
    if (
        exists(unfixed_sequence_components)
        and f"{src_chain}{src_resid}" in unfixed_sequence_components
    ):
        ranked_logger.info(
            "Unfixing sequence for motif {}{}".format(src_chain, src_resid)
        )
        subarray.set_annotation(
            "is_motif_atom_with_fixed_seq",
            np.zeros(subarray.shape[0], dtype=int),
        )

    # ... Double check that required annotations are set within the scope of this function only
    check_has_required_conditioning_annotations(subarray)
    subarray = set_common_annotations(subarray)
    subarray.set_annotation("res_id", np.full(subarray.shape[0], 1))  # Reset to 1
    return subarray


def create_diffused_residues_(n):
    if n <= 0:
        raise ValueError(f"Negative/null residue count ({n}) not allowed.")

    atoms = []
    [
        atoms.extend(
            [
                struc.Atom(
                    np.array([0.0, 0.0, 0.0], dtype=np.float32),
                    res_name="ALA",
                    res_id=idx,
                )
                for _ in range(5)
            ]
        )
        for idx in range(1, n + 1)
    ]
    array = struc.array(atoms)
    array.set_annotation(
        "element", np.array(["N", "C", "C", "O", "C"] * n, dtype="<U2")
    )
    array.set_annotation(
        "atom_name", np.array(["N", "CA", "C", "O", "CB"] * n, dtype="<U2")
    )
    array = set_default_conditioning_annotations(array, motif=False)
    array = set_common_annotations(array)
    return array


def accumulate_components(
    components,
    src_atom_array,
    redesign_motif_sidechains,
    unindexed_components: list[str],
    unfixed_sequence_components: list[str],
    breaks: list,
    fixed_atoms: dict,
    unfix_all: bool,
    optional_conditions: list[str],
    flexible_backbone: bool,
    *,
    start_chain="A",
    unfix_residues: list[str],
    start_resid=1,
):
    """
    Subcomponents have three types, specifying either the end of a chain ("/0),
    a motif (e.g. "A20" or "A21"), or a number indicating the number of diffused residues to create.
    This function accumulates these components into a single atom array.

    Arguments:
        - components: list of components, where each component is either a string
        e.g. [2, A20, A21, 2, A25, 3, A30, /0, 3]
        - src_atom_array: the source atom array to fetch motifs from, or None if no input is provided.
        - unindexed_components: list of components to unindex e.g. [A20, A21]
        - redesign_motif_sidechains: whether to diffuse the sidechains of the input motifs (indexed components)
        - fixed_atoms: dictionary of fixed atoms for each component (previously called `contig_atoms`)
        - unfix_all: whether to fully unfix the motif coordinates
        - unfix_residues: list of residues to unfix. Overrides `unfix_all` for specific residues.
        - flexible_backbone: whether to allow flexible backbone for motifs

    Returns:
        - Accumulated atom array with components, and is_motif labels
    """
    # ... Create component assignment functions
    fetch_motif_residue = functools.partial(
        fetch_motif_residue_,
        components=components,
        src_atom_array=src_atom_array,
        redesign_motif_sidechains=redesign_motif_sidechains,
        unindexed_components=unindexed_components,
        unfixed_sequence_components=unfixed_sequence_components,
        fixed_atoms=fixed_atoms,
        unfix_all=unfix_all,
        flexible_backbone=flexible_backbone,
        unfix_residues=unfix_residues,
    )
    create_diffused_residues = create_diffused_residues_

    # ... For loop accum variables
    breaks = [None] * len(components) if breaks is None else breaks
    unindexed_components_started = (
        False  # once one unindexed component is added, stop adding diffused residues
    )
    atom_array_accum = []
    chain = start_chain
    res_id = start_resid
    molecule_id = 0
    # 2) Insert contig information one- by one-
    for component, is_break in zip(components, breaks):
        if component == "/0":
            # reset iterators on next chain
            chain = chr(ord(chain) + 1)
            molecule_id += 1
            res_id = 1
            continue

        # Create array to insert
        if str(component)[0].isalpha():  # motif (e.g. "A22")
            atom_array_insert = fetch_motif_residue(*split_contig(component))
            n = 1
            if exists(is_break) and is_break:
                if not unindexed_components_started:
                    chain = start_chain
                unindexed_components_started = True
                atom_array_insert.set_annotation(
                    "is_motif_atom_unindexed_motif_breakpoint",
                    np.ones(atom_array_insert.shape[0], dtype=int),
                )
        else:
            n = int(component)
            if n == 0 or unindexed_components_started:
                res_id += n
                continue
            atom_array_insert = create_diffused_residues(n)
            for key in optional_conditions:
                atom_array_insert.set_annotation(
                    key,
                    np.full(
                        atom_array_insert.array_length(),
                        OPTIONAL_CONDITIONING_VALUES[key],
                        dtype=int,
                    ),
                )

        # ... Set index of insertion
        atom_array_insert = set_indices(
            array=atom_array_insert,
            chain=chain,
            res_id_start=res_id,
            molecule_id=molecule_id,
            component=component,
        )

        assert (
            len(get_token_starts(atom_array_insert)) == n
        ), f"Mismatch in number of residues: expected {n}, got {len(get_token_starts(atom_array_insert))} in \n{atom_array_insert}"

        # ... Insert & Increment residue ID
        atom_array_accum.append(atom_array_insert)
        res_id += n

    atom_array_accum = struc.concatenate(atom_array_accum)
    atom_array_accum.set_annotation("pn_unit_iid", atom_array_accum.chain_id)

    # Reset res_id for unindexed residues to avoid duplicates (ridiculously long lines of code, cleanup later)
    if np.any(atom_array_accum.is_motif_atom_unindexed.astype(bool)) and not np.all(
        atom_array_accum.is_motif_atom_unindexed.astype(bool)
    ):
        max_id = np.max(
            atom_array_accum[
                ~atom_array_accum.is_motif_atom_unindexed.astype(bool)
            ].res_id
        )
        min_id_udx = np.min(
            atom_array_accum[
                atom_array_accum.is_motif_atom_unindexed.astype(bool)
            ].res_id
        )
        atom_array_accum.res_id[
            atom_array_accum.is_motif_atom_unindexed.astype(bool)
        ] += max_id - min_id_udx + 1

    return atom_array_accum


#################################################################################
# Custom conditioning functions
#################################################################################


def create_atom_array_from_design_specification_legacy(
    *,
    # Specification args:
    input: PathLike = None,
    length: str = "100-300",
    contig: str = None,
    fixed_atoms: dict = None,
    unindex: str = None,
    unfix_sequence: str = None,
    redesign_motif_sidechains: bool = False,
    unfix_all=False,
    unfix_specific: str = None,
    flexible_backbone: bool = False,
    # Args for biomolecular design (Enzymes, DNA/PNA):
    ligand: str = None,
    ori_token: list[float] = None,
    infer_ori_strategy: str | None = None,
    atomwise_rasa: dict = None,
    atomwise_hbond: dict = None,
    # Additional args:
    out_path=None,
    cif_parser_args=None,
    # PPI Kwargs
    atom_level_hotspots: dict | None = None,
    # SS conditioning kwargs
    is_helix: dict | None = None,
    is_sheet: dict | None = None,
    is_loop: dict | None = None,
    spoof_helical_bundle_ss_conditioning: bool = False,
    symmetry: dict = None,
    # Low-temperature global conditioning args
    plddt_enhanced: bool = True,
    is_non_loopy: bool | None = None,
    # Partial diff args:
    partial_t: float | None = None,  # Optional noise scale for partial diffusion
    **_,  # dump additional args
):
    """
    Create pre-pipeline CIF file.

    Arguments:
        - input: path to input pdb containing coordinate data
        - contig: your typical contig string '10-10,A20-21,5-5,A25-25,5-5,A30-30,10-10'.
        - unindex: string of residue indices to unindex, e.g. "A20,A21" or "A20-21". Note the latter will be treated as two contiguous
            residues whereas the former will end up as two uncorrelated residues.
        - unfix_sequence: contig string of components to unfix sequence for.
        - unfix_specific: comma separated residues to unfix coordinates for. "ALL" to unfix every motif.
        - length: required total length (optional)
        - ligand: name of ligand to keep from input pdb, or path to a cif file containing the ligand
        - ori_token: coordinates for origin relative to coordinates in input file.
        - infer_ori_strategy: string argument controlling how the ori token is inferred if not otherwise specified.
            If None, the ori token will be set to the COM of the motif, or to [0,0,0] for unconditional generation.
            Currently supported strategies:
                - "hotspots": move 10A along an outward normal vector from the COM of the hotspots.

    Returns:
        - atom_array with all required conditioning annotations set appropriately.
    """
    ###########################################################################################################################

    # 1) Load input data if provided
    if exists(input):
        atom_array_input = inference_load_(input, cif_parser_args=cif_parser_args)[
            "atom_array"
        ]
        # If we are doing symmetric design, we need to center the full input atom array at the origin (for getting symmetry frames)
        if exists(symmetry) and symmetry.get("id"):
            atom_array_input = center_symmetric_src_atom_array(atom_array_input)
    elif exists(contig) or exists(length):
        atom_array_input = None
    else:
        raise ValueError("Either 'input' or 'contig' / 'length' must be provided.")
    if isinstance(length, int):
        length = f"{length}-{length}"
    if exists(length) and not exists(contig):
        # Handle cases where contigs aren't specified
        if not exists(unindex) and not exists(flexible_backbone):
            if exists(fixed_atoms):
                # ensure that fixed atoms are in the input, else raise error
                _ = [
                    fetch_mask_from_component(key, atom_array=atom_array_input)
                    for key in fixed_atoms.keys()
                ]
            ranked_logger.warning(
                "No input contig specified and no motif, running unconditional generation"
            )
        indexed_components_provided = False
        contig = length
    else:
        indexed_components_provided = True
    if not exists(fixed_atoms):
        fixed_atoms = {}

    optional_conditions = []
    if exists(atomwise_rasa):
        set_atom_level_argument(atom_array_input, atomwise_rasa, "rasa_bin")
        optional_conditions.append("rasa_bin")
    if exists(atomwise_hbond):
        for key, value in atomwise_hbond.items():
            set_atom_level_argument(atom_array_input, value, key)
            optional_conditions.append(key)
    if exists(atom_level_hotspots):
        set_atom_level_argument(
            atom_array_input, atom_level_hotspots, "is_atom_level_hotspot"
        )
        optional_conditions.append("is_atom_level_hotspot")

    # 2) Parse contigs into components
    indexed_components = get_design_pattern_with_constraints(
        contig, length
    )  # e.g. [2, A20, A21, 2, A25, 3, A30, /0, 3]

    # Parse redesign_motif_sidechains if necessary
    if isinstance(redesign_motif_sidechains, str):
        redesign_motif_sidechains = get_design_pattern_with_constraints(
            redesign_motif_sidechains
        )
    ###########################################################################################################################

    # ... Add unindexed components
    unindexed_components, unindexed_breaks = (
        get_motif_components_and_breaks(unindex) if exists(unindex) else ([], [])
    )
    breaks = [None] * len(indexed_components) + unindexed_breaks
    assert_non_intersecting_contigs(indexed_components, unindexed_components)

    # ... Expand unfix_sequence into components
    unfixed_sequence_components = (
        get_design_pattern_with_constraints(unfix_sequence) if unfix_sequence else []
    )

    # Determine which residues to unfix
    unfix_residues = []
    if isinstance(unfix_specific, list):
        unfix_residues = [str(u) for u in unfix_specific]
    elif isinstance(unfix_specific, str):
        if unfix_specific.upper() == "ALL":
            unfix_all = True
        elif unfix_specific:
            unfix_residues, _ = get_motif_components_and_breaks(
                unfix_specific, index_all=True
            )

    # 3) Create atom array from components
    if exists(partial_t):
        ranked_logger.info("Using partial diffusion with t=%s", partial_t)
        atom_array = assign_types_(copy.deepcopy(atom_array_input))
        atom_array = atom_array[atom_array.is_protein]

        # Set the whole thing without constraints
        atom_array = set_default_conditioning_annotations(
            atom_array, motif=False, unindexed=False
        )
        atom_array = set_common_annotations(
            atom_array, set_src_component_to_res_name=False
        )

        # Fix parts in the atom array as fixed components
        set_default_conditioning_annotations(atom_array, motif=False, unindexed=False)
        if indexed_components and indexed_components_provided:
            for component in indexed_components:
                if str(component)[0].isalpha():
                    mask = fetch_mask_from_component(component, atom_array=atom_array)

                    # Set the component as a motif token
                    set_default_conditioning_annotations(
                        atom_array, motif=True, unindexed=False, mask=mask
                    )

                    # Set the fixed atoms of the component
                    if mask.any():
                        # Also handle fixed atoms
                        if component in fixed_atoms:
                            atom_mask = get_name_mask(
                                atom_array.atom_name[mask],
                                fixed_atoms[component],
                                atom_array.res_name[mask][0],
                            )
                            # If specific fixed atoms are selected, set fixed coordinates to those specified
                            atom_array.is_motif_atom_with_fixed_coord[mask] = atom_mask
                        else:
                            # Otherwise fix the entire token.
                            atom_array.is_motif_atom_with_fixed_coord[mask] = 1

        # Append unindexed components from input specifcation
        if unindexed_components:
            start_resid = np.max(atom_array.res_id) + 1
            start_chain = atom_array.chain_id[
                0
            ]  # HACK: set chain ID for unindexed residues as whatever the input has
            atom_array_append = accumulate_components(
                # Normal stuff:
                components=unindexed_components,
                breaks=unindexed_breaks,
                # Regular other stuff
                src_atom_array=atom_array_input,
                redesign_motif_sidechains=redesign_motif_sidechains,
                unindexed_components=unindexed_components,
                unfixed_sequence_components=unfixed_sequence_components,
                fixed_atoms=fixed_atoms,
                unfix_all=unfix_all,
                optional_conditions=optional_conditions,
                flexible_backbone=flexible_backbone,
                unfix_residues=unfix_residues,
                start_chain=start_chain,
                start_resid=start_resid,
            )
            atom_array = atom_array + atom_array_append
    else:
        atom_array = accumulate_components(
            components=indexed_components + unindexed_components,
            src_atom_array=atom_array_input,
            redesign_motif_sidechains=redesign_motif_sidechains,
            unindexed_components=unindexed_components,
            unfixed_sequence_components=unfixed_sequence_components,
            breaks=breaks,
            fixed_atoms=fixed_atoms,
            unfix_all=unfix_all,
            optional_conditions=optional_conditions,
            flexible_backbone=flexible_backbone,
            unfix_residues=unfix_residues,
        )

    # Spoof assignments for is_motif_token
    f = get_motif_features(atom_array)
    is_motif_token = f["is_motif_token"]
    atom_array.set_annotation("is_motif_token", is_motif_token.astype(int))
    is_motif_atom = f["is_motif_atom"]
    atom_array.set_annotation("is_motif_atom", is_motif_atom.astype(int))

    # ... If ligand, post-pend it
    if exists(ligand):
        ligand_array = extract_ligand_array(
            atom_array_input,
            ligand,
            fixed_atoms,
            additional_annotations=set(
                list(atom_array.get_annotation_categories())
                + list(atom_array_input.get_annotation_categories())
                + ["is_motif_atom", "is_motif_token"]
            ),
        )
        ligand_array.res_id = (
            ligand_array.res_id
            - np.min(ligand_array.res_id)
            + np.max(atom_array.res_id)
            + 1
        )
        atom_array = atom_array + ligand_array

    # ... Apply symmetry if it exists ahead of any other processing
    if exists(symmetry) and symmetry.get("id"):
        atom_array = make_symmetric_atom_array(
            atom_array, symmetry, sm=ligand, src_atom_array=atom_array_input
        )

    # ... Input frame and ORI token handling
    if exists(partial_t):
        # For symmetric structures, avoid COM centering that would collapse chains
        if exists(symmetry) and symmetry.get("id"):
            ranked_logger.info(
                "Partial diffusion with symmetry: skipping COM centering to preserve chain spacing"
            )
        else:
            atom_array = set_com(atom_array, ori_token=None, infer_ori_strategy="com")
        atom_array.set_annotation(
            "partial_t", np.full(atom_array.shape[0], partial_t, dtype=float)
        )
    else:
        atom_array = set_com(
            atom_array, ori_token=ori_token, infer_ori_strategy=infer_ori_strategy
        )
        # diffused atoms initialized at origin
        atom_array.coord[~atom_array.is_motif_atom_with_fixed_coord.astype(bool), :] = (
            0.0
        )

    # This is an annotation on the diffused regions, so must be added after accumulate_components
    if spoof_helical_bundle_ss_conditioning:
        is_helix = spoof_helical_bundle_ss_conditioning_fn(atom_array)
        is_sheet = None
        is_loop = None
    if exists(is_helix):
        set_atom_level_argument(atom_array, is_helix, "is_helix")
    if exists(is_sheet):
        set_atom_level_argument(atom_array, is_sheet, "is_sheet")
        optional_conditions.append("is_sheet")
    if exists(is_loop):
        set_atom_level_argument(atom_array, is_loop, "is_loop")
        optional_conditions.append("is_loop")

    is_non_loopy_annot = np.zeros(atom_array.array_length(), dtype=int)
    diffused_region_mask = ~(atom_array.is_motif_token.astype(bool))
    if exists(is_non_loopy):
        is_non_loopy_annot[diffused_region_mask] = 1 if is_non_loopy else -1

    atom_array.set_annotation("is_non_loopy", is_non_loopy_annot)
    atom_array.set_annotation("is_non_loopy_atom_level", is_non_loopy_annot)

    if plddt_enhanced:
        atom_array.set_annotation(
            "ref_plddt", np.ones((atom_array.array_length(),), dtype=int)
        )

    # Ensure correct annotations before saving
    check_has_required_conditioning_annotations(
        atom_array, required=REQUIRED_INFERENCE_ANNOTATIONS
    )
    convert_existing_annotations_to_bool(atom_array)

    if "atom_id" in atom_array.get_annotation_categories():
        ranked_logger.info("Removing atom_id annotation...")
        atom_array.del_annotation("atom_id")

    if out_path is not None:
        to_cif_file(atom_array, out_path, extra_fields=INFERENCE_ANNOTATIONS)

    return atom_array
