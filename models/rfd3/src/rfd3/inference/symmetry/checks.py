import numpy as np
from rfd3.inference.symmetry.contigs import expand_contig_unsym_motif
from rfd3.transforms.conditioning_base import get_motif_features

from foundry.utils.ddp import RankedLogger

MIN_ATOMS_ALIGN = 100
MAX_TRANSFORMS = 10
RMSD_CUT = 1.0  # Angstroms

ranked_logger = RankedLogger(__name__, rank_zero_only=True)


def check_symmetry_config(
    atom_array, sym_conf, sm, has_dist_cond, src_atom_array=None, partial=False
):
    """
    Check if the symmetry configuration is valid. Add all basic checks here.
    """

    assert sym_conf.get("id"), "symmetry_id is required. e.g. {'id': 'C2'}"
    # if unsym motif is provided, check that each motif name is in the atom array
    if sym_conf.get("is_unsym_motif"):
        assert (
            src_atom_array is not None
        ), "Source atom array must be provided for symmetric motifs"
        unsym_motif_names = sym_conf["is_unsym_motif"].split(",")
        unsym_motif_names = expand_contig_unsym_motif(unsym_motif_names)
        for n in unsym_motif_names:
            if (sm and n not in sm.split(",")) and (n not in atom_array.src_component):
                raise ValueError(f"Unsym motif {n} not found in atom_array")
    if (
        get_motif_features(atom_array)["is_motif_token"].any()
        and not sym_conf.get("is_symmetric_motif")
        and not has_dist_cond
    ):
        raise ValueError(
            "Asymmetric motif inputs should be distance constrained. "
            "Use atomwise_fixed_dist to constrain the distance between the motif atoms."
        )
    # else: if unconditional symmetry, no need to have symmetric input motif

    if partial and not sym_conf.get("is_symmetric_motif"):
        raise ValueError(
            "Partial diffusion with symmetry is only supported for symmetric inputs."
        )


def check_atom_array_is_symmetric(atom_array):
    """
    Check if the atom array is symmetric. This is NOT to check that the atom array symmetry matches that of the symmetry_id.
    Arguments:
        atom_array: atom arrays to check
    Returns:
        bool: True if the atom array is symmetric, False otherwise
    """
    # TODO: Implement something like this https://github.com/baker-laboratory/ipd/blob/main/ipd/sym/sym_detect.py#L303
    #       and maybe this https://github.com/baker-laboratory/ipd/blob/main/ipd/sym/sym_detect.py#L231

    import biotite.structure as struc
    from rfd3.inference.symmetry.atom_array import (
        apply_symmetry_to_atomarray_coord,
    )
    from rfd3.inference.symmetry.frames import (
        get_symmetry_frames_from_symmetry_id,
    )

    # remove hetero atoms
    atom_array = atom_array[~atom_array.hetero]
    if len(atom_array) == 0:
        ranked_logger.info("Atom array has no protein chains. Please check your input.")
        return False

    chains = np.unique(atom_array.chain_id)
    asu_mask = atom_array.chain_id == chains[0]
    asu_atoms = atom_array[asu_mask].copy()

    # Check that all atom arrays have the same number of atoms
    for chain in chains[1:]:
        chain_mask = atom_array.chain_id == chain
        if len(asu_atoms) != len(atom_array[chain_mask]):
            ranked_logger.info(
                f"Atom array has different number of atoms in chain {chain}. {len(asu_atoms)} != {len(atom_array[chain_mask])}"
            )
            return False

    # Check that all atom arrays have the same atoms
    for chain in chains[1:]:
        chain_mask = atom_array.chain_id == chain
        for i in range(len(asu_atoms)):
            if asu_atoms.atom_name[i] != atom_array[chain_mask].atom_name[i]:
                ranked_logger.info(
                    f"Atom array has different atoms in chain {chain}. {asu_atoms.atom_name[i]} != {atom_array[chain_mask].atom_name[i]}"
                )
                return False

    # Check that the atom array aligns with the standard symmetry frames
    standard_frames = get_symmetry_frames_from_symmetry_id(atom_array.symmetry_id[0])
    standard_atom_array = []
    for frame in standard_frames:
        symmed_atoms = apply_symmetry_to_atomarray_coord(asu_atoms, frame)
        standard_atom_array.append(symmed_atoms)
    standard_atom_array = struc.concatenate(standard_atom_array)

    R_standard_obtained = find_optimal_rotation(
        standard_atom_array.coord, atom_array.coord
    )

    if R_standard_obtained is None:
        ranked_logger.info(
            "Atom array does not align with the standard symmetry frames."
        )
        return False

    return True


def find_optimal_rotation(coords1, coords2, max_points=1000):
    """
    Find optimal rotation matrix between two sets of coordinates using Kabsch algorithm.

    Args:
        coords1: reference coordinates (N, 3)
        coords2: target coordinates (N, 3)
        max_points: maximum number of points to use for efficiency

    Returns:
        rotation_matrix: 3x3 rotation matrix or None if failed
    """
    if len(coords1) > max_points:
        indices = np.random.choice(len(coords1), max_points, replace=False)
        coords1 = coords1[indices]
        coords2 = coords2[indices]

    # Ensure same number of points
    min_len = min(len(coords1), len(coords2))
    coords1 = coords1[:min_len]
    coords2 = coords2[:min_len]
    if min_len < 3:
        return None

    # Kabsch algorithm
    try:
        centroid1 = np.mean(coords1, axis=0)
        centroid2 = np.mean(coords2, axis=0)
        coords1_centered = coords1 - centroid1
        coords2_centered = coords2 - centroid2

        # Compute covariance matrix
        H = coords1_centered.T @ coords2_centered

        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        # Ensure proper rotation matrix
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        return R

    except Exception as e:
        print(f"Error in rotation calculation: {e}")
        return None


def check_input_frames_match_symmetry_frames(computed_frames, original_frames) -> None:
    """
    Check if the atom array matches the symmetry_id.
    Arguments:
        computed_frames: list of computed frames
        original_frames: list of original frames
    """
    assert len(computed_frames) == len(
        original_frames
    ), "Number of computed frames does not match number of original frames"


def check_valid_multiplicity(nids_by_entity) -> None:
    """
    Check if the multiplicity is valid.
    Arguments:
        nids_by_entity: dict mapping entity to ids
    """
    # get multiplicities of subunits
    multiplicity = min([len(i) for i in nids_by_entity.values()])
    if multiplicity == 1:  # no possible symmetry
        raise ValueError(
            "Input has no possible symmetry. If asymmetric motif, please use 2D conditioning inference instead."
        )

    # Check that the input is not asymmetric
    multiplicity_good = [len(i) % multiplicity == 0 for i in nids_by_entity.values()]
    if not all(multiplicity_good):
        raise ValueError("Invalid multiplicities of subunits. Please check your input.")


def check_valid_subunit_size(nids_by_entity, pn_unit_id) -> None:
    """
    Check that the subunits in the input are of the same size.
    Arguments:
        nids_by_entity: dict mapping entity to ids
    """
    for i, js in nids_by_entity.items():
        for j in js[1:]:
            if (pn_unit_id == js[0]).sum() != (pn_unit_id == j).sum():
                raise ValueError("Size mismatch in the input. Please check your file.")


def check_min_atoms_to_align(natm_per_unique, reference_entity) -> None:
    """
    Check that we have enough atoms to align.
    Arguments:
        nids_by_entity: dict mapping entity to ids
    """
    if natm_per_unique[reference_entity] < MIN_ATOMS_ALIGN:
        raise ValueError("Not enough atoms to align. Please check your input.")


def check_max_transforms(chains_to_consider) -> None:
    """
    Check that we are not exceeding the max number of transforms.
    Arguments:
        chains_to_consider: list of chains to consider
        max_transforms: max number of transforms
    """
    if len(chains_to_consider) > MAX_TRANSFORMS:
        raise ValueError(
            "Number of transforms exceeds the max number of transforms (10)"
        )


def check_max_rmsds(rmsds) -> None:
    """
    Check that the RMSD between the reference molecule and the other molecules is not too big.
    Arguments:
        rmsds: dict mapping chain to RMSD
    """
    if max(rmsds.values()) > RMSD_CUT:
        ranked_logger.warning(
            f"RMSD between the reference molecule and the other molecules is too big ({max(rmsds.values())} > {RMSD_CUT}). Please provide a symmetric input PDB file."
        )
        # raise ValueError(f"RMSD between the reference molecule and the other molecules is too big ({max(rmsds.values())} > {RMSD_CUT}). Please provide a symmetric input PDB file.")
