import numpy as np
import torch


def get_symmetry_frames_from_symmetry_id(symmetry_id):
    """
    Get symmetry frames from a symmetry id.
    Arguments:
        symmetry_id: string of the symmetry id
    Returns:
        frames: list of rotation matrices
    """

    # Get frames from symmetry id
    sym_conf = {}
    if isinstance(symmetry_id, dict):
        sym_conf = symmetry_id
        symmetry_id = symmetry_id.get("id")

    if symmetry_id.lower().startswith("c"):
        order = int(symmetry_id[1:])
        frames = get_cyclic_frames(order)
    elif symmetry_id.lower().startswith("d"):
        order = int(symmetry_id[1:])
        frames = get_dihedral_frames(order)
    elif symmetry_id.lower() == "input_defined":
        assert (
            "symmetry_file" in sym_conf
        ), "symmetry_file is required for input_defined symmetry"
        frames = get_frames_from_file(sym_conf.get("symmetry_file"))
    else:
        raise ValueError(f"Symmetry id {symmetry_id} not supported")

    # Check that the frames are valid rotation matrices
    for R, _ in frames:
        assert is_valid_rotation_matrix(R), f"Frame {R} is not a valid rotation matrix"

    return frames


def get_symmetry_frames_from_atom_array(src_atom_array, input_frames):
    """
    Get symmetry frames from an atom array. Adapted from code from FD
    Arguments:
        src_atom_array: atom array with coordinates and chain/residue information
        input_frames: list of (rotation_matrix, translation_vector) tuples
    Returns:
        computed_frames: list of (rotation_matrix, translation_vector) tuples (updated)
    """
    # import within the function to avoid circular import
    from rfd3.inference.symmetry.checks import (
        check_input_frames_match_symmetry_frames,
        check_max_rmsds,
        check_max_transforms,
        check_min_atoms_to_align,
        check_valid_multiplicity,
        check_valid_subunit_size,
    )

    # remove non-protein elements
    src_atom_array = src_atom_array[src_atom_array.chain_type == 6]

    # get entities and ids from the src atom array
    pn_unit_ent = src_atom_array.get_annotation("pn_unit_entity")
    pn_unit_id = src_atom_array.get_annotation("pn_unit_iid")
    unique_entities = np.unique(pn_unit_ent)
    nids_by_entity = {
        i: np.unique(pn_unit_id[pn_unit_ent == i]) for i in unique_entities
    }

    # get coordinates
    coords = src_atom_array.coord

    # get/check multiplicities of subunits
    check_valid_multiplicity(nids_by_entity)

    multiplicity = min([len(i) for i in nids_by_entity.values()])
    n_per_asu = {i: len(j) // multiplicity for i, j in nids_by_entity.items()}

    # check that the subunits in the input are of the same size
    check_valid_subunit_size(nids_by_entity, pn_unit_id)

    # align the largest set of entities
    natm_per_unique = {
        i: (pn_unit_id == nids_by_entity[i][0]).sum()
        for i in unique_entities
        if n_per_asu[i] == 1
    }
    reference_entity = max(natm_per_unique, key=natm_per_unique.get)

    # check that we have enough atoms to align
    check_min_atoms_to_align(natm_per_unique, reference_entity)

    # chains for the alignment (will generate complete set of frames)
    chains_to_consider = nids_by_entity[reference_entity]
    reference_molecule = nids_by_entity[reference_entity][0]

    # check that we are not exceeding the max number of transforms
    check_max_transforms(chains_to_consider)

    # align reference molecule to all others
    xforms = {
        i: _align(coords[pn_unit_id == i], coords[pn_unit_id == reference_molecule])
        for i in chains_to_consider
    }
    rmsds = {
        i: _rms(coords[pn_unit_id == i], coords[pn_unit_id == reference_molecule], *j)
        for i, j in xforms.items()
    }

    # check that there is not too big of a RMSD difference between subunits
    check_max_rmsds(rmsds)

    # check that the frames are valid rotation matrices
    Rs = [R for _, R, _ in xforms.values()]
    for R in Rs:
        assert is_valid_rotation_matrix(
            R
        ), f"Computed frame {R} is not a valid rotation matrix"
    computed_frames = [(R, np.array([0, 0, 0])) for R in Rs]

    # check that the computed frames match the input frames
    check_input_frames_match_symmetry_frames(computed_frames, input_frames)

    return computed_frames


def _align(X_fixed, X_moving):
    """
    Align two sets of coordinates using Kabsch algorithm.
    Arguments:
        X_fixed: fixed coordinates
        X_moving: moving coordinates
    Returns:
        u_X_moving: mean of the moving coordinates
        R: rotation matrix
        u_X_fixed: mean of the fixed coordinates
    """
    is_torch = isinstance(X_fixed, torch.Tensor)

    def _mean_along_dim(X, dim):
        if is_torch:
            return X.mean(dim=dim)
        else:
            return X.mean(axis=dim)

    assert X_fixed.shape == X_moving.shape

    if X_fixed.ndim == 2:
        X_fixed = X_fixed[None, ...]
        X_moving = X_moving[None, ...]
    B = X_fixed.shape[0]

    if is_torch:
        mask = (~torch.isnan(X_fixed) & ~torch.isnan(X_moving)).all(dim=-1).all(dim=0)
    else:
        mask = (~np.isnan(X_fixed) & ~np.isnan(X_moving)).all(axis=-1).all(axis=0)

    X_fixed = X_fixed[:, mask]
    X_moving = X_moving[:, mask]

    u_X_fixed = _mean_along_dim(X_fixed, dim=-2)
    u_X_moving = _mean_along_dim(X_moving, dim=-2)

    X_fixed_centered = X_fixed - u_X_fixed[..., None, :]
    X_moving_centered = X_moving - u_X_moving[..., None, :]

    if is_torch:
        C = torch.einsum("...ji,...jk->...ik", X_fixed_centered, X_moving_centered)
        U, S, V = torch.linalg.svd(C, full_matrices=False)
    else:
        C = np.einsum("...ji,...jk->...ik", X_fixed_centered, X_moving_centered)
        U, S, V = np.linalg.svd(C, full_matrices=False)

    R = U @ V
    if is_torch:
        F = torch.eye(3, 3, device=R.device).expand(B, 3, 3).clone()
        F[..., -1, -1] = torch.sign(torch.linalg.det(R))
    else:
        F = np.broadcast_to(np.eye(3, 3), (B, 3, 3)).copy()
        F[..., -1, -1] = np.sign(np.linalg.det(R))
    R = U @ F @ V

    if R.shape[0] == 1:
        return u_X_moving[0], R[0], u_X_fixed[0]

    return u_X_moving, R, u_X_fixed


def _rms(X_fixed, X_moving, t_pre, R, t_post):
    """
    Calculate the RMSD between two sets of coordinates.
    Arguments:
        X_fixed: fixed coordinates
        X_moving: moving coordinates
        t_pre: pre-rotation translation
        R: rotation matrix
        t_post: post-rotation translation
    Returns:
        rms: RMSD
    """
    mask = (~np.isnan(X_fixed) & ~np.isnan(X_moving)).all(axis=-1)
    X_fixed = X_fixed[mask]
    X_moving = X_moving[mask]

    X_moving_aln = np.einsum("ij,bj->bi", R, (X_moving - t_pre[None])) + t_post[None]
    rms = np.sqrt(np.sum(np.square(X_moving_aln - X_fixed)) / X_moving_aln.shape[-2])
    return rms


def is_valid_rotation_matrix(R):
    """
    check if a matrix is a valid rotation matrix.
    Arguments:
        R: rotation matrix
    Returns:
        bool: True if R is a valid rotation matrix, False otherwise
    """

    return np.allclose(R @ R.T, np.eye(3), atol=1e-6)


def get_cyclic_frames(order):
    """
    Get cyclic frames from a number of subunits.
    Arguments:
        order: number of subunits
    Returns:
        frames: list of rotation matrices
    """

    frames = []
    for i in range(order):
        angle = 2 * np.pi * i / order
        R = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        frames.append((R, np.array([0, 0, 0])))

    return frames


def get_dihedral_frames(order):
    """
    Get dihedral frames from a number of subunits.
    Arguments:
        order: number of subunits // 2 (since each dihedral has two frames)
    Returns:
        frames: list of rotation matrices
    """

    frames = []

    for i in range(order):
        angle = 2 * np.pi * i / order
        R = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )

        # 180 degree rotation in the xy-plane
        phi = angle + np.pi / order
        u = np.array([np.cos(phi), np.sin(phi), 0])
        flip = -np.eye(3) + 2 * np.outer(u, u)

        # add both frames for the dihedral
        frames.append((R, np.array([0, 0, 0])))
        frames.append((R @ flip, np.array([0, 0, 0])))

    return frames


def get_frames_from_file(file_path):
    raise NotImplementedError("Input defined symmetry not implemented")


###################################
# Kinematics
###################################


# fd - two routines that convert between:
#    a) a "virtual frame" consisting of three atoms; and
#    b) a translation and rotation
# uses Gram-Schmidt orthogonalziation, handles stacked/unstacked
# support np and torch inputs
def RTs_to_framecoords(Rs, ts, sig=1.0):
    if isinstance(Rs, np.ndarray):
        Rs = torch.from_numpy(Rs)
        ts = torch.from_numpy(ts)
    Ori = ts
    X = Ori + sig * Rs[..., 0, :] / (
        torch.norm(Rs[..., 0, :], dim=-1, keepdim=True) + 1e-6
    )
    Y = Ori + sig * Rs[..., 1, :] / (
        torch.norm(Rs[..., 1, :], dim=-1, keepdim=True) + 1e-6
    )
    return Ori, X, Y


# RTs_to_framecoords is used in loss and expects torch inputs
# (and must support backwards)
def framecoords_to_RTs(Ori, X, Y, eps=1e-6):
    R1 = X - Ori
    R1 = (R1 + torch.tensor([eps, 0, 0], device=R1.device)) / (
        torch.linalg.norm(R1, axis=-1, keepdims=True) + eps
    )

    Y_rel = Y - Ori
    proj = torch.sum(Y_rel * R1, axis=-1, keepdims=True) * R1
    R2 = Y_rel - proj
    R2 = (R2 + torch.tensor([0, eps, 0], device=R1.device)) / (
        torch.linalg.norm(R2, axis=-1, keepdims=True) + eps
    )

    R3 = torch.cross(R1, R2, dim=-1)

    # Stack into rotation matrix
    R = torch.stack([R1, R2, R3], axis=-2)  # shape (..., 3, 3)
    T = Ori

    return R, T


def pack_vector(v: np.ndarray) -> np.ndarray:
    """
    v: 1-D array of shape (3,) and arbitrary dtype
    returns: 1-element of shape 1
    """
    dt = np.dtype([("x", v.dtype, (3,))])
    a = np.zeros(1, dtype=dt)
    a["x"][0] = v
    return a


def unpack_vector(a: np.ndarray) -> np.ndarray:
    """
    a: stuctured array of shape (1,)
    returns: original vector
    """
    return a["x"]


def decompose_symmetry_frame(frame):
    R, T = frame
    Ori, X, Y = RTs_to_framecoords(R, T)
    Ori, X, Y = pack_vector(Ori.numpy()), pack_vector(X.numpy()), pack_vector(Y.numpy())
    return Ori, X, Y
