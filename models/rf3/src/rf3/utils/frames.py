# TODO: REFACTOR; COPIED FROM RF2AA. WE NEED TO ADD DOCSTRINGS, EXAMPLES, HOPEFULLY TESTS, AND CLEAN UP

import torch
from rf3.chemical import NFRAMES, NNAPROTAAS, costgtNA


def is_atom(seq):
    return seq > NNAPROTAAS


def get_frames(xyz_in, xyz_mask, seq, frame_indices, atom_frames=None):
    # B,L,natoms = xyz_in.shape[:3]
    frames = frame_indices[seq]
    atoms = is_atom(seq)
    if torch.any(atoms):
        frames[:, atoms[0].nonzero().flatten(), 0] = atom_frames

    frame_mask = ~torch.all(frames[..., 0, :] == frames[..., 1, :], axis=-1)

    # frame_mask *= torch.all(
    #     torch.gather(xyz_mask,2,frames.reshape(B,L,-1)).reshape(B,L,-1,3),
    #     axis=-1)

    return frames, frame_mask


# build a frame from 3 points
# fd  -  more complicated version splits angle deviations between CA-N and CA-C (giving more accurate CB position)
# fd  -  makes no assumptions about input dims (other than last 1 is xyz)
def rigid_from_3_points(N, Ca, C, is_na=None, eps=1e-4):
    dims = N.shape[:-1]

    v1 = C - Ca
    v2 = N - Ca
    e1 = v1 / (torch.norm(v1, dim=-1, keepdim=True) + eps)
    u2 = v2 - (torch.einsum("...li, ...li -> ...l", e1, v2)[..., None] * e1)
    e2 = u2 / (torch.norm(u2, dim=-1, keepdim=True) + eps)
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.cat(
        [e1[..., None], e2[..., None], e3[..., None]], axis=-1
    )  # [B,L,3,3] - rotation matrix

    v2 = v2 / (torch.norm(v2, dim=-1, keepdim=True) + eps)
    cosref = torch.sum(e1 * v2, dim=-1)

    costgt = torch.full(dims, -0.3616, device=N.device)
    if is_na is not None:
        costgt[is_na] = costgtNA

    cos2del = torch.clamp(
        cosref * costgt
        + torch.sqrt((1 - cosref * cosref) * (1 - costgt * costgt) + eps),
        min=-1.0,
        max=1.0,
    )

    cosdel = torch.sqrt(0.5 * (1 + cos2del) + eps)

    sindel = torch.sign(costgt - cosref) * torch.sqrt(1 - 0.5 * (1 + cos2del) + eps)

    Rp = torch.eye(3, device=N.device).repeat(*dims, 1, 1)
    Rp[..., 0, 0] = cosdel
    Rp[..., 0, 1] = -sindel
    Rp[..., 1, 0] = sindel
    Rp[..., 1, 1] = cosdel
    R = torch.einsum("...ij,...jk->...ik", R, Rp)

    return R, Ca


def mask_unresolved_frames_batched(frames, frame_mask, atom_mask):
    """
    reindex frames tensor from relative indices to absolute indices and masks out frames with atoms that are unresolved
    in the structure
    Input:
        - frames: relative indices for frames (B, L, nframes, 3)
        - frame_mask: mask for which frames are valid to compute FAPE/losses (B, L, nframes)
        - atom_mask: mask for seen coordinates (B, L, natoms)
    Output:
        - frames_reindex: absolute indices for frames
        - frame_mask_update: updated frame mask with frames with unresolved atoms removed
    """
    B, L, natoms = atom_mask.shape

    # reindex frames for flat X
    frames_reindex = (
        torch.arange(L, device=frames.device)[None, :, None, None] + frames[..., 0]
    ) * natoms + frames[..., 1]

    masked_atom_frames = torch.any(
        frames_reindex > L * natoms, dim=-1
    )  # find frames with atoms that aren't resolved
    masked_atom_frames *= torch.any(frames_reindex < 0, dim=-1)
    # There are currently indices for frames that aren't in the coordinates bc they arent resolved, reset these indices to 0 to avoid
    # indexing errors
    frames_reindex[masked_atom_frames, :] = 0

    frame_mask_update = frame_mask.clone()
    frame_mask_update *= ~masked_atom_frames
    frame_mask_update *= torch.all(
        torch.gather(
            atom_mask.reshape(B, L * natoms),
            1,
            frames_reindex.reshape(B, L * NFRAMES * 3),
        ).reshape(B, L, -1, 3),
        axis=-1,
    )

    return frames_reindex, frame_mask_update
