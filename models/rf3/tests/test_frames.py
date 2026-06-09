"""Unit tests for rf3.utils.frames.

These RF2AA-derived helpers build the rigid frames that anchor RF3's structural
losses. `rigid_from_3_points` constructs a per-residue orientation from the
backbone N/Ca/C atoms via Gram-Schmidt, then applies an idealization rotation
that nudges the frame toward canonical backbone geometry; the contract that is
not obvious from the body is that the returned `R` is still a proper rotation
(orthonormal, det +1) and the returned origin is exactly Ca. The module ships
with a `# TODO: ... HOPEFULLY TESTS` note and no tests, so the contracts are
pinned here on small CPU inputs. `is_atom` splits the sequence alphabet at
NNAPROTAAS (atom tokens are strictly above it). `get_frames` looks up each
token's frames from a `[token, NFRAMES, 3, 2]` table (the trailing 2 is a
`(residue_offset, atom_index)` pair) and marks a frame valid only when its first
two atoms differ; `mask_unresolved_frames_batched` reindexes those relative
frames to absolute flat atom indices and drops any frame with an unresolved atom.
"""

import torch
from rf3.chemical import NFRAMES, NNAPROTAAS
from rf3.utils.frames import (
    get_frames,
    is_atom,
    mask_unresolved_frames_batched,
    rigid_from_3_points,
)


def _is_proper_rotation(R: torch.Tensor, atol: float = 1e-3) -> bool:
    """True iff every R[..., 3, 3] is orthonormal and has det +1 (no reflection).

    The idealization step leaves det within ~2e-4 of 1, so det is checked with a
    looser tolerance than orthonormality.
    """
    eye = torch.eye(3).expand_as(R)
    orthonormal = torch.allclose(R @ R.transpose(-1, -2), eye, atol=atol)
    det = torch.linalg.det(R)
    return (
        orthonormal
        and bool((det > 0).all())
        and torch.allclose(det, torch.ones_like(det), atol=1e-2)
    )


# --- rigid_from_3_points ------------------------------------------------------


def test_rigid_from_3_points_returns_proper_rotation():
    N = torch.tensor([[0.0, 1.0, 0.0]])
    Ca = torch.tensor([[0.0, 0.0, 0.0]])
    C = torch.tensor([[1.0, 0.0, 0.0]])
    R, _ = rigid_from_3_points(N, Ca, C)
    assert R.shape == (1, 3, 3)
    assert _is_proper_rotation(R)


def test_rigid_from_3_points_origin_is_ca():
    N = torch.tensor([[0.0, 1.0, 0.0]])
    Ca = torch.tensor([[2.0, -1.0, 3.0]])
    C = torch.tensor([[1.0, 0.0, 0.0]])
    _, t = rigid_from_3_points(N, Ca, C)
    assert torch.equal(t, Ca)


def test_rigid_from_3_points_preserves_batch_dims():
    torch.manual_seed(0)
    N, Ca, C = (torch.randn(2, 4, 3) for _ in range(3))
    R, t = rigid_from_3_points(N, Ca, C)
    assert R.shape == (2, 4, 3, 3)
    assert t.shape == (2, 4, 3)
    assert _is_proper_rotation(R)


def test_rigid_from_3_points_na_path_is_proper_and_differs():
    # The is_na flag swaps the idealization target angle (costgt -> costgtNA), so
    # the nucleic-acid frame is a different proper rotation than the protein one.
    N = torch.tensor([[0.0, 1.0, 0.0]])
    Ca = torch.tensor([[0.0, 0.0, 0.0]])
    C = torch.tensor([[1.0, 0.0, 0.0]])
    R_protein, _ = rigid_from_3_points(N, Ca, C)
    R_na, _ = rigid_from_3_points(N, Ca, C, is_na=torch.tensor([True]))
    assert _is_proper_rotation(R_na)
    assert not torch.allclose(R_protein, R_na, atol=1e-3)


# --- is_atom ------------------------------------------------------------------


def test_is_atom_splits_strictly_above_nnaprotaas():
    seq = torch.tensor([0, NNAPROTAAS - 1, NNAPROTAAS, NNAPROTAAS + 1])
    assert is_atom(seq).tolist() == [False, False, False, True]


# --- get_frames ---------------------------------------------------------------


def test_get_frames_looks_up_per_token_and_flags_degenerate():
    # frame_indices[token] holds NFRAMES frames, each 3 atoms of (offset, atom_idx).
    # A frame is valid iff its first two atoms differ: type 0 differs (valid),
    # type 1 repeats atom 0 (degenerate -> invalid). xyz_in/xyz_mask are unused
    # (production passes literal 0s), so the lookup is driven only by seq.
    frame_indices = torch.tensor(
        [
            [[[0, 0], [0, 1], [0, 2]]],  # type 0: atom0 != atom1 -> valid
            [[[0, 5], [0, 5], [0, 7]]],  # type 1: atom0 == atom1 -> invalid
        ],
        dtype=torch.long,
    )
    seq = torch.tensor([[0, 1]])  # B=1, L=2, both protein tokens (is_atom False)
    frames, frame_mask = get_frames(0, 0, seq, frame_indices)

    assert frames.shape == (1, 2, 1, 3, 2)
    assert torch.equal(frames[0, 0], frame_indices[0])
    assert torch.equal(frames[0, 1], frame_indices[1])
    assert frame_mask.tolist() == [[[True], [False]]]


# --- mask_unresolved_frames_batched -------------------------------------------


def test_mask_unresolved_reindexes_and_drops_frames_with_unresolved_atoms():
    # frames must carry exactly NFRAMES frames — the function reshapes with the
    # chemical NFRAMES constant, not the input's frame dim. All frames here are
    # in-bounds (residue offset 0, atoms 0/1/2), so residue r reindexes to the
    # flat atom indices [3r, 3r+1, 3r+2] (natoms = 3).
    frames = torch.zeros(1, 2, NFRAMES, 3, 2, dtype=torch.long)
    frames[..., 1] = torch.tensor([0, 1, 2])  # atom indices; residue offset stays 0
    frame_mask = torch.ones(1, 2, NFRAMES, dtype=torch.bool)
    # residue 0 fully resolved, residue 1 has atom 2 unresolved.
    atom_mask = torch.tensor([[[True, True, True], [True, True, False]]])

    reindex, mask_update = mask_unresolved_frames_batched(frames, frame_mask, atom_mask)

    assert reindex.shape == (1, 2, NFRAMES, 3)
    assert reindex[0, 0, 0].tolist() == [0, 1, 2]
    assert reindex[0, 1, 0].tolist() == [3, 4, 5]
    # A frame survives only if all 3 of its atoms are resolved: residue 0's frames
    # are kept, residue 1's are dropped (atom 2 unresolved).
    assert bool(mask_update[0, 0].all())
    assert not bool(mask_update[0, 1].any())
