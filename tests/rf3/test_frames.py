"""Unit tests for rf3.utils.frames.

These RF2AA-derived helpers build the rigid frames that anchor RF3's structural
losses. `rigid_from_3_points` constructs a per-residue orientation from the
backbone N/Ca/C atoms via Gram-Schmidt, then applies an idealization rotation
that nudges the frame toward canonical backbone geometry; the contract that is
not obvious from the body is that the returned `R` is still a proper rotation
(orthonormal, det +1) and the returned origin is exactly Ca. The module ships
with a `# TODO: ... HOPEFULLY TESTS` note and no tests, so the contracts are
pinned here on small CPU inputs. `is_atom` splits the sequence alphabet at
NNAPROTAAS (atom tokens are strictly above it).
"""

import torch
from rf3.chemical import NNAPROTAAS
from rf3.utils.frames import is_atom, rigid_from_3_points


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
