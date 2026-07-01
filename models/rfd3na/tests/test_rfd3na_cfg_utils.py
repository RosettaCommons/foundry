"""Unit tests for ``rfd3na.model.cfg_utils`` (classifier-free-guidance helpers).

``strip_f`` does two things to a conditioning-feature dict:

1. **Crop** — it removes the unindexed-motif tail. Cropping only happens when there is at
   least one motif *atom* (``crop = any(is_motif_atom_unindexed)``); then token-shaped
   features are cropped to the first motif *token* and atom-shaped features to the first
   motif *atom* (2-D square features are cropped on both axes). A feature is identified as
   token- or atom-shaped by whether ``is_motif_token_unindexed``'s / ``is_motif_atom_unindexed``'s
   length appears in its shape.
2. **Zero** — any key in ``cfg_features`` is reset to its default: zeros, except
   ``bp_partners``, whose default is a "no partner" mask (``[..., 0] = 1``, ``[..., 1:] = 0``).

``strip_X`` crops the atom axis of the coordinate tensor to the (already-cropped) atom count.
"""

import torch
from rfd3na.model.cfg_utils import strip_f, strip_X

# Token dim (4) and atom dim (6) are kept distinct so feature classification is unambiguous.


def _base_masks(token_motif: list[bool], atom_motif: list[bool]) -> dict:
    return {
        "is_motif_token_unindexed": torch.tensor(token_motif),
        "is_motif_atom_unindexed": torch.tensor(atom_motif),
    }


# --- strip_f: no motif atoms => no cropping ---------------------------------


def test_no_motif_atoms_leaves_shapes_unchanged():
    f = _base_masks([False, False, False, False], [False] * 6)
    f["tok1d"] = torch.arange(4).float()
    f["atom1d"] = torch.arange(6).float()
    f["pair2d"] = torch.arange(16).float().reshape(4, 4)
    out = strip_f(f, cfg_features=[])
    assert out["tok1d"].shape == (4,)
    assert out["atom1d"].shape == (6,)
    assert out["pair2d"].shape == (4, 4)


def test_cfg_feature_is_zeroed_even_without_cropping():
    f = _base_masks([False] * 4, [False] * 6)
    f["tok1d"] = torch.arange(4).float() + 1
    out = strip_f(f, cfg_features=["tok1d"])
    assert torch.equal(out["tok1d"], torch.zeros(4))


# --- strip_f: motif atoms present => crop at first motif index --------------


def test_crops_token_and_atom_features_at_first_motif_index():
    # first motif token at index 2, first motif atom at index 3.
    f = _base_masks([False, False, True, True], [False, False, False, True, True, True])
    f["tok1d"] = torch.arange(4).float()
    f["atom1d"] = torch.arange(6).float()
    out = strip_f(f, cfg_features=[])
    assert out["tok1d"].tolist() == [0.0, 1.0]
    assert out["atom1d"].tolist() == [0.0, 1.0, 2.0]


def test_crops_2d_square_token_feature_on_both_axes():
    f = _base_masks([False, False, True, True], [False, False, False, True, True, True])
    f["pair2d"] = torch.arange(16).float().reshape(4, 4)
    out = strip_f(f, cfg_features=[])
    assert out["pair2d"].shape == (2, 2)
    assert out["pair2d"].tolist() == [[0.0, 1.0], [4.0, 5.0]]


def test_crop_is_driven_by_atom_mask_not_token_mask():
    # Motif tokens but NO motif atoms => crop is False => nothing is cropped.
    f = _base_masks([False, False, True, True], [False] * 6)
    f["tok1d"] = torch.arange(4).float()
    out = strip_f(f, cfg_features=[])
    assert out["tok1d"].shape == (4,)


# --- strip_f: bp_partners default is a mask, not zeros ----------------------


def test_bp_partners_reset_to_no_partner_mask():
    f = _base_masks([False, False], [False, False])
    f["bp_partners"] = torch.arange(2 * 2 * 3).float().reshape(2, 2, 3)
    out = strip_f(f, cfg_features=["bp_partners"])
    assert torch.equal(out["bp_partners"][..., 0], torch.ones(2, 2))
    assert torch.equal(out["bp_partners"][..., 1:], torch.zeros(2, 2, 2))


# --- strip_X ----------------------------------------------------------------


def test_strip_x_crops_atom_axis_to_stripped_atom_count():
    x = torch.arange(2 * 10 * 3).float().reshape(2, 10, 3)
    f_stripped = {"is_motif_atom_unindexed": torch.zeros(4)}
    out = strip_X(x, f_stripped)
    assert out.shape == (2, 4, 3)
    assert torch.equal(out, x[:, :4, :])
