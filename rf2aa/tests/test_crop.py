import numpy as np
import torch

from rf2aa.data.loaders.crop import (
    contiguous_crop_index,
    get_preferred_chain_or_interface,
    radial_crop_index,
    select_preferred_token,
)
from rf2aa.util import get_protein_bond_feats

merged_outs = {
    "ch_letters_poly": ["A", "B", "C"],
    "akeys_sm": [
        [("a", "b", "c", "1"), ("a", "b", "c", "2")],
        [("x", "y", "z", "1"), ("x", "y", "z", "2")],
        [("l", "m", "n", "1")],
    ],
    "Ls_poly": [20, 25, 30],
    "Ls_sm": [10, 15, 10],
    "xyz": torch.concatenate(
        [
            torch.randn(1, 20, 36, 3) * 0.1,
            torch.randn(1, 25, 36, 3) * 0.1
            + torch.tensor([15.0, 0.0, 0.0]).reshape(1, 1, 1, 3),
            torch.randn(1, 30, 36, 3) * 0.1
            + torch.tensor([30.0, 0.0, 0.0]).reshape(1, 1, 1, 3),
            torch.randn(1, 10, 36, 3) * 0.1
            + torch.tensor([0.0, 5.0, 0.0]).reshape(1, 1, 1, 3),
            torch.randn(1, 15, 36, 3) * 0.1
            + torch.tensor([10.0, 5.0, 0.0]).reshape(1, 1, 1, 3),
            torch.zeros(1, 10, 36, 3),
        ],
        dim=1,
    ),
    "mask": torch.concatenate(
        [
            torch.zeros(1, 20, 36),
            torch.ones(1, 25, 36),
            torch.ones(1, 30, 36),
            torch.zeros(1, 10, 36),
            torch.ones(1, 15, 36),
            torch.ones(1, 10, 36),
        ],
        dim=1,
    ).bool(),
}
merged_outs["mask"][0, 0] = True
merged_outs["mask"][0, 75] = True
merged_outs["xyz"][0, 100:110, 1, 1] = torch.arange(10) * 5.0 + 11.0

bond_feats = [
    get_protein_bond_feats(L) for L in merged_outs["Ls_poly"] + merged_outs["Ls_sm"]
]
bond_feats = torch.block_diag(*bond_feats).long()
merged_outs["bond_feats"] = bond_feats

rng = np.random.default_rng(0)


def test_get_chain():
    preferred_chain, preferred_interface = get_preferred_chain_or_interface(
        merged_outs,
        {"preferred_chain": "A", "preferred_chain_type": "polypeptide(L)"},
        rng=rng,
    )
    assert preferred_chain == 0
    assert preferred_interface is None

    preferred_chain, preferred_interface = get_preferred_chain_or_interface(
        merged_outs,
        {"preferred_chain": [("a", "b", "c")], "preferred_chain_type": "nonpoly"},
        rng=rng,
    )
    assert preferred_chain == 3
    assert preferred_interface is None

    preferred_chain, preferred_interface = get_preferred_chain_or_interface(
        merged_outs,
        {
            "preferred_interface": ["C", [("x", "y", "z")]],
            "preferred_interface_type": ["polypeptide(L)", "nonpoly"],
        },
        rng=rng,
    )
    assert preferred_chain is None
    assert preferred_interface == (2, 4)


def test_get_token():
    token = select_preferred_token(
        merged_outs, preferred_chain=0, preferred_interface=None, rng=rng
    )
    assert token == 0
    token = select_preferred_token(
        merged_outs, preferred_chain=1, preferred_interface=None, rng=rng
    )
    assert token >= 20 and token < 45

    token = select_preferred_token(
        merged_outs, preferred_chain=None, preferred_interface=(0, 3), rng=rng
    )
    assert token == 0 or token == 75

    token = select_preferred_token(
        merged_outs, preferred_chain=None, preferred_interface=(1, 4), rng=rng
    )
    assert (token >= 20 and token < 45) or (token >= 85 and token < 100)


def test_radial_crop():
    crop_sel = radial_crop_index(merged_outs, crop_index=0, crop_size=20, rng=rng)
    assert torch.all(crop_sel < 20)

    crop_sel = radial_crop_index(merged_outs, crop_index=0, crop_size=30, rng=rng)
    assert torch.all(
        torch.isin(
            crop_sel, torch.concatenate([torch.arange(0, 20), torch.arange(75, 85)])
        )
    )


def test_contiguous_crop():
    crop_sel = contiguous_crop_index(merged_outs, crop_index=109, crop_size=5, rng=rng)
    assert torch.all(torch.isin(crop_sel, torch.arange(105, 110)))

    crop_sel = contiguous_crop_index(merged_outs, crop_index=109, crop_size=20, rng=rng)
    assert torch.all(
        torch.isin(
            crop_sel,
            torch.concatenate(
                [
                    torch.arange(75, 85),
                    torch.arange(100, 110),
                ]
            ),
        )
    )


test_contiguous_crop()
