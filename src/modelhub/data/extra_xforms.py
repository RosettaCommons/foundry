import torch
from atomworks.ml.transforms._checks import (
    check_contains_keys,
)
from atomworks.ml.transforms.base import Transform


class CheckForNaNsInInputs(Transform):
    """
    This component marks atoms as occ=0 based on bfactor values

    It takes as input 'brange', a list specifying the Mminimum and maximum B factors to
    keep.

    Example:
        brange = [-1.0,70.0] will mark with occ=0 any atom with b>70 or b<-1
    """

    def check_input(self, data: dict):
        check_contains_keys(data, ["coord_atom_lvl_to_be_noised"])
        check_contains_keys(data, ["noise"])

    def forward(self, data: dict) -> dict:
        assert not torch.isnan(
            data["coord_atom_lvl_to_be_noised"]
        ).any(), "NaN found in network input"
        assert not torch.isnan(data["noise"]).any(), "NaN found in network noise"

        return data
