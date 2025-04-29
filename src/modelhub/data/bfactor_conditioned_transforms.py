from biotite.structure import AtomArray
from datahub.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_is_instance,
)
from datahub.transforms.base import Transform


class SetOccToZeroOnBfactor(Transform):
    """
    This component marks atoms as occ=0 based on bfactor values

    It takes as input 'brange', a list specifying the Mminimum and maximum B factors to
    keep.

    Example:
        brange = [-1.0,70.0] will mark with occ=0 any atom with b>70 or b<-1
    """

    def __init__(
        self,
        bmin=None,
        bmax=None,
    ):
        self.bmin = bmin
        self.bmax = bmax

    def check_input(self, data: dict):
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["occupancy"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        if self.bmin is None and self.bmax is None:
            return data

        assert (
            "b_factor" in atom_array.get_annotation_categories()
        ), "B factor annotation not found"
        bfact = atom_array.get_annotation("b_factor")
        if self.bmin is not None:
            mask = bfact < self.bmin
            if self.bmax is not None:
                mask = mask | (bfact > self.bmax)
        else:
            mask = bfact > self.bmax

        occ = atom_array.get_annotation("occupancy")
        occ[mask] = 0.0

        atom_array.set_annotation("occupancy", occ)

        data["atom_array"] = atom_array

        return data
