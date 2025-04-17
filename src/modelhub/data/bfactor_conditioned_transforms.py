from collections import defaultdict
from typing import Any, Literal

import numpy as np
import toolz
import torch
from biotite.structure import AtomArray
from datahub.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_is_instance,
    check_nonzero_length,
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
        brange,
    ):
        self.bmin = brange[0]
        self.bmax = brange[1]

    def check_input(self, data: dict):
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(
            data, ["b_factor", "occupancy"]
        )

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        bfact = atom_array.get_annotation('b_factor')
        mask = (bfact<self.bmin) | (bfact>self.bmax)
        occ = atom_array.get_annotation('occupancy')
        occ[mask] = 0.0
        atom_array.set_annotation('occupancy',occ)

        data["atom_array"] = atom_array

        return data

