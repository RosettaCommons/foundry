import os
import torch
import pytest
from icecream import ic
from rf2aa.alignment import weighted_rigid_align, get_rmsd
from rf2aa.util import kabsch

def pseudobatched_kabsch(xyz1, xyz2):
    B = xyz1.shape[0]
    out = []
    for i in range(B):
        out.append(kabsch(xyz1[i], xyz2[i])[0])
    return torch.stack(out)

def test_align():
    torch.manual_seed(0)

    B = 9
    L = 5
    x_from = torch.rand((B, L, 3))
    x_to = torch.rand((B, L, 3))
    w = torch.ones((B, L))

    rmsd_kabsch = pseudobatched_kabsch(x_from, x_to)

    is_resolved = torch.ones((L), dtype=torch.bool)
    x_from_align = weighted_rigid_align(x_from, x_to, is_resolved, w)
    rmsd_weighted_rigid = get_rmsd(x_to, x_from_align)
    ic(rmsd_weighted_rigid, rmsd_kabsch)
    assert (torch.abs(rmsd_weighted_rigid - rmsd_kabsch) < 1e-5).all(), f'{rmsd_weighted_rigid} != {rmsd_kabsch}'    

if __name__ == '__main__':
    test_align()