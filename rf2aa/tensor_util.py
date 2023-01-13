import torch
import pprint
import assertpy
import dataclasses
from collections import OrderedDict

def assert_shape(t, s):
    assertpy.assert_that(tuple(t.shape)).is_equal_to(s)

def assert_equal(got, want):
    assertpy.assert_that(got.dtype).is_equal_to(want.dtype)
    assertpy.assert_that(got.shape).is_equal_to(want.shape)
    is_eq = got.nan_to_num()==want.nan_to_num()
    unequal_idx = torch.nonzero(~is_eq)
    unequal_got = got[~is_eq]
    unequal_want = want[~is_eq]
    uneq_idx_got_want = list(zip(unequal_idx.tolist(), unequal_want, unequal_got))[:3]

    uneq_msg = '\t'.join(f'idx:{idx}, got:{got}, want:{want}' for idx, got, want in uneq_idx_got_want)
    msg = f'tensors with shape {got.shape}: first unequal indices: {uneq_msg}'
    if torch.numel(got) < 10:
        msg = f'got {got}, want: {want}'
    assert len(unequal_idx) == 0, msg

# Dataclass functions

def to_ordered_dict(dc):
    return OrderedDict((field.name, getattr(dc, field.name)) for field in dataclasses.fields(dc))

def to_device(dc, device):
    d = to_ordered_dict(dc)
    for k, v in d.items():
        setattr(dc, k, v.to(device))

def shapes(dc):
    d = to_ordered_dict(dc)
    return {k:v.shape if hasattr(v, 'shape') else None for k,v in d.items()}

def pprint_obj(obj):
    pprint.pprint(obj.__dict__, indent=4)
