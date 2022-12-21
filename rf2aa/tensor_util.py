import torch
import assertpy

def assert_shape(t, s):
    assertpy.assert_that(tuple(t.shape)).is_equal_to(s)

def assert_equal(got, want):
    assertpy.assert_that(got.dtype).is_equal_to(want.dtype)
    assertpy.assert_that(got.shape).is_equal_to(want.shape)
    is_eq = got.nan_to_num()==want.nan_to_num()
    unequal_idx = torch.nonzero(~is_eq)
    unequal_got = got[~is_eq]
    unequal_want = want[~is_eq]
    uneq_idx_got_want = list(zip(unequal_idx, unequal_want, unequal_got))[:3]
    uneq_msg = '\t'.join(f'idx:{idx}, got:{got}, want:{want}' for idx, got, want in uneq_idx_got_want)
    msg = f'tensors with shape {got.shape}: first unequal indices: {uneq_idx_got_want}'
    # m = is_eq.float().mean().item()
    assertpy.assert_that(len(unequal_idx)).described_as(msg).is_equal_to(0)