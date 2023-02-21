import torch
import numpy as np
import json
from deepdiff import DeepDiff
import pprint
import assertpy
import dataclasses
from collections import OrderedDict
import contextlib
from deepdiff.operator import BaseOperator

def assert_shape(t, s):
    assertpy.assert_that(tuple(t.shape)).is_equal_to(s)

def assert_same_shape(t, s):
    assertpy.assert_that(tuple(t.shape)).is_equal_to(tuple(s.shape))

class ExceptionLogger(contextlib.AbstractContextManager):

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            print("***Logging exception {}***".format((exc_type, exc_value, 
                traceback)))


def assert_equal(got, want):
    assertpy.assert_that(got.dtype).is_equal_to(want.dtype)
    assertpy.assert_that(got.shape).is_equal_to(want.shape)
    is_eq = got.nan_to_num()==want.nan_to_num()
    unequal_idx = torch.nonzero(~is_eq)
    unequal_got = got[~is_eq]
    unequal_want = want[~is_eq]
    uneq_idx_got_want = list(zip(unequal_idx.tolist(), unequal_want, unequal_got))[:3]

    uneq_msg = '    '.join(f'idx:{idx}, got:{got}, want:{want}' for idx, got, want in uneq_idx_got_want)
    msg = f'tensors with shape {got.shape}: first unequal indices: {uneq_msg}'
    if torch.numel(got) < 10:
        msg = f'got {got}, want: {want}'
    assert len(unequal_idx) == 0, msg

def cpu(e):
    if isinstance(e, dict):
        return {cpu(k): cpu(v) for k,v in e.items()}
    if isinstance(e, list) or isinstance(e, tuple):
        return tuple(cpu(i) for i in e)
    if hasattr(e, 'cpu'):
        return e.cpu().detach()
    return e

# Dataclass functions

def to_ordered_dict(dc):
    return OrderedDict((field.name, getattr(dc, field.name)) for field in dataclasses.fields(dc))

def to_device(dc, device):
    d = to_ordered_dict(dc)
    for k, v in d.items():
        if v is not None:
            setattr(dc, k, v.to(device))

def shapes(dc):
    d = to_ordered_dict(dc)
    return {k:v.shape if hasattr(v, 'shape') else None for k,v in d.items()}


def pprint_obj(obj):
    pprint.pprint(obj.__dict__, indent=4)

def assert_squeeze(t):
    assert t.shape[0] == 1, f'{t.shape}[0] != 1'
    return t[0]

def apply_to_tensors(e, op):
    # if isinstance(e, dataclasses.)
    if dataclasses.is_dataclass(e):
        # return to_ordered_dict
        return type(e)(*(apply_to_tensors(getattr(e, field.name), op) for field in dataclasses.fields(e)))
    if isinstance(e, dict):
        return {k: apply_to_tensors(v, op) for k,v in e.items()}
    if isinstance(e, list) or isinstance(e, tuple):
        return tuple(apply_to_tensors(i, op) for i in e)
    if hasattr(e, 'cpu'):
        return op(e)
    return e

def apply_to_matching(e, op, filt):
    # if isinstance(e, dataclasses.)
    if filt(e):
        return op(e)
    if dataclasses.is_dataclass(e):
        # return to_ordered_dict
        return type(e)(*(apply_to_tensors(getattr(e, field.name), op) for field in dataclasses.fields(e)))
    if isinstance(e, dict):
        return {k: apply_to_tensors(v, op) for k,v in e.items()}
    if isinstance(e, list) or isinstance(e, tuple):
        return tuple(apply_to_tensors(i, op) for i in e)
    return e

def set_grad(t):
    t.requires_grad = True

def require_grad(e):
    apply_to_tensors(e, set_grad)

def get_grad(e):
    return apply_to_tensors(e, lambda x: x.grad)

def info(e):
    shap = apply_to_tensors(e, lambda x: x.shape)
    shap = apply_to_matching(shap, str, dataclasses.is_dataclass)
    return json.dumps(shap, indent=4)

def minmax(e):
    return apply_to_tensors(e, lambda x: (torch.log10(torch.min(x)), torch.log10(torch.max(x))))

class TensorMatchOperator(BaseOperator):


    def __init__(self, atol=1e-3, rtol=0, **kwargs):
        super(TensorMatchOperator, self).__init__(**kwargs)
        self.atol = atol
        self.rtol = rtol
    
    def _equal_msg(self, got, want):
        if got.shape != want.shape:
            return f'got shape {got.shape} want shape {want.shape}'
        if got.dtype != want.dtype:
            return f'got dtype {got.dtype} want dtype {want.dtype}'
        if torch.isclose(got, want, equal_nan=True, atol=self.atol, rtol=self.rtol).all():
            return ''
        is_eq = torch.isclose(got, want, equal_nan=True, atol=self.atol, rtol=self.rtol)
        unequal_idx = torch.nonzero(~is_eq)
        unequal_got = got[~is_eq]
        unequal_want = want[~is_eq]
        uneq_idx_got_want = list(zip(unequal_idx.tolist(), unequal_got, unequal_want))[:3]
        uneq_msg = '    '.join(f'idx:{idx}, got:{got}, want:{want}' for idx, got, want in uneq_idx_got_want)
        uneq_msg += f'    fraction unequal:{unequal_got.numel()}/{got.numel()}'
        msg = f'tensors with shape {got.shape}: first unequal indices: {uneq_msg}'
        if torch.numel(got) < 10:
            msg = f'got {got}, want: {want}'
        return msg

    
    def give_up_diffing(self, level, diff_instance):
        msg = self._equal_msg(level.t1, level.t2)
        if msg:
            print(f'got:\n{level.t1}\n\nwant:\n{level.t2}')
            msg = self._equal_msg(level.t1, level.t2)
            if msg:
                diff_instance.custom_report_result('tensors unequal', level, {
                        "msg": msg
                    })
        return True

class NumpyMatchOperator(TensorMatchOperator):
    def give_up_diffing(self, level, diff_instance):
        level.t1 = torch.Tensor(level.t1)
        level.t2 = torch.Tensor(level.t2)
        return super(NumpyMatchOperator, self).give_up_diffing(level, diff_instance)


def cmp(got, want, **kwargs):
    
    dd = DeepDiff(got, want, custom_operators=[
        NumpyMatchOperator(types=[np.ndarray], **kwargs),
        TensorMatchOperator(types=[torch.Tensor], **kwargs)])
    if dd:
        return dd
    return ''
