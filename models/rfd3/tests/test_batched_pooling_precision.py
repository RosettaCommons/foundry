"""Batched atom means retain precision through division and ignore padding."""

import pytest
import torch
from rfd3.inference.batching import collate_examples
from rfd3.model.layers.batched import pool_atoms
from rfd3_batch_fixtures import example


def pooling_case(device, dtype):
    f = collate_examples(
        [example((3, 2), d=1), example((1,), d=1)],
        atom_capacity=8,
        token_capacity=4,
        slot_capacity=4,
        batch_capacity=3,
    )["f"]
    f = {k: v.to(device) for k, v in f.items()}
    x = torch.full((3, 1, 8, 1), float("nan"), device=device, dtype=dtype)
    values = (
        [-0.7578125, -0.6015625, -0.39453125]
        if dtype == torch.bfloat16
        else [-1.189453125, 1.3935546875, 2.10546875]
    )
    x[0, 0, :3, 0] = torch.tensor(values, device=device, dtype=dtype)
    x[0, 0, 3:5, 0] = 40000  # FP16 sum would overflow, but mean is representable.
    x[1, 0, 0, 0] = 7
    return x, f


def check_precision(device, dtype, compiled=False):
    x, f = pooling_case(device, dtype)
    x.requires_grad_()
    reference_input = x.detach().double().requires_grad_()
    reference = torch.stack(
        [
            reference_input[0, 0, :3].mean(0),
            reference_input[0, 0, 3:5].mean(0),
            reference_input[1, 0, 0],
        ]
    )
    fn = torch.compile(pool_atoms, fullgraph=True) if compiled else pool_atoms
    actual = fn(x, f)
    selected = torch.stack([actual[0, 0, 0], actual[0, 0, 1], actual[1, 0, 0]])
    torch.testing.assert_close(selected, reference.to(dtype), rtol=0, atol=0)
    assert torch.isfinite(actual).all()
    assert (actual[2] == 0).all() and (actual[:, :, 2:] == 0).all()
    weights = torch.tensor([[0.7], [1.2], [0.3]], dtype=dtype, device=device)
    got_grad = torch.autograd.grad((selected * weights).sum(), x)[0]
    ref_grad = torch.autograd.grad(
        (reference * weights.double()).sum(), reference_input
    )[0]
    torch.testing.assert_close(got_grad, ref_grad.to(dtype), rtol=0, atol=0)
    # This fixture must catch the former sum-in-output-dtype bug.
    old = x.detach()[0, 0, :3].sum(0) / 3
    assert not torch.equal(old, reference[0].to(dtype))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cpu_pooling_against_fp64(dtype):
    check_precision("cpu", dtype)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("compiled", [False, True])
def test_cuda_pooling_against_fp64(dtype, compiled):
    check_precision("cuda", dtype, compiled)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA"
            ),
        ),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_pair_pooling_keeps_fp32_under_autocast(device, dtype):
    from rfd3.model.layers.batched import pool_pairs

    x, f = pooling_case(device, dtype)
    pairs = x.unsqueeze(3).expand(-1, -1, -1, 8, -1).clone().requires_grad_()
    ref = pairs.detach().double().requires_grad_()
    expected = ref[0, 0, :3, :3].mean((0, 1)).to(dtype)
    with torch.autocast(device_type=device, dtype=dtype):
        out = pool_pairs(pairs, f)
    torch.testing.assert_close(out[0, 0, 0, 0], expected, atol=0, rtol=0)
    assert torch.isfinite(out).all()
    assert out[0, 0, 1, 1].item() == 40000
    assert (out[2] == 0).all() and (out[:, :, 2:] == 0).all()
    got_grad = torch.autograd.grad(out[0, 0, 0, 0].sum(), pairs)[0]
    ref_grad = torch.autograd.grad(ref[0, 0, :3, :3].mean((0, 1)).sum(), ref)[0]
    torch.testing.assert_close(got_grad, ref_grad.to(dtype), atol=0, rtol=0)
