import pytest
import torch
from itertools import product
from rf2aa.data.chain_crop import crop_chirals
from rf2aa.tensor_util import assert_equal

CHIRALS = [
    torch.zeros((0, 5)),
    torch.tensor(
        [
            [0, 1, 2, 3, -0.5],
            [2, 4, 8, 3, 0.5],
            [5, 80, 1, 10, -0.5],
            [100, 103, 104, 105, 0.5],
        ]
    ),
]

CROP_INDICES = [
    torch.tensor([]),
    torch.arange(2),
    torch.arange(4),
    torch.arange(8),
    torch.arange(10),
    torch.tensor([1, 5, 10, 80]),
    torch.tensor([0, 1, 2, 3, 4, 5, 8, 10, 80, 100, 103, 104, 105]),
    torch.arange(200),
]

INPUTS = list(product(CHIRALS, CROP_INDICES))

DESIRED_OUTPUTS = [torch.zeros((0, 5))] * len(CROP_INDICES) + [
    torch.zeros((0, 5)),
    torch.zeros((0, 5)),
    torch.tensor([[0, 1, 2, 3, -0.5]]),
    torch.tensor([[0, 1, 2, 3, -0.5]]),
    torch.tensor([[0, 1, 2, 3, -0.5], [2, 4, 8, 3, 0.5]]),
    torch.tensor([[1, 3, 0, 2, -0.5]]),
    torch.tensor(
        [
            [0, 1, 2, 3, -0.5],
            [2, 4, 6, 3, 0.5],
            [5, 8, 1, 7, -0.5],
            [9, 10, 11, 12, 0.5],
        ]
    ),
    torch.tensor(
        [
            [0, 1, 2, 3, -0.5],
            [2, 4, 8, 3, 0.5],
            [5, 80, 1, 10, -0.5],
            [100, 103, 104, 105, 0.5],
        ]
    ),
]


@pytest.mark.parametrize("inputs, desired_output", zip(INPUTS, DESIRED_OUTPUTS))
def test_chirals(inputs, desired_output):
    chiral, crop_index = inputs
    output = crop_chirals(chiral, crop_index)
    assert_equal(output, desired_output)
