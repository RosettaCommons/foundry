import torch

from modelhub.util_module import Dropout


def test_dropout():
    torch.manual_seed(0)
    drop_row = Dropout(broadcast_dim=0, p_drop=0.5)
    d = 8
    x = torch.rand((d, d))
    x = drop_row(x)
    print("x:")
    print(x)

    assert not torch.all(x == 0)
    for i in range(d):
        row = x[i]
        if torch.any(row == 0):
            assert torch.all(row == 0)

    has_all_zero_row = False
    for i in range(d):
        row = x[i]
        if torch.all(row == 0):
            has_all_zero_row = True

    assert has_all_zero_row
