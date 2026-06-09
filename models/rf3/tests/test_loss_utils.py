"""Unit tests for rf3.utils.loss — the batched-loss bookkeeping helpers.

- ``convert_batched_losses_to_list_of_dicts`` splits a loss dict into one dict per
  batch index (carrying only the 1-D / batched entries, tagged with ``batch_idx``)
  followed by a single trailing dict of the 0-D / non-batched (scalar) entries.
- ``mean_losses`` reduces each tensor in a loss dict to its scalar mean.
"""

import pytest
import torch
from rf3.utils.loss import convert_batched_losses_to_list_of_dicts, mean_losses


def test_convert_splits_batched_and_scalar_entries():
    loss_dict = {
        "diffusion_loss": torch.tensor([0.05, 0.006]),
        "t": torch.tensor([1.7, 9.3]),
        "distogram_loss": torch.tensor(1.76),  # scalar (0-D)
    }
    out = convert_batched_losses_to_list_of_dicts(loss_dict)

    # Two per-batch dicts (batch_size inferred from the 1-D entries) + one scalar dict.
    assert len(out) == 3
    assert out[0]["batch_idx"] == 0 and out[1]["batch_idx"] == 1
    assert out[0]["diffusion_loss"] == pytest.approx(0.05, abs=1e-4)
    assert out[1]["t"] == pytest.approx(9.3, abs=1e-4)
    # Scalar entries never leak into the per-batch dicts...
    assert "distogram_loss" not in out[0]
    # ...they land in the trailing non-batched dict, which carries no batch_idx.
    assert "batch_idx" not in out[2]
    assert out[2]["distogram_loss"] == pytest.approx(1.76, abs=1e-4)


def test_convert_all_scalar_defaults_to_single_batch():
    # No 1-D entries → batch_size falls back to 1: one empty-but-tagged batch dict
    # plus the scalar dict.
    out = convert_batched_losses_to_list_of_dicts({"total_loss": torch.tensor(1.2)})
    assert len(out) == 2
    assert out[0] == {"batch_idx": 0}
    assert out[1]["total_loss"] == pytest.approx(1.2, abs=1e-4)


def test_mean_losses_reduces_each_entry():
    out = mean_losses({"loss1": torch.tensor([0.5, 0.7]), "loss2": torch.tensor([1.0])})
    assert out["loss1"] == pytest.approx(0.6, abs=1e-6)
    assert out["loss2"] == pytest.approx(1.0, abs=1e-6)
