"""Unit tests for the pure pieces of foundry.trainers.fabric.

``FabricTrainer`` is training-orchestration glue (fit / train / validation loops,
checkpoint save-load, optimizer/scheduler stepping) that needs a real model,
Fabric, and data loaders — integration territory, not CPU unit tests. The one
self-contained, pure piece is the static ``get_latest_checkpoint``, whose
directory-scan + lexicographic-sort contract is pinned here.

Note: ``get_latest_checkpoint`` returns ``None`` for an existing-but-empty
directory (and the ``fit()`` resume path does not guard that — tracked as a
latent bug in the roadmap). These tests pin the *current* behaviour.
"""

import pytest

from foundry.trainers.fabric import FabricTrainer


def test_get_latest_checkpoint_returns_none_for_missing_dir(tmp_path):
    assert FabricTrainer.get_latest_checkpoint(tmp_path / "does_not_exist") is None


def test_get_latest_checkpoint_returns_none_for_empty_dir(tmp_path):
    assert FabricTrainer.get_latest_checkpoint(tmp_path) is None


def test_get_latest_checkpoint_picks_highest_epoch(tmp_path):
    for epoch in (0, 1, 2):
        (tmp_path / f"epoch-{epoch:04d}.ckpt").write_text("x")

    latest = FabricTrainer.get_latest_checkpoint(tmp_path)

    assert latest == tmp_path / "epoch-0002.ckpt"


def test_get_latest_checkpoint_with_single_file(tmp_path):
    only = tmp_path / "epoch-0007.ckpt"
    only.write_text("x")

    assert FabricTrainer.get_latest_checkpoint(tmp_path) == only


if __name__ == "__main__":
    pytest.main(["-v", __file__])
