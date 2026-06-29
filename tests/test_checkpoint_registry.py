"""Unit tests for the pure path-resolution logic in
foundry.inference_engines.checkpoint_registry.

The download/dotenv-writing helpers are side-effecting glue. The path search
order is load-bearing (it decides which checkpoint directory wins) and is pinned
here, including the fact that directories from ``FOUNDRY_CHECKPOINT_DIRS`` are
searched *before* the default ``~/.foundry/checkpoints`` directory.
"""

from pathlib import Path

import pytest

from foundry.inference_engines import checkpoint_registry as cr
from foundry.inference_engines.checkpoint_registry import (
    RegisteredCheckpoint,
    _normalize_paths,
    get_default_checkpoint_dir,
    get_default_checkpoint_dirs,
)


def test_normalize_paths_absolutizes_and_dedupes_preserving_order(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    result = _normalize_paths([a, b, a])

    assert result == [a, b]  # duplicate 'a' dropped, order preserved
    assert all(p.is_absolute() for p in result)


def test_normalize_paths_treats_equivalent_relative_paths_as_one():
    result = _normalize_paths([Path("x"), Path("./x")])
    assert result == [Path("x").absolute()]


def _clear_env(monkeypatch):
    monkeypatch.delenv("FOUNDRY_CHECKPOINT_DIRS", raising=False)
    monkeypatch.delenv("FOUNDRY_CHECKPOINTS_DIR", raising=False)


def test_default_dirs_is_just_default_when_env_unset(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    default = tmp_path / "default"
    monkeypatch.setattr(cr, "DEFAULT_CHECKPOINT_DIR", default)

    assert get_default_checkpoint_dirs() == [default]


def test_env_dirs_are_searched_before_the_default(tmp_path, monkeypatch):
    default = tmp_path / "default"
    monkeypatch.setattr(cr, "DEFAULT_CHECKPOINT_DIR", default)
    a, b = tmp_path / "a", tmp_path / "b"
    monkeypatch.setenv("FOUNDRY_CHECKPOINT_DIRS", f"{a}:{b}")

    assert get_default_checkpoint_dirs() == [a, b, default]
    # The "primary" dir is the first env dir, not the default.
    assert get_default_checkpoint_dir() == a


def test_legacy_env_var_is_used_when_new_one_is_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("FOUNDRY_CHECKPOINT_DIRS", raising=False)
    default = tmp_path / "default"
    monkeypatch.setattr(cr, "DEFAULT_CHECKPOINT_DIR", default)
    legacy = tmp_path / "legacy"
    monkeypatch.setenv("FOUNDRY_CHECKPOINTS_DIR", str(legacy))

    assert get_default_checkpoint_dirs() == [legacy, default]


def test_get_default_path_returns_first_existing_file(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    default = tmp_path / "default"
    monkeypatch.setattr(cr, "DEFAULT_CHECKPOINT_DIR", default)
    extra = tmp_path / "extra"
    extra.mkdir()
    (extra / "model.ckpt").write_text("weights")
    monkeypatch.setenv("FOUNDRY_CHECKPOINT_DIRS", str(extra))

    ckpt = RegisteredCheckpoint(url="u", filename="model.ckpt", description="d")
    assert ckpt.get_default_path() == extra / "model.ckpt"


def test_get_default_path_falls_back_to_primary_dir_when_missing(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    default = tmp_path / "default"
    monkeypatch.setattr(cr, "DEFAULT_CHECKPOINT_DIR", default)

    ckpt = RegisteredCheckpoint(url="u", filename="absent.ckpt", description="d")
    # Nothing exists anywhere -> the primary (first) dir / filename.
    assert ckpt.get_default_path() == default / "absent.ckpt"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
