"""Unit tests for foundry_cli.download_checkpoints.

The actual download path is network/file I/O glue and is not tested. The pieces
with a non-obvious, deterministic contract are pinned here: ``_resolve_checkpoint_dirs``
puts the user-requested directory first (inserting or moving it to the front),
and the ``list-available`` / ``list-installed`` commands report correctly on an
empty vs populated checkpoint directory.
"""

import pytest
from typer.testing import CliRunner

from foundry_cli import download_checkpoints as dc

runner = CliRunner()


@pytest.fixture
def no_env_persistence(monkeypatch):
    """Stop _resolve_checkpoint_dirs from touching a real .env file."""
    monkeypatch.setattr(dc, "append_checkpoint_to_env", lambda dirs: False)


def test_resolve_dirs_returns_defaults_when_none(monkeypatch, tmp_path):
    base = [tmp_path / "a", tmp_path / "b"]
    monkeypatch.setattr(dc, "get_default_checkpoint_dirs", lambda: list(base))

    assert dc._resolve_checkpoint_dirs(None) == base


def test_resolve_dirs_prepends_a_new_directory(
    monkeypatch, tmp_path, no_env_persistence
):
    base = [tmp_path / "a", tmp_path / "b"]
    monkeypatch.setattr(dc, "get_default_checkpoint_dirs", lambda: list(base))
    extra = (tmp_path / "extra").absolute()

    assert dc._resolve_checkpoint_dirs(extra) == [extra, *base]


def test_resolve_dirs_moves_an_existing_directory_to_front(
    monkeypatch, tmp_path, no_env_persistence
):
    a, b = (tmp_path / "a").absolute(), (tmp_path / "b").absolute()
    monkeypatch.setattr(dc, "get_default_checkpoint_dirs", lambda: [a, b])

    # Requesting 'b' (already present) moves it to the front without duplicating.
    assert dc._resolve_checkpoint_dirs(b) == [b, a]


def test_list_available_lists_registered_models():
    result = runner.invoke(dc.app, ["list-available"])

    assert result.exit_code == 0
    assert "Available models" in result.stdout
    assert "rf3" in result.stdout


def test_list_installed_reports_empty_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(dc, "get_default_checkpoint_dirs", lambda: [tmp_path])

    result = runner.invoke(dc.app, ["list-installed"])

    assert result.exit_code == 0
    assert "No checkpoint files found" in result.stdout


def test_list_installed_totals_populated_directory(monkeypatch, tmp_path):
    (tmp_path / "model.ckpt").write_bytes(b"x" * 1024)
    monkeypatch.setattr(dc, "get_default_checkpoint_dirs", lambda: [tmp_path])

    result = runner.invoke(dc.app, ["list-installed"])

    assert result.exit_code == 0
    assert "No checkpoint files found" not in result.stdout
    assert "Total:" in result.stdout


if __name__ == "__main__":
    pytest.main(["-v", __file__])
