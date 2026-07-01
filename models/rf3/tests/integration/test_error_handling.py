"""Integration tests for graceful failure on bad inputs.

``rf3 fold`` should exit non-zero with a clear error rather than hang or crash
uncontrollably.  ``run_rf3_fold`` raises :class:`RuntimeError` on a non-zero
exit, so each test simply asserts that.

Both cases below fail during input discovery / parsing — *before* the model is
loaded — so they are fast (~15-20 s, dominated by import time).

Note: an *unknown CCD code* (e.g. ``ccd_code: "ZZ9"``) is deliberately not
tested here — ``rf3 fold`` tolerates it, folding the component as unknown atoms
with a warning rather than failing.
"""

import pytest
from conftest import run_rf3_fold


@pytest.mark.integration
def test_nonexistent_input_raises(require_ckpt, tmp_path):
    """Pointing ``inputs`` at a missing file fails cleanly."""
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(RuntimeError):
        run_rf3_fold(missing, tmp_path / "out")


@pytest.mark.integration
def test_malformed_json_raises(require_ckpt, tmp_path):
    """A syntactically invalid JSON input fails cleanly."""
    bad = tmp_path / "malformed.json"
    bad.write_text('[{"name": "malformed" "components": broken')
    with pytest.raises(RuntimeError):
        run_rf3_fold(bad, tmp_path / "out")
