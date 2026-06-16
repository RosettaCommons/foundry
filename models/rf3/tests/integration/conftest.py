"""Shared fixtures for RF3 end-to-end integration tests.

These tests invoke the real ``rf3 fold`` CLI against a downloaded model
checkpoint and are excluded from the default ``pytest`` run (``testpaths``
only covers ``tests/``). Run them explicitly with::

    pytest models/rf3/tests/integration/ -m integration

The RF3 checkpoint must be available. Set the ``RF3_CKPT_PATH`` environment
variable to its absolute path, or place it at the default location::

    ~/.foundry/checkpoints/rf3_foundry_01_24_latest_remapped.ckpt

Download with::

    wget -P ~/.foundry/checkpoints \\
        http://files.ipd.uw.edu/pub/rf3/rf3_foundry_01_24_latest_remapped.ckpt

All ``rf3 fold`` calls in these tests use reduced parameters to keep the total
wall-clock time under 15 minutes on a GitHub Actions CPU runner::

    n_recycles=1          (default 10)
    num_steps=20          (default 50)
    diffusion_batch_size=1  (default 5)
    seed=1

Session-scoped fixtures amortise model-loading cost: each distinct flag
combination gets exactly one ``rf3 fold`` subprocess call, and multiple test
functions share that result.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data"
GPU_BASELINE_DIR = DATA_DIR / "integration_baselines"

# Resolve the rf3 executable from the same venv that is running pytest so the
# subprocess inherits the correct installation without relying on PATH.
_RF3_BIN = Path(sys.executable).parent / "rf3"

_env_ckpt = os.environ.get("RF3_CKPT_PATH")
CKPT_PATH = (
    Path(_env_ckpt)
    if _env_ckpt
    else Path.home()
    / ".foundry"
    / "checkpoints"
    / "rf3_foundry_01_24_latest_remapped.ckpt"
)

# Reduce compute so the full suite finishes within the CI time budget.
# early_stopping_plddt_threshold=0.0 disables the default threshold (0.5) so
# that no fixture unexpectedly early-stops on a future low-pLDDT test input.
SPEED_FLAGS = [
    "n_recycles=1",
    "num_steps=20",
    "diffusion_batch_size=1",
    "seed=1",
    "early_stopping_plddt_threshold=0.0",
]

# Per-fold subprocess timeout (seconds).  Set high enough to cover the
# worst-case fixture (basic_folds_dir batches three inputs in one call).
# Individual hangs are still caught; CI runners finish well within this limit.
_FOLD_TIMEOUT = 1800


# ---------------------------------------------------------------------------
# Helpers (importable by test modules via `from conftest import ...`)
# ---------------------------------------------------------------------------


def run_rf3_fold(inputs, out_dir, extra_flags=None):
    """Invoke ``rf3 fold`` via subprocess and return the output directory.

    Parameters
    ----------
    inputs:
        A single ``Path``/``str`` or a list of paths. Lists are formatted
        with Hydra list syntax automatically.
    out_dir:
        Destination passed to ``out_dir=``.
    extra_flags:
        Additional Hydra overrides appended after the speed flags.

    Returns
    -------
    tuple[Path, str]
        ``(out_dir, stderr)`` — the output directory and the captured stderr text.

    Raises
    ------
    RuntimeError
        When ``rf3 fold`` exits with a non-zero return code.
    subprocess.TimeoutExpired
        When the call exceeds ``_FOLD_TIMEOUT`` seconds.
    """
    if isinstance(inputs, (str, Path)):
        inputs_arg = f"inputs={inputs}"
    else:
        joined = ", ".join(str(p) for p in inputs)
        inputs_arg = f"inputs=[{joined}]"

    cmd = (
        [str(_RF3_BIN), "fold"]
        + SPEED_FLAGS
        + [f"ckpt_path={CKPT_PATH}", inputs_arg, f"out_dir={out_dir}"]
        + (extra_flags or [])
    )
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=_FOLD_TIMEOUT)
    if result.returncode != 0:
        raise RuntimeError(
            f"rf3 fold failed (exit {result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    return Path(out_dir), result.stderr


def load_summary(out_dir, name):
    """Return the parsed ``summary_confidences.json`` for *name*."""
    path = out_dir / name / f"{name}_summary_confidences.json"
    return json.loads(path.read_text())


def assert_standard_outputs(out_dir, name):
    """Assert that all four standard output files exist for *name*."""
    base = out_dir / name
    assert base.is_dir(), f"output directory missing: {base}"
    for filename in [
        f"{name}_model.cif",
        f"{name}_summary_confidences.json",
        f"{name}_confidences.json",
        f"{name}_ranking_scores.csv",
    ]:
        assert (base / filename).exists(), f"missing output file: {base / filename}"


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def require_ckpt():
    """Skip the whole integration session when the checkpoint is absent."""
    if not CKPT_PATH.exists():
        pytest.skip(
            f"RF3 checkpoint not found at {CKPT_PATH}. "
            "Set RF3_CKPT_PATH or see the module docstring for download instructions."
        )


@pytest.fixture(scope="session")
def basic_folds_dir(require_ckpt, tmp_path_factory):
    """Single ``rf3 fold`` call covering all three basic input modes.

    Batching the three inputs amortises the model-loading overhead::

        1cyo_from_json.json   — protein-only JSON
        1cyo_with_ligand.json — protein + HEM via CCD code
        1cyo.cif              — CIF file containing protein + HEM
    """
    out_dir = tmp_path_factory.mktemp("rf3_basic")
    out_dir, _ = run_rf3_fold(
        inputs=[
            DATA_DIR / "1cyo_from_json.json",
            DATA_DIR / "1cyo_with_ligand.json",
            DATA_DIR / "1cyo.cif",
        ],
        out_dir=out_dir,
    )
    return out_dir


@pytest.fixture(scope="session")
def annotate_b_factor_dir(require_ckpt, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("rf3_annotate_b")
    out_dir, _ = run_rf3_fold(
        DATA_DIR / "1cyo_from_json.json",
        out_dir,
        extra_flags=["annotate_b_factor_with_plddt=true"],
    )
    return out_dir


@pytest.fixture(scope="session")
def early_stopping_dir(require_ckpt, tmp_path_factory):
    """Fold with threshold=1.0, which pLDDT can never reach → always exits early."""
    out_dir = tmp_path_factory.mktemp("rf3_early_stop")
    out_dir, stderr = run_rf3_fold(
        DATA_DIR / "1cyo_from_json.json",
        out_dir,
        extra_flags=["early_stopping_plddt_threshold=1.0"],
    )
    return out_dir, stderr


@pytest.fixture(scope="session")
def one_model_per_file_dir(require_ckpt, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("rf3_one_model")
    out_dir, _ = run_rf3_fold(
        DATA_DIR / "1cyo_from_json.json",
        out_dir,
        extra_flags=["one_model_per_file=true"],
    )
    return out_dir


@pytest.fixture(scope="session")
def seed_dirs(require_ckpt, tmp_path_factory):
    """Two identical runs with the same seed for reproducibility checks."""
    dirs = []
    for _ in range(2):
        d = tmp_path_factory.mktemp("rf3_seed")
        d, _ = run_rf3_fold(DATA_DIR / "1cyo_from_json.json", d)
        dirs.append(d)
    return dirs[0], dirs[1]


@pytest.fixture(scope="session")
def template_selection_dir(require_ckpt, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("rf3_template")
    out_dir, _ = run_rf3_fold(
        DATA_DIR / "1cyo.cif",
        out_dir,
        extra_flags=["template_selection=[A]"],
    )
    return out_dir


@pytest.fixture(scope="session")
def ground_truth_conformer_dir(require_ckpt, tmp_path_factory):
    """1cyo chain B is HEM — use it as the ground-truth conformer."""
    out_dir = tmp_path_factory.mktemp("rf3_gt_conformer")
    out_dir, _ = run_rf3_fold(
        DATA_DIR / "1cyo.cif",
        out_dir,
        extra_flags=["ground_truth_conformer_selection=[B]"],
    )
    return out_dir


@pytest.fixture(scope="session")
def skip_existing_dirs(require_ckpt, tmp_path_factory):
    """Run fold twice into the same out_dir; second run uses skip_existing=true."""
    out_dir = tmp_path_factory.mktemp("rf3_skip_existing")
    run_rf3_fold(DATA_DIR / "1cyo_from_json.json", out_dir)

    model_cif = out_dir / "1cyo_from_json" / "1cyo_from_json_model.cif"
    mtime_after_first = model_cif.stat().st_mtime if model_cif.exists() else None

    run_rf3_fold(
        DATA_DIR / "1cyo_from_json.json",
        out_dir,
        extra_flags=["skip_existing=true"],
    )
    mtime_after_second = model_cif.stat().st_mtime if model_cif.exists() else None

    return out_dir, mtime_after_first, mtime_after_second
