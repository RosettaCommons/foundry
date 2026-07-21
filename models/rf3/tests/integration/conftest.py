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
# Repo root (…/foundry): the CWD that repo-relative paths inside input JSONs
# (e.g. an SDF ligand's ``path``) are written against.
REPO_ROOT = Path(__file__).resolve().parents[4]

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


def assert_valid_plddt(summary):
    """Assert ``overall_plddt`` is a sane confidence value in the open (0, 1)."""
    plddt = summary["overall_plddt"]
    assert 0 < plddt < 1, f"overall_plddt outside expected (0, 1) range: {plddt}"


def assert_chain_count(summary, expected, detail=""):
    """Assert the fold produced *expected* chains (one ``chain_ptm`` entry each).

    *detail* optionally names the expected composition for the failure message.
    """
    actual = len(summary["chain_ptm"])
    suffix = f" ({detail})" if detail else ""
    assert actual == expected, f"expected {expected} chains{suffix}; got {actual}"


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


def assert_valid_plddt(summary):
    """Assert ``overall_plddt`` is a sane confidence value in the open (0, 1)."""
    plddt = summary["overall_plddt"]
    assert 0 < plddt < 1, f"overall_plddt outside expected (0, 1) range: {plddt}"
def residue_names_in_cif(cif_path):
    """Return the set of ``label_comp_id`` (residue name) values in a model CIF.

    Parses the ``_atom_site`` loop header to locate the residue-name column,
    then collects it from each atom row.  More robust than a substring search
    for asserting that a specific residue (a nucleotide, a ligand, ...) is
    present in a predicted structure.
    """
    names = set()
    col_names = []
    comp_col = None
    in_atom_loop = False

    for line in Path(cif_path).read_text().splitlines():
        stripped = line.strip()
        if stripped == "loop_":
            in_atom_loop = False
            col_names = []
            comp_col = None
        elif stripped.startswith("_atom_site."):
            col_names.append(stripped)
            if stripped == "_atom_site.label_comp_id":
                comp_col = len(col_names) - 1
            in_atom_loop = True
        elif (
            in_atom_loop
            and comp_col is not None
            and stripped
            and not stripped.startswith("_")
            and stripped != "#"
        ):
            parts = stripped.split()
            if len(parts) > comp_col:
                names.add(parts[comp_col])
    return names


def materialize_json_with_abs_paths(json_path, dest_dir):
    """Copy a components JSON into *dest_dir*, resolving relative component paths.

    A component may reference an external file (e.g. an SDF ligand) via a
    ``path`` field. rf3 resolves that path relative to the *process* CWD, not to
    the JSON file's location, and ``run_rf3_fold`` sets no ``cwd``. Such paths are
    written repo-root-relative, so the checked-in JSON only loads when pytest is
    launched from the repo root. Rewriting each relative path to an absolute path
    (anchored at ``REPO_ROOT``, the CWD it was written against) makes the fixture
    CWD-independent. The path cannot simply be committed as absolute because it
    is machine-specific.

    Returns the path to the rewritten JSON in *dest_dir*.
    """
    data = json.loads(Path(json_path).read_text())
    for entry in data if isinstance(data, list) else [data]:
        for component in entry.get("components", []):
            raw = component.get("path")
            if raw and not Path(raw).is_absolute():
                component["path"] = str((REPO_ROOT / raw).resolve())
    out_path = dest_dir / Path(json_path).name
    out_path.write_text(json.dumps(data))
    return out_path


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

        glke_from_json.json          — protein-only JSON (GLKE, 4 residues)
        glke_with_ligands.json       — GLKE + MG (ccd_code) + HEM (sdf path)
                                       + imidazole (smiles)
        glke_with_ligands_from_cif.cif — CIF containing GLKE + the same ligands

    NOTE: the batched examples share a single seeded RNG stream, so each
    example's stochastic outputs depend on what was folded *before* it in this
    list. Reordering (or adding/removing) inputs changes those draws — most
    visibly ``has_clash``, which tests here assert on. If you change the batch,
    re-run the suite and update any ``has_clash`` assertions that flip.
    """
    out_dir = tmp_path_factory.mktemp("rf3_basic")
    # glke_with_ligands.json references HEM.sdf by a repo-relative path; rewrite
    # it to an absolute path so the fold does not depend on the pytest CWD.
    ligands_json = materialize_json_with_abs_paths(
        DATA_DIR / "glke_with_ligands.json", out_dir
    )
    out_dir, _ = run_rf3_fold(
        inputs=[
            DATA_DIR / "glke_from_json.json",
            ligands_json,
            DATA_DIR / "glke_with_ligands_from_cif.cif",
        ],
        out_dir=out_dir,
    )
    return out_dir


@pytest.fixture(scope="session")
def annotate_b_factor_dir(require_ckpt, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("rf3_annotate_b")
    out_dir, _ = run_rf3_fold(
        DATA_DIR / "glke_from_json.json",
        out_dir,
        extra_flags=["annotate_b_factor_with_plddt=true"],
    )
    return out_dir


@pytest.fixture(scope="session")
def early_stopping_dir(require_ckpt, tmp_path_factory):
    """Fold with threshold=1.0, which pLDDT can never reach → always exits early."""
    out_dir = tmp_path_factory.mktemp("rf3_early_stop")
    out_dir, stderr = run_rf3_fold(
        DATA_DIR / "glke_from_json.json",
        out_dir,
        extra_flags=["early_stopping_plddt_threshold=1.0"],
    )
    return out_dir, stderr


@pytest.fixture(scope="session")
def seed_dirs(require_ckpt, tmp_path_factory):
    """Two identical runs with the same seed for reproducibility checks."""
    dirs = []
    for _ in range(2):
        d = tmp_path_factory.mktemp("rf3_seed")
        d, _ = run_rf3_fold(DATA_DIR / "glke_from_json.json", d)
        dirs.append(d)
    return dirs[0], dirs[1]


@pytest.fixture(scope="session")
def template_selection_dir(require_ckpt, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("rf3_template")
    out_dir, _ = run_rf3_fold(
        DATA_DIR / "glke.cif",
        out_dir,
        extra_flags=["template_selection=[A]"],
    )
    return out_dir


@pytest.fixture(scope="session")
def ground_truth_conformer_dir(require_ckpt, tmp_path_factory):
    """Chain C of the ligand CIF is HEM — use it as the ground-truth conformer."""
    out_dir = tmp_path_factory.mktemp("rf3_gt_conformer")
    out_dir, _ = run_rf3_fold(
        DATA_DIR / "glke_with_ligands_from_cif.cif",
        out_dir,
        extra_flags=["ground_truth_conformer_selection=[C]"],
    )
    return out_dir


@pytest.fixture(scope="session")
def complex_folds_dir(require_ckpt, tmp_path_factory):
    """Single ``rf3 fold`` call covering multi-entity and multi-example inputs.

    Batching amortises the model-loading overhead across five predictions::

        two_protein_chains.json     — two protein chains (interface metrics)
        protein_dna_complex.json    — protein + single-stranded DNA complex
        peptide_glycan_bond.json    — peptide + NAG with an explicit covalent bond
        two_examples_from_json.json — two examples defined in one JSON file
                                      (→ two_examples_first, two_examples_second)

    NOTE: the batched examples share a single seeded RNG stream, so each
    example's stochastic outputs depend on what was folded *before* it in this
    list. Reordering (or adding/removing) inputs changes those draws — most
    visibly ``has_clash`` (which is why ``test_fold_with_covalent_bond``
    deliberately does not assert on it). If you change the batch, re-run the
    suite and update any ``has_clash`` assertions that flip.
    """
    out_dir = tmp_path_factory.mktemp("rf3_complex")
    out_dir, _ = run_rf3_fold(
        inputs=[
            DATA_DIR / "two_protein_chains.json",
            DATA_DIR / "protein_dna_complex.json",
            DATA_DIR / "peptide_glycan_bond.json",
            DATA_DIR / "two_examples_from_json.json",
        ],
        out_dir=out_dir,
    )
    return out_dir


@pytest.fixture(scope="session")
def directory_input_dir(require_ckpt, tmp_path_factory):
    """Fold a *directory* of inputs, exercising directory-globbing resolution.

    Two minimal single-chain JSON inputs are written into a directory, which is
    then passed as ``inputs=<dir>``; ``rf3 fold`` should discover and fold both.
    """
    input_dir = tmp_path_factory.mktemp("rf3_dir_inputs")
    for name, seq in [("dir_pep_a", "GLKE"), ("dir_pep_b", "AGLK")]:
        (input_dir / f"{name}.json").write_text(
            json.dumps([{"name": name, "components": [{"seq": seq, "chain_id": "A"}]}])
        )

    out_dir = tmp_path_factory.mktemp("rf3_dir_out")
    out_dir, _ = run_rf3_fold(input_dir, out_dir)
    return out_dir


@pytest.fixture(scope="session")
def skip_existing_dirs(require_ckpt, tmp_path_factory):
    """Run fold twice into the same out_dir; second run uses skip_existing=true."""
    out_dir = tmp_path_factory.mktemp("rf3_skip_existing")
    run_rf3_fold(DATA_DIR / "glke_from_json.json", out_dir)

    model_cif = out_dir / "glke_from_json" / "glke_from_json_model.cif"
    mtime_after_first = model_cif.stat().st_mtime if model_cif.exists() else None

    run_rf3_fold(
        DATA_DIR / "glke_from_json.json",
        out_dir,
        extra_flags=["skip_existing=true"],
    )
    mtime_after_second = model_cif.stat().st_mtime if model_cif.exists() else None

    return out_dir, mtime_after_first, mtime_after_second


@pytest.fixture(scope="session")
def msa_fold_dir(require_ckpt, tmp_path_factory):
    """Single ``rf3 fold`` call covering MSA inputs across chain counts and modes.

    Batching amortises the model-loading overhead across four folds that each
    supply pre-computed MSAs. The set spans both chain count (monomer /
    homodimer / heteromer) and both ways rf3 accepts an MSA path (JSON
    per-component ``msa_path`` vs CIF ``_msa_paths_by_chain_id`` header)::

        monomer_msa.json          — monomer;   JSON ``msa_path`` (1 chain)
        monomer_msa_from_cif.cif  — monomer;   CIF ``_msa_paths_by_chain_id``
                                    (same sequence, MSA declared in the CIF)
        homodimer_msa.json        — homodimer; two identical chains sharing one
                                    a3m via JSON ``msa_path``
        heteromer_paired_msa.json — heteromer; two distinct chains, each with its
                                    own a3m (a shared TaxID makes it a genuine
                                    paired MSA)

    All inputs use short synthetic sequences (~12 residues) with hand-built a3m
    files so the whole fixture folds quickly on CPU — these tests exercise MSA
    format/plumbing, not structure quality, so a real complex (much slower on
    CPU) is unnecessary. The MSA paths inside the inputs are written relative to
    the repo root, so (as documented for the integration suite) ``rf3 fold``
    must be launched from the repo root for them to resolve.

    NOTE: the batched examples share a single seeded RNG stream, so each
    example's stochastic outputs depend on what was folded *before* it. The
    assertions in ``test_msa_fold.py`` (chain count, ``iptm > 0``, pLDDT range)
    are order-insensitive; if you add order-sensitive checks, re-run after any
    reordering.
    """
    out_dir = tmp_path_factory.mktemp("rf3_msa")
    out_dir, _ = run_rf3_fold(
        inputs=[
            DATA_DIR / "monomer_msa.json",
            DATA_DIR / "monomer_msa_from_cif.cif",
            DATA_DIR / "homodimer_msa.json",
            DATA_DIR / "heteromer_paired_msa.json",
        ],
        out_dir=out_dir,
    )
    return out_dir


@pytest.fixture(scope="session")
def msa_present_flag_dir(require_ckpt, tmp_path_factory):
    """Fold an input that *has* MSAs with ``raise_if_missing_msa_...`` enabled.

    ``monomer_msa.json`` is a 12-residue protein (above the length-10 threshold)
    with an MSA supplied, so the missing-MSA guard must not trip and the fold
    should complete normally. This is the success counterpart to
    ``test_raise_if_missing_msa_errors_when_absent``.
    """
    out_dir = tmp_path_factory.mktemp("rf3_msa_flag")
    out_dir, _ = run_rf3_fold(
        DATA_DIR / "monomer_msa.json",
        out_dir,
        extra_flags=["raise_if_missing_msa_for_protein_of_length_n=10"],
    )
    return out_dir
