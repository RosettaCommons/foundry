"""Integration tests for individual ``rf3 fold`` CLI options.

Each test verifies that a specific flag produces the expected behaviour.
All runs use the speed flags defined in conftest (n_recycles=1, num_steps=20,
diffusion_batch_size=1, seed=1) to stay within the 15-minute CI budget.

Session-scoped fixtures run each flag scenario exactly once; test functions
only inspect the resulting files and metrics.
"""

import pytest

from conftest import assert_standard_outputs, load_summary


@pytest.mark.integration
def test_skip_existing_does_not_overwrite(skip_existing_dirs):
    """Second fold with skip_existing=true leaves existing outputs untouched."""
    _, mtime_after_first, mtime_after_second = skip_existing_dirs
    assert mtime_after_first is not None, "first fold produced no model.cif"
    assert mtime_after_second is not None
    assert mtime_after_first == mtime_after_second, (
        "skip_existing=true should not overwrite the existing model.cif "
        f"(mtime changed from {mtime_after_first} to {mtime_after_second})"
    )


@pytest.mark.integration
def test_early_stopping_suppresses_model_output(early_stopping_dir):
    """early_stopping_plddt_threshold=1.0 always triggers early exit.

    pLDDT can never reach 1.0, so the model exits after the first recycle
    without writing a structure file.  The ranking CSV is still produced and
    records the early-stop event.
    """
    result_dir = early_stopping_dir / "1cyo_from_json"
    assert result_dir.is_dir(), "output directory should still be created on early stop"
    assert not (result_dir / "1cyo_from_json_model.cif").exists(), (
        "early stopping should suppress model output"
    )
    scores_text = (result_dir / "1cyo_from_json_ranking_scores.csv").read_text()
    assert "early_stopped" in scores_text.lower(), (
        "ranking_scores.csv should record the early_stopped field"
    )


@pytest.mark.integration
def test_annotate_b_factor_with_plddt(annotate_b_factor_dir):
    """annotate_b_factor_with_plddt=true forces one_model_per_file=true.

    pLDDT values annotated on B-factors should be in (0, 1) rather than the
    large values (> 1) typical of crystallographic B-factors.
    """
    result_dir = annotate_b_factor_dir / "1cyo_from_json"
    assert result_dir.is_dir()

    # With one_model_per_file forced on, individual sample CIFs are written.
    cif_files = list(result_dir.rglob("*.cif"))
    assert len(cif_files) > 0, "expected at least one CIF output"

    # Spot-check B-factor values in the first CIF found: all should be in
    # (0, 1) since they store raw pLDDT, not Å² B-factors.
    sample_cif = cif_files[0]
    content = sample_cif.read_text()
    b_factor_values = _parse_b_factors_from_cif(content)
    assert len(b_factor_values) > 0, "no B-factor values found in CIF"
    assert all(0.0 <= v <= 1.0 for v in b_factor_values), (
        f"B-factors should be pLDDT values in [0, 1]; got range "
        f"[{min(b_factor_values):.3f}, {max(b_factor_values):.3f}]"
    )


@pytest.mark.integration
def test_one_model_per_file(one_model_per_file_dir):
    """one_model_per_file=true writes individual CIF files per sample."""
    result_dir = one_model_per_file_dir / "1cyo_from_json"
    assert result_dir.is_dir()
    cif_files = list(result_dir.rglob("*.cif"))
    assert len(cif_files) > 0, "expected at least one per-sample CIF"


@pytest.mark.integration
def test_seed_reproducibility(seed_dirs):
    """Two runs with identical flags (including seed=1) produce identical scores."""
    dir_a, dir_b = seed_dirs
    summary_a = load_summary(dir_a, "1cyo_from_json")
    summary_b = load_summary(dir_b, "1cyo_from_json")

    for key in ("ranking_score", "overall_plddt", "ptm"):
        val_a = summary_a.get(key)
        val_b = summary_b.get(key)
        assert val_a == val_b, (
            f"seed=1 produced different {key}: run_a={val_a}, run_b={val_b}"
        )


@pytest.mark.integration
def test_template_selection(template_selection_dir):
    """template_selection=[A] completes without error and produces valid output."""
    assert_standard_outputs(template_selection_dir, "1cyo")
    summary = load_summary(template_selection_dir, "1cyo")
    assert 0 < summary["overall_plddt"] < 1


@pytest.mark.integration
def test_ground_truth_conformer_selection(ground_truth_conformer_dir):
    """ground_truth_conformer_selection=[B] keeps HEM in the predicted structure."""
    assert_standard_outputs(ground_truth_conformer_dir, "1cyo")
    model_cif = ground_truth_conformer_dir / "1cyo" / "1cyo_model.cif"
    assert "HEM" in model_cif.read_text(), (
        "HEM should remain in the output when used as a ground-truth conformer"
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _parse_b_factors_from_cif(content):
    """Extract numeric B-factor values from mmCIF ATOM/HETATM records.

    Reads the column index of ``_atom_site.B_iso_or_equiv`` from the loop
    header, then extracts that column from each data row.
    """
    lines = content.splitlines()
    in_atom_loop = False
    col_names = []
    b_col = None
    values = []

    for line in lines:
        stripped = line.strip()

        if stripped == "loop_":
            in_atom_loop = False
            col_names = []
            b_col = None
            continue

        if stripped.startswith("_atom_site."):
            col_names.append(stripped)
            if stripped == "_atom_site.B_iso_or_equiv":
                b_col = len(col_names) - 1
            in_atom_loop = True
            continue

        if in_atom_loop and b_col is not None and stripped and not stripped.startswith("_") and stripped != "#":
            parts = stripped.split()
            if len(parts) > b_col:
                try:
                    values.append(float(parts[b_col]))
                except ValueError:
                    pass

    return values
