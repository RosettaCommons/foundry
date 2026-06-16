"""CPU vs GPU parity tests for RF3 inference.

These tests compare scalar confidence metrics from a CPU run against a
committed GPU baseline to confirm that running on CPU does not degrade
prediction quality — only speed.

Metrics compared (all scalars from ``summary_confidences.json``):

    overall_plddt   per-atom confidence averaged over all atoms
    ptm             predicted TM-score
    iptm            interface predicted TM-score
    ranking_score   weighted combination used for ranking

Tolerance: ±0.05 per metric.  Raw coordinates and PAE matrices are NOT
compared — floating-point non-determinism between CPU and GPU makes
exact agreement impossible.

Known limitations
-----------------
1. **Stale baselines.** The GPU baseline is a committed JSON file generated
   once.  If the inference code changes (even as a bug fix), the test may
   pass against an outdated baseline or begin failing spuriously.  Regenerate
   and commit a fresh baseline whenever the inference engine output changes.
   See ``integration_baselines/README.md`` for the regeneration command.

2. **``iptm=0.0`` bug for single-chain inputs.** ``ComputeIPTM`` returns
   ``0.0`` instead of ``None`` when there are no interfaces, causing
   ``compute_ranking_score`` to weight iptm in when it should not.  Both the
   CPU run and the committed GPU baseline contain this wrong value, so the
   parity check passes — but it is validating a shared bug, not correct
   behaviour.  Once the bug is fixed the baseline must be regenerated.

3. **Narrow input coverage.** Only the protein-only input (``1cyo_from_json``)
   has a committed GPU baseline.  Ligand inputs (``1cyo_with_ligand``,
   ``1cyo.cif``) exercise different code paths but are only range-checked by
   other tests.  Add baselines for those inputs to extend parity coverage.

4. **Low-quality speed-flag outputs.** The baseline was generated with
   ``n_recycles=1 num_steps=20`` — the same flags used to keep CI fast.
   These produce valid but low-quality predictions, so the ±0.05 tolerance
   is relative to an already-noisy reference point.

Generating the GPU baseline
---------------------------
Run on a machine with a GPU, then commit the output::

    rf3 fold \\
        inputs='models/rf3/tests/data/1cyo_from_json.json' \\
        ckpt_path='<path_to_checkpoint>' \\
        n_recycles=1 num_steps=20 diffusion_batch_size=1 seed=1 \\
        out_dir='models/rf3/tests/data/integration_baselines'

Commit the ``summary_confidences.json`` (and optionally the model CIF) from
that directory.  Once committed, this test will run automatically.
"""

import json

import pytest
from conftest import GPU_BASELINE_DIR, load_summary

_BASELINE_DIR = GPU_BASELINE_DIR / "1cyo_from_json"
_BASELINE_SUMMARY = _BASELINE_DIR / "1cyo_from_json_summary_confidences.json"

_TOLERANCE = 0.05
_METRICS = ("overall_plddt", "ptm", "iptm", "ranking_score")


@pytest.mark.integration
@pytest.mark.skipif(
    not _BASELINE_SUMMARY.exists(),
    reason=(
        "GPU baseline missing at integration_baselines/1cyo_from_json/. "
        "See module docstring to regenerate."
    ),
)
def test_confidence_metrics_match_gpu_baseline(basic_folds_dir):
    """CPU scalar metrics agree with the GPU baseline within ±0.05."""
    cpu_summary = load_summary(basic_folds_dir, "1cyo_from_json")
    gpu_summary = json.loads(_BASELINE_SUMMARY.read_text())

    mismatches = []
    for key in _METRICS:
        cpu_val = cpu_summary.get(key)
        gpu_val = gpu_summary.get(key)
        assert cpu_val is not None, f"CPU summary missing expected metric: {key!r}"
        assert gpu_val is not None, f"GPU baseline missing expected metric: {key!r}"
        diff = abs(cpu_val - gpu_val)
        if diff > _TOLERANCE:
            mismatches.append(
                f"  {key}: CPU={cpu_val:.4f}, GPU={gpu_val:.4f}, diff={diff:.4f}"
            )

    assert not mismatches, (
        f"CPU/GPU metric divergence exceeds ±{_TOLERANCE}:\n" + "\n".join(mismatches)
    )
