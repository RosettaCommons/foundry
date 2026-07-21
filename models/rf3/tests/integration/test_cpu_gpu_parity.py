"""CPU vs GPU parity tests for RF3 inference.

These tests compare scalar confidence metrics from a CPU run against a
committed GPU baseline to confirm that running on CPU does not degrade
prediction quality — only speed.

Metrics compared (all scalars from ``summary_confidences.json``):

    overall_plddt   per-atom confidence averaged over all atoms
    ptm             predicted TM-score
    iptm            interface predicted TM-score
    ranking_score   weighted combination used for ranking

Tolerance: ±0.02 per metric.  Raw coordinates and PAE matrices are NOT
compared — floating-point non-determinism between CPU and GPU makes
exact agreement impossible.

The ±0.02 window was chosen empirically.  CPU output for this input is
bit-identical across BLAS thread counts (1/8/32 threads produce byte-identical
summaries), so within-backend reduction order contributes no drift; the only
measured movement is the metric's intrinsic seed-to-seed variance at these
speed flags (~2.6e-3 for ptm, ≤1e-3 for the other three).  ±0.02 sits ~8x above
that noise floor, leaving headroom for the (unmeasured, GPU-only) RNG- and
kernel-level CPU/GPU divergence while still being tight enough to catch a real
regression.  If a real GPU baseline later shows a larger systematic offset,
widen this; if it confirms a small offset, it can be tightened further.

Known limitations
-----------------
1. **Stale baselines.** The GPU baseline is a committed JSON file generated
   once.  If the inference code changes (even as a bug fix), the test may
   pass against an outdated baseline or begin failing spuriously.  Regenerate
   and commit a fresh baseline whenever the inference engine output changes.
   See ``integration_baselines/README.md`` for the regeneration command.

2. **Interface metrics are exercised, but small.** ``two_protein_chains`` is a
   two-chain input, so it has a real interface: iptm and ranking_score are
   genuine computed values here (iptm≈0.002, ranking_score≈0.019 at speed-flag
   quality), unlike single-chain inputs which hit the ``iptm=0.0`` bug
   (``ComputeIPTM`` returns ``0.0`` instead of ``None`` when there are no
   interfaces).  Because those two values are small, the absolute ±0.02
   tolerance is loose *relative* to iptm/ranking_score specifically — a gross
   regression in them would be caught, but a small drift would not.  Higher
   ``num_steps`` would raise the values (and quality) at the cost of CI speed.

3. **Narrow input coverage.** Only ``two_protein_chains`` has a committed GPU
   baseline.  The protein-only (``glke_from_json``) and ligand inputs
   (``glke_with_ligands``, ``glke_with_ligands_from_cif``) exercise different
   code paths but are only range-checked by other tests.  Add baselines for
   those inputs to extend parity coverage.

4. **Low-quality speed-flag outputs.** The baseline was generated with
   ``n_recycles=1 num_steps=20`` — the same flags used to keep CI fast.
   These produce valid but low-quality predictions, so the ±0.02 tolerance
   is relative to an already-noisy reference point.

Generating the GPU baseline
---------------------------
Run on a machine with a GPU, then commit the output::

    rf3 fold \\
        inputs='models/rf3/tests/data/two_protein_chains.json' \\
        ckpt_path='<path_to_checkpoint>' \\
        n_recycles=1 num_steps=20 diffusion_batch_size=1 seed=1 \\
        early_stopping_plddt_threshold=0.0 \\
        out_dir='models/rf3/tests/data/integration_baselines'

The flags must match the ``SPEED_FLAGS`` used by the ``complex_folds_dir``
fixture exactly (including ``early_stopping_plddt_threshold=0.0``, which
disables the default 0.5 threshold) so the CPU run and GPU baseline are
compared apples-to-apples.  ``two_protein_chains`` is folded *first* in that
fixture's batch, so its draws from the shared seeded RNG stream match a
standalone fold of the same input (which is what the command above produces).

Commit the ``summary_confidences.json`` (and optionally the model CIF) from
that directory.  Once committed, this test will run automatically.
"""

import json

import pytest
from conftest import GPU_BASELINE_DIR, load_summary

_BASELINE_DIR = GPU_BASELINE_DIR / "two_protein_chains"
_BASELINE_SUMMARY = _BASELINE_DIR / "two_protein_chains_summary_confidences.json"

_TOLERANCE = 0.02
_METRICS = ("overall_plddt", "ptm", "iptm", "ranking_score")


@pytest.mark.integration
@pytest.mark.skipif(
    not _BASELINE_SUMMARY.exists(),
    reason=(
        "GPU baseline missing at integration_baselines/two_protein_chains/. "
        "See module docstring to regenerate."
    ),
)
def test_confidence_metrics_match_gpu_baseline(complex_folds_dir):
    """CPU scalar metrics agree with the GPU baseline within ±0.02."""
    cpu_summary = load_summary(complex_folds_dir, "two_protein_chains")
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
