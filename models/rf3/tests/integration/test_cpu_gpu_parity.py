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

Generating the GPU baseline
---------------------------
Run on a machine with a GPU, then commit the output::

    rf3 fold \\
        inputs='models/rf3/tests/data/1cyo_from_json.json' \\
        ckpt_path='<path_to_checkpoint>' \\
        n_recycles=1 num_steps=20 diffusion_batch_size=1 seed=1 \\
        out_dir='models/rf3/tests/data/integration_baselines/1cyo_from_json'

Commit the ``summary_confidences.json`` (and optionally the model CIF) from
that directory.  Once committed, this test will run automatically.
"""

import json
from pathlib import Path

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
        "GPU baseline not yet committed for 1cyo_from_json. "
        "See module docstring for generation instructions."
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
        if cpu_val is None or gpu_val is None:
            continue
        diff = abs(cpu_val - gpu_val)
        if diff > _TOLERANCE:
            mismatches.append(
                f"  {key}: CPU={cpu_val:.4f}, GPU={gpu_val:.4f}, diff={diff:.4f}"
            )

    assert not mismatches, (
        f"CPU/GPU metric divergence exceeds ±{_TOLERANCE}:\n" + "\n".join(mismatches)
    )
