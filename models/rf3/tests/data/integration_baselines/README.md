# RF3 Integration Test — GPU Baselines

This directory holds GPU-generated outputs used by `test_cpu_gpu_parity.py`
to verify that CPU inference produces metrics within tolerance of GPU inference.

## What lives here

Each subdirectory corresponds to one test input and contains the
`summary_confidences.json` (and optionally `model.cif`) from a GPU run.

```
integration_baselines/
  two_protein_chains/
    two_protein_chains_summary_confidences.json
```

## Known limitations

See the module docstring in `models/rf3/tests/integration/test_cpu_gpu_parity.py`
for a full list of known limitations, including:

- Baselines go stale silently if the inference engine output changes.
- The parity input (`two_protein_chains`) has a real interface, so iptm and
  ranking_score are genuine computed values — but at speed-flag quality they are
  small, so the absolute ±0.02 tolerance is loose relative to those two metrics.
- The protein-only and ligand inputs have no committed baseline yet.

## Generating a baseline

Run on a machine with a GPU using the same speed flags as the integration
tests (so the comparison is apples-to-apples):

```bash
cd /path/to/foundry

rf3 fold \
    inputs='models/rf3/tests/data/two_protein_chains.json' \
    ckpt_path='<path_to_rf3_foundry_01_24_latest_remapped.ckpt>' \
    n_recycles=1 num_steps=20 diffusion_batch_size=1 seed=1 \
    early_stopping_plddt_threshold=0.0 \
    out_dir='models/rf3/tests/data/integration_baselines'
```

The flags must match the `SPEED_FLAGS` used by the `complex_folds_dir` fixture in
`conftest.py` exactly — in particular `early_stopping_plddt_threshold=0.0`, which
disables the default 0.5 threshold. Otherwise the CPU run and the GPU baseline are
not compared apples-to-apples (a low-pLDDT input could early-stop under the
default and shift the metrics). `two_protein_chains` is folded first in that
fixture's batch, so its draws from the shared seeded RNG stream match this
standalone fold.

`rf3 fold` automatically creates a `two_protein_chains/` subdirectory inside `out_dir`,
so the output lands at
`integration_baselines/two_protein_chains/two_protein_chains_summary_confidences.json`
— exactly where the parity test looks for it.

Commit at minimum the `summary_confidences.json` from the output.
Once committed, `test_cpu_gpu_parity.py::test_confidence_metrics_match_gpu_baseline`
will run automatically in the integration CI job.
