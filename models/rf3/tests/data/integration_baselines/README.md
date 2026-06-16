# RF3 Integration Test — GPU Baselines

This directory holds GPU-generated outputs used by `test_cpu_gpu_parity.py`
to verify that CPU inference produces metrics within tolerance of GPU inference.

## What lives here

Each subdirectory corresponds to one test input and contains the
`summary_confidences.json` (and optionally `model.cif`) from a GPU run.

```
integration_baselines/
  1cyo_from_json/
    1cyo_from_json_summary_confidences.json
```

## Generating a baseline

Run on a machine with a GPU using the same speed flags as the integration
tests (so the comparison is apples-to-apples):

```bash
cd /path/to/foundry

rf3 fold \
    inputs='models/rf3/tests/data/1cyo_from_json.json' \
    ckpt_path='<path_to_rf3_foundry_01_24_latest_remapped.ckpt>' \
    n_recycles=1 num_steps=20 diffusion_batch_size=1 seed=1 \
    out_dir='models/rf3/tests/data/integration_baselines'
```

`rf3 fold` automatically creates a `1cyo_from_json/` subdirectory inside `out_dir`, so
the output lands at `integration_baselines/1cyo_from_json/1cyo_from_json_summary_confidences.json`
— exactly where the parity test looks for it.

Commit at minimum the `summary_confidences.json` from the output.
Once committed, `test_cpu_gpu_parity.py::test_confidence_metrics_match_gpu_baseline`
will run automatically in the integration CI job.
