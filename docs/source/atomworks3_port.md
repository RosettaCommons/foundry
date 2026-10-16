# Upgrading to AtomWorks 3

RFD3, RFD3NA, MPNN and RF3 require AtomWorks `>=3.0.1,<4` and PyTorch `>=2.11`.
Upgrade Foundry and its declared dependencies together; rebuild containers rather
than overlaying new source on an AtomWorks 2 environment. Run `pip check` after
installation. Model checkpoints do not need conversion.

## Input and preprocessing compatibility

Existing design specifications remain supported. Custom pipelines should use
AtomWorks 3's `ParseConfig` / `PrepareConfig` APIs and updated annotation/query
imports. Foundry's dataset configurations use its `StructuralDatasetWrapper`
adapter for AtomWorks' native `PandasDataset` and structure loader.

AtomWorks 3 corrects reference-conformer rotations and charges on covalently
modified atoms. It also retains a ligand reference conformer after some UFF
optimization failures where AtomWorks 2 masked it. These changes can alter
predictions even with the same checkpoint and seed. Preserve an old environment
if you need to reproduce AtomWorks 2 preprocessing.

RF3 preserves its training-only bond-distance cutoff of strictly less than
2.4 Å. Inference retains explicit finite-length bonds between components that
have not yet been folded together. Template and conformer constraints survive
CCD atom completion and residue renumbering; newly added atoms are unconditioned.

## Inference behavior

RFD3 defaults to `inference_mode=fast`, which enables compiled inference and
corrected pooling. Compilation adds startup cost for new shapes. Use
`inference_mode=legacy` for the serial eager execution policy and original mean;
it retains AtomWorks 3 preprocessing and cannot guarantee historical coordinates.
The old CUDA mean is nondeterministic, and fast/legacy consume different random
streams. See {doc}`RFD3 input options </models/rfd3/input>` and
{doc}`inference acceleration </models/rfd3/inference_acceleration>`.

RF3 automatically selects the optimized LayerNorm on validated H200 / Torch
2.11.0 / CUDA 12.8 configurations and sufficiently large tensors. Other inference
calls use native PyTorch, with one warning per fallback reason. Training and
autograd always use native LayerNorm. Whole-model bitwise comparisons require
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialization and
`torch.use_deterministic_algorithms(True)` in both runs; inference does not enable
these process-wide controls automatically.

For device debugging, `FOUNDRY_DEVICE` accepts `auto` (default), `cpu`, `cuda`,
`mps` or `xpu`. The shared inference engines use one device/node and FP32 when
CPU or MPS is explicitly requested. MPNN also reads this variable unless its
Python `device` argument is supplied.

## Quality and reproducibility limits

Corrected pooling improves numerical accumulation, but has not demonstrated a
model-quality improvement. The tested RFD3 enzyme task showed no clear quality
separation between fast and legacy; see the
{doc}`measurements and limits </models/rfd3/inference_acceleration>`.
RF3 fixed-feature replay matched across AtomWorks versions in the paired audit,
while native preprocessing changed one peptide–DNA conformation substantially.
These bounded comparisons do not establish broad folding or docking equivalence.
Old and corrected BF16 training trajectories differ; full training convergence
and downstream training quality have not been compared.
