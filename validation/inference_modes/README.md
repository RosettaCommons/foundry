# RFD3 fast/legacy preset validation

The inference preset and batched precision changes are in the consolidated
[Foundry PR #401](https://github.com/RosettaCommons/foundry/pull/401).

## Code and precision coverage

- `fast` retains configurable acceleration with corrected pooling by default.
- `legacy` forces the old serial/global-RNG eager execution and original
  `index_reduce` atom pooling on each model copy. The switch is instance-local;
  training continues to use corrected pooling.
- Batched atom means previously rounded the low-precision sum before dividing.
  Batched pair means could also downcast the contractions under Fabric autocast.
  Both keep FP32 intermediates through division for FP16/BF16; pair pooling uses
  integer counts and a local disabled-autocast context.
- Unit regressions compare values and gradients to FP64 references of quantized
  activations, including NaN padding, empty examples, FP16 overflow, and BF16
  rounding. GPU tests exercise CUDA Inductor/fullgraph and autocast.
- Public JSON metadata reports the chosen preset and actual execution path.

Local complete CPU suite: **1,112 passed, 84 skipped**. The skips include GPU,
MPS, optional compiler and unavailable RF3 checkpoint integration cases.
H200 focused precision/preset/compile suite: **21 passed, 4 optional CPU-compiler
and MPS skips**. Mypy passes all 256 source files; pinned Ruff check and format
pass all 460 Python files. CLI Hydra composition accepts both presets.

The full CPU run initially exposed two newly merged upstream documentation tests:
AW3 rejects descending ligand residue IDs in the M0255 PDB. Reordering its ACT
records after NAI leaves every atom record intact. All eight documentation
example tests and the subsequent full suite pass. No test was disabled.

## Public-checkpoint quality protocol

Slurm array **664847**, CoreWeave H200, same released checkpoint and AW3 environment
as the [64-output-per-arm enzyme benchmark](../enzyme_tipatom/expanded64/README.md).
The unchanged 150-residue 1euv tutorial uses 32 predetermined seeds × two outputs,
**64 outputs per preset**, public 200-step sampler, no refolding, and no external
pooling patch. Every output is regenerated through the actual new preset. Read
only native JSON `metrics.insertion_rmsd`; success is <1 Å. All outputs count.

`plan.json` fixes the seed list and input hash before submission. `launch.sh`
records the GPU test gate and both invocations. `analyze.py` requires completion,
checks every output's resolved mode/path/pooling metadata, and writes individual
scores and summaries. Confidence intervals resample whole paired seed groups
(20,000 draws, RNG seed 20261014); same seed does not mean identical random noise
between the two execution paths. This one-fixture pilot has no prespecified
population-equivalence margin.

## Results after the final batched precision correction

All **128/128** outputs completed successfully, and every native JSON reports the
expected preset, pooling policy and actual execution path. All GPU allocations
are released. Both modes use AtomWorks 3.0.1 and the same checkpoint.

| Preset | <1 Å | Mean insertion RMSD (Å) | Median (Å) | Maximum (Å) |
| --- | ---: | ---: | ---: | ---: |
| fast | 62/64 | 0.485 | 0.426 | 2.040 |
| legacy | 63/64 | 0.447 | 0.392 | 1.707 |

Fast minus legacy mean difference: **+0.038 Å**, paired seed-group 95% bootstrap
interval **[−0.051, +0.131] Å**. Pass-rate difference: **−1.5625 percentage
points**, interval **[−6.25, +3.125]**. This fixture shows no clear quality
separation; the intervals are not proof of equivalence or experimental function.
The fast maximum is retained, not filtered out.

The prior AW3 padded benchmark before the final atom/pair pooling correction
had 61/64 below 1 Å and a 0.478 Å mean. The corrected default has 62/64 and
0.485 Å: the precision correction changes some trajectories, without an evident
large quality collapse on this fixture. This comparison does not establish a
quality benefit from the arithmetic correction.

The actual fast run used compiled padded Torch kernels on H200 with compiled
CUDA graphs enabled, as selected by `auto` on this device. Legacy JSON confirms
no compilation/graphs and the original serial random stream/reduction. Both
used two diffusion samples per seed. Fast run timing includes cold compilation
and its preceding GPU tests, so raw Slurm elapsed time is not a speed benchmark.

- [All native scores and JSON paths](evidence/scores.csv)
- [Mode metadata, summary, and intervals](evidence/summary.json)
- [Runtime provenance](evidence/provenance.json)
- [Slurm accounting](evidence/scheduler.txt)
- [Validation checks](evidence/checks.json)

## Compatibility boundary

See [the inference-mode documentation](../../models/rfd3/docs/inference_acceleration.md).
Legacy restores execution choices, not the whole pre-port environment. Both modes
use AW3's parser/bond/conformer behavior, and legacy CUDA reductions can remain
nondeterministic. Previous AW2 results therefore are not a direct expectation for
legacy/AW3. A historical checkout/environment is still needed to reproduce those
input features; even that cannot guarantee old CUDA reduction replay.
