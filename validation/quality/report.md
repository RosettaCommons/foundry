# RFD3 pooling, padded execution, and AtomWorks 3 quality pilot

## Decision

**The deterministic/FP32 pooling fix has a numerical-correctness rationale, but
this pilot does not show better design quality and does not clear its registered
exploratory noninferiority margins.** Keep Foundry #400 separate and in draft
pending broader quality evidence with the production sampler; do not infer that
the arithmetic fix is ready for an automatic default rollout. Numerical
correctness does not require reproducing legacy coordinates, but downstream
quality remains a separate gate.

The padded-execution and AtomWorks 3 contrasts clear all three exploratory
margins in this bounded configuration. These results support continuing the
port review. They do not bypass the pooling quality gate or establish general
equivalence, quality improvement, or production readiness for untested modes.

## Completed experiment

CoreWeave H200 job **664474 completed successfully (exit 0:0) in 24m59s** on
2026-10-13. All **64 planned backbones, 128 MPNN sequences, and 128 RF3 refolds**
completed. There were **zero generation/refolding execution failures, zero
recovery jobs, and zero omitted cases**. The preflight job 664458 completed in
1m11s; its two refolds were excluded, and its native L64/seed11 backbone was
reused under the same generation protocol. Both allocations have ended.

The four arms share 16 length/seed cases: lengths 64/96/128/160 and seeds
11/22/33/44. RFD3 uses 200 schedule points, bfloat16 autocast, matched serialized
AW2 inputs/noise, no realignment, diffusion batch size one, and the PyTorch
kernel backend. The padded arms compile the denoiser at capacity 2304 atoms /
192 tokens / 14 slots. Every backbone receives two ProteinMPNN sequences at
temperature 0.1, followed by sequence-only RF3 refolding (50 steps, 10 recycles,
one sample) in the same AW3 environment with the same pooling overlay. See
[README.md](README.md) for seed formulas and replay instructions.

The primary endpoint is a backbone with at least one refold jointly satisfying
CA RMSD **< 2 A** and overall pLDDT **>= 0.8**. Two sequence siblings are one
backbone case, not independent replicates. Secondary best RMSD and best pLDDT
are separate sibling extrema and need not come from the same refold.

| Arm | Backbone passes | Sequence passes | Mean best RMSD (A) | Mean best pLDDT |
|---|---:|---:|---:|---:|
| Native AW2 | 15/16 (93.75%) | 30/32 | 0.757930 | 0.862063 |
| Pooling fix, AW2 | 14/16 (87.50%) | 28/32 | 1.068046 | 0.856894 |
| Padded + pooling fix, AW2 | 16/16 (100%) | 30/32 | 0.770233 | 0.854794 |
| Padded + pooling fix, AW3 | 16/16 (100%) | 30/32 | 0.770885 | 0.854794 |

## Paired contrasts

All differences are candidate minus baseline. Intervals are paired percentile
bootstrap 90% intervals, 10,000 resamples, seed 914. The second interval clusters
the four recurring seed streams across lengths. The exploratory gate uses the
worse interval bound against the registered margins: success loss no more than
10 percentage points (pp), pLDDT loss no more than 0.03, RMSD increase no more
than 0.25 A. Both intervals are fragile with only four seed clusters; passing a
gate here is not a population-wide noninferiority result.

### Pooling fix minus native AW2

| Endpoint | Mean difference | Case 90% interval | Seed-cluster 90% interval | Exploratory gate |
|---|---:|---:|---:|---|
| Backbone success (pp) | -6.25 | [-18.75, 0.00] | [-12.50, 0.00] | **Fail** |
| Best pLDDT | -0.005169 | [-0.019494, 0.009282] | [-0.018225, 0.010225] | Pass |
| Best RMSD (A) | +0.310116 | [-0.103172, 0.965524] | [-0.102911, 0.723143] | **Fail** |

The fix lost one passing case (L64/seed44: best RMSD 0.580 to 5.918 A) and gained
none. The preexisting L160/seed33 failure improved from 2.975 to 2.344 A but still
failed. All 16 cases remain included. This is insufficient
to establish a general quality regression, but it also provides no support for
a quality-improvement claim and cannot rule out the prespecified harmful
effects. Zero of 32 designed sequences match the corresponding native sibling;
trajectory/sequence divergence itself is not the quality verdict.

### Padded AW2 minus pooling-fix AW2

| Endpoint | Mean difference | Case 90% interval | Seed-cluster 90% interval | Exploratory gate |
|---|---:|---:|---:|---|
| Backbone success (pp) | +12.50 | [0.00, 25.00] | [0.00, 25.00] | Pass |
| Best pLDDT | -0.002100 | [-0.014312, 0.009950] | [-0.017950, 0.011300] | Pass |
| Best RMSD (A) | -0.297813 | [-0.937700, 0.171522] | [-0.718454, 0.109183] | Pass |

Padded execution gained L64/seed44 and L160/seed33 and lost none. All exploratory
margins pass; the success interval still includes zero, so this is encouraging
pilot evidence rather than an established improvement. Zero of 32 sibling
sequences match the eager fixed arm.

### Padded AW3 minus padded AW2

| Endpoint | Mean difference | Case 90% interval | Seed-cluster 90% interval | Exploratory gate |
|---|---:|---:|---:|---|
| Backbone success (pp) | 0.00 | [0.00, 0.00] | [0.00, 0.00] | Pass |
| Best pLDDT | 0.000000 | [0.000000, 0.000000] | [0.000000, 0.000000] | Pass |
| Best RMSD (A) | +0.000652 | [-0.001130, 0.003182] | [-0.001174, 0.003130] | Pass |

All 16 generated CA arrays are **bitwise identical**, all PDB ATOM records are
identical, and all 32 designed sequences match. PDB files differ only because
AW3/Biotite writes additional CONECT records. All per-case model-feature
comparison lists are empty. The tiny refolding RMSD differences are downstream
common-oracle/output variability after identical generation; their exact cause
was not isolated. They are not a port generation effect. The largest
absolute per-sequence RMSD difference is 0.019825 A. This study does not establish
bitwise oracle repeatability. The degenerate success/pLDDT intervals reflect
identical observed paired outcomes and do not prove a zero population effect.

## Audit and limits

- All registered script hashes remain unchanged. Runtime source bytes match
  registered revisions despite later ancillary test/documentation commits in
  two source checkouts. Exact package versions, source revisions, pooling-file
  hash, and all checkpoint hashes are in
  [runtime provenance](evidence/runtime_provenance.json).
- The independent audit passed planned-case coverage, sequence/chain identity,
  finite coordinates/confidences, feature comparisons, and output hashes.
  Missing-output inventories are saved before assertions. Five CPU analysis
  tests pass, including proper Kabsch alignment, reflection/NaN rejection,
  clustered intervals, and missing-artifact retention.
- cuEquivariance is installed only in the AW2 environment, but source review
  found its RFD3 attention branch disabled and no active RFD3 triangle-kernel
  path. The common RF3 oracle uses the AW3 environment for every arm. The
  optional-kernel discrepancy therefore does not explain these generator
  contrasts.
- This unconditional, batch-one, matched-noise, forced-PyTorch pilot excludes
  production-default RNG sequencing, multi-example batching, custom Triton
  kernels, motifs, charges, nucleic acids, training convergence, and experimental
  function. Identical unconditional features do not validate the port's
  chemistry changes. Timings include compilation and are not speed benchmarks.

## Evidence

- [Exact arm summaries, paired intervals, and per-backbone outcomes](evidence/quality_summary.json)
- [All 128 sequence-level scores](evidence/quality_sequences.csv)
- [Identity, finite-value, provenance, and artifact hash audit](evidence/quality_audit.json)
- [All planned output records, including missing-output fields](evidence/quality_inventory.json)
- [Completion counts and bitwise port comparison](evidence/completion.json)
- [Slurm completion/accounting](evidence/scheduler_accounting.txt)

Full local artifacts are retained under
`/mnt/home/jiechen7/other/foundry-quality-validation/validation/quality/results`.
Weights and raw structure/trajectory payloads are not committed.
