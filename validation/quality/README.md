# Paired RFD3 generation and refolding pilot

Read `report.md` for the observed results and limitations. This directory keeps
the registered plan, exact executed scripts, an independent artifact audit,
analysis tests, and compact evidence. Weights, serialized model inputs, and raw
structure outputs are excluded from Git.

## Protocol

`plan.json` and `provenance.json` record the protocol and script hashes. The four
arms each contain the same 16 cases: lengths 64/96/128/160 and seeds 11/22/33/44.
All use 200 RFD3 schedule points, bfloat16 autocast, PyTorch attention kernels,
and no sampler realignment. The native arm serializes its AW2 example and 200
Gaussian tensors for each case. Fixed and padded AW2 consume that exact payload;
AW3 builds its own example and records any model-feature difference before using
the same Gaussian tensors. Padding capacity is 2304 atoms, 192 tokens, 14 slots,
batch size one. This pilot does not measure batching throughput.

Every backbone gets two ProteinMPNN sequences at temperature 0.1. The same AW3
source, pooling helper, MPNN checkpoint, and RF3 checkpoint are used for all arms.
RF3 gets sequence-only inputs without supplied MSAs or templates and uses 50
diffusion steps, 10 recycles, one sample, no pLDDT early stopping, and seed
initialization before constructing each inference input. MPNN seed is
`5000 + generation_seed`; RF3 seed is `6000 + 10 * generation_seed + draw`.

A backbone passes if at least one of its two refolds jointly has aligned CA
RMSD < 2 A and overall pLDDT >= 0.8. The primary endpoint never combines RMSD
from one sibling with confidence from another. Secondary best RMSD and best
pLDDT are separate sibling extrema, and need not describe the same structure.

`score.py` reports paired percentile bootstrap intervals (90%, 10,000 draws,
bootstrap seed 914), plus a sensitivity analysis clustering the four repeated
generation seeds across lengths. The exploratory margin gate uses the worse
bound of the two intervals. Four seed clusters cannot establish general
equivalence; the pilot covers unconditional proteins only.

## Reproduce

Use an H200 GPU and separate Python 3.12 environments for AtomWorks 2.2.1 and
3.0.1. See `evidence/runtime_provenance.json` for actual package versions, source
revisions, and checkpoint hashes. Create clean source checkouts at the commits
in `provenance.json`. The shared pooling helper is
`src/foundry/utils/torch.py` at commit `aa97ded` in Foundry PR #400.

`run.sh` is the exact executed CoreWeave Slurm launcher, retained unchanged to
preserve its registered hash. `reproduce.sh` is a portable wrapper for the same
driver; run it inside an existing GPU allocation with these absolute paths set:

```bash
export AW2_PY=/path/to/aw2-env/bin/python
export AW3_PY=/path/to/aw3-env/bin/python
export NATIVE_SOURCE=/path/to/native-foundry
export PADDED_SOURCE=/path/to/pr397-foundry
export AW3_SOURCE=/path/to/aw3-foundry
export POOL_SOURCE=/path/to/pool-fix-foundry/src/foundry/utils/torch.py
export RFD3_CKPT=/path/to/rfd3_latest.ckpt
export MPNN_CKPT=/path/to/proteinmpnn_v_48_020.pt
export RF3_CKPT=/path/to/rf3_foundry_01_24_latest_remapped.ckpt
export RESULTS_DIR=/path/to/new-results
bash validation/quality/reproduce.sh
```

Install the model dependencies corresponding to each checkout into its
environment. The wrapper sets `PYTHONPATH` explicitly so a neighboring editable
checkout cannot silently replace the selected source. Check checkpoint hashes
before running. `RESULTS_DIR` must be a fresh directory for a new experiment;
reuse only to resume the same protocol. Generation resumes from completed JSON
records; refolding resumes from existing model CIFs. If a run stops while writing
a model, remove only that incomplete case's outputs before resuming and retain
the failure record. Do not treat partial runs as quality evidence.

To analyze an intact retained result directory without new inference:

```bash
"$AW3_PY" validation/quality/audit.py "$RESULTS_DIR"
"$AW3_PY" validation/quality/score.py "$RESULTS_DIR"
"$AW3_PY" -m pytest validation/quality/test_score.py -q
```

The audit checks case coverage, immutable script hashes, model-feature equality,
finite coordinates, refold sequence identity, confidence bounds, and artifact
hashes. It runs before quality scoring. The test file covers rigid alignment,
reflection rejection, missing coordinates, and paired/clustered intervals.

## Scope

Backbones are written as all-alanine N/CA/C/O PDBs for sequence design. Reference
CA coordinates are retained in full precision for scoring; PDB roundoff is
shared by all arms. The generated-conformation RMSD is descriptive, not a
quality gate: a scientifically better arithmetic implementation need not
reproduce a legacy stochastic trajectory exactly. An improvement claim still
requires support from the refolding endpoints.

This study does not establish training convergence, nucleic-acid quality,
motif fidelity, experimental folding, binding, or function. It uses one GPU
architecture, one capacity bucket, no target conditioning, and four seed
streams. Identical unconditional AW2/AW3 features do not exercise the port's
chemistry, charge, or motif preprocessing changes. The custom matched noise,
diffusion batch size one, and forced PyTorch kernels do not test public-default
random-number sequencing, multi-example batching, or the custom Triton kernel
backend. The padded arms still use `torch.compile`/Inductor, whose generated GPU
kernels may themselves use Triton.
Timings include compilation and initialization effects and must not
be used as production speed benchmarks.

The AW2 environment has cuEquivariance installed; AW3 does not. A source audit
found no active RFD3 path using it: `LocalAttentionPairBias.forward` explicitly
sets `use_kernel = False` in both source revisions; RFD3's `PairformerBlock`
has no triangle multiplication or triangle attention modules; its padded layer
implementation has no cuEquivariance calls. RF3 can use optional triangle kernels
in other configurations, but all refolds here share the same AW3 environment.
