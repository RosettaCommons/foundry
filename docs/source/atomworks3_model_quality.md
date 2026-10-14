# RF3 and MPNN paired checkpoint audit

This bounded audit compares PR #397 (`754d1ca`, AtomWorks 2.2.1) with the
AtomWorks 3 port (`d0474f0`, AtomWorks 3.0.1). Both arms carry the same
PR #400 deterministic pooling implementation (`aa97ded`; helper SHA256
`5f1a285700caa0571db64980ffcedd1bf013137365b889bbe0519d4d2ff79248`).
It evaluates dependency migration with the pooling correction present; it does
not establish equivalence of PR #397 to the implementation preceding #397.

## Controlled protocol

Both environments use Torch 2.11.0+cu128, RDKit 2026.09.1, the same H200 GPU,
and identical checkpoints. Biotite is 1.4.0 in the old environment and 1.6.0 in
the new environment. RF3 uses bfloat16 AMP, vanilla triangle/dense attention,
10 recycles, 50 diffusion steps, one sample, and no early stopping. TF32 is off.
`DISABLE_CUEQUIVARIANCE=1` is set before imports: triangle multiplication has
its own optional backend independently of the triangle attention setting.
Both actual backend flags and pooling hashes are recorded in provenance.

Seeds 11, 29, and 47 reset Python, NumPy, Torch CPU, and CUDA RNG before each
input is constructed and run. Seven fixtures cover a designed 32-residue
protein, GNGG + four DNA nucleotides, GNGG + covalent NAG, ACE-capped GNGG,
GLKE + ligands, the 32-residue structure as an explicit template, and a
supplementary 32-residue protein + eight DNA nucleotides. The latter is a
separate diagnostic; it does not replace the short-peptide stress fixture.
These are regression fixtures, not experimentally validated complexes.

Fixed-feature replay loads the archived AtomWorks 2 preprocessing result and
restores its Python/NumPy/Torch CPU/CUDA RNG states in AtomWorks 3. Actual
pre-validation tensors are also archived. This isolates the model from the
known reference-rotation and covalent-charge preprocessing changes.

ProteinMPNN and LigandMPNN consume identical archived coordinates in each arm,
with two designs per seed, temperature 0.1, and residue A4 fixed. They cover the
original six fixtures, giving 36 paired runs and 72 sequences per arm. Their
inputs come from the initial archived RF3 baseline; RF3 backend differences
cannot confound MPNN comparisons because both arms read exactly the same files.

Thresholds were set before results: numerical `atol=rtol=1e-5`, identical
MPNN sampled sequences and fixed residues, no nonfinite outputs, identical
RF3 atom identities, median pLDDT decline at most 0.05 and per-case mean decline
at most 0.10, covalent outlier fraction increase at most 0.10, no added severe
nonbonded clashes below 0.8 Å, and template CA RMSD at most 1 Å.
Covalent outliers use a deliberately coarse 0.9–2.0 Å range, excluding metal
coordination; clashes exclude bonds and 1–3 neighbors. Native CA RMSD above
2 Å separately flags loss of seeded structure equivalence.

## Results

This summary records the controlled audit; generated per-case JSON reports are
not shipped as documentation. The runners below can produce a new report.

- MPNN: all 36 comparisons are bitwise identical for feature tensors, logits,
  log probabilities, sampled/argmax sequences, and reported sequence confidence.
  Every fixed A4 is retained in both arms.
- RF3 fixed-feature replay: all 21 comparisons reproduce all 154 network
  tensors bitwise, including diffusion trajectories and confidence outputs;
  maximum absolute difference is zero.
- RF3 native preprocessing: all 21 comparisons pass the preregistered quality
  gate. Median pLDDT change is zero; per-case mean changes span −0.00297 to
  +0.00637. Outputs are finite, atom identities match, and no severe clashes
  are added. Template CA RMSDs are 0.164–0.173 Å. The supplementary protein–DNA
  case changes by 0.040–0.157 Å CA RMSD.
- Native seeded structure equivalence fails for the short-peptide + DNA
  seed-11 fixture: CA RMSD is 3.044 Å. In that comparison AtomWorks 3 removes
  33 severe clashes and reduces the covalent outlier fraction by 0.145; the
  AtomWorks 2 peptide had collapsed. The changed conformation remains a
  reported failure of the prespecified structural-equivalence criterion.

The initial study lacked a backend control: cuEquivariance was available only
in the old environment, despite both arms requesting vanilla attention. It
showed a short-peptide seed-11 clash increase of 21 and failed fixed-feature
replay. Those results remain in `.validation/paired/`; an intermediate
seed-before-construction study is in `.validation/paired-seeded/`. An old-to-old
replay control reproduced all 154 network tensors bitwise in all three protein
seeds. The final controlled experiment is the basis for dependency conclusions.

## Interpretation and limits

The paired MPNN evidence supports exact inference preservation on these inputs.
RF3's native preprocessing is not numerically identical: AtomWorks 3 corrects
reference rotations and covalent charges, and one stress input changes its
sampled conformation substantially. The measured quality criteria do not
regress in the controlled sample. Three seeds and short synthetic fixtures do
not establish broad folding, docking, or training-quality equivalence. The
arbitrary protein–DNA fixture has low interface confidence and cannot establish
binding correctness. Both arms lacked the optional MACE embedding cache and
used the same fallback. Separate training and broader design/refolding studies
are required for those claims.

For CI, retain CPU transform/constraint tests and scorer tests, and run the
checkpoint audit on a GPU lane with explicit matched backends. A missing
checkpoint should make that dedicated lane fail; a skipped GPU job is not
quality evidence. Do not loosen numerical or geometry thresholds to hide a
failed fixture.

## Reproduction

The runner is `scripts/validate_atomworks_model_quality.py`; the comparison is
`scripts/compare_atomworks_model_quality.py`. Keep separate old/new source trees
and environments, and place the chosen arm's `src`, `models/rf3/src`, and
`models/mpnn/src` first on `PYTHONPATH`. Both source trees must use the identical
pooling helper above. Use the same GPU and checkpoints:

| File | SHA256 |
| --- | --- |
| `rf3_foundry_01_24_latest_remapped.ckpt` | `364ef592fd8042a9cf4176d045015190f8322f961ccca38d891b20ca578d3bb0` |
| `proteinmpnn_v_48_020.pt` | `c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd` |
| `ligandmpnn_v_32_010_25.pt` | `161cd264061fda9680cbb940255522ae42f2966c552d045d87913d9452a80970` |

Set `OMP_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=4`, `PYTHONHASHSEED=0`, and
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before launching. For each arm run the runner
with `--arm aw2` or `--arm aw3`, `--model rf3`, `--fixture-root NEW_TREE`,
`--checkpoint-dir CHECKPOINTS`, and `--out RUN/aw2-rf3` or `RUN/aw3-rf3`.
Run a third RF3 invocation in the new environment with
`--replay RUN/aw2-rf3 --out RUN/aw3-rf3-replay`.
Run MPNN once in each environment with `--model mpnn`,
`--structures RUN/aw2-rf3`, and the corresponding `--out RUN/aw2-mpnn` or
`RUN/aw3-mpnn`. Finally run the comparer with `RUN --rf3-pairs 21 --require-pass`. The last flag fails on numerical or
quality gates; seeded structural equivalence remains an explicitly separate
reported criterion.

The scorer's nine CPU tests cover tolerance, missing/discrete changes,
nonfinite and empty outputs, and fixed-residue preservation.

## Exact RF3 inference kernels

The RF3 inference engine can use the standalone CUDA LayerNorm forward kernel
from [uplifting-biomolecular-modeling](https://github.com/anthropics/uplifting-biomolecular-modeling),
revision `f4f62fa6592ae4938d49b1757bea0cfeff9f468e`. It reproduces ATen's
Welford reduction and affine arithmetic while processing multiple rows per warp.
The port retains those operations, uses per-device compilation and thread-local
launch buffers, and omits the upstream provider registry, prebuilt binaries,
disk cache, and unused residual fusion. Source and license notices ship in the
wheel; CUDA Torch's NVRTC compiles the kernel on first use without `nvcc`.

`inference_kernel_backend=exact` is the default. The measured dispatch requires
H200, Torch 2.11.0/CUDA 12.8, widths 64/128/256, at least 131,072 rows, and either
FP32 input or BF16 input under CUDA autocast with FP32 parameters. Training,
gradient-enabled evaluation, unsupported layouts/dtypes, smaller tensors, and
other stacks use PyTorch. `inference_kernel_backend=torch` disables the adapters.
Parameter objects and checkpoint keys are preserved; attention backends and the
AtomWorks 3 port are unchanged by this switch.

### Equivalence protocol

`scripts/benchmark_rf3_exact_kernels.py` compares complete checkpoint forwards,
including all diffusion trajectory and confidence tensors, using identical
prepared features and restored Python/NumPy/CPU/CUDA RNG states. Each invocation
gets a fresh copy of the prepared batch. Both arms use the corrected pooling
implementation. The reference is the current Foundry/AW3 implementation, not
pre-fix PR #397 or AtomWorks 2 preprocessing.

Bitwise model reproducibility requires a controlled runtime:
`CUBLAS_WORKSPACE_CONFIG=:4096:8` must be set before CUDA initialization, and the
benchmark enables `torch.use_deterministic_algorithms(True)` for both arms.
Initial comparisons without these controls exposed allocation-sensitive full
prediction differences even though every replaced LayerNorm output matched its
native operation. A follow-up with only the fixed cuBLAS workspace also produced
different outputs between two native repeats; that setting alone was insufficient.
Thus the kernel's arithmetic equivalence is not an
unconditional guarantee of seeded whole-model reproducibility in an uncontrolled
runtime. The engine does not change process-wide determinism settings.

The benchmark rejects nonfinite outputs, checks every tensor's dtype/shape and
raw bytes (including signed zero), and counts actual kernel executions so a
native fallback cannot masquerade as an acceleration result. It excludes model
loading, input preparation and the first call of each backend from its paired,
alternating warm-run medians. Generated reports remain local benchmark artifacts.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python scripts/benchmark_rf3_exact_kernels.py \
  --checkpoint /path/to/rf3_foundry_01_24_latest_remapped.ckpt \
  --lengths 128 256 400 512 --seeds 11 29 --repeats 3 \
  --out outs/rf3-kernels/benchmark.json
```

### Candidate selection

On the H200, the warmed isolated FP32 LayerNorm case `[256,256,128]` took
106.3 µs with native LayerNorm, 33.0 µs with upstream's CUDA candidate and
48.1 µs with its Triton candidate (median of five 20-call CUDA-event batches).
Both candidates matched all tested output bits. These are isolated operation
timings; whole-model gains are smaller.

The fused transition candidate also passed the controlled checkpoint comparison,
but adding it to CUDA LayerNorm did not improve that run (16.85 s versus
16.44 s), so it is not included. The upstream exact triangle-attention provider
has proven cells for cuEquivariance 0.10.0/0.11.1, while this environment uses
0.12.0; those replacements are not enabled here. Approximate kernels and the
upstream source overlay are not part of this port.

### Whole-model measurements

H200, Torch 2.11.0+cu128, AtomWorks 3.0.1, cuEquivariance 0.12.0,
`rf3_foundry_01_24_latest_remapped.ckpt` (SHA256 above), 10 recycles,
50 steps, no early stopping. Fixed heterogeneous synthetic protein sequences
are numerical/performance fixtures, not a protein-quality benchmark. The table
uses one diffusion sample, vanilla triangle/dense attention and the existing
cuEquivariance triangle-multiplication backend. Times are medians over three
warm repetitions for each of seeds 11 and 29, then averaged across those seeds.

| Residues | Native | Exact LayerNorm | Speedup |
| --- | ---: | ---: | ---: |
| 400 | 16.060 s | 15.555 s | 1.032× |
| 512 | 26.204 s | 25.291 s | 1.036× |

All 154 output tensors, including the full diffusion trajectories, matched
bitwise in all eight length/seed cases (128/256/400/512 residues × seeds 11/29),
for cold and warm calls. The initial 128-residue dispatch was 2–3% slower, and
256 residues provided negligible benefit; a five-sample/auto-attention check at
256 residues was 1.2% slower. These sizes now use native LayerNorm. The final
128-residue fallback check was effectively tied (5.453 s native, 5.466 s with
the adapter, zero custom kernel calls). The 400-residue activation census had
no calls in the removed 65,536–131,071-row range.

With the **final dispatch and default automatic triangle attention**, five
samples, seed 11, and two warm repetitions per arm, 512 residues took
**10.495 s native versus 9.094 s exact (1.154×, 13.4% less time)**. All 154
output tensors matched, and 3,632 custom kernel calls executed per prediction.
At 256 residues the final dispatch made zero custom calls: 5.984 s native versus
5.965 s with the adapter, effectively tied. These timings exclude checkpoint
loading and preprocessing; they are measured cases, not a universal speedup.
The H200 kernel suite passes **110 tests**, including all affine variants,
BF16 autocast, transposed/expanded/column-strided inputs, large dynamic ranges,
nearly constant inputs, signed zeros, alignment and dtype fallback, nondefault
streams, and preservation of native training/gradient behavior.
