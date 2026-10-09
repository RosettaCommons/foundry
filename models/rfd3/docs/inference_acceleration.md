# RFD3 CUDA inference acceleration

The optional accelerator combines native Triton kernels, cached atom indexing,
and CUDA graph replay of the token transformer, atom encoder and atom decoder.
All measurements and examples here use
`low_memory_mode=false`, the full resident atom-pair representation, two recycles,
and the released checkpoint. Sampling schedules and step counts are unchanged.
Inference now defaults to `inference_kernel_backend=auto` and
`compile_model=true`. Auto selects native operators for compiled inference,
except on SM12x Blackwell GPUs with at least four independent inputs, where it
selects native attention plus the independent Triton transition. Explicit
`inference_kernel_backend=triton-transition` selects that combination on other
hardware; `torch` forces all-native operators. The matched small-length A4000
runs did not show a custom-kernel speedup there.
For standard eager CUDA inference on Ampere or newer, auto selects Triton when
installed, including padded input batches. Explicit `triton` also supports
compiled padded denoisers; other supported devices use PyTorch.
`inference_cuda_graph` remains an explicit eager option.

## Run

Use this checkout's CUDA environment (tested with Python 3.12.13, PyTorch
2.11.0+cu130, Triton 3.6.0, AtomWorks 2.2.1, NVIDIA driver 595.71.05):

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 FOUNDRY_DEVICE=cuda RFD3_LOW_MEMORY_MODE=0 \
  .venv/bin/rfd3 design inputs=null '+specification.length=128' \
  out_dir=outs/accelerated-128 diffusion_batch_size=1 n_batches=1 \
  seed=42 low_memory_mode=false inference_sampler.num_timesteps=200 \
  compile_model=false inference_kernel_backend=triton inference_cuda_graph=true
```

Use a fresh output directory: the engine skips designs it already finds there.
`diffusion_batch_size` is the number of samples of the same specification and can
exceed one. The explicit `inference_cuda_graph=true` option requires
`inference_batch_size=1` and `compile_model=false`; incompatible settings fail
explicitly. The kernels themselves also support padded `inference_batch_size>1`
and `compile_model=true`. CUDA is required.
Triton is supplied by the tested Linux CUDA PyTorch wheel; no Apex, external kit
installation, checkpoint conversion, or package overlay is required.

The options are independent. `inference_kernel_backend=torch
inference_cuda_graph=true` enables graph replay and invariant caching with native
PyTorch arithmetic. `inference_kernel_backend=triton
inference_cuda_graph=false` enables kernels and caching without graph replay.
The existing `dense_attention_backend` remains independently configurable; the
reported accelerated configuration keeps `vanilla` because token SDPA did not
improve the small A4000 workload.

### Padded kernel execution

Both batch axes are supported: `inference_batch_size` batches different inputs,
while `diffusion_batch_size` generates samples of each input. Compatible inputs
share atom/token/slot padding buckets. A tail batch can contain entirely masked
dummy examples, and masked atom rows produce zero outputs.

For example, the two inputs `{"short": {"length": 48}, "long": {"length": 50}}`
share a compilation bucket. With `inference_batch_size=2`,
`diffusion_batch_size=2`, `compile_model=true`, and
`inference_kernel_backend=triton`, the measured denoiser input shape was
`[2, 2, 768, 3]`. The 200-point checkpoint run saved four valid-length, finite
structures under `outs/padded-kernels/compiled-mixed-l48-50-d2/structures/`.
Its warm sampling time was 55.61 ms/step, with 2.22 GiB peak allocated memory.
These are execution checks, not biological quality validation; matched native
comparisons are recorded separately before drawing speedup conclusions.

CUDA compilation keeps length buckets static and marks diffusion batches greater
than one as dynamic. A batch of one retains its own specialization. CUDA tests
verify reuse of one graph when changing diffusion batch 2 to 4, with both native
and custom kernels. Padded attention tests cover strided biases, duplicate slots,
empty rows, mixed lengths, independent B/D axes, and FP32/BF16 fullgraph execution.
The compiled transition test also changes weights without recompiling and checks
that outputs reflect the new weights.

## Integration

The gather-attention device kernel was extracted from
[Anthropic's RFD3 optimization kit](https://github.com/anthropics/uplifting-biomolecular-modeling/tree/f4f62fa6592ae4938d49b1757bea0cfeff9f468e/rfdiffusion3)
at commit `f4f62fa6592ae4938d49b1757bea0cfeff9f468e`:

- `common/opt_core/opt_core/kernels/gather_attn.py`: atom attention gathers keys,
  values and pair biases directly inside the kernel, uses online softmax, and
  fuses output gating. Sorted duplicate neighbors count once, matching the dense
  masked attention path. The full `P_LL` representation remains resident.
- The transition is now a repository-native implementation of
  `layers/layer_utils.py::Transition.forward`. A Triton kernel computes separate
  input projections and SwiGLU on a two-dimensional row/hidden grid, then PyTorch
  performs the output matrix multiplication. RMSNorm remains unchanged, with
  BF16 rounding at the native Linear, SiLU and multiplication boundaries.
  Dispatch serves BF16 autocast at channel widths 128/256; other transitions use
  the original PyTorch body. Local measurements selected 64-row tiles and
  64/128-hidden tiles (the wider tile for at least 8,192 rows), with 32-channel
  reduction tiles, four warps and two stages.
- Padded atom attention uses a separate repository-native sparse slot kernel.
  It accepts gathered pair biases and explicit neighbor validity, preserves
  duplicate-slot semantics, and returns zero for empty rows. Input examples
  (`B`) and diffusion samples (`D`) remain independent. Dense token attention
  retains its existing implementation. The padded attention and transition have
  compiler-visible custom operators with fake tensor implementations, so
  `fullgraph=True` captures them without Python graph breaks. Transition weights
  are graph inputs, so loading new weights does not reuse stale cast weights.
- The kit's invariant-projection caching approach is implemented using a native
  rollout-local context: six atom-pair bias projections and conditioning
  downcasting are reused. References keep cache identities valid; CFG reference
  inputs have separate entries; normal return and exceptions both clear the
  context. Separate BF16 transition weights are scoped to the same lifetime.

The graph boundaries are a native adaptation of the kit's graph-replay approach.
Token, atom-encoder and atom-decoder stacks replay independently. Neighbor
selection, diffusion noise, motif handling, recycling control and sampling remain
outside the graphs. Variable tensor inputs, including conditioning and neighbor
indices, are copied into graph buffers before replay. Immutable rollout pair
tensors and decoder atom-to-token mappings are retained directly, with their
identities included in the graph key; another same-shaped CFG mapping or pair
tensor gets a separate graph. Callers must not mutate these constants during a
rollout. Returned outputs are copied so later replay cannot overwrite an earlier
result. Graphs are also keyed by input shape/dtype/device and autocast state, and
released after each rollout.

Atom validity masks and flat gather/scatter indices are computed once per fixed
mapping. Integer `index_copy_` / `index_select` replace repeated masked scatter
and dynamically sized boolean indexing inside the accelerated rollout. This
removes repeated `unique`, scalar reads and `nonzero` from the decoder and allows
its upcast/downcast blocks to be captured. Training and the default backend keep
the original implementations.
Capture failures propagate. The padded denoiser is compiled when requested;
no global monkeypatch is installed, and training keeps its autograd implementation. Checkpoint state-dict
keys are unchanged.

Licenses are mapped per component in `src/rfd3/model/kernels/NOTICE`.
`gather.py` retains Anthropic's Apache-2.0 license (`LICENSE`), copyright and
modification notices. The new `transition.py` uses the repository's BSD-3-Clause
license (`FOUNDRY_LICENSE`). The imported `fpf_transition` device code, helpers,
interleaved weight packing and launch tables have been replaced; its associated
`PROTENIX_LICENSE` and component notice are no longer shipped. This is scoped to
that removed component, not a claim about every pre-existing reference in the
repository.

The replacement was written from the existing RFD3 operations and Triton's public
API, using a different two-stage structure. Development included prior review of
the imported kernel; this is documented provenance, not a clean-room legal
certification. The upstream H100/A100 exact-mode guarantee is not claimed.

The remaining Anthropic code still requires its applicable notices and license
on redistribution. Apache-2.0 allows commercial use without requiring source
publication; these code licenses do not replace the repository's or model
checkpoint's separate terms. See
[Apache-2.0 sections 2–4](https://www.apache.org/licenses/LICENSE-2.0).

## Reproduce measurements

```bash
# Fixed real checkpoint denoiser inputs; five warmups, 20 timed repetitions.
FOUNDRY_DEVICE=cuda OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python models/rfd3/benchmark_inference.py \
  --length 128 --batch 1 --warmup 5 --repeats 20 \
  --modes baseline kernels graph kernels_graph \
  --out local/results/step-128

# Profile one mode per process, after its timing has finished.
FOUNDRY_DEVICE=cuda .venv/bin/python models/rfd3/benchmark_inference.py \
  --length 128 --modes kernels_graph --profile \
  --out local/results/trace-128

# Complete 200-step sampling, alternating baseline/accelerated order.
FOUNDRY_DEVICE=cuda .venv/bin/python models/rfd3/benchmark_sampling.py \
  --length 64 --steps 200 --repeats 2 --out local/results/sampling-64
```

Both scripts accept `--checkpoint /absolute/path/rfd3_latest.ckpt`. They select
four CPU threads. Run one benchmark process at a time, with no concurrent tests
or other GPU workload. The step script captures one real denoiser call from the
engine, then compares backends on those same tensors. Reports retain every wall
and CUDA-event sample, first-call latency, peak allocated memory and output error.
CUDA event spans include GPU idle gaps while the host dispatches operations;
they are not summed kernel execution time. Cold latency excludes Python imports,
checkpoint loading and featurization. Earlier modes can warm Triton specializations
used by later ones. Profiling is deliberately limited to one mode per process:
on this stack collecting a trace inflated subsequent eager timings in that
process. Trace runs are diagnostic, not the source of the speedup table.

The sampling script warms preprocessing and CUDA with a short rollout, then times
full engine calls at the requested schedule length. The default schedule of 200
points executes 199 denoiser calls, each with two recycles. These timings include
featurization, initialization, graph capture, sampling and output construction,
but exclude loading the engine and writing CIF/JSON files. Every measured design
is saved after timing, checked for finite coordinates and the requested C-alpha
count. Repeated seed-42 runs expose unchanged-model variation as well as
accelerated differences. These checks do not establish biological designability.

## First integration: token graphs (2026-10-06)

The first two measurement sections below used the subsequently removed imported
transition. Their raw records are retained as historical measurements. The
independent replacement is measured separately below; current benchmark modes
use that replacement unless `kernels_graph_torch_transition` is selected.

GPU: NVIDIA RTX A4000, 16 GiB. Baseline: this repository at
`829b3a1806ddee6e2dccebf911a6a9bf841075d5`, default atom dense-SDPA rule and
`dense_attention_backend=vanilla`. Checkpoint SHA-256:
`9b3f85923e0d51e9453e15cdd2f8c666e7ce096a60577f57d11bbc54ae6d67c1`.
Raw timed samples and errors are retained in
[the measurement record](../performance/rtx_a4000_2026-10-06.json).

Warmed median milliseconds per denoising call, including both recycles:

| Residues | Samples | Baseline ms | Kernels + cache ms | Graph + cache ms | Both ms | Speedup | Peak allocated GiB (baseline / both) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1 | 123.7 | — | 67.2 | 63.9 | 1.94× | 1.37 / 1.42 |
| 128 | 1 | 129.9 | 128.0 | 73.3 | 70.7 | 1.84× | 1.51 / 1.68 |
| 256 | 1 | 199.6 | 152.4 | 127.2 | 104.0 | 1.92× | 2.10 / 2.71 |
| 128 | 4 | 177.4 | 145.8 | 124.8 | 99.8 | 1.78× | 1.75 / 1.82 |

The 64-residue case used 3 warmups / 10 samples; the other cases used 5 / 20.
All timings came from serial runs without profiler instrumentation. The 128- and
256-residue runs also measured kernels and graph replay separately. The paired
features and two recycles were kept in every case. These memory figures cover
the retained denoiser inputs and forward calls, not the initializer's peak.

**Complete 64-residue sampling, 200 schedule points / 199 denoiser calls:**
baseline 28.75 and 29.77 s; accelerated 16.12 and 14.77 s.
The mean fell from 29.26 to 15.45 s
(**1.89×**). The first accelerated rollout includes its first Triton
use in that process; each accelerated rollout includes graph capture. Peak
allocated memory across these full runs was 1.94 GiB. Loading the engine
took 11.77 s and is excluded from these rollout times, as are Python imports.

Both accelerated designs had 64 C-alpha atoms, finite coordinates, and aligned
C-alpha RMSDs of 0.039 and 0.041 Å relative to the first baseline design. Median
adjacent C-alpha distances were 3.808–3.811 Å. The second same-seed baseline
itself differed by 6.832 Å: this sampling harness is not bitwise deterministic.
This is a small execution/numerical check on unconditional monomers, not a
biological-quality or conditioned binder-design validation.

On identical fixed denoiser inputs, the unchanged 128-residue baseline repeated
with coordinate RMS error 0.0536 Å; the accelerated error was 0.0623 Å relative
to the first baseline. At 256 residues these were 0.0772 and 0.0758 Å. The full
error records include sequence logits and maximum absolute differences. Kernel
unit tests compare against dense attention / native SwiGLU; token-stack graph
tests require bitwise equality with eager execution while changing conditioning,
neighbor topology and input shapes, and verify that replay preserves prior outputs.

## Profile findings and remaining work

The baseline 64-residue profile showed roughly 26.3 ms of GPU kernel execution
inside a 124 ms uninstrumented wall-time span. Repeated CPU launches dominate
this small workload. There were 615 `aten::mm` calls, 1,955 `aten::copy_` calls,
and 200 fused RMSNorm calls per denoising call. Weight/activation conversion was
visible throughout the trace. The graph removes Python dispatch from the 18-block
token stack, while the gather kernel avoids dense masked-bias materialization and
the transition kernel removes intermediate writes. The kernels' contribution
grows with atom count and diffusion batch size. The final accelerated trace shows
22.6 ms of GPU kernel execution, 182 CPU `aten::mm` calls and 717 CPU
`aten::copy_` calls. It records two token-stack graph replays, nine fused gather
launches and eight fused transition launches per denoising call. Device GEMMs
inside graph replay still execute; the reduced CPU counts reflect bypassed
Python/ATen dispatch rather than removal of those matrix products.

The follow-up integration below addresses atom encoder/decoder dispatch and
fixed atom indexing. Remaining opportunities include repeated precision
conversions and coordinate/index preparation. Capturing a whole denoiser would
require removing its remaining host scalar reads and dynamic indexing.
The upstream whole-model compile/graph overlay was not copied over this repo's
newer padded-batching and training code. Large batches and other GPU architectures
need their own measurements before changing the defaults.

Local diagnostic traces, logs, frozen environment and saved CIF designs are in
`local/results/rfd3-kernels/` (ignored by Git). `profile64/baseline.trace.json`
is the baseline trace; `final-profile64/kernels_graph.trace.json` is the final
accelerated trace. Earlier exploratory runs in that directory are not included
in the measurement record. In particular, timings collected after an earlier
profiler session and an overlapping-test run were discarded.


## Follow-up: cached atom indexing and atom graphs (2026-10-06)

Same GPU, checkpoint, precision, two recycles and full resident `P_LL`. Paired
fixed-input runs used five warmups and 20 timed calls per mode. The previous
accelerator is reproduced by `kernels_token_graph`, which disables the atom
layout cache and atom graph boundaries. `kernels_cached_graph` adds only the
indexing change; `kernels_graph` now includes both changes. All timed samples,
including host-latency outliers, are retained; these are medians, not minima.

| Residues | Samples | Original PyTorch ms | Previous acceleration ms | Current acceleration ms | Improvement over previous | Overall speedup | Current peak GiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 134.0 | 69.9 | 42.6 | 1.64× | 3.15× | 1.72 |
| 256 | 1 | 195.5 | 101.3 | 74.3 | 1.36× | 2.63× | 2.75 |
| 128 | 4 | 176.9 | 101.8 | 81.2 | 1.25× | 2.18× | 1.87 |

At 128 residues / one sample, cached indexing alone reduces the previous
69.9 ms to 54.3 ms; atom graph replay reduces this to 42.6 ms. Peak allocated
memory rises from 1.67 GiB with the previous accelerator to 1.72 GiB. The
low-memory mode is never needed or enabled.

**Full 64-residue sampling, 200 schedule points / 199 denoiser calls:** the two
baseline runs took 29.20 and 27.44 s; accelerated runs took 10.77 and 9.17 s.
Means are **28.32 → 9.97 s (2.84×)**, including graph setup and the first
accelerated run's process-local Triton startup. Engine loading (11.33 s) and
file writing are excluded. Maximum allocated memory across these runs was
1.98 GiB. The order was baseline, accelerated, accelerated, baseline.

Both accelerated designs contain 64 C-alpha atoms and finite coordinates, with
median adjacent C-alpha distances of 3.814 and 3.812 Å. Their aligned C-alpha
RMSDs to the first baseline are 7.657 and 7.667 Å, while the second baseline
itself differs by 7.661 Å. Against that second baseline, accelerated RMSDs are
0.101 and 0.062 Å. The complete pairwise comparison is retained; these results
do not establish exact trajectory equivalence or biological quality. On fixed
128-residue denoiser inputs, unchanged-baseline repeat RMS error is 0.0505 Å and
the current accelerator's error is 0.0546 Å; at 256 residues they are 0.0700 and
0.0723 Å, respectively.

Separate 128-residue diagnostic traces show five graph replays (one encoder,
two token-stack and two decoder replays), compared with two previously. CPU
`aten::mm` calls fall from 182 to 58, `aten::copy_` from 717 to 298,
`aten::_unique2` from 12 to 3, and `aten::nonzero` from 12 to 6. Summed GPU
kernel execution is similar (29.94 vs 29.22 ms); captured matrix products still
execute on the device. The main gain is reduced host dispatch and synchronization.
Profiler timings are diagnostic and are not used in the performance table.

Reproduce the ablation with:

```bash
FOUNDRY_DEVICE=cuda RFD3_LOW_MEMORY_MODE=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python models/rfd3/benchmark_inference.py \
  --length 128 --warmup 5 --repeats 20 \
  --modes baseline kernels_token_graph kernels_cached_graph kernels_graph \
  --out local/results/index-graph-ablation
```

Raw measurement records are in
[the follow-up report](../performance/rtx_a4000_atom_graphs_2026-10-06.json).
Local traces, logs and generated designs are retained under
`local/results/rfd3-index-graphs/`.

## Independent transition replacement (2026-10-06)

The transition was profiled before replacement, with only transition dispatch
disabled in `kernels_graph_torch_transition`. Gather attention, caches, atom
graphs and token graphs stay enabled in both modes. Each denoiser measurement
used five warmups and 30 synchronized samples. Imported and replacement kernels
were measured in separate serial processes, each with a native-transition
control; they are not an interleaved comparison in the same process.

| Residues | Native-transition controls, before / after (ms) | Imported transition (ms) | Independent transition (ms) |
|---:|---:|---:|---:|
| 128 | 44.21 / 44.41 | 42.65 | 42.91 |
| 256 | 79.44 / 79.66 | 74.23 | 74.19 |

Removing the imported transition alone cost roughly 4–7% of denoiser time.
The replacement preserves denoiser speed within 1% in these runs. It is not
uniformly faster in isolation: with FP32 post-normalization inputs, the smallest
tested transition (4,096 rows, 128 channels, 256 hidden units) takes 0.063 ms,
versus 0.032 ms for the imported kernel. At 65,536 rows / 512 hidden units,
times are 0.823 vs 0.835 ms. Both sets of raw samples are retained.

A diagnostic run records the actual normalized activation shapes and dtypes:
the accelerated pair transitions have shape `[1, 128, 128, 128]`, BF16 dtype,
and 256 or 512 hidden channels. The 384-channel FP32 token transitions still
use the original implementation. The trace contains eight `_project_swiglu`
launches, no removed `_fused_transition_kernel` launches, nine gather launches
and five graph replays. Summed GPU kernel execution is 29.16 ms; trace timings
are diagnostic and excluded from the table.

On BF16 synthetic activations at pair-grid sizes of 4,096, 16,384 and 65,536
rows, with hidden widths 256/512, the post-normalization replacement is
1.55–2.68× faster than native PyTorch in the isolated graph benchmark. All six
BF16 checks and all six FP32 checks were bitwise equal to the native expression
for those seeded inputs; this is not a general exactness guarantee. Timing
excludes RMSNorm, compilation and initial weight preparation. Native graph
capture disables autocast's weight cache, while the accelerator explicitly
retains BF16 weights; these microbenchmarks include that implementation
difference and do not predict the denoiser speedup by themselves.

Reproduce current measurements with:

```bash
FOUNDRY_DEVICE=cuda RFD3_LOW_MEMORY_MODE=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python models/rfd3/benchmark_inference.py \
  --length 128 --warmup 5 --repeats 30 \
  --modes kernels_graph_torch_transition kernels_graph \
  --out local/results/independent-transition-128

RFD3_LOW_MEMORY_MODE=0 .venv/bin/python models/rfd3/benchmark_transition.py \
  --dtype bfloat16 --label independent --out local/results/transition-bf16
```

Add `--record-transition-shapes --profile` to a single-mode denoiser run to
record real activation metadata and a trace. The before/after timing records,
offline tile sweep and provenance are retained in
[the transition measurement record](../performance/rtx_a4000_independent_transition_2026-10-06.json).
Copies of the latest artifacts are under `outs/independent-transition/`.
The full 64-token, 199-denoiser-call sampling measurement was 28.82 s native
versus 10.82 s with the independent transition (2.66x). That is slower than
9.97 s in the previous imported-transition run; the near-equal 128/256-token
step measurements do not establish a uniform full-sampling improvement.
Coordinates were finite, but accelerated samples differed by 6.95–7.45 Å
CA-aligned RMSD from the first native sample (native repeat: 2.20 Å).
This is numerical validation, not biological quality validation.

## Validation

The first integration's targeted suite passed **71 tests** with 5 opt-in platform/compiler tests
skipped. It covers the new CUDA kernels and graph/cache lifetime tests, dense
attention, legacy-versus-padded batching, full-graph reuse across masks/times,
and CUDA Inductor parity. Kernel tolerances are explicit in
`tests/test_inference_kernels.py`; the tensor-only token graph is checked bitwise.
Ruff 0.8.3 checks and formatting pass. A wheel was built and inspected to verify
that the optional kernels, their wrapper, and all three license/notice files are
included. A real `rfd3 design` CLI smoke run also exercised both new config
flags and logged `kernels=triton, token_cuda_graph=True, low_memory_mode=false`.
Scoped mypy checks pass for `inference_acceleration.py` and `model/RFD3.py`.
Checking `engine.py` reports two existing issues (the `torch.compile` options
dictionary type and an unused override suppression); checking the unmodified
file from the base commit reproduces both errors.
Validation logs are retained with the local benchmark artifacts.

After adding cached indexing and atom graphs, the targeted regression suite
passed **44 tests**, with one Metal-only test skipped. This covers kernel/cache
tests, dense attention, batch components and batched model parity. New encoder
and decoder tests require bitwise equality against the original indexing path
for both PyTorch and Triton, changing conditioning, neighbor indices, same-sized
atom mappings and pair constants, and checking that replay preserves earlier
outputs. Uneven token sizes and non-contiguous atom activations are also covered.
Ruff checks and formatting pass, as does scoped mypy for the acceleration helper.
All follow-up tests and benchmarks were run
with `RFD3_LOW_MEMORY_MODE=0`.

The independent transition suite passed **48 tests**, with one Metal-only skip.
The built wheel contains the independent transition, gather kernel, Apache
license/notice, and repository BSD license; it contains no Protenix kernel license.

## Persistent compilation cache

`RFD3_COMPILE_CACHE_DIR` supplies the default `compile_cache_dir` in both the
YAML inference config and `RFD3InferenceConfig`. If unset, this value is `null` /
`None`: RFD3 does not choose a cache directory or change the framework's cache
environment. An explicit config value overrides this environment default.

For example, `RFD3_COMPILE_CACHE_DIR=./outs/compile-cache` persists Inductor FX
graphs and generated code across processes. Custom Triton kernels default to
the directory's `triton/` subdirectory unless `TRITON_CACHE_DIR` is already set.
An explicit non-default `TORCHINDUCTOR_CACHE_DIR` takes precedence; `null` leaves
PyTorch's cache configuration alone. The compiler's automatically populated temporary
location does not override the configured directory. Explicit `TORCHINDUCTOR_FX_GRAPH_CACHE=0`
and PyTorch's force-disable-cache setting are respected. This stores generated
code, not checkpoints or live CUDA graphs. Matching architecture, shapes,
dtypes, GPU and compiler configuration can reuse code after weights are loaded.
New shapes or changed compiler configuration may compile new variants. A cache
hit still incurs process startup, tracing/guards and device initialization.
Live CUDA graphs contain process-local pointers and are captured again.

### Digs cache location

For reuse across jobs/nodes, the recommended shared location is:

```bash
export RFD3_COMPILE_CACHE_DIR=/projects/ml/pipelines/compile-cache/rfd3/$USER
```

This is an opt-in setting for the job environment; the repository does not set
it globally. On `g2125`, `/projects/ml/pipelines` is on the shared
`nfs-5:/data/projects/ml` mount (about 8.9 TiB available when checked), and is
writable by this account. Its existing `cache/` directory is owned by another
account and is not writable here, so the proposed path uses a new sibling.

A small filesystem probe on 2026-10-06 compared 128 files of 8 KiB, three rounds
per location. Median buffered write+rename / stat+read times were:

| Storage | Write+rename, seconds | Stat+read, seconds |
| --- | ---: | ---: |
| Local `/scratch/jbutch` (NVMe) | 0.020 | 0.008 |
| Home NFS (this checkout's `outs/`) | 1.196 | 1.132 |
| `/net/scratch/jbutch` (NFS) | 1.081 | 1.214 |
| `/projects/ml/pipelines` (NFS) | 0.445 | 0.300 |

The project mount was the fastest shared location in this probe. Local scratch
is faster for active cache I/O, but is node-local; neither cross-node visibility
nor retention across cleanup should be assumed. `$TMPDIR` currently contains the
Slurm job ID and should not be the sole cache for reuse across jobs. For jobs
where startup latency dominates, a node-local cache seeded from a shared cache
is an additional option. This probe measures small-file I/O, not end-to-end
compiler startup, and ran while kernel validation was active. Source and raw
measurements are in `outs/cache_storage_probe.py` and
`outs/cache-storage-probe.json`; all probe directories were temporary.

The length-bucket benchmark uses separate processes for capacity probes and
warm runs, preserving the disk cache. Batch means diffusion samples for one
input. It excludes the first three denoiser calls from the warm sampling window,
includes sampler work between subsequent calls, and reports setup separately.
The default 200-point schedule gives 196 measured warm calls. Full rollout
wall time and throughput projected from the warm window are reported separately.
Each successful run saves CIF.gz structures and metadata beside its report;
7-point capacity probes are labeled separately from the final samples:

```bash
RFD3_LOW_MEMORY_MODE=0 TORCH_COMPILE_DISABLE=0 .venv/bin/python \
  models/rfd3/benchmark_buckets.py --out outs/buckets \
  --lengths 64 128 256 384 512 --steps 200
```

## Warm length comparison (2026-10-06)

The finalized local figure and its source data are
`outs/rfd3_speed_by_length.{png,svg,pdf,csv}`. Reproduce the figure with
`outs/plot-env/bin/python models/rfd3/plot_benchmarks.py` (Matplotlib is installed
in that separate plotting environment). Successful runs retain CIF.gz structures
and metadata in `outs/buckets/plot-{mode}-l{tokens}-d1/structures/`.

These measurements use diffusion batch 1, two recycles, the full 200-point
schedule, and standard memory mode. The first three of 199 denoiser calls are
excluded, leaving 196 warm calls. The CUDA-event window includes sampler work
between calls. Model loading, compilation, preprocessing, and file writing are
outside the warm window. The job has one CPU core available; PyTorch uses four
threads consistently across these runs.

These original compiled measurements used the earlier auto policy (native
PyTorch operators compiled by Inductor). They predate custom kernel integration
into the padded path and are retained as the baseline, not relabeled as new runs.

| Tokens | Native eager, ms/step | Compiled native, ms/step | Custom kernels + graphs, ms/step |
| ---: | ---: | ---: | ---: |
| 64 | 140.23 | 48.75 | 35.87 |
| 128 | 140.43 | 49.57 | 43.66 |
| 256 | 204.96 | 50.01 | 76.39 |
| 384 | Initialization OOM | Initialization OOM | Initialization OOM |
| 512 | Initialization OOM | Initialization OOM | Initialization OOM |

The compiled path uses atom/token padding buckets; 64-token inputs pad to 128
tokens. At 256 tokens it processes 4,096 padded atoms and peaks at 14.03 GiB
allocated, versus 3,584 atoms and 11.04 GiB for the eager paths. At 384 and 512
tokens, the dense sinusoidal initializer exhausts VRAM before sampling begins.
No low-memory fallback was used. The main plot focuses on 64–256 tokens;
the original figure including the larger-length failures is retained in
`outs/length-plot-64-512/`. No speed is extrapolated for failed runs. All nine completed trajectories have finite coordinates
and the requested CA count; these checks do not establish biological quality.

The 64-token fresh-process compiled run recorded two FX graph cache hits and
zero FX graph cache misses. The first denoiser input is BF16 and subsequent
inputs are FP32, so there are two legitimate specializations. Cache reuse still
required 50.16 seconds of setup through warmup on this one-core job, excluding
checkpoint loading. Integer neighbor sorts remain native CUDA calls inside the
compiled graph: fusing those sorts into large multi-output Triton kernels caused
excessive assembly time. CUDA fusion also limits unique I/O buffers to 16.


The plotting command accepts `--lengths` and reads the GPU name from the reports,
so separate hardware sweeps can retain separate figures:

```bash
outs/plot-env/bin/python models/rfd3/plot_benchmarks.py \
  --input outs/h200/buckets --out outs/h200 --lengths 64 128 256 384 512
```

`benchmark_buckets.py --worker --profile` records a separate diagnostic trace of
one warm denoiser call. Its report is marked `profiler_instrumented=true`; these
runs must not be used as uninstrumented latency measurements.

## CPU preprocessing and workers (2026-10-06)

The padded inference path currently prepares inputs in the main process.
A CPU-only prototype measured 32 independent 150–200-token inputs on an
8-core Slurm allocation, using the frozen source at `2b0fa6a` and PyTorch 2.11.
Timings include transferring complete examples through the DataLoader, but not
GPU inference or collation. Two later passes reuse persistent worker processes.

| Spawned workers | First batch, seconds | Mean later batch, seconds |
| ---: | ---: | ---: |
| 0 | 6.59 | 6.09 |
| 2 | 14.20 | 3.44 |
| 4 | 11.39 | 1.94 |

Features, coordinate tensors, and example order match the synchronous run
exactly with example-ID-derived seeds. Spawned workers construct the pipeline
from config because the current transform objects contain unpicklable lambdas.
The prototype retains input order, limits prefetching to two examples per worker,
and sets PyTorch to one thread per worker. This is evidence for optional,
persistent workers in bulk inference; it is not a measurement of end-to-end GPU
pipeline overlap. The first-batch cost argues against unconditional worker startup
for short requests. Re-measure after the separate virtual-atom padding optimization.

The prototype, raw timings, preprocessing profile, and a list of independent
optimization candidates are in `outs/non-model/`. No production worker pool is
enabled by these experiments.

## Production worker integration and B=32 attempt

`inference_num_workers` now enables optional spawned CPU preprocessing for the
padded inference path. Its default is two; set `inference_num_workers=0` to run
preprocessing synchronously. Workers reconstruct the transform
pipeline from config, preserve example-ID-derived seeds and input ordering, and
prefetch approximately one inference batch (at least two examples per worker).
The per-worker prefetch factor is
`max(2, ceil(inference_batch_size / inference_num_workers))`. With B=4 and the
default two workers, up to four examples are queued for preprocessing ahead of
consumption. Workers load and transform inputs on CPU while the parent runs GPU
inference; this does not prefetch tensors onto the GPU.
The workers serve all batches within one `engine.run()` call and stop when its
input stream is exhausted; they are not retained across calls. Batching, GPU
inference, output reconstruction, and file writing remain in the parent process.

A production B=4 check with the default two workers measured each main-process
`next(example_iterator)` call without synchronizing CUDA in the retrieval hook.
Across seven later batches, total retrieval time per batch averaged 64 ms and
peaked at 75 ms, compared with approximately 1.81 seconds per model/output
forward call. Retrieval includes IPC, deserialization, and any queue wait;
prefetching does not guarantee zero GPU idle time between batches. Worker startup
added 9.3 seconds before the first batch, and shutdown took 1.9 seconds after the
last batch. There is no per-denoising-step worker synchronization. Raw timings
and 32 sampled structures are under `outs/production-workers/prefetch/`.

With the newly vectorized input padding, an unmodified production initialization
was attempted for 32 independent 150–200-token inputs, D=1, 3072-atom/256-token
buckets, compilation enabled, and a 50-point schedule. All three settings
(0, 2, and 4 workers) exhausted the 96 GB RTX PRO 6000 before the first denoiser
call. The sinusoidal atom-pair initializer's `torch.cos(angles)` requested
36 GiB with only 14.8 GiB free; peak PyTorch allocated memory was 77.48 GiB.
No initialization modification, smaller batch, or low-memory fallback was used.
Consequently these attempts provide no production throughput comparison.
Reports and tracebacks are in `outs/production-workers/unmodified/`, with the
outcome summarized in `outs/production-workers/summary.json`.

### Conservative B=4 production comparison

The subsequent comparison uses B=4 as requested, with the same unmodified
initializer and vectorized padding. It measures 16 independent inputs per
`engine.run()` (four batches), lengths 150–200, D=1, 3072-atom/256-token buckets,
50 schedule points, compiled native kernels, and full resident atom pairs on the
96 GB RTX PRO 6000. The job has eight CPU cores; the parent uses four PyTorch
threads and each worker uses one. Two repetitions reverse the worker-count order.
Checkpoint loading and a 147.3-second compilation/warmup run are excluded;
preprocessing, initialization, sampling, output reconstruction, CIF/JSON writing,
and worker startup/shutdown are included in each measured request.

| CPU workers | Mean complete 16-design request, s | Mean later B=4 batch, s | Later designs/s |
| ---: | ---: | ---: | ---: |
| 0 | 8.79 | 2.188 | 1.828 |
| 2 | 18.26 | 2.091 | 1.913 |
| 4 | 20.99 | 2.038 | 1.963 |

Later-batch means cover the last three batches of each repeat. Four workers
improved sustained throughput by 7.4%, but startup dominates this short request:
the two complete four-worker runs took 18.04 and 23.94 seconds, versus 8.79
seconds for both zero-worker runs. Override `inference_num_workers=0` for short
requests when worker startup outweighs overlap. Longer
streams can amortize startup, but this experiment does not measure persistence
across separate engine calls. Parent-process structure writing remains serial
and took approximately 0.60 seconds per 16-design request in every setting.

Peak GPU allocation was 29.96 GiB in all six runs. Input features and output
coordinate hashes matched exactly across worker settings and repetitions; all
96 structures had finite coordinates and their saved CIF/JSON files were checked.
The compact measurement record is
[rtx_pro6000_workers_2026-10-06.json](../performance/rtx_pro6000_workers_2026-10-06.json).
Full reports, source hashes, plots, and sampled structures are under
`outs/production-workers/b4/`; the reproduction harness and Slurm script are
`outs/production-workers/benchmark_fitting.py` and `b4.sbatch`.

## H200 length sweep

The completed batch-one H200 NVL sweep is in
`outs/h200/rfd3_speed_by_length.{png,svg,pdf,csv}`. All 15 successful reports share
the same model-source hashes, use full resident atom pairs, and retain sampled
structures. The same 196-call warm window is used as in the A4000 comparison.

| Tokens | Native eager, ms/step | Compiled native, ms/step | Custom kernels + graphs, ms/step |
| ---: | ---: | ---: | ---: |
| 64 | 38.71 | 14.86 | 18.37 |
| 128 | 40.63 | 14.81 | 19.40 |
| 256 | 57.79 | 15.42 | 25.15 |
| 384 | 85.97 | 22.98 | 35.44 |
| 512 | 124.05 | 31.62 | 47.33 |

The compiled curve is nearly flat through 256 tokens, then increases at 384 and
512 tokens. These H200 jobs have eight allocated CPU cores rather than the
A4000 job's one; cross-device timing differences therefore also include host
hardware and CPU-allocation differences.

Separate diagnostic traces show 2,668 GPU kernels per compiled denoiser call at
64 tokens and 2,793 at 512 tokens, without CUDA graph replay. Summed kernel
execution was 7.18 and 27.65 ms, respectively. This supports launch/dispatch gaps
as a remaining cost at short lengths, while GPU computation grows substantially
at longer lengths. Profiling perturbs wall time, so these diagnostic sums must
not be subtracted from the uninstrumented timings to estimate an exact overhead
percentage. The traces and their summary are in `outs/h200/profiles/`.

An isolated H200 benchmark of the existing recycling distogram (B=1, FP32,
50 warm repetitions) measured 0.143 ms eager versus 0.030 ms compiled at 512
tokens. At 256 tokens it measured 0.048 versus 0.032 ms. These are wall times per
distogram, not full recycling passes. Compilation fused distance calculation
with binning and replaced the eager integer one-hot plus float conversion with
a direct float output; the 512-token trace had three GPU kernels. The output
still occupies 65 MiB per sample. Exact eager/compiled output equality passed
for the benchmark inputs at all five lengths. This operation is a small cost in
the measured compiled batch-one model; the microbenchmark does not establish
its cost for larger batches or prove equality at every bin boundary. Raw timings
and traces are in `outs/h200/distogram/`.

## Inference synchronization and topology reuse

The padded inference sampler now resolves timestep-dependent Python decisions
in one transfer before the rollout, instead of reading a GPU boolean at each
step. Noise retains its CPU seed contract but uses owned pinned buffers and
asynchronous host-to-device copies. Entropy trajectories, when requested, stay
on device until the rollout ends and then transfer together. Final coordinates
transfer once per batch before per-example finite checks, and output construction
uses the existing CPU example rather than copying its features back to CUDA.
Training forward and loss paths are unchanged.

For B=4, 150–200-token inputs, 3072-atom/256-token buckets and a 50-point schedule,
full-forward diagnostic traces dropped from 142 to 37 CUDA stream synchronizations.
Inside sampling, the count fell from 99 to one. Profiler-added device waits are
excluded from these counts. Synchronization wait duration includes useful GPU
work completing and is not a directly recoverable performance budget. This
first change alone reduced measured model/output forward time by 2.6%, with
exact coordinate equality across before/after runs. Raw results and traces are
in `outs/inference-syncs/`.

Sequence-forced neighbor indices are now sorted once during topology preparation.
Geometric nearest neighbors still update at every denoiser/recycle. This removes
the repeated full atom/token square-grid integer sorts, while retaining the
original slot mask, padding, chain and unindexed-motif rules. The isolated change
reduced B=4 model/output forward time by 7.5% in the alternating-order comparison
under `outs/neighbor-cache/timing/`.

The CUDA worker loader also enables PyTorch's background pinning thread, which
receives and deserializes prefetched examples while the parent samples. At B=4
with two workers, mean retrieval time for later batches fell from 61.1 ms to
0.254 ms (seven later batches per setting). All 32 saved structures matched
exactly between settings. This does not remove initial worker startup or make
workers persistent across engine calls. See `outs/worker-prefetch/`.

The combined synchronization/topology comparison uses the same two-worker pinned
loader for both variants, so it excludes that additional retrieval improvement.
Compilation is warmed, initialization remains unmodified, and structure writing
is included in batch intervals. Two repetitions reverse the variant order:

| RTX PRO 6000, B=4 | Before | After |
| --- | ---: | ---: |
| Mean later batch, seconds | 1.980 | 1.752 |
| Mean later model/output forward, seconds | 1.829 | 1.603 |
| Warm designs/second | 2.020 | 2.283 |

Warm throughput improved 13.0%. Short-request total time remains dominated by
variable process startup: the complete 16-design requests averaged 17.93 seconds
before and 19.63 seconds after, including one slow worker startup in the latter.
Do not interpret the warm improvement as a guaranteed short-request speedup.
Peak allocated GPU memory stayed at approximately 30 GiB. Reports, source hashes,
and sampled structures are in `outs/inference-final/`.

Neighbor selection tests cover changing coordinates, independent diffusion
samples, padding and empty rows, including CUDA fullgraph execution. The cached
and original eager denoisers are bitwise identical on the captured real inputs,
and the CUDA neighbor indices/masks match. The changed compilation graph is not
bitwise equivalent: one real denoiser comparison had coordinate RMS difference
0.0477 versus the old compiled version, compared with 0.0670 between the original
eager and compiled versions. Over 50 schedule points, unaligned C-alpha RMSDs
between compiled versions ranged from 0.176 to 3.807 Angstrom across 16 examples.
Each version reproduced its own outputs exactly across repeats. These checks
establish implementation consistency, not biological quality equivalence.

Two additional experiments were not adopted: enabling Inductor CUDA graph replay
increased mean forward time from 1.643 to 1.700 seconds, and batching all sequence
output transfers did not improve B=4 forward time (1.637 versus 1.654 seconds).
The latter's output-staging implementation was removed; its experiment artifacts
remain under `outs/output-staging/`. Graph-option measurements are under
`outs/compiled-graphs/`.

### Longer inputs and transition-only dispatch

On an H200 NVL, B=1 at 512 tokens with a 200-point schedule, the same
synchronization/topology changes reduced warm sampling from 31.609 to 20.850
ms/step: 34.0% less time, or 51.6% more warm sampling throughput. Two repetitions
reverse the order and exclude the first three of 199 denoiser calls, leaving
196 warm calls. Peak allocated GPU memory remained approximately 52.2 GiB.
Native operators are used in both variants. Each report records source hashes
and the old-function overrides used for the baseline; sampled structures are
retained under `outs/h200-inference-updates/`.

A separate RTX PRO 6000 B=4 comparison retained native attention and enabled only
the independent Triton `Transition` implementation. Across both execution orders
(14 later batches per setting), mean model/output forward time fell from 1.627
to 1.472 seconds, a 9.5% reduction beyond the synchronization/topology changes.
Raw results are in `outs/mixed-transitions/` and its `reverse/` subdirectory.

`auto` now chooses this `triton-transition` policy for compiled inference on SM12x
Blackwell GPUs when `inference_batch_size >= 4`. Other compiled hardware and
smaller input batches retain the native policy. An explicit `torch` selection
always remains native; `triton` retains custom attention and transitions, while
`triton-transition` can be selected explicitly for further hardware comparisons.
The independent kernel still checks supported channel sizes and BF16 autocast
before dispatching. This is inference-only selection; checkpoint weights and
training forward paths are unchanged.

### Final automatic-policy comparison

With the synchronization, topology and transition dispatch changes combined, the
final alternating-order B=4 comparison measured 1.961 seconds per later batch
before and 1.614 seconds after: 17.7% less time and 21.4% more warm throughput.
Both variants use the pinned two-worker loader, so this excludes its separate
retrieval benefit. The six later batches per variant include output construction
and structure writing; compilation and the first batch are excluded. Full
16-design requests averaged 17.85 versus 16.53 seconds over two repetitions,
including variable worker startup. Peak allocated memory remained about 30 GiB.

The final automatic policy was verified as `triton-transition` on the RTX PRO
6000 and remains `torch` for the H200 comparison above. All runs used full resident
atom pairs and unmodified production initialization. The targeted CUDA suite
passed 42 tests. The numerical limitations described above still apply.

The finalized plot is at
[`outs/inference-auto/inference_improvements.png`](../../../outs/inference-auto/inference_improvements.png)
(also SVG/PDF). Raw measurements and all 64 sampled structures from this final
comparison are under `outs/inference-auto/`. A compact checked-in measurement
record, including source hashes, is at
[`performance/inference_optimization_2026-10-06.json`](../performance/inference_optimization_2026-10-06.json).

### Computational biological pilot (2026-10-07)

Existing Pipelines ProteinMPNN, ESMFold2 and RMSD blocks evaluated 16 paired
150–200-residue designs from `before-0` and `after-1` above. Four ProteinMPNN
sequences per backbone were refolded once each: 128 refolds total. Both variants
used the same evaluation settings and model snapshots. A backbone passes when
at least one sequence has aligned CA RMSD below 2 Angstrom and mean ESMFold2
pLDDT at least 80/100. The same 15 of 16 backbones passed in each variant;
49/64 versus 56/64 individual sequences passed. Mean best-of-four CA RMSD was
0.933 versus 0.778 Angstrom, and mean pLDDT was 88.37 versus 89.68.

This small computational pilot found no loss in backbone pass rate, but does
not establish biological equivalence. Each pass rate has a Wilson 95% interval
of 71.7–98.9%; no equivalence margin was predefined. Generator outputs differ
by median 0.676 and maximum 6.802 Angstrom aligned CA RMSD. Inputs cover one
seed and one unconditional length range, with no functional or experimental
validation. Sequence replicates were grouped by backbone for paired bootstrap
intervals; identical pass outcomes make that particular bootstrap degenerate.

The baseline substitutes the sampler, RNG, denoiser and output-forward methods
from revision `749d94d` into the current engine and selects native kernels.
Both variants retain current pinned loading and topology preparation, including
unused cached indices in the baseline. This is a controlled comparison of the
recent inference changes, not a pristine checkout or container comparison.
The exact generation runs supplying the biological inputs took 1.966 versus
1.590 seconds per warm batch (19.1% less time). Complete 16-design requests took
18.265 versus 15.965 seconds; these exclude checkpoint loading and prewarming.
The reversed-order repeat estimate remains 17.7% less warm-batch time as above.
Refolding job durations are not measurements of RFD3 acceleration.

The plot, structures, sequences, protocol, provenance and raw results are in
`outs/biological-equivalence/`. The compact results and exact generation timing
are recorded in
[`performance/biological_pilot_2026-10-07.json`](../performance/biological_pilot_2026-10-07.json).


### Seed-paired cysteine-hydrolase check (2026-10-07)

Four paired 160-residue designs used the 1EUV tutorial constraints and full
200-point generation, followed by one LigandMPNN sequence and one apo ESMFold2
refold per design (eight refolds total). All generation and evaluation completed
on the local 16 GB RTX A4000 without OOM or low-memory mode. Generation used
compiled B=1, unmodified production initialization and zero loader workers.
Feature hashes match between arms. The baseline uses the same method substitution
as above; the optimized arm explicitly selects `triton-transition`, whereas
compiled `auto` selects native kernels on this GPU.

Whole-fold refold CA RMSDs for pairs 1–4 were 3.19→2.87, 2.24→8.38,
9.20→11.91 and 11.68→17.60 Angstrom. Neither arm passed the combined 2 Angstrom /
80 pLDDT criterion. Catalytic atom RMSDs after catalytic-backbone alignment were
6.07→2.08, 2.27→4.81, 6.47→8.69 and 9.00→8.81 Angstrom. Catalytic identities
were preserved; fixed-atom conditioning agreement is not evidence of functional
preservation. Three pairs worsened in whole-fold RMSD, so this small pilot does
not support declaring equivalence. Apo prediction also cannot validate the
covalent ligand complex or catalysis.

The first warmed generation pair took 9.665 versus 9.717 seconds: no speedup
on A4000 B=1. Remaining pairs overlapped host-side PR validation and are retained
as diagnostic timings, not a controlled throughput benchmark. No additional
refolds were launched. Plots, all structures and raw measurements are under
`outs/cysteine-hydrolase/`; the compact record is
[`performance/cysteine_hydrolase_pilot_2026-10-07.json`](../performance/cysteine_hydrolase_pilot_2026-10-07.json).


### Cysteine-hydrolase diagnostic ablation

A single real input (`cys_pair_1`) was reused across baseline, synchronization-only,
cached-neighbor/native-kernel and custom-transition runs on the local A4000.
All runs retained the full 200-point schedule and full memory. No extra refolds
were performed. The baseline repeated bitwise exactly within the diagnostic
process; synchronization removal also preserved all captured trajectories and
outputs exactly, with identical random draws and timesteps.

Cached-neighbor compilation alone changed first-call denoised coordinates by
0.136 Angstrom RMSD over valid model atoms; the transition kernel added a
0.151 Angstrom difference relative to the cached/native run. These grew late in
the rollout. Final cleaned CA RMSDs were 2.55 Angstrom (cached versus uncached)
and 3.22 Angstrom (transition versus cached/native). This isolates numerical
sensitivity, not biological quality.

Eight real-input neighbor checks produced identical indices and validity masks.
A direct eager full-denoiser comparison with/without cached neighbors was also
bitwise identical, while the corresponding compiled calls differed. Thus the
compiled numerical behavior changes despite equivalent eager calculations.
Separately, isolated 128-channel transitions matched eager PyTorch exactly but
differed from default compilation by approximately 0.36% relative L2. Enabling
Inductor's `emulate_precision_casts` on that small reference eliminated the
difference. This establishes a BF16 rounding-contract mismatch with compiled
execution; eager-only tolerance tests do not establish trajectory equivalence.

Uninstrumented forward timings in baseline/native/transition/reverse order were
10.279, 9.237, 9.144, 9.054, 8.950 and 9.202 seconds. Native and custom-transition
means were effectively equal (9.094 and 9.099 seconds). The variable baseline
prevents a robust large-speedup claim. A separate 11-call-denoiser profile saw
30,510 / 30,399 / 30,211 GPU kernels and 36 / 13 / 13 stream synchronizations.
Kernel execution time fell, but launch count barely changed at B1. Profiling
approximately doubled per-call elapsed time; its busy fraction is not a
production-utilization measurement.

A further limitation emerged: fresh-process baseline designs did not reproduce
the original saved baseline (2.41 Angstrom aligned CA RMSD for an untraced replay),
despite identical input feature hashes. Original initial-coordinate tensors and
full runtime state were not retained, so its precise cause cannot be reconstructed.
The production replay controls below independently reproduce cross-process
variation and identify a configuration that eliminated it for this input.
The same-process ablation remains controlled, but is not an exact replay of the
original pilot. Downstream LigandMPNN sequence identities were 78%, 38%, 16% and
41%; it was seeded once per arm rather than independently per design. ESMFold2
was independently seeded per prediction. Establish baseline replay controls and
independent sequence-design seeds before interpreting a larger quality study.

Plots, diagnostic structures, traces and the detailed report are under
`outs/cysteine-hydrolase/investigation/`. The compact record is
[`performance/cysteine_hydrolase_diagnostics_2026-10-07.json`](../performance/cysteine_hydrolase_diagnostics_2026-10-07.json).

### Seeded replay across processes

The seed audit matched all 807 model parameter/buffer tensors, 43 input feature
tensors, 53 prepared feature tensors, all five initializer tensors, initial
coordinates and the first two noise draws exactly across two fresh processes.
These captures used the baseline diagnostic path; the earlier same-process
ablation also matched every random draw across implementations. Missing initial
seeding does not explain that diagnostic's numerical differences.

Separate uninstrumented runs used the current production methods, native kernels,
compilation, full memory, B1/D1, 3072-atom/256-token buckets and the same shared
compile cache. Each configuration generated the same 160-residue cysteine
hydrolase input twice in one process and once in a fresh process. All configurations
repeated bitwise within a process. Fresh-process comparisons were:

| Configuration | Backbone RMSD, N/CA/C/O (Angstrom) | Warm end-to-end seconds |
| --- | ---: | ---: |
| Default | 3.546 | 11.63 |
| `TORCHINDUCTOR_DETERMINISTIC=1` alone | 4.526 | 11.69 |
| Full deterministic controls below | 0, raw coordinates bitwise equal | 23.29 |
| Full controls except `torch.use_deterministic_algorithms(True)` | 3.096 | 22.24 |

The full deterministic reference used these environment variables **before
starting Python**:

```bash
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCHINDUCTOR_DETERMINISTIC=1
```

It also called `torch.use_deterministic_algorithms(True)` before constructing the
inference engine. Removing that call failed the fresh-process check and did not
recover default runtime. The responsible individual CUDA operation has not been
isolated; deterministic compiler tuning alone was insufficient. This is an opt-in
reference recipe; the engine defaults were
not changed. `RFD3_REPRO_PROBE_STRICT=1` in the saved experiment is a switch in
`repeat_production.py` that makes that call, **not** a production inference option.

The local environment was PyTorch 2.11.0+cu130 on RTX A4000. Each warm time is one
complete call including preprocessing and structure writing, after loading and
compilation. These timings suggest roughly 2x overhead for the full controls;
they are not a multi-repeat throughput benchmark. Two process launches for one
input do not establish reproducibility across GPUs, software versions, compiler
caches, backend changes or batch/padding shapes. In particular, reproducibility
within one implementation does not make different compiled graphs or BF16
rounding contracts equivalent.

Keep seed identity stable too: the padded sampler derives CPU streams from the
global seed, example ID, diffusion sample index and event tag. Example IDs can
include the input JSON basename, `global_prefix` and batch index. Pipelines also
adds the chunk index to `seed_offset` and uses chunk-dependent prefixes. Reusing
the integer seed while renaming inputs or changing chunk assignments is not
necessarily replaying the same random stream. The inspected Pipelines code also
omits the seed override when `seed_offset=0`; that separate edge case did not
affect these explicit seed-42 experiments.

Backbone RMSD uses Pipelines IO and rigid alignment of all 640 matched N/CA/C/O
atoms, with separate CA alignment reported alongside it. There is no outlier
trimming or downstream sequence design/refolding. The harness, JSON/CSV results
and structures are in `outs/cysteine-hydrolase/investigation/`; the compact record
is [`performance/reproducibility_2026-10-07.json`](../performance/reproducibility_2026-10-07.json).

### Pipelines image defaults versus additional transition acceleration

The earlier cysteine-hydrolase comparison used compilation in **both** arms.
Its native and custom-transition means of 9.094 and 9.099 seconds measured
incremental kernel changes, not the gain over the deployed eager image.

A direct local image comparison used the same checkpoint, cysteine-hydrolase
input, seed 42, B1/D1, 200 schedule points, native kernels, full memory and four
PyTorch CPU threads. The fast image was also tested with compilation disabled.
Both containers use PyTorch 2.11.0+cu128. Each mode had two warmed repeats:

| Runtime | Warm inference-call seconds |
| --- | ---: |
| Original `foundry.sif` | 41.77, 40.69 (mean 41.23) |
| `foundry_fast.sif`, compilation off | 41.37, 40.79 (mean 41.08) |
| `foundry_fast.sif`, compilation on | 13.38, 12.56 (mean 12.97) |

Thus the fast image was 3.18x faster warm (68.5% less elapsed time) in this case.
These calls include input preparation and sampling, but exclude model loading,
container startup, output writing and scheduling. Preprocessing workers were
disabled for these repeated one-input calls; the production image default remains
two. At B1 the compile switch also selects padded inference instead of the legacy
eager path, so this measures the practical configuration switch rather than
compiler code generation in isolation. It is not a numerical-equivalence test.

The first cold compiled call took 388.55 seconds, including compilation. The
measured timings imply roughly 14 calls to amortize that cold cost versus eager
execution; this is an estimate for this input, not a general threshold. Short
cold jobs can be slower. Persistent caches and multiple batches per process are
important, and a cache still does not retain loaded weights or live CUDA graphs.

The custom transition's limited end-to-end gain has two concrete explanations.
The checkpoint has 48 conditioned transition modules, whose implementation does
not dispatch to this kernel, versus 16 plain transition modules; plain transitions
are additionally restricted to channels 128/256. The existing denoiser profile
reduced GPU launch count only from 30,399 to 30,211 (0.62%).

A new counterbalanced microbenchmark compared compiled native and compiled custom
post-normalization transitions inside the shipped image. At 3,072 rows, 128
channels and expansion 4, CUDA-graph device time improved from 60.01 to 33.98
microseconds, but ordinary host-driven calls worsened from 142.52 to 246.77
microseconds. At 65,536 rows, host-driven calls improved from 1.227 to 0.722 ms.
Thus GPU savings can survive on large matrices while dispatch overhead erases
them on small workloads. The operator remains a Python-dispatched custom-op
boundary, and its output projection still uses native matrix multiplication.
This microbenchmark excludes normalization/weight conversion and is evidence
about that boundary, not a full-model runtime decomposition. Keep `auto` dispatch
rather than assuming a forced transition kernel improves every workload.

The Pipelines integration branch now exposes optional `compile_model`,
`inference_kernel_backend`, `inference_num_workers` and `inference_batch_size`
overrides. Unset fields inherit the image: compile=true, backend=auto, workers=2,
independent-example batch=1. Pipelines `batch_size` instead sets diffusion samples
per input. RFD3's CPU request is four cores to accommodate preprocessing workers.
The actual image's composed configuration was validated with both default and
explicit override arguments. A site seed-argument mapping also fixes the newer
adapter's `model_seeds` argument for this private image, which requires `seed`;
zero is preserved as an explicit seed. These Pipelines updates are on the
`claude-rfd3-fast-digs` integration branch; deployment to main requires merging
that branch.

Artifacts, sampled structures, raw timings and the image-comparison plot are in
`outs/inference-defaults/`. See
[`performance/pipelines_defaults_2026-10-07.json`](../performance/pipelines_defaults_2026-10-07.json).
