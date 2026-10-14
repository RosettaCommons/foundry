# RFD3 CUDA inference acceleration

## Fast and legacy inference presets

RFD3 defaults to `inference_mode=fast`. To select the existing serial inference
path and original pooling operation:

```bash
rfd3 design inputs=designs.json out_dir=outputs/fast inference_mode=fast
rfd3 design inputs=designs.json out_dir=outputs/legacy inference_mode=legacy
```

Python uses the same setting:

```python
config = RFD3InferenceConfig(inference_mode="legacy", ckpt_path="rfd3")
engine = RFD3InferenceEngine(**config)
```

Use separate output directories (or `skip_existing=false`) when comparing modes,
and construct a new engine for each mode. This preset is currently an **RFD3**
option; RFD3NA, MPNN, and RF3 receive the AtomWorks 3 port but do not acquire
RFD3's padded inference implementation or this dial.

| Setting | `fast` (default) | `legacy` |
| --- | --- | --- |
| Compilation | Enabled by default; existing overrides supported | Disabled |
| Independent-example batch | Configurable, default 1 | Forced to 1 |
| Preprocessing workers | Configurable, default 2 | Forced to 0 |
| Kernels | `auto`, with existing overrides | Forced to `torch` |
| Dense attention | `vanilla` by default | Forced to `vanilla` |
| CUDA graph options | Existing fast-mode settings | Both disabled |
| Atom-to-token mean | Corrected accumulation; batched sum and division in FP32 for FP16/BF16 | Original `index_reduce(..., "mean", include_self=False)` |
| Pair mean on padded path | FP32 sum/division for FP16/BF16, integer counts, pooling autocast disabled | Existing legacy pair-pooling path |
| Random stream | Example-keyed on padded path | Existing global-RNG serial path |

The legacy preset **takes precedence over acceleration settings**, including
explicit `compile_model=true`, `inference_batch_size>1`, kernel and graph flags.
It preserves your checkpoint, diffusion sample count, sampler parameters, input
specification, seed, and low-memory setting. `fast` leaves existing expert
settings configurable; for example, `compile_model=false inference_batch_size=1
inference_kernel_backend=torch` still uses corrected pooling on the eager path.
Unsupported padded inputs can use an explicitly logged eager fallback.

With prediction metadata enabled (the default), output JSON includes
`inference_settings`: requested mode, resolved compilation/kernel/attention/batch
settings, pooling policy, and **actual** execution path/random-stream family for
that output. The latter distinguishes a fast-mode eager fallback from padded
execution. Pooling selection is local to each model instance, including EMA
copies. Training always retains corrected pooling; the legacy option does not
change checkpoint parameters or the process-wide helper.

### What legacy reproduces

`legacy` restores the listed pre-#397 **execution choices** on CPU/CUDA: serial
examples, eager model execution, native Torch/vanilla attention, the legacy RNG
path, and the original atom-to-token reduction. It is useful as an execution
fallback and for comparisons on identical prepared inputs.

It **does not promise bitwise historical coordinates**, even with the same seed:

- The original CUDA `index_reduce` mean can be nondeterministic and uses the old
  low-precision arithmetic. Selecting it deliberately restores those limitations.
- AtomWorks 3 remains installed in both modes. Parser/bond/conformer behavior is
  unchanged by this switch, including the identified UFF-failure reference-mask
  change. A legacy run under AW3 is not an AW2 run.
- Other source fixes, checkpoint choice, Torch/CUDA/RDKit versions, hardware,
  precision flags, input order and batch size can affect results. The preset
  does not restore an entire historical checkout or environment.
- The same integer seed does not imply the same random draws between fast and
  legacy paths. Compare distributions or explicitly matched model inputs/noise.
- MPS legacy execution is not supported by this original `index_reduce` path;
  CUDA/CoreWeave is the validated target for this compatibility option.

For historical reproduction, preserve the old checkout, checkpoint, input and
AW2 environment separately. Even that environment cannot guarantee exact replay
of an inherently nondeterministic reduction. These modes are execution policies,
not a guarantee of protein folding, binding or experimental function.

### Pooling precision regression coverage

The old padded atom mean materialized a FP16/BF16 sum before division; padded
pair contractions could also downcast under autocast. Small fixtures demonstrate
one-rounding-step errors and FP16 sum overflow despite a representable mean.
Both now retain FP32 intermediates through division and cast the final mean back.
Regression tests compare values and gradients against FP64 references built from
the already-quantized inputs, cover NaN padding and empty examples, and exercise
CUDA Inductor/fullgraph execution and autocast. This prevents avoidable rounding;
it does not assert that different reduction trees or complete trajectories are
bitwise equal. It cannot recover precision already lost in input activations.


## One container for both inference modes

Fast and legacy are runtime settings in the same Foundry installation. Upgrade
and rebuild the existing Foundry image with the desired source revision and its
AtomWorks 3 dependencies; no separate fast image or source-overlay build is needed.
The standard [Dockerfile](../../../examples/docker/Dockerfile) accepts a release
tag or commit through `FOUNDRY_REVISION`:

```bash
# Set FOUNDRY_REVISION to the exact merged commit or release being deployed.
docker build -f examples/docker/Dockerfile \
  --build-arg FOUNDRY_REVISION="$FOUNDRY_REVISION" -t foundry:aw3 .
```

For Chaperone, keep the existing `foundry` image identity and update its Foundry
revision, dependency pins and version labels together. Its old recipe pins
AtomWorks 2.2.1 and Biotite 1.4.0: those pins must be replaced with the compatible
AW3 stack, not carried forward or bypassed with `--no-deps`. Run `pip check`, all
four model import checks, and pipeline smoke tests after rebuilding. The runtime
option `inference_mode=legacy` does not restore AtomWorks 2; retain the historical
image by its immutable digest if that environment is needed for comparison.

## Execution settings

Compilation defaults to `compile_shape_policy=buckets`, which reuses compiled
graphs across nearby lengths. Set `compile_shape_policy=batch_max` to use each
batch's actual maximum post-transform atom, token and per-token atom-slot counts. This
includes ligand and motif features, rather than assuming that protein residue
count equals token count. Lengths are not rounded to the preset buckets or
padding multiples in this mode. A new shape may require another compilation;
the optional persistent cache can reuse artifacts for repeated shapes.

The bucket policy uses `compile_atom_buckets`, `compile_token_buckets` and
`compile_slot_buckets`. Both policies are inference-engine options; eager
inference retains its existing padding-multiple settings. Measurements below
that explicitly report buckets used the bucket policy.

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

Use an environment installed from this checkout with PyTorch >=2.11 and its
AtomWorks 3 dependencies.
The current CoreWeave validation used PyTorch 2.11.0+cu128 and AtomWorks 3.0.1:

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
its upcast/downcast blocks to be captured. Training and the unaccelerated eager backend keep
the original indexing implementations.
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
  --out outs/step-128

# Profile one mode per process, after its timing has finished.
FOUNDRY_DEVICE=cuda .venv/bin/python models/rfd3/benchmark_inference.py \
  --length 128 --modes kernels_graph --profile \
  --out outs/trace-128

# Complete 200-step sampling, alternating baseline/accelerated order.
FOUNDRY_DEVICE=cuda .venv/bin/python models/rfd3/benchmark_sampling.py \
  --length 64 --steps 200 --repeats 2 --out outs/sampling-64
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

## Regression and quality results

See [RFD3 regression answers](../../../docs/source/atomworks3_port.md#rfd3-regression-answers)
for the pooling diagnosis, non-exact acceleration, training limits, paired
refolding, 64-output enzyme comparisons, final fast/legacy results and rollout
policy. The AtomWorks 3 port and permanent regression tests remain part of the
same change.

Earlier hardware timing reports remain in [performance](../performance/).
The [historical development record](https://github.com/RosettaCommons/foundry/blob/1abfe1978216dac2066b42d7cedfa064bc6d4d15/models/rfd3/docs/inference_acceleration.md)
links the original environments and measurements. Historical AW2 timings are
not new AW3 performance claims.
