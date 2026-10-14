# RFD3 inference acceleration

## Fast and legacy presets

RFD3 defaults to `inference_mode=fast`. Both modes use the same Foundry
installation, checkpoint format and AtomWorks 3 dependencies.

```bash
rfd3 design inputs=designs.json out_dir=outputs/fast inference_mode=fast
rfd3 design inputs=designs.json out_dir=outputs/legacy inference_mode=legacy
```

For Python, pass `inference_mode="legacy"` to `RFD3InferenceConfig`. Construct
a new engine and use a separate output directory for each mode when comparing.

| Setting | `fast` (default) | `legacy` |
| --- | --- | --- |
| Compilation | Enabled; overrides supported | Disabled |
| Independent-example batch | Configurable, default 1 | Forced to 1 |
| Preprocessing workers | Configurable, default 2 | Forced to 0 |
| Kernels | `auto`; overrides supported | Forced to `torch` |
| Dense attention | `vanilla` by default | Forced to `vanilla` |
| CUDA graphs | Configurable | Disabled |
| Atom-to-token mean | Corrected accumulation | Original `index_reduce` mean |
| Pair mean on padded path | FP32 sum/division for FP16/BF16 | Original serial path |
| Random stream | Example-keyed on padded path | Global RNG, serial path |

Legacy overrides acceleration flags, including explicit compilation, batching,
kernel and graph settings. It preserves checkpoint, sample count, sampler,
constraints, seed and low-memory settings. Fast allows expert overrides;
`compile_model=false inference_kernel_backend=torch` uses corrected pooling
with eager native operators. Unsupported padded inputs take a logged eager
fallback. Output JSON's `inference_settings` records the requested preset,
resolved settings, actual execution path and random-stream family.

### What legacy reproduces

Legacy restores pre-#397 execution choices, **not exact historical coordinates**:

- The original CUDA reduction is nondeterministic and uses low-precision sums.
- AtomWorks 3 preprocessing remains active, including its ligand conformer
  fallback. Reproducing AW2 inputs requires the historical environment.
- Fast and legacy consume different random streams; equal seeds do not imply
  identical noise. Hardware, dependencies, input ordering and batching matter.
- The original reduction does not support MPS legacy execution; CUDA is the
  validated target.

Pooling selection is model-instance-local, including EMA copies. Training uses
corrected pooling in both modes. Neither preset changes checkpoint parameters.
The preset is specific to RFD3; it does not configure RFD3NA, RF3 or MPNN.

## Batching and compilation

`inference_batch_size` batches independent inputs; `diffusion_batch_size`
generates samples of each input. These axes stay independent. Compatible inputs
share padded atom/token/slot shapes; masked tail examples do not produce designs.

`compile_shape_policy=buckets` (default) uses `compile_atom_buckets`,
`compile_token_buckets` and `compile_slot_buckets` to reuse graphs across nearby
lengths. `batch_max` uses each batch's actual post-transform maximum counts,
including ligand and motif atoms, without rounding to buckets or padding
multiples. New shapes can trigger compilation. Diffusion batches greater than
one share a dynamic CUDA dimension; a batch of one has a separate specialization.

`inference_kernel_backend=auto` selects:

- Compiled inference: native operators, except SM12x GPUs with at least four
  independent inputs, which use native attention plus the Triton transition.
- Eager CUDA inference on Ampere or newer: Triton when available.
- Other devices or low-memory inference: native PyTorch.

Explicit `torch`, `triton` and `triton-transition` overrides remain available.
The transition kernel serves BF16 autocast at widths 128/256; unsupported calls
use native operators. Triton comes with the tested Linux CUDA PyTorch wheel.
Dense attention is independently controlled by `dense_attention_backend`.

`inference_cuda_graph=true` enables eager token/atom-stack graph replay and
requires `compile_model=false inference_batch_size=1`. It can use native or
Triton kernels. Graphs and invariant projection caches live for one rollout;
CFG inputs are kept separate, variable inputs are refreshed, and outputs are
copied before subsequent replay. Capture errors propagate. Padded compiled
inference is a separate path and supports both batch axes.

## Persistent compiler cache

`RFD3_COMPILE_CACHE_DIR` supplies the default `compile_cache_dir`. An explicit
configuration value overrides it; `null` leaves PyTorch's cache settings alone.
A non-default `TORCHINDUCTOR_CACHE_DIR` takes precedence, and an existing
`TRITON_CACHE_DIR` is respected. Framework cache-disable settings are retained.

This stores generated code, not checkpoints or live CUDA graphs. Cache reuse
still requires compatible shapes, dtypes, GPU and compiler settings; startup,
tracing and CUDA graph capture can still incur costs.

## Installation and provenance

Use PyTorch >=2.11 and the declared AtomWorks 3 dependencies. Upgrade the standard
Foundry image; no separate fast container or source overlay is required. The
[Dockerfile](../../../examples/docker/Dockerfile) accepts `FOUNDRY_REVISION` to
pin a release or commit. Update dependency pins together and run `pip check`.

The gather-attention kernel derives from Anthropic's optimization kit at
`f4f62fa6592ae4938d49b1757bea0cfeff9f468e`. The padded attention and two-stage
transition implementations are repository-native. Component provenance and
redistribution notices are in `src/rfd3/model/kernels/NOTICE`; retain the supplied
licenses. No universal bitwise-exactness guarantee is made for the accelerated path.

## Regression results

See [RFD3 regression answers](../../../docs/source/atomworks3_port.md#rfd3-regression-answers)
for the pooling correction, final 64-output enzyme comparison, quality limits,
and AtomWorks 3 behavior. Permanent tests cover numerical values and gradients,
NaN padding, empty groups, independent batch axes, CFG, graph lifetimes, compiler
reuse and legacy settings. Development timing reports are not shipped.
