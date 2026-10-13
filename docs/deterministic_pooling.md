# Deterministic pooling prerequisite (2026-10-13)

This change starts from public production `69d5f735f272b78b6b23c230abba849f18315234`.
The pooling and model code is unchanged from `8bffbcb9`, the parent used in the
independent [PR #397 precision audit](https://github.com/Ubiquinone-dot/foundry/pull/1).
It includes no #397 acceleration or AtomWorks 3 changes. **This is not approval to
merge #397, and reproducibility is not evidence of designability.**

The follow-up [controlled training validation](pooling_training_validation.md)
checks real RFD3/RF3 modules, short optimizer trajectories, and a checkpoint
denoiser backward pass.

## Diagnosis

Production's shared `scatter_mean` calls `index_reduce(..., "mean",
include_self=False)`. RFD3 uses it in `LinearEmbedWithPool`; RF3 uses it in
`AtomAttentionEncoderDiffusion` and its pairformer atom pooling. The RFD3 40-residue
fixture pools 14 atoms per token. Residue groups are normally small; 23-member
nucleic-acid-sized and 64-member groups are also tested. Atomized ligand tokens
normally contain one atom. Large groups below are numerical stress cases, not a
claim that ordinary RFD3 protein groups contain thousands of atoms.

There are two distinct problems:

1. CUDA floating-point atomic additions have an unspecified order, so the
   same values can give different rounded sums. PyTorch explicitly documents
   [CUDA nondeterminism for index_reduce](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.index_reduce_.html).
   Strict deterministic mode rejects this operation. The prior audit observed
   4.904 Å aligned final CA RMSD between fresh production processes with the same
   serialized inputs and diffusion noise. This is numerical amplification,
   not evidence of wrong indices or a wrong include-self denominator for normal
   small groups.
2. Low-precision source sums and counts can incur substantial error. The
   [PyTorch 2.11 CUDA implementation](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/Indexing.cu)
   constructs counts with `zeros_like(result)` and accumulates ones with
   `index_add_`. BF16 integer counts stop resolving consecutive integers above
   256; FP16 above 2048. For an already-quantized 3,000-member BF16 linspace from
   -1 to 2, native CUDA returned 2.0 versus the FP64 mean 0.499992. FP16 returned
   0.7793 versus 0.500001. These are real arithmetic failures at stress sizes.
   Even 14/23-member BF16 groups have larger rounding error than an FP32 sum.

The MPS fallback also added a nonzero destination to touched groups, contrary to
its documented `include_self=False` contract. Actual model callers allocate
zeros, so this was a latent helper contract bug, not the observed CUDA cause.

## Implementation and semantics

Stable-sort the indices, gather values in their original dtype, promote FP16/BF16
values to FP32, and use an ordered segmented mean. Counts are integers. Cast only
after division. FP32/FP64 retain their accumulation dtype; floating-point addition
is still finite precision. Touched destinations are excluded; untouched groups
retain their original destination and its gradient. Empty inputs and destinations
preserve both autograd edges. Neither tensor is mutated.

A small autograd function implements mean's linear backward with gather/repeat
operations because PyTorch's native segment backward lacks second derivatives.
First and second derivatives are checked, including gradients of the destination.
The helper does not set global deterministic flags or change RNG state.

MPS retains scatter-add, with FP32 accumulation and correct destination masking.
MPS hardware was unavailable; its runtime and determinism are not certified here.
Sorting, bincount's dynamic shape, and segment validation have overhead and may
synchronize the host. This change targets eager CPU/CUDA execution; fullgraph
compilation, CUDA graph capture, torch.func transforms and performance on other
PyTorch versions require separate validation. It promises neither identical
results across devices/versions nor equivalence to old BF16-rounded outputs.

## Validation

Environment: CoreWeave H200, PyTorch 2.11.0+cu128, AtomWorks 2.2.1, Biotite 1.4.
Official RFD3 checkpoint SHA256:
`9b3f85923e0d51e9453e15cdd2f8c666e7ce096a60577f57d11bbc54ae6d67c1`.

- **69 tests passed**, combining `tests/test_scatter_mean.py` and existing
  `tests/test_torch_utils.py` on CPU/CUDA: arbitrary/negative axes, non-contiguous
  batches, int32 unsorted indices, invalid indices, absent groups, nonzero
  destinations, empty source/destination, four dtypes, first/double derivatives,
  default/strict CUDA forward/backward repeats, and groups 14/23/64/300/3000.
- For 40 residues, batch one, 200 steps, identical serialized features/noise and
  checkpoint: two fresh default processes were bitwise identical in all five
  initializer tensors, first denoiser outputs, final coordinates/logits/sequence,
  all 200 noisy and denoised states, and entropy trajectories.
- Two further fresh processes using `torch.use_deterministic_algorithms(True)`,
  `PYTHONHASHSEED=0`, and `CUBLAS_WORKSPACE_CONFIG=:4096:8` were also bitwise equal.
  **Default and strict settings differ from each other**: first-coordinate maximum
  difference 0.155 Å, final-coordinate maximum 9.923 Å. Changing the arithmetic
  regime changes results; it is not a failed repeat within a regime.
- Two ordinary fresh-process CLI runs, explicit seed 42, length 40, batch one,
  normal 200 steps, with normal preprocessing and RNG: CIF coordinate fields,
  residue sequence and atom identifiers were exactly equal (221 saved atoms).
  This CLI result is equality at CIF writer precision; the separate serialized
  tests above establish bitwise tensor equality. No strict flags were required.
- No scientific refolding/designability comparison, RF3 checkpoint replay,
  full training run, MPS hardware run, or cross-platform reproducibility claim.
  The shared helper's arithmetic and gradient tests cover the RF3 callsites,
  but checkpoint behavior can change because pooling is now more accurate.

### Cost

The standalone `examples/benchmark_scatter_mean.py` measures synchronized wall
latency (including host overhead), forward/backward, and peak extra allocated
CUDA memory. BF16, 200 tokens, 14 atoms/token, 768 channels:

| Batch | Native forward | Fixed forward | Native extra memory | Fixed extra memory |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.060 ms | 0.215 ms | 5.2 MB | 12.9 MB |
| 8 | 0.435 ms | 0.329 ms | 41.5 MB | 103.2 MB |

This is a helper microbenchmark, not a whole-model throughput estimate. The two
ordinary 40-residue CLI diffusion batches took 11.71 and 12.17 seconds, without a
matched pre-fix latency comparison. See the [machine-readable measurements](validation/deterministic_pooling_2026-10-13.json).

### Reproduction

Run from this checkout with its source installed, the RFD3 dependencies, and an
existing official checkpoint (no download or GPU allocation is done by these commands):

```bash
python -m pytest tests/test_scatter_mean.py tests/test_torch_utils.py -q
python examples/benchmark_scatter_mean.py
```

Create `input.json` containing `{"pooling_repeat":{"length":"40"}}`, then run
these in separate processes with the same environment:

```bash
python -m rfd3.cli design inputs=input.json out_dir=repeat1 ckpt_path=/path/to/rfd3.ckpt seed=42 diffusion_batch_size=1 n_batches=1
python -m rfd3.cli design inputs=input.json out_dir=repeat2 ckpt_path=/path/to/rfd3.ckpt seed=42 diffusion_batch_size=1 n_batches=1
```

Compare CIF coordinate/sequence fields, excluding timing metadata and gzip headers.
The serialized-noise replay uses `models/rfd3/benchmark_precision.py` from the
separate audit PR, with this source on PYTHONPATH and `--legacy-only`; its diagnostic
pooling monkeypatch must **not** be enabled for testing this implementation.
