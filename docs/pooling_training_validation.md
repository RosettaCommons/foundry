# Pooling: controlled training validation

This extends the [pooling investigation](deterministic_pooling.md). Tests ran on a
CoreWeave H200 with PyTorch 2.11.0+cu128, AtomWorks 2.2.1 and production source plus
the pooling fix. No #397 acceleration or AtomWorks 3 source changes are present.

**Conclusion:** the fix follows the mathematical mean more closely in these
training computations. BF16 training trajectories are **not equivalent to the
legacy implementation**. These controlled numerical tests do not establish
training convergence, checkpoint designability, or preservation of a benchmark's
scientific quality.

## Controls and criteria

Three arms share identical weights, inputs, targets, RNG seeds and optimizer
settings: native `index_reduce(mean)`, fixed pooling, and an FP64 pooling oracle.
The oracle averages the *same already-quantized activations*, casts to the original
dtype, and uses ordinary differentiable PyTorch operations. It does not replace the
whole network with an FP64 model. Comparisons use relative L2 error, with the oracle
norm as denominator and a `1e-12` floor. Optimizer comparisons use parameter
**changes**, not total parameter norms that would conceal update differences.

Tolerances were written before running the experiments:

| Module runs | Output | Scalar loss | Parameter gradients | Parameter updates |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 1e-5 | 1e-5 | 1e-4 | 1e-3 |
| BF16 autocast | 1e-3 | 1e-3 | 5e-3 | 2e-2 |

For the deeper checkpoint test: FP32 output/loss `1e-4`, gradients `1e-3`; BF16
output/loss `0.02`, gradients `0.1`. These are numerical diagnostic criteria,
not scientific quality thresholds. The acceptance gate applies to the fixed
implementation against the oracle; native errors are reported without hiding them.

## Actual pooling modules and optimizer steps

The real `LinearEmbedWithPool` in RFD3, `AtomAttentionEncoderPairformer` in RF3,
and `AtomAttentionEncoderDiffusion` in RF3 were run in training mode. RF3 uses an
actual attention/transition block and activation checkpointing. Widths are reduced
for this test; these module weights are initialized, not pretrained RF3 weights.

There are 20 cases: FP32/BF16, 14/23 atoms per token, four tokens, unsorted indices,
and batch 1/4 for RFD3 and RF3 diffusion (the pairformer encoder has no diffusion
batch dimension). Each arm runs five AdamW steps, `lr=1e-3`, `weight_decay=0.01`,
against the same synthetic MSE targets. All fixed arms passed.

In BF16, fixed outputs, parameter gradients, and parameter updates were identical
to the oracle across every step/case. Scalar loss differences were at most
`1.15e-7` relative. The largest differences for native BF16 were:

| Module | Output relative L2 | Gradient relative L2 | Update relative L2 |
| --- | ---: | ---: | ---: |
| RFD3 pooling | 0.944% | 0.307% | 0.135% |
| RF3 pairformer encoder | 0.563% | 3.63% | 5.66% |
| RF3 diffusion encoder | 0.510% | 3.17% | 7.75% |

For FP32, fixed output/gradient differences were approximately `1e-7`; maximum
update error was `2.21e-5`. Native FP32 behaved similarly. This establishes correct
training plumbing at all three real callsites and shows that BF16 legacy rounding
can change optimizer trajectories. It does not measure which trajectory trains a
better scientific model.

## RFD3 checkpoint denoiser backward

The official checkpoint from the earlier audit was loaded. Inference selects
EMA weights, which are detached; gradients were explicitly enabled on the denoiser
without changing those weights. Conditioning from the token initializer was held
fixed and detached. A 40-residue generated structure from the matched-noise audit
served as the target, with fixed Gaussian noise at sigma 16, two recycles, and the
production `DiffusionLoss` implementation. All 560 coordinate slots, including
virtual atoms, are marked valid for this numerical fixture. Source production weights are used:
`weight=4`, `lddt_weight=0.25`. Both weighted coordinate MSE and smooth LDDT terms
contribute; the MSE term is below its clipping threshold.

This is a prepared numerical fixture using a generated target, **not a natural
training example or a resumed training run**. No full-checkpoint optimizer steps
were taken. The backward pass emitted gradients for 159,166,466 denoiser parameter
entries; unconnected outputs such as the sequence head are not supervised here.

| Precision / arm vs FP64-pooling oracle | Coordinate relative L2 | Loss relative error | Gradient relative L2 | Gradient cosine |
| --- | ---: | ---: | ---: | ---: |
| FP32 fixed | 6.43e-7 | 6.86e-8 | 0.0527% | approximately 1 |
| FP32 native | 8.08e-7 | 4.80e-7 | 0.0662% | approximately 1 |
| BF16 fixed | **0** | **0** | **0** | **1** |
| BF16 native | 1.66% | 2.22% | **45.92%** | **0.9075** |

The BF16 fixed coordinates, loss, and all emitted gradients are bitwise identical
to the oracle. Native BF16's pooling projection weight gradient differs by 42.33%
relative L2. This large difference concerns the backward result through the entire
network after different rounded forward activations; it does not imply that the
native mean's local small-group derivative uses a wrong denominator.

The recorded native measurements are one realization of a nondeterministic CUDA
reduction. Do not treat them as a stable expected error or a population estimate.
The loss dictionary's empty low-noise bin for batch one is undefined and stored as
`null`; absent DNA/RNA categories are not evidence about nucleic-acid quality.
Reported peak GPU allocation is the process high-water mark, not an arm-by-arm
memory comparison.

## Regression tests and reproduction

- Combined helper, existing torch utility and new module training suite:
  **75 passed** on the GPU node (CPU and CUDA helper cases).
- The six new module training regressions also **passed on CPU**.
- New Python files pass the repository-pinned Ruff 0.8.3 checks and formatting.

With both RFD3 and RF3 packages available from this source checkout:

```bash
python -m pytest tests/test_scatter_mean_training.py tests/test_scatter_mean.py tests/test_torch_utils.py -q
python examples/validate_pooling_training.py --device cuda --steps 5 --out callsites.json
python examples/validate_pooling_checkpoint_training.py --checkpoint /path/to/rfd3.ckpt --input /path/to/input.pt --target /path/to/generated_legacy.pt --out checkpoint_training.json
```

The latter inputs are trusted local snapshots produced by the separate
[matched-noise audit utility](https://github.com/Ubiquinone-dot/foundry/pull/1),
without its diagnostic pooling monkeypatch. Input/target/checkpoint hashes, loss
trajectories, per-case maxima and full checkpoint measurements are in the
[machine-readable results](validation/pooling_training_2026-10-13.json).

A natural-data training/validation experiment and external design/refolding
quality evaluation are still needed before claiming scientific equivalence.
