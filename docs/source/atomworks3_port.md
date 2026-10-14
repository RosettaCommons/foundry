# AtomWorks 3 port and regression results

Foundry PR #401 combines PR #397's batched inference, corrected pooling, and the
AtomWorks `>=3.0.1,<4` port for **RFD3, RFD3NA, MPNN and RF3**. CI pins
AtomWorks 3.0.1; the supported Torch minimum is the validated 2.11 runtime.

RFD3 defaults to `inference_mode=fast`; `inference_mode=legacy` restores its
old execution choices. Both modes retain AtomWorks 3 preprocessing, including
its conformer-failure policy. See the [inference options and exact compatibility
boundary](https://github.com/RosettaCommons/foundry/blob/10b9c7407fa71b0f97dd491f7b0b853df63a7d28/models/rfd3/docs/inference_acceleration.md). The dial applies only to RFD3.

## Compatibility changes

- Use `ParseConfig` / `PrepareConfig` and the new annotation/query imports.
  RFD3 and RFD3NA explicitly retain bonds on synthetic coordinates and preserve
  their conditioning annotations.
- Replace the removed polymer-bond helper with CCD attachment-site lookup for
  nonstandard residues. Preserve existing bonds and respect chain, assembly,
  gap, unindexed-motif, and protein/nucleic-acid boundaries.
- Set the former covalent-modification transform defaults explicitly. Preserve
  RF3's training bond cutoff (strictly below 2.4 Å) and inference bond semantics
  in a model-local feature transform.
- Restore RF3 template and conformer constraints after CCD atom completion,
  including nonpolymer `auth_seq_id` renumbering. Added atoms remain unconditioned.
  Missing formal charges are filled from the CCD without replacing supplied
  charges or mutating the caller's array.
- Adapt the existing nested dataset configurations to AtomWorks' native
  `PandasDataset` and structure loader. Portable Generic, PN-unit, and interface
  metadata examples are covered; full training campaigns were not run.

## Port regression coverage

The complete portable suite passes: **1,112 passed, 84 skipped**. Mypy passes
all 256 source files. The package builds as both a source distribution and a
wheel; the wheel contains the new dataset adapter, RF3 bond transform, and
migrated model configs. `uv pip check` reports no dependency conflicts.

`tests/test_atomworks3.py` compares all 41 model input features against saved
AtomWorks 2.2.1 references for RFD3 and mixed RNA/DNA/protein RFD3NA. All values
match exactly. Tests also cover real engine imports, motif file parsing,
conditioning, nonstandard residue/cap bonds, and boundary handling.

RF3 and MPNN tests exercise protein, DNA, glycan, and ACE-cap inputs; file and
in-memory parsing; ion charges; template transport; and RF3 bond distances.
The RF3 NAG regression specifically guards against silently losing template
constraints when the parser changes residue ID 1 to author residue ID 101.

A separate feature comparison on those four modalities found:

| Model | AtomWorks 2.2.1 versus 3.0.1 |
|---|---|
| ProteinMPNN + LigandMPNN | 144/144 arrays exactly equal |
| RF3 | 154/160 arrays exactly equal |

The six RF3 differences are understood: four `ref_pos` arrays differ by rigid
rotations following AtomWorks' corrected QR sign convention; their intra-residue
distances agree within 1.398e-6 Å. Two `ref_charge` arrays change +1 to 0 for
glycosylated ASN ND2 and ACE carbonyl C, matching AtomWorks 3's corrected bonded
chemistry. Both comparisons used the same RDKit version. These are intentional
upstream semantic changes, so universal bitwise equivalence is not claimed.

AtomWorks 3 also retains CCD bonds on synthetic coordinates that AtomWorks 2
discarded. The golden tests include the actual model token-bond features, which
are unchanged in those examples; they do not freeze the raw intermediate graph.

## RFD3 regression answers

### Pooling correction and acceleration exactness

The original CUDA `index_reduce` mean has nondeterministic floating-point
addition order and low-precision sum/count arithmetic. With identical prepared
inputs and diffusion noise, pristine upstream produced a 4.904 Å aligned CA
replay difference in the 40-residue diagnostic. Corrected eager pooling made
fresh-process repeats bitwise identical within the tested execution settings.
Different devices, precision settings or software versions are not guaranteed
to match.

The shared fix uses stable grouping, integer counts and FP32 accumulation for
FP16/BF16. The padded atom and pair means also retain FP32 intermediates through
division, with pair-pooling autocast disabled. Value and gradient tests compare
against FP64 means of the same quantized inputs, including rounding, overflow,
empty groups, NaN padding and compiled CUDA execution.

**The accelerated path is not numerically exact, even after controlling the
original nondeterminism.** The independent 40-residue, 200-step audit found a
2.515 Å aligned CA difference from its deterministic diagnostic reference.
Padding/mask/batch-isolation and graph-lifetime tests found no defect on their
tested inputs; trajectory differences still require quality measurements.
Fast and legacy also use different random streams, so the same seed does not
mean identical draws.

### Effect on design quality

**No quality improvement has been demonstrated.** It corrects arithmetic, but
that is not a guarantee of better predictions from weights trained with the old
arithmetic. The earlier paired experiment used 16 length/seed cases per arm,
two MPNN sequences per backbone, and 128 total RF3 refolds. A backbone passed
when at least one refold had CA RMSD <2 Å and pLDDT >=0.8:

| Historical arm | Backbone passes |
| --- | ---: |
| Original AW2 | 15/16 |
| Corrected eager pooling, AW2 | 14/16 |
| Corrected padded inference, AW2 | 16/16 |
| Corrected padded inference, AW3 | 16/16 |

Pooling alone failed this small pilot's exploratory noninferiority margins;
acceleration and the dependency contrast passed them. This is neither proof of
broad equivalence nor a demonstrated general regression. It predates the final
batched-pooling precision correction.

The expanded enzyme test used the 150-residue 1euv tutorial, 32 predetermined
seeds × two outputs per arm, the public 200-step sampler, and **native JSON
`metrics.insertion_rmsd` <1 Å**, without refolding or filtering outliers:

| Historical arm, before final batched correction | <1 Å | Mean RMSD (Å) |
| --- | ---: | ---: |
| Original AW2 | 63/64 | 0.435 |
| Corrected eager pooling, AW2 | 61/64 | 0.485 |
| Corrected padded inference, AW2 | 57/64 | 0.532 |
| Corrected padded inference, AW3 | 61/64 | 0.478 |

Pooling alone had a +0.050 Å mean penalty on this fixture (paired seed-group
95% bootstrap interval +0.009 to +0.094 Å). The AW3 port had four more passes
and a 0.054 Å lower mean than padded AW2; its contrast intervals included zero.
The initial eight-output AW3 disadvantage did not persist at 64 outputs.

### Final fast and legacy results

After the final batched atom/pair precision fix, all 128 fresh designs completed
on CoreWeave H200 using the same enzyme task, checkpoint, seeds and AW3 stack.
Every native JSON confirmed the requested preset and actual execution path.

| Final AW3 preset | <1 Å | Mean RMSD (Å) | Maximum (Å) |
| --- | ---: | ---: | ---: |
| Fast (compiled padded, auto-selected Torch kernels) | 62/64 | 0.485 | 2.040 |
| Legacy (serial eager, original pooling) | 63/64 | 0.447 | 1.707 |

Fast minus legacy was **+0.038 Å**, paired seed-group 95% interval
**[−0.051, +0.131] Å**. The pass-rate difference was −1.5625 percentage points
[−6.25, +3.125]. There is **no clear quality separation on this fixture**;
these intervals do not establish population equivalence. This H200 comparison
does not certify every optional Triton backend, task or hardware configuration.

### Legacy reproducibility and AW3 behavior

Legacy restores serial eager execution, Torch/vanilla attention, no CUDA graphs,
the global RNG stream and the original atom mean. It cannot guarantee historical
coordinates: the old reduction remains nondeterministic and AW3 remains installed.

For the enzyme fixture, AW3 retains a ligand conformer after UFF optimization
fails, whereas AW2 masks it. The diagnostic changed 22 ligand reference-mask
entries; restoring only the old mask reproduced all eight earlier AW2 outputs.
The port **accepts AW3's upstream fallback in both modes**. The 64-output results
do not justify forcing the old mask on quality grounds. Historical AW2 input
reproduction needs the old environment.

### Training and regression coverage

The numerical training checks passed for real RFD3/RF3 pooling modules and short
optimizer trajectories. In the prepared RFD3 checkpoint backward fixture, fixed
BF16 coordinates, loss and gradients matched the quantized-input FP64 pooling
oracle exactly. Native BF16 gradients differed by 45.92% relative L2 in that
recorded realization. This is a numerical fixture, not a training-quality score.
**Old and corrected BF16 training trajectories are not equivalent; full training
convergence and downstream quality have not been tested.**

Permanent regressions remain in `tests/test_scatter_mean.py`,
`tests/test_scatter_mean_training.py`, `tests/test_atomworks3.py`, and the RFD3
pooling, batching, compilation, kernel and inference-mode tests. The focused
H200 precision/preset/compile suite passed 21 tests with four optional skips.
Review head `1abfe19` passed all CI: 1,112 CPU tests (84 skipped), 25 RF3
checkpoint integration tests, mypy, Ruff and docs. These are recorded results,
not a substitute for running CI on subsequent changes.

**Rollout:** keep AW3, corrected pooling and speedups as the default, with the
legacy execution option for fallback/comparison. Describe pooling as a numerical
correction, not a proven design-quality upgrade. Broader tasks, long sequences,
large checkpoint batches and full training remain outside this validation.

## Other model checks

CoreWeave H200 public-checkpoint smoke tests completed for RFD3 eager/compiled,
RFD3NA mixed RNA/DNA/protein, ProteinMPNN, LigandMPNN, RF3 protein/ligands and
RFD3 → MPNN → RF3. Generated coordinates were finite; MPNN retained its expected
unplaced zero-occupancy terminal OXT. These are functionality checks.

The paired AW2/AW3 checkpoint audit used identical corrected pooling in both
arms. RF3 fixed-feature replay matched all 154 output tensors bitwise in 21
cases (seven fixtures, three seeds). With native preprocessing, all 21 passed
the confidence/geometry gates, but one short peptide–DNA case failed seeded
structural equivalence (3.044 Å CA RMSD, with 33 fewer severe clashes in AW3).
MPNN matched features, logits and sequences bitwise in all 36 comparisons.
These small fixtures do not establish broad folding or training-quality equivalence.

The RF3 exact LayerNorm port matched all 154 output tensors across four protein
lengths (128–512 residues) and two seeds under deterministic runtime settings.
With final dispatch, default attention and five samples, the 512-residue H200
case took 10.495 s native versus 9.094 s optimized (1.15×); 256 residues used
native kernels. The focused kernel suite passed 110 tests. See the
[RF3 kernel options](https://github.com/RosettaCommons/foundry/blob/095cdf2c5a225f9be54c4dccc73094bb35b6769e/models/rf3/README.md#exact-inference-kernels).

Historical campaign scripts and detailed reports remain in Git history at
`095cdf2`; permanent model, pooling and AtomWorks regression tests remain in
the test suites.
