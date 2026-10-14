# AtomWorks 3 port and regression results

Foundry PR #401 combines #397's batched inference, corrected pooling, and the
AtomWorks `>=3.0.1,<4` port for RFD3, RFD3NA, MPNN and RF3. CI pins AtomWorks
3.0.1; the supported Torch minimum is 2.11.

## Compatibility and tests

- Migrate parser/preparation configurations, annotations and query imports.
  Preserve RFD3/RFD3NA conditioning annotations and bonds on synthetic coordinates.
- Replace the removed polymer-bond helper with CCD attachment-site lookup while
  respecting chain, assembly, gap, motif and protein/nucleic-acid boundaries.
- Preserve RF3's training bond cutoff (strictly below 2.4 Å), inference bonds,
  and template/conformer constraints after atom completion and residue renumbering.
  Fill missing CCD charges without overwriting supplied values.
- Adapt nested dataset configurations to AtomWorks' `PandasDataset` and loader.
  Tests cover generic, PN-unit and interface metadata; full training was not run.

Golden tests compare all 41 RFD3/RFD3NA model input features with AW2.2.1 references
exactly, including token bonds. RF3/MPNN tests cover protein, DNA, glycan, ACE cap,
file/in-memory parsing, charges, templates and bond distances. In the separate
feature audit, MPNN matched 144/144 arrays; RF3 matched 154/160. The six RF3
changes were corrected reference rotations and covalent charges in AtomWorks 3.
Universal preprocessing equivalence is not claimed.

## RFD3 regression answers

### Pooling and exactness

The old CUDA `index_reduce` mean has nondeterministic addition order and
low-precision sum/count arithmetic. Identical prepared inputs and noise produced
a 4.904 Å aligned CA replay difference in the upstream 40-residue diagnostic.
Corrected eager pooling made repeats bitwise identical in that tested setting.

The fix uses stable grouping, integer counts and FP32 accumulation/division for
FP16/BF16. Padded pair pooling also disables autocast locally. Permanent tests
compare values and gradients against FP64 means of the same quantized inputs,
including overflow, empty groups, NaN padding and compiled CUDA execution.

**Acceleration is not bitwise exact:** a controlled 40-residue, 200-step audit
still differed by 2.515 Å aligned CA RMSD from its deterministic reference.
Fast and legacy also use different random streams. Numerical correctness alone
does not establish improved design quality.

### Quality measurements

After the final batched precision fix, the enzyme test used the 150-residue
1euv tutorial, 32 predetermined seeds × two outputs per mode, the public 200-step
sampler and native JSON `metrics.insertion_rmsd`, without refolding or filtering.

| Final AW3 preset | Insertion RMSD <1 Å | Mean (Å) | Maximum (Å) |
| --- | ---: | ---: | ---: |
| Fast | 62/64 | 0.485 | 2.040 |
| Legacy | 63/64 | 0.447 | 1.707 |

Fast minus legacy was +0.038 Å, paired seed-group 95% bootstrap interval
[−0.051, +0.131] Å. **No clear quality separation was measured on this fixture**;
these intervals do not establish population equivalence or a quality improvement.

Earlier evidence predates the final batched precision correction: the paired
16-backbone-per-arm pilot yielded refolding passes of 15/16 original AW2, 14/16
corrected eager AW2, 16/16 padded AW2 and 16/16 padded AW3. A separate 64-output
enzyme comparison gave 63/64, 61/64, 57/64 and 61/64 below 1 Å, respectively.
Pooling alone failed that pilot's exploratory noninferiority margin and had a
+0.050 Å enzyme mean penalty (95% interval +0.009 to +0.094 Å). These small studies
are not proof of broad equivalence or a general regression.

### Legacy and training limits

RFD3 defaults to fast; legacy restores serial eager execution, native kernels,
vanilla attention, no graphs and the original mean. It cannot guarantee old
coordinates because that mean remains nondeterministic and AW3 remains installed.
Both presets accept AW3's conformer fallback: on the enzyme fixture, AW3 retains
a ligand reference conformer after UFF failure where AW2 masked it. Restoring
only that old mask reproduced eight diagnostic AW2 outputs, but the expanded
quality results do not justify forcing it. See the
{doc}`inference guide </models/rfd3/inference_acceleration>`
for the preset contract.

Real RFD3/RF3 pooling-module tests and short optimizer trajectories pass against
the quantized-input FP64 oracle. Old and corrected BF16 training trajectories
are not equivalent; training convergence and downstream quality were not tested.
Describe pooling as a numerical correction, not a proven quality upgrade.

## RF3 and MPNN checks

The paired AW2/AW3 checkpoint audit used identical corrected pooling in both
arms. RF3 fixed-feature replay matched all 154 output tensors bitwise in 21
cases (seven fixtures, three seeds). With native preprocessing, all 21 passed
the confidence/geometry gates, but one short peptide–DNA case failed seeded
structural equivalence (3.044 Å CA RMSD, with 33 fewer severe clashes in AW3).
MPNN matched features, logits and sequences bitwise in all 36 comparisons.
These small fixtures do not establish broad folding or training-quality equivalence.

RF3's exact LayerNorm matched all 154 output tensors across four lengths
(128–512 residues) and two seeds with deterministic algorithms and a fixed
cuBLAS workspace. With default attention and five samples, the 512-residue H200
case took 10.495 s native versus 9.094 s optimized (1.15×); 256 residues used
native kernels. RF3 selects these kernels automatically and warns once per
inference fallback reason. Training and autograd use native PyTorch.

CoreWeave public-checkpoint smoke tests also completed for all four model
families and RFD3 → MPNN → RF3, with finite outputs. These are functionality
checks. Historical campaign scripts and detailed reports remain in Git history;
production regression tests remain in the test suites.
