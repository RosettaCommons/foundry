# Enzyme insertion benchmark: 64 outputs per configuration

Expansion of the [eight-output pilot](../README.md), requested after the user
clarified that the native output JSON metric alone is the desired check.

## Frozen protocol

- Same bundled 1euv cysteine-hydrolase tutorial, length 150, six unindexed motif
  residues, ligand and exposure constraints, public 200-step RFD3 sampler.
- Four configurations: original pooling/AW2, fixed pooling/AW2,
  fixed pooling + #397/AW2, and fixed pooling + #397/AW3.
- **64 outputs per configuration, 256 total:** 32 predetermined seeds (11 through
  352 in increments of 11), two outputs per seed. Seeds 11/22/33/44 retain all
  eight prior outputs unchanged; the remaining 28 seeds add 56 new outputs.
- Native JSON **`metrics.insertion_rmsd <1 Å`**. No refolding or supplementary
  tip-atom metric is used to decide success. This metric averages guidepost
  residue RMSDs before coordinate replacement.
- Report all 64 and the new 56 separately. No score-based exclusions or retries.
- BF16, H200, identical public checkpoint, cuEquivariance disabled. Legacy arms
  are eager; #397 arms use compiled auto kernels. Model source is unchanged from
  the original pilot; dependency versions and checkpoint hash are in
  [runtime.json](runtime.json).
- Seeds/task are matched across configurations. The #397 random-stream contract
  differs from legacy, so legacy-versus-#397 contrasts do not share exact noise.

The expansion plan and seed list were written before submission of array
**664743**. [Retained hashes](retained_manifest.json) verify that none of the
initial outputs changed. This is an expanded descriptive experiment after
seeing the first eight outcomes, not a newly prespecified confirmatory test.

## Results

All **256/256** outputs are present and finite, all retained artifact hashes match,
and all four array tasks completed with exit 0. Each configuration has 64 outputs.
Scores below are native output JSON insertion RMSDs in Å.

| Configuration | <1 Å | Mean | Median | 90th percentile | Maximum | New outputs <1 Å |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original pooling, AW2 | 63/64 | 0.435 | 0.381 | 0.604 | 1.337 | 55/56 |
| Fixed pooling, AW2 | 61/64 | 0.485 | 0.422 | 0.604 | 1.911 | 54/56 |
| Fixed + #397, AW2 | 57/64 | 0.532 | 0.430 | 1.003 | 2.182 | 50/56 |
| Fixed + #397, AW3 | 61/64 | 0.478 | 0.407 | 0.759 | 1.237 | 54/56 |

### Paired differences and interpretation

Intervals below are descriptive 95% seed-group bootstrap intervals, without
multiple-comparison adjustment. Negative RMSD differences favor the newer
configuration; positive pass-rate differences favor it.

| Contrast | Mean RMSD difference (Å), 95% interval | Sub-angstrom pass-rate difference (percentage points), 95% interval |
| --- | ---: | ---: |
| Fixed pooling − original | +0.050 [+0.009, +0.094] | -3.12 [-7.81, +0.00] |
| Fixed + #397 − fixed eager | +0.047 [-0.059, +0.163] | -6.25 [-17.19, +3.12] |
| AW3 − AW2, both fixed + #397 | -0.054 [-0.164, +0.042] | +6.25 [-3.12, +15.62] |

**Pooling:** 61/64 versus 63/64 pass, with a small +0.050 Å mean insertion
penalty. The mean-difference interval is above zero on this fixture. There is
no quality-improvement claim; these results are consistent with a modest effect,
not a drastic collapse. The numerical-correctness motivation for the pooling
fix remains separate from design quality.

**AtomWorks 3:** 61/64 versus 57/64 pass for matched #397 configurations;
mean RMSD improves by 0.054 Å in this sample. Both difference intervals include
zero. The apparent AW3 disadvantage in the initial eight outputs did not persist
in the expanded sample. This supports comparable insertion quality on this
fixture, without establishing population-wide equivalence or superiority.

**Decision supported by this benchmark:** do not treat the original eight-output
quality difference as evidence to reject the port, and do not advertise the
pooling fix as a design-quality improvement. The previously identified ligand
reference-mask policy change still explains why AW3 trajectories can differ;
this larger quality sample does not make those inputs identical. Its rollout
policy should be explicit, but these results alone do not justify requiring the
old failure mask on quality grounds.

The new-only 56-output results are shown above to keep the extension distinct
from the pilot that motivated it. All outliers remain in the results; no further
sampling was triggered by these outcomes. No refolding was run.

- [Every output and native JSON path](evidence/scores.csv)
- [Summary, intervals, and all outliers](evidence/summary.json)
- [Runtime provenance for the additional 224 outputs](evidence/new_provenance.json)
- [Slurm accounting](evidence/scheduler.txt)

## Analysis

`analyze.py` reads each native JSON directly, requires all planned outputs and
completion markers, rejects missing/nonfinite scores, and verifies all retained
artifact hashes. It writes per-output CSV and JSON summaries under `evidence/`.

Report mean, median, 90th percentile, maximum, and number below 1 Å. The 95%
percentile bootstrap uses 20,000 draws with fixed RNG seed 20261013 and resamples
whole seed groups (both outputs together). Paired differences resample the same
seed indices across configurations. New-only summaries use their 28 seed groups.
No equivalence/noninferiority margin was specified: a confidence interval
containing zero does not prove equivalence. One enzyme fixture still cannot
establish general enzyme-design success or experimental activity.

## Visualizations

`render.py` uses actual PyMOL geometry from the pre-insertion structures. Cyan is
the scaffold, magenta the generated constrained tip atoms, green the target
motif, and gray the ligand; heteroatoms retain element colors. Display alignment
uses the ligand only and does not alter JSON scores. The PNGs carry native JSON
RMSD labels.

- `results/median_overviews.png`: one median-ranked design (rank 33/64) per arm.
- `results/median_active_sites.png`: close-ups of those same designs.
- `results/worst_active_sites.png`: the largest native insertion RMSD per arm.
- `results/<arm>_all64.png`: all 64 designs in seed/sample order, with scores
  ≥1 Å labeled red.
- `results/enzyme64_comparison.pse`: all 256 designs and their target motifs,
  plus input structure, with median and worst comparison scenes.

Four-panel order is original AW2, fixed AW2, fixed+#397 AW2, fixed+#397 AW3,
reading left-to-right and top-to-bottom. The session opens on median overviews.
Only PNG figures are generated. Raw structures, PNGs, and the session stay local
under the ignored `results/` directory; their filenames and selections are
recorded in the image manifest.

## Reproduction

`launch.sh` records site paths and Slurm settings. After the array completes, run
`analyze.py` in the Foundry environment and `render.py` in the isolated PyMOL
Python environment (PyMOL 3.2.0a0, Pillow 12.3.0). The launcher adds only the new
seeds; retained artifacts are required for the full 64-output analysis.
