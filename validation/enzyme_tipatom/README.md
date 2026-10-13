# Basic enzyme tip-atom insertion benchmark

This extends the unconditional [design/refolding pilot](../quality/report.md)
with Foundry's basic **1euv cysteine-hydrolase tutorial**. This run ends at RFD3;
there is no ProteinMPNN sequence redesign or refolding. The sequence labels in
the structures come from RFD3's sequence head and motif insertion.

## Protocol

- Bundled input: `models/rfd3/docs/tutorials/enzyme_tutorial_files/1euv_lig.pdb`.
- Original tutorial specification, with length fixed at 150 instead of 100–200.
- Six unindexed residues: A514, A531, A574, A579, A580, A581; ligand `l:g`.
- Four seeds (11/22/33/44), two diffusion samples per seed, eight structures per
  arm and **32 primary structures total**. One separate batch-one preflight is
  excluded. The full job completed with exit 0 in 7m39s on a CoreWeave H200.
- Public inference engine, 200-step sampler defaults, BF16; compiled arms use
  default `auto` inference kernels. The compiled paths were exercised without
  legacy-fallback messages. All arms disable cuEquivariance. The two legacy arms
  are eager; the two #397 arms are compiled. Input feature workers are zero.
- Identical task and seed labels; #397's random-stream contract differs from
  legacy. This is not a matched-Gaussian-noise comparison between those paths.
- One fixed pooling implementation is shared by all three corrected arms.

The original-pooling smoke submission first failed before inference because the
legacy public engine requires a string filename rather than `Path`. The harness
was corrected; the subsequent smoke and all primary designs completed. There
were no omitted primary designs or score-based recovery runs.

## What is measured

**Score before cleanup.** RFD3 normally overwrites matched motif coordinates with
the fixed guideposts. Measuring those cleaned atoms alone would hide placement
errors. A read-only hook captures the generated positions before that overwrite
and uses RFD3's own assigned residue map. Internal atom14 slot labels are mapped
back to `gt_atom_name` chemical identities.

The independent tip score pools the squared distances of these **15 atoms** in
their shared model frame; it applies no tip fitting and drops no atoms:

| Input residue | Tip atoms |
| --- | --- |
| HIS A514 | CG, ND1, CD2, CE1, NE2 |
| ASP A531 | CB, CG, OD1, OD2 |
| GLN A574 | CG, CD, OE1, NE2 |
| CYS A580 | CB, SG |

A579/A581 have backbone-only constraints and therefore contribute to the native
insertion metric, but not the independent tip subset. The native `insertion_rmsd`
is the mean of six residue-level guidepost RMSDs; it is not the atom-weighted
15-tip RMSD. The independently captured native metric matches every output JSON.

After the preflight suggested an ASP oxygen-label exchange, a secondary score
was specified before inspecting the full-arm outcomes. It permits **only ASP
OD1/OD2 exchange**, without fitting or dropping atoms. GLN O/N identities remain
distinct. Strict scores stay primary. The 0.5 Å cutoff is a descriptive
insertion-quality reference, not a population-equivalence test.

## Primary results

All values below are mean RMSDs in Å across all eight designs per arm.

| Arm | Native insertion | Strict 15-tip | ASP-symmetry-aware tip | Strict tip <0.5 Å |
| --- | ---: | ---: | ---: | ---: |
| Original pooling, AW2 | 0.445 | 0.640 | 0.592 | 4/8 |
| Corrected pooling, AW2 | 0.579 | 0.938 | 0.872 | 2/8 |
| Corrected pooling + #397, AW2 | 0.522 | 0.710 | 0.682 | 3/8 |
| Corrected pooling + #397, AW3 | 0.692 | 1.192 | 1.170 | 2/8 |

These measurements do not show a pooling quality improvement. The corrected
eager arm's worst sample (seed33/sample1) has 2.652 Å pooled tip RMSD, dominated by
HIS A514 at 4.320 Å. The primary AW3 arm has three GLN A574 placement outliers
(3.675–4.224 Å residue tip RMSD). Thus the higher AW3 error is not explained away
by equivalent ASP oxygen names. There are only four seed groups per arm and one
enzyme task; this is not a general estimate of enzyme-design success or activity.

## Identified AtomWorks 3 semantic change

Replaying preprocessing with the public padded loader's seed rules finds exactly
one changed model feature in every seed: **`ref_mask` for all 22 ligand atoms**.
It is false in AW2 and true in AW3. All other model features, starting coordinates,
atom annotations and example IDs are identical. Raw bond graphs differ, as noted
in the broader port, but no additional model feature differs here.

AtomWorks 3 catches an RDKit UFF optimization `RuntimeError` and retains an
unoptimized conformer. AtomWorks 2 propagated that failure to conformer handling,
which marked the reference unavailable. RFD3's current configuration does not
expose the generated reference positions, so the mask alone reaches the model
as a changed feature. This is an upstream fallback-policy change, not evidence
that the pooling operation itself caused the AW2→AW3 difference.

The additional `diagnostic_aw3_mask` arm restores only the archived AW2 reference
mask before model execution and asserts that all other features already match.
It is a diagnostic intervention, not a change to production code or a replacement
for the primary AW3 measurements.

**Diagnostic result:** all eight designs reproduce the corrected #397/AW2 arm
exactly: atom identities, PDB coordinates (maximum difference zero),
full-precision pre-insertion motif coordinates, and both RMSD metrics. Mean native
insertion RMSD returns to **0.522 Å** and strict tip RMSD to **0.710 Å**. Every
runtime assertion confirmed that `ref_mask` was the sole changed input feature.
The diagnostic job completed successfully in a separate short allocation.
See [all eight comparisons](evidence/mask_diagnostic.json).

This causally isolates the migration difference for this fixture. Before
promoting the port, explicitly decide how current checkpoints should handle
conformer optimization failure. Preserving the historical failure mask is the
conservative compatibility choice for this checkpoint; silently changing the
feature or hard-coding a mask for this ligand is not an adequate production fix.
No production conformer policy was changed by this benchmark. An explicit,
tested failure-policy interface remains needed for a compatibility-preserving
rollout. The experiment does not establish whether a model trained with the newer
fallback policy would perform better.

## PyMOL deliverables

`results/enzyme_tipatom_comparison.pse` contains all 32 primary designs plus the
eight mask-restoration diagnostic designs. Scenes:
`representative_overviews`, `representative_active_sites`, and
`worst_tip_insertion`. The four-panel images use seed11/sample0 from each arm,
chosen before scoring; the worst scene is the maximum strict tip error across
all primary arms.

Panel order: top-left original AW2; top-right corrected AW2; bottom-left corrected
#397/AW2; bottom-right corrected #397/AW3. Cyan is scaffold, magenta generated tips,
green target motif, gray ligand; heteroatoms retain element colors. Small spheres
keep unbonded misplaced atoms visible. Structures are aligned on the ligand for
display only. Scores use original float coordinates, not rounded PDBs or PyMOL
fitting. The images show **pre-insertion** positions, not artificially corrected
motif coordinates. Each case also retains its usual cleaned CIF/PDB.

PNG files: `results/overview.png`, `results/active_sites.png`, and
`results/worst_insertion.png`. No PDF or SVG files were generated.

## Evidence and reproduction

- [All primary scores](evidence/scores.csv)
- [Summary and seed-group differences](evidence/summary.json)
- [Coverage](evidence/coverage.json)
- [Source, pooling hashes, and runtime settings](evidence/provenance.json)
- [Input feature comparison](evidence/input_comparison.json)
- [Scheduler accounting](evidence/scheduler.txt)

`run.py` accepts explicit source/environment selection through `PYTHONPATH`, plus
`--input`, `--root`, `--arm`, `--checkpoint`, optional `--pooling-source`, and
`--seeds`/`--batch-size`. `launch.sh` records the CoreWeave paths/resources used;
edit them for another cluster. Use `summarize.py` after all four arms finish and
`render.py` with a Python environment containing `pymol-open-source` and NumPy.
PyMOL version used here is 3.2.0a0. Raw structures and the PSE remain local under
this directory's `results/`; they are excluded from Git.

Four scorer tests check atom-slot/chemical-name correspondence, preservation of
unfitted insertion displacement, missing/nonfinite atom rejection, and the exact
scope of the ASP symmetry rule. Pinned Ruff check/format pass.
