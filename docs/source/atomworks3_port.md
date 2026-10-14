# AtomWorks 3 port validation

This port starts at public Foundry PR #397 revision
`754d1caae5a74b35a3bfcf68169e459474b7dbc4`. It supports AtomWorks
`>=3.0.1,<4`; CI pins 3.0.1. Dependency compatibility and approval of PR #397's
acceleration defaults are separate decisions.

## Consolidated rollout policy

[PR #401](https://github.com/RosettaCommons/foundry/pull/401) consolidates the
speedups, pooling fixes, and dependency port. RFD3 defaults to `inference_mode=fast`
and offers `inference_mode=legacy` for the old execution choices. See the
[precise compatibility contract](models/rfd3/inference_acceleration.md).
Both modes use AW3 preprocessing, including its upstream conformer-failure
fallback. This is an explicit acceptance of the newer dependency policy; the
execution dial does not emulate AW2 inputs. The original eight-output diagnostic
below remains historical evidence, not a requirement to force the old ligand
mask. Historical quality measurements also predate the final batched atom/pair
precision correction; the [new mode validation](../../validation/inference_modes/README.md)
reports runs after that correction.

The latest production example tests also exposed a bundled PDB with ligand
residues ordered 900 then 850. AW3 checks per-chain residue order. The bundled
M0255 fixture now places 850 first, preserving every atom record, identifier and
coordinate. Unsorted user inputs require normalization before parsing; no global
parser-validation bypass was introduced.

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

## Regression evidence

The complete portable suite passes: **1,051 passed, 77 skipped**. Mypy passes
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

## CoreWeave checkpoint validation, 2026-10-13

Clean Python 3.12 environment, AtomWorks 3.0.1, Biotite 1.6.0,
Torch 2.11.0+cu128, RDKit 2026.9.1. One H200, NVIDIA driver 595.71.05,
Slurm interactive allocation with `--gres=gpu:1`, 8 CPUs, 64 GB host memory.
The allocation was released after validation. Apex was not installed.

| Run | Configuration and result |
|---|---|
| RFD3 eager | Public checkpoint, 32 residues, 200 steps, one valid output |
| RFD3 compiled | Default compiled/prefetched path, two batches of one, two valid outputs |
| RFD3NA | Public checkpoint, 8 RNA + 8 DNA + 16 protein residues, 200 steps, three output chains |
| ProteinMPNN | Public legacy weights, two distinct 32-residue sequences from the eager RFD3 backbone |
| LigandMPNN | Public legacy weights, two outputs retaining the four-chain ligand fixture |
| RF3 protein/ligands | Public checkpoint, 50 steps, 10 recycles; pLDDT 0.7379, pTM 0.8176, no reported clash |
| RFD3 → ProteinMPNN → RF3 | Both designed sequences predicted; pLDDT 0.8213 / 0.8183, no reported clash |

All generated RFD3, RFD3NA, and RF3 coordinates are finite and have the expected
residue/chain counts. MPNN exports retain one unplaced terminal OXT with zero
occupancy per output; every other coordinate is finite. RF3 confidence values
are finite and within their expected range. These are functional checks, not a
benchmark of designability, binding, or scientific equivalence.

The checkpoint configuration references an optional IPD residue cache that is
absent on CoreWeave. Its normal fallback completed successfully. AtomWorks also
warns when writing a plain AtomArray using bundled CCD definitions; these runs
use standard public CCD components, not custom local residue definitions.

### Reproduce the checkpoint runs

Install this branch, using the CUDA wheel appropriate to the cluster. The tested
environment used a separate Torch installation from the cu128 index, followed by
`uv pip install -e '.[all]' -c ci-constraints.txt`. Download public model aliases
`rfd3`, `rfd3na`, `rf3`, `proteinmpnn`, and `ligandmpnn` with `foundry install`;
set the five path variables below to the downloaded files. No checkpoint is
committed to the repository.

From the repository root, in a GPU allocation:

```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p validation-aw3
echo '{"protein":{"length":"32"}}' > validation-aw3/rfd3.json
echo '{"multipolymer":{"contig":"8R,/0,8D,/0,16","length":"32","input":"models/rfd3na/docs/input_pdbs/AMP.pdb"}}' > validation-aw3/rfd3na.json

rfd3 design inputs=validation-aw3/rfd3.json out_dir=validation-aw3/rfd3-eager \
  ckpt_path="$RFD3_CHECKPOINT" diffusion_batch_size=1 n_batches=1 \
  compile_model=false inference_num_workers=0 skip_existing=false seed=123
rfd3 design inputs=validation-aw3/rfd3.json out_dir=validation-aw3/rfd3-compiled \
  ckpt_path="$RFD3_CHECKPOINT" diffusion_batch_size=1 n_batches=2 \
  skip_existing=false seed=123
rfd3na design inputs=validation-aw3/rfd3na.json out_dir=validation-aw3/rfd3na \
  ckpt_path="$RFD3NA_CHECKPOINT" diffusion_batch_size=1 n_batches=1 \
  read_sequence_from_sequence_head=false skip_existing=false seed=123
mpnn --model_type protein_mpnn --checkpoint_path "$PROTEINMPNN_CHECKPOINT" \
  --is_legacy_weights True --batch_size 2 --seed 123 \
  --structure_path validation-aw3/rfd3-eager/rfd3_protein_0_model_0.cif.gz \
  --out_directory validation-aw3/mpnn-protein
mpnn --model_type ligand_mpnn --checkpoint_path "$LIGANDMPNN_CHECKPOINT" \
  --is_legacy_weights True --batch_size 2 --seed 123 \
  --structure_path models/rf3/tests/data/glke_with_ligands_from_cif.cif \
  --out_directory validation-aw3/mpnn-ligand
rf3 fold inputs=models/rf3/tests/data/glke_with_ligands.json \
  out_dir=validation-aw3/rf3-ligands ckpt_path="$RF3_CHECKPOINT" \
  diffusion_batch_size=1 early_stopping_plddt_threshold=0 seed=123
```

For the chained run, put each ProteinMPNN FASTA sequence in a separate RF3 JSON
example (`components: [{chain_id: A, seq: ..., chain_type: polypeptide(l)}]`),
then use the same RF3 command with that input file. Do not concatenate siblings.

## Precision and remaining limits

The [independent precision audit](https://github.com/Ubiquinone-dot/foundry/pull/1)
of PR #397 used AtomWorks 2.2.1, without this port. Its 58 tests passed, with four
optional/platform skips. It found
no tested padding, mask, example-mixing, or stale-graph defect. It also found that
the accelerated path is **not numerically exact**: in one real 40-residue,
200-step case, compiled output differed by 2.515 Å aligned CA RMSD from a
disclosed deterministic diagnostic reference, with 37.5% sequence-head agreement.
Pristine upstream itself was nondeterministic (4.904 Å CA replay RMSD); replacing
only its nondeterministic pooling for diagnosis made upstream and PR legacy
inference bitwise identical. The diagnostic intervention is not pristine behavior.

The acceleration also changes the seed contract. The upstream
[pooling prerequisite](https://github.com/RosettaCommons/foundry/pull/400),
including its regression tests and training diagnostics, is now included in this
branch as separate commits. It makes CPU/CUDA pooling deterministic and improves
low-precision arithmetic; preservation of scientific quality is a separate question.

The completed [paired design/refolding pilot](../../validation/quality/report.md)
generated 64 backbones and refolded 128 designed sequences. Successful backbones
were 15/16 for pristine production, 14/16 for corrected production pooling, and
16/16 for each corrected accelerated arm (AtomWorks 2 and 3). The pooling contrast
did not meet the preset exploratory noninferiority margins; it does not establish
better design quality. Keep the pooling prerequisite separate and draft pending
broader production-sampler validation. The acceleration and dependency contrasts
passed the pilot's exploratory gates, with identical generated CA arrays and
designed sequences across the two AtomWorks versions. This small unconditional,
single-sample Torch-backend study does not validate default RNG behavior,
multi-example batching, Triton kernels, motifs, or nucleic-acid design quality.

The separate [RF3/MPNN checkpoint audit](atomworks3_model_quality.md) found all
36 MPNN paired runs and all 21 RF3 fixed-feature replays bitwise identical with
matched backends. Native RF3 preprocessing passed the bounded quality criteria,
but one short peptide–DNA stress case failed seeded structural equivalence while
improving geometry. These results support the dependency migration on the tested
inputs; they do not remove the pooling rollout decision or prove broad scientific
equivalence. CPU/MPS checkpoint inference, Apex, long sequences, large
real-checkpoint batches, and full training runs remain outside this validation.

### Basic enzyme insertion follow-up

The [1euv enzyme benchmark](../../validation/enzyme_tipatom/README.md) adds 32
150-residue designs with unindexed catalytic atoms, public inference sampling,
two diffusion samples per batch, and compiled auto kernels. The requested score
is `metrics.insertion_rmsd` read directly from the native output JSON, with <1 Å
as the criterion; no refolding is involved. Mean native insertion RMSD is 0.445 Å for original pooling, 0.579 Å
for corrected eager pooling, 0.522 Å for corrected #397/AW2, and 0.692 Å for
corrected #397/AW3. Sub-angstrom counts are respectively 8/8, 7/8, 7/8, and
7/8. These small samples demonstrate neither a pooling quality gain nor a drastic
general regression. The earlier independent tip score is supplementary.

This conditional input exposes a previously uncovered dependency change:
AtomWorks 3 retains an unoptimized ligand conformer after UFF optimization raises
an error, whereas AtomWorks 2 masks the failed conformer. All 22 ligand reference
mask entries change; all other model features and starting coordinates match.
Restoring only the old mask in a diagnostic reproduces all eight AW2 designs,
atom identities, saved coordinates, and full-precision pre-insertion motif
coordinates exactly. Thus the earlier unconditional equivalence does not extend
to this failure path. An explicit, tested conformer-failure policy is required
before promoting a compatibility-preserving port; the diagnostic does not change
the production implementation. The report retains primary results, symmetry-aware
secondary scores, the mask intervention, and PyMOL reproduction instructions.

### Expanded enzyme insertion benchmark (64 outputs per configuration)

The [expanded benchmark](../../validation/enzyme_tipatom/expanded64/README.md)
retains the original eight outputs and adds 56 predetermined outputs per arm:
256/256 total, all jobs successful. Native JSON insertion RMSD <1 Å counts are
63/64 original AW2, 61/64 fixed eager AW2, 57/64 fixed+#397 AW2, and 61/64
fixed+#397 AW3. Corresponding means are 0.435, 0.485, 0.532, and 0.478 Å.

The pooling fix has a small +0.050 Å mean penalty on this fixture (95% paired
seed-group bootstrap interval +0.009 to +0.094 Å), with two fewer successes.
It corrects arithmetic but does not improve measured design quality. The AW3
port's initial eight-output disadvantage does not persist: compared with matched
#397/AW2, it has four more successes and a 0.054 Å lower mean, though both
contrast intervals include zero. This supports comparable insertion quality on
this fixture, without a population-equivalence or superiority claim.

These data do not support blocking the port because of the original quality
estimate. The conformer-failure policy still needs an explicit rollout decision;
the larger sample does not justify requiring the historical mask on quality
grounds. All 256 structures are included in the saved PyMOL session, with median,
worst-case, and all-design PNG views. No refolding was needed.
