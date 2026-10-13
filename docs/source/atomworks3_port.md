# AtomWorks 3 port validation

This port starts at public Foundry PR #397 revision
`754d1caae5a74b35a3bfcf68169e459474b7dbc4`. It supports AtomWorks
`>=3.0.1,<4`; CI pins 3.0.1. Dependency compatibility and approval of PR #397's
acceleration defaults are separate decisions.

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

The acceleration also changes the seed contract. These results do not establish
equivalent designability or actual motif preservation. Require a paired quality
benchmark against pristine production, with predetermined acceptance margins,
before adopting those speedups by default. This dependency port does not resolve
that separate merge gate. CPU/MPS checkpoint inference, Apex, long sequences,
large real-checkpoint batches, and full training runs remain outside this validation.
