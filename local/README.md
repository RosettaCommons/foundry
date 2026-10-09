# Local Foundry installation

Installed in `/Users/jbutch/Projects/foundry-latest/.venv` on an Apple M3 Mac with 16 GiB unified memory. This is an editable install of the existing checkout (base commit `673fa32`), with Python 3.12.12, PyTorch 2.14.0 and AtomWorks 2.2.1. Exact installed versions are in `requirements-macos.lock.txt`.

RFD3 and RF3 reuse the existing official checkpoints in `/Users/jbutch/.foundry/checkpoints`. ProteinMPNN uses `local/checkpoints/proteinmpnn_v_48_020.pt`, downloaded from the official IPD checkpoint registry. The repository `.env` includes this additional checkpoint directory.

## Run a model

The launcher explicitly selects CPU or MPS and uses this project's environment:

```bash
cd /Users/jbutch/Projects/foundry-latest

# Generate one 32-residue backbone with the standard 200 diffusion steps.
./local/run.sh cpu rfd3 design inputs=null '+specification.length=32' \
  out_dir=/Users/jbutch/Projects/foundry-latest/local/results/my-design \
  diffusion_batch_size=1 n_batches=1 seed=42

# Design a sequence for that backbone.
./local/run.sh cpu mpnn --model_type protein_mpnn --is_legacy_weights True \
  --structure_path /Users/jbutch/Projects/foundry-latest/local/results/my-design/_0_model_0.cif.gz \
  --out_directory /Users/jbutch/Projects/foundry-latest/local/results/my-sequences \
  --seed 42

# Predict a structure from a sequence JSON, with one sampled structure.
./local/run.sh cpu rf3 fold \
  inputs=/Users/jbutch/Projects/foundry-latest/local/inputs/rf3.json \
  out_dir=/Users/jbutch/Projects/foundry-latest/local/results/my-fold \
  diffusion_batch_size=1 seed=42
```

Replace `cpu` with `mps` to use the Apple GPU. Run MPS commands from a normal macOS terminal; Codex's restricted shell cannot see Metal and needs its approved GPU-access execution mode. CPU and MPS use float32. The launcher defaults to four CPU threads. No automatic unsupported-operator fallback is enabled by the launcher.

You can also activate `.venv` and set `FOUNDRY_DEVICE=cpu` or `FOUNDRY_DEVICE=mps` before the regular model CLIs. Without an explicit setting, Foundry retains its automatic hardware selection.

## Repeat the inference verification

```bash
cd /Users/jbutch/Projects/foundry-latest
.venv/bin/python local/verify.py --device cpu
.venv/bin/python local/verify.py --device mps
```

The verifier runs models sequentially to limit peak memory, uses 32-residue inputs and batch size 1, retains individual logs, validates outputs, and writes `local/results/<device>-full/verification.json`. Repeated runs use a timestamped subdirectory so previous outputs are retained and FASTA records cannot accumulate. RFD3 uses 200 steps; RF3 uses 50 diffusion steps and 10 recycles, with confidence early stopping disabled. ProteinMPNN designs a sequence on the generated RFD3 backbone. RF3 folds a separate fixed test sequence. Add `--quick` for a 10-step / 1-recycle execution check.

Checks require finite RFD3/RF3 coordinates, expected residue counts, a valid ProteinMPNN sequence, and finite RF3 pLDDT. ProteinMPNN's structure export can include an unplaced terminal oxygen (`OXT`) with NaN coordinates and zero occupancy. This is reported explicitly and retained; every other non-finite coordinate fails verification. ProteinMPNN designs sequences and does not build complete side chains for those new sequences.

These are local inference checks, not a validation of biological quality or a capacity test for large proteins/complexes. Keep batch size 1 on this 16 GiB machine and increase input size cautiously.

## Local compatibility changes

- `FOUNDRY_DEVICE` overrides automatic device selection for RFD3, RF3 and ProteinMPNN.
- Explicit CPU/MPS selection uses one device and float32.
- RF3 attention no longer forces bfloat16 activations on CPU when its linear weights are float32.
- Device-selection and float32 attention regression tests are included.

Model weights, outputs and logs are excluded from Git by `local/.gitignore`.

## Verified results (2026-09-29)

| Model | CPU float32 | MPS float32 | Test |
|---|---|---|---|
| RFD3 | Passed | Passed | 32 residues, 200 diffusion steps, one design |
| ProteinMPNN | Passed | Passed | One 32-residue sequence on each RFD3 backbone |
| RF3 | Passed | Passed | 32 residues, 50 steps, 10 recycles, one sample |

Both RFD3 and RF3 produced finite coordinates with the expected residue counts. ProteinMPNN produced valid sequences and finite backbone coordinates, with the terminal OXT placeholder documented above. RF3 mean pLDDT was CPU 0.8348, MPS 0.8349. These scores are for this test input only.

Reports: [CPU verification](results/cpu-full/verification.json), [MPS verification](results/mps-full/verification.json). Each report points to the model logs and output files. The MPS run completed without enabling PyTorch's automatic unsupported-operator CPU fallback; the models retain their existing CPU preprocessing and compatibility helpers.

Validation also passed 15 targeted unit/regression tests, Ruff checks, and `uv pip check`.
