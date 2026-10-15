# Installing RFOptimization dependencies

RFO orchestrates three separate model processes. Use separate Python
environments for Foundry/RF3, Boltz, and upstream MPNN so their dependency
versions can differ. No containers, weights, or Baker Lab filesystem paths
are supplied or assumed.

## 1. Foundry and the driver

From the root of this Foundry checkout, with Python 3.12 or newer:

```bash
python -m pip install -e '.[rf3,rfo]'
foundry install rf3
export RF3_CKPT="$HOME/.foundry/checkpoints/rf3_foundry_01_24_latest_remapped.ckpt"
```

The Foundry/RF3 environment uses AtomWorks 3 (`atomworks[ml]>=3.0.1,<4`).
Keep Boltz and upstream MPNN in their separate environments; their NumPy
requirements need not match Foundry's.

RFO requires a confidence-enabled RF3 checkpoint with its `train_cfg`.
Use `RF3_CKPT` to select a checkpoint explicitly. The command above uses
Foundry's public registry; historical RFO research configurations used
other checkpoints, so matching their results requires the corresponding
checkpoint and experiment settings. Full RF3 inference needs the AtomWorks
resources described in the [RF3 README](../../rf3/README.md).

When the driver runs in the RF3 environment, no `RF3_PYTHON` is needed.
Otherwise set it to that environment's absolute Python path:

```bash
export RF3_PYTHON=/path/to/foundry-env/bin/python
```

To run only the lightweight source driver in a different environment:

```bash
python -m pip install numpy biopython pyyaml
cd models/rfo
./rfo --help
```

Set `RFO_PYTHON` if the source launcher should use a specific Python binary.
The RF3 branch uses an RFO-specific differentiable forward pass: gradients
reach the sequence through the RF3 trunk and prediction heads, while sampled
diffusion coordinates are held constant. Standard Foundry RF3 inference is
unchanged. If the installed cuEquivariance triangle kernel fails in the
gradient path, use Foundry's supported PyTorch implementation:

```bash
export DISABLE_CUEQUIVARIANCE=1
```

RFO overrides the checkpoint's training residue cache with `None`, and disables
geometry-based cyclic-bond detection: sequence-derived placeholder coordinates
must not be interpreted as physical terminal closure bonds. This does not add
RF3 cyclic-peptide support; `cyclic` still affects only the Boltz branch.

Template selection accepts whole chains (`B`), compact residue ranges
(`B1103-1120`), and AtomWorks path queries (`B/*/[1103-1120]/CA`).

## 2. Install upstream MPNN yourself

Clone [dauparas/LigandMPNN](https://github.com/dauparas/LigandMPNN). This
repository includes both ProteinMPNN and LigandMPNN inference and checkpoints;
RFO expects its `run.py` interface. The separate ProteinMPNN repository's
`protein_mpnn_run.py` interface is not compatible with this adapter.

For example, on a Linux CUDA machine, following the upstream environment setup:

```bash
git clone https://github.com/dauparas/LigandMPNN.git /path/to/LigandMPNN
cd /path/to/LigandMPNN
conda create -n rfo-mpnn python=3.11 -y
conda activate rfo-mpnn
python -m pip install -r requirements.txt
bash get_model_params.sh ./model_params

export MPNN_REPO="$(pwd)"
export MPNN_PYTHON="$(command -v python)"
export MPNN_MODEL_TYPE=protein_mpnn
"$MPNN_PYTHON" "$MPNN_REPO/run.py" --help
```

Return to the driver environment, retaining these exports. RFO invokes this
Python interpreter directly; it never imports Foundry's `mpnn` package.
The upstream requirements include specific CUDA/PyTorch versions; follow
upstream guidance if your machine needs a different CUDA build.

| Setting | Meaning |
| --- | --- |
| `MPNN_REPO` | Absolute path to the upstream checkout containing `run.py`; required. |
| `MPNN_PYTHON` | Its environment's Python executable; defaults to the driver's Python. |
| `MPNN_MODEL_TYPE` | `protein_mpnn` (default) or `ligand_mpnn`. |
| `MPNN_CKPT` | Optional checkpoint override; must match the chosen model type. |

Default checkpoints under `$MPNN_REPO/model_params/` are
`proteinmpnn_v_48_020.pt` and `ligandmpnn_v_32_010_25.pt`, respectively.
To use ligand-aware sequence design:

```bash
export MPNN_MODEL_TYPE=ligand_mpnn
# Leave MPNN_CKPT unset to use the matching default weights.
```

Do not pass converted Foundry MPNN weights: this adapter expects upstream
weights. Old `MPNN_LEGACY_WEIGHTS` and `MPNN_CONTAINER_PYTHONPATH` settings
are not used. MPNN is required for both the RF3 and Boltz cycling branches.

RFO designs only unfixed residues in chain A and maps positional masks onto
actual PDB residue numbers and insertion codes. Standard amino acids and
complete N/CA/C/O backbone atoms are required in the designed chain. The
adapter preserves other chains and ligands from the predicted complex,
removes obsolete side-chain atoms at mutations, and selects the first sample.
When all residues are fixed, it copies the complex without calling MPNN.

## 3. Install Boltz-2

Follow the [Boltz installation instructions](https://github.com/jwohlwend/boltz#installation)
in a separate environment, for example:

```bash
conda create -n rfo-boltz python=3.12 -y
conda activate rfo-boltz
python -m pip install 'boltz[cuda]'
export BOLTZ_EXECUTABLE="$(command -v boltz)"
export BOLTZ_CACHE="$HOME/.cache/boltz"
"$BOLTZ_EXECUTABLE" predict --help
```

Return to the driver environment with these exports retained. Boltz downloads
its public model assets into its cache when needed. For offline execution,
prepare that cache and set `BOLTZ_CKPT` to your Boltz-2 checkpoint. RFO supplies
`--model boltz2`, `--no_kernels`, and the requested diffusion sample count.

The [Boltz prediction guide](https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md)
describes its input/MSA formats. RFO supplies `msa: empty` when no MSA is
provided, and does not send sequences to an MSA server. Set `msa_paths` in the
RFO YAML for target MSAs. Boltz is not needed with `backprop_fraction: 1.0`;
RF3 is not needed with `backprop_fraction: 0.0`.

## Optional Apptainer execution

Each model can independently run in a user-provided image by setting
`RF3_SIF`, `BOLTZ_SIF`, or `MPNN_SIF`. A model with no image configured runs
in its selected local environment. Install each model's dependencies inside
its image; there is no required shared image or `/net/software/lab/rfo` store.

```bash
export RF3_SIF=/path/to/foundry.sif
export RF3_PYTHON=python
export BOLTZ_SIF=/path/to/boltz.sif
export BOLTZ_EXECUTABLE=boltz
export MPNN_SIF=/path/to/ligandmpnn.sif
export MPNN_PYTHON=python
# MPNN_REPO still points to the host upstream checkout; RFO binds it in.
```

RFO supplies `apptainer exec --nv` and binds the source, input, output,
checkpoint and MSA paths it uses. All bound paths retain their host names.
Add comma-separated paths with `RF3_BIND_PATHS`, `BOLTZ_BIND_PATHS` or
`MPNN_BIND_PATHS` for additional resources. Bind paths must not contain commas.
RF3 forwards `RF3_CKPT`, `CCD_MIRROR_PATH`, `PDB_MIRROR_PATH`, `X3DNA`, and
`DSSP` when set. Configure these to your own resources if required by your
RF3 checkpoint's transform pipeline.

## Run and verify

From `models/rfo`, after exporting the settings above:

```bash
./rfo --config src/rfo/cycling/configs/example.yaml --total-cycles 1
```

Use `--backprop-fraction 0` and `--backprop-fraction 1` in separate output
directories to exercise each branch. The default example needs both models.
A completed run writes a records JSON and the redesigned complex under
`inputs/`. Missing MPNN installations/checkpoints fail before model inference;
failed or incomplete runs return a nonzero exit status.

The portable adapter, CLI and gradient tests can be run from the repository root:

```bash
PYTHONPATH=models/rfo/src python -m pytest models/rfo/tests
```

These tests use a stand-in external CLI and small differentiable tensors;
they do not replace a GPU smoke test with real model weights.
