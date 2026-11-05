# RFOptimization: Guiding Design Optimization with All-Atom Structure Prediction

<div align="center">
  <img src="docs/assets/overview.png" alt="RFOptimization across protein binders, cyclic peptides, ligand binders, and enzyme designs" width="900">
</div>

RFOptimization (RFO) improves an initial biomolecular design through repeated
sequence optimization and structure prediction. Each cycle selects RF3
gradient-guided sequence optimization with probability `backprop_fraction`,
or Boltz-2 structure prediction otherwise. MPNN then redesigns the unfixed
residues of chain A, and the redesigned complex seeds the next cycle.

## Installation

Follow the [installation guide](docs/installation.md) before running RFO.
It covers Foundry/RF3, Boltz, checkpoints, and optional Apptainer execution.

**Install MPNN separately.** RFO uses the upstream
[LigandMPNN repository](https://github.com/dauparas/LigandMPNN), which provides
both ProteinMPNN and LigandMPNN. Configure `MPNN_REPO` and `MPNN_PYTHON` to
point to that checkout and its Python environment. RFO does not use Foundry's
bundled `mpnn` package or command.

For the driver and RF3 in one environment, install from this Foundry checkout:

```bash
python -m pip install --upgrade pip   # pip 26.2.2 or newer
python -m pip install -e '.[rf3,rfo]'
```

A lightweight host driver can instead use the `models/rfo/rfo` launcher with
`numpy`, `biopython`, and `pyyaml`, and launch the models in separate
environments or Apptainer images. GPU inference is intended for Linux with
an NVIDIA GPU; the portable tests do not require model weights.

## Run optimization

After configuring the external tools as described in the installation guide:

```bash
cd models/rfo
./rfo --config src/rfo/cycling/configs/example.yaml
```

The installed `rfo` command provides the same interface:

```bash
rfo --input /path/to/start.cif --out-dir outputs/run1 \
    --backprop-fraction 0.5 --total-cycles 10
```

Chain **A** is the designed protein. Other chains are retained as partners;
the RF3 branch uses contacts with chain B to select fixed residues for MPNN.
In the Boltz branch, contiguous high-confidence regions are fixed instead.
See [cycling details](src/rfo/cycling/README.md) and the
[standalone RF3 optimizer](src/rfo/backprop/README.md).

## Configuration

```yaml
input_structure: start.cif
out_dir: outputs/run1
backprop_fraction: 0.5  # 1 = RF3 only; 0 = Boltz only; both still use MPNN
total_cycles: 10

# Optional conditioning; paths are relative to the working directory.
# loss: pae_interface_mean
# msa_paths: {B: /path/to/target.a3m}
# templates: {B: /path/to/target.cif}  # RF3 branch only

mpnn_num_seqs: 1
mpnn_temperature: 0.1
seed: 42

# Same settings can be exported as environment variables.
assets:
  mpnn_repo: /path/to/LigandMPNN
  mpnn_python: /path/to/ligandmpnn-env/bin/python
  mpnn_model_type: protein_mpnn  # use ligand_mpnn for ligand-aware design
  rf3_ckpt: /path/to/rf3_checkpoint.ckpt
  boltz_executable: /path/to/boltz-env/bin/boltz
```

Command-line flags override YAML values; explicit `assets:` values override
environment variables. Use `rfo --help` for supported flags. The default RF3
configuration reads the checkpoint from `RF3_CKPT` (or `assets.rf3_ckpt`).

The Boltz input is derived from the complex by default. Use
`af3_json_template` to specify SMILES ligands or embedded MSAs explicitly.
Without an MSA, Boltz runs in single-sequence mode. `cyclic: true` marks chain
A as cyclic in the Boltz input; it does not add cyclic bond constraints to the
RF3 optimizer.

## Outputs

`out_dir/` contains one `recycle_<n>/` directory per completed cycle,
`inputs/` with the redesigned complexes, and `<name>_records.json` with
per-cycle scores, timing and designed sequences. Each cycle's `mpnn/`
directory contains `mpnn_design.pdb`, `mpnn_result.json`, the design mask in
`mpnn_spec.json`, and upstream samples. RFO carries the **first** MPNN sample
forward when `mpnn_num_seqs > 1`; it does not rank the samples.

RF3's `modelhub_pred/` output name is retained for compatibility with existing
analysis scripts. Independent AlphaFold3 filtering is outside this pipeline.

## Citation

If you use this code or data in your work, please consider citing:

```bibtex
@article{zhang2026_rfoptimization,
    author = {Zhang, Odin and Wang, Jiaqi and Thompson, Tuscan Rock and You, Ziyi and Song, Zihao and DiMaio, Frank and Baker, David},
    title = {{RFOptimization}: Guiding Design Optimization with All-Atom Structure Prediction},
    elocation-id = {2026.09.04.749184},
    year = {2026},
    doi = {10.64898/2026.09.04.749184},
    publisher = {openRxiv},
    URL = {https://www.biorxiv.org/content/10.64898/2026.09.04.749184v2},
    eprint = {https://www.biorxiv.org/content/10.64898/2026.09.04.749184v2.full.pdf},
    journal = {bioRxiv}
}
```
