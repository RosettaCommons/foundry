# MPNN Documentation

```{caution}
The MPNN model is still being benchmarked for comparison to the 
original ProteinMPNN, LigandMPNN, and SolubleMPNN implementations.
```

## Available Models

Foundry attempts to collect the functionality of various MPNN design methods under one unified package, 
with the backend support of [atomworks](https://github.com/RosettaCommons/atomworks) 
and [foundry](https://github.com/RosettaCommons/foundry). The currently supported models and weights are listed below. More MPNN
models and weights will be added to foundry in the future.

Are we missing a model you would like to work with? Create a PR with the
necessary files and/or code changes or open an issue requesting it.

* [ProteinMPNN](https://doi.org/10.1126/science.add2187), ([original repository](https://github.com/dauparas/ProteinMPNN)) enables protein sequence design given a fixed backbone structure of a protein. ProteinMPNN works soley on canonical amino acids, ignoring any non-protein residues. 
* [LigandMPNN](https://doi.org/10.1038/s41592-025-02626-1), ([original repository](https://github.com/dauparas/LigandMPNN)) extends ProteinMPNN functionality to enable fixed-backbone sequence design of proteins in the context of ligands (i.e. small molecules, ions, DNA/RNA, etc.).
* [SolubleMPNN](https://doi.org/10.1038/s41586-024-07601-y) is a version of ProteinMPNN which has been trained specifically on soluble (e.g. non-membrane) proteins. This has been shown to perform better with solubilizing proteins with membrane-protein-like folds, as the default ProteinMPNN recapitulates surface hydrophobics.

Standard weights for ProteinMPNN and LigandMPNN are included with the base-models install. Additional weights for other models/training settings are also available:

```{toctree}
:maxdepth: 1

weights.md
```

## Basic Usage

The foundry version of MPNN can take either PDB or CIF input structures.
Model type, input path and output directory must all be specified.

    mpnn --model_type protein_mpnn --structure_path input.cif --out_directory input/

    mpnn --model_type ligand_mpnn --structure_path with_ligand.cif --out_directory ligand/

    mpnn --model_type soluble_mpnn --structure_path input.cif --out_directory soluble/

The output directory will be created if it does not already exist.

```{toctree}
:maxdepth: 1

inference_input.md
config_json.md
conversion.md
```

## Outputs

By default, in the output directory there will be a FASTA-formatted file with the sequence outputs,
as well as a backbone-only CIF-formatted output structure of the designs. 
(The sidechain packing model of LigandMPNN is not currently implemented.)

Both the FASTA and CIF output files should contain the sequence and confidence values. The CIF output should also contain additional information in the `_mpnn_input` and `_mpnn_output` tables. These can be extracted via standard CIF-file readers, or through text processing tools (e.g. grep).

```{toctree}
:maxdepth: 1

outputs.md
```

## Retraining

```{toctree}
:maxdepth: 1

training.md
```
