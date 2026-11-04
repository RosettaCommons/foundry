# ProteinMPNN, LigandMPNN, and SolubleMPNN

> [!WARNING]
> **Benchmarking**: Please use the old repositories of ProteinMPNN, LigandMPNN, and SolubleMPNN for model benchmarking/comparison until the API and public weights stabilize. We are in the process of validating that the re-implementation (both a retrained version and the old weight loading option) is as performant as the original models.

> [!IMPORTANT]
> **Issues**: Please provide feedback on any issues you encounter with the ProteinMPNN/LigandMPNN/SolubleMPNN re-implementation. We are particularly interested in discrepancies between the original models and this re-implementation, issues with performance when loading the original weights from the old repositories, problems with inference hyperparameters/conditioning, and input/output bugs.

ProteinMPNN enables protein sequence design given a fixed backbone structure of a protein. LigandMPNN extends this functionality to enable fixed-backbone sequence design of proteins in the context of ligands (i.e. small molecules, ions, DNA/RNA, etc.). This module represents a re-implementation of the original ProteinMPNN and LigandMPNN models within the modelforge/atomworks framework.

## Installation

The MPNN models are installed with the standard foudry install. See the [general README](../../README.md) or the [online documentation](https://rosettacommons.github.io/foundry/index.html) for installation instructions.

## Usage

See the [model documentation](docs/index.md) or the [online documentation](https://rosettacommons.github.io/foundry/models/mpnn/index.html) for usage information.
