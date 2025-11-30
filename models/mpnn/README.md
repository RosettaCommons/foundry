# ProteinMPNN and LigandMPNN

> [!WARNING]
> **Benchmarking**: Please use the old repositories of ProteinMPNN and LigandMPNN for model benchmarking/comparison until the API and public weights stabilize. We are in the process of validating that the re-implementation (both the retrained version and the old weight loading option) is as performant as the original models.

> [!IMPORTANT]
> **Issues**: Please provide feedback on any issues you encounter with the ProteinMPNN/LigandMPNN re-implementation. We are particularly interested in discrepancies between the original models and this re-implementation, issues with performance when loading the original weights from the old repositories, problems with inference hyperparameters/conditioning, and input/output bugs.

> [!IMPORTANT]
> **API Instability**: We are currently finalizing some cleanup work on the inference API and training code. Please expect the API (including input formats and outputs) to stabilize in the upcoming weeks. Thank you for your patience!

> [!IMPORTANT]
> **Training Code and New Weights**: We are working to release the dataframes used for retrianing the ProteinMPNN and LigandMPNN re-implementations. Also, we are finalizing the retraining runs and will release weights retrained within this repository shortly.

ProteinMPNN enables protein sequence design given a fixed backbone structure of a protein. LigandMPNN extends this functionality to enable fixed-backbone sequence design of proteins in the context of ligands (i.e. small molecules, ions, DNA/RNA, etc.). This module represents a re-implementation of the original ProteinMPNN and LigandMPNN models within the modelforge/atomworks framework.

For more information on the original models, please see:
- [Robust deep learning–based protein sequence design using ProteinMPNN](https://doi.org/10.1126/science.add2187) | [ProteinMPNN Original Github](https://github.com/dauparas/ProteinMPNN)
- [Atomic context-conditioned protein sequence design using LigandMPNN](https://doi.org/10.1038/s41592-025-02626-1) | [LigandMPNN Original Github](https://github.com/dauparas/LigandMPNN)

This guide provides instructions on preparing inputs and running inference for ProteinMPNN/LigandMPNN, as well as training these models.

## Inference

> [!IMPORTANT] 
> When using weights from the original ProteinMPNN/LigandMPNN repositories, please ensure to set `is_legacy_weights` to `True` when running inference.

### Notes on Programmatic (Scripted) Inference
- Currently, 'mpnn_bias' and 'mpnn_pair_bias' annotations cannot be saved to CIF files due to shape limitations. As a result, these annotations must be recreated (either directly with annotation on the atom array or via the input config dictionary) when reloading designed structures from CIF files.