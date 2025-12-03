# ProteinMPNN, LigandMPNN, and SolubleMPNN

> [!WARNING]
> **Benchmarking**: Please use the old repositories of ProteinMPNN, LigandMPNN, and SolubleMPNN for model benchmarking/comparison until the API and public weights stabilize. We are in the process of validating that the re-implementation (both a retrained version and the old weight loading option) is as performant as the original models.

> [!IMPORTANT]
> **Issues**: Please provide feedback on any issues you encounter with the ProteinMPNN/LigandMPNN/SolubleMPNN re-implementation. We are particularly interested in discrepancies between the original models and this re-implementation, issues with performance when loading the original weights from the old repositories, problems with inference hyperparameters/conditioning, and input/output bugs.

ProteinMPNN enables protein sequence design given a fixed backbone structure of a protein. LigandMPNN extends this functionality to enable fixed-backbone sequence design of proteins in the context of ligands (i.e. small molecules, ions, DNA/RNA, etc.). This module represents a re-implementation of the original ProteinMPNN and LigandMPNN models within the modelforge/atomworks framework.

For more information on the original models, please see:
- ProteinMPNN: [Robust deep learning–based protein sequence design using ProteinMPNN](https://doi.org/10.1126/science.add2187) | [ProteinMPNN Original Github](https://github.com/dauparas/ProteinMPNN)
- LigandMPNN: [Atomic context-conditioned protein sequence design using LigandMPNN](https://doi.org/10.1038/s41592-025-02626-1) | [LigandMPNN Original Github](https://github.com/dauparas/LigandMPNN)
- SolubleMPNN: [Computational design of soluble and functional membrane protein analogues](https://doi.org/10.1038/s41586-024-07601-y)

This guide provides instructions on preparing inputs and running inference for ProteinMPNN/LigandMPNN, as well as training these models.

## Installation
### A. Installation using `uv`
```bash
git clone https://github.com/RosettaCommons/rc-foundry.git \
  && cd rc-foundry \
  && uv python install 3.12 \
  && uv venv --python 3.12 \
  && source .venv/bin/activate \
  && uv pip install -e ".[mpnn]"
```

### B. Download Model Weights

<details>
<summary><strong>ProteinMPNN</strong></summary>

Please use the following settings with these ProteinMPNN weights:
- `model_type`: `"protein_mpnn"`
- `is_legacy_weights`: `True`

48 Nearest Neighbors, $\sigma = 0.20 Å$ Gaussian noise during training:
```bash
wget https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_020.pt
```
<details>
<summary>Additional ProteinMPNN Weights</summary>

48 Nearest Neighbors, $\sigma = 0.02 Å$ Gaussian noise during training:
```bash
wget https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_002.pt
```
48 Nearest Neighbors, $\sigma = 0.10 Å$ Gaussian noise during training:
```bash
wget https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_010.pt
```
48 Nearest Neighbors, $\sigma = 0.30 Å$ Gaussian noise during training:
```bash
wget https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_030.pt
```
</details>
</details>

<details>
<summary><strong>LigandMPNN</strong></summary>

Please use the following settings with these LigandMPNN weights:
- `model_type`: `"ligand_mpnn"`
- `is_legacy_weights`: `True`

32 Nearest Neighbors, $\sigma = 0.10 Å$ of Gaussian noise during training, 25 ligand atom context:
```bash
wget https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_010_25.pt
```

<details>
<summary>Additional LigandMPNN Weights</summary>

32 Nearest Neighbors, $\sigma = 0.05 Å$ of Gaussian noise during training, 25 ligand atom context:
```bash
wget https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_005_25.pt
```
32 Nearest Neighbors, $\sigma = 0.20 Å$ of Gaussian noise during training, 25 ligand atom context:
```bash
wget https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_020_25.pt
```
32 Nearest Neighbors, $\sigma = 0.30 Å$ of Gaussian noise during training, 25 ligand atom context:
```bash
wget https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_030_25.pt
```
</details>
</details>

<details>
<summary><strong>SolubleMPNN</strong></summary>

Please use the following settings with these SolubleMPNN weights:
- `model_type`: `"protein_mpnn"`
- `is_legacy_weights`: `True`

48 Nearest Neighbors, $\sigma = 0.20 Å$ Gaussian noise during training:
```bash
wget https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_020.pt
```

<details>
<summary>Additional SolubleMPNN Weights</summary>

48 Nearest Neighbors, $\sigma = 0.02 Å$ Gaussian noise during training:
```bash
wget https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_002.pt
```
48 Nearest Neighbors, $\sigma = 0.10 Å$ Gaussian noise during training:
```bash
wget https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_010.pt
```
48 Nearest Neighbors, $\sigma = 0.30 Å$ Gaussian noise during training:
```bash
wget https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_030.pt
```
</details> 
</details>

## Inference
> [!WARNING]
> **Known Bug**: There is currently an issue with loading MPNN user annotation (temperature, designed residues, etc.) from CIF/atom array annotations. Command line passing of these options works as expected, as does `input_dict` specificiation with MPNNInferenceEngine.

> [!IMPORTANT]
> **API Instability**: We are currently finalizing some cleanup work on the inference API. Please expect the API (including input formats and outputs) to stabilize in the upcoming weeks. Thank you for your patience!

> [!IMPORTANT] 
> When using weights from the original ProteinMPNN/LigandMPNN/SolubleMPNN repositories, please ensure to set `is_legacy_weights` to `True` when running inference.

### A. Command Line Inference
Detailed documentation coming soon!

### B. JSON-based Inference
Detailed documentation coming soon!

### C. Programmatic (Scripted) Inference
Detailed documentation coming soon!

> [!IMPORTANT]
> Currently, 'mpnn_bias' and 'mpnn_pair_bias' annotations cannot be saved to CIF files due to shape limitations. As a result, these annotations must be recreated (either directly with annotation on the atom array or via the input config dictionary) when reloading designed structures from CIF files.

## Training
Instructions for training ProteinMPNN/LigandMPNN/SolubleMPNN models will be updated here shortly.

> [!IMPORTANT]
> **Training Code and New Weights**: We are working to release the dataframes used for retrianing the ProteinMPNN, LigandMPNN, and SolubleMPNN re-implementations. Also, we are finalizing the retraining runs and will release weights retrained within this repository shortly.