# MPNN Models and Weights

The trained weights to be used with the model can be specified with

    --checkpoint_path checkpoint.pt

For all of the checkpoints available below, you must also specify

    --is_legacy_weights True

To use the standard weights (i.e. those downloaded with `foundry install`), simply omit the `--checkpoint_path` and `--is_legacy_weights` options.

## ProteinMPNN

[ProteinMPNN](https://doi.org/10.1126/science.add2187), ([original repository](https://github.com/dauparas/ProteinMPNN)) enables protein sequence design given a fixed backbone structure of a protein. ProteinMPNN works soley on canonical amino acids, ignoring any non-protein residues. 

### Weights 

Standard weights (48 Nearest Neighbors, $\sigma = 0.20 Å$ Gaussian noise) are included with the base-models install, or can be downloaded separately with

    foundry install proteinmpnn

Additional weights can be downloaded manually:

    # 48 Nearest Neighbors, $\sigma = 0.02 Å$ Gaussian noise during training:
    wget https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_002.pt
    # 48 Nearest Neighbors, $\sigma = 0.10 Å$ Gaussian noise during training:
    wget https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_010.pt
    # 48 Nearest Neighbors, $\sigma = 0.30 Å$ Gaussian noise during training:
    wget https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_030.pt

## LigandMPNN

[LigandMPNN](https://doi.org/10.1038/s41592-025-02626-1), ([original repository](https://github.com/dauparas/LigandMPNN)) extends ProteinMPNN functionality to enable fixed-backbone sequence design of proteins in the context of ligands (i.e. small molecules, ions, DNA/RNA, etc.).

### Weights 

Standard weights (32 Nearest Neighbors, $\sigma = 0.10 Å$ Gaussian noise during training, 25 ligand atom context) are included with the base-models install, or can be downloaded separately with

    foundry install ligandmpnn

Additional werights can be downloaded manually:

    # 32 Nearest Neighbors, $\sigma = 0.05 Å$ of Gaussian noise during training, 25 ligand atom context:
    wget https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_005_25.pt
    # 32 Nearest Neighbors, $\sigma = 0.20 Å$ of Gaussian noise during training, 25 ligand atom context:
    wget https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_020_25.pt
    # 32 Nearest Neighbors, $\sigma = 0.30 Å$ of Gaussian noise during training, 25 ligand atom context:
    wget https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_030_25.pt


## SolubleMPNN

[SolubleMPNN](https://doi.org/10.1038/s41586-024-07601-y) is a version of ProteinMPNN which has been trained specifically on soluble (e.g. non-membrane) proteins. This has been shown to perform better with solubilizing proteins with membrane-protein-like folds, as the default ProteinMPNN recapitulates surface hydrophobics.

### Weights 

Standard weights (48 Nearest Neighbors, $\sigma = 0.20 Å$ Gaussian noise) are not included with the base-models install, and must be downloaded with

    foundry install solublempnn

Additional weights can be downloaded manually:

    # 48 Nearest Neighbors, $\sigma = 0.02 Å$ Gaussian noise during training:
    wget https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_002.pt
    # 48 Nearest Neighbors, $\sigma = 0.10 Å$ Gaussian noise during training:
    wget https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_010.pt
    # 48 Nearest Neighbors, $\sigma = 0.30 Å$ Gaussian noise during training:
    wget https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_030.pt


