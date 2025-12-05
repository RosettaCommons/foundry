# De novo Design of Biomolecular Interactions with RFdiffusion3

RFdiffusion3 (RFD3) is a diffusion method that can design protein structures 
under complex constraints. 

This repository contains both the training and inference code, and
both are described in more detail below. 

<p align="center">
  <img src="docs/.assets/overview.png" alt="All-atom design with RFD3">
</p>

> [!NOTE]
> Looking for config documentation? See [here](./docs/input.md)

## Getting Started
1. Install RFdiffusion3. See [Main README](../../README.md) for instructions how to install all models to run full pipeline (recommended). If you have already installed all the models skip [here](#run-inference).
```bash
pip install rc-foundry[rfd3]
```
2. Download checkpoint to your desired checkpoint location.
```bash
foundry install rfd3 --checkpoint-dir <path/to/ckpt/dir>
```
This sets `FOUNDRY_CHECKPOINT_DIRS` and will in future look for checkpoints in that directory (alongside the default `~/.foundry/checkpoints` location), allowing you to run inference without supplying the checkpoint path. The checkpoint directory is optional, defaulting to `~/.foundry/checkpoints` if unset.

## Running Inference

To run inference (with foundry installed in your environment, or RFD3 & Foundry src in PYTHONPATH):
```bash
rfd3 design out_dir=logs/inference_outs/demo/0 inputs=models/rfd3/docs/demo.json skip_existing=False dump_trajectories=True prevalidate_inputs=True
```

Additional unnecessary args here are added:
- Including dumping and aligning trajectory structures can be useful for debugging your setup or making cool gifs.
- Printing the config and dumping trajectories are turned off by default, but turned on here for verbosity
- `prevalidate_inputs` will check that your inputs are valid before running inference. Helpful if your json has a number of different configs you want to debug / double check are valid before loading the checkpoints.
- Only `out_dir` and `inputs` are required. The output directory will automatically be created. 


There are various interesting ways you can use RFD3 beyond Atom14 design as it's trained on a large array of different tasks.
For example, you can fix sequence and not structure (prediction-type task), fix the backbone and unfix the sequence (MPNN-type inverse folding) or unfix the sidechains only (PLACER/ChemNet-style):

<p align="center">
  <img src="docs/.assets/conditioning.png" alt="Conditioning options for RFD3">
</p>

For full details on how to specify inputs, see the [input specification documentation](./docs/input.md). You can also see `foundry/models/rfd3/configs/inference_engine/rfdiffusion3.yaml` for even more options.

## Further example jsons for different applications
Additional examples are broken up by use case. If you have cloned the
repository, matching `.json` files are in `foundry/models/rfd3/docs`
that can be run directly, similar to the previous example. 

In the examples the paths to the input files are specified assuming
that you are running the examples from the `foundry/models/rfd3/docs`
directory. If you would like to run RFD3 from a different location, 
you will need to change the path in the `.json` file(s) before running.

<table>
  <tr>
    <td align="center">
      <h3><a href="./docs/na_binder_design.md">Nucleic acid binder design</a></h3>
      <img src="docs/.assets/dna.png" height="150" />
    </td>
    <td align="center">
      <h3><a href="./docs/sm_binder_design.md">Small molecule binder design</a></h3>
      <img src="docs/.assets/sm.png" height="150" />
    </td>
    <td align="center">
      <h3><a href="./docs/protein_binder_design.md">Protein binder design</a></h3>
      <img src="docs/.assets/ppi.png" height="150" />
    </td>
  </tr>
  <tr>
    <td align="center">
      <h3><a href="./docs/enzyme_design.md">Enzyme design</a></h3>
      <img src="docs/.assets/enzyme.png" height="150" />
    </td>
    <td align="center">
      <h3><a href="./docs/symmetry.md">Symmetric design</a></h3>
      <img src="docs/.assets/symm.png" height="150" />
    </td>
  </tr>
</table>

## Training and Fine-Tuning

We make available to the community not only the weights to run RFdiffusion3 but also the complete training code, easily extendable to additional use cases. Any AtomWorks-compatible dataset (and thus, any collection of structure files) can be readily incorporated and used for training or fine-tuning.

### Dataset Configuration

#### PDB Training

To train on the PDB:

1. Set up PDB and CCD mirrors as described in the [AtomWorks documentation](https://rosettacommons.github.io/atomworks/latest/mirrors.html)
2. Update the [path configs](/models/rfd3/configs/paths/) to point to the correct base directories for the metadata parquets
3. Set the `PDB_MIRROR` and `CCD_PATH` variables in your `.env` file

#### Custom Datasets

RFdiffusion3 supports arbitrary datasets of structure files for training and fine-tuning via AtomWorks. See the [AtomWorks dataset documentation](https://rosettacommons.github.io/atomworks/latest/auto_examples/dataset_exploration.html) for details on creating custom datasets.

### Running Training

After setting up Hydra configs, launch a training run:
```bash
uv run python models/rfd3/src/rfd3/train.py experiment=pretrain ckpt_path=<path/to/ckpt>
```

Supplying `ckpt_path=null` (default) will start with fresh weights.
See the [path configs](/models/rfd3/configs/paths/) to customize data input and log output directories.

### Logging Configuration

Training runs support logging via [Weights & Biases](https://wandb.ai/). To enable wandb logging:

```bash
uv run python models/rfd3/src/rfd3/train.py experiment=pretrain logger=wandb
```

To run training without wandb (default):
```bash
uv run python models/rfd3/src/rfd3/train.py experiment=pretrain logger=csv
``` 

### Install HBPLUS for training with hydrogen bond conditioning:

1. Download hbplus from here: https://www.ebi.ac.uk/thornton-srv/software/HBPLUS/download.html (available for free)
2. Follow the installation instruction here: https://www.ebi.ac.uk/thornton-srv/software/HBPLUS/install.html
3. Update `HBPLUS_PATH` in `foundry/.env` file with the path to your `hbplus` executable.

## Distributed Training
To use distributed training, you could use a command such as this (we use Lightning Fabric to handle ddp)
```
EFFECTIVE_BATCH_SIZE=16
DEVICES_PER_NODE= #INSERT NUMBER OF DEVICES PER NODE
NNODES = # INSERT NUMBER OF NODES
GRAD_ACCUM_STEPS=$((EFFECTIVE_BATCH_SIZE / (DEVICES_PER_NODE * NNODES)))
uv run python models/rfd3/src/rfd3/train.py \
    experiment=pretrain \
    trainer.devices_per_node=$DEVICES_PER_NODE \
    trainer.num_nodes=$SLURM_NNODES \
    trainer.grad_accum_steps=$GRAD_ACCUM_STEPS"
```
Notably, fabric must receive `devices_per_node` and the number of nodes (`num_nodes`) you're training on.

**Dataset Paths:** See the paths [configs](/models/rfd3/configs/paths/) to customize the paths where data is read from and where logs are written. There is also a wandb config that can be enabled if you want to log training through wandb. 

**Hydra configs and experiments:** In the example above, the `experiment` argument is a hydra-native argument. For RFD3, it will look for config overrides in `/models/rfd3/configs/experiment/<experiment-name>.yaml` and apply them on top of the base configs

**Conditioning during training:** RFD3 is trained on a multitude of conditioning tasks, and does so by randomly 'creating problems' for it to solve during training. For example, for a random training example it gets a random set of tokens to be 'motif tokens', then subsets those to whether specific atoms should be fixed, and further subsets the information to whether, say, sequence, coordinates or the sequence index should be fixed. It's pretty complicated to evaluate and how it was put together was more of an art than a science. There's likely still room for 
further optimization!

In `models/rfd3/configs/datasets/design_base.yaml` there's the shared configs for all datasets under `global_transform_args`. The dials that control the conditioning described above go under `training_conditions`, where for example `tipatom` - a specific preset conditioning sampler which more frequently fixes few tokens with few atoms - and others can be found.

**Training with WandB:** We strongly recommend tracking your runs via wandb. To use it, simply have your WANDB_API_KEY set and use the wandb logger. For more details see [here](wandb.ai)

# Appendix

## Install HBPLUS for hydrogen bond conditioning:
One of the examples shows how to incorporate hydrogen bond conditioning 
into your designs. To make use of this feature, you will need to 
additionally complete the following steps:

1. Download hbplus from here: https://www.ebi.ac.uk/thornton-srv/software/HBPLUS/download.html (available for free)
2. Follow the installation instruction here: https://www.ebi.ac.uk/thornton-srv/software/HBPLUS/install.html
3. Update `HBPLUS_PATH` in `foundry/.env` file with the path to your `hbplus` executable.

## Citation

If you use this code or data in your work, please consider citing:

```bibtex
@article {butcher2025_rfdiffusion3,
	author = {Butcher, Jasper and Krishna, Rohith and Mitra, Raktim and Brent, Rafael Isaac and Li, Yanjing and Corley, Nathaniel and Kim, Paul T and Funk, Jonathan and Mathis, Simon Valentin and Salike, Saman and Muraishi, Aiko and Eisenach, Helen and Thompson, Tuscan Rock and Chen, Jie and Politanska, Yuliya and Sehgal, Enisha and Coventry, Brian and Zhang, Odin and Qiang, Bo and Didi, Kieran and Kazman, Maxwell and DiMaio, Frank and Baker, David},
	title = {De novo Design of All-atom Biomolecular Interactions with RFdiffusion3},
	elocation-id = {2025.09.18.676967},
	year = {2025},
	doi = {10.1101/2025.09.18.676967},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/11/19/2025.09.18.676967},
	eprint = {https://www.biorxiv.org/content/early/2025/11/19/2025.09.18.676967.full.pdf},
	journal = {bioRxiv}
}
```
