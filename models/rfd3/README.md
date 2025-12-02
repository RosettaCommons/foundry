# De novo Design of Biomolecular Interactions with RFdiffusion3

<p align="center">
  <img src="docs/.assets/trajectory.png" alt="All-atom diffusion with RFD3">
</p>


##  Installation, Setup, and a Basic Design
### A. Installation using `uv`
```bash
git clone https://github.com/RosettaCommons/modelforge.git \
  && cd modelforge \
  && uv python install 3.12 \
  && uv venv --python 3.12 \
  && source .venv/bin/activate \
  && uv pip install -e ".[rfd3]"
```

### B. Download model weights for RFD3
```bash
wget http://files.ipd.uw.edu/pub/rfd3/rfd3_foundry_2025_12_01.ckpt
```

### C. Run a test prediction
```bash
rf3 fold inputs='tests/data/5vht_from_json.json'
```

You may then specify the specific checkpoint, if desired, with:
```bash
rf3 fold inputs='tests/data/5vht_from_json.json' ckpt_path='/path/to/rf3_921.pt'
```

**Setup**
RFD3 currently requires specific branches for `cifutils` and `datahub` so we recommend cloning the branch with submodules:
```bash
git clone -b aa_design/main git@github.com:baker-laboratory/modelhub.git
git submodule init
git submodule update --init
export PROJECT_PATH="$(pwd)/projects/aa_design"
```

Files for RFD3 exist under this folder (`projects/aa_design`), and wrap around the components for the AF3 repro under `src/modelhub`. 
The AF3 repro might not work on this branch since it is currently not kept up to date.

If you run `inference.py` as a script (via `./src/inference.py`), then the shebang file should take care of the submodule paths for you (if you cloned the submodules). Otherwise, add the following to your environment;
```
chmod +x src/modelhub/*.py
```

## Inference:
The following checkpoint is updated continuously (see channel https://chat.ipd.uw.edu/ipd/channels/rfdiffusion3):
```bash
cur_ckpt=/projects/ml/aa_design/models/rfd3_latest.ckpt
```

To run inference, use:
```bash
./src/modelhub/inference.py out_dir=logs/inference_outs/demo/0 ckpt_path=$cur_ckpt inputs=projects/aa_design/tests/test_data/demo.json print_config=True dump_trajectories=True
```

Additional args here are added for verbosity, aligning trajectory structures, printing the config and dumping trajectories are turned off by default.

For full details on how to specify inputs, see the `input.md` documentation.

## PPI-Design

See `input.md`, please feel free to reach out to me (Rafi Brent) if you have any questions or concerns. I'm happy to help out however I can!

## Enzyme-Design
See `input.md`, feel free to send jbutch or jfunk21 a message if in doubt!

## Symmetric-Design
See `symmetry.md`, plese feel free to reach out to aimura or heisen if you have any questions!

## Training (w & w/o WandB):

Add `export PROJECT_PATH=$(pwd)/projects/aa_design` to `scripts/slurm/launch.sh`, where `$(pwd)` is the repositories' absolute path
You will also want to add your cifutils, datahub and modelhub (`$(pwd)`) paths to `launch.sh`.

To launch a training run, use:
```
sbatch -J rfd3-full-sparse launch.sh
```

Optionally ensure your `WANDB_API_KEY` is an environment variable. You can disable wandb by including the following at the top of your experiment config:
```yaml
defaults:
  - override /logger: csv  # turns off wandb logger
```

## Conditioining Pipeline
Both inference and validation passes arguments to `create_atom_array_from_design_specification`, to create an atom array with all the information needed to run inference. 
This is then passed through the same processing pipeline as in training with `is_inference=True` (pipeline in `./projects/aa_design/transforms/pipelines.py`).

<p align="center">
  <img src="docs/.assets/pipeline.png" alt="Atom14 Design Pipelines">
  <figcaption>Overview of important transforms in the Atom14 conditioning pipeline.
  </figcaption>
</p>



## Citation

If you use this code or data in your work, please cite:

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