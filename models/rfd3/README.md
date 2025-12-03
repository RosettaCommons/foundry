# De novo Design of Biomolecular Interactions with RFdiffusion3

<p align="center">
  <img src="docs/.assets/trajectory.png" alt="All-atom diffusion with RFD3">
</p>


##  Installation, Setup, and a Basic Design
### A. Installation using `uv`
```bash
git clone https://github.com/RosettaCommons/foundry.git \
  && cd foundry \
  && uv python install 3.12 \
  && uv venv --python 3.12 \
  && source .venv/bin/activate \
  && uv pip install -e ".[rfd3]"
```
<!--
> [!IMPORTANT]
> You must install `foundry` (the root package) with `-e` first, then install `rfd3`. This ensures both packages are in editable mode for proper development workflow.
-->
> [!NOTE]
> optionally make installed venv available as ipynb kernel (helpful for running examples in `examples/all.ipynb`)
`python -m ipykernel install --user --name=foundry --display-name "foundry"`

### B. Download model weights for RFD3
```bash
wget http://files.ipd.uw.edu/pub/rfd3/rfd3_foundry_2025_12_01.ckpt
```
*You can store these weights anywhere you would like,
but if you do not store them in the root directory
you will need to change the `cur_ckpt` variable discussed
later on.*

**Setup**
```bash
export PROJECT_PATH="$(pwd)/models/rfd3/src:$(pwd)/src:$(pwd)/lib/atomworks/src"
```
If your virtual environment is not already active you will 
also need to run:
```
source .venv/bin/activate
```

Files for RFD3 exist under this folder (`models/rfd3`), and wrap around the components of RF3 under `src/foundry/`. 
```
chmod +x src/foundry/*.py
```

## Inference:
```bash 
cur_ckpt=rfd3_foundry_2025_12_01.ckpt
```

To run inference
```bash
python ./models/rfd3/src/rfd3/run_inference.py out_dir=logs/inference_outs/demo/0 ckpt_path=$cur_ckpt inputs=./models/rfd3/docs/demo.json verbose=True dump_trajectories=True
```

> [!NOTE]
> This demo will take a very long amount of time if run on a
> CPU instead of a GPU. On a GPU, this should take on the
> order of 10 minutes.

Additional args here are added for verbosity, aligning trajectory structures, printing the config and dumping trajectories are turned off by default.

The output directory will automatically be created.

For full details on how to specify inputs, see the [input specification documentation](./docs/input.md). You can also see `models/rfd3/configs/inference_engine/rfdiffusion3.yaml`.

## Further example jsons for different applications

<table>
  <tr>
    <td align="center">
      <h3><a href="./docs/binder_design.md">Nucleic acid binder design</a></h3>
      <img src="docs/.assets/dna.png" height="150" />
    </td>
    <td align="center">
      <h3><a href="./docs/binder_design.md">Small molecule binder design</a></h3>
      <img src="docs/.assets/sm.png" height="150" />
    </td>
    <td align="center">
      <h3><a href="./docs/binder_design.md">Protein binder design</a></h3>
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

## Training (w & w/o WandB): #TODO make sure correct

Add `export PROJECT_PATH=$(pwd)/models/rfd3` to `scripts/slurm/launch.sh`, where `$(pwd)` is the repositories' absolute path
You will also want to add your atomworks and foundry (`$(pwd)`) paths to `launch.sh`.

To launch a training run, use:
```
sbatch -J rfd3-full-sparse launch.sh
```

Optionally ensure your `WANDB_API_KEY` is an environment variable. You can disable wandb by including the following at the top of your experiment config:
```yaml
defaults:
  - override /logger: csv  # turns off wandb logger
```

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
