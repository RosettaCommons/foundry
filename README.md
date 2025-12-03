# Protein design with Foundry

Foundry provides tooling and infrastructure for using and training all classes of models for protein design, including design (RFD3), inverse folding (ProteinMPNN) and protein folding (RF3).

All models within Foundry rely on [AtomWorks](https://github.com/RosettaCommons/atomworks) - a unified framework for manipulating and processing biomolecular structures - for both training and inference. 

## Getting Started
### Quickstart guide
**Installation**
```bash
pip install rc-foundry[all]
```

**Downloading weights** All models can be downloaded to a target folder with:

```
foundry install all --checkpoint_dir <path/to/ckpt/dir>
```
This will download all the models supported (including multiple checkpoints of RF3) but as a beginner you can start with:
```
foundry install rfd3 ligandmpnn rf3 --checkpoint_dir  <path/to/ckpt/dir>
```

>*See `examples/all.ipynb` for how to run each model in a notebook.*

### RFdiffusion3 (RFD3)

[RFdiffusion3](https://www.biorxiv.org/content/10.1101/2025.09.18.676967v2) is an all-atom generative model capable of designing protein structures under complex constraints. 

> *See [models/rfd3/README.md](models/rfd3/README.md) for complete documentation.*

<div align="center">
  <img src="models/rfd3/docs/.assets/trajectory.png" alt="RFdiffusion3 generation trajectory." width="700">
</div>

### ProteinMPNN
[ProteinMPNN](https://www.science.org/doi/10.1126/science.add2187) and [LigandMPNN](https://www.nature.com/articles/s41592-025-02626-1) are lightweight inverse-folding models which can be use to design diverse sequences for backbones under constrained conditions.

> *See [models/mpnn/README.md](models/mpnn/README.md) for complete documentation.*


### RosettaFold3 (RF3)

[RF3](https://doi.org/10.1101/2025.08.14.670328) is a structure prediction neural network that narrows the gap between closed-source AF-3 and open-source alternatives.

<div align="center">
  <img src="docs/_static/prot_dna.png" alt="Protein-DNA complex prediction" width="400">
</div>

> *See [models/rf3/README.md](models/rf3/README.md) for complete documentation.*

---

## Development

### Code Organization

**Strict dependency flow:** `foundry` → `atomworks`

- **atomworks**: Structure I/O, preprocessing, featurization
- **foundry**: Model architectures, training, inference endpoints
- **models/\<model\>:** Released models.

#### For Core Developers (Multiple Packages)

Install both `foundry` and models in editable mode for development:

```bash
# Install foundry and RF3 in editable mode
uv pip install -e . -e ./models/rf3 -e ./models/rfd3 -e ./models/mpnn

# Or install only foundry (no models)
uv pip install -e .
```

This approach allows you to:
- Modify `foundry` shared utilities and see changes immediately
- Work on specific models without installing all models
- Add new models as independent packages in `models/`

### Adding New Models

To add a new model:

1. Create `models/<model_name>/` directory with its own `pyproject.toml`
2. Add `foundry` as a dependency
3. Implement model-specific code in `models/<model_name>/src/`
4. Users can install with: `uv pip install -e ./models/<model_name>`

### Pre-commit Formatting

We ship a `.pre-commit-config.yaml` that runs `make format` (via `ruff format`) before each commit. Enable it once per clone:

```bash
pip install pre-commit  # if not already installed
pre-commit install
```

After installation the hook automatically formats the repo whenever you `git commit`. Use `pre-commit run --all-files` to apply it manually.

## Citation

If you use this repository code or data in your work, please cite the relavant work as below:

```bibtex
@article{corley2025accelerating,
  title={Accelerating biomolecular modeling with atomworks and rf3},
  author={Corley, Nathaniel and Mathis, Simon and Krishna, Rohith and Bauer, Magnus S and Thompson, Tuscan R and Ahern, Woody and Kazman, Maxwell W and Brent, Rafael I and Didi, Kieran and Kubaney, Andrew and others},
  journal={bioRxiv},
  year={2025}
}

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

@article{dauparas2022robust,
  title={Robust deep learning--based protein sequence design using ProteinMPNN},
  author={Dauparas, Justas and Anishchenko, Ivan and Bennett, Nathaniel and Bai, Hua and Ragotte, Robert J and Milles, Lukas F and Wicky, Basile IM and Courbet, Alexis and de Haas, Rob J and Bethel, Neville and others},
  journal={Science},
  volume={378},
  number={6615},
  pages={49--56},
  year={2022},
  publisher={American Association for the Advancement of Science}
}

@article{dauparas2025atomic,
  title={Atomic context-conditioned protein sequence design using LigandMPNN},
  author={Dauparas, Justas and Lee, Gyu Rie and Pecoraro, Robert and An, Linna and Anishchenko, Ivan and Glasscock, Cameron and Baker, David},
  journal={Nature Methods},
  pages={1--7},
  year={2025},
  publisher={Nature Publishing Group US New York}
}
```
## Acknowledgments
We thank Rachel Clune and Hope Woods from the RosettaCommons for their collaboration on the codebase, documentation, tutorials and examples. 
