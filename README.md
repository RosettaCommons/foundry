# Open-Source Neural Networks for Biomolecular Tasks

`ModelForge` is a repository of open-source models for common biomolecular tasks, including structure prediction, fixed-backbone sequence design ("inverse folding"), and *de novo* protein design.

All models within `ModelForge` share a common training harness and integrate with [AtomWorks](https://github.com/RosettaCommons/atomworks) – our generalized computational framework for biomolecular modeling.

For more information, please see our preprint, [Accelerating Biomolecular Modeling with AtomWorks and RF3](https://doi.org/10.1101/2025.08.14.670328).

> [!WARNING]
> We fixed an inference bug on 8/29 that arose during codebase migration and impacted predictions from JSON and from mmCIF/PDB; the issue is now resolved but for the purposes of model benchmarking predictions should be re-run.

> [!IMPORTANT]
> We are currently finalizing some cleanup work within our repositories. Please expect the APIs (e.g., function and class names, inputs and outputs) to stabilize within the next two weeks. Thank you for your patience!

> [!NOTE]
> Training code coming very soon, with documentation on how to fine-tune on new datasets! 

## RosettaFold3 (RF3)

[RF3](https://doi.org/10.1101/2025.08.14.670328) is a structure prediction neural network that narrows the gap between closed-source AF-3 and open-source alternatives.

<div align="center">
  <img src="docs/_static/prot_dna.png" alt="Protein-DNA complex prediction" width="400">
</div>

> [!TIP]
> Complete inference instructions for RF3 are provided [here](models/rf3/README.md).

### RF3 Quick Start - Installation & Usage

Follow these steps to set up **ModelForge** and run a test prediction with **RF3**.

---

#### 1. Install the source repository and RF3 model using `uv`

```bash
git clone https://github.com/RosettaCommons/modelforge.git \
  && cd modelforge \
  && uv python install 3.12 \
  && uv venv --python 3.12 \
  && source .venv/bin/activate \
  && uv pip install -e ./models/rf3
```

> [!NOTE]
> Installing `rf3` automatically installs `modelhub` (shared utilities) as a dependency.

#### 2. Download model weights for RF3 
```bash
wget http://files.ipd.uw.edu/pub/rf3/rf3_latest.pt
```

#### 3. Run a test prediction
```bash
rf3 fold tests/data/5vht_from_json.json
```

Details on the exact formatting of the json files are available [here](models/rf3/README.md).

## Development

### Package Structure

ModelForge uses a multi-package architecture:

- **`modelhub`**: Core package containing shared utilities, training infrastructure, and base classes
- **`models/rf3/`**: RF3 model package with model-specific code and dependencies
- **`models/<future>/`**: Additional models can be added as separate packages

### Installation Options

#### For Users (Single Model)

Install only the model you need. Dependencies (including `modelhub`) are automatically installed:

```bash
# Install RF3 (includes modelhub automatically)
uv pip install -e ./models/rf3

# Future: Install other models
# uv pip install -e ./models/other_model
```

#### For Core Developers (Multiple Packages)

Install both `modelhub` and models in editable mode for development:

```bash
# Install modelhub and RF3 in editable mode
uv pip install -e . -e ./models/rf3

# Or install only modelhub (no models)
uv pip install -e .
```

This approach allows you to:
- Modify `modelhub` shared utilities and see changes immediately
- Work on specific models without installing all models
- Add new models as independent packages in `models/`

### Adding New Models

To add a new model:

1. Create `models/<model_name>/` directory with its own `pyproject.toml`
2. Add `modelhub` as a dependency
3. Implement model-specific code in `models/<model_name>/src/`
4. Users can install with: `uv pip install -e ./models/<model_name>`

## Development