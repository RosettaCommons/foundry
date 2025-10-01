# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

### Setup
It is **CRITICAL** that before you run **ANY** commands, you activate the python environment like below.
Otherwise, you will always run into import and package errors.
```bash
# IMPORTANT! ALWAYS ACTIVATE THE PYTHON ENVIRONMENT!
source .venv/bin/activate
```

## Coding Practices

Read CAREFULLY the following coding practices. These are central tenants of the way we code. Whenver you write code, retroactively examine your code and ensure it conforms to the principles oulined below.
(1) Adhere to the do-not-repeat-yourself (DRY) practice. Break out common operations into shared functions or classes.
(2) Prefer functions over classes where possible; the more general the function, the better. Functional programming makes the code more extendable.
(3) Follow the YAGNI principle - "You Ain't Gunna Need It." Don't speculatively build functionality that I do not explicitly ask for. Prioritize code simplicity and brevity above all else.

## Project Overview

**ModelForge** is a repository of open-source neural networks for biomolecular structure prediction and design. The flagship model is **RosettaFold3 (RF3)**, a structure prediction network competitive with AlphaFold3. The repository uses a shared training harness and integrates with [AtomWorks](https://github.com/RosettaCommons/atomworks) for biomolecular data processing.

### Current Status
The repository is undergoing active refactoring. Code is organized into:
- `releases/rf3/`: Stable RF3 model release with complete inference and training
  - `src/rf3/`: RF3-specific implementation code
  - `configs/`: Hydra configuration files for RF3
  - `tests/`: RF3 test suite
- `src/modelhub/`: Shared utilities and base classes used across all models
- `tests/`: Shared tests at the repository level
- `lib/`: Git submodules including AtomWorks


## Key Commands

### Installation and Setup
```bash
# Clone and install
git clone https://github.com/RosettaCommons/modelforge.git
cd modelforge
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .

# Download RF3 weights
wget http://files.ipd.uw.edu/pub/rf3/rf3_latest.pt
```

### Development Commands
```bash
# IMPORTANT: Always activate virtual environment first
source .venv/bin/activate

# Code formatting and linting
make format                     # Format code using ruff (preferred)
ruff format src tests           # Format code directly
ruff check --fix src tests      # Lint and fix issues

# Cleanup
make clean                      # Delete compiled and cached files
```

### Testing
```bash
# IMPORTANT: Always activate virtual environment first
source .venv/bin/activate

# Run RF3-specific tests (from releases/rf3/)
cd releases/rf3/
pytest tests/                                    # All RF3 tests
pytest tests/test_inference_regression.py        # Inference regression tests
pytest tests/test_write_confidence.py            # Confidence output tests

# Run shared/root-level tests (from repository root)
cd /path/to/modelhub_latent
pytest tests/                                    # All shared tests
pytest tests/test_metrics.py                     # Metric tests
pytest tests/test_weight_loading.py              # Weight loading tests
pytest tests/test_torch_utils.py                 # Torch utility tests

# Run GPU-dependent tests (requires GPU)
pytest tests/test_inference_regression.py -m gpu

# Run with verbose output
pytest tests/ -v
```

### Inference (RF3)
```bash
# IMPORTANT: Navigate to RF3 release directory
cd releases/rf3/

# Basic structure prediction
rf3 fold inputs='tests/data/5vht_from_json.json'

# With MSA
rf3 fold inputs='../../docs/rf3/examples/3en2_from_json_with_msa.json'

# Batch processing multiple files
rf3 fold inputs='[file1.cif, file2.json, file3.pdb]'
rf3 fold inputs='path/to/directory'  # Process all CIF/PDB/JSON in directory

# Advanced inference options
rf3 fold inputs='input.json' \
    ckpt_path='/path/to/rf3_latest.pt' \
    out_dir='./predictions' \
    n_recycles=10 \
    diffusion_batch_size=5 \
    num_steps=50 \
    annotate_b_factor_with_plddt=true \
    early_stopping_plddt_threshold=0.5

# Templating (fix specific regions during prediction)
rf3 fold inputs='input.cif' \
    template_selection='[A, B/*/1-42, B/*/49-63]' \
    ground_truth_conformer_selection='[C, D]'

# Alternative: Direct Python invocation (more informative error messages)
python src/rf3/inference.py inputs='tests/data/5vht_from_json.json'
```

**Note**: RF3 uses Hydra for configuration, so arguments use `key=value` syntax (not `--key value`).

**Selection Syntax** (for templating): `CHAIN/RES_NAME/RES_ID/ATOM_NAME`
- Exact: `A/ALA/15/CA`
- Wildcard: `A/*/*/CA` (all CA atoms in chain A)
- Range: `A/*/5-10` (residues 5-10 in chain A)
- Union: `A, B` (chains A and B)

### Training (RF3)
```bash
# IMPORTANT: Navigate to RF3 release directory
cd releases/rf3/

# Train with specific experiment config
python src/rf3/train.py experiment=pretrained/rf3

# Override specific parameters
python src/rf3/train.py \
    experiment=pretrained/rf3 \
    trainer.max_steps=100000 \
    seed=42

# Resume from checkpoint
python src/rf3/train.py \
    experiment=pretrained/rf3 \
    ckpt_path=/path/to/checkpoint.ckpt

# Debug mode (quick testing)
python src/rf3/train.py \
    experiment=pretrained/rf3 \
    debug=default
```

## Architecture

### Package Structure
- `releases/rf3/`: Complete RF3 model release (self-contained)
  - `src/rf3/`: RF3 implementation package
    - `model/`: Neural network architecture (Pairformer, Diffusion Module, auxiliary heads)
      - `layers/`: Network components (attention, triangle multiplication, outer product)
      - `RF3.py`: Main model class
      - `RF3_structure.py`: Diffusion module and structural components
    - `data/`: Data pipelines and transformations
    - `inference_engines/`: RF3 inference implementations
    - `loss/`: Loss functions (diffusion loss, confidence loss, distogram loss)
    - `metrics/`: RF3-specific evaluation metrics
    - `trainers/`: Training logic using Lightning Fabric
    - `training/`: Training utilities (EMA, schedulers, checkpoints)
    - `utils/`: RF3-specific utilities
    - `callbacks/`: RF3-specific callbacks
    - `cli.py`: RF3 CLI entry point
    - `inference.py`: Inference script
    - `train.py`: Training script
    - `validate.py`: Validation script
  - `configs/`: Hydra configuration files for RF3
    - `model/`: Model architecture configs
    - `trainer/`: Training configs (loss, metrics)
    - `datasets/`: Dataset configs
    - `inference_engine/`: Inference engine configs
    - `callbacks/`, `logger/`, `paths/`, etc.
  - `tests/`: RF3 test suite
    - `test_inference_regression.py`: Regression tests
    - `test_write_confidence.py`: Confidence output tests
    - `data/`: Test data and baselines
- `src/modelhub/`: Shared utilities across all models
  - `callbacks/`: Shared training callbacks (health logging, timing, base classes)
  - `inference_engines/`: Base inference engine interface
  - `metrics/`: Shared evaluation metrics (base classes, common metrics)
  - `trainers/`: Shared training infrastructure (Fabric wrappers)
  - `utils/`: Common utilities (weights, logging, instantiators, DDP, torch utils)
  - `hydra/`: Hydra resolvers and utilities
- `tests/`: Repository-level shared tests
  - `test_metrics.py`: Shared metric tests
  - `test_weight_loading.py`: Weight loading tests
  - `test_torch_utils.py`: Torch utility tests
- `lib/`: Git submodules (AtomWorks)
- `docs/`: Documentation and examples

### Key Concepts
- **RF3 Architecture**: Pairformer (token-level) → Diffusion Module (atom-level)
  - Token-level representations: Single (`S`, I×C_s), Pair (`Z`, I×I×C_z)
  - Atom-level representations: Single (`Q`, L×C_atom), Pair (`P`, L×L×C_atompair)
  - Distogram head for token-level distance predictions
- **Hydra Configuration**: Composable configs with defaults and overrides
- **Lightning Fabric**: Distributed training with DDP support
- **AtomWorks Integration**: Unified data processing for structures, MSAs, templates
- **Input Flexibility**: Supports CIF, PDB, JSON, SMILES, and CCD codes

### Development Environment
- Python 3.12 required (3.11+ supported)
- Uses `ruff` for linting and formatting (configured in pyproject.toml)
- Testing with `pytest` including GPU-specific tests
- Environment variables: Configure in `.env` file (see `.env` template)
  - `PDB_MIRROR_PATH`: Local PDB mirror for training data
  - `CCD_MIRROR_PATH`: Chemical Component Dictionary mirror
  - `LOCAL_MSA_DIRS`: MSA search directories
  - Tool paths: `HHFILTER_PATH`, `MMSEQS2_PATH`, etc.

### Data Dependencies
The training and inference pipelines expect:
- PDB mirror with mmCIF files in standard RCSB sharding pattern
- CCD mirror for small molecule definitions
- Optional MSA data for protein chains
- Pre-computed metadata as parquet files (for training)

## RF3 Input Formats

### JSON Format
```json
{
  "name": "example_name",
  "components": [
    {
      "seq": "MKTAYIA...",           // Protein/NA sequence (supports non-canonical)
      "msa_path": "path/to/msa.a3m", // Optional MSA (a3m or fasta)
      "chain_id": "A"                 // Optional chain ID
    },
    {
      "smiles": "CC(=O)O"             // Small molecule via SMILES
    },
    {
      "ccd_code": "HEM"               // Chemical Component Dictionary code
    },
    {
      "path": "ligand.sdf"            // Structure file (SDF/CIF)
    }
  ],
  "bonds": [                          // Optional covalent modifications
    ["A/ASN/133/ND2", "B/NAG/1/C1"]
  ],
  "template_selection": ["A/*/1-50"], // Optional token-level templating
  "ground_truth_conformer_selection": ["C"] // Optional atom-level templating
}
```

### CIF/PDB Files
Standard RCSB format with optional MSA specification in CIF header:
```cif
data_3EN2
_msa_paths_by_chain_id.A   path/to/msa_A.a3m.gz
_msa_paths_by_chain_id.B   path/to/msa_B.a3m.gz
```

## RF3 Outputs

Inference produces:
- `{name}_model_{i}.cif.gz`: Predicted structures (gzipped mmCIF, one per diffusion seed)
- `{name}_metrics.csv`: Overall confidence metrics (pTM, ipTM, pLDDT, etc.)
- `{name}.score`: Detailed per-residue confidence scores

All CIF outputs can be directly opened in PyMol or parsed with AtomWorks.

## Testing

RF3 tests are located in `releases/rf3/tests/`, while shared tests are in the root `tests/` directory.

- **RF3 Regression tests**: Compare inference outputs against frozen baselines
  - Location: `releases/rf3/tests/test_inference_regression.py`
  - Test data: `releases/rf3/tests/data/` (mini examples and baseline predictions)
  - Run from: `releases/rf3/` directory
- **RF3 Confidence tests**: Test confidence output writing
  - Location: `releases/rf3/tests/test_write_confidence.py`
- **Shared Unit tests**: Test shared utilities across models
  - Location: Root `tests/` directory
  - `test_metrics.py`: Shared metrics
  - `test_weight_loading.py`: Weight loading utilities
  - `test_torch_utils.py`: Torch utility functions
- **GPU tests**: Marked with `@pytest.mark.gpu` decorator
- **Test execution**: Navigate to appropriate directory before running pytest

## Git Workflow

- **Main branch**: `trunk` (use for PRs, **not** `main`)
- Current refactoring branch: `refactor/rf3-lab`
- The repository is actively being refactored; expect API changes

## Common Development Patterns

### Adding New Metrics
1. **RF3-specific**: Implement in `releases/rf3/src/rf3/metrics/`
2. **Shared across models**: Implement in `src/modelhub/metrics/`
3. Inherit from `BaseMetric` in `src/modelhub/metrics/base.py`
4. Register in appropriate config file (`releases/rf3/configs/trainer/metrics/`)
5. Add tests:
   - RF3-specific tests in `releases/rf3/tests/`
   - Shared tests in root `tests/test_metrics.py`

### Adding New Callbacks
1. **RF3-specific**: Implement in `releases/rf3/src/rf3/callbacks/`
2. **Shared across models**: Implement in `src/modelhub/callbacks/`
3. Inherit from `BaseCallback` in `src/modelhub/callbacks/base.py`
4. Register in `releases/rf3/configs/callbacks/`
5. Hook into training loop via callback methods (`on_train_batch_end`, etc.)

### Modifying Model Architecture
1. Edit modules in `releases/rf3/src/rf3/model/`
2. Update corresponding configs in `releases/rf3/configs/model/`
3. Verify with regression tests from `releases/rf3/`:
   ```bash
   cd releases/rf3/
   pytest tests/test_inference_regression.py
   ```
4. Check weight loading from root:
   ```bash
   pytest tests/test_weight_loading.py
   ```

### Working with Hydra Configs
```bash
# Override nested config values
python script.py model.c_s=512 model.c_z=256

# Override list items (must quote)
python script.py inputs='[file1.cif, file2.json]'

# Change config group defaults
python script.py inference_engine=rf3 trainer=ddp

# Compose multiple configs
defaults:
  - trainer: rf3
  - model: rf3
  - datasets: pdb_and_distillation
  - _self_
```

## Performance Tips

- **Early stopping**: Set `early_stopping_plddt_threshold=0.5` to skip low-confidence predictions (10-20x faster)
- **Batch processing**: Use multiple inputs in single command to amortize startup cost
- **Diffusion steps**: Reduce `num_steps=50` (from default 200) for 2x speedup with minimal quality loss
- **Recycling**: Default `n_recycles=10`; reduce for faster inference

## Troubleshooting

### Import Errors
- Ensure virtual environment is activated: `source .venv/bin/activate`
- Check you're in correct directory:
  - For RF3 tests/inference/training: `cd releases/rf3/`
  - For shared tests: Stay in repository root
- The `rf3` CLI command (via `pyproject.toml`) should work from any directory once installed

### Missing Data
- Set `PDB_MIRROR_PATH` and `CCD_MIRROR_PATH` in `.env`
- For MSA: Set `LOCAL_MSA_DIRS` or use `raise_if_missing_msa_for_protein_of_length_n=0` to require MSAs

### CUDA/GPU Issues
- Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`
- Set precision: `torch.set_float32_matmul_precision("medium")`

### Hydra Configuration Errors
- Use `=` not `--` for arguments: `inputs='file.cif'` not `--inputs file.cif`
- Quote lists and strings: `inputs='[file1, file2]'`
- For detailed errors, use Python directly: `python src/modelhub/inference.py ...`

## Docstring Guidelines

Follow Google-style docstrings with Sphinx optimization. These comprehensive guidelines ensure consistent, high-quality documentation across the codebase.

### Primary Goals
- **Concise**: Keep docstrings as short as possible while being clear
- **No redundancy**: Don't repeat function/class names, types when annotated, or obvious behavior
- **Sphinx-first**: Prefer reStructuredText roles/directives that render beautifully
- **Google section headers**: Use standard section names with colons (e.g., `Args:`), not underlined headings

### When to Include Sections
- **Args**: Include only if non-trivial or non-obvious. Omit types when PEP 484 annotations are present
- **Returns**:
  - Omit if the function returns `None`
  - Omit if the summary sentence fully describes the return
  - Otherwise include, using rST for clarity (including literal blocks when helpful)
- **Yields**: Use instead of Returns for generators
- **Raises**: Include only unusual, explicitly raised exceptions that matter to users
- **Examples**: Strongly encouraged when usage isn't obvious. See "Examples formatting" below
- **References**: Include when citing standards, papers, or external docs; also when adding rST link targets
- **Todo**: Allowed; requires the Sphinx `sphinx.ext.todo` extension
- For classes, put argument documentation in `__init__`, not in the class docstring; the class docstring is a high-level overview

### General Formatting Rules
- **One-line summary**: Imperative, single sentence, ends with a period
- **Second paragraph**: Optional short elaboration only if genuinely useful
- **Inline code**: Use double backticks for identifiers and literals, e.g., ``"same"`` or ``pathlib.Path``
- **Cross-references**: Prefer semantic roles (see "Cross-referencing rules" below)
- **Line length**: Wrap naturally; don't force hard 79-char wraps if it harms readability
- **Defaults**: Use "Defaults to X." at end of the parameter description
- **None vs Optional**: Prefer "Defaults to None."; avoid repeating "Optional" if type hints already indicate optionality

### Section Ordering (Typical)
1. Summary
2. Optional elaboration
3. Args
4. Returns or Yields
5. Raises
6. Notes / Warnings (only if needed)
7. Examples
8. References
9. See Also
10. Todo
11. Attributes (for modules/classes only when helpful; not for `__init__` args)

### Args Formatting
Use the Google format; types omitted if PEP 484 present. Keep descriptions concise.

Example with type annotations:
```python
def fn(a: int, path: str | None = None):
    """Do X.

    Args:
      a: Number of items to process.
      path: Optional file path. Defaults to ``None``.
    """
```

Without type annotations:
```python
def fn(a, path=None):
    """Do X.

    Args:
      a (int): Number of items to process.
      path (str, optional): Optional file path. Defaults to ``None``.
    """
```

### Returns / Yields Formatting
Omit if returning `None` or already clear from the summary. Otherwise, short text; use rST formatting when useful.

Examples:
```python
def compute() -> dict[str, int]:
    """Compute counts.

    Returns:
      Mapping of names to counts.

      The ``Returns`` section supports any reStructuredText formatting,
      including literal blocks::

          {
              'param1': 1,
              'param2': 2
          }
    """
```

```python
def stream(n: int):
    """Yield integers up to ``n``.

    Yields:
      int: Next value in ``range(n)``.
    """
```

### Raises Formatting
Include only unusual, explicit exceptions that users should anticipate.
```python
Raises:
  ValueError: If ``threshold`` is negative.
```

### Examples Formatting
- Use the `Examples:` section
- Prefer doctest-style for simple usage with expected output
- Use `.. code-block:: python` for multi-line, setup-heavy, or non-doctest examples
- Precede each code block with a short sentence describing the case
- Keep examples minimal; 1–3 cases is typical. Avoid redundant examples
- Don't mix doctest and code-block in the same example unless clearly justified

Doctest example:
```python
def square(x: int) -> int:
    """Return the square of ``x``.

    Examples:
      >>> square(3)
      9
    """
```

Multi-case using code blocks:
```python
def from_selection_str(s: str):
    """Create an ``AtomSelection`` from a selection string.

    Examples:
      Select CA at chain A, residue 1::

        AtomSelection.from_selection_str("A/ALA/1/CA")

      Select CB in any chain at any residue::

        AtomSelection.from_selection_str("*/ALA/*/CB")
    """
```

Or with an explicit directive:
```python
Examples:
  Basic usage:

  .. code-block:: python

     result = fn("input")
     print(result)
```

### References Formatting
Use the `References:` section to:
- Cite external resources (standards, papers, docs)
- Define hyperlink targets (label definitions) for clean inline linking in the docstring
- Place external link definitions directly in the `References:` section or right after it

Examples:
```python
def parse():
    """Parse input.

    References:
      * `Google Python Style Guide`_
      * `PEP 484`_

      .. _Google Python Style Guide: http://google.github.io/styleguide/pyguide.html
      .. _PEP 484: https://www.python.org/dev/peps/pep-0484/
    """
```

### Cross-referencing Rules (Use Liberally)
Prefer semantic roles over plain backticks for important code entities:
- **Functions**: `:py:func:`package.module.fn`` (use `~` to shorten)
- **Classes**: `:py:class:`package.module.Class``; methods: `:py:meth:`~package.module.Class.method``
- **Modules**: `:py:mod:`package.module``; **Attributes**: `:py:attr:`~package.module.Class.attr``
- Use `:ref:` for intra-doc labels (sections/figures/tables)
- Use `:doc:` to link other documents by path
- Use `:any:` when unsure of the role; Sphinx will try to resolve it
- Prefer cross-references over repeating type info or behavior

### Notes, Warnings, and See Also
Keep these sparse; only when clarity is improved. Prefer Google sections (`Notes:`, `Warnings:`, `See Also:`). If a visual callout is needed, you may use admonitions (`.. note::`, `.. warning::`) inside the docstring; keep content concise.

Example:
```python
Notes:
  Uses :py:mod:`asyncio` for scheduling.

See Also:
  :py:class:`~package.Scheduler`, :py:func:`~package.schedule_task`
```

### Class and __init__ Rules
- **Class docstring**: High-level overview and purpose; avoid argument docs
- **__init__ docstring**: Document constructor parameters in `Args:`; follow the same formatting rules as functions

```python
class Cache:
    """In-memory cache with time-based eviction.

    See Also:
      :py:class:`~package.LRUCache`
    """

    def __init__(self, ttl: float, capacity: int = 1024):
        """Initialize the cache.

        Args:
          ttl: Time-to-live in seconds.
          capacity: Max entries. Defaults to ``1024``.
        """
```

### Module Docstrings
Short overview of module purpose and key public objects. Optionally an `Attributes:` section for significant module-level constants.

```python
"""Tools for model inference and orchestration.

Attributes:
  DEFAULT_TIMEOUT (float): Default timeout for network calls.

Todo:
  * Add support for streaming outputs.
  * Enable tracing hooks.
"""
```

### Style/Consistency Conventions
- Use double backticks for inline code/literals: ``None``, ``"text"``, ``/path``
- Use present tense and active voice
- Avoid restating obvious types when PEP 484 annotations exist
- Prefer short paragraphs and bulleted lists only when they clarify meaning
- Keep "Examples" and "References" well-formed and visually separated

### Quick Templates

Function (concise):
```python
def fn(x: int) -> int:
    """Return the square of ``x``.

    Examples:
      >>> fn(3)
      9
    """
```

Function (fuller):
```python
def load(path: str, *, strict: bool = False) -> dict:
    """Load a configuration from ``path``.

    Args:
      path: File path to load.
      strict: Validate schema strictly. Defaults to ``False``.

    Returns:
      Mapping representing the configuration.

    Raises:
      FileNotFoundError: If the file does not exist.

    Examples:
      Strict load with error handling:

      .. code-block:: python

         cfg = load("config.yaml", strict=True)

    References:
      * `YAML Spec`_

      .. _YAML Spec: https://yaml.org/spec/
    """
```

Class + __init__:
```python
class Runner:
    """Execute tasks with configurable concurrency."""

    def __init__(self, max_workers: int = 4):
        """Initialize the runner.

        Args:
          max_workers: Number of worker threads. Defaults to ``4``.
        """
```

Generator:
```python
def items():
    """Yield items from the queue.

    Yields:
      Item type: Next available item.
    """
```
