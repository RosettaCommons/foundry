# `molecore_foundry` (uv) — cross-platform default bio environment

This directory defines a **portable** environment spec you can use on:

- **macOS (CPU)**: day-to-day bio + notebooks + visualization helpers
- **Linux (GPU)**: same base env, with an **opt-in** CUDA PyTorch install path

It is designed to work well with the Foundry repo at:
- `/Users/ariel/dev/molCore/foundry`

---

## Prerequisites

- Install `uv` (one-time).
- Standardize on **Python 3.12** (matches Foundry’s `requires-python = ">=3.12"`).

---

## Create the environment

Choose where you want the env to live. Recommended: `~/.venvs/molecore_foundry`.

```bash
uv python install 3.12
uv venv ~/.venvs/molecore_foundry --python 3.12
source ~/.venvs/molecore_foundry/bin/activate
```

Install the base utilities:

```bash
uv pip install -e /Users/ariel/dev/molCore/foundry/tools/envs/molecore_foundry
```

This base install now includes **AtomWorks + Biotite + Torch** by default.

---

## Install Foundry into this environment (optional)

### Option A: editable install (when developing locally)

```bash
uv pip install -e "/Users/ariel/dev/molCore/foundry[all]"
```

### Option B: pip install released package

```bash
uv pip install "rc-foundry[all]"
```

---

## Linux GPU: CUDA PyTorch (opt-in)

### Why opt-in?

Package managers (pip/uv) **cannot reliably auto-detect** your server’s driver/CUDA runtime and choose the “right” CUDA wheel at resolution time. The robust pattern is:

- install the env normally (CPU-safe everywhere)
- on GPU machines, explicitly install CUDA-enabled PyTorch

### Recommended default

Prefer **cu124** (or **cu121** if your fleet is older). Your driver reports CUDA 12.9 capability, so it should run cu124/cu121 wheels.

Example (choose one):

```bash
# cu124 (recommended)
uv pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# or cu121
uv pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

If you want a fully automatic “pick a CUDA wheel based on `nvidia-smi`”, that generally requires a small wrapper script (because the package resolver itself doesn’t see runtime driver state). If you want that, we can add a `scripts/install_torch_cuda.py` later.

Sanity check:

```bash
python -c "import torch; print('torch', torch.__version__); print('cuda?', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

---

## PyRosetta (not available for Python 3.12)

**Status: ❌ Not compatible with Python 3.12**

PyRosetta builds for Python 3.12 are not yet available from Rosetta Commons. The environment includes `pyrosetta-installer` for when compatibility is added.

### Future availability
When PyRosetta supports Python 3.12, you can install it:

```bash
python -c "import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()"
```

### Alternative approaches
- Use Python 3.11 for PyRosetta-specific work
- Consider web-based alternatives like Rosetta Online
- Use the ColabFold/RoseTTAFold servers for structure prediction

---

## AtomWorks (recommended)

Foundry relies on AtomWorks; it’s also very useful as a “daily driver” for structure IO/cleanup.
See [`RosettaCommons/atomworks`](https://github.com/RosettaCommons/atomworks).

- Installed by default (IO-only; no torch requirement):

```bash
uv pip install -e /Users/ariel/dev/molCore/foundry/tools/envs/molecore_foundry
```

- ML-enabled (pulls torch):

```bash
uv pip install "molecore-foundry-env[atomworks_ml]"
```

---

## PyMOL (installed via wrapper)

**Status: ✅ Python wrapper included**

The environment includes `pymolPy3` - a Python wrapper that allows scripting PyMOL when PyMOL itself is installed separately.

### Setup
1. **Install PyMOL system-wide** (outside this environment):
   - macOS: `brew install pymol` (Homebrew)
   - Linux: `conda install -c conda-forge pymol-bundle` or system packages

2. **Use the wrapper** for Python scripting:
   ```python
   import pymolPy3
   pm = pymolPy3.pymolPy3()
   pm("load my_structure.pdb")
   ```

### Alternative: Pure Python visualization
For notebook-friendly 3D visualization without external PyMOL:

```bash
uv pip install py3Dmol nglview
```

---

## Agentic workflows + Modal (optional)

If you want agentic scripting + cloud execution:

- Agentic tooling (SDKs + helpers):

```bash
uv pip install "molecore-foundry-env[agentic]"
```

- Modal client (serverless GPU/CPU from Python):

```bash
uv pip install "molecore-foundry-env[modal]"
```

Modal reference: [`modal` on PyPI](https://pypi.org/pypi/modal)

---

## What’s included?

See:
- `PACKAGES.md` for a curated list (from your old env + Foundry + suggestions)
- `pyproject.toml` for the actual dependency groups


