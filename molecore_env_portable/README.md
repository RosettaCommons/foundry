# molecore_foundry Portable Environment

This is a portable Python environment specification designed for biomolecular modeling workflows, optimized for both macOS and Linux platforms.

## What's Included

### Core Dependencies (Always Installed)
- **atomworks** (2.2.0) - Unified framework for biomolecular structure processing
- **biotite** (1.4.0) - Computational structural biology toolkit
- **torch** (2.9.1) - Machine learning framework
- **numpy** (2.2.6) - Scientific computing
- **pandas** (2.3.3) - Data analysis
- **scipy** (1.16.3) - Scientific computing
- **requests** - HTTP library
- **python-dotenv** - Environment variable management
- **pyyaml** - YAML parsing
- **ipykernel** - Jupyter notebook support
- **jupyterlab** - Advanced notebook interface
- **tqdm** - Progress bars

### Bio + Database Access (Recommended)
- **biopython** - Biological computation
- **rcsbsearchapi** - PDB database search
- **bioservices** - Biological web services
- **httpx** - Modern HTTP client
- **gget** - Sequence/structure fetching

### Visualization
- **matplotlib** - Plotting
- **seaborn** - Statistical visualization
- **py3dmol** - 3D molecular visualization
- **nglview** - Jupyter molecular viewer

### Agentic Workflows
- **openai** - OpenAI API client
- **anthropic** - Anthropic API client
- **groq** - Groq API client

### Cloud Computing
- **modal** - Serverless cloud platform

## Setup Instructions

### On macOS (CPU)
```bash
uv python install 3.12
uv venv molecore_foundry --python 3.12
source molecore_foundry/bin/activate
pip install -r requirements.txt
```

### On Linux (GPU - Optional CUDA)
```bash
uv python install 3.12
uv venv molecore_foundry --python 3.12
source molecore_foundry/bin/activate
pip install -r requirements.txt
# For CUDA support (optional):
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Verification
```bash
python -c "
import torch, atomworks, biotite, numpy as np, pandas as pd, scipy
import requests, rcsbsearchapi, openai, modal
print('✅ All packages imported successfully!')
print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
"
```

## Usage Notes

- **AtomWorks**: Requires PDB/CCD mirror paths for full functionality (set `PDB_MIRROR_PATH` and `CCD_MIRROR_PATH`)
- **PyTorch**: CPU version installed by default; upgrade to CUDA version on GPU machines
- **Modal**: Requires account setup and API keys for cloud deployment
- **PyRosetta**: Install separately using `pyrosetta-installer` if needed

## Environment Variables

Optional but recommended:
```bash
export PDB_MIRROR_PATH=/path/to/pdb/mirror
export CCD_MIRROR_PATH=/path/to/ccd/mirror
```

## File Structure
- `requirements.txt` - Complete package list for reproduction
- `README.md` - This documentation
