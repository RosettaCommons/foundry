# RF3 Apptainer Containers

This directory contains two Apptainer definition files for different use cases.

## Container Options

### 1. `rf3-standalone.def` - Standalone Container
Contains a complete snapshot of the modelhub repository at build time. Use this for:
- Production deployments
- Reproducible inference with a fixed codebase
- Running on systems where you don't have the repository

### 2. `rf3-dev.def` - Development Container
Contains only Python dependencies (from `requirements.txt`). Use this for:
- Active development and testing
- Working with your local modelhub code
- Debugging and modifying RF3 code

**Note**: Generate `requirements.txt` first using `uv pip compile pyproject.toml -o requirements.txt` before building this container.

## Prerequisites

- Apptainer/Singularity installed
- NVIDIA GPU with CUDA 12.1+ support
- Sufficient disk space (~10GB per container)

## Building Containers

Build from the `models/rf3/` directory:

```bash
cd models/rf3/

# Build standalone container (includes modelhub snapshot)
apptainer build rf3-standalone.sif rf3-standalone.def

# Build development container (dependencies only)
apptainer build rf3-dev.sif rf3-dev.def

# Or build with sandbox for debugging
apptainer build --sandbox rf3_sandbox/ rf3-standalone.def
```

## Using the Standalone Container

The standalone container has the full repository baked in.

### Basic Inference

```bash
# Run inference on a single input
apptainer exec --nv rf3-standalone.sif rf3 fold inputs='input.json'

# Process CIF/PDB files
apptainer exec --nv rf3-standalone.sif rf3 fold inputs='structure.cif'

# Batch processing
apptainer exec --nv rf3-standalone.sif rf3 fold inputs='[file1.cif,file2.json,file3.pdb]'
```

### With Custom Weights

```bash
# Mount weights directory and specify checkpoint
apptainer exec --nv \
    --bind /path/to/weights:/weights \
    rf3-standalone.sif \
    rf3 fold inputs='input.json' ckpt_path='/weights/rf3_latest.pt'
```

## Using the Development Container

The development container requires mounting your local modelhub repository.

### Basic Usage

```bash
# Run with local modelhub repository
apptainer exec --nv \
    --bind /path/to/modelhub:/opt/modelhub \
    rf3-dev.sif \
    rf3 fold inputs='input.json'

# Example with actual path
apptainer exec --nv \
    --bind $PWD/../..:/opt/modelhub \
    rf3-dev.sif \
    rf3 fold inputs='input.json'
```
