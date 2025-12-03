# Running ModelForge at IPD

This guide covers how to run training, inference, and validation within the IPD environment using our shebang and apptainer infrastructure.

## Table of Contents

- [Overview](#overview)
- [The Shebang System](#the-shebang-system)
- [Running Inference](#running-inference)
- [Running Training](#running-training)
- [Running Validation](#running-validation)
- [Container Information](#container-information)
- [Debugging](#debugging)

## Overview

The IPD environment provides two complementary approaches for running models:

1. **Shebang System (Development)**: Executes Python scripts inside containers while allowing you to edit code on the host. Changes to your code are immediately reflected without rebuilding containers.

2. **Pre-built Containers (Production)**: Fully containerized execution using `apptainer exec` with code baked into the container.

For active development, the shebang approach is recommended as it provides the fastest iteration cycle.

## The Shebang System

### How It Works

The shebang system allows you to execute Python scripts directly (e.g., `./models/rf3/src/rf3/train.py`) without explicitly calling `apptainer exec`. Here's what happens behind the scenes:

1. **Entry Point Scripts** (train.py, validate.py, inference.py) include a special shebang line:
   ```bash
   #!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../../../../.ipd/shebang/rf3_exec.sh" "$0" "$@"'
   ```

2. **The rf3_exec.sh Script** (`.ipd/shebang/rf3_exec.sh`) then:
   - Locates the repository root by searching for `.project-root`
   - Sets up PYTHONPATH to include foundry, rf3, and atomworks
   - Finds the development container at `.ipd/apptainer/rf3-dev.sif`
   - Detects GPU support (uses `--nvccli` if available, falls back to `--nv`, or runs without GPU)
   - Executes your script inside the container with the repository bind-mounted

3. **Your Code Runs** inside the container but reads from the host filesystem, so any edits you make are immediately active.

### Usage

Simply make the script executable and run it directly:

```bash
# Run inference
./models/rf3/src/rf3/inference.py inputs=example.json

# Run training
./models/rf3/src/rf3/train.py experiment=my_experiment

# Run validation
./models/rf3/src/rf3/validate.py experiment=my_experiment
```

You can also use the `rf3` CLI command if the package is installed in your environment, but the shebang approach ensures you're always using the container environment.

## Running Inference

### Method 1: Development (Shebang - Recommended)

Execute the inference script directly:

```bash
./models/rf3/src/rf3/inference.py \
  inputs=./path/to/input.json \
  out_dir=./output \
  ckpt_path=/net/software/containers/versions/modelhub_inference/ckpts/rf3-w-conf-run10-ep922-remapped.ckpt \
  diffusion_batch_size=5 \
  n_recycles=10 \
  num_steps=200
```

> Inference arguments for RF3 are provided [here](models/rf3/README.md).

### Method 2: Production (Pre-built Container)

For production inference or when you don't need to modify code:

```bash
apptainer exec --nvccli \
  /net/software/containers/versions/modelhub_inference/rf3.sif \
  rf3 fold \
  inputs=./path/to/input.json \
  out_dir=./output \
  ckpt_path=/net/software/containers/versions/modelhub_inference/ckpts/rf3-w-conf-run10-ep922-remapped.ckpt \
  diffusion_batch_size=1 \
  one_model_per_file=True \
  annotate_b_factor_with_plddt=True
```

This approach is ideal for:
- Large-scale inference jobs (e.g., distillation, design campaigns)
- When you want to ensure exact reproducibility

## Running Training

### Method 1: Direct Execution (Development)

For quick training runs or debugging:

```bash
./models/rf3/src/rf3/train.py \
  experiment=quick-rf3 \
  debug=default
```

The experiment name determines which config is loaded from `models/rf3/configs/experiment/`.

### Method 2: SLURM Job Submission (Recommended for Production Training)

For multi-GPU or multi-node training, use SLURM:

```bash
# Submit a training job
sbatch .ipd/slurm/launch_rf3.sh
```

#### Customizing SLURM Jobs

Edit `.ipd/slurm/launch_rf3.sh` or create a custom launch script:

```bash
#!/bin/bash
#SBATCH -p gpu-train                    # Partition
#SBATCH --nodes 1                       # Number of nodes
#SBATCH --gres=gpu:l40:8               # GPUs per node
#SBATCH --ntasks-per-node 8            # Tasks per node (usually = GPUs)
#SBATCH -c 4                           # CPUs per task
#SBATCH --mem=512g                     # Memory per node
#SBATCH -t 1-00:00:00                  # Time limit (1 day)
#SBATCH -J my_experiment               # Job name

# Environment setup
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1              # For L40 GPUs (no NVLink)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Calculate gradient accumulation
EFFECTIVE_BATCH_SIZE=32                 # Total effective batch size
BATCH_SIZE_PER_GPU=1                    # Batch size per GPU
DEVICES_PER_NODE=$SLURM_GPUS_ON_NODE
TOTAL_GPUS=$((SLURM_NNODES * DEVICES_PER_NODE))
GRAD_ACCUM_STEPS=$((EFFECTIVE_BATCH_SIZE / (BATCH_SIZE_PER_GPU * TOTAL_GPUS)))

# Launch training
srun --kill-on-bad-exit \
  ../../models/rf3/src/rf3/train.py \
  experiment=$SLURM_JOB_NAME \
  ++trainer.devices_per_node=$DEVICES_PER_NODE \
  ++trainer.num_nodes=$SLURM_NNODES \
  ++trainer.grad_accum_steps=$GRAD_ACCUM_STEPS
```

#### Key SLURM Configuration Options

**Resource Allocation:**
- `-p gpu-train`: GPU training partition
- `--nodes`: Number of nodes (1 for single-node, >1 for distributed)
- `--gres=gpu:l40:8`: GPU type and count 
- `--ntasks-per-node`: Number of processes per node (typically equals GPU count)
- `-c`: CPU cores per task
- `--mem`: Memory per node

**Time and Naming:**
- `-t`: Time limit (format: days-hours:minutes:seconds)
- `-J`: Job name (used as experiment name if you use `$SLURM_JOB_NAME`)

**Environment Variables:**
- `OMP_NUM_THREADS`: OpenMP threads (usually set to CPUs per task)
- `NCCL_P2P_DISABLE=1`: Disable peer-to-peer for GPUs without NVLink (which we don't have at the IPD)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: Enable expandable CUDA memory


## Running Validation

Validate a trained model or checkpoint:

```bash
# Using shebang (development)
./models/rf3/src/rf3/validate.py \
  experiment=my_experiment \
  ckpt_path=/path/to/checkpoint.ckpt
```

Validation datasets are configured in `models/rf3/configs/datasets/val/`.

## Debugging

### Remote Debugging with debugpy

The shebang system supports remote debugging:

```bash
# Set DEBUG_PORT environment variable
DEBUG_PORT=5678 ./models/rf3/src/rf3/train.py experiment=quick-rf3
```

This will:
1. Launch debugpy server on the specified port
2. Wait for a debugger to attach
3. Print connection instructions

Then attach your IDE debugger to `localhost:5678`.

### Viewing Logs

Training logs are saved to the experiment directory. For SLURM jobs:

```bash
# View real-time logs
tail -f slurm-<job_id>.out

# Check job status
squeue -u $USER

# View completed job info
sacct -j <job_id> --format=JobID,JobName,Partition,State,ExitCode,Elapsed
```