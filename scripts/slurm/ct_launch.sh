#!/bin/bash
#SBATCH -p gpu-train
#SBATCH --nodes 1
#SBATCH --gres=gpu:l40:8
#SBATCH --ntasks-per-node 8
#SBATCH -c 4
#SBATCH --mem=256G
#SBATCH --exclude=gpu83,gpu78,gpu79,gpu82,gpu73,gpu74,gpu76,gpu81,gpu75,gpu72,gpu80
#SBATCH -t 1-00:00:00
#SBATCH -J crosstalk
#SBATCH -o slurm_logs/%x_%j.log
#SBATCH -e slurm_logs/%x_%j.log
#SBATCH --no-kill=off

### Excluded Nodes:
### #SBATCH --exclude=gpu83,gpu78,gpu79,gpu82,gpu73,gpu74,gpu76,gpu81,gpu75,gpu72,gpu80
# gpu78 - device >=0 && device < num_gpus, Internal Assert Failed (3/12) (multi-node only)
# gpu83 - device >=0 && device < num_gpus, Internal Assert Failed (3/12) (multi-node only)
# gpu79 - device >=0 && device < num_gpus, Internal Assert Failed (3/12) (multi-node only)
# gpu82 - device >=0 && device < num_gpus, Internal Assert Failed (3/12) (multi-node only)
# gpu73 - device >=0 && device < num_gpus, Internal Assert Failed (3/12) (multi-node only)
# gpu74 - device >=0 && device < num_gpus, Internal Assert Failed (3/12) (multi-node only)
# gpu81 - device >=0 && device < num_gpus, Internal Assert Failed (3/12) (multi-node only)
# gpu75 - device >=0 && device < num_gpus, Internal Assert Failed (3/12) (multi-node only)
# gpu76 - flagged by Bo, not confirmed by me

### To call this script run:  `sbatch launch.sh` from this directory
### For reference, see the Lightning Fabric + SLURM guide: https://lightning.ai/docs/fabric/stable/guide/multi_node/slurm.html

# (In case we're still running in debug mode)
unset DEBUG_PORT

# (SLURM setup, ensuring we have a unique port per job, and setting the master address to Rank 0)
export MASTER_PORT=$((1024 + RANDOM % 64512))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

### Set custom paths
# WARNING: You will need to update these paths to match your local setup
# ... cifutils and datahub
# export PYTHONPATH="/home/ncorley/projects/datahub/src:/home/ncorley/projects/cifutils/src:/home/ncorley/projects/modelhub/src"
# export PYTHONPATH="/home/jbutch/Projects/HT25/af3/modelhub_refactor/src:/home/jbutch/software/stable/cifutils/src:/home/jbutch/Projects/HT25/af3/datahub/src"
export PYTHONPATH="/home/jfunk21/projects/datahub/src:/home/jfunk21/projects/cifutils/src:/home/jfunk21/projects/modelhub/src"

# ... project path (if not using root src/modelhub)
export PROJECT_PATH="/home/jfunk21/projects/modelhub/projects/aa_design"
export HYDRA_FULL_ERROR=1

# ... cache directory for Triton kernels (e.g., DeepSpeed4Science fused kernels)
# export TRITON_CACHE_DIR="/home/ncorley/.triton" # Change this to a directory with write permissions
export TRITON_CACHE_DIR="/home/jfunk21/.triton"

### Environment flags

# Debugging flags (optional)
export NCCL_DEBUG=INFO # NCCL internal debugging
export PYTHONFAULTHANDLER=1 # Catches Python core dumps (e.g., segmentation faults)

# Expand CUDA memory
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Turn off NVLink (L40 do not have NVLink)
export NCCL_P2P_DISABLE=1

# OPENMP and OPENBLAS optimizations
# https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#utilize-openmp
# NOTE: Must be optimized per-system; see: https://github.com/pytorch/pytorch/blob/65e6194aeb3269a182cfe2c05c122159da12770f/torch/distributed/run.py#L596-L608
export OMP_NUM_THREADS=4    
export OPENBLAS_NUM_THREADS=4

#######################################################################################################
### WARNING: The command below is just an example. It will fail if you don't update the experiment  ###
###          config in the command below. Please adapt according to your target experiment          ###
#######################################################################################################

### Set the effective batch size
EFFECTIVE_BATCH_SIZE=16

### Compose the training script
DEVICES_PER_NODE=${SLURM_NTASKS_PER_NODE:-8}  # Default to 8 if not set
echo "Running on $SLURM_NNODES nodes with $DEVICES_PER_NODE tasks per node"

### Calculate grad_accum_steps
GRAD_ACCUM_STEPS=$((EFFECTIVE_BATCH_SIZE / (DEVICES_PER_NODE * SLURM_NNODES)))
echo "Grad Accumulation Steps: $GRAD_ACCUM_STEPS"

echo "Experiment: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_NODELIST"
echo "Number of GPUs per node: $SLURM_GPUS_ON_NODE"

command="srun --kill-on-bad-exit ../../src/modelhub/train.py \
    experiment=$SLURM_JOB_NAME \
    ++trainer.devices_per_node=$DEVICES_PER_NODE \
    ++trainer.num_nodes=$SLURM_NNODES \
    ++trainer.grad_accum_steps=$GRAD_ACCUM_STEPS"

echo -e "command\t$command"

# Let 'er rip
$command