#!/bin/bash
#SBATCH --partition=<your_partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem 32gb
#SBATCH --time 00:59:00
#SBATCH --job-name="rfd3"

# Run RFdiffusion3
rfd3 design \
  out_dir="./pdf1_sm_binder_outputs" \
  inputs="./pdf1.yaml" \
  n_batches=1 \
  diffusion_batch_size=8 \
  dump_trajectories=1 # OPTIONAL, FOR VISUALIZATION PURPOSES

