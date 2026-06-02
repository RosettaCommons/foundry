#!/bin/bash
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem 32gb
#SBATCH --time 00:59:00
#SBATCH --job-name="rfd3"

source ~/.bashrc
conda activate rc-foundry

# Set variables
INFILE="./pdf1.yaml" #or .json
OUTDIR="./diffusion_outs"

# Run RFdiffusion3
rfd3 design \
  out_dir="$OUTDIR" \
  inputs="$INFILE" \
  n_batches=1 \
  diffusion_batch_size=1 \
  dump_trajectories=1 # OPTIONAL, FOR VISUALIZATION PURPOSES

