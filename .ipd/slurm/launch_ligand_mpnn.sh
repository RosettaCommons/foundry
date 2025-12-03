#!/bin/bash
#SBATCH --job-name=foundry_ligand
#SBATCH --partition=gpu-train
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/train_ligand_%j.out
#SBATCH --error=logs/train_ligand_%j.err

set -a
source ../.env
set +a

srun ../../models/mpnn/src/mpnn/train.py ligand_mpnn