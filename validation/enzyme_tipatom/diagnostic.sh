#!/bin/bash
#SBATCH --partition=interactive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=enzyme_mask_check
#SBATCH --output=/mnt/home/jiechen7/other/foundry-atomworks3/validation/enzyme_tipatom/mask-%j.log
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 DISABLE_CUEQUIVARIANCE=1
REPO=/mnt/home/jiechen7/other/foundry-atomworks3
TASK_DIR="$REPO/validation/enzyme_tipatom"
export PYTHONPATH="$REPO/src:$REPO/models/rfd3/src:$REPO/models/rf3/src:$REPO/models/mpnn/src"
cd "$REPO"
.venv/bin/python "$TASK_DIR/run.py" --input "$TASK_DIR/input.json" --root "$TASK_DIR/results/diagnostic_aw3_mask" --arm padded_aw3_aw2mask --checkpoint /mnt/home/jiechen7/other/foundry-pr397-precision/audit_artifacts/rfd3_latest.ckpt --reference-mask-dir "$TASK_DIR/results/probe_aw2"
