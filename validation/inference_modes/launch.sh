#!/bin/bash
#SBATCH --partition=interactive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --job-name=rfd3_modes
#SBATCH --output=/mnt/home/jiechen7/other/foundry-atomworks3/validation/inference_modes/job-%A_%a.log
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 DISABLE_CUEQUIVARIANCE=1
cd /mnt/home/jiechen7/other/foundry-atomworks3
TASK_DIR="$PWD/validation/inference_modes"
CKPT=/mnt/home/jiechen7/other/foundry-pr397-precision/audit_artifacts/rfd3_latest.ckpt
if [ "$SLURM_ARRAY_TASK_ID" = 0 ]; then
 .venv/bin/pytest -q models/rfd3/tests/test_batched_pooling_precision.py models/rfd3/tests/test_inference_modes.py models/rfd3/tests/test_rfd3_batch_compile.py > "$TASK_DIR/gpu-tests.log" 2>&1
 mode=fast
 arm=padded_fast
else
 mode=legacy
 arm=legacy
fi
.venv/bin/python validation/enzyme_tipatom/run.py --arm "$arm" --inference-mode "$mode" --input "$TASK_DIR/results/input.json" --root "$TASK_DIR/results/$mode" --checkpoint "$CKPT" --seeds 11 22 33 44 55 66 77 88 99 110 121 132 143 154 165 176 187 198 209 220 231 242 253 264 275 286 297 308 319 330 341 352 --batch-size 2
