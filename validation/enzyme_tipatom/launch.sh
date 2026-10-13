#!/bin/bash
#SBATCH --partition=interactive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=enzyme_tipatom
#SBATCH --output=/mnt/home/jiechen7/other/foundry-atomworks3/validation/enzyme_tipatom/job-%j.log
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 DISABLE_CUEQUIVARIANCE=1
TASK_DIR=/mnt/home/jiechen7/other/foundry-atomworks3/validation/enzyme_tipatom
AW2_PY=/mnt/home/jiechen7/other/foundry-pr397-precision/.venv/bin/python
AW3_PY=/mnt/home/jiechen7/other/foundry-atomworks3/.venv/bin/python
POOL_SOURCE=/mnt/home/jiechen7/other/foundry-deterministic-pooling/src/foundry/utils/torch.py
CKPT=/mnt/home/jiechen7/other/foundry-pr397-precision/audit_artifacts/rfd3_latest.ckpt
# A minimal preflight gates the pilot within this one short allocation.
SOURCE_DIR=/mnt/home/jiechen7/other/foundry-pr397-base
export PYTHONPATH="$SOURCE_DIR/src:$SOURCE_DIR/models/rfd3/src:$SOURCE_DIR/models/rf3/src:$SOURCE_DIR/models/mpnn/src"
cd "$SOURCE_DIR"
"$AW2_PY" "$TASK_DIR/run.py" --input "$TASK_DIR/input.json" --root "$TASK_DIR/results/smoke" --arm native_aw2 --checkpoint "$CKPT" --seeds 11 --batch-size 1
for arm in ${ENZYME_ARMS:-native_aw2 fixed_aw2 padded_aw2 padded_aw3}; do
 RUN_PY="$AW2_PY"
 POOL_ARGS=(--pooling-source "$POOL_SOURCE")
 case "$arm" in
  native_aw2) SOURCE_DIR=/mnt/home/jiechen7/other/foundry-pr397-base; POOL_ARGS=();;
  fixed_aw2) SOURCE_DIR=/mnt/home/jiechen7/other/foundry-pr397-base;;
  padded_aw2) SOURCE_DIR=/mnt/home/jiechen7/other/foundry-pr397-precision;;
  padded_aw3) SOURCE_DIR=/mnt/home/jiechen7/other/foundry-atomworks3; RUN_PY="$AW3_PY";;
 esac
 export PYTHONPATH="$SOURCE_DIR/src:$SOURCE_DIR/models/rfd3/src:$SOURCE_DIR/models/rf3/src:$SOURCE_DIR/models/mpnn/src"
 cd "$SOURCE_DIR"
 "$RUN_PY" "$TASK_DIR/run.py" --input "$TASK_DIR/input.json" --root "$TASK_DIR/results/$arm" --arm "$arm" --checkpoint "$CKPT" "${POOL_ARGS[@]}" --seeds ${ENZYME_SEEDS:-11 22 33 44} --batch-size "${ENZYME_BATCH:-2}"
done
