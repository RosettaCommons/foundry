#!/bin/bash
#SBATCH --partition=hpc-mid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=foundry_quality
#SBATCH --output=/mnt/home/jiechen7/other/foundry-quality-validation/validation/quality/pilot-%j.log
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
STUDY_DIR=/mnt/home/jiechen7/other/foundry-quality-validation/validation/quality
AW2_PY=/mnt/home/jiechen7/other/foundry-pr397-precision/.venv/bin/python
AW3_PY=/mnt/home/jiechen7/other/foundry-atomworks3/.venv/bin/python
POOL_SOURCE=/mnt/home/jiechen7/other/foundry-deterministic-pooling/src/foundry/utils/torch.py
RFD3_CKPT=/mnt/home/jiechen7/other/foundry-pr397-precision/audit_artifacts/rfd3_latest.ckpt
for arm in native_aw2 fixed_aw2 padded_aw2 padded_aw3; do
 RUN_PY="$AW2_PY"
 POOL_ARGS=(--pooling-source "$POOL_SOURCE")
 case "$arm" in
  native_aw2) SOURCE_DIR=/mnt/home/jiechen7/other/foundry-pr397-base; POOL_ARGS=();;
  fixed_aw2) SOURCE_DIR=/mnt/home/jiechen7/other/foundry-pr397-base;;
  padded_aw2) SOURCE_DIR=/mnt/home/jiechen7/other/foundry-pr397-precision;;
  padded_aw3) SOURCE_DIR=/mnt/home/jiechen7/other/foundry-quality-validation; RUN_PY="$AW3_PY";;
 esac
 export PYTHONPATH="$SOURCE_DIR/src:$SOURCE_DIR/models/rfd3/src:$SOURCE_DIR/models/rf3/src:$SOURCE_DIR/models/mpnn/src"
 cd "$SOURCE_DIR"
 "$RUN_PY" "$STUDY_DIR/driver.py" generate --root "$STUDY_DIR/results" --arm "$arm" --checkpoint "$RFD3_CKPT" "${POOL_ARGS[@]}"
done
SOURCE_DIR=/mnt/home/jiechen7/other/foundry-quality-validation
export PYTHONPATH="$SOURCE_DIR/src:$SOURCE_DIR/models/rfd3/src:$SOURCE_DIR/models/rf3/src:$SOURCE_DIR/models/mpnn/src"
cd "$SOURCE_DIR"
"$AW3_PY" "$STUDY_DIR/driver.py" mpnn --root "$STUDY_DIR/results" --checkpoint /mnt/home/jiechen7/other/foundry-atomworks3/.validation/checkpoints/proteinmpnn_v_48_020.pt --pooling-source "$POOL_SOURCE"
"$AW3_PY" "$STUDY_DIR/driver.py" fold --root "$STUDY_DIR/results" --checkpoint /mnt/home/jiechen7/other/foundry-atomworks3/.validation/checkpoints/rf3_foundry_01_24_latest_remapped.ckpt --pooling-source "$POOL_SOURCE"
touch "$STUDY_DIR/results/COMPLETE"
