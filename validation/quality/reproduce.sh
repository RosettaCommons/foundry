#!/usr/bin/env bash
# Replay the registered pilot in an existing GPU allocation. All paths explicit.
set -euo pipefail
: "${AW2_PY:?Python environment containing AtomWorks 2.2.1}"
: "${AW3_PY:?Python environment containing AtomWorks 3.0.1}"
: "${NATIVE_SOURCE:?Checkout of 8bffbcb9bec7eb8c3be0813f87d33c98faa0f5d2}"
: "${PADDED_SOURCE:?Checkout of 754d1caae5a74b35a3bfcf68169e459474b7dbc4}"
: "${AW3_SOURCE:?Checkout of d0474f0f467793d7dd7151399499470fe9475efc}"
: "${POOL_SOURCE:?Path to src/foundry/utils/torch.py from aa97ded}"
: "${RFD3_CKPT:?Path to rfd3_latest.ckpt}"
: "${MPNN_CKPT:?Path to proteinmpnn_v_48_020.pt}"
: "${RF3_CKPT:?Path to rf3_foundry_01_24_latest_remapped.ckpt}"
: "${RESULTS_DIR:?Absolute path to a separate replay output directory}"
STUDY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
for arm in native_aw2 fixed_aw2 padded_aw2 padded_aw3; do
  RUN_PY="$AW2_PY"
  POOL_ARGS=(--pooling-source "$POOL_SOURCE")
  case "$arm" in
    native_aw2) SOURCE_DIR="$NATIVE_SOURCE"; POOL_ARGS=();;
    fixed_aw2) SOURCE_DIR="$NATIVE_SOURCE";;
    padded_aw2) SOURCE_DIR="$PADDED_SOURCE";;
    padded_aw3) SOURCE_DIR="$AW3_SOURCE"; RUN_PY="$AW3_PY";;
  esac
  export PYTHONPATH="$SOURCE_DIR/src:$SOURCE_DIR/models/rfd3/src:$SOURCE_DIR/models/rf3/src:$SOURCE_DIR/models/mpnn/src"
  cd -- "$SOURCE_DIR"
  "$RUN_PY" "$STUDY_DIR/driver.py" generate --root "$RESULTS_DIR" --arm "$arm" --checkpoint "$RFD3_CKPT" "${POOL_ARGS[@]}"
done
export PYTHONPATH="$AW3_SOURCE/src:$AW3_SOURCE/models/rfd3/src:$AW3_SOURCE/models/rf3/src:$AW3_SOURCE/models/mpnn/src"
cd -- "$AW3_SOURCE"
"$AW3_PY" "$STUDY_DIR/driver.py" mpnn --root "$RESULTS_DIR" --checkpoint "$MPNN_CKPT" --pooling-source "$POOL_SOURCE"
"$AW3_PY" "$STUDY_DIR/driver.py" fold --root "$RESULTS_DIR" --checkpoint "$RF3_CKPT" --pooling-source "$POOL_SOURCE"
touch "$RESULTS_DIR/COMPLETE"
"$AW3_PY" "$STUDY_DIR/audit.py" "$RESULTS_DIR"
"$AW3_PY" "$STUDY_DIR/score.py" "$RESULTS_DIR"
