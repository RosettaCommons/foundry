#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 cpu|mps rfd3|rf3|mpnn [model CLI arguments...]" >&2
  exit 2
fi
device="$1"
model="$2"
shift 2
case "$device" in cpu|mps) ;; *) echo "Device must be cpu or mps" >&2; exit 2;; esac
case "$model" in rfd3|rf3|mpnn) ;; *) echo "Model must be rfd3, rf3 or mpnn" >&2; exit 2;; esac
export FOUNDRY_DEVICE="$device"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
cd "$root"
exec "$root/.venv/bin/$model" "$@"
