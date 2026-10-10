# Sourced by verify_unconditional.sh and verify_partial_diffusion.sh. Not executable on its own.
#
# A fixed `seed=` must give a fixed design. Each verify script runs the same RFD3 inference
# REPEATS times with the same seed (fresh out_dir, fresh process each time) and requires the
# designs to be bitwise identical; then runs once with a different seed and requires a
# different design, so a PASS cannot come from the comparison being blind.
#
# Requires a GPU, the RFD3 checkpoint, and an environment with foundry + rfd3 installed
# (`uv pip install -e '.[rfd3]'`). Knobs (environment variables):
#   SEED=2000  SEED2=2001   seeds for the repeat runs / the control run
#   REPEATS=3               identical runs per check (>= 2)
#   GPU=0                   CUDA_VISIBLE_DEVICES for every run
#   OUT_ROOT=<mktemp dir>   where runs and logs go
#   RFD3="rfd3"  PYTHON="python"   rfd3 CLI / interpreter (need biotite) to use
#   CKPT_PATH=<unset>       passed as ckpt_path= if set (default: foundry's `rfd3` checkpoint)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FOUNDRY_ROOT="$(cd "${HERE}/../../../.." && pwd)"

SEED="${SEED:-2000}"
SEED2="${SEED2:-2001}"
REPEATS="${REPEATS:-3}"
GPU="${GPU:-0}"
RFD3="${RFD3:-rfd3}"
PYTHON="${PYTHON:-python}"
OUT_ROOT="${OUT_ROOT:-$(mktemp -d -t rfd3_seed_verify.XXXXXX)}"
mkdir -p "${OUT_ROOT}"
echo "outputs and logs: ${OUT_ROOT}"

# run_design NAME SEED [hydra overrides...]  -> ${OUT_ROOT}/NAME  (one design, own process)
run_design() {
  local name="$1" seed="$2"
  shift 2
  local out="${OUT_ROOT}/${name}"
  local extra=()
  [[ -n "${CKPT_PATH:-}" ]] && extra+=("ckpt_path=${CKPT_PATH}")
  echo "--- ${name} (seed=${seed}, gpu=${GPU}) ---"
  if ! CUDA_VISIBLE_DEVICES="${GPU}" "${RFD3}" design \
      "out_dir=${out}" "n_batches=1" "diffusion_batch_size=1" "seed=${seed}" \
      "${extra[@]}" "$@" > "${OUT_ROOT}/${name}.log" 2>&1; then
    echo "run failed, see ${OUT_ROOT}/${name}.log:" >&2
    tail -n 15 "${OUT_ROOT}/${name}.log" >&2
    exit 2
  fi
}

# verify_seed LABEL [hydra overrides...]
verify_seed() {
  local label="$1"
  shift
  local repeats=()
  for ((r = 1; r <= REPEATS; r++)); do
    run_design "${label}_seed${SEED}_r${r}" "${SEED}" "$@"
    repeats+=("${OUT_ROOT}/${label}_seed${SEED}_r${r}")
  done
  run_design "${label}_seed${SEED2}" "${SEED2}" "$@"

  local status=0
  echo "== ${label}: same seed (${SEED}) x${REPEATS} must be bitwise identical"
  "${PYTHON}" "${HERE}/compare_runs.py" equal "${repeats[@]}" || status=1
  echo "== ${label}: different seed (${SEED2}) must give a different design (control)"
  "${PYTHON}" "${HERE}/compare_runs.py" differ "${repeats[0]}" "${OUT_ROOT}/${label}_seed${SEED2}" || status=1
  return "${status}"
}
