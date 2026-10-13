#!/usr/bin/env bash
# Seed determinism check, unconditional generation (no input structure, length 100-150).
#   bash models/rfd3/scripts/verify_seed_determinism/verify_unconditional.sh
# Exit code 0 = PASS (same seed -> bitwise identical; different seed -> different).
# See _common.sh for the environment variables it honours.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

UNCOND_LENGTH="${UNCOND_LENGTH:-100-150}"

verify_seed unconditional "inputs=null" "+specification.length=${UNCOND_LENGTH}" \
  && echo "PASS: unconditional generation is seed-deterministic" \
  || { echo "FAIL: unconditional generation is not seed-deterministic"; exit 1; }
