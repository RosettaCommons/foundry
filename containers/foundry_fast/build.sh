#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/../.." && pwd)
base_image=${1:-/projects/ml/pipelines/containers/foundry.sif}
output_dir=${2:-"$repo_dir/outs/foundry-fast-$(date +%Y%m%d-%H%M%S)"}
mkdir -p "$output_dir"
output_dir=$(cd "$output_dir" && pwd)

python3 "$repo_dir/containers/foundry_fast/snapshot.py" "$output_dir/source"
apptainer build --fakeroot --mksquashfs-args '-processors 4' \
    --build-arg "base_image=$base_image" \
    --build-arg "snapshot=$output_dir/source" \
    "$output_dir/foundry_fast.sif" \
    "$repo_dir/containers/foundry_fast/foundry_fast.def" \
    2>&1 | tee "$output_dir/build.log"

sha256sum "$output_dir/foundry_fast.sif" > "$output_dir/foundry_fast.sif.sha256"
