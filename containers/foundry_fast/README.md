# Foundry fast image

Extends the existing Foundry SIF with this repository's tracked working-tree
source, including staged inference changes. The snapshot excludes `.env`, Git
metadata, checkpoints, compilation caches and `outs/`. Dependencies, CUDA,
PyTorch and Triton are inherited from the base image. Foundry is reinstalled
editable at `/opt/foundry` **inside the image**; no host source mount is needed.
Kernel licenses and provenance notices are included with the source.

Build from the repository root:

```bash
APPTAINER_TMPDIR=/scratch/jbutch-foundry-fast-build \
  bash containers/foundry_fast/build.sh
```

Optional positional arguments are the base SIF and a new output directory.
The build emits `foundry_fast.sif`, a SHA-256 checksum, its log, and the frozen
source. `/opt/foundry/IMAGE_SOURCE.json` inside the image records the Git base
revision and individual source hashes; the image includes uncommitted changes.

The RFD3 defaults are compiled inference with preset buckets
(`compile_shape_policy=buckets`), automatic kernel selection, two CPU
loader workers and full-memory operation. `RFD3_COMPILE_CACHE_DIR` is not set by
the image; supply it explicitly if a persistent compilation cache is wanted.
The base image used for the initial build has PyTorch 2.11.0+cu128 and Triton 3.6.0.

## Use with Pipelines without changing its code

The existing block's `command_template` override selects this image:

```python
from blocks import Blocks

RFD3 = Blocks.get("rfd3")
command = RFD3(cloud=False).command_template
command = [
    "/projects/ml/pipelines/containers/foundry_fast.sif"
    if part == "${CONTAINER:foundry}" else part
    for part in command
]
rfd3 = RFD3(
    cloud=False,
    command_template=command,
    # Other RFD3 block options go here.
)
```

This retains Pipelines' existing mounts and environment settings. The original
`foundry.sif` and the Pipelines repository are not modified. Tests can run using
the source and pytest baked into the image; run from `/opt/foundry` to ensure
the host checkout cannot supply imports.

Initial build validation (2026-10-07): all 946 snapshot file hashes matched,
47 targeted CUDA tests passed on an RTX A4000, and the unchanged Pipelines RFD3
block generated and collected one 32-residue, seven-step smoke-test structure
with compilation enabled. This short sample tests integration, not biological
quality. Logs, the structure, source/image manifests and checksums are under
`outs/foundry-fast/`. The shared image is
`/projects/ml/pipelines/containers/foundry_fast.sif`.

Compiled CUDA graph replay is enabled by default (`compile_cuda_graphs=true`).
Use `compile_cuda_graphs=false` to disable replay while retaining compilation.
