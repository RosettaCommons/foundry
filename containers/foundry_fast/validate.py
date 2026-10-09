"""Verify baked source hashes, inference defaults, and GPU/runtime availability."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import torch
import triton
from rfd3.engine import RFD3InferenceConfig
from rfd3.model import kernel_ops

root = Path("/opt/foundry")
manifest = json.loads((root / "IMAGE_SOURCE.json").read_text())
failures = []
for name, expected in manifest["files_sha256"].items():
    path = root / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        failures.append(name)
assert not failures, failures
assert Path(kernel_ops.__file__).is_relative_to(root)
config = RFD3InferenceConfig()
assert config.compile_model and not config.low_memory_mode
assert config.compile_shape_policy == "buckets"
assert config.compile_cuda_graphs
assert config.inference_kernel_backend == "auto" and config.inference_num_workers == 2
assert "RFD3_COMPILE_CACHE_DIR" not in os.environ
assert config.compile_cache_dir is None
assert torch.cuda.is_available()
report = {
    "source_revision": manifest["git_revision"],
    "verified_source_files": len(manifest["files_sha256"]),
    "kernel_ops": kernel_ops.__file__,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "triton": triton.__version__,
    "atomworks": importlib.metadata.version("atomworks"),
    "gpu": torch.cuda.get_device_name(),
    "compile_model": config.compile_model,
    "compile_shape_policy": config.compile_shape_policy,
    "compile_cuda_graphs": config.compile_cuda_graphs,
    "inference_kernel_backend": config.inference_kernel_backend,
    "inference_num_workers": config.inference_num_workers,
    "low_memory_mode": config.low_memory_mode,
    "compile_cache_dir": config.compile_cache_dir,
}
print(json.dumps(report, indent=2))
