"""Compare native and exact kernels with frozen inputs, weights and RNG state.

Example (H200, Torch 2.11/cu128):
  CUBLAS_WORKSPACE_CONFIG=:4096:8 python scripts/benchmark_rf3_exact_kernels.py \
      --checkpoint /path/to/rf3.ckpt --lengths 128 256 400 512 \
      --seeds 11 29 --out outs/rf3-kernels/benchmark.json

Reports synchronized model+metric wall time; excludes preprocessing, checkpoint
loading and the first call of each backend (reported separately). Any tensor
mismatch or nonfinite output fails the run. No speedup is asserted by the script.
"""

import argparse
import copy
import json
import os
import random
import statistics
import time
from pathlib import Path

# Must be configured before CUDA/cuBLAS initialize in this standalone process.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from rf3.inference_engines.rf3 import RF3InferenceEngine
from rf3.model.layers.inference_kernels import set_exact_kernels, validated_stack
from rf3.utils.inference import InferenceInput


def tensors(value, path=""):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from tensors(child, f"{path}/{key}")
    elif isinstance(value, (list, tuple)):
        for i, child in enumerate(value):
            yield from tensors(child, f"{path}/{i}")


def rng_state():
    return (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all(),
    )


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    torch.cuda.set_rng_state_all(state[3])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 256, 400, 512])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 29])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument(
        "--triangle-backend", choices=["auto", "vanilla"], default="vanilla"
    )
    args = parser.parse_args()
    if args.repeats < 1 or min(args.lengths) < 1:
        parser.error("repeats and lengths must be positive")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    if not validated_stack(torch.cuda.current_device()):
        raise RuntimeError(
            "This stack would use native fallback; cannot measure kernels"
        )
    engine = RF3InferenceEngine(
        ckpt_path=args.checkpoint,
        n_recycles=10,
        num_steps=50,
        diffusion_batch_size=args.samples,
        early_stopping_plddt_threshold=0,
        triangle_attention_backend=args.triangle_backend,
        dense_attention_backend="vanilla",
        inference_kernel_backend="torch",
    )
    engine.initialize()
    model = engine.trainer.state["model"]
    original = engine.trainer.validation_step
    # Count actual kernel launches, so an all-native fallback cannot pass this audit.
    from rf3.model.kernels import exact_layer_norm

    launch = exact_layer_norm.layer_norm
    calls = 0

    def counted(*a, **kw):
        nonlocal calls
        result = launch(*a, **kw)
        calls += 1
        return result

    exact_layer_norm.layer_norm = counted
    report = dict(
        complete=False,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        deterministic=True,
        cublas_workspace=os.environ["CUBLAS_WORKSPACE_CONFIG"],
        checkpoint=str(args.checkpoint.resolve()),
        recycles=10,
        steps=50,
        samples=args.samples,
        triangle_backend=args.triangle_backend,
        cases=[],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    context = {}

    def compare(*a, **kw):
        nonlocal calls
        rng = rng_state()
        frozen = copy.deepcopy(kw)
        reference = None
        row = dict(**context, runs=[])
        # Alternate the order of paired timed runs after warming both backends.
        order = ["torch", "exact"]
        for repeat in range(args.repeats):
            order.extend(["torch", "exact"] if repeat % 2 == 0 else ["exact", "torch"])
        for i, backend in enumerate(order):
            set_exact_kernels(model, backend == "exact")
            inputs = copy.deepcopy(frozen)
            restore_rng(rng)
            calls = 0
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = original(*a, **inputs)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            values = {p: t.detach().cpu() for p, t in tensors(result)}
            assert all(
                torch.isfinite(t).all() for t in values.values()
            ), "nonfinite output"
            if reference is None:
                reference = values
            assert reference.keys() == values.keys(), "output schema changed"
            for path, value in values.items():
                ref = reference[path]
                assert ref.dtype == value.dtype and ref.shape == value.shape, path
                assert torch.equal(
                    ref.contiguous().reshape(-1).view(torch.uint8),
                    value.contiguous().reshape(-1).view(torch.uint8),
                ), f"bitwise mismatch: {context}, {backend}, {path}"
            if backend == "exact" and context["length"] >= 363:
                assert calls > 0, "candidate fell back everywhere"
            run = dict(
                backend=backend,
                cold=i < 2,
                seconds=seconds,
                kernel_calls=calls,
                tensors=len(values),
                bitwise_equal=True,
            )
            row["runs"].append(run)
            print(json.dumps(dict(**context, **run)), flush=True)
        medians = {
            b: statistics.median(
                r["seconds"] for r in row["runs"] if r["backend"] == b and not r["cold"]
            )
            for b in ("torch", "exact")
        }
        row.update(median_seconds=medians, speedup=medians["torch"] / medians["exact"])
        report["cases"].append(row)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        return result

    engine.trainer.validation_step = compare
    for length in args.lengths:
        # Fixed heterogeneous sequence shared between seeds/backends. This is a
        # numerical/performance fixture, not a protein-quality benchmark.
        seq = "".join(random.Random(length).choices("ACDEFGHIKLMNPQRSTVWY", k=length))
        item = InferenceInput.from_json_dict(
            dict(
                name=f"protein{length}",
                components=[dict(seq=seq, chain_id="A", chain_type="polypeptide(l)")],
            )
        )
        for seed in args.seeds:
            context.update(length=length, seed=seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            engine.seed = seed
            engine.run(item, out_dir=None, dump_predictions=False)
    report["complete"] = True
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
