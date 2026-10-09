"""Time complete standard-mode sampling with real weights and retained designs."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from lightning.fabric import seed_everything
from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine


def ca_rmsd(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    u, _, vh = np.linalg.svd(a.T @ b)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vh)
    return float(np.sqrt(np.mean(np.sum((a @ u @ correction @ vh - b) ** 2, axis=-1))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--checkpoint", default="rfd3")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    engine = RFD3InferenceEngine(
        **RFD3InferenceConfig(
            ckpt_path=args.checkpoint,
            diffusion_batch_size=args.batch,
            specification={"length": str(args.length)},
            seed=42,
            inference_sampler={"num_timesteps": args.steps},
            low_memory_mode=False,
            compile_model=False,
            inference_kernel_backend="torch",
        )
    )
    started = time.perf_counter()
    engine.initialize()
    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "length": args.length,
        "batch": args.batch,
        "steps": args.steps,
        "denoiser_calls": args.steps - 1,
        "low_memory_mode": False,
        "load_seconds": time.perf_counter() - started,
        "runs": [],
    }
    # Warm libraries and preprocessing without changing the measured step count.
    sampler = engine._rfd3_net().inference_sampler.sampler
    sampler.num_timesteps = 3
    engine.run(inputs=None, n_batches=1)
    sampler.num_timesteps = args.steps
    reference = None
    for repetition in range(args.repeats):
        # Alternate order to reduce monotonic CPU/GPU clock or thermal bias.
        modes = (
            ("baseline", "accelerated")
            if repetition % 2 == 0
            else ("accelerated", "baseline")
        )
        for mode in modes:
            engine.inference_kernel_backend = (
                "triton" if mode == "accelerated" else "torch"
            )
            engine.inference_cuda_graph = mode == "accelerated"
            seed_everything(42, workers=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            outputs = engine.run(inputs=None, n_batches=1)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            peak = torch.cuda.max_memory_allocated()
            designs = [d for batch in outputs.values() for d in batch]
            assert len(designs) == args.batch
            checks = []
            run_dir = args.out / f"{mode}-{repetition}"
            run_dir.mkdir(exist_ok=False)
            for design in designs:
                aa = design.atom_array
                ca = aa.coord[aa.atom_name == "CA"]
                assert len(ca) == args.length and np.isfinite(aa.coord).all()
                if reference is None:
                    reference = ca.copy()
                checks.append(
                    {
                        "finite": True,
                        "ca_count": len(ca),
                        "ca_rmsd_to_first_baseline": ca_rmsd(ca, reference),
                        "median_adjacent_ca_distance": float(
                            np.median(np.linalg.norm(np.diff(ca, axis=0), axis=-1))
                        ),
                    }
                )
                design.dump(run_dir, verbose=False)
            row = {
                "mode": mode,
                "repetition": repetition,
                "seconds": elapsed,
                "peak_allocated_bytes": peak,
                "designs": checks,
            }
            report["runs"].append(row)
            print(json.dumps(row), flush=True)
            (args.out / "sampling.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
