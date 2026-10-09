"""Isolated standard-mode batch capacity probes and warm sampling measurements.

Workers retain full atom-pair tensors. Batch means diffusion samples for one
unconditional input. Warm speed excludes the first three denoiser calls but
includes sampler work between subsequent calls. No compilation fallback is used.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path


def worker(args):
    source_root = Path(importlib.util.find_spec("rfd3").origin).parent
    source_hashes = {
        str(path.relative_to(source_root)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(source_root.rglob("*.py"))
    }
    os.environ["RFD3_LOW_MEMORY_MODE"] = "0"
    os.environ["TORCH_COMPILE_DISABLE"] = "0"
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "4")
    if args.mode.startswith("compiled"):
        os.environ.setdefault("TORCH_LOGS", "recompiles")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(
        (args.out.parent / "compile-cache").resolve()
    )
    os.environ["TRITON_CACHE_DIR"] = str((args.out.parent / "triton-cache").resolve())
    import numpy as np
    import torch
    import triton
    from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine
    from torch._dynamo.utils import counters

    torch.set_num_threads(4)
    compiled = args.mode.startswith("compiled")
    lengths = args.input_lengths or [args.length]
    input_path = None
    if len(lengths) > 1:
        input_path = args.result.with_suffix(".inputs.json")
        input_path.write_text(
            json.dumps(
                {f"input_{i}": {"length": length} for i, length in enumerate(lengths)}
            )
        )
    engine = RFD3InferenceEngine(
        **RFD3InferenceConfig(
            ckpt_path=args.checkpoint,
            specification={"length": str(lengths[0])} if input_path is None else None,
            diffusion_batch_size=args.batch,
            inference_batch_size=len(lengths),
            compile_model=compiled,
            compile_cache_dir=str(args.out.parent / "compile-cache"),
            inference_kernel_backend=args.kernel_backend
            or ("torch" if args.mode in ("eager", "compiled-native") else "auto"),
            inference_cuda_graph=args.mode == "accelerated",
            low_memory_mode=False,
            seed=42,
            inference_sampler={"num_timesteps": args.steps},
        )
    )
    started = time.perf_counter()
    engine.initialize()
    load = time.perf_counter() - started
    model = engine._rfd3_net()
    assert not model.token_initializer.use_chunked_pll
    dm = model.diffusion_module
    method = "forward_batched" if compiled or len(lengths) > 1 else "forward"
    original = getattr(dm, method)
    events, starts, cold = [], [], []
    shape = {}
    profiler = None
    if args.profile:
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        )

    def timed(*pos, **kw):
        profile_this_call = profiler is not None and len(events) == 5
        if profile_this_call:
            torch.cuda.synchronize()
            profiler.start()
        if not events:
            x = pos[0] if pos else kw["X_noisy_L"]
            shape["coordinate_shape"] = list(x.shape)
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        starts.append(time.perf_counter())
        begin.record()
        result = original(*pos, **kw)
        end.record()
        if profile_this_call:
            end.synchronize()
            profiler.stop()
        if not events:
            end.synchronize()
            cold.append(time.perf_counter() - starts[-1])
        events.append((begin, end))
        return result

    setattr(dm, method, timed)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    outputs = engine.run(
        inputs=str(input_path) if input_path is not None else None, n_batches=1
    )
    torch.cuda.synchronize()
    finished = time.perf_counter()
    if profiler is not None:
        profiler.export_chrome_trace(str(args.result.with_suffix(".trace.json")))
        args.result.with_suffix(".profile.txt").write_text(
            profiler.key_averages().table(
                sort_by="self_device_time_total", row_limit=80
            )
        )
    designs = [d for batch in outputs.values() for d in batch]
    if len(lengths) > 1:
        assert shape["coordinate_shape"][:2] == [
            len(lengths),
            args.batch,
        ], "Mixed lengths must share padding buckets to measure one input batch"
        assert len(events) == args.steps - 1
    assert len(designs) == args.batch * len(lengths)
    finite = all(np.isfinite(d.atom_array.coord).all() for d in designs)
    assert finite
    assert Counter(
        int(sum(d.atom_array.atom_name == "CA")) for d in designs
    ) == Counter(length for length in lengths for _ in range(args.batch))
    structures_dir = args.result.with_suffix("") / "structures"
    structures_dir.mkdir(parents=True, exist_ok=True)
    for design in designs:
        design.dump(structures_dir, verbose=False)
    skip = 3
    assert len(events) > skip
    ms = [b.elapsed_time(e) for b, e in events]
    span = events[skip][0].elapsed_time(events[-1][1]) / 1000
    n = len(events) - skip
    report = dict(
        status="ok",
        mode=args.mode,
        length=args.length,
        batch=args.batch,
        input_batch_size=len(lengths),
        input_lengths=lengths,
        steps=args.steps,
        denoiser_calls=len(events),
        warm_calls=n,
        warm_sampling_seconds=span,
        warm_sampling_step_ms=span * 1000 / n,
        warm_design_steps_per_second=len(designs) * n / span,
        warm_denoiser_median_ms=statistics.median(ms[skip:]),
        projected_199_step_designs_per_second=len(designs) * n / (span * 199),
        first_denoiser_seconds=cold[0],
        setup_through_warmup_seconds=starts[skip] - started,
        load_seconds=load,
        total_rollout_seconds=finished - started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        device_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        triton=triton.__version__,
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        torch_num_threads=torch.get_num_threads(),
        checkpoint=str(engine.ckpt_path),
        source_files_sha256=source_hashes,
        source_package_path=str(source_root),
        compile_cache_dir=os.environ["TORCHINDUCTOR_CACHE_DIR"],
        cuda_max_fusion_unique_io_buffers=16 if compiled else None,
        resolved_kernel_backend=engine.resolved_kernel_backend,
        requested_kernel_backend=engine.inference_kernel_backend,
        compile_model=compiled,
        low_memory_mode=False,
        full_resident_atom_pairs=True,
        profiler_instrumented=args.profile,
        finite_coordinates=bool(finite),
        structures_dir=str(structures_dir),
        saved_designs=len(designs),
        compiler_counters={k: dict(v) for k, v in counters.items() if v},
        denoiser_ms=ms,
        **shape,
    )
    args.result.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, default=str), flush=True)


def orchestrate(args):
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []

    def run(length, mode, batch, steps, label):
        stem = f"{mode}-l{length}-d{batch}-{label}"
        result = args.out / (stem + ".json")
        if result.exists():
            return json.loads(result.read_text())
        command = [
            sys.executable,
            __file__,
            "--worker",
            "--length",
            str(length),
            "--batch",
            str(batch),
            "--steps",
            str(steps),
            "--mode",
            mode,
            "--checkpoint",
            args.checkpoint,
            "--out",
            str(args.out),
            "--result",
            str(result),
        ]
        if args.kernel_backend:
            command.extend(["--kernel-backend", args.kernel_backend])
        started = time.perf_counter()
        with (args.out / (stem + ".log")).open("w") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if not result.exists():
            log_text = (args.out / (stem + ".log")).read_text()
            row = dict(
                status="oom" if "out of memory" in log_text.lower() else "error",
                length=length,
                mode=mode,
                batch=batch,
                exit_code=process.returncode,
                process_seconds=time.perf_counter() - started,
                log=stem + ".log",
            )
            result.write_text(json.dumps(row, indent=2) + "\n")
        row = json.loads(result.read_text())
        print(stem, row["status"], row.get("warm_sampling_step_ms"), flush=True)
        return row

    for length in args.lengths:
        for mode in args.modes:
            # Each OOM lives in its own process. Probe geometric growth then
            # bisect to an integer boundary. Compiled shapes persist on disk.
            lo, hi = 0, None
            batch = args.start_batch
            while True:
                row = run(length, mode, batch, 7, "probe")
                if row["status"] == "error":
                    rows.append(row)
                    (args.out / "summary.json").write_text(
                        json.dumps(rows, indent=2) + "\n"
                    )
                    raise RuntimeError(f"Capacity probe failed: {row}")
                if row["status"] == "ok":
                    lo = batch
                    if hi is None:
                        batch *= 2
                    elif hi - lo > 1:
                        batch = (lo + hi) // 2
                    else:
                        break
                else:
                    hi = batch
                    if hi - lo <= 1:
                        break
                    batch = (lo + hi) // 2
            if lo:
                warm = run(length, mode, lo, args.steps, "warm")
                warm_batch = lo
                while warm["status"] == "oom" and warm_batch > 1:
                    warm_batch -= 1
                    warm = run(length, mode, warm_batch, args.steps, "warm")
                warm["largest_passing_probe_batch"] = lo
                warm["first_failing_batch"] = hi
                rows.append(warm)
            else:
                rows.append(
                    row | dict(largest_passing_probe_batch=0, first_failing_batch=hi)
                )
            (args.out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Diagnostic trace of one warm call; timings are instrumented",
    )
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--lengths", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument(
        "--modes", nargs="+", default=["compiled-native", "accelerated"]
    )
    parser.add_argument(
        "--mode",
        choices=["accelerated", "compiled", "compiled-native", "eager"],
        default="compiled",
    )
    parser.add_argument("--start-batch", type=int, default=16)
    parser.add_argument(
        "--kernel-backend", choices=["auto", "torch", "triton", "triton-transition"]
    )
    parser.add_argument(
        "--input-lengths", nargs="+", type=int, help="Worker: mixed input batch"
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--checkpoint", default="rfd3")
    parser.add_argument("--out", type=Path, default=Path("outs/buckets"))
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.worker:
        try:
            worker(args)
        except Exception as error:
            import traceback

            traceback.print_exc()
            args.result.write_text(
                json.dumps(
                    dict(
                        status="oom"
                        if "out of memory" in str(error).lower()
                        else "error",
                        length=args.length,
                        batch=args.batch,
                        mode=args.mode,
                        input_lengths=args.input_lengths or [args.length],
                        requested_kernel_backend=args.kernel_backend,
                        compile_model=args.mode.startswith("compiled"),
                        low_memory_mode=False,
                        error_type=type(error).__name__,
                        error=str(error),
                    ),
                    indent=2,
                )
                + "\n"
            )
            raise
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
