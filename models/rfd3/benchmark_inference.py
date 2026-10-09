"""Profile real checkpoint denoising on identical inputs, excluding model loading.

Run from the repo: python models/rfd3/benchmark_inference.py --length 64 --out DIR
All workloads use the standard dense pair representation (low_memory_mode=False).
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine
from rfd3.model import inference_acceleration as accel
from rfd3.model.layers.blocks import CompactStreamingDecoder, LocalAtomTransformer
from rfd3.model.layers.layer_utils import Transition


class Captured(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--checkpoint", default="rfd3")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--record-transition-shapes", action="store_true")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=[
            "baseline",
            "sdpa",
            "kernels",
            "kernels_sdpa",
            "graph",
            "kernels_token_graph",
            "kernels_cached_graph",
            "kernels_graph_torch_transition",
            "kernels_graph",
        ],
        default=[
            "baseline",
            "sdpa",
            "kernels",
            "kernels_sdpa",
            "graph",
            "kernels_graph",
        ],
    )
    args = parser.parse_args()
    if min(args.length, args.batch, args.repeats) < 1 or args.warmup < 0:
        parser.error(
            "length, batch and repeats must be positive; warmup must be nonnegative"
        )
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    engine = RFD3InferenceEngine(
        **RFD3InferenceConfig(
            ckpt_path=args.checkpoint,
            diffusion_batch_size=args.batch,
            specification={"length": str(args.length)},
            seed=42,
            inference_sampler={"num_timesteps": 10},
            low_memory_mode=False,
            compile_model=False,
            inference_kernel_backend="torch",
        )
    )
    engine.initialize()
    dm = engine._rfd3_net().diffusion_module
    captured = {}

    def capture(module, positional, kwargs):
        captured.update(kwargs)
        raise Captured

    handle = dm.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        engine.run(inputs=None, n_batches=1)
    except Captured:
        pass
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError("No denoiser inputs captured")

    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "length": args.length,
        "batch": args.batch,
        "low_memory_mode": False,
        "atoms": captured["X_noisy_L"].shape[1],
        "recycles": dm.n_recycle,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "results": {},
    }
    reference = None
    modes = [
        ("baseline", "torch", "vanilla", False),
        ("sdpa", "torch", "sdpa", False),
        ("kernels", "triton", "vanilla", False),
        ("kernels_sdpa", "triton", "sdpa", False),
        ("graph", "torch", "vanilla", True),
        ("kernels_token_graph", "triton", "vanilla", True),
        ("kernels_cached_graph", "triton", "vanilla", True),
        ("kernels_graph_torch_transition", "triton", "vanilla", True),
        ("kernels_graph", "triton", "vanilla", True),
    ]
    modes = [m for m in modes if m[0] in args.modes]
    if args.profile and len(modes) != 1:
        raise ValueError(
            "Profile one mode per process to avoid perturbing later timings"
        )
    for mode, backend, dense, graph in modes:
        for module in dm.modules():
            module.inference_kernel_backend = backend
            if mode == "kernels_graph_torch_transition" and isinstance(
                module, Transition
            ):
                module.inference_kernel_backend = "torch"
            module.inference_cuda_graph = graph
            if mode in ("kernels_token_graph", "kernels_cached_graph") and isinstance(
                module, (LocalAtomTransformer, CompactStreamingDecoder)
            ):
                module.inference_cuda_graph = False
            if hasattr(module, "dense_attention_backend"):
                module.dense_attention_backend = dense
        torch.cuda.empty_cache()
        with (
            torch.no_grad(),
            torch.autocast("cuda", dtype=torch.bfloat16),
            accel.rollout_cache(
                backend == "triton" or graph,
                constants=(captured["P_LL"],),
                atom_layout=mode != "kernels_token_graph",
            ),
        ):
            transition_shapes = {}
            handles = []
            if args.record_transition_shapes:
                for name, module in dm.named_modules():
                    if isinstance(module, Transition):

                        def record_shape(
                            norm, inputs, output, name=name, module=module
                        ):
                            transition_shapes[name] = {
                                "shape": list(output.shape),
                                "dtype": str(output.dtype),
                                "hidden": module.linear_1.out_features,
                            }

                        handles.append(
                            module.layer_norm_1.register_forward_hook(record_shape)
                        )
            start = time.perf_counter()
            try:
                out = dm(**captured)
            finally:
                for handle in handles:
                    handle.remove()
            torch.cuda.synchronize()
            cold = time.perf_counter() - start
            if reference is None:
                report["reference_mode"] = mode
                reference = {k: v.detach().clone() for k, v in out.items()}
                repeat = dm(**captured)
                report["reference_repeat_error"] = {
                    k: {
                        "max_abs": (repeat[k].float() - v.float()).abs().max().item(),
                        "rms": (repeat[k].float() - v.float())
                        .square()
                        .mean()
                        .sqrt()
                        .item(),
                    }
                    for k, v in reference.items()
                    if k in ("X_L", "sequence_logits_I")
                }
            errors = {}
            for key in ("X_L", "sequence_logits_I"):
                delta = out[key].float() - reference[key].float()
                errors[key] = {
                    "max_abs": delta.abs().max().item(),
                    "rms": delta.square().mean().sqrt().item(),
                    "reference_rms": reference[key]
                    .float()
                    .square()
                    .mean()
                    .sqrt()
                    .item(),
                    "finite": bool(torch.isfinite(out[key]).all()),
                }
            for _ in range(args.warmup):
                dm(**captured)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            wall, device = [], []
            for _ in range(args.repeats):
                begin, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                torch.cuda.synchronize()
                start = time.perf_counter()
                begin.record()
                out = dm(**captured)
                end.record()
                end.synchronize()
                wall.append(1000 * (time.perf_counter() - start))
                device.append(begin.elapsed_time(end))
            result = {
                "transition_shapes": transition_shapes,
                "transition_backend": "torch"
                if mode == "kernels_graph_torch_transition"
                else backend,
                "atom_layout_cache": mode != "kernels_token_graph"
                and (backend == "triton" or graph),
                "atom_graphs": graph
                and mode not in ("kernels_token_graph", "kernels_cached_graph"),
                "cold_seconds": cold,
                "wall_ms": wall,
                "cuda_ms": device,
                "median_wall_ms": statistics.median(wall),
                "median_cuda_ms": statistics.median(device),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "error": errors,
            }
            if args.profile:
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=True,
                ) as prof:
                    dm(**captured)
                    torch.cuda.synchronize()
                prof.export_chrome_trace(str(args.out / f"{mode}.trace.json"))
                (args.out / f"{mode}.profile.txt").write_text(
                    prof.key_averages().table(
                        sort_by="self_cuda_time_total", row_limit=40
                    )
                )
            report["results"][mode] = result
            print(mode, json.dumps(result), flush=True)
            (args.out / "benchmark.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
