"""Compare post-RMSNorm RFD3 transition implementations on CUDA.

Uses synthetic activations at token-pair shapes encountered by the denoiser.
Normalization is unchanged and excluded. Weight preparation and compilation are
outside the timed region; graph replay isolates device execution from dispatch.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from rfd3.model import inference_acceleration as accel
from rfd3.model.layers.layer_utils import Transition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[4096, 16384, 65536])
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--expansion", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(*args.rows, args.channels, *args.expansion, args.repeats, args.replays) < 1:
        parser.error("all dimensions and repeat counts must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    kernel_path = Path(accel.__file__).parent / "kernels" / "transition.py"
    report = {
        "label": args.label,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "kernel_sha256": hashlib.sha256(kernel_path.read_bytes()).hexdigest(),
        "repeats": args.repeats,
        "replays_per_sample": args.replays,
        "includes_normalization": False,
        "input_dtype": args.dtype,
        "workloads": [],
    }
    for rows in args.rows:
        for expansion in args.expansion:
            layer = Transition(c=args.channels, n=expansion).cuda().eval()
            x = torch.randn(rows, args.channels, device="cuda").to(
                getattr(torch, args.dtype)
            )

            def native():
                return layer.linear_3(F.silu(layer.linear_1(x)) * layer.linear_2(x))

            def kernel():
                result = accel.fused_transition(layer, x)
                if result is None:
                    raise ValueError("This shape is not supported by the kernel")
                return result

            with (
                torch.no_grad(),
                torch.autocast("cuda", dtype=torch.bfloat16),
                accel.rollout_cache(),
            ):
                expected = native()
                actual = kernel()
                torch.testing.assert_close(actual, expected, atol=0.003, rtol=0.02)
                row = {
                    "rows": rows,
                    "channels": args.channels,
                    "hidden": args.channels * expansion,
                    "max_abs_error": float(
                        (actual.float() - expected.float()).abs().max()
                    ),
                    "rms_error": float(
                        (actual.float() - expected.float()).square().mean().sqrt()
                    ),
                    "results": {},
                }
                # Reverse mode order on alternate shapes to limit order bias.
                funcs = [("torch", native), (args.label, kernel)]
                if len(report["workloads"]) % 2:
                    funcs.reverse()
                for name, fn in funcs:
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with (
                        torch.cuda.stream(stream),
                        torch.autocast(
                            "cuda", dtype=torch.bfloat16, cache_enabled=False
                        ),
                    ):
                        for _ in range(3):
                            fn()
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with (
                        torch.cuda.graph(graph, stream=stream),
                        torch.autocast(
                            "cuda", dtype=torch.bfloat16, cache_enabled=False
                        ),
                    ):
                        output = fn()
                    for _ in range(5):
                        graph.replay()
                    samples = []
                    for _ in range(args.repeats):
                        begin, end = (
                            torch.cuda.Event(enable_timing=True) for _ in range(2)
                        )
                        begin.record()
                        for _ in range(args.replays):
                            graph.replay()
                        end.record()
                        end.synchronize()
                        samples.append(begin.elapsed_time(end) / args.replays)
                    torch.testing.assert_close(output, expected, atol=0.003, rtol=0.02)
                    row["results"][name] = {
                        "median_ms": statistics.median(samples),
                        "samples_ms": samples,
                    }
                    del graph, output
                report["workloads"].append(row)
                print(json.dumps(row), flush=True)
                (args.out / "transition.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )


if __name__ == "__main__":
    main()
