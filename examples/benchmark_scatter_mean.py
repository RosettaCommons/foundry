"""CUDA pooling precision and synchronized latency/memory microbenchmark."""

import json
import time

import torch

from foundry.utils.torch import scatter_mean


def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    records = []
    for batch, groups, size, channels in [
        (1, 40, 14, 128),
        (1, 200, 14, 768),
        (8, 200, 14, 768),
        (1, 200, 23, 768),
    ]:
        x = torch.randn(
            batch, groups * size, channels, device="cuda", dtype=torch.bfloat16
        )
        idx = torch.arange(groups, device="cuda").repeat_interleave(size)
        z = torch.zeros(batch, groups, channels, device="cuda", dtype=x.dtype)
        oracle = x.double().reshape(batch, groups, size, channels).mean(2).to(x.dtype)
        row = {"shape": [batch, groups, size, channels]}
        for label, fn in [
            (
                "native",
                lambda z, d, i, x: z.index_reduce(d, i, x, "mean", include_self=False),
            ),
            ("fixed", scatter_mean),
        ]:
            for _ in range(10):
                fn(z, 1, idx, x)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            start = time.perf_counter()
            for _ in range(100):
                y = fn(z, 1, idx, x)
            torch.cuda.synchronize()
            wall = (time.perf_counter() - start) * 10
            row[label] = {
                "forward_ms": wall,
                "peak_extra_bytes": torch.cuda.max_memory_allocated() - before,
                "max_abs_vs_fp64": float((y.float() - oracle.float()).abs().max()),
            }
            x.requires_grad_(True)
            for _ in range(5):
                torch.autograd.grad(fn(z, 1, idx, x).sum(), x)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(30):
                torch.autograd.grad(fn(z, 1, idx, x).sum(), x)
            torch.cuda.synchronize()
            row[label]["forward_backward_ms"] = (
                (time.perf_counter() - start) * 1000 / 30
            )
            x.requires_grad_(False)
        records.append(row)
    precision = []
    for dtype in [torch.float16, torch.bfloat16, torch.float32]:
        for n in [14, 23, 64, 300, 3000]:
            x = torch.randn(4, n, 128, device="cuda", dtype=dtype)
            idx = torch.zeros(n, device="cuda", dtype=torch.long)
            z = torch.zeros(4, 1, 128, device="cuda", dtype=dtype)
            oracle = x.double().mean(1, keepdim=True)
            ys = [
                z.index_reduce(1, idx, x, "mean", include_self=False) for _ in range(10)
            ]
            fixed = scatter_mean(z, 1, idx, x)
            precision.append(
                {
                    "dtype": str(dtype),
                    "n": n,
                    "native_max_error_vs_quantized_fp64_mean": float(
                        (ys[0].double() - oracle).abs().max()
                    ),
                    "fixed_max_error_vs_quantized_fp64_mean": float(
                        (fixed.double() - oracle).abs().max()
                    ),
                    "native_repeat_max_delta": max(
                        float((ys[0] - y).abs().max()) for y in ys
                    ),
                    "fixed_equals_rounded_fp64": bool(
                        torch.equal(fixed, oracle.to(dtype))
                    ),
                }
            )
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(),
                "benchmarks": records,
                "precision": precision,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
