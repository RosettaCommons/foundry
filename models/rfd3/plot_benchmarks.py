"""Render measured warm RFD3 speed, preserving missing/OOM points explicitly."""

import argparse
import csv
import json
import os
from pathlib import Path

LENGTHS = [64, 128, 256]
MODES = ["eager", "compiled", "accelerated"]
LABELS = {
    "eager": "Uncompiled · native",
    "compiled": "Compiled · native",
    "accelerated": "Uncompiled · custom kernels + graphs",
}
COLORS = {"eager": "#606C79", "compiled": "#CB7223", "accelerated": "#2169B3"}


def plot_capacity(summary, out):
    """Compare measured throughput at each path's largest verified batch."""
    records = json.loads(summary.read_text())
    if not records:
        raise RuntimeError("No capacity measurements yet")
    modes = ["compiled-native", "accelerated"]
    labels = {
        "eager": "Uncompiled · native",
        "compiled-native": "Compiled · native",
        "accelerated": "Uncompiled · custom + graphs",
        "compiled": "Compiled · custom kernels",
    }
    colors = {
        "eager": "#606C79",
        "compiled-native": "#CB7223",
        "accelerated": "#2169B3",
        "compiled": "#368368",
    }
    indexed, rows = {}, []
    for record in records:
        if record["mode"] not in modes or record["length"] not in LENGTHS:
            continue
        if record["status"] not in {"ok", "oom"}:
            raise RuntimeError("Resolve capacity benchmark errors before plotting")
        if record["status"] == "ok":
            assert record["steps"] == 200 and record["warm_calls"] == 196
            assert not record["low_memory_mode"]
            assert not record.get("profiler_instrumented", False)
        key = record["mode"], record["length"]
        if key in indexed:
            raise RuntimeError(f"Duplicate capacity result: {key}")
        indexed[key] = record
        rows.append(
            dict(
                mode=record["mode"],
                tokens=record["length"],
                status=record["status"],
                verified_batch=record["batch"] if record["status"] == "ok" else 0,
                largest_passing_probe_batch=record.get("largest_passing_probe_batch"),
                first_failing_batch=record.get("first_failing_batch"),
                warm_sampling_step_ms=record.get("warm_sampling_step_ms"),
                projected_199_step_designs_per_second=record.get(
                    "projected_199_step_designs_per_second"
                ),
                peak_allocated_gib=record.get("peak_allocated_bytes", 0) / 2**30
                or None,
                structures=record.get("structures_dir"),
            )
        )
    complete = all((mode, n) in indexed for mode in modes for n in LENGTHS)
    stem = "rfd3_batch_throughput_by_length" + ("" if complete else "_partial")
    with (out / f"{stem}.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.environ.setdefault("MPLCONFIGDIR", str(out.resolve() / "matplotlib-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, (rate, batch) = plt.subplots(1, 2, figsize=(12, 6.6))
    fig.subplots_adjust(left=0.085, right=0.975, bottom=0.27, top=0.71, wspace=0.28)
    title = "RFD3 throughput after batch capacity tuning"
    if not complete:
        title += f" · {len(indexed)}/{len(modes) * len(LENGTHS)} runs"
    fig.text(0.085, 0.935, title, fontsize=18, fontweight="bold", color="#18232E")
    fig.text(
        0.085,
        0.89,
        "RTX A4000 (16 GB) · 2 recycles · full resident atom pairs",
        fontsize=11.5,
        color="#526171",
    )
    for ax in (rate, batch):
        ax.set_xticks(LENGTHS)
        ax.set_xlim(42, 278)
        ax.set_xlabel("Tokens")
        ax.grid(axis="y", color="#E6EAF0")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for mode in modes:
        values = [indexed.get((mode, n), {}) for n in LENGTHS]
        throughput = [
            r.get("projected_199_step_designs_per_second", np.nan) for r in values
        ]
        batches = [r["batch"] if r.get("status") == "ok" else np.nan for r in values]
        rate.plot(
            LENGTHS, throughput, "o-", label=labels[mode], color=colors[mode], lw=2.4
        )
        batch.plot(LENGTHS, batches, "o-", color=colors[mode], lw=2.4)
    rate.set_ylabel("Designs/s projected from warm sampling")
    rate.set_ylim(bottom=0)
    rate.set_title("Warm throughput · higher is faster", loc="left", pad=12)
    batch.set_ylabel("Largest batch verified over full schedule")
    batch.set_yscale("log", base=2)
    batch.yaxis.set_major_formatter(lambda value, pos: f"{value:g}")
    batch.set_title("Diffusion samples per input", loc="left", pad=12)
    for n in LENGTHS:
        failed = sum(
            indexed.get((mode, n), {}).get("status") == "oom" for mode in modes
        )
        if failed:
            for ax in (rate, batch):
                ax.axvspan(n - 10, n + 10, color="#FAECEC", zorder=0)
                ax.text(
                    n,
                    0.95,
                    "OOM" if failed == len(modes) else f"{failed} OOM",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    color="#A34040",
                    fontsize=10,
                    fontweight="bold",
                )
    handles, names = rate.get_legend_handles_labels()
    fig.legend(
        handles,
        names,
        loc="upper left",
        bbox_to_anchor=(0.075, 0.857),
        ncol=2,
        frameon=False,
        fontsize=10,
        columnspacing=2.5,
    )
    note = (
        "Geometric batch probes + integer bisection; successful capacity verified with a 200-point schedule.\n"
        "Warm window: 196 calls after 3 warmups, including sampler work; loading, compilation and file writing excluded.\n"
        "Throughput projects batch / (199 × warm ms/step); it is not full end-to-end throughput.\n"
        "One input per batch; compiled paths use padding buckets. OOM at batch 1 has no throughput value."
    )
    fig.text(0.085, 0.045, note, fontsize=8.8, color="#526171", linespacing=1.5)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(out / f"{stem}.{extension}", dpi=220, facecolor="white")
    plt.close(fig)
    print(out / f"{stem}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outs/buckets"))
    parser.add_argument("--out", type=Path, default=Path("outs"))
    parser.add_argument("--capacity-summary", type=Path)
    parser.add_argument("--lengths", nargs="+", type=int, default=LENGTHS)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    lengths = sorted(set(args.lengths))
    if args.capacity_summary is not None:
        plot_capacity(args.capacity_summary, args.out)
        return
    records = {}
    rows = []
    for mode in MODES:
        for length in lengths:
            path = args.input / f"plot-{mode}-l{length}-d1.json"
            if not path.exists():
                continue
            record = json.loads(path.read_text())
            if record["status"] not in {"ok", "oom"}:
                raise RuntimeError(f"Resolve benchmark error before plotting: {path}")
            if record["status"] == "ok":
                assert record["batch"] == 1 and record["steps"] == 200
                assert record["warm_calls"] == 196 and not record["low_memory_mode"]
                assert not record.get("profiler_instrumented", False)
            records[mode, length] = record
            rows.append(
                dict(
                    mode=mode,
                    tokens=length,
                    batch=1,
                    status=record["status"],
                    model_atom_capacity=record.get("coordinate_shape", [None, None])[
                        -2
                    ],
                    warm_sampling_step_ms=record.get("warm_sampling_step_ms"),
                    warm_denoiser_median_ms=record.get("warm_denoiser_median_ms"),
                    projected_199_step_designs_per_second=record.get(
                        "projected_199_step_designs_per_second"
                    ),
                    peak_allocated_gib=record["peak_allocated_bytes"] / 2**30
                    if "peak_allocated_bytes" in record
                    else None,
                    full_rollout_seconds=record.get("total_rollout_seconds"),
                    first_denoiser_seconds=record.get("first_denoiser_seconds"),
                    structures=record.get("structures_dir"),
                    source=str(path),
                )
            )
    if not rows:
        raise RuntimeError("No completed length-plot measurements yet")
    for row in rows:
        reference = records.get(("eager", row["tokens"]), {})
        row["speedup_over_native"] = (
            reference["warm_sampling_step_ms"] / row["warm_sampling_step_ms"]
            if row["status"] == "ok" and reference.get("status") == "ok"
            else None
        )
    complete = len(records) == len(MODES) * len(lengths)
    stem = "rfd3_speed_by_length" if complete else "rfd3_speed_by_length_partial"
    with (args.out / (stem + ".csv")).open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    os.environ.setdefault("MPLCONFIGDIR", str(args.out.resolve() / "matplotlib-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, (latency, speedup) = plt.subplots(
        1, 2, figsize=(12, 6.2), gridspec_kw={"width_ratios": [1.45, 1]}
    )
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.28, top=0.74, wspace=0.26)
    title = f"Warm RFD3 sampling, {min(lengths)}–{max(lengths)} tokens"
    if not complete:
        title += f"  ·  {len(records)}/{len(MODES) * len(lengths)} runs complete"
    fig.text(0.075, 0.935, title, fontsize=19, fontweight="bold", color="#18232E")
    successful = [r for r in records.values() if r["status"] == "ok"]
    devices = {r["gpu"] for r in successful}
    if len(devices) != 1:
        raise RuntimeError(f"Expected one GPU type in length plot: {devices}")
    gpu = successful[0]["gpu"]
    fig.text(
        0.075,
        0.89,
        f"{gpu}  ·  diffusion batch 1  ·  2 recycles  ·  standard memory mode",
        fontsize=11.5,
        color="#526171",
    )
    for ax in (latency, speedup):
        ax.set_xticks(lengths)
        ax.set_xlim(min(lengths) - 22, max(lengths) + 22)
        ax.set_xlabel("Tokens")
        ax.grid(axis="y", color="#E6EAF0", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#BAC4CE")
        ax.tick_params(color="#BAC4CE")
    for mode in MODES:
        values = [
            records.get((mode, n), {}).get("warm_sampling_step_ms", np.nan)
            for n in lengths
        ]
        latency.plot(
            lengths,
            values,
            "o-",
            label=LABELS[mode],
            color=COLORS[mode],
            linewidth=2.5,
            markersize=5.5,
        )
        if mode != "eager":
            ratios = [
                records.get(("eager", n), {}).get("warm_sampling_step_ms", np.nan) / v
                for n, v in zip(lengths, values)
            ]
            speedup.plot(
                lengths, ratios, "o-", color=COLORS[mode], linewidth=2.5, markersize=5.5
            )
    latency.set_ylabel("Warm sampling time (ms/step)")
    latency.set_ylim(bottom=0)
    latency.set_title(
        "Latency  ·  lower is faster", loc="left", pad=13, color="#526171"
    )
    speedup.axhline(1, color=COLORS["eager"], linewidth=1.2, linestyle="--")
    speedup.set_ylabel("Speedup over native eager")
    speedup.set_ylim(bottom=0)
    speedup.set_title(
        "Relative speed  ·  higher is faster", loc="left", pad=13, color="#526171"
    )
    speedup.yaxis.set_major_formatter(lambda x, pos: f"{x:g}×")
    handles, labels = latency.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.067, 0.858),
        ncol=3,
        frameon=False,
        fontsize=10,
        handlelength=2,
        columnspacing=1.8,
    )
    note = (
        "Released checkpoint · 200 schedule points / 199 denoiser calls · first 3 calls excluded from warm timing.\n"
        "Warm window includes work between denoiser calls; excludes loading, compilation and file writing.\n"
        "Compiled path uses atom/token padding buckets (64 tokens pad to 128); custom kernels use the legacy path."
    )
    failures = {}
    for length in lengths:
        failed = [
            mode
            for mode in MODES
            if records.get((mode, length), {}).get("status") == "oom"
        ]
        if failed:
            failures[length] = failed
            for ax in (latency, speedup):
                ax.axvspan(length - 10, length + 10, color="#FAECEC", zorder=0)
                ax.text(
                    length,
                    0.96,
                    "OOM" if len(failed) == 3 else f"{len(failed)} OOM",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    color="#A34040",
                    fontsize=10,
                    fontweight="bold",
                )
    if failures:
        summary = "; ".join(
            f"{n} tokens: {len(modes)}/3 paths" for n, modes in failures.items()
        )
        note += (
            "\nOut of memory during initialization (no warm speed measured): "
            + summary
            + "."
        )
    fig.text(0.075, 0.045, note, fontsize=8.8, color="#526171", linespacing=1.55)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(args.out / f"{stem}.{extension}", dpi=220, facecolor="white")
    plt.close(fig)
    print(args.out / f"{stem}.png")


if __name__ == "__main__":
    main()
