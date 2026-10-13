"""RFD3 checkpoint backward comparison on trusted local audit snapshots.

The prepared features and generated target are a controlled numerical fixture,
not a natural-structure training dataset. No optimizer updates or scientific
convergence claims are made for the full checkpoint.
"""

import argparse
import copy
import json
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch
from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine
from rfd3.metrics.losses import DiffusionLoss
from rfd3.model.layers import blocks
from validate_pooling_training import native, oracle, relative_error

from foundry.utils.torch import scatter_mean

# Numerical comparison criteria fixed before this checkpoint experiment.
LIMITS = {
    "float32": {"output": 1e-4, "loss": 1e-4, "gradient": 1e-3},
    "bfloat16": {"output": 0.02, "loss": 0.02, "gradient": 0.1},
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    engine = RFD3InferenceEngine(
        **RFD3InferenceConfig(
            ckpt_path=a.checkpoint,
            diffusion_batch_size=1,
            seed=42,
            compile_model=False,
            prevalidate_inputs=False,
        )
    )
    engine.initialize()
    net = engine.trainer.state["model"]
    for _ in range(8):
        if hasattr(net, "diffusion_module"):
            break
        for attr in ("_forward_module", "module", "shadow", "model"):
            if hasattr(net, attr):
                net = getattr(net, attr)
                break
    net.eval()
    dm = net.diffusion_module
    # Inference selects detached EMA weights; enable gradients without changing them.
    dm.requires_grad_(True)
    for module in net.modules():
        module.inference_kernel_backend = "torch"
    data = torch.load(a.input, weights_only=False)
    example = engine.trainer.fabric.to_device(data["example"])
    f = example["feats"]
    target = torch.load(a.target, weights_only=False)["legacy"]["X_L"].cuda().detach()
    noise = data["noise"][0].cuda()
    t = torch.full((1,), 16.0, device="cuda")
    noisy = target + 16 * noise
    loss_input = dict(
        crd_mask_L=torch.ones(target.shape[-2], device="cuda", dtype=torch.bool),
        is_original_unindexed_token=torch.zeros(
            f["token_index"].shape[0], dtype=torch.bool, device="cuda"
        ),
        X_gt_L_in_input_frame=target,
    )
    loss_fn = DiffusionLoss(weight=4.0, sigma_data=dm.sigma_data, lddt_weight=0.25)
    report = dict(
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        limits=LIMITS,
        target="generated structure from matched-noise inference snapshot",
        cases=[],
    )
    for dtype in [torch.float32, torch.bfloat16]:
        reference = None
        for name, pooling in [
            ("oracle", oracle),
            ("fixed", scatter_mean),
            ("native", native),
        ]:
            torch.manual_seed(2026)
            net.zero_grad(set_to_none=True)
            context = (
                torch.autocast("cuda", dtype=dtype)
                if dtype != torch.float32
                else nullcontext()
            )
            # Initializer is held fixed: the trained denoiser and all its parameters
            # receive gradients, while the fixture tests diffusion conditioning.
            with torch.no_grad(), context:
                init = net.token_initializer(copy.deepcopy(f))
            dm.train()
            with patch.object(blocks, "scatter_mean", pooling), context:
                output = dm(noisy, t, copy.deepcopy(f), **init, n_recycle=2)
                loss, terms = loss_fn({"f": f, "t": t}, output, loss_input)
            with patch.object(blocks, "scatter_mean", pooling):
                loss.backward()
            gradients = {
                k: v.grad.detach().float().cpu().clone()
                for k, v in dm.named_parameters()
                if v.grad is not None
            }
            assert gradients and any(
                bool(torch.count_nonzero(v)) for v in gradients.values()
            )
            row = dict(
                dtype=str(dtype),
                arm=name,
                loss=float(loss.detach()),
                terms={
                    k: float(v) if torch.isfinite(v) else None for k, v in terms.items()
                },
                gradient_parameter_count=sum(v.numel() for v in gradients.values()),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            )
            current = dict(
                output=output["X_L"].detach().float().cpu(),
                loss=loss.detach().float().cpu(),
            )
            if reference is None:
                reference = (current, gradients)
            else:
                ref, ref_grads = reference
                assert gradients.keys() == ref_grads.keys()
                delta2 = norm2 = dot = other2 = 0.0
                for key, grad in gradients.items():
                    g, r = grad.double(), ref_grads[key].double()
                    assert torch.isfinite(g).all() and torch.isfinite(r).all()
                    delta2 += float((g - r).square().sum())
                    norm2 += float(r.square().sum())
                    other2 += float(g.square().sum())
                    dot += float((g * r).sum())
                row["relative_errors_vs_oracle"] = {
                    k: relative_error(current[k], ref[k]) for k in current
                }
                row["relative_errors_vs_oracle"]["gradient"] = (
                    delta2 / max(norm2, 1e-24)
                ) ** 0.5
                row["gradient_cosine"] = dot / (max(norm2 * other2, 1e-48) ** 0.5)
                key = "process_a.linear.weight"
                row["pool_weight_gradient_relative_error"] = relative_error(
                    gradients[key], ref_grads[key]
                )
                limits = LIMITS[str(dtype).split(".")[-1]]
                row["within_declared_limits"] = all(
                    row["relative_errors_vs_oracle"][k] <= v for k, v in limits.items()
                )
            report["cases"].append(row)
            print(json.dumps(row), flush=True)
            a.out.write_text(json.dumps(report, indent=2) + "\n")
            dm.eval()
            del gradients, output, loss
    if not all(
        x.get("within_declared_limits", True)
        for x in report["cases"]
        if x["arm"] == "fixed"
    ):
        raise SystemExit("Fixed pooling exceeded declared checkpoint tolerances.")


if __name__ == "__main__":
    main()
