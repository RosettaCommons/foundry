"""Regression checks at the actual optional RFD3 and RF3 training callsites."""

import copy
from contextlib import nullcontext
from unittest.mock import patch

import pytest
import torch

from foundry.utils.torch import scatter_mean

rfd3_blocks = pytest.importorskip("rfd3.model.layers.blocks")
rf3_pairformer = pytest.importorskip("rf3.model.layers.pairformer_layers")
rf3_diffusion = pytest.importorskip("rf3.model.layers.af3_diffusion_transformer")

# Defined before running experiments. Relative norms use the oracle norm, with
# floor 1e-12; optimizer update comparisons use changes, not full parameters.
TOLERANCES = {
    "float32": {"output": 1e-5, "loss": 1e-5, "gradient": 1e-4, "update": 1e-3},
    "bfloat16": {"output": 1e-3, "loss": 1e-3, "gradient": 5e-3, "update": 2e-2},
}


def oracle(destination, dim, index, source):
    """FP64 mean of the same already-quantized activations, cast to output dtype."""
    outputs = []
    for i in range(destination.shape[dim]):
        selected = source.index_select(dim, (index == i).nonzero().flatten())
        outputs.append(
            selected.double().mean(dim).to(source.dtype)
            if selected.shape[dim]
            else destination.select(dim, i)
        )
    return torch.stack(outputs, dim)


def native(destination, dim, index, source):
    return destination.index_reduce(dim, index, source, "mean", include_self=False)


def relative_error(candidate, reference):
    a, b = candidate.double().flatten(), reference.double().flatten()
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    return float(
        torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1e-12)
    )


def make_case(kind, group_size, batch, device):
    torch.manual_seed(731)
    tokens, channels = 4, 32
    atoms = tokens * group_size
    index = torch.arange(tokens, device=device).repeat_interleave(group_size)
    # Permute complete source positions to exercise unsorted index grouping.
    index = index[torch.randperm(atoms, device=device)]
    if kind == "rfd3":
        module = rfd3_blocks.LinearEmbedWithPool(channels).to(device)
        args = (torch.randn(batch, atoms, 3, device=device), index)
        target = torch.randn(batch, tokens, channels, device=device)
        return module, rfd3_blocks, args, target
    kwargs = dict(
        c_atom=32,
        c_atompair=8,
        c_token=channels,
        c_tokenpair=8,
        c_s=16,
        atom_1d_features=["ref_pos"],
        c_atom_1d_features=3,
        atom_transformer=dict(
            n_queries=32,
            n_keys=128,
            diffusion_transformer=dict(
                n_block=1,
                diffusion_transformer_block=dict(
                    n_head=4,
                    no_residual_connection_between_attention_and_transition=False,
                    kq_norm=True,
                ),
            ),
        ),
        use_inv_dist_squared=True,
    )
    f = dict(
        atom_to_token_map=index,
        ref_pos=torch.randn(atoms, 3, device=device),
        ref_space_uid=index,
        ref_atom_name_chars=torch.zeros(atoms, 4, 64, device=device),
    )
    if kind == "rf3_pairformer":
        module = rf3_pairformer.AtomAttentionEncoderPairformer(**kwargs).to(device)
        args = (f, None, None, None)
        target = torch.randn(tokens, channels, device=device)
        owner = rf3_pairformer
    else:
        module = rf3_diffusion.AtomAttentionEncoderDiffusion(
            **kwargs,
            broadcast_trunk_feats_on_1dim_old=False,
            use_chiral_features=False,
            no_grad_on_chiral_center=True,
        ).to(device)
        args = (
            f,
            torch.randn(batch, atoms, 3, device=device),
            torch.randn(tokens, 16, device=device),
            torch.randn(tokens, tokens, 8, device=device),
        )
        target = torch.randn(batch, tokens, channels, device=device)
        owner = rf3_diffusion
    return module, owner, args, target


def run_arm(module, owner, args, target, pooling, dtype, steps):
    model = copy.deepcopy(module).train()
    initial = {k: p.detach().clone() for k, p in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    records = []
    for step in range(steps):
        torch.manual_seed(1900 + step)
        optimizer.zero_grad(set_to_none=True)
        context = (
            torch.autocast(target.device.type, dtype=dtype)
            if dtype != torch.float32
            else nullcontext()
        )
        # Keep patch active through checkpoint recomputation in backward.
        with patch.object(owner, "scatter_mean", pooling), context:
            output = model(*copy.deepcopy(args))
            if isinstance(output, tuple):
                output = output[0]
            loss = (output.float() - target).square().mean()
        with patch.object(owner, "scatter_mean", pooling):
            loss.backward()
        gradients = torch.cat(
            [
                p.grad.detach().float().flatten()
                for p in model.parameters()
                if p.grad is not None
            ]
        )
        optimizer.step()
        update = torch.cat(
            [(p.detach() - initial[k]).flatten() for k, p in model.named_parameters()]
        )
        records.append(
            dict(
                output=output.detach().float(),
                loss=loss.detach().float(),
                gradient=gradients,
                update=update,
            )
        )
    return records


def compare_case(kind, group_size, batch, device, dtype, steps):
    module, owner, args, target = make_case(kind, group_size, batch, device)
    results = {
        name: run_arm(module, owner, args, target, pool, dtype, steps)
        for name, pool in [
            ("oracle", oracle),
            ("fixed", scatter_mean),
            ("native", native),
        ]
    }
    comparisons = {}
    for name in ("fixed", "native"):
        comparisons[name] = [
            {key: relative_error(arm[key], ref[key]) for key in ref}
            for arm, ref in zip(results[name], results["oracle"], strict=True)
        ]
    limits = TOLERANCES[str(dtype).split(".")[-1]]
    passed = all(
        row[key] <= limits[key] for row in comparisons["fixed"] for key in limits
    )
    return dict(
        module=kind,
        group_size=group_size,
        batch=batch,
        dtype=str(dtype),
        steps=steps,
        tolerance=limits,
        fixed_pass=passed,
        relative_errors_vs_oracle=comparisons,
        losses={
            name: [float(r["loss"]) for r in rows] for name, rows in results.items()
        },
    )


@pytest.mark.parametrize("module", ["rfd3", "rf3_pairformer", "rf3_diffusion"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_pooling_training_matches_quantized_activation_oracle(module, dtype):
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        result = compare_case(
            module,
            group_size=23,
            batch=1,
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=dtype,
            steps=2,
        )
        assert result["fixed_pass"], result
    finally:
        torch.set_num_threads(previous_threads)
