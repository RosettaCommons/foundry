"""Compile capacities are ceilings, and grouping and allocation use one policy."""

import pytest
import torch
from rfd3.inference.batching import (
    batch_signature,
    iter_compatible_batches,
    padding_capacity,
    validate_buckets,
)
from rfd3_batch_fixtures import example, tiny_model
from test_rfd3_batch_engine import engine


@pytest.mark.parametrize(
    "length,expected",
    [(1, 256), (256, 256), (257, 384), (384, 384), (385, 768), (768, 768)],
)
def test_bucket_ceilings(length, expected):
    assert padding_capacity(length, buckets=(256, 384, 768)) == expected


def test_bucket_overflow_never_truncates():
    with pytest.raises(ValueError, match="769.*768"):
        padding_capacity(769, buckets=(256, 384, 768))


@pytest.mark.parametrize(
    "values", [[], [0, 4], [-1, 4], [4, 4], [8, 4], [True, 4], [4.5, 8]]
)
def test_invalid_buckets(values):
    with pytest.raises(ValueError, match="strictly increasing positive integers"):
        validate_buckets(values, "test")


def test_grouping_uses_bucketed_atoms_tokens_and_slots():
    exs = [example((2, 2), seed=1), example((3, 3), seed=2), example((3, 3, 3), seed=3)]
    policy = dict(atom_buckets=(8, 16), token_buckets=(4, 8), slot_buckets=(4, 8))
    assert batch_signature(exs[0], **policy)[:3] == (8, 4, 4)
    groups = list(iter_compatible_batches(exs, 2, **policy))
    assert [[e["example_id"] for e in g] for g in groups] == [
        [e["example_id"] for e in exs[:2]],
        [exs[2]["example_id"]],
    ]


def test_compiled_engine_cfg_and_tail_share_all_capacities(monkeypatch):
    model = tiny_model()
    model.use_classifier_free_guidance = True
    model.cfg_features = ["active_donor"]
    obj = engine(model)
    obj.compile_model = True
    obj.compile_atom_buckets = (8, 16)
    obj.compile_token_buckets = (4, 8)
    obj.compile_slot_buckets = (4, 8)
    ex = example()
    ex["feats"]["is_motif_atom_unindexed"][3:] = True
    ex["feats"]["is_motif_token_unindexed"][2:] = True
    seen = []
    original = model.forward

    def record(*args, **kwargs):
        for f in [kwargs["input"]["f"], kwargs["input"]["f_ref"]]:
            seen.append(
                (
                    tuple(f["atom_valid"].shape),
                    tuple(f["token_valid"].shape),
                    tuple(f["atom_slots"].shape),
                )
            )
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "forward", record)
    output = obj._model_forward_batch([ex])
    assert seen == [((3, 8), (3, 4), (3, 4, 4))] * 2
    assert len(output[ex["example_id"]]) == 3
    assert all(torch.isfinite(o.atom_array).all() for o in output[ex["example_id"]])


def test_compile_at_batch_one_uses_padded_route(monkeypatch):
    obj = engine(tiny_model())
    obj.inference_batch_size = 1
    obj.compile_model = True
    monkeypatch.setattr(obj, "_run_multi_batched", lambda specs: "padded")
    assert obj._run_multi({}) == "padded"


def test_eager_mode_keeps_existing_capacity_policy():
    obj = engine(tiny_model())
    obj.compile_atom_buckets = (128,)
    obj.compile_token_buckets = (256,)
    assert obj._batch_capacities([example()]) == dict(
        atom_capacity=6, token_capacity=3, batch_capacity=3
    )


def test_engine_buckets_reuse_graph_across_lengths_slots_and_tail_rows():
    from rfd3.inference.batching import collate_examples, prepare_attention
    from rfd3.model.layers.batched import initialize

    torch._dynamo.reset()
    model = tiny_model(bf16=False)
    obj = engine(model)
    obj.compile_model = True
    obj.compile_atom_buckets = (8, 16)
    obj.compile_token_buckets = (4, 8)
    obj.compile_slot_buckets = (4, 8)
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    compiled = torch.compile(
        model.diffusion_module.forward_batched,
        backend=backend,
        fullgraph=True,
        dynamic=False,
    )
    groups = [
        [example((2, 2), seed=1), example((3, 3), seed=2)],
        [example((1, 1, 1), seed=3)],
        [example((3, 3, 3), seed=4)],
    ]
    with torch.no_grad():
        for index, exs in enumerate(groups):
            batch = collate_examples(exs, **obj._batch_capacities(exs))
            f = prepare_attention(
                batch["f"],
                atom_keys=4,
                atom_neighbors=1,
                token_keys=32,
                token_neighbors=8,
            )
            init = initialize(model.token_initializer, f)
            t = torch.ones(3, 3)
            expected = model.diffusion_module.forward_batched(
                batch["coords"], t, f, **init
            )
            actual = compiled(batch["coords"], t, f, **init)
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key])
            assert len(graphs) == (1 if index < 2 else 2)


def test_compile_bucket_config_validates_before_loading_model():
    from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine

    with pytest.raises(ValueError, match="compile_token_buckets"):
        RFD3InferenceEngine(
            **RFD3InferenceConfig(compile_model=True, compile_token_buckets=(384, 256))
        )


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_singleton_compile_wraps_padded_denoiser_only(monkeypatch, device):
    from types import SimpleNamespace

    obj = engine(tiny_model())
    obj.inference_batch_size = 1
    obj.compile_model = True
    obj.compile_atom_buckets = (8,)
    obj.compile_token_buckets = (4,)
    obj.compile_slot_buckets = (4,)
    calls = []
    monkeypatch.setattr(
        obj._rfd3_net().diffusion_module,
        "parameters",
        lambda: iter([SimpleNamespace(device=torch.device(device))]),
    )

    def compile_fn(fn, **kwargs):
        calls.append((fn.__name__, kwargs))
        return fn

    monkeypatch.setattr(torch, "compile", compile_fn)
    obj._compile_diffusion_submodules()
    options = {"max_fusion_unique_io_buffers": 30} if device == "mps" else {}
    if device == "cuda":
        options = {"max_fusion_unique_io_buffers": 16}
    assert calls == [
        ("forward_batched", dict(dynamic=False, fullgraph=True, options=options))
    ]


def test_bucket_padding_preserves_real_cfg_rollout():
    model = tiny_model(bf16=False)
    model.use_classifier_free_guidance = True
    model.cfg_features = ["active_donor"]
    model.inference_sampler.sampler.use_classifier_free_guidance = True
    model.inference_sampler.sampler.cfg_scale = 1.5
    obj = engine(model)
    ex = example()
    ex["feats"]["is_motif_atom_unindexed"][3:] = True
    ex["feats"]["is_motif_token_unindexed"][2:] = True
    expected = obj._model_forward_batch([ex])
    obj.compile_model = True
    obj.compile_atom_buckets = (8, 16)
    obj.compile_token_buckets = (4, 8)
    obj.compile_slot_buckets = (4, 8)
    actual = obj._model_forward_batch([ex])
    for before, after in zip(expected[ex["example_id"]], actual[ex["example_id"]]):
        torch.testing.assert_close(
            after.atom_array, before.atom_array, atol=2e-5, rtol=2e-5
        )


def test_compile_disabled_environment_is_not_reported_as_compilation(monkeypatch):
    obj = engine(tiny_model())
    obj.compile_model = True
    monkeypatch.setattr(torch._dynamo.config, "disable", True)
    with pytest.raises(ValueError, match="TORCH_COMPILE_DISABLE=0"):
        obj.initialize()
