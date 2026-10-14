"""The legacy preset restores execution without changing other models or training."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from rfd3._core.inference_engines.base import BaseInferenceEngine
from rfd3._core.utils.torch import scatter_mean
from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine
from rfd3.model.layers.blocks import LinearEmbedWithPool


def make_engine(tmp_path, **kwargs):
    return RFD3InferenceEngine(
        **RFD3InferenceConfig(ckpt_path=tmp_path / "test.ckpt", **kwargs)
    )


def test_preset_defaults_and_unknown_value(tmp_path):
    assert RFD3InferenceConfig().inference_mode == "fast"
    yaml = Path(__file__).parents[1] / "configs/inference_engine/rfdiffusion3.yaml"
    assert OmegaConf.load(yaml).inference_mode == "fast"
    fast = make_engine(tmp_path)
    assert fast.compile_model and fast.compile_cuda_graphs
    assert fast.inference_kernel_backend == "auto"
    with pytest.raises(ValueError, match="inference_mode"):
        make_engine(tmp_path, inference_mode="typo")


def test_legacy_preset_overrides_acceleration_and_is_scoped(tmp_path, monkeypatch):
    legacy = make_engine(
        tmp_path,
        inference_mode="legacy",
        inference_batch_size=4,
        inference_kernel_backend="triton",
        inference_cuda_graph=True,
        dense_attention_backend="sdpa",
    )
    assert not legacy.compile_model and not legacy.compile_cuda_graphs
    assert not legacy.inference_cuda_graph
    assert legacy.inference_batch_size == 1 and legacy.inference_num_workers == 0
    assert legacy.inference_kernel_backend == "torch"
    assert legacy.dense_attention_backend == "vanilla"
    # Both primary and EMA copies are configured; another engine stays corrected.
    fast = make_engine(
        tmp_path,
        compile_model=False,
        inference_num_workers=0,
        inference_kernel_backend="torch",
    )
    monkeypatch.setattr(BaseInferenceEngine, "initialize", lambda self: None)
    for engine in (legacy, fast):
        model = torch.nn.ModuleList(
            [LinearEmbedWithPool(3), LinearEmbedWithPool(3)]
        ).eval()
        engine.trainer = SimpleNamespace(state={"model": model})
        engine.initialize()
        assert all(m.legacy_pooling == (engine is legacy) for m in model)
    assert scatter_mean.__module__ == "rfd3._core.utils.torch"
    # A legacy request must never route into padded execution.
    monkeypatch.setattr(
        legacy, "_run_multi_batched", lambda _: pytest.fail("padded path")
    )
    monkeypatch.setattr(
        "rfd3.engine.assemble_distributed_inference_loader_from_json", lambda **_: []
    )
    legacy.pipeline = None
    legacy.trainer.fabric = SimpleNamespace(
        world_size=1, global_rank=0, setup_dataloaders=lambda x, **_: x
    )
    legacy.out_dir = None
    assert legacy._run_multi({}) == {}


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_legacy_reduction_matches_original_and_training_stays_corrected(dtype):
    layer = LinearEmbedWithPool(3).to(dtype).eval()
    with torch.no_grad():
        layer.linear.weight.copy_(torch.eye(3))
    x = torch.tensor(
        [[[-0.7578125] * 3, [-0.6015625] * 3, [-0.39453125] * 3]], dtype=dtype
    )
    index = torch.zeros(3, dtype=torch.long)
    zeros = torch.zeros(1, 1, 3, dtype=dtype)
    layer.legacy_pooling = True
    expected = zeros.index_reduce(
        -2, index, layer.linear(x), "mean", include_self=False
    )
    torch.testing.assert_close(layer(x, index), expected, rtol=0, atol=0)
    layer.train()
    corrected = scatter_mean(zeros, -2, index, layer.linear(x))
    torch.testing.assert_close(layer(x, index), corrected, rtol=0, atol=0)


def test_output_json_records_resolved_mode_and_path(tmp_path):
    engine = make_engine(tmp_path, inference_mode="legacy")
    engine.resolved_kernel_backend = "torch"
    output = dict(predicted_atom_array_stack=[None], prediction_metadata={0: {}})
    result = engine._format_model_output({"example_id": "test"}, output)[0]
    settings = result.metadata["inference_settings"]
    assert settings["mode"] == "legacy"
    assert settings["pooling"] == "legacy_index_reduce"
    assert settings["execution_path"] == "legacy"
    assert settings["random_stream"] == "legacy_global"
    assert not settings["compile_model"]
