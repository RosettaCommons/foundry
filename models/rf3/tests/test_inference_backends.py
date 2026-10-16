"""Inference configuration and automatic kernel installation."""

from types import SimpleNamespace

import pytest
import torch
from rf3.inference_engines.rf3 import RF3InferenceEngine


def test_backend_validated_before_loading():
    with pytest.raises(ValueError, match="triangle_attention_backend"):
        RF3InferenceEngine(ckpt_path="unused.ckpt", triangle_attention_backend="bad")


def test_cli_composes_real_config(monkeypatch):
    import rf3.inference as inference
    from rf3.cli import app
    from typer.testing import CliRunner

    calls = []

    def instantiate(cfg, **kwargs):
        assert cfg.triangle_attention_backend == "auto"
        assert cfg.dense_attention_backend == "auto"
        assert "inference_kernel_backend" not in cfg
        assert cfg.num_steps == 50 and cfg.n_recycles == 10
        assert "inputs" not in cfg and "out_dir" not in cfg
        return SimpleNamespace(run=lambda **options: calls.append(options))

    monkeypatch.setattr(inference, "instantiate", instantiate)
    result = CliRunner().invoke(app, ["fold", "inputs=example.cif", "out_dir=output"])
    assert result.exit_code == 0, result.output
    assert calls[0]["inputs"] == "example.cif"
    assert calls[0]["out_dir"] == "output"


def test_backend_override_reaches_all_model_copies(monkeypatch):
    from rf3._core.inference_engines.base import BaseInferenceEngine
    from rf3.model.layers.attention import TriangleAttention

    obj = RF3InferenceEngine.__new__(RF3InferenceEngine)
    obj.initialized_ = True
    from rf3.model.layers.af3_diffusion_transformer import AttentionPairBiasDiffusion
    from rf3.model.layers.pairformer_layers import AttentionPairBiasPairformerDeepspeed

    copies = torch.nn.ModuleList([TriangleAttention(8), TriangleAttention(8)])
    dense = torch.nn.ModuleList(
        [
            AttentionPairBiasPairformerDeepspeed(16, 16, 8, 4),
            AttentionPairBiasDiffusion(16, 16, 8, 4, True),
        ]
    )
    norms = torch.nn.ModuleList([torch.nn.LayerNorm(128), torch.nn.LayerNorm(128)])
    obj.trainer = SimpleNamespace(
        state={"model": torch.nn.ModuleList([copies, dense, norms])}
    )
    obj.triangle_attention_backend = "vanilla"
    obj.dense_attention_backend = "sdpa"
    monkeypatch.setattr(BaseInferenceEngine, "initialize", lambda self: None)
    obj.initialize()
    assert all(m.attention_backend == "vanilla" for m in copies)
    assert all(m.dense_attention_backend == "sdpa" for m in dense)
    from rf3.model.layers.inference_kernels import ExactLayerNorm

    assert all(isinstance(m, ExactLayerNorm) for m in norms)
    installed = list(norms)
    obj.initialize()
    assert all(a is b for a, b in zip(norms, installed))
