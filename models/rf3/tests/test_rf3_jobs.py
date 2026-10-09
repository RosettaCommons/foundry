"""Reuse weights while isolating every job's RNG and output identity."""

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from rf3.inference_engines.jobs import validate_jobs
from rf3.inference_engines.rf3 import RF3InferenceEngine


def job(name, seed=12):
    return dict(inputs=f"{name}.cif", out_dir=f"output/{name}", seed=seed)


@pytest.mark.parametrize(
    "jobs",
    [
        [],
        {},
        [dict(inputs="x")],
        [job("x", True)],
        [job("x", -1)],
        [job("x", 2**32)],
        [dict(job("x"), unexpected=1)],
        [dict(job("x"), inputs="")],
        [job("x"), job("x")],
    ],
)
def test_invalid_jobs(jobs):
    with pytest.raises(ValueError):
        validate_jobs(jobs)


def engine():
    obj = RF3InferenceEngine.__new__(RF3InferenceEngine)
    obj.seed = 99
    obj.loads = 0

    def initialize():
        obj.loads += 1
        # Model initialization consumes randomness. Per-job seeding must follow it.
        torch.rand(17)

    obj.initialize = initialize

    def run(**kwargs):
        return kwargs, obj.seed, (random.random(), np.random.rand(), torch.rand(3))

    obj.run = run
    return obj


def test_jobs_reuse_engine_and_preserve_order_independent_randomness():
    obj = engine()
    jobs = [job("a", 1), job("b", 2)]
    result = obj.run_jobs(jobs, template_selection=[])
    assert obj.loads == 1
    assert obj.seed == 99
    reverse = engine().run_jobs(list(reversed(jobs)), template_selection=[])
    for a, b in zip(result, reversed(reverse)):
        assert a[:2] == b[:2]
        assert a[2][:2] == b[2][:2]
        torch.testing.assert_close(a[2][2], b[2][2], rtol=0, atol=0)
    assert jobs == [job("a", 1), job("b", 2)]


def test_failure_restores_seed_and_stops_later_jobs():
    obj = engine()
    calls = []

    def fail(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("prediction failed")

    obj.run = fail
    with pytest.raises(RuntimeError, match="prediction failed"):
        obj.run_jobs([job("a"), job("b")])
    assert obj.seed == 99
    assert len(calls) == 1


def test_invalid_run_options_fail_before_loading():
    obj = engine()
    with pytest.raises(ValueError):
        obj.run_jobs([job("a")], seed=123)
    assert obj.loads == 0


def test_cli_jobs_dispatch_without_loading_model(tmp_path, monkeypatch):
    import json

    import rf3.inference as cli

    path = tmp_path / "jobs.json"
    path.write_text(json.dumps([job("a")]))
    calls = []
    fake = SimpleNamespace(run_jobs=lambda jobs, **kw: calls.append((jobs, kw)))

    def instantiate(cfg, **kwargs):
        assert "jobs" not in cfg and "inputs" not in cfg and "out_dir" not in cfg
        return fake

    monkeypatch.setattr(cli, "instantiate", instantiate)
    cfg = OmegaConf.create(
        dict(
            jobs=str(path),
            inputs="???",
            out_dir="???",
            triangle_attention_backend="auto",
        )
    )
    cli.run_inference.__wrapped__(cfg)
    assert calls[0][0] == [job("a")]
    assert "inputs" not in calls[0][1] and "out_dir" not in calls[0][1]


def test_backend_validated_before_loading():
    with pytest.raises(ValueError, match="triangle_attention_backend"):
        RF3InferenceEngine(ckpt_path="unused.ckpt", triangle_attention_backend="bad")


def test_cli_composes_real_config_for_jobs(tmp_path, monkeypatch):
    import json

    import rf3.inference as inference
    from rf3.cli import app
    from typer.testing import CliRunner

    path = tmp_path / "jobs.json"
    path.write_text(json.dumps([job("a")]))
    calls = []

    def instantiate(cfg, **kwargs):
        assert cfg.triangle_attention_backend == "auto"
        assert cfg.dense_attention_backend == "auto"
        assert cfg.num_steps == 50 and cfg.n_recycles == 10
        assert "jobs" not in cfg and "inputs" not in cfg and "out_dir" not in cfg
        return SimpleNamespace(run_jobs=lambda jobs, **options: calls.append(jobs))

    monkeypatch.setattr(inference, "instantiate", instantiate)
    result = CliRunner().invoke(app, ["fold", f"jobs={path}"])
    assert result.exit_code == 0, result.output
    assert calls == [[job("a")]]


def test_backend_override_reaches_all_model_copies(monkeypatch):
    from rf3.model.layers.attention import TriangleAttention

    from foundry.inference_engines.base import BaseInferenceEngine

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
    obj.trainer = SimpleNamespace(state={"model": torch.nn.ModuleList([copies, dense])})
    obj.triangle_attention_backend = "vanilla"
    obj.dense_attention_backend = "sdpa"
    monkeypatch.setattr(BaseInferenceEngine, "initialize", lambda self: None)
    obj.initialize()
    assert all(m.attention_backend == "vanilla" for m in copies)
    assert all(m.dense_attention_backend == "sdpa" for m in dense)
